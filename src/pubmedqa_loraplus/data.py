from typing import Any

import torch
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from .config import LoRAPlusConfig
from .prompts import ChatFormat, build_completion_content, build_user_message
from .splits import (
    balanced_indices,
    class_capped_sample,
    official_test_labels,
    stratified_holdout,
)

ARTIFICIAL_CONFIG = "pqa_artificial"


def flatten_context(context: object) -> str:
    if isinstance(context, dict) and "contexts" in context:
        return " ".join(context["contexts"])
    if isinstance(context, list):
        return " ".join(str(part) for part in context)
    return str(context)


def normalise_example(example: dict[str, Any]) -> dict[str, str]:
    """The one record shape every split uses, whatever PubMedQA subset it came from."""
    return {
        "pubid": str(example["pubid"]),
        "question": example["question"],
        "context": flatten_context(example["context"]),
        "long_answer": example.get("long_answer") or "",
        "final_decision": example["final_decision"].strip().lower(),
    }


EXAMPLE_FEATURES = Features(
    {
        name: Value("string")
        for name in ("pubid", "question", "context", "long_answer", "final_decision")
    }
)


def _normalise(dataset: Dataset) -> Dataset:
    # Explicit features: datasets otherwise keeps pqa_labeled's int64 pubid type.
    return dataset.map(
        normalise_example, remove_columns=dataset.column_names, features=EXAMPLE_FEATURES
    )


def encode_prompt(
    tokenizer: Any,
    chat_format: ChatFormat,
    question: str,
    context: str,
    target_format: str,
    budget: int,
) -> list[int]:
    """Render the chat prompt and tokenize it into at most ``budget`` tokens.

    The rendered string is tokenized whole, so the model sees exactly what the
    chat template produces. When it is too long, context tokens are dropped
    from the left (the end of the abstract holds the results) and the prompt is
    re-rendered, which keeps the instruction and the question intact.
    """

    def render(text: str) -> list[int]:
        prompt = chat_format.render_prompt(
            tokenizer, build_user_message(question, text, target_format)
        )
        return tokenizer(prompt, add_special_tokens=False)["input_ids"]

    prompt_ids = render(context)
    if len(prompt_ids) <= budget:
        return prompt_ids

    overhead = len(render(""))
    if overhead > budget:
        raise ValueError(f"The prompt without context needs {overhead} tokens; budget is {budget}")
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    keep = budget - overhead
    while keep > 0:
        prompt_ids = render(tokenizer.decode(context_ids[-keep:]))
        if len(prompt_ids) <= budget:
            return prompt_ids
        keep -= len(prompt_ids) - budget
    return render("")


def tokenize_example(
    example: dict[str, Any],
    tokenizer: Any,
    chat_format: ChatFormat,
    config: LoRAPlusConfig,
) -> dict[str, list[int]]:
    content = build_completion_content(
        example["final_decision"], example.get("long_answer"), config.target_format
    )
    completion_ids = tokenizer(chat_format.completion_text(content), add_special_tokens=False)[
        "input_ids"
    ]
    budget = config.max_length - len(completion_ids)
    if budget < 1:
        raise ValueError(f"max_length={config.max_length} leaves no room for the prompt")
    prompt_ids = encode_prompt(
        tokenizer,
        chat_format,
        example["question"],
        example["context"],
        config.target_format,
        budget,
    )
    input_ids = prompt_ids + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate empty feature batch")
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _load_labeled(config: LoRAPlusConfig) -> Dataset:
    if config.dataset_disk_path.exists():
        dataset = load_from_disk(str(config.dataset_disk_path))
        return dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    return load_dataset(config.dataset_id, config.dataset_config, revision=config.dataset_revision)[
        "train"
    ]


def load_pubmedqa(config: LoRAPlusConfig) -> DatasetDict:
    """Official 500-question test set; train/validation from the other 500.

    The test split is selected by PMID from PubMedQA's published ground truth,
    so the reported numbers are comparable with the leaderboard and no test
    question is ever trained on. The validation split, used only for checkpoint
    selection, is a label-stratified ``validation_size`` of the remaining pool.
    """
    labeled = _normalise(_load_labeled(config))
    truth = official_test_labels()
    pubids = labeled["pubid"]
    decisions = labeled["final_decision"]

    if len(set(pubids)) != len(pubids):
        raise ValueError("Dataset contains duplicate PMIDs")
    test_rows = [index for index, pubid in enumerate(pubids) if pubid in truth]
    if len(test_rows) != len(truth):
        raise ValueError(
            f"Dataset holds {len(test_rows)} of the {len(truth)} official test PMIDs; "
            "expected qiaojin/PubMedQA pqa_labeled"
        )
    disagreements = [pubids[i] for i in test_rows if truth[pubids[i]] != decisions[i]]
    if disagreements:
        raise ValueError(f"Labels disagree with the official ground truth: {disagreements[:5]}")

    pool_rows = [index for index, pubid in enumerate(pubids) if pubid not in truth]
    held_out = set(
        stratified_holdout([decisions[i] for i in pool_rows], config.validation_size, config.seed)
    )
    return DatasetDict(
        {
            "train": labeled.select(
                [row for position, row in enumerate(pool_rows) if position not in held_out]
            ),
            "validation": labeled.select(
                [row for position, row in enumerate(pool_rows) if position in held_out]
            ),
            "test": labeled.select(test_rows),
        }
    )


def _artificial_examples(config: LoRAPlusConfig, exclude: set[str]) -> Dataset:
    """A yes/no-balanced sample of ``pqa_artificial`` (which has no ``maybe``)."""
    artificial = load_dataset(
        config.dataset_id, ARTIFICIAL_CONFIG, revision=config.dataset_revision
    )["train"]
    rows = class_capped_sample(
        [decision.lower() for decision in artificial["final_decision"]],
        config.pqa_artificial_samples // 2,
        ("yes", "no"),
        config.seed,
    )
    sample = _normalise(artificial.select(rows))
    return sample.filter(lambda example: example["pubid"] not in exclude)


def build_train_dataset(dataset: DatasetDict, config: LoRAPlusConfig) -> Dataset:
    """The examples the optimizer sees: optional extra data, then class balancing."""
    train = dataset["train"]
    if config.pqa_artificial_samples:
        exclude = set(dataset["validation"]["pubid"]) | set(dataset["test"]["pubid"])
        train = concatenate_datasets([train, _artificial_examples(config, exclude)])
    if config.balance_train_classes:
        train = train.select(balanced_indices(train["final_decision"], config.seed))
    return train


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: Any,
    chat_format: ChatFormat,
    config: LoRAPlusConfig,
) -> Dataset:
    return dataset.map(
        lambda example: tokenize_example(example, tokenizer, chat_format, config),
        remove_columns=dataset.column_names,
    )
