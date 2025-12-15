from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk

from .config import LoRAPlusConfig


def flatten_context(context: object) -> str:
    if isinstance(context, dict) and "contexts" in context:
        return " ".join(context["contexts"])
    if isinstance(context, list):
        return " ".join(str(part) for part in context)
    return str(context)


def build_instruction_prompt(question: str, context: str, bos_token: str = "<s>") -> str:
    return f"{bos_token}[INST] Context: {context}\n\nQuestion: {question} [/INST]"


def build_answer_text(answer: str, eos_token: str = "</s>") -> str:
    return f" {answer.lower()}{eos_token}"


def build_prompt_answer_pair(
    example: dict[str, Any],
    bos_token: str,
    eos_token: str,
) -> tuple[str, str]:
    context = flatten_context(example["context"])
    prompt = build_instruction_prompt(example["question"], context, bos_token)
    answer = build_answer_text(example["final_decision"], eos_token)
    return prompt, answer


def tokenize_example(
    example: dict[str, Any],
    tokenizer: Any,
    config: LoRAPlusConfig,
) -> dict[str, list[int]]:
    prompt, answer = build_prompt_answer_pair(
        example,
        tokenizer.bos_token or "<s>",
        tokenizer.eos_token or "</s>",
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    max_prompt_tokens = max(config.max_length - len(answer_ids), 1)
    prompt_ids = prompt_ids[:max_prompt_tokens]
    input_ids = (prompt_ids + answer_ids)[: config.max_length]
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * len(prompt_ids) + answer_ids)[: config.max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
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


def load_pubmedqa(config: LoRAPlusConfig) -> DatasetDict:
    if config.dataset_disk_path.exists():
        dataset = load_from_disk(str(config.dataset_disk_path))
        train_split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    else:
        source = load_dataset(config.dataset_id, config.dataset_config)
        train_split = source["train"]

    return train_split.train_test_split(test_size=config.test_size, seed=config.seed)


def tokenize_dataset(dataset: DatasetDict, tokenizer: Any, config: LoRAPlusConfig) -> DatasetDict:
    return dataset.map(
        lambda example: tokenize_example(example, tokenizer, config),
        remove_columns=dataset["train"].column_names,
    )
