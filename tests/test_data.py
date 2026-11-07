"""CPU-only tests for the label-masking data path.

``tokenize_example`` decides what the model learns: the prompt must be
masked with ``-100`` so only the answer tokens contribute to the loss. An
off-by-one here trains on the wrong tokens without failing any other test,
so the alignment is pinned directly against a character-level tokenizer.
"""

import pytest

from src.pubmedqa_loraplus.config import LoRAPlusConfig
from src.pubmedqa_loraplus.data import (
    CompletionOnlyCollator,
    encode_prompt,
    tokenize_example,
)
from src.pubmedqa_loraplus.prompts import build_answer_text, build_prompt_parts

IGNORE_INDEX = -100


class StubTokenizer:
    """Character-level tokenizer: one token per character, ids are ordinals."""

    bos_token = "<s>"
    eos_token = "</s>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, ids):
        return "".join(chr(token_id) for token_id in ids)


def _config(tmp_path, max_length=1024):
    return LoRAPlusConfig(
        run_name="test-run",
        base_model_id="mistralai/Mistral-7B-Instruct-v0.3",
        base_model_path=tmp_path / "base_model",
        dataset_id="qiaojin/PubMedQA",
        dataset_config="pqa_labeled",
        dataset_disk_path=tmp_path / "dataset",
        output_root=tmp_path / "out",
        reports_root=tmp_path / "reports",
        max_length=max_length,
    )


def _example(decision="yes", question="Does X help?", context=("An abstract.",)):
    return {
        "final_decision": decision,
        "question": question,
        "context": {"contexts": list(context)},
    }


@pytest.mark.parametrize("decision", ["yes", "no", "maybe"])
def test_prompt_tokens_are_masked_and_only_answer_is_supervised(tmp_path, decision):
    tokenizer = StubTokenizer()
    config = _config(tmp_path)
    example = _example(decision)

    encoded = tokenize_example(example, tokenizer, config)

    answer_ids = tokenizer(
        build_answer_text(decision, tokenizer.eos_token),
        add_special_tokens=False,
    )["input_ids"]
    prompt_length = len(encoded["input_ids"]) - len(answer_ids)

    assert prompt_length > 0
    assert encoded["labels"][:prompt_length] == [IGNORE_INDEX] * prompt_length
    assert encoded["labels"][prompt_length:] == answer_ids


def test_labels_align_with_input_ids(tmp_path):
    tokenizer = StubTokenizer()
    encoded = tokenize_example(_example(), tokenizer, _config(tmp_path))

    assert len(encoded["labels"]) == len(encoded["input_ids"])
    assert len(encoded["attention_mask"]) == len(encoded["input_ids"])
    supervised = [
        token_id
        for token_id, label in zip(encoded["input_ids"], encoded["labels"])
        if label != IGNORE_INDEX
    ]
    assert supervised == [label for label in encoded["labels"] if label != IGNORE_INDEX]


def test_answer_text_starts_with_the_decision_token(tmp_path):
    tokenizer = StubTokenizer()
    encoded = tokenize_example(_example("maybe"), tokenizer, _config(tmp_path))

    decoded_labels = tokenizer.decode(
        [label for label in encoded["labels"] if label != IGNORE_INDEX]
    )
    assert decoded_labels == " maybe</s>"


def test_sequence_never_exceeds_max_length(tmp_path):
    tokenizer = StubTokenizer()
    config = _config(tmp_path, max_length=64)

    encoded = tokenize_example(
        _example(context=["abstract " * 200]),
        tokenizer,
        config,
    )

    assert len(encoded["input_ids"]) <= config.max_length
    assert len(encoded["labels"]) <= config.max_length
    assert len(encoded["attention_mask"]) <= config.max_length


def test_prompt_keeps_the_instruction_suffix_when_truncated(tmp_path):
    tokenizer = StubTokenizer()
    config = _config(tmp_path, max_length=64)

    encoded = tokenize_example(
        _example(context=["abstract " * 200]),
        tokenizer,
        config,
    )
    prompt_length = len(encoded["input_ids"]) - len(
        tokenizer(
            build_answer_text("yes", tokenizer.eos_token),
            add_special_tokens=False,
        )["input_ids"]
    )

    prompt = tokenizer.decode(encoded["input_ids"][:prompt_length])
    assert "Question: Does X help? [/INST]" in prompt


def test_encode_prompt_respects_budget(tmp_path):
    tokenizer = StubTokenizer()

    prompt_ids = encode_prompt(tokenizer, "Q?", "context " * 100, budget=32)

    assert len(prompt_ids) == 32


def test_collator_pads_labels_with_ignore_index():
    collator = CompletionOnlyCollator(pad_token_id=0)
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4], "attention_mask": [1], "labels": [5]},
    ]

    batch = collator(features)

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [5, -100, -100]]


def test_collator_rejects_empty_batch():
    with pytest.raises(ValueError):
        CompletionOnlyCollator(pad_token_id=0)([])


def test_prompt_parts_are_reassembled_by_tokenize_example(tmp_path):
    tokenizer = StubTokenizer()
    example = _example(context=["First sentence.", "Second sentence."])

    encoded = tokenize_example(example, tokenizer, _config(tmp_path))
    prompt_length = len(encoded["input_ids"]) - len(
        tokenizer(
            build_answer_text("yes", tokenizer.eos_token),
            add_special_tokens=False,
        )["input_ids"]
    )
    prompt = tokenizer.decode(encoded["input_ids"][:prompt_length])
    head, _, tail = build_prompt_parts(
        example["question"],
        "First sentence. Second sentence.",
    )

    assert prompt.startswith(head)
    assert prompt.endswith(tail)
