"""CPU-only tests for the label-masking data path.

``tokenize_example`` decides what the model learns: the prompt must be masked
with ``-100`` so only the completion contributes to the loss. An off-by-one
here trains on the wrong tokens without failing any other test.
"""

import pytest
from _stubs import CHATML, MISTRAL_V3, StubTokenizer

from src.pubmedqa_loraplus.config import LoRAPlusConfig
from src.pubmedqa_loraplus.data import CompletionOnlyCollator, tokenize_example
from src.pubmedqa_loraplus.prompts import ChatFormat

IGNORE_INDEX = -100


def _config(tmp_path, **overrides):
    return LoRAPlusConfig(
        run_name="test-run",
        base_model_id="mistralai/Mistral-7B-Instruct-v0.3",
        base_model_path=tmp_path / "base_model",
        dataset_id="qiaojin/PubMedQA",
        dataset_config="pqa_labeled",
        dataset_disk_path=tmp_path / "dataset",
        output_root=tmp_path / "out",
        reports_root=tmp_path / "reports",
        **overrides,
    )


def _example(decision="yes", context="An abstract.", long_answer="X helps."):
    return {
        "pubid": "1",
        "final_decision": decision,
        "question": "Does X help?",
        "context": context,
        "long_answer": long_answer,
    }


def _supervised_text(tokenizer, encoded):
    return tokenizer.decode([label for label in encoded["labels"] if label != IGNORE_INDEX])


@pytest.mark.parametrize("template", [MISTRAL_V3, CHATML])
@pytest.mark.parametrize(
    ("target_format", "expected_content"),
    [("label", "maybe"), ("rationale", "X helps.\nAnswer: maybe")],
)
def test_only_the_completion_is_supervised(tmp_path, template, target_format, expected_content):
    tokenizer = StubTokenizer(template)
    chat_format = ChatFormat.from_tokenizer(tokenizer)
    config = _config(tmp_path, target_format=target_format)

    encoded = tokenize_example(_example("maybe"), tokenizer, chat_format, config)

    supervised = sum(1 for label in encoded["labels"] if label != IGNORE_INDEX)
    prompt_length = len(encoded["labels"]) - supervised
    assert encoded["labels"][:prompt_length] == [IGNORE_INDEX] * prompt_length
    assert encoded["labels"][prompt_length:] == encoded["input_ids"][prompt_length:]
    assert _supervised_text(tokenizer, encoded) == chat_format.completion_text(expected_content)
    assert len(encoded["attention_mask"]) == len(encoded["input_ids"])


def test_long_context_is_trimmed_but_the_question_and_instruction_survive(tmp_path):
    tokenizer = StubTokenizer()
    chat_format = ChatFormat.from_tokenizer(tokenizer)
    config = _config(tmp_path, max_length=80, target_format="label")

    encoded = tokenize_example(
        _example(context="early " * 200 + "final results"), tokenizer, chat_format, config
    )

    prompt = tokenizer.decode(
        [token for token, label in zip(encoded["input_ids"], encoded["labels"]) if label == -100]
    )
    assert len(encoded["input_ids"]) <= config.max_length
    assert prompt.startswith("<s>[INST] Context:")
    assert "final results" in prompt
    assert "Question: Does X help?" in prompt
    assert prompt.endswith("yes, no, or maybe.[/INST]")


def test_rationale_example_without_long_answer_is_rejected(tmp_path):
    tokenizer = StubTokenizer()
    chat_format = ChatFormat.from_tokenizer(tokenizer)

    with pytest.raises(ValueError):
        tokenize_example(_example(long_answer=""), tokenizer, chat_format, _config(tmp_path))


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
