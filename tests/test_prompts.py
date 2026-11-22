"""CPU-only tests for chat-template prompts and decision-token scoring.

The old pipeline hand-built ``<s>[INST] ...`` strings, which silently produce
the wrong format for any other base model. These tests pin that the template
comes from the tokenizer and that the scored token is the one training
supervises.
"""

import pytest
from _stubs import CHATML, MISTRAL_V3, StubTokenizer

from src.pubmedqa_loraplus.prompts import (
    ANSWER_MARKER,
    ChatFormat,
    build_completion_content,
    build_user_message,
    choice_token_ids,
    extract_rationale,
)
from src.pubmedqa_loraplus.scoring import PUBMEDQA_LABELS


@pytest.mark.parametrize(
    ("template", "prefix", "suffix"),
    [(MISTRAL_V3, " ", "</s>"), (CHATML, "", "<|im_end|>\n")],
)
def test_chat_format_recovers_the_assistant_wrapper_from_the_template(template, prefix, suffix):
    chat_format = ChatFormat.from_tokenizer(StubTokenizer(template))

    assert chat_format.completion_prefix == prefix
    assert chat_format.completion_suffix == suffix


def test_chatml_prompt_carries_no_mistral_markers():
    tokenizer = StubTokenizer(CHATML)
    chat_format = ChatFormat.from_tokenizer(tokenizer)

    prompt = chat_format.render_prompt(tokenizer, build_user_message("Q?", "ctx", "label"))

    assert "[INST]" not in prompt and "<s>" not in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("template", [MISTRAL_V3, CHATML])
@pytest.mark.parametrize("target_format", ["label", "rationale"])
@pytest.mark.parametrize("decision", PUBMEDQA_LABELS)
def test_scored_decision_token_is_the_token_training_supervises(template, target_format, decision):
    tokenizer = StubTokenizer(template)
    chat_format = ChatFormat.from_tokenizer(tokenizer)
    rationale = "Treatment X reduced Y in adults."
    content = build_completion_content(decision, rationale, target_format)
    trained = tokenizer(chat_format.completion_text(content))["input_ids"]

    lead_text = chat_format.completion_prefix
    if target_format == "rationale":
        lead_text += f"{rationale}\n{ANSWER_MARKER} "
    lead_ids, choice_ids = choice_token_ids(tokenizer, lead_text, PUBMEDQA_LABELS)

    assert trained[: len(lead_ids)] == lead_ids
    assert trained[len(lead_ids)] == choice_ids[decision]


def test_choice_tokens_must_be_distinct():
    tokenizer = StubTokenizer()

    with pytest.raises(ValueError):
        choice_token_ids(tokenizer, " ", ("yes", "yes"))


def test_rationale_target_requires_a_long_answer():
    with pytest.raises(ValueError):
        build_completion_content("yes", "  ", "rationale")


def test_label_target_is_the_lowercased_decision():
    assert build_completion_content("Maybe", None, "label") == "maybe"


def test_extract_rationale_stops_at_the_answer_marker():
    assert extract_rationale("It helps.\nAnswer: yes") == "It helps."
    assert extract_rationale("No marker at all ") == "No marker at all"
