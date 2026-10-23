"""CPU-only tests for prompt formatting and token budgeting.

Left-slicing a joined prompt drops the ``Question: ... [/INST]`` suffix once
the abstract exceeds the budget. That did not happen on the published run (its
longest of 100 prompts was 688 tokens against a 1024 budget), but the guard is
what keeps the instruction suffix intact for longer inputs.
"""

import pytest

from src.pubmedqa_loraplus.prompts import (
    build_answer_text,
    build_instruction_prompt,
    fit_prompt_ids,
)


class StubTokenizer:
    """Character-level tokenizer: one token per character, ids are ordinals."""

    bos_token = "<s>"
    eos_token = "</s>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


def _decode(ids):
    return "".join(chr(token_id) for token_id in ids)


def test_instruction_prompt_keeps_mistral_markers():
    prompt = build_instruction_prompt("Does X help?", "Some context.", "<s>")

    assert prompt.startswith("<s>[INST] Context: ")
    assert "Question: Does X help? [/INST]" in prompt


def test_answer_text_is_space_prefixed_and_lowercase():
    assert build_answer_text("YES") == " yes</s>"


def test_fit_prompt_ids_keeps_head_and_tail_when_context_overflows():
    head = [1, 2]
    context = list(range(10, 30))
    tail = [40, 41, 42]

    fitted = fit_prompt_ids(head, context, tail, budget=10)

    assert len(fitted) == 10
    assert fitted[: len(head)] == head
    assert fitted[-len(tail) :] == tail


def test_fit_prompt_ids_drops_context_from_the_left_first():
    fitted = fit_prompt_ids([1], [10, 11, 12, 13], [20], budget=4)

    # 1 head + 2 context + 1 tail, context trimmed from the left.
    assert fitted == [1, 12, 13, 20]


def test_fit_prompt_ids_preserves_tail_when_budget_smaller_than_tail():
    fitted = fit_prompt_ids([1, 2], [3, 4], [5, 6, 7], budget=2)

    assert fitted == [6, 7]


def test_fit_prompt_ids_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        fit_prompt_ids([1], [2], [3], budget=0)


def test_joined_prompt_truncation_would_drop_the_question():
    """Regression guard: document why the old left-slice was wrong.

    Slicing the joined prompt to the last N tokens is fine, but slicing it to
    the *first* N tokens — which is what the old code did — removes the
    question and the [/INST] marker entirely once the abstract is long.
    """
    tokenizer = StubTokenizer()
    long_context = "x" * 500
    prompt = build_instruction_prompt("Does X help?", long_context, "<s>")
    joined = tokenizer(prompt)["input_ids"]
    budget = 40

    left_slice = joined[:budget]
    assert "Question:" not in _decode(left_slice)

    # The replacement keeps the instruction suffix.
    head = tokenizer("<s>[INST] Context: ")["input_ids"]
    context = tokenizer(long_context)["input_ids"]
    tail = tokenizer("\n\nQuestion: Does X help? [/INST]")["input_ids"]
    fitted = fit_prompt_ids(head, context, tail, budget)

    assert len(fitted) <= budget
    assert "Question: Does X help? [/INST]" in _decode(fitted)
