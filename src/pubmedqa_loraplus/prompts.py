"""Prompt formatting and token budgeting helpers.

Kept free of torch/datasets imports so the logic can be unit-tested on CPU.
"""

from __future__ import annotations

from typing import Sequence

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


def build_prompt_parts(
    question: str,
    context: str,
    bos_token: str = BOS_TOKEN,
) -> tuple[str, str, str]:
    """Split the instruction prompt into (head, context, tail) text segments.

    Keeping the context as its own segment lets the token budget drop context
    text while preserving the instruction markers and the question itself,
    which a plain head-truncation of the joined prompt would cut off.
    """
    head = f"{bos_token}[INST] Context: "
    tail = f"\n\nQuestion: {question} [/INST]"
    return head, context, tail


def build_instruction_prompt(
    question: str,
    context: str,
    bos_token: str = BOS_TOKEN,
) -> str:
    head, body, tail = build_prompt_parts(question, context, bos_token)
    return f"{head}{body}{tail}"


def build_answer_text(answer: str, eos_token: str = EOS_TOKEN) -> str:
    return f" {answer.lower()}{eos_token}"


def fit_prompt_ids(
    head_ids: Sequence[int],
    context_ids: Sequence[int],
    tail_ids: Sequence[int],
    budget: int,
) -> list[int]:
    """Fit a prompt into ``budget`` tokens without losing the instruction suffix.

    The head and the tail (``Question: ... [/INST]``) are always kept. Context
    tokens are dropped from the left because the end of a PubMedQA abstract —
    the conclusion — is the part that answers the question.

    The published run never hit this path: the longest of 100 real
    ``pqa_labeled`` prompts tokenized to 688 tokens against a 1024 budget, so
    left-truncating the joined prompt would not have removed the question. The
    guard matters for longer abstracts and smaller ``max_length`` values, so it
    is still used for both training and evaluation.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")

    tail = list(tail_ids)
    if len(tail) >= budget:
        return tail[-budget:]

    head = list(head_ids)
    remaining = budget - len(tail)
    if len(head) >= remaining:
        return head[:remaining] + tail
    remaining -= len(head)

    context = list(context_ids)
    if remaining > 0:
        context = context[-remaining:]
    else:
        context = []
    return head + context + tail
