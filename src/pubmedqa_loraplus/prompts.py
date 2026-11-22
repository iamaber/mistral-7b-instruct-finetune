"""Prompt and completion formatting.

Kept free of torch/datasets imports so the logic can be unit-tested on CPU.
Every prompt goes through the tokenizer's own chat template, so swapping the
base model does not leave a Mistral-specific ``[INST]`` string behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

TargetFormat = Literal["label", "rationale"]
TARGET_FORMATS: tuple[str, ...] = ("label", "rationale")
ANSWER_MARKER = "Answer:"
# The trained completion puts the marker at the start of a line; matching that
# form keeps a quoted "Answer:" inside the conclusion from ending generation.
ANSWER_LINE = f"\n{ANSWER_MARKER}"

_INSTRUCTIONS: dict[str, str] = {
    "label": "Answer the question with exactly one word: yes, no, or maybe.",
    "rationale": (
        "Write a short conclusion supported by the abstract, then give the final "
        f"answer on its own line as '{ANSWER_MARKER} yes', '{ANSWER_MARKER} no', "
        f"or '{ANSWER_MARKER} maybe'."
    ),
}

_SENTINEL = "\x00PUBMEDQA_ASSISTANT\x00"


def build_user_message(question: str, context: str, target_format: str) -> str:
    return f"Context: {context}\n\nQuestion: {question}\n\n{_INSTRUCTIONS[target_format]}"


def build_completion_content(decision: str, long_answer: str | None, target_format: str) -> str:
    """The assistant turn the model is trained to produce.

    ``label`` supervises the decision alone. ``rationale`` supervises PubMedQA's
    ``long_answer`` (the abstract's own conclusion, which the context omits)
    followed by the decision, so the loss covers tokens that depend on the
    abstract instead of two label tokens.
    """
    label = decision.strip().lower()
    if target_format == "label":
        return label
    if not long_answer or not long_answer.strip():
        raise ValueError("rationale targets need a non-empty long_answer")
    return f"{long_answer.strip()}\n{ANSWER_MARKER} {label}"


def extract_rationale(generated: str) -> str:
    """Text the model wrote before its ``Answer:`` line (all of it if absent)."""
    return generated.split(ANSWER_LINE, 1)[0].strip()


@dataclass(frozen=True)
class ChatFormat:
    """How the tokenizer's chat template wraps one user turn and its reply.

    ``completion_prefix``/``completion_suffix`` are the template text around the
    assistant content (a single space and ``</s>`` for Mistral v0.3, nothing and
    ``<|im_end|>\\n`` for ChatML). They are recovered by rendering a sentinel
    reply, so no template is hard-coded here.
    """

    completion_prefix: str
    completion_suffix: str
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> "ChatFormat":
        kwargs = dict(template_kwargs or {})
        user = [{"role": "user", "content": "question"}]
        prompt = tokenizer.apply_chat_template(
            user, tokenize=False, add_generation_prompt=True, **kwargs
        )
        full = tokenizer.apply_chat_template(
            user + [{"role": "assistant", "content": _SENTINEL}],
            tokenize=False,
            **kwargs,
        )
        if not full.startswith(prompt):
            raise ValueError(
                "The chat template does not render the generation prompt as a prefix of "
                "the full conversation; pass chat_template_kwargs that make them agree."
            )
        prefix, found, suffix = full[len(prompt) :].partition(_SENTINEL)
        if not found:
            raise ValueError("The chat template dropped the assistant content")
        return cls(prefix, suffix, kwargs)

    def render_prompt(self, tokenizer: Any, user_text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )

    def completion_text(self, content: str) -> str:
        return f"{self.completion_prefix}{content}{self.completion_suffix}"


def choice_token_ids(
    tokenizer: Any,
    lead_text: str,
    choices: Sequence[str],
) -> tuple[list[int], dict[str, int]]:
    """Return ``(lead_ids, {choice: token_id})`` for scoring one-token decisions.

    Each ``lead_text + choice`` is tokenized whole, exactly as the training
    completion is. Their shared token prefix is the lead the model is fed; the
    next token is the choice's decision token. Scoring that single position
    replaces the old sum over ``" yes</s>"``, whose EOS term was a stop
    probability rather than a label probability.
    """
    encoded = {
        choice: tokenizer(lead_text + choice, add_special_tokens=False)["input_ids"]
        for choice in choices
    }
    sequences = list(encoded.values())
    shared = 0
    while (
        all(len(ids) > shared for ids in sequences) and len({ids[shared] for ids in sequences}) == 1
    ):
        shared += 1
    if any(len(ids) <= shared for ids in sequences):
        raise ValueError(f"choices {list(choices)} share every token after {lead_text!r}")
    ids = {choice: tokens[shared] for choice, tokens in encoded.items()}
    if len(set(ids.values())) != len(choices):
        raise ValueError(f"choices {list(choices)} do not start with distinct tokens")
    return sequences[0][:shared], ids
