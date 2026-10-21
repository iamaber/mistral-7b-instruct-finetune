"""Pure prediction-scoring and aggregation helpers.

Nothing here imports torch, datasets or sklearn, so the label decision and the
metric computation can be unit-tested on CPU without loading a model.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence

PUBMEDQA_LABELS = ("yes", "no", "maybe")


def score_candidate(
    prompt_length: int,
    candidate_ids: Sequence[int],
    log_prob_at: Callable[[int, int], float],
) -> float:
    """Sum the log-probability of every candidate token, given its own logits.

    A causal LM's logits at position ``i`` predict the token at position
    ``i + 1``. The first candidate token therefore has to be scored with the
    logits at ``prompt_length - 1`` (the last prompt position). Indexing from
    ``prompt_length`` instead never reads the position that predicts the first
    answer token, so the only logit carrying the label decision is skipped:
    for ``" yes</s>"`` / ``" no</s>"`` / ``" maybe</s>"`` the position that
    predicts ``yes`` / ``no`` / ``maybe`` is never consulted.
    """
    if prompt_length < 1:
        raise ValueError("prompt_length must be >= 1")
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty")

    total = 0.0
    for offset, token_id in enumerate(candidate_ids):
        total += log_prob_at(prompt_length + offset - 1, token_id)
    return total


def normalise_candidate_scores(
    raw_scores: Mapping[str, float],
    candidate_token_counts: Mapping[str, int],
) -> dict[str, float]:
    """Convert summed log-probabilities into per-token average log-probabilities.

    Summing log-probabilities prefers shorter candidates when the candidates
    have different token counts, because every extra token adds another
    negative term. Averaging removes that bias.

    The published PubMedQA candidates were ``" yes</s>"``, ``" no</s>"`` and
    ``" maybe</s>"``; each is two tokens, so this bias did not contribute to
    that particular collapse. It is fixed regardless because labels such as
    ``" maybe"`` split differently with other tokenizers, and the MMLU
    candidates are scored by the same function.
    """
    if set(raw_scores) != set(candidate_token_counts):
        raise ValueError("raw_scores and candidate_token_counts must have identical keys")

    normalised: dict[str, float] = {}
    for label, total in raw_scores.items():
        count = candidate_token_counts[label]
        if count <= 0:
            raise ValueError(f"candidate {label!r} has no scorable tokens")
        normalised[label] = total / count
    return normalised


def choose_label(
    scores: Mapping[str, float],
    order: Sequence[str] = PUBMEDQA_LABELS,
    tolerance: float = 1e-9,
) -> tuple[str, bool]:
    """Return ``(label, tied)`` for the highest-scoring candidate.

    A tie is reported explicitly instead of being hidden behind dict order.
    The winner is still deterministic (first label in ``order``), but callers
    record ``tied`` so that a collapsed scorer is visible in the report rather
    than silently producing an all-majority-label prediction set.
    """
    if not scores:
        raise ValueError("Cannot choose a label from an empty score mapping")
    ranked = sorted(order, key=lambda label: scores[label], reverse=True)
    best = ranked[0]
    tied = len(ranked) > 1 and abs(scores[ranked[0]] - scores[ranked[1]]) <= tolerance
    return best, tied


def confusion_matrix_counts(
    references: Sequence[str],
    predictions: Sequence[str],
    labels: Sequence[str] = PUBMEDQA_LABELS,
) -> list[list[int]]:
    """Row = reference label, column = predicted label (sklearn-compatible)."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for reference, prediction in zip(references, predictions):
        if reference not in index:
            raise ValueError(f"unknown reference label: {reference!r}")
        if prediction not in index:
            raise ValueError(f"unknown predicted label: {prediction!r}")
        matrix[index[reference]][index[prediction]] += 1
    return matrix


def accuracy(references: Sequence[str], predictions: Sequence[str]) -> float:
    if not references:
        return 0.0
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    correct = sum(
        1
        for reference, prediction in zip(references, predictions)
        if reference == prediction
    )
    return correct / len(references)


def _per_label_f1(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
) -> dict[str, tuple[float, float, float, int]]:
    """Return ``label -> (precision, recall, f1, support)`` from a confusion matrix."""
    results: dict[str, tuple[float, float, float, int]] = {}
    for row_index, label in enumerate(labels):
        true_positive = matrix[row_index][row_index]
        predicted_positive = sum(matrix[row][row_index] for row in range(len(labels)))
        support = sum(matrix[row_index])
        false_positive = predicted_positive - true_positive
        false_negative = support - true_positive

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        results[label] = (precision, recall, f1, support)
    return results


def macro_f1(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str] = PUBMEDQA_LABELS,
) -> float:
    if not labels:
        return 0.0
    per_label = _per_label_f1(matrix, labels)
    return sum(per_label[label][2] for label in labels) / len(labels)


def weighted_f1(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str] = PUBMEDQA_LABELS,
) -> float:
    per_label = _per_label_f1(matrix, labels)
    total_support = sum(per_label[label][3] for label in labels)
    if total_support == 0:
        return 0.0
    return sum(per_label[label][2] * per_label[label][3] for label in labels) / total_support


def prediction_counts(
    predictions: Iterable[str],
    labels: Sequence[str] = PUBMEDQA_LABELS,
) -> dict[str, int]:
    counter = Counter(predictions)
    return {label: counter.get(label, 0) for label in labels}


def reference_counts(
    references: Iterable[str],
    labels: Sequence[str] = PUBMEDQA_LABELS,
) -> dict[str, int]:
    counter = Counter(references)
    return {label: counter.get(label, 0) for label in labels}


def majority_class_rate(
    references: Iterable[str],
    labels: Sequence[str] = PUBMEDQA_LABELS,
) -> float:
    """Share of the most frequent reference label; the trivial-predictor baseline."""
    counts = reference_counts(references, labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


def is_collapsed(
    predictions: Iterable[str],
    labels: Sequence[str] = PUBMEDQA_LABELS,
) -> bool:
    """True when the predictor emitted only one distinct label."""
    counts = prediction_counts(predictions, labels)
    return sum(1 for value in counts.values() if value > 0) <= 1


def subject_sample_plan(
    subject_totals: Mapping[str, int],
    total_limit: int,
    subjects: Sequence[str],
) -> dict[str, int]:
    """Split an evaluation budget evenly across the requested subjects.

    Taking the first ``total_limit`` rows of a subject-filtered MMLU split
    spends the whole budget on the first subject in row order (``anatomy``)
    and leaves every other subject with zero samples. Distributing the budget
    over the subjects keeps each one represented, so the aggregate figure is
    not silently a single-subject number.
    """
    if total_limit < 1:
        raise ValueError("total_limit must be >= 1")
    if not subjects:
        raise ValueError("subjects must not be empty")

    per_subject = total_limit // len(subjects)
    remainder = total_limit % len(subjects)
    plan: dict[str, int] = {}
    for position, subject in enumerate(subjects):
        share = per_subject + (1 if position < remainder else 0)
        plan[subject] = min(max(share, 1), subject_totals.get(subject, 0))
    return plan


def select_subject_rows(
    row_subjects: Sequence[str],
    plan: Mapping[str, int],
) -> list[int]:
    """Return row indices that satisfy ``plan``, preserving the original order.

    Kept separate from ``subject_sample_plan`` so the selection can be tested
    against a row order without a dataset or a model. The published run did not
    use a plan at all: it kept the first rows of the filtered split, which are
    all ``anatomy``.
    """
    remaining = dict(plan)
    selected: list[int] = []
    for row_index, subject in enumerate(row_subjects):
        if remaining.get(subject, 0) > 0:
            selected.append(row_index)
            remaining[subject] -= 1
    return selected
