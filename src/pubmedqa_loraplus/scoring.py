"""Pure prediction-scoring and aggregation helpers.

Nothing here imports torch, datasets or sklearn, so the label decision and the
metric computation can be unit-tested on CPU without loading a model.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence

PUBMEDQA_LABELS = ("yes", "no", "maybe")


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
        1 for reference, prediction in zip(references, predictions) if reference == prediction
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
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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


def macro_f1_score(references: Sequence[str], predictions: Sequence[str]) -> float:
    return macro_f1(confusion_matrix_counts(references, predictions))


def max_prediction_share(predictions: Sequence[str]) -> float:
    """Share of the most frequent predicted label; near 1.0 means near-collapse."""
    if not predictions:
        return 0.0
    return max(Counter(predictions).values()) / len(predictions)


Metric = Callable[[Sequence[str], Sequence[str]], float]


def _percentile_interval(values: list[float], alpha: float) -> list[float]:
    ordered = sorted(values)
    low = ordered[int((alpha / 2) * (len(ordered) - 1))]
    high = ordered[int((1 - alpha / 2) * (len(ordered) - 1))]
    return [low, high]


def bootstrap_interval(
    references: Sequence[str],
    predictions: Sequence[str],
    metric: Metric,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> list[float]:
    """Percentile bootstrap interval of ``metric`` over resampled examples."""
    if samples < 1 or not references:
        raise ValueError("bootstrap needs at least one sample and one example")
    rng = random.Random(seed)
    size = len(references)
    values = []
    for _ in range(samples):
        rows = [rng.randrange(size) for _ in range(size)]
        values.append(metric([references[i] for i in rows], [predictions[i] for i in rows]))
    return _percentile_interval(values, alpha)


def paired_bootstrap_delta(
    references: Sequence[str],
    candidate: Sequence[str],
    baseline: Sequence[str],
    metric: Metric,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float | list[float]]:
    """``metric(candidate) - metric(baseline)`` with a paired bootstrap interval.

    Both predictors are resampled on the same rows, so the interval reflects the
    difference on shared questions rather than two independent noise bands.
    """
    if samples < 1 or not references:
        raise ValueError("bootstrap needs at least one sample and one example")
    rng = random.Random(seed)
    size = len(references)
    values = []
    for _ in range(samples):
        rows = [rng.randrange(size) for _ in range(size)]
        sampled = [references[i] for i in rows]
        values.append(
            metric(sampled, [candidate[i] for i in rows])
            - metric(sampled, [baseline[i] for i in rows])
        )
    return {
        "delta": metric(references, candidate) - metric(references, baseline),
        "interval": _percentile_interval(values, alpha),
    }


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
