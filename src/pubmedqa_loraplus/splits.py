"""Split and sampling plans over plain label lists.

Kept free of torch/datasets imports so the plans can be unit-tested on CPU.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

OFFICIAL_TEST_SPLIT = "pqal_official_test_500"
_GROUND_TRUTH = Path(__file__).with_name("pqal_test_ground_truth.json")


def official_test_labels() -> dict[str, str]:
    """PubMedQA's fixed 500-question test set as ``{pubid: decision}``.

    Vendored verbatim from ``data/test_ground_truth.json`` in
    https://github.com/pubmedqa/pubmedqa (MIT). ``qiaojin/PubMedQA`` ships all
    1000 labeled questions as one ``train`` split, so the official test set has
    to be selected by PMID.
    """
    return json.loads(_GROUND_TRUTH.read_text(encoding="utf-8"))


def _indices_by_label(labels: Sequence[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    return dict(sorted(grouped.items()))


def stratified_holdout(labels: Sequence[str], fraction: float, seed: int) -> list[int]:
    """Indices of a label-stratified ``fraction`` of ``labels``, sorted."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    rng = random.Random(seed)
    held_out: list[int] = []
    for indices in _indices_by_label(labels).values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        held_out.extend(shuffled[: max(1, round(fraction * len(shuffled)))])
    return sorted(held_out)


def balanced_indices(labels: Sequence[str], seed: int) -> list[int]:
    """Oversample every label up to the majority count, then shuffle.

    Each minority example is repeated as evenly as possible (whole passes over a
    shuffled copy, then a partial pass), so no example is dropped and none is
    repeated more than once beyond the others of its label.
    """
    rng = random.Random(seed)
    grouped = _indices_by_label(labels)
    if not grouped:
        return []
    target = max(len(indices) for indices in grouped.values())
    plan: list[int] = []
    for indices in grouped.values():
        order = list(indices)
        rng.shuffle(order)
        whole, partial = divmod(target, len(order))
        plan.extend(order * whole + order[:partial])
    rng.shuffle(plan)
    return plan


def class_capped_sample(
    labels: Sequence[str],
    per_label: int,
    allowed: Sequence[str],
    seed: int,
) -> list[int]:
    """Up to ``per_label`` seeded indices for each label in ``allowed``, sorted."""
    rng = random.Random(seed)
    chosen: list[int] = []
    for label, indices in _indices_by_label(labels).items():
        if label not in allowed:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        chosen.extend(shuffled[:per_label])
    return sorted(chosen)
