"""The single run-report schema consumed by the model-card generator.

There are three producers (``train_model``, the Modal notebook, and
``evaluate_artifact``) and one consumer (``hub.build_model_card``). When the
notebook wrote its training metadata to ``<run>_training.json`` while the card
read ``<run>.json``, the card's "Reported run details" table silently rendered
``not available`` for every row on the primary path. Building the payload in
one place keeps the producer and consumer from drifting apart again.
"""

from __future__ import annotations

from typing import Any, Mapping

REPORT_FIELDS = (
    "run_name",
    "base_model_id",
    "base_model_source",
    "dataset_id",
    "dataset_config",
    "dataset_disk_path",
    "train_examples",
    "validation_examples",
    "eval_examples",
    "training",
    "training_summary",
    "artifacts",
    "evaluation",
)

CARD_TABLE_FIELDS = (
    "train_examples",
    "eval_examples",
    "training",
    "training_summary",
)


def build_run_report(
    *,
    run_name: str,
    base_model_id: str,
    training: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    dataset_id: str | None = None,
    dataset_config: str | None = None,
    dataset_disk_path: str | None = None,
    base_model_source: str | None = None,
    train_examples: int | None = None,
    validation_examples: int | None = None,
    eval_examples: int | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the unified report consumed by ``build_model_card``."""
    return {
        "run_name": run_name,
        "base_model_id": base_model_id,
        "base_model_source": base_model_source,
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "dataset_disk_path": dataset_disk_path,
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "eval_examples": eval_examples,
        "training": dict(training),
        "training_summary": dict(training_summary),
        "artifacts": dict(artifacts) if artifacts is not None else None,
        "evaluation": dict(evaluation),
    }


def missing_card_fields(report: Mapping[str, Any]) -> list[str]:
    """Return the fields the card needs for its run-details table that are absent.

    ``validation_examples`` is optional because older reports predate it;
    everything else the table renders must be present for the row to show a
    value instead of ``not available``.
    """
    return [field for field in CARD_TABLE_FIELDS if not report.get(field)]
