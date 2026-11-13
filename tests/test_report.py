"""CPU-only tests for the unified run-report schema.

The card reads ``reports/<run_name>.json`` and renders ``train_examples``,
``training`` and ``training_summary``. The notebook used to write its training
metadata to ``<run_name>_training.json`` and only the evaluation payload to
``<run_name>.json``, so the card's run-details table was always
``not available`` on the primary path.
"""

import json

from src.pubmedqa_loraplus.config import LoRAPlusConfig
from src.pubmedqa_loraplus.hub import build_model_card
from src.pubmedqa_loraplus.report import (
    CARD_TABLE_FIELDS,
    REPORT_FIELDS,
    build_run_report,
    missing_card_fields,
)

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def _report(**overrides):
    payload = {
        "run_name": "pubmedqa-loraplus",
        "base_model_id": BASE_MODEL_ID,
        "base_model_source": "/vol/pubmedqa-loraplus/base_model",
        "dataset_id": "qiaojin/PubMedQA",
        "dataset_config": "pqa_labeled",
        "dataset_disk_path": "/vol/pubmedqa-loraplus/dataset",
        "train_examples": 810,
        "validation_examples": 90,
        "eval_examples": 100,
        "training": {
            "global_step": 171,
            "train_metrics": {"train_loss": 0.4810},
        },
        "training_summary": {
            "best_eval_loss": 0.5188,
            "latest_eval_loss": 0.5188,
            "train_runtime": 566.5676,
        },
        "artifacts": {"adapter": "/vol/pubmedqa-loraplus/runs/x/adapter"},
        "evaluation": {
            "pubmedqa": {
                "accuracy": 0.45,
                "macro_f1": 0.2069,
                "weighted_f1": 0.2793,
                "majority_class_rate": 0.45,
                "collapsed": True,
                "label_order": ["yes", "no", "maybe"],
                "confusion_matrix": [[45, 0, 0], [40, 0, 0], [15, 0, 0]],
                "prediction_counts": {"yes": 100, "no": 0, "maybe": 0},
                "num_samples": 100,
            },
            "mmlu_medical": {
                "overall_accuracy": 0.16,
                "covered_subjects": ["anatomy"],
                "subject_results": {"anatomy": {"correct": 8, "total": 50}},
                "total_samples": 50,
            },
        },
    }
    payload.update(overrides)
    return payload


def test_builder_emits_every_declared_field():
    report = build_run_report(
        run_name="pubmedqa-loraplus",
        base_model_id=BASE_MODEL_ID,
        training={"global_step": 1},
        training_summary={"best_eval_loss": 0.5},
        evaluation={},
    )

    assert set(REPORT_FIELDS) <= set(report)


def test_builder_has_no_missing_card_fields_when_fully_populated():
    report = build_run_report(
        run_name="pubmedqa-loraplus",
        base_model_id=BASE_MODEL_ID,
        training={"global_step": 1, "train_metrics": {"train_loss": 0.4}},
        training_summary={"best_eval_loss": 0.5, "latest_eval_loss": 0.6},
        evaluation={},
        train_examples=810,
        eval_examples=100,
    )

    assert missing_card_fields(report) == []


def test_missing_card_fields_reports_absent_training_summary():
    report = build_run_report(
        run_name="pubmedqa-loraplus",
        base_model_id=BASE_MODEL_ID,
        training={"global_step": 1},
        training_summary={},
        evaluation={},
        train_examples=810,
        eval_examples=100,
    )

    assert "training_summary" in missing_card_fields(report)


def test_card_renders_run_details_from_the_shared_report(tmp_path):
    """The card's run-details table must not fall back to `not available`.

    This is the exact failure the notebook hit: training metadata lived in a
    different file, so every row rendered "not available".
    """
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    (reports_root / "pubmedqa-loraplus.json").write_text(
        json.dumps(_report()),
        encoding="utf-8",
    )
    config = LoRAPlusConfig(
        run_name="pubmedqa-loraplus",
        base_model_id=BASE_MODEL_ID,
        base_model_path=tmp_path / "base_model",
        dataset_id="qiaojin/PubMedQA",
        dataset_config="pqa_labeled",
        dataset_disk_path=tmp_path / "dataset",
        output_root=tmp_path / "out",
        reports_root=reports_root,
    )

    card = build_model_card(config, "adapter")

    assert "### Reported run details" in card
    assert "| Train examples | 810 |" in card
    assert "| Eval examples | 100 |" in card
    assert "| Global step | 171 |" in card
    assert "| Train loss (final) | 0.4810 |" in card
    assert "| Best eval loss | 0.5188 |" in card
    assert "| Latest eval loss | 0.5188 |" in card
    assert "| Train runtime (s) | 566.5676 |" in card
    assert "not available" not in card


def test_declared_card_table_fields_are_the_ones_the_card_reads():
    assert CARD_TABLE_FIELDS == (
        "train_examples",
        "eval_examples",
        "training",
        "training_summary",
    )
