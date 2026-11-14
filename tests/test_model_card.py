"""CPU-only tests for model-card generation.

The published cards presented accuracy 0.45 and "medical MMLU 0.16" as if they
were results, said nothing about the collapse, and shipped no usage snippet,
limitations, language or citation. These tests pin the honesty requirements.
"""

import json

import pytest

from src.pubmedqa_loraplus.config import LoRAPlusConfig
from src.pubmedqa_loraplus.hub import build_model_card

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

PUBLISHED_REPORT = {
    "run_name": "pubmedqa-loraplus",
    "base_model_id": BASE_MODEL_ID,
    "dataset_id": "qiaojin/PubMedQA",
    "dataset_config": "pqa_labeled",
    "train_examples": 900,
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
    "evaluation": {
        "pubmedqa": {
            "accuracy": 0.45,
            "macro_f1": 0.2069,
            "weighted_f1": 0.2793,
            "majority_class_rate": 0.45,
            "collapsed": True,
            "label_order": ["yes", "no", "maybe"],
            "confusion_matrix": [[45, 0, 0], [40, 0, 0], [15, 0, 0]],
            "reference_counts": {"yes": 45, "no": 40, "maybe": 15},
            "prediction_counts": {"yes": 100, "no": 0, "maybe": 0},
            "num_samples": 100,
            "warnings": [],
        },
        "mmlu_medical": {
            "overall_accuracy": 0.16,
            "subject_results": {
                "anatomy": {"correct": 8, "total": 50},
                "clinical_knowledge": {"correct": 0, "total": 0},
                "college_medicine": {"correct": 0, "total": 0},
                "medical_genetics": {"correct": 0, "total": 0},
                "professional_medicine": {"correct": 0, "total": 0},
                "virology": {"correct": 0, "total": 0},
            },
            "total_samples": 50,
        },
    },
}


def _config(tmp_path, **overrides):
    reports_root = tmp_path / "reports"
    reports_root.mkdir(exist_ok=True)
    (reports_root / "pubmedqa-loraplus.json").write_text(
        json.dumps(PUBLISHED_REPORT),
        encoding="utf-8",
    )
    return LoRAPlusConfig(
        run_name="pubmedqa-loraplus",
        base_model_id=BASE_MODEL_ID,
        base_model_path=tmp_path / "base_model",
        dataset_id="qiaojin/PubMedQA",
        dataset_config="pqa_labeled",
        dataset_disk_path=tmp_path / "dataset",
        output_root=tmp_path / "out",
        reports_root=reports_root,
        **overrides,
    )


@pytest.fixture
def config(tmp_path):
    return _config(tmp_path)


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_declares_experiment_status(config, artifact):
    card = build_model_card(config, artifact)

    assert "not a usable medical model" in card
    assert "Do not use it for clinical decision-making" in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_states_the_predictor_collapsed(config, artifact):
    card = build_model_card(config, artifact)

    assert "collapsed predictor" in card
    assert "All 100 held-out PubMedQA predictions were the same label" in card
    assert "majority class" in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_labels_mmlu_as_anatomy_only(config, artifact):
    card = build_model_card(config, artifact)

    assert "covers `anatomy` only" in card
    assert "single-subject accuracy" in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_says_metrics_were_computed_once_and_copied(config, artifact):
    card = build_model_card(config, artifact)

    assert "The metrics were computed once" in card
    assert "copied into both cards" in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_keeps_published_numbers(config, artifact):
    card = build_model_card(config, artifact)

    assert "0.4500" in card
    assert "0.2069" in card
    assert "0.2793" in card
    assert "0.1600" in card
    assert "| 45 | 0 | 0 |" in card
    assert "0.5188" in card
    assert "0.4810" in card
    assert "566.5676" in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_does_not_claim_a_cause_that_was_not_established(config, artifact):
    card = build_model_card(config, artifact)

    assert "The cause was not established" in card


def test_adapter_card_names_its_own_repo_in_the_snippet(tmp_path):
    config = _config(tmp_path, adapter_hub_repo_id="owner/some-adapter")

    card = build_model_card(config, "adapter")

    assert 'PeftModel.from_pretrained(base_model, "owner/some-adapter")' in card
    assert "<this-repo-id>" not in card


def test_merged_card_names_its_own_repo_in_the_snippet(tmp_path):
    config = _config(tmp_path, merged_hub_repo_id="owner/some-merged")

    card = build_model_card(config, "merged")

    assert 'AutoModelForCausalLM.from_pretrained(\n    "owner/some-merged",' in card
    assert "<this-repo-id>" not in card


def test_card_falls_back_to_a_placeholder_without_a_repo_id(config):
    card = build_model_card(config, "adapter")

    assert "<this-repo-id>" in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_declares_english_language(config, artifact):
    card = build_model_card(config, artifact)

    assert "language:\n  - en" in card


def test_adapter_card_uses_peft_usage_snippet(config):
    card = build_model_card(config, "adapter")

    assert "library_name: peft" in card
    assert "from peft import PeftModel" in card
    assert "LoRA adapter, not full model weights" in card


def test_merged_card_uses_transformers_usage_snippet(config):
    card = build_model_card(config, "merged")

    assert "library_name: transformers" in card
    assert "merged full-precision weights" in card
    assert "from peft import PeftModel" not in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_has_required_sections(config, artifact):
    card = build_model_card(config, artifact)

    for heading in (
        "## Intended use",
        "## Out-of-scope use",
        "## Usage",
        "## Training data",
        "## Limitations",
        "## Licence",
        "## Citation",
    ):
        assert heading in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_cites_pubmedqa_and_base_licence(config, artifact):
    card = build_model_card(config, artifact)

    assert "jin2019pubmedqa" in card
    assert "aclanthology.org/D19-1259" in card
    assert "Apache-2.0" in card
    assert "PubMedQA is MIT-licensed" in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_declares_base_model_in_frontmatter(config, artifact):
    card = build_model_card(config, artifact)

    assert f"base_model: {BASE_MODEL_ID}" in card
    assert "  - qiaojin/PubMedQA" in card


def test_card_without_report_reports_no_metrics(tmp_path):
    config = LoRAPlusConfig(
        run_name="missing-run",
        base_model_id=BASE_MODEL_ID,
        base_model_path=tmp_path / "base_model",
        dataset_id="qiaojin/PubMedQA",
        dataset_config="pqa_labeled",
        dataset_disk_path=tmp_path / "dataset",
        output_root=tmp_path / "out",
        reports_root=tmp_path / "reports",
    )

    card = build_model_card(config, "adapter")

    assert "No evaluation report was found" in card
    assert "does not borrow metrics from another run" in card
    assert "Accuracy |" not in card
