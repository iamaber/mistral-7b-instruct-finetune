"""CPU-only tests for model-card generation.

The published cards presented accuracy 0.45 and "medical MMLU 0.16" as if they
were results, said nothing about the collapse, and shipped no usage snippet,
limitations, language or citation. These tests pin the honesty requirements.
"""

import json
from fnmatch import fnmatchcase
from pathlib import Path
from unittest.mock import Mock

import huggingface_hub
import pytest

from src.pubmedqa_loraplus.config import LoRAPlusConfig
from src.pubmedqa_loraplus.hub import build_model_card, push_artifact_to_hub

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
def test_push_uploads_only_model_files(tmp_path, monkeypatch, artifact):
    config = _config(
        tmp_path,
        adapter_hub_repo_id="owner/test-adapter",
        merged_hub_repo_id="owner/test-merged",
    )
    checkpoint = tmp_path / "checkpoint-40"
    checkpoint.mkdir()
    adapter_files = {"adapter_config.json", "adapter_model.safetensors"}
    merged_files = {
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    }
    tokenizer_files = {
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model.v3",
        "special_tokens_map.json",
        "chat_template.jinja",
    }
    trainer_files = {
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    }
    model_files = adapter_files if artifact == "adapter" else merged_files
    extra_files = merged_files if artifact == "adapter" else {"model.bin"}
    for filename in model_files | tokenizer_files | trainer_files | extra_files:
        (checkpoint / filename).touch()
    if artifact == "adapter":
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")

    api = Mock()
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=api))
    push_artifact_to_hub(config, artifact, source_dir=checkpoint)

    upload = api.upload_folder.call_args.kwargs
    patterns = upload["allow_patterns"]
    assert patterns
    assert not any(fnmatchcase(name, pattern) for name in trainer_files for pattern in patterns)
    matched_files = {
        path.name
        for path in Path(upload["folder_path"]).iterdir()
        if path.is_file() and any(fnmatchcase(path.name, pattern) for pattern in patterns)
    }
    assert matched_files == model_files | tokenizer_files | {"README.md"}


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_declares_experiment_status(config, artifact):
    card = build_model_card(config, artifact)

    assert "not a usable medical model" in card
    assert "Do not use it for clinical decision-making" in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_states_the_predictor_collapsed(config, artifact):
    card = build_model_card(config, artifact)

    assert "The predictor collapsed" in card
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
    assert "contains merged weights" in card
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


HEALTHY_REPORT = {
    "run_name": "pubmedqa-loraplus",
    "base_model_id": BASE_MODEL_ID,
    "dataset_id": "qiaojin/PubMedQA",
    "dataset_config": "pqa_labeled",
    "train_examples": 810,
    "validation_examples": 90,
    "eval_examples": 100,
    "training": {"global_step": 171, "train_metrics": {"train_loss": 0.41}},
    "training_summary": {
        "best_eval_loss": 0.44,
        "latest_eval_loss": 0.45,
        "train_runtime": 500.0,
    },
    "evaluation": {
        "pubmedqa": {
            "accuracy": 0.62,
            "macro_f1": 0.58,
            "weighted_f1": 0.61,
            "majority_class_rate": 0.45,
            "collapsed": False,
            "label_order": ["yes", "no", "maybe"],
            "confusion_matrix": [[30, 10, 5], [5, 30, 5], [2, 3, 10]],
            "reference_counts": {"yes": 45, "no": 40, "maybe": 15},
            "prediction_counts": {"yes": 37, "no": 43, "maybe": 20},
            "num_samples": 100,
            "warnings": [],
        },
        "mmlu_medical": {
            "overall_accuracy": 0.41,
            "covered_subjects": [
                "anatomy",
                "clinical_knowledge",
                "college_medicine",
                "medical_genetics",
                "professional_medicine",
                "virology",
            ],
            "subject_results": {
                "anatomy": {"correct": 4, "total": 9},
                "clinical_knowledge": {"correct": 3, "total": 9},
                "college_medicine": {"correct": 3, "total": 8},
                "medical_genetics": {"correct": 4, "total": 8},
                "professional_medicine": {"correct": 3, "total": 8},
                "virology": {"correct": 4, "total": 8},
            },
            "total_samples": 50,
        },
    },
}


@pytest.fixture
def healthy_config(tmp_path):
    reports_root = tmp_path / "reports"
    reports_root.mkdir(exist_ok=True)
    (reports_root / "pubmedqa-loraplus.json").write_text(
        json.dumps(HEALTHY_REPORT),
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
    )


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_healthy_run_card_does_not_claim_a_collapse(healthy_config, artifact):
    """The Limitations block must be derived from the report, not boilerplate.

    A non-collapsed, multi-subject run must not inherit the published run's
    "all predictions were the same label" or "only anatomy received samples"
    text.
    """
    card = build_model_card(healthy_config, artifact)

    assert "The predictor collapsed" not in card
    assert "same label" not in card
    assert "All 100 held-out PubMedQA predictions" not in card
    assert "one subject" not in card
    assert "Only `anatomy` received samples" not in card
    assert "No collapse in the recorded run" in card
    assert "covers 6 subjects" in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_healthy_run_card_keeps_the_real_numbers(healthy_config, artifact):
    card = build_model_card(healthy_config, artifact)

    assert "0.6200" in card
    assert "0.4100" in card
    assert "| 30 | 10 | 5 |" in card


@pytest.mark.parametrize("artifact", ["adapter", "merged"])
def test_card_without_a_report_claims_no_collapse(tmp_path, artifact):
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

    card = build_model_card(config, artifact)

    assert "The predictor collapsed" not in card
    assert "No evaluation was recorded" in card
    assert "one subject" not in card
    assert "No evaluation report was available" in card


def test_report_limitations_reference_their_own_sample_count(tmp_path):
    """Card text must quote the report's sample count, not a hardcoded 100."""
    report = json.loads(json.dumps(HEALTHY_REPORT))
    report["evaluation"]["pubmedqa"]["collapsed"] = True
    report["evaluation"]["pubmedqa"]["num_samples"] = 250
    reports_root = tmp_path / "reports"
    reports_root.mkdir(exist_ok=True)
    (reports_root / "pubmedqa-loraplus.json").write_text(
        json.dumps(report),
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

    assert "All 250 held-out PubMedQA predictions" in card
    assert "All 100 held-out PubMedQA predictions" not in card
