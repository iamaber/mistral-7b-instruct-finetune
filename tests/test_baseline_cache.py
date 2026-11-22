"""The zero-shot baseline is cached per base-model setup and shared across sweep variants."""

from src.pubmedqa_loraplus.config import LoRAPlusConfig, baseline_cache_key


def _config(tmp_path, **overrides):
    fields = {
        "run_name": "test-run",
        "base_model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_model_path": tmp_path / "base_model",
        "dataset_id": "qiaojin/PubMedQA",
        "dataset_config": "pqa_labeled",
        "dataset_disk_path": tmp_path / "dataset",
        "output_root": tmp_path / "out",
        "reports_root": tmp_path / "reports",
    }
    return LoRAPlusConfig(**{**fields, **overrides})


def test_training_hyperparameters_share_one_baseline(tmp_path):
    base = baseline_cache_key(_config(tmp_path))

    assert base == baseline_cache_key(
        _config(
            tmp_path, run_name="other", learning_rate=5e-5, lora_r=8, pqa_artificial_samples=4000
        )
    )


def test_anything_the_baseline_depends_on_changes_the_key(tmp_path):
    base = baseline_cache_key(_config(tmp_path))

    for overrides in (
        {"base_model_revision": "abc"},
        {"target_format": "label"},
        {"max_length": 1024},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"mmlu_eval_samples": 30},
    ):
        assert baseline_cache_key(_config(tmp_path, **overrides)) != base, overrides
