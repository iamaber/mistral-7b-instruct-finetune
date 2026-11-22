"""CPU-only tests for adapter publish fixups.

The published adapter recorded the trainer-local directory
``/vol/pubmedqa-loraplus/base_model`` as its base model, so
``PeftModel.from_pretrained`` could not resolve the base from the Hub.
"""

import json

import pytest

from src.pubmedqa_loraplus.artifact import (
    prepare_adapter_for_publish,
    remove_trainer_sidecars,
    rewrite_base_model_reference,
)

HUB_ID = "mistralai/Mistral-7B-Instruct-v0.3"
TRAINER_PATH = "/vol/pubmedqa-loraplus/base_model"


def test_rewrite_replaces_trainer_path_with_hub_id(write_json):
    path = write_json(
        "adapter_config.json",
        {"base_model_name_or_path": TRAINER_PATH, "peft_type": "LORA", "r": 16},
    )

    payload = rewrite_base_model_reference(path, HUB_ID)

    assert payload["base_model_name_or_path"] == HUB_ID
    assert payload["r"] == 16
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["base_model_name_or_path"] == HUB_ID


def test_rewrite_is_idempotent(write_json):
    path = write_json("adapter_config.json", {"base_model_name_or_path": TRAINER_PATH})

    rewrite_base_model_reference(path, HUB_ID)
    payload = rewrite_base_model_reference(path, HUB_ID)

    assert payload["base_model_name_or_path"] == HUB_ID


def test_rewrite_rejects_local_looking_base_id(write_json):
    path = write_json("adapter_config.json", {"base_model_name_or_path": TRAINER_PATH})

    with pytest.raises(ValueError):
        rewrite_base_model_reference(path, "/vol/pubmedqa-loraplus/base_model")


def test_rewrite_rejects_existing_local_directory(write_json, tmp_path, monkeypatch):
    """A relative path that resolves to a real directory is trainer-local state."""
    (tmp_path / "models" / "base_model").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    path = write_json("adapter_config.json", {"base_model_name_or_path": TRAINER_PATH})

    with pytest.raises(ValueError):
        rewrite_base_model_reference(path, "models/base_model")


def test_rewrite_rejects_empty_base_id(write_json):
    path = write_json("adapter_config.json", {"base_model_name_or_path": TRAINER_PATH})

    with pytest.raises(ValueError):
        rewrite_base_model_reference(path, "")


def test_rewrite_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rewrite_base_model_reference(tmp_path / "missing.json", HUB_ID)


def test_remove_trainer_sidecars_deletes_training_args(tmp_path):
    (tmp_path / "training_args.bin").write_bytes(b"\x00")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"\x00")

    removed = remove_trainer_sidecars(tmp_path)

    assert removed == ["training_args.bin"]
    assert not (tmp_path / "training_args.bin").exists()
    assert (tmp_path / "adapter_model.safetensors").exists()


def test_remove_trainer_sidecars_is_noop_when_absent(tmp_path):
    assert remove_trainer_sidecars(tmp_path) == []


def test_prepare_adapter_for_publish_applies_both_fixups(write_json, tmp_path):
    config_path = write_json(
        "adapter_config.json",
        {"base_model_name_or_path": TRAINER_PATH},
    )
    (tmp_path / "training_args.bin").write_bytes(b"\x00")

    result = prepare_adapter_for_publish(tmp_path, HUB_ID)

    assert result["base_model_name_or_path"] == HUB_ID
    assert result["removed_files"] == ["training_args.bin"]
    assert not (tmp_path / "training_args.bin").exists()
    assert json.loads(config_path.read_text())["base_model_name_or_path"] == HUB_ID


def test_published_adapter_pins_the_base_revision_peft_loads(write_json, tmp_path):
    from peft import PeftConfig

    write_json(
        "adapter_config.json", {"base_model_name_or_path": TRAINER_PATH, "peft_type": "LORA"}
    )

    prepare_adapter_for_publish(tmp_path, HUB_ID, "c170c708c41dac9275d15a8fff4eca08d52bab71")

    loaded = PeftConfig.from_pretrained(str(tmp_path))
    assert loaded.base_model_name_or_path == HUB_ID
    assert loaded.revision == "c170c708c41dac9275d15a8fff4eca08d52bab71"
