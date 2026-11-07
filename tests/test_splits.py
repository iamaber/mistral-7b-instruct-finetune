"""CPU-only tests for the train/validation/test split.

The published run selected its checkpoint with ``load_best_model_at_end`` on
``eval_loss`` and then reported PubMedQA on the same 100 examples, so model
selection leaked into the headline number. The split must keep the
validation set used for selection disjoint from the reported test set.
"""

import pytest
from datasets import Dataset, load_from_disk

from src.pubmedqa_loraplus.config import LoRAPlusConfig
from src.pubmedqa_loraplus.data import load_pubmedqa


def _config(tmp_path, **overrides):
    return LoRAPlusConfig(
        run_name="test-run",
        base_model_id="mistralai/Mistral-7B-Instruct-v0.3",
        base_model_path=tmp_path / "base_model",
        dataset_id="qiaojin/PubMedQA",
        dataset_config="pqa_labeled",
        dataset_disk_path=tmp_path / "dataset",
        output_root=tmp_path / "out",
        reports_root=tmp_path / "reports",
        **overrides,
    )


@pytest.fixture
def cached_dataset(tmp_path):
    dataset = Dataset.from_dict(
        {
            "pubid": [str(index) for index in range(1000)],
            "question": [f"question {index}" for index in range(1000)],
            "context": [{"contexts": [f"abstract {index}"]} for index in range(1000)],
            "final_decision": ["yes", "no", "maybe"] * 333 + ["yes"],
        }
    )
    destination = tmp_path / "dataset"
    dataset.save_to_disk(str(destination))
    return load_from_disk(str(destination))


def test_split_has_train_validation_and_test(tmp_path, cached_dataset):
    dataset = load_pubmedqa(_config(tmp_path))

    assert set(dataset) == {"train", "validation", "test"}


def test_validation_is_disjoint_from_the_reported_test_set(tmp_path, cached_dataset):
    dataset = load_pubmedqa(_config(tmp_path))

    validation_ids = set(dataset["validation"]["pubid"])
    test_ids = set(dataset["test"]["pubid"])
    train_ids = set(dataset["train"]["pubid"])

    assert validation_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(train_ids)
    assert train_ids.isdisjoint(test_ids)


def test_split_covers_every_example_exactly_once(tmp_path, cached_dataset):
    dataset = load_pubmedqa(_config(tmp_path))

    total = sum(len(dataset[split]) for split in ("train", "validation", "test"))
    assert total == len(cached_dataset)


def test_split_sizes_follow_the_configured_fractions(tmp_path, cached_dataset):
    dataset = load_pubmedqa(_config(tmp_path, test_size=0.1, validation_size=0.1))

    assert len(dataset["test"]) == 100
    assert len(dataset["validation"]) == 90
    assert len(dataset["train"]) == 810


def test_split_is_deterministic_for_a_fixed_seed(tmp_path, cached_dataset):
    first = load_pubmedqa(_config(tmp_path, seed=42))
    second = load_pubmedqa(_config(tmp_path, seed=42))

    assert list(first["test"]["pubid"]) == list(second["test"]["pubid"])
    assert list(first["validation"]["pubid"]) == list(second["validation"]["pubid"])
