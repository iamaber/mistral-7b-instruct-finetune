"""CPU-only tests for the official test split, class balancing and the step guard.

The published run reported a private 100-question re-split of the labeled pool
(not PubMedQA's test set) and, with ~150 optimizer steps against
``eval_steps=200``, never ran checkpoint selection at all.
"""

from collections import Counter

import pytest
from datasets import Dataset

from src.pubmedqa_loraplus.config import LoRAPlusConfig, planned_optimizer_steps
from src.pubmedqa_loraplus.data import build_train_dataset, load_pubmedqa
from src.pubmedqa_loraplus.splits import balanced_indices, official_test_labels

TRUTH = official_test_labels()


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


def _save_labeled(tmp_path, truth=TRUTH):
    pool_labels = ["yes"] * 276 + ["no"] * 169 + ["maybe"] * 55
    pubids = [int(pubid) for pubid in truth] + [90_000_000 + i for i in range(500)]
    decisions = list(truth.values()) + pool_labels
    Dataset.from_dict(
        {
            "pubid": pubids,
            "question": [f"question {pubid}" for pubid in pubids],
            "context": [{"contexts": [f"abstract {pubid}"]} for pubid in pubids],
            "long_answer": [f"conclusion {pubid}" for pubid in pubids],
            "final_decision": decisions,
        }
    ).save_to_disk(str(tmp_path / "dataset"))


def test_the_vendored_ground_truth_is_the_official_500(tmp_path):
    assert len(TRUTH) == 500
    assert Counter(TRUTH.values()) == {"yes": 276, "no": 169, "maybe": 55}


def test_test_split_is_exactly_the_official_test_set(tmp_path):
    _save_labeled(tmp_path)

    dataset = load_pubmedqa(_config(tmp_path))

    assert set(dataset["test"]["pubid"]) == set(TRUTH)
    assert dict(zip(dataset["test"]["pubid"], dataset["test"]["final_decision"])) == TRUTH


def test_train_and_validation_partition_the_remaining_pool(tmp_path):
    _save_labeled(tmp_path)

    dataset = load_pubmedqa(_config(tmp_path))

    train, validation = set(dataset["train"]["pubid"]), set(dataset["validation"]["pubid"])
    assert train.isdisjoint(validation)
    assert (train | validation).isdisjoint(TRUTH)
    assert len(train) + len(validation) == 500


def test_validation_is_label_stratified(tmp_path):
    _save_labeled(tmp_path)

    dataset = load_pubmedqa(_config(tmp_path, validation_size=0.1))

    assert Counter(dataset["validation"]["final_decision"]) == {"yes": 28, "no": 17, "maybe": 6}


def test_split_is_deterministic_for_a_fixed_seed(tmp_path):
    _save_labeled(tmp_path)

    first = load_pubmedqa(_config(tmp_path, seed=42))
    second = load_pubmedqa(_config(tmp_path, seed=42))

    assert first["validation"]["pubid"] == second["validation"]["pubid"]


def test_a_dataset_missing_official_test_questions_is_rejected(tmp_path):
    partial = dict(list(TRUTH.items())[:499])
    _save_labeled(tmp_path, truth=partial)

    with pytest.raises(ValueError, match="official test PMIDs"):
        load_pubmedqa(_config(tmp_path))


def test_a_label_that_disagrees_with_the_ground_truth_is_rejected(tmp_path):
    pubid = next(iter(TRUTH))
    wrong = {**TRUTH, pubid: "maybe" if TRUTH[pubid] != "maybe" else "yes"}
    _save_labeled(tmp_path, truth=wrong)

    with pytest.raises(ValueError, match="disagree"):
        load_pubmedqa(_config(tmp_path))


def test_balanced_train_set_has_equal_classes_and_keeps_every_example(tmp_path):
    _save_labeled(tmp_path)
    dataset = load_pubmedqa(_config(tmp_path))

    train = build_train_dataset(dataset, _config(tmp_path, balance_train_classes=True))

    counts = Counter(train["final_decision"])
    assert len(set(counts.values())) == 1
    assert set(train["pubid"]) == set(dataset["train"]["pubid"])


def test_balanced_indices_repeat_minority_examples_evenly():
    plan = balanced_indices(["yes"] * 7 + ["maybe"] * 3, seed=0)

    repeats = Counter(plan)
    assert Counter(["yes" if i < 7 else "maybe" for i in plan]) == {"yes": 7, "maybe": 7}
    assert sorted(repeats[i] for i in range(7, 10)) == [2, 2, 3]


def test_published_schedule_is_rejected_because_selection_never_ran(tmp_path):
    published = _config(
        tmp_path,
        train_batch_size=4,
        gradient_accumulation_steps=4,
        num_epochs=3,
        eval_steps=200,
        save_steps=200,
    )

    with pytest.raises(ValueError, match="never evaluate"):
        planned_optimizer_steps(published, train_examples=810)


def test_default_schedule_evaluates_several_times(tmp_path):
    config = _config(tmp_path)

    steps = planned_optimizer_steps(config, train_examples=744)

    assert steps // config.eval_steps >= 5


def test_saving_less_often_than_evaluating_is_rejected(tmp_path):
    config = _config(tmp_path, eval_steps=20, save_steps=40)

    with pytest.raises(ValueError, match="save_steps must equal eval_steps"):
        planned_optimizer_steps(config, train_examples=744)
