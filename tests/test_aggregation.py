"""CPU-only tests for metric aggregation and collapse detection.

The published card reported accuracy 0.45 with every prediction "yes". The
aggregation has to make that visible rather than presenting 0.45 as a result.
"""

import pytest

from src.pubmedqa_loraplus.scoring import (
    PUBMEDQA_LABELS,
    accuracy,
    confusion_matrix_counts,
    is_collapsed,
    macro_f1,
    majority_class_rate,
    prediction_counts,
    reference_counts,
    subject_sample_plan,
    weighted_f1,
)

# The published test split: 45 yes, 40 no, 15 maybe.
REFERENCES = ["yes"] * 45 + ["no"] * 40 + ["maybe"] * 15


def test_published_collapse_reproduces_published_metrics():
    predictions = ["yes"] * 100
    matrix = confusion_matrix_counts(REFERENCES, predictions, PUBMEDQA_LABELS)

    assert matrix == [[45, 0, 0], [40, 0, 0], [15, 0, 0]]
    assert accuracy(REFERENCES, predictions) == pytest.approx(0.45)
    assert macro_f1(matrix, PUBMEDQA_LABELS) == pytest.approx(0.206896, abs=1e-6)
    assert weighted_f1(matrix, PUBMEDQA_LABELS) == pytest.approx(0.279310, abs=1e-6)


def test_collapsed_predictor_accuracy_equals_majority_rate():
    predictions = ["yes"] * 100

    assert is_collapsed(predictions) is True
    assert majority_class_rate(REFERENCES) == pytest.approx(0.45)
    assert accuracy(REFERENCES, predictions) == pytest.approx(
        majority_class_rate(REFERENCES)
    )


def test_non_collapsed_predictor_is_not_flagged():
    predictions = ["yes"] * 45 + ["no"] * 40 + ["maybe"] * 15

    assert is_collapsed(predictions) is False
    assert accuracy(REFERENCES, predictions) == pytest.approx(1.0)
    matrix = confusion_matrix_counts(REFERENCES, predictions, PUBMEDQA_LABELS)
    assert macro_f1(matrix, PUBMEDQA_LABELS) == pytest.approx(1.0)
    assert weighted_f1(matrix, PUBMEDQA_LABELS) == pytest.approx(1.0)


def test_single_prediction_is_treated_as_collapsed():
    assert is_collapsed(["maybe"]) is True


def test_confusion_matrix_is_reference_rows_by_prediction_columns():
    matrix = confusion_matrix_counts(
        ["yes", "no", "maybe"],
        ["no", "no", "yes"],
        PUBMEDQA_LABELS,
    )
    assert matrix == [[0, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_confusion_matrix_rejects_unknown_labels():
    with pytest.raises(ValueError):
        confusion_matrix_counts(["unknown"], ["yes"], PUBMEDQA_LABELS)


def test_confusion_matrix_requires_equal_lengths():
    with pytest.raises(ValueError):
        confusion_matrix_counts(["yes"], [], PUBMEDQA_LABELS)


def test_distributions_cover_every_label_even_at_zero():
    predictions = ["yes"] * 3

    assert prediction_counts(predictions) == {"yes": 3, "no": 0, "maybe": 0}
    assert reference_counts(REFERENCES) == {"yes": 45, "no": 40, "maybe": 15}


def test_accuracy_of_empty_evaluation_is_zero_not_division_error():
    assert accuracy([], []) == 0.0


def test_macro_f1_of_empty_matrix_is_zero():
    assert macro_f1([[0, 0, 0], [0, 0, 0], [0, 0, 0]], PUBMEDQA_LABELS) == 0.0


def test_subject_sample_plan_spreads_budget_across_subjects():
    """The published run spent all 50 MMLU samples on anatomy.

    Selecting the first 50 rows of a subject-filtered split yields
    anatomy=50 and 0 for the other five subjects.
    """
    subjects = (
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "medical_genetics",
        "professional_medicine",
        "virology",
    )
    totals = {subject: 100 for subject in subjects}

    plan = subject_sample_plan(totals, 50, subjects)

    assert plan == {
        "anatomy": 9,
        "clinical_knowledge": 9,
        "college_medicine": 8,
        "medical_genetics": 8,
        "professional_medicine": 8,
        "virology": 8,
    }
    assert sum(plan.values()) == 50
    assert all(count > 0 for count in plan.values())


def test_subject_sample_plan_respects_subject_size():
    plan = subject_sample_plan({"anatomy": 3, "virology": 100}, 10, ("anatomy", "virology"))
    assert plan == {"anatomy": 3, "virology": 5}


def test_subject_sample_plan_gives_every_subject_a_sample():
    subjects = ("anatomy", "virology", "medical_genetics")
    plan = subject_sample_plan({subject: 10 for subject in subjects}, 2, subjects)
    assert all(count >= 1 for count in plan.values())


def test_subject_sample_plan_validates_inputs():
    with pytest.raises(ValueError):
        subject_sample_plan({}, 0, ("anatomy",))
    with pytest.raises(ValueError):
        subject_sample_plan({}, 5, ())
