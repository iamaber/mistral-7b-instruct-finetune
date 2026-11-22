"""CPU-only tests for label choice and the bootstrap intervals."""

import pytest

from src.pubmedqa_loraplus.scoring import (
    accuracy,
    bootstrap_interval,
    choose_label,
    macro_f1_score,
    max_prediction_share,
    paired_bootstrap_delta,
)


def test_choose_label_picks_highest_score():
    label, tied = choose_label({"yes": -0.1, "no": -0.2, "maybe": -0.3})
    assert label == "yes"
    assert tied is False


def test_choose_label_reports_ties_instead_of_hiding_them():
    label, tied = choose_label({"yes": -0.1, "no": -0.1, "maybe": -0.4})
    assert label == "yes"
    assert tied is True


def test_choose_label_requires_scores():
    with pytest.raises(ValueError):
        choose_label({})


REFERENCES = ["yes"] * 50 + ["no"] * 35 + ["maybe"] * 15
GOOD = ["yes"] * 45 + ["no"] * 5 + ["no"] * 30 + ["yes"] * 5 + ["maybe"] * 10 + ["no"] * 5
ALL_YES = ["yes"] * 100


def test_bootstrap_interval_brackets_the_point_estimate():
    low, high = bootstrap_interval(REFERENCES, GOOD, accuracy, samples=500, seed=0)

    assert low < accuracy(REFERENCES, GOOD) < high
    assert 0.0 <= low and high <= 1.0


def test_bootstrap_interval_is_reproducible_for_a_seed():
    first = bootstrap_interval(REFERENCES, GOOD, macro_f1_score, samples=200, seed=7)
    second = bootstrap_interval(REFERENCES, GOOD, macro_f1_score, samples=200, seed=7)

    assert first == second


def test_paired_delta_of_a_predictor_against_itself_is_exactly_zero():
    result = paired_bootstrap_delta(REFERENCES, GOOD, GOOD, accuracy, samples=200, seed=0)

    assert result["delta"] == 0.0
    assert result["interval"] == [0.0, 0.0]


def test_paired_delta_separates_a_real_model_from_the_majority_predictor():
    result = paired_bootstrap_delta(REFERENCES, GOOD, ALL_YES, macro_f1_score, 500, seed=0)

    assert result["delta"] > 0
    assert result["interval"][0] > 0


def test_max_prediction_share_flags_near_collapse_that_is_collapsed_misses():
    assert max_prediction_share(["yes"] * 99 + ["no"]) == pytest.approx(0.99)
    assert max_prediction_share([]) == 0.0
