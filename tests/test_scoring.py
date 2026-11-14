"""CPU-only tests for candidate scoring and label choice.

These pin the causal-index contract and the length normalisation. They do not
reproduce the published collapse: that run came from the notebook, whose
scoring loop already used the correct ``prompt_length - 1`` offset, so the
off-by-one fixed here was present only in ``src/pubmedqa_loraplus/evaluation.py``.
"""

import pytest

from src.pubmedqa_loraplus.scoring import (
    choose_label,
    normalise_candidate_scores,
    score_candidate,
)


def test_first_candidate_token_is_scored_with_last_prompt_logits():
    """The candidate's first token must be read at prompt_length - 1.

    A causal LM's logits at position i predict the token at i + 1, so scoring
    from prompt_length skips the logit that decides which label the model
    wants and never reads it at all.
    """
    prompt_length = 5
    candidate_ids = [10, 11, 12]
    seen: list[tuple[int, int]] = []

    def log_prob_at(position: int, token_id: int) -> float:
        seen.append((position, token_id))
        return -1.0

    score_candidate(prompt_length, candidate_ids, log_prob_at)

    assert seen == [(4, 10), (5, 11), (6, 12)]


def test_score_candidate_sums_all_candidate_tokens():
    def log_prob_at(position: int, token_id: int) -> float:
        return -0.5

    assert score_candidate(3, [7, 8], log_prob_at) == pytest.approx(-1.0)


def test_score_candidate_rejects_empty_candidate():
    with pytest.raises(ValueError):
        score_candidate(3, [], lambda position, token_id: 0.0)


def test_score_candidate_rejects_empty_prompt():
    with pytest.raises(ValueError):
        score_candidate(0, [1], lambda position, token_id: 0.0)


def test_length_normalisation_prevents_multi_token_label_bias():
    """A shorter label must not win purely by having fewer tokens.

    With equal per-token probability, the summed log-probability of a
    three-token candidate is lower than a one-token candidate's, which would
    favour the short label regardless of content.
    """
    per_token_log_prob = -0.1
    raw = {"yes": per_token_log_prob, "maybe": per_token_log_prob * 3}
    counts = {"yes": 1, "maybe": 3}

    normalised = normalise_candidate_scores(raw, counts)

    assert normalised["yes"] == pytest.approx(-0.1)
    assert normalised["maybe"] == pytest.approx(-0.1)


def test_normalisation_requires_matching_keys():
    with pytest.raises(ValueError):
        normalise_candidate_scores({"yes": -1.0}, {"no": 1})


def test_normalisation_rejects_zero_token_candidate():
    with pytest.raises(ValueError):
        normalise_candidate_scores({"yes": -1.0}, {"yes": 0})


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


def test_causal_offset_decides_the_label_and_the_off_by_one_erases_it():
    """Prove the consequence of the old ``prompt_length + offset`` index.

    The stub pretends the last prompt position strongly prefers the ``yes``
    token and is neutral elsewhere. Reading from ``prompt_length - 1`` picks
    ``yes``; reading from ``prompt_length`` (the old behaviour) never looks at
    that position, so every candidate scores the same and the winner is just
    whichever label ``max`` sees first.
    """
    prompt_length = 4
    yes_token = 900
    scores_at = {3: {yes_token: -0.01, 901: -9.0, 902: -9.0}}

    def log_prob_at(position: int, token_id: int) -> float:
        return scores_at.get(position, {}).get(token_id, -5.0)

    candidates = {"yes": [yes_token], "no": [901], "maybe": [902]}

    fixed = {
        label: score_candidate(prompt_length, ids, log_prob_at)
        for label, ids in candidates.items()
    }
    buggy = {
        label: sum(
            log_prob_at(prompt_length + offset, token_id)
            for offset, token_id in enumerate(ids)
        )
        for label, ids in candidates.items()
    }

    assert choose_label(fixed) == ("yes", False)
    # The old index gives every label the identical score.
    assert len(set(buggy.values())) == 1
