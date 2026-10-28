import random

import pytest

from module3_student_model.bkt import (
    BKTParams,
    BKTTracker,
    apply_learning,
    fit_params_grid,
    posterior_after_answer,
    posterior_after_score,
)


def test_params_reject_uninformative_guess_and_slip():
    with pytest.raises(ValueError):
        BKTParams(p_guess=0.6, p_slip=0.5)


def test_probabilities_stay_in_bounds():
    t = BKTTracker()
    for correct in [True] * 30 + [False] * 30:
        p = t.observe("s", "c", correct=correct)
        assert 0.0 < p < 1.0


def test_repeated_correct_answers_raise_mastery():
    t = BKTTracker()
    values = [t.observe("s", "c", correct=True) for _ in range(5)]
    assert values == sorted(values)
    assert values[-1] > 0.9


def test_lucky_guess_moves_weak_student_only_a_little():
    p = posterior_after_answer(0.1, correct=True, guess=0.25, slip=0.1)
    assert 0.1 < p < 0.4


def test_slip_moves_strong_student_only_a_little():
    p = posterior_after_answer(0.95, correct=False, guess=0.2, slip=0.1)
    assert 0.6 < p < 0.95


def test_correct_with_hint_is_weaker_evidence():
    with_hint = BKTTracker().observe("s", "c", correct=True, hint_used=True)
    without = BKTTracker().observe("s", "c", correct=True, hint_used=False)
    assert with_hint < without


def test_partial_credit_is_monotonic_and_half_is_neutral():
    low = posterior_after_score(0.4, 0.2, guess=0.2, slip=0.1)
    mid = posterior_after_score(0.4, 0.5, guess=0.2, slip=0.1)
    high = posterior_after_score(0.4, 0.9, guess=0.2, slip=0.1)
    assert low < mid < high
    assert mid == pytest.approx(0.4, abs=1e-6)


def test_learning_step_never_decreases_mastery():
    assert apply_learning(0.3, 0.2) == pytest.approx(0.44)


def test_grid_fit_recovers_rough_parameters():
    true = BKTParams(p_init=0.3, p_learn=0.2, p_guess=0.15, p_slip=0.05)
    rng = random.Random(0)
    sequences = []
    for _ in range(150):
        known = rng.random() < true.p_init
        seq = []
        for _ in range(8):
            seq.append(rng.random() > true.p_slip if known else rng.random() < true.p_guess)
            known = known or rng.random() < true.p_learn
        sequences.append(seq)
    fitted, _ = fit_params_grid(sequences)
    assert abs(fitted.p_init - true.p_init) <= 0.15
    assert abs(fitted.p_learn - true.p_learn) <= 0.1
    assert abs(fitted.p_guess - true.p_guess) <= 0.1
    assert abs(fitted.p_slip - true.p_slip) <= 0.1