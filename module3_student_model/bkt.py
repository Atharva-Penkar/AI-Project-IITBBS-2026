"""Bayesian Knowledge Tracing.

The latent state per (student, concept) is binary: known or not known.
After each observation we apply Bayes' rule, then the learning transition.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from shared.schema import ActionType


@dataclass(frozen=True)
class BKTParams:
    p_init: float = 0.2
    p_learn: float = 0.15
    p_guess: float = 0.2
    p_slip: float = 0.1
    p_guess_hint: float = 0.5

    def __post_init__(self) -> None:
        for name in ("p_init", "p_learn", "p_guess", "p_slip", "p_guess_hint"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.p_guess + self.p_slip >= 1.0:
            raise ValueError("p_guess + p_slip must be < 1, otherwise answers carry no information")


ACTION_LEARNING_WEIGHT: dict[ActionType, float] = {
    ActionType.LECTURE: 0.5,
    ActionType.EXAMPLE: 0.7,
    ActionType.PEER_DISCUSSION: 0.6,
    ActionType.PEER_LEARNING: 0.8,
    ActionType.TA_INTERVENTION: 1.0,
    ActionType.EXERCISE: 0.0,
    ActionType.REVISION: 0.8,
    ActionType.DIAGNOSTIC_QUIZ: 0.0,
    ActionType.ASSESSMENT: 0.0,
    ActionType.NEXT_CONCEPT: 0.0,
}
"""How strongly each action counts as a learning opportunity.

Quizzes and exercises get 0 here because their learning step is applied
when the answer itself is observed.
"""


def _clip(p: float) -> float:
    return min(max(p, 1e-6), 1.0 - 1e-6)


def posterior_after_answer(p_known: float, correct: bool, guess: float, slip: float) -> float:
    """P(known | answer) by Bayes' rule."""
    if correct:
        num = p_known * (1.0 - slip)
        den = num + (1.0 - p_known) * guess
    else:
        num = p_known * slip
        den = num + (1.0 - p_known) * (1.0 - guess)
    return _clip(num / den)


def posterior_after_score(p_known: float, score: float, guess: float, slip: float) -> float:
    """Soft update for partial credit: score is treated as a fractional correct answer."""
    like_known = score * (1.0 - slip) + (1.0 - score) * slip
    like_unknown = score * guess + (1.0 - score) * (1.0 - guess)
    num = p_known * like_known
    return _clip(num / (num + (1.0 - p_known) * like_unknown))


def apply_learning(p_known: float, p_learn: float) -> float:
    """Chance that an unknown concept becomes known after a learning opportunity."""
    return _clip(p_known + (1.0 - p_known) * p_learn)


def prior_from_prerequisites(
    base_prior: float, prerequisite_mastery: Sequence[float], weight: float = 0.5
) -> float:
    """Optional: shrink a concept's starting prior when its prerequisites are weak."""
    if not prerequisite_mastery:
        return base_prior
    avg = sum(prerequisite_mastery) / len(prerequisite_mastery)
    return _clip(base_prior * ((1.0 - weight) + weight * avg))


@dataclass
class BKTTracker:
    """Point estimates of P(known) for every (student, concept)."""

    default_params: BKTParams = field(default_factory=BKTParams)
    concept_params: dict[str, BKTParams] = field(default_factory=dict)
    _p: dict[tuple[str, str], float] = field(default_factory=dict)

    def params(self, concept_id: str) -> BKTParams:
        return self.concept_params.get(concept_id, self.default_params)

    def get(self, student_id: str, concept_id: str) -> float:
        key = (student_id, concept_id)
        if key not in self._p:
            self._p[key] = self.params(concept_id).p_init
        return self._p[key]

    def set_prior(self, student_id: str, concept_id: str, p: float) -> None:
        self._p[(student_id, concept_id)] = _clip(p)

    def observe(
        self,
        student_id: str,
        concept_id: str,
        correct: bool | None = None,
        score: float | None = None,
        hint_used: bool = False,
    ) -> float:
        """Update after a quiz answer or exercise, then apply the learning step."""
        if correct is None and score is None:
            raise ValueError("observe() needs correct or score")
        prm = self.params(concept_id)
        guess = prm.p_guess_hint if hint_used else prm.p_guess
        p = self.get(student_id, concept_id)
        if score is not None:
            p = posterior_after_score(p, score, guess, prm.p_slip)
        else:
            p = posterior_after_answer(p, bool(correct), guess, prm.p_slip)
        p = apply_learning(p, prm.p_learn)
        self._p[(student_id, concept_id)] = p
        return p

    def expose(self, student_id: str, concept_id: str, strength: float) -> float:
        """Learning step only, for instruction with no observed answer."""
        prm = self.params(concept_id)
        p = apply_learning(self.get(student_id, concept_id), prm.p_learn * strength)
        self._p[(student_id, concept_id)] = p
        return p


def sequence_log_likelihood(sequence: Iterable[bool], params: BKTParams) -> float:
    """Log P(answer sequence | params) via the BKT forward pass."""
    p = params.p_init
    ll = 0.0
    for correct in sequence:
        p_correct = p * (1.0 - params.p_slip) + (1.0 - p) * params.p_guess
        ll += math.log(p_correct if correct else 1.0 - p_correct)
        p = posterior_after_answer(p, correct, params.p_guess, params.p_slip)
        p = apply_learning(p, params.p_learn)
    return ll


def fit_params_grid(
    sequences: Sequence[Sequence[bool]],
    p_init_grid: Sequence[float] = (0.05, 0.15, 0.3, 0.5, 0.7),
    p_learn_grid: Sequence[float] = (0.02, 0.05, 0.1, 0.2, 0.3),
    p_guess_grid: Sequence[float] = (0.05, 0.15, 0.25, 0.35),
    p_slip_grid: Sequence[float] = (0.02, 0.05, 0.1, 0.2),
) -> tuple[BKTParams, float]:
    """Grid search for the parameters that best explain the answer sequences."""
    best: tuple[BKTParams, float] | None = None
    for p_init, p_learn, p_guess, p_slip in itertools.product(
        p_init_grid, p_learn_grid, p_guess_grid, p_slip_grid
    ):
        if p_guess + p_slip >= 1.0:
            continue
        prm = BKTParams(p_init=p_init, p_learn=p_learn, p_guess=p_guess, p_slip=p_slip)
        ll = sum(sequence_log_likelihood(seq, prm) for seq in sequences)
        if best is None or ll > best[1]:
            best = (prm, ll)
    assert best is not None
    return best