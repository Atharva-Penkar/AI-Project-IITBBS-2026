"""Uncertainty in mastery estimates.

A single BKT probability does not say how sure we are of it. We run an
ensemble of BKT trackers whose parameters are sampled around the base
values. The mean is the estimate; the spread is the uncertainty. With little
evidence the copies disagree (high uncertainty); as evidence accumulates
they converge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from module3_student_model.bkt import BKTParams, BKTTracker


def _beta_around(rng: np.random.Generator, mean: float, concentration: float) -> float:
    mean = min(max(mean, 1e-3), 1 - 1e-3)
    return float(rng.beta(mean * concentration, (1.0 - mean) * concentration))


def sample_params(
    base: BKTParams, rng: np.random.Generator, init_concentration: float = 4.0, concentration: float = 30.0
) -> BKTParams:
    """Draw one plausible parameter set around `base`."""
    while True:
        guess = _beta_around(rng, base.p_guess, concentration)
        slip = _beta_around(rng, base.p_slip, concentration)
        if guess + slip < 0.95:
            break
    return BKTParams(
        p_init=_beta_around(rng, base.p_init, init_concentration),
        p_learn=_beta_around(rng, base.p_learn, concentration),
        p_guess=guess,
        p_slip=slip,
        p_guess_hint=max(guess, _beta_around(rng, base.p_guess_hint, concentration)),
    )


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@dataclass
class EnsembleTracker:
    """K BKT trackers with sampled parameters; reports mean and spread."""

    default_params: BKTParams = field(default_factory=BKTParams)
    concept_params: dict[str, BKTParams] = field(default_factory=dict)
    n_members: int = 15
    seed: int = 0
    members: list[BKTTracker] = field(init=False)
    _n_obs: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        concepts = list(self.concept_params)
        self.members = []
        for _ in range(self.n_members):
            self.members.append(
                BKTTracker(
                    default_params=sample_params(self.default_params, rng),
                    concept_params={c: sample_params(self.concept_params[c], rng) for c in concepts},
                )
            )

    def observe(
        self,
        student_id: str,
        concept_id: str,
        correct: bool | None = None,
        score: float | None = None,
        hint_used: bool = False,
    ) -> None:
        for m in self.members:
            m.observe(student_id, concept_id, correct=correct, score=score, hint_used=hint_used)
        key = (student_id, concept_id)
        self._n_obs[key] = self._n_obs.get(key, 0) + 1

    def expose(self, student_id: str, concept_id: str, strength: float) -> None:
        if strength <= 0:
            return
        for m in self.members:
            m.expose(student_id, concept_id, strength)

    def set_prior_scale(self, student_id: str, concept_id: str, scale: float) -> None:
        """Scale every member's prior, e.g. from prerequisite mastery."""
        for m in self.members:
            m.set_prior(student_id, concept_id, m.get(student_id, concept_id) * scale)

    def values(self, student_id: str, concept_id: str) -> np.ndarray:
        return np.array([m.get(student_id, concept_id) for m in self.members])

    def mean(self, student_id: str, concept_id: str) -> float:
        return float(self.values(student_id, concept_id).mean())

    def std(self, student_id: str, concept_id: str) -> float:
        return float(self.values(student_id, concept_id).std())

    def n_observations(self, student_id: str, concept_id: str) -> int:
        return self._n_obs.get((student_id, concept_id), 0)