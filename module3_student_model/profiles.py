"""Student profiles, archetypes and ready-made classrooms for evaluation scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from shared.schema import CourseCatalog


class StudentProfile(BaseModel):
    """Ground-truth description of one simulated student."""

    student_id: str
    archetype: str = "average"
    prior_knowledge: dict[str, float] = Field(default_factory=dict)
    misconceptions: list[str] = Field(default_factory=list)
    learning_speed: float = 1.0
    confidence: float = 0.5
    participation_tendency: float = 0.5
    engagement: float = 0.7
    guess: float = 0.2
    slip: float = 0.1


@dataclass(frozen=True)
class Archetype:
    prior: float
    learning_speed: float
    confidence: float
    participation: float
    engagement: float
    misconception_rate: float
    guess: float = 0.2
    slip: float = 0.1


ARCHETYPES: dict[str, Archetype] = {
    "strong": Archetype(0.80, 1.3, 0.75, 0.50, 0.85, 0.00, slip=0.05),
    "average": Archetype(0.40, 1.0, 0.55, 0.35, 0.70, 0.15),
    "struggling": Archetype(0.10, 0.6, 0.35, 0.20, 0.55, 0.30, slip=0.15),
    "quiet_knower": Archetype(0.80, 1.2, 0.40, 0.05, 0.75, 0.00, slip=0.05),
    "dominant": Archetype(0.50, 1.0, 0.85, 0.90, 0.90, 0.15),
    "confident_wrong": Archetype(0.25, 0.9, 0.90, 0.60, 0.75, 0.35),
    "disengaged": Archetype(0.30, 0.8, 0.40, 0.10, 0.30, 0.15),
}

SCENARIOS = (
    "mixed",
    "widespread_misconception",
    "small_group_behind",
    "dominant_talkers",
    "confident_wrong",
    "quiet_knowers",
)


def make_profile(
    student_id: str, archetype: str, catalog: CourseCatalog, rng: np.random.Generator
) -> StudentProfile:
    a = ARCHETYPES[archetype]
    prior: dict[str, float] = {}
    for c in catalog.concepts:
        difficulty = 0.8 if c.prerequisites else 1.0
        prior[c.id] = float(np.clip(a.prior * difficulty + rng.normal(0, 0.08), 0.01, 0.99))
    misconceptions = [m.id for m in catalog.misconceptions if rng.random() < a.misconception_rate]
    if archetype == "confident_wrong" and not misconceptions and catalog.misconceptions:
        misconceptions = [catalog.misconceptions[int(rng.integers(len(catalog.misconceptions)))].id]
    return StudentProfile(
        student_id=student_id,
        archetype=archetype,
        prior_knowledge=prior,
        misconceptions=misconceptions,
        learning_speed=a.learning_speed,
        confidence=a.confidence,
        participation_tendency=a.participation,
        engagement=a.engagement,
        guess=a.guess,
        slip=a.slip,
    )


def _mix(n: int, weights: dict[str, float], rng: np.random.Generator) -> list[str]:
    names = list(weights)
    p = np.array([weights[k] for k in names], dtype=float)
    return list(rng.choice(names, size=n, p=p / p.sum()))


def make_classroom(
    catalog: CourseCatalog,
    scenario: str = "mixed",
    n_students: int = 24,
    seed: int = 0,
    widespread_misconception: str | None = None,
) -> list[StudentProfile]:
    """Profiles for one of the evaluation scenarios in SCENARIOS."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")
    rng = np.random.default_rng(seed)
    base = {"strong": 0.2, "average": 0.45, "struggling": 0.1, "quiet_knower": 0.1, "dominant": 0.1, "confident_wrong": 0.05}

    if scenario in ("mixed", "widespread_misconception"):
        kinds = _mix(n_students, base, rng)
    elif scenario == "small_group_behind":
        k = max(3, n_students // 6)
        kinds = ["struggling"] * k + _mix(n_students - k, {"strong": 0.4, "average": 0.6}, rng)
    elif scenario == "dominant_talkers":
        k = max(2, n_students // 8)
        kinds = ["dominant"] * k + _mix(n_students - k, {"average": 0.6, "quiet_knower": 0.25, "strong": 0.15}, rng)
    elif scenario == "confident_wrong":
        k = max(3, n_students // 5)
        kinds = ["confident_wrong"] * k + _mix(n_students - k, {"average": 0.7, "strong": 0.3}, rng)
    else:
        k = max(3, n_students // 4)
        kinds = ["quiet_knower"] * k + _mix(n_students - k, {"average": 0.7, "dominant": 0.1, "strong": 0.2}, rng)

    profiles = [make_profile(f"s{i + 1:02d}", kind, catalog, rng) for i, kind in enumerate(kinds)]

    if scenario == "widespread_misconception":
        target = widespread_misconception or catalog.misconceptions[-2].id
        for p in profiles:
            if rng.random() < 0.7 and target not in p.misconceptions:
                p.misconceptions.append(target)
    return profiles