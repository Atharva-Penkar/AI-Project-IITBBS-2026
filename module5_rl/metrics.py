"""Outcome metrics computed from the simulator's TRUE state (never from the model's estimates)."""

from __future__ import annotations

import numpy as np

from module3_student_model.simulator import ClassroomSimulator
from module3_student_model.engine import StudentModelEngine
from shared.schema import CourseCatalog

LEFT_BEHIND_KNOWN_FRACTION = 0.4


def known_matrix(sim: ClassroomSimulator, catalog: CourseCatalog) -> np.ndarray:
    """(students x concepts) matrix of true mastery (0/1)."""
    return np.array(
        [[float(sim.true_mastery(s, c)) for c in catalog.concept_ids()] for s in sim.student_ids()]
    )


def misconception_count(sim: ClassroomSimulator) -> float:
    """Mean number of misconceptions a student currently holds."""
    return float(np.mean([len(sim.true_misconceptions(s)) for s in sim.student_ids()]))


def mean_engagement(sim: ClassroomSimulator) -> float:
    return float(np.mean([sim.true_engagement(s) for s in sim.student_ids()]))


def left_behind_fraction(known: np.ndarray) -> float:
    return float((known.mean(axis=1) < LEFT_BEHIND_KNOWN_FRACTION).mean())


def compute_metrics(
    sim: ClassroomSimulator,
    engine: StudentModelEngine,
    catalog: CourseCatalog,
    n_completed: int,
    known0: np.ndarray,
    time_used: float,
    steps: int,
) -> dict[str, float]:
    known = known_matrix(sim, catalog)
    gain = known.mean(axis=1) - known0.mean(axis=1)
    order = np.argsort(known0.mean(axis=1), kind="stable")
    third = max(1, len(order) // 3)
    low, high = order[:third], order[-third:]
    return {
        "known": float(known.mean()),  # fraction of (student, concept) pairs truly mastered
        "gain": float(gain.mean()),
        "left_behind": left_behind_fraction(known),
        "coverage": n_completed / len(catalog.concepts),
        "misconceptions_left": misconception_count(sim),
        "engagement": mean_engagement(sim),
        "participation_balance": float(1.0 - engine.get_class_summary().participation_gini),
        "time_used": float(time_used),
        "steps": float(steps),
        "gain_low": float(gain[low].mean()),
        "gain_high": float(gain[high].mean()),
        "equity_gap": float(gain[low].mean() - gain[high].mean()),
    }
