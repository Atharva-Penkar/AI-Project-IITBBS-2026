"""Group and class summaries, and a discretised state for tabular RL."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from shared.schema import ConceptSummary, GroupSummary, StudentEstimate


@dataclass(frozen=True)
class SummaryThresholds:
    left_behind_mastery: float = 0.4
    high_uncertainty: float = 0.15
    misconception_present: float = 0.5


def gini(values: Sequence[float]) -> float:
    """0 = perfectly even participation, 1 = one student does everything."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    n = x.size
    index = np.arange(1, n + 1)
    return float((2 * (index * x).sum()) / (n * x.sum()) - (n + 1) / n)


def summarise(
    students: Sequence[StudentEstimate],
    concept_ids: Sequence[str],
    thresholds: SummaryThresholds = SummaryThresholds(),
) -> GroupSummary:
    if not students:
        raise ValueError("summarise() needs at least one student")

    concepts: dict[str, ConceptSummary] = {}
    for cid in concept_ids:
        mastery = np.array([s.concepts[cid].p_mastery for s in students])
        uncert = np.array([s.concepts[cid].uncertainty for s in students])
        mids = sorted({m for s in students for m in s.concepts[cid].misconceptions})
        prevalence = {
            m: float(
                np.mean(
                    [s.concepts[cid].misconceptions.get(m, 0.0) >= thresholds.misconception_present for s in students]
                )
            )
            for m in mids
        }
        concepts[cid] = ConceptSummary(
            concept_id=cid,
            mean_mastery=float(mastery.mean()),
            std_mastery=float(mastery.std()),
            mean_uncertainty=float(uncert.mean()),
            fraction_below_threshold=float((mastery < thresholds.left_behind_mastery).mean()),
            misconception_prevalence=prevalence,
        )

    left_behind = [
        s.student_id
        for s in students
        if np.mean([s.concepts[c].p_mastery for c in concept_ids]) < thresholds.left_behind_mastery
    ]
    high_uncertainty = [
        s.student_id
        for s in students
        if np.mean([s.concepts[c].uncertainty for c in concept_ids]) > thresholds.high_uncertainty
    ]

    return GroupSummary(
        student_ids=[s.student_id for s in students],
        concepts=concepts,
        mean_engagement=float(np.mean([s.engagement for s in students])),
        mean_participation=float(np.mean([s.participation for s in students])),
        participation_gini=gini([s.participation for s in students]),
        students_left_behind=left_behind,
        students_high_uncertainty=high_uncertainty,
        confidently_wrong_students=[s.student_id for s in students if s.confidently_wrong],
    )


def _level(value: float, cuts: tuple[float, float]) -> int:
    """0 = low, 1 = medium, 2 = high."""
    return 0 if value < cuts[0] else (1 if value < cuts[1] else 2)


def discretise(summary: GroupSummary, concept_id: str) -> tuple[int, int, int, int, int]:
    """Small state tuple for tabular Q-learning.

    (mastery, misconception prevalence, engagement, participation balance,
    share of students left behind on this concept), each 0/1/2.
    Participation balance is 2 when even and 0 when a few students dominate.
    """
    cs = summary.concepts[concept_id]
    worst_misconception = max(cs.misconception_prevalence.values(), default=0.0)
    return (
        _level(cs.mean_mastery, (0.4, 0.7)),
        _level(worst_misconception, (0.15, 0.4)),
        _level(summary.mean_engagement, (0.4, 0.7)),
        2 - _level(summary.participation_gini, (0.3, 0.5)),
        _level(cs.fraction_below_threshold, (0.1, 0.3)),
    )