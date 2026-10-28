"""Small typed fixtures used by the unit tests (no simulator required)."""

from __future__ import annotations

from shared.schema import ConceptEstimate, ConceptSummary, GroupSummary, StudentEstimate


def make_students(
    student_ids: list[str],
    concept_id: str = "topic",
    *,
    mastery: dict[str, float] | None = None,
    observations: dict[str, int] | None = None,
    uncertainty: dict[str, float] | None = None,
    misconceptions: dict[str, dict[str, float]] | None = None,
    participation: dict[str, float] | None = None,
    confidence: dict[str, float] | None = None,
    confidently_wrong: set[str] | None = None,
    engagement: dict[str, float] | None = None,
) -> list[StudentEstimate]:
    """Build model-compatible student records with sensible defaults."""
    mastery = mastery or {}
    observations = observations or {}
    uncertainty = uncertainty or {}
    misconceptions = misconceptions or {}
    participation = participation or {}
    confidence = confidence or {}
    confidently_wrong = confidently_wrong or set()
    engagement = engagement or {}
    students = []
    for sid in student_ids:
        estimate = ConceptEstimate(
            concept_id=concept_id,
            p_mastery=mastery.get(sid, 0.65),
            uncertainty=uncertainty.get(sid, 0.10),
            n_observations=observations.get(sid, 2),
            misconceptions=misconceptions.get(sid, {}),
        )
        students.append(
            StudentEstimate(
                student_id=sid,
                concepts={concept_id: estimate},
                engagement=engagement.get(sid, 0.70),
                participation=participation.get(sid, 0.50),
                confidence=confidence.get(sid, 0.50),
                overconfidence=0.0,
                confidently_wrong=sid in confidently_wrong,
            )
        )
    return students


def make_summary(
    students: list[StudentEstimate],
    concept_id: str = "topic",
    *,
    mean_mastery: float | None = None,
    mean_uncertainty: float | None = None,
    fraction_below_threshold: float | None = None,
    misconception_prevalence: dict[str, float] | None = None,
    mean_engagement: float = 0.70,
    mean_participation: float = 0.50,
    participation_gini: float = 0.0,
) -> GroupSummary:
    """Build a compact GroupSummary while allowing targeted test overrides."""
    estimates = [s.concepts[concept_id] for s in students]
    if mean_mastery is None:
        mean_mastery = sum(e.p_mastery for e in estimates) / len(estimates)
    if mean_uncertainty is None:
        mean_uncertainty = sum(e.uncertainty for e in estimates) / len(estimates)
    if fraction_below_threshold is None:
        fraction_below_threshold = sum(e.p_mastery < 0.4 for e in estimates) / len(estimates)
    concept_summary = ConceptSummary(
        concept_id=concept_id,
        mean_mastery=mean_mastery,
        std_mastery=0.10,
        mean_uncertainty=mean_uncertainty,
        fraction_below_threshold=fraction_below_threshold,
        misconception_prevalence=misconception_prevalence or {},
    )
    return GroupSummary(
        student_ids=[s.student_id for s in students],
        concepts={concept_id: concept_summary},
        mean_engagement=mean_engagement,
        mean_participation=mean_participation,
        participation_gini=participation_gini,
        students_left_behind=[s.student_id for s in students if s.concepts[concept_id].p_mastery < 0.4],
        students_high_uncertainty=[s.student_id for s in students if s.concepts[concept_id].uncertainty > 0.15],
        confidently_wrong_students=[s.student_id for s in students if s.confidently_wrong],
    )
