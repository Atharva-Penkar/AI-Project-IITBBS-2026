"""Course progression, prerequisite validation, and completion tests."""

import pytest

from shared.schema import Concept, ConceptEstimate, ConceptSummary, CourseCatalog, GroupSummary, StudentEstimate
from module4_planner.planner import AdaptivePlanner
from module4_planner.models import ActionDecision
from shared.schema import ActionType


class FakeEngine:
    """Minimal StudentModelEngine-compatible test double with a stable snapshot."""

    def __init__(self, summary: GroupSummary, students: list[StudentEstimate]):
        self._summary = summary
        self._students = {student.student_id: student for student in students}

    def student_ids(self) -> list[str]:
        return list(self._students)

    def get_class_summary(self) -> GroupSummary:
        return self._summary

    def get_student(self, student_id: str) -> StudentEstimate:
        return self._students[student_id]


def _fake_course_and_engine(*, mastery_root: float = 0.85, mastery_child: float = 0.2):
    # Child is deliberately listed before its prerequisite to ensure the
    # planner uses the dependency graph rather than list order alone.
    catalog = CourseCatalog(
        concepts=[
            Concept(id="child", name="Dependent concept", prerequisites=["root"]),
            Concept(id="root", name="Prerequisite concept"),
        ]
    )
    student = StudentEstimate(
        student_id="s01",
        concepts={
            "root": ConceptEstimate(
                concept_id="root", p_mastery=mastery_root, uncertainty=0.05, n_observations=4
            ),
            "child": ConceptEstimate(
                concept_id="child", p_mastery=mastery_child, uncertainty=0.1, n_observations=2
            ),
        },
        engagement=0.7,
        participation=0.5,
        confidence=0.5,
        overconfidence=0.0,
        confidently_wrong=False,
    )
    summary = GroupSummary(
        student_ids=["s01"],
        concepts={
            "root": ConceptSummary(
                concept_id="root", mean_mastery=mastery_root, std_mastery=0.0,
                mean_uncertainty=0.05, fraction_below_threshold=0.0,
            ),
            "child": ConceptSummary(
                concept_id="child", mean_mastery=mastery_child, std_mastery=0.0,
                mean_uncertainty=0.1, fraction_below_threshold=1.0 if mastery_child < 0.4 else 0.0,
            ),
        },
        mean_engagement=0.7,
        mean_participation=0.5,
        participation_gini=0.0,
        students_left_behind=["s01"] if mastery_child < 0.4 else [],
        students_high_uncertainty=[],
        confidently_wrong_students=[],
    )
    return catalog, FakeEngine(summary, [student])


def test_planner_respects_prerequisites_even_when_catalog_order_does_not() -> None:
    catalog, engine = _fake_course_and_engine()
    planner = AdaptivePlanner(catalog)

    first = planner.plan_next(engine)
    assert first is not None
    assert first.action == ActionType.NEXT_CONCEPT
    assert first.concept_id == "root"

    # The root already meets the completion criterion, so the next call may
    # unlock the dependent child, but never before the root is recorded complete.
    second = planner.plan_next(engine)
    assert second is not None
    assert second.action == ActionType.NEXT_CONCEPT
    assert second.concept_id == "child"
    assert "root" in planner.completed_concepts


def test_planner_stops_after_all_concepts_meet_completion_criterion() -> None:
    catalog, engine = _fake_course_and_engine(mastery_child=0.8)
    planner = AdaptivePlanner(catalog)

    assert planner.plan_next(engine).concept_id == "root"
    assert planner.plan_next(engine).concept_id == "child"
    assert planner.plan_next(engine) is None
    assert planner.is_complete


def test_unknown_prerequisite_is_rejected() -> None:
    catalog = CourseCatalog(concepts=[Concept(id="child", name="Child", prerequisites=["missing"])])
    with pytest.raises(ValueError, match="unknown prerequisite"):
        AdaptivePlanner(catalog)


def test_prerequisite_cycle_is_rejected() -> None:
    catalog = CourseCatalog(
        concepts=[
            Concept(id="a", name="A", prerequisites=["b"]),
            Concept(id="b", name="B", prerequisites=["a"]),
        ]
    )
    with pytest.raises(ValueError, match="cycle"):
        AdaptivePlanner(catalog)


def test_no_registered_students_returns_no_decision() -> None:
    catalog, engine = _fake_course_and_engine()
    empty_engine = FakeEngine(engine.get_class_summary(), [])
    assert AdaptivePlanner(catalog).plan_next(empty_engine) is None


def test_action_history_is_tracked_per_concept() -> None:
    catalog, _ = _fake_course_and_engine()
    planner = AdaptivePlanner(catalog)
    planner.record_execution(ActionDecision(action=ActionType.DIAGNOSTIC_QUIZ, concept_id="root"))
    planner.record_execution(ActionDecision(action=ActionType.EXAMPLE, concept_id="child"))

    assert planner.action_counts_for("root") == {"diagnostic_quiz": 1}
    assert planner.action_counts_for("child") == {"example": 1}
    assert planner.action_counts == {"diagnostic_quiz": 1, "example": 1}
