"""Unit tests for policy ordering and its interpretable action choices."""

from shared.schema import ActionType

from module4_planner.models import PlannerConfig
from module4_planner.policy import HeuristicTeachingPolicy
from module4_planner.tests.helpers import make_students, make_summary


def test_unobserved_concept_gets_diagnostic_quiz() -> None:
    students = make_students(["s01", "s02", "s03"], observations={"s01": 0, "s02": 0, "s03": 0})
    summary = make_summary(students, mean_mastery=0.5, mean_uncertainty=0.30)

    decision = HeuristicTeachingPolicy().choose_action(summary, students, "topic")

    assert decision.action == ActionType.DIAGNOSTIC_QUIZ
    assert decision.metadata["rule"] == "diagnostic_evidence"


def test_prevalent_misconception_gets_targeted_ta_intervention() -> None:
    students = make_students(
        ["s01", "s02", "s03", "s04"],
        misconceptions={
            "s01": {"confuses_g_and_h": 0.9},
            "s02": {"confuses_g_and_h": 0.8},
            "s03": {"confuses_g_and_h": 0.1},
            "s04": {"confuses_g_and_h": 0.1},
        },
    )
    summary = make_summary(
        students,
        mean_mastery=0.55,
        mean_uncertainty=0.10,
        misconception_prevalence={"confuses_g_and_h": 0.5},
    )

    decision = HeuristicTeachingPolicy().choose_action(summary, students, "topic")

    assert decision.action == ActionType.TA_INTERVENTION
    assert decision.target_misconception == "confuses_g_and_h"
    assert set(decision.target_student_ids) == {"s01", "s02"}


def test_diagnostic_quiz_is_bounded() -> None:
    students = make_students(
        ["s01", "s02"],
        observations={"s01": 0, "s02": 0},
        uncertainty={"s01": 0.3, "s02": 0.3},
    )
    summary = make_summary(students, mean_mastery=0.5, mean_uncertainty=0.3)
    policy = HeuristicTeachingPolicy(PlannerConfig(max_diagnostic_actions_per_concept=2))

    decision = policy.choose_action(
        summary,
        students,
        "topic",
        action_counts={ActionType.DIAGNOSTIC_QUIZ.value: 2},
    )

    assert decision.action != ActionType.DIAGNOSTIC_QUIZ


def test_single_learner_below_threshold_gets_targeted_ta_support() -> None:
    students = make_students(
        ["s01", "s02", "s03", "s04"],
        mastery={"s01": 0.2, "s02": 0.7, "s03": 0.8, "s04": 0.8},
    )
    summary = make_summary(students, mean_mastery=0.63, mean_uncertainty=0.1)

    decision = HeuristicTeachingPolicy().choose_action(
        summary,
        students,
        "topic",
        action_counts={ActionType.DIAGNOSTIC_QUIZ.value: 2},
    )

    assert decision.action == ActionType.TA_INTERVENTION
    assert decision.target_student_ids == ("s01",)


def test_participation_imbalance_uses_peer_discussion() -> None:
    students = make_students(["s01", "s02", "s03"], participation={"s01": 0.05, "s02": 0.4, "s03": 0.9})
    summary = make_summary(
        students,
        mean_mastery=0.7,
        mean_uncertainty=0.08,
        participation_gini=0.52,
        mean_participation=0.45,
    )

    decision = HeuristicTeachingPolicy().choose_action(summary, students, "topic")

    assert decision.action == ActionType.PEER_DISCUSSION
    assert "s01" in decision.target_student_ids
