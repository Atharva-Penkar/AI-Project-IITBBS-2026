from module3_student_model import ClassroomSimulator, StudentModelEngine, astar_catalog, make_classroom
from module4_planner.action_model import ACTION_MODEL, GOAL, plan_concept
from module4_planner.executor import AdaptiveExecutor
from module4_planner.plan_policy import PlanningTeachingPolicy
from module4_planner.planner import AdaptivePlanner
from shared.schema import ActionType


def test_every_action_has_preconditions_effects_and_duration():
    for spec in ACTION_MODEL.values():
        assert spec.duration > 0
        assert spec.requires or spec.forbids
        assert spec.adds or spec.removes
        assert spec.expected_effect


def test_fresh_concept_gets_lecture_quiz_discussion_sequence():
    plan = plan_concept(frozenset())
    assert plan[:2] == [ActionType.LECTURE, ActionType.DIAGNOSTIC_QUIZ]
    assert ActionType.PEER_DISCUSSION in plan and plan[-1] == ActionType.ASSESSMENT


def test_misconception_and_prerequisite_gap_change_the_plan():
    assert plan_concept(frozenset({"prereq_gap"}))[0] == ActionType.REVISION
    facts = frozenset({"introduced", "evidence", "misconception"})
    assert plan_concept(facts)[0] == ActionType.TA_INTERVENTION


def test_goal_state_needs_no_actions():
    assert plan_concept(GOAL) == []


def test_planning_policy_runs_a_course_and_explains_each_action():
    cat = astar_catalog()
    sim = ClassroomSimulator(cat, make_classroom(cat, "widespread_misconception", 12, seed=1), seed=1)
    eng = StudentModelEngine(cat, sim.student_ids())
    planner = AdaptivePlanner(cat, policy=PlanningTeachingPolicy(cat))
    executor = AdaptiveExecutor()
    seen = set()
    for _ in range(60):
        decision = planner.plan_next(eng)
        if decision is None:
            break
        assert decision.rationale
        seen.add(decision.action)
        executor.execute(decision, sim, eng, planner)
    assert {ActionType.LECTURE, ActionType.DIAGNOSTIC_QUIZ}.issubset(seen)
    assert len(planner.completed_concepts) >= 2
