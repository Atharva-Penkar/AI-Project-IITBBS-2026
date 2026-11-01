"""End-to-end smoke tests using the real Team 3 engine and simulator."""

import pytest

pytest.importorskip("module3_student_model")

from module3_student_model import ClassroomSimulator, StudentModelEngine, astar_catalog, make_classroom
from shared.schema import ActionType
from module4_planner.executor import AdaptiveExecutor
from module4_planner.planner import AdaptivePlanner


def test_adaptive_loop_updates_model_and_emits_auditable_decisions() -> None:
    catalog = astar_catalog()
    profiles = make_classroom(catalog, scenario="mixed", n_students=8, seed=11)
    simulator = ClassroomSimulator(catalog, profiles, seed=11)
    engine = StudentModelEngine(catalog, student_ids=simulator.student_ids())
    planner = AdaptivePlanner(catalog)
    executor = AdaptiveExecutor()

    decisions = []
    for _ in range(12):
        decision = planner.plan_next(engine)
        if decision is None:
            break
        decisions.append(decision)
        executor.execute(decision, simulator, engine, planner=planner)

    assert decisions
    assert any(decision.action == ActionType.NEXT_CONCEPT for decision in decisions)
    assert any(decision.action != ActionType.NEXT_CONCEPT for decision in decisions)
    assert engine.get_class_summary().student_ids == simulator.student_ids()
    assert all(decision.rationale for decision in decisions)


def test_evaluation_harness_runs_both_modes() -> None:
    from module4_planner.evaluation import evaluate_policies

    rows = evaluate_policies(scenarios=("mixed",), seeds=(3,), n_students=6, max_steps=8)
    assert {row["mode"] for row in rows} == {"fixed", "adaptive"}
    assert all(0.0 <= row["mean_mastery"] <= 1.0 for row in rows)
