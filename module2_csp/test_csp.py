import copy

import pytest

from module2_csp.adapter import GroupScheduler
from module2_csp.classroom_csp import make_demo_classroom, solve_classroom, validate_schedule
from module3_student_model import ClassroomSimulator, StudentModelEngine, astar_catalog, make_classroom
from shared.schema import ActionType


@pytest.mark.parametrize("n_students", [9, 16, 24])
def test_valid_arrangement_for_different_class_sizes(n_students):
    cfg = make_demo_classroom(n_students=n_students, seed=1)
    cfg.max_parallel_groups = 3
    res = solve_classroom(cfg)
    assert res.feasible
    assert validate_schedule(cfg, res.solution) == []
    placed = sorted(s for g in res.solution.values() for s in g["students"])
    assert placed == sorted(s.sid for s in cfg.students)


def test_too_few_ta_slots_is_infeasible_and_explained():
    cfg = make_demo_classroom()
    for t in cfg.tas:
        t.max_groups = 1
    res = solve_classroom(cfg)
    assert res.status == "INFEASIBLE"
    assert any("[ta_workload]" in line for line in res.explanation)


def test_student_never_free_when_a_ta_is_is_explained():
    cfg = make_demo_classroom()
    cfg.students[0].available_slots = {"Sat-10"}
    res = solve_classroom(cfg)
    assert not res.feasible
    assert any("[student_availability]" in line and cfg.students[0].sid in line for line in res.explanation)


def test_all_weak_class_cannot_be_mixed():
    cfg = make_demo_classroom()
    for s in cfg.students:
        s.ability = 0.2
    cfg.students[0].ability = 0.9
    res = solve_classroom(cfg)
    assert not res.feasible
    assert any("[ability_mix]" in line for line in res.explanation)


def test_scheduler_uses_live_student_model_estimates():
    cat = astar_catalog()
    sim = ClassroomSimulator(cat, make_classroom(cat, "mixed", 18, seed=2), seed=2)
    eng = StudentModelEngine(cat, sim.student_ids())
    for cid in cat.concept_ids():
        for action in (ActionType.LECTURE, ActionType.EXAMPLE, ActionType.EXAMPLE, ActionType.DIAGNOSTIC_QUIZ):
            eng.update_many(sim.step(action, cid))
    plan = GroupScheduler(cat, sim.student_ids(), seed=2).schedule(eng, "path_cost", sim.student_ids(), "peer")
    placed = [s for g in plan.groups for s in g.student_ids] + plan.unplaced
    assert sorted(placed) == sorted(sim.student_ids())
    assert all(g.kind in ("peer", "revision", "ta") for g in plan.groups)
