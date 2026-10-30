"""Run ANY ``TeachingPolicy`` through the production path: AdaptivePlanner + simulator + engine.

This is the evaluation harness. It never uses the training environment, so a policy is
scored exactly the way it would run when plugged into Team 4's planner.
"""

from __future__ import annotations

import dataclasses

from module3_student_model.engine import EngineConfig, StudentModelEngine
from module3_student_model.profiles import make_classroom
from module3_student_model.sample_catalog import astar_catalog
from module3_student_model.simulator import ClassroomSimulator
from module5_rl.actions import time_cost
from module5_rl.metrics import compute_metrics, known_matrix
from module5_rl.policies import AdvancingPlanner
from shared.schema import ActionType, CourseCatalog


def run_policy_episode(
    policy,
    scenario: str,
    seed: int,
    n_students: int = 24,
    time_budget: float = 45.0,
    catalog: CourseCatalog | None = None,
    engine_config: EngineConfig | None = None,
    trace: bool = False,
):
    catalog = catalog or astar_catalog()
    profiles = make_classroom(catalog, scenario, n_students, seed=seed)
    sim = ClassroomSimulator(catalog, profiles, seed=seed)
    engine = StudentModelEngine(
        catalog, sim.student_ids(), dataclasses.replace(engine_config or EngineConfig(), seed=seed)
    )
    planner = AdvancingPlanner(catalog, policy=policy)
    if hasattr(policy, "reset"):
        policy.reset()
    known0 = known_matrix(sim, catalog)

    time_left, steps, log = time_budget, 0, []
    while time_left > 1e-9:
        if hasattr(policy, "set_time_left"):
            policy.set_time_left(time_left)
        d = planner.plan_next(engine)
        if d is None:
            break
        events = sim.step(d.action, d.concept_id, list(d.target_student_ids) or None, d.target_misconception)
        engine.update_many(events)
        planner.record_execution(d)
        time_left -= time_cost(d.action)
        if d.action != ActionType.NEXT_CONCEPT:
            steps += 1
        if trace:
            log.append(d)

    metrics = compute_metrics(
        sim, engine, catalog, len(planner.completed_concepts), known0, time_budget - max(time_left, 0.0), steps
    )
    if hasattr(policy, "fallback_rate"):
        metrics["fallback_rate"] = policy.fallback_rate
    return (metrics, log) if trace else metrics
