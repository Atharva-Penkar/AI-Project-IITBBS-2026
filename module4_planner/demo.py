"""Run an end-to-end adaptive-planning demo with Team 3's seeded simulator.

Run from the repository root so `shared`, `module3_student_model`, and
`module4_planner` are importable:

    python -m module4_planner.demo --students 12 --steps 40 --seed 7
"""

from __future__ import annotations

import argparse

from module3_student_model import ClassroomSimulator, StudentModelEngine, astar_catalog, make_classroom
from shared.schema import ActionType

from module4_planner.executor import AdaptiveExecutor
from module4_planner.planner import AdaptivePlanner


def run_demo(n_students: int = 12, max_steps: int = 40, seed: int = 7) -> None:
    """Print each decision and the final class-level state estimate."""
    catalog = astar_catalog()
    profiles = make_classroom(catalog, scenario="mixed", n_students=n_students, seed=seed)
    simulator = ClassroomSimulator(catalog, profiles, seed=seed)
    engine = StudentModelEngine(catalog, student_ids=simulator.student_ids())
    planner = AdaptivePlanner(catalog)
    executor = AdaptiveExecutor()

    print(f"Adaptive Classroom Digital Twin | students={n_students} | seed={seed}")
    print("-" * 78)
    for number in range(1, max_steps + 1):
        decision = planner.plan_next(engine)
        if decision is None:
            print("Course completion criterion met for every concept.")
            break

        result = executor.execute(decision, simulator, engine, planner=planner)
        targets = ",".join(decision.target_student_ids) if decision.target_student_ids else "whole class"
        print(
            f"{number:02d}. {decision.action.value:16s} | {decision.concept_id:16s} | "
            f"targets={targets:14s} | events={result.event_count:2d} | {decision.rationale}"
        )
        if decision.action == ActionType.NEXT_CONCEPT:
            continue

    summary = engine.get_class_summary()
    print("\nFinal class summary")
    print(f"Mean engagement:       {summary.mean_engagement:.3f}")
    print(f"Mean participation:    {summary.mean_participation:.3f}")
    print(f"Participation Gini:    {summary.participation_gini:.3f}")
    print(f"Completed concepts:    {len(planner.completed_concepts)}/{len(catalog.concepts)}")
    for concept_id, estimate in summary.concepts.items():
        print(
            f"  {concept_id:16s} mastery={estimate.mean_mastery:.3f} "
            f"uncertainty={estimate.mean_uncertainty:.3f} "
            f"below-threshold={estimate.fraction_below_threshold:.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=int, default=12, help="Number of simulated students")
    parser.add_argument("--steps", type=int, default=40, help="Maximum simulator steps")
    parser.add_argument("--seed", type=int, default=7, help="Seed for reproducibility")
    args = parser.parse_args()
    if args.students < 1 or args.steps < 1:
        parser.error("--students and --steps must be positive integers")
    run_demo(n_students=args.students, max_steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
