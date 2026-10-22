"""Run one simulated lesson and print what the engine believes.

    python -m module3_student_model.demo --scenario confident_wrong
"""

from __future__ import annotations

import argparse

from module3_student_model.engine import StudentModelEngine
from module3_student_model.profiles import SCENARIOS, make_classroom
from module3_student_model.sample_catalog import astar_catalog
from module3_student_model.simulator import ClassroomSimulator
from shared.schema import ActionType

LESSON = [
    ActionType.DIAGNOSTIC_QUIZ,
    ActionType.LECTURE,
    ActionType.EXAMPLE,
    ActionType.DIAGNOSTIC_QUIZ,
    ActionType.PEER_DISCUSSION,
    ActionType.EXERCISE,
    ActionType.DIAGNOSTIC_QUIZ,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    parser.add_argument("--students", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    catalog = astar_catalog()
    profiles = make_classroom(catalog, args.scenario, args.students, seed=args.seed)
    sim = ClassroomSimulator(catalog, profiles, seed=args.seed)
    engine = StudentModelEngine(catalog, sim.student_ids())

    n_events = 0
    for cid in catalog.concept_ids():
        for action in LESSON:
            events = sim.step(action, cid)
            engine.update_many(events)
            n_events += len(events)

    summary = engine.get_class_summary()
    print(f"Scenario: {args.scenario}, {len(profiles)} students, {n_events} events\n")
    print(f"{'concept':<16}{'mastery':>9}{'below 0.4':>11}{'top misconception':>36}")
    for cid, cs in summary.concepts.items():
        top = max(cs.misconception_prevalence.items(), key=lambda kv: kv[1], default=("-", 0.0))
        print(f"{cid:<16}{cs.mean_mastery:>9.2f}{cs.fraction_below_threshold:>11.0%}{top[0]:>30} {top[1]:>4.0%}")

    print(f"\nEngagement {summary.mean_engagement:.2f}, participation Gini {summary.participation_gini:.2f}")
    print(f"Left behind: {summary.students_left_behind or 'none'}")
    print(f"Confidently wrong: {summary.confidently_wrong_students or 'none'}")
    print(f"RL state for 'astar': {engine.discretised_state('astar')}\n")

    print("Sample explanations for A* (estimate vs truth):")
    for sid in sim.student_ids()[:5]:
        est = engine.get_student(sid).concepts["astar"]
        truth = "knows" if sim.true_mastery(sid, "astar") else "does not know"
        held = sorted(sim.true_misconceptions(sid)) or "none"
        print(f"  {sid} [{sim.profiles[sid].archetype}] truth: {truth}, misconceptions {held}")
        print(f"      {est.explanation}")


if __name__ == "__main__":
    main()