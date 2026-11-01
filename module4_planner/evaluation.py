"""Reproducible simulation harness for adaptive-vs-fixed lesson comparison.

This is an evaluation harness, not a claim that the heuristic policy is better.
Run identical scenarios/seeds for both policies and inspect mean mastery,
remaining learners below threshold, participation inequality, and action cost.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from module3_student_model import (
    SCENARIOS,
    ClassroomSimulator,
    StudentModelEngine,
    astar_catalog,
    make_classroom,
)
from shared.schema import ActionType

from module4_planner.executor import AdaptiveExecutor
from module4_planner.models import ActionDecision, PlannerConfig
from module4_planner.planner import AdaptivePlanner

FIXED_ACTION_CYCLE = (
    ActionType.LECTURE,
    ActionType.EXAMPLE,
    ActionType.EXERCISE,
    ActionType.REVISION,
)


def run_episode(
    *,
    mode: str,
    scenario: str = "mixed",
    seed: int = 0,
    n_students: int = 24,
    max_steps: int = 60,
    config: PlannerConfig | None = None,
) -> dict[str, Any]:
    """Run one seeded classroom episode and return comparable scalar metrics.

    Both modes share the same catalog, profile generation, simulator seed,
    class size, and maximum number of simulator steps. The adaptive mode may
    spend steps on explicit ``NEXT_CONCEPT`` transitions; those no-op steps are
    counted separately so the overhead is visible in the report.
    """
    if mode not in {"adaptive", "fixed"}:
        raise ValueError("mode must be 'adaptive' or 'fixed'")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    catalog = astar_catalog()
    profiles = make_classroom(catalog, scenario=scenario, n_students=n_students, seed=seed)
    simulator = ClassroomSimulator(catalog, profiles, seed=seed)
    # Register the full class before the first plan call. NEXT_CONCEPT emits no
    # events, so relying on the first event to register students would deadlock.
    engine = StudentModelEngine(catalog, student_ids=simulator.student_ids())
    executor = AdaptiveExecutor()
    planner = AdaptivePlanner(catalog, config=config) if mode == "adaptive" else None

    actions: list[str] = []
    transitions = 0
    teaching_actions = 0
    for step_index in range(max_steps):
        if mode == "adaptive":
            assert planner is not None
            decision = planner.plan_next(engine)
            if decision is None:
                break
            result = executor.execute(decision, simulator, engine, planner=planner)
        else:
            # Fixed sequence: cover concepts in catalog order, cycling through
            # the same four action types without observing mastery to choose.
            actions_per_concept = len(FIXED_ACTION_CYCLE)
            concept_index = (step_index // actions_per_concept) % len(catalog.concepts)
            action_index = step_index % actions_per_concept
            concept_id = catalog.concepts[concept_index].id
            decision = ActionDecision(
                action=FIXED_ACTION_CYCLE[action_index],
                concept_id=concept_id,
                rationale="Fixed baseline sequence; no state-dependent selection.",
                metadata={"rule": "fixed_lesson_baseline"},
            )
            result = executor.execute(decision, simulator, engine)

        actions.append(decision.action.value)
        if decision.action == ActionType.NEXT_CONCEPT:
            transitions += 1
        else:
            teaching_actions += 1

    final_summary = engine.get_class_summary()
    estimates = [engine.get_student(sid) for sid in simulator.student_ids()]
    all_mastery = [estimate.p_mastery for student in estimates for estimate in student.concepts.values()]
    all_below = [value < 0.4 for value in all_mastery]
    concept_mastery = [item.mean_mastery for item in final_summary.concepts.values()]
    concept_below = [item.fraction_below_threshold for item in final_summary.concepts.values()]

    metric_config = config or PlannerConfig()
    concepts_meeting_mastery_criterion = sum(
        item.mean_mastery >= metric_config.mastery_target
        and item.fraction_below_threshold
        <= metric_config.max_fraction_below_threshold_for_completion
        for item in final_summary.concepts.values()
    )
    planner_completed = len(planner.completed_concepts) if planner is not None else None
    return {
        "mode": mode,
        "scenario": scenario,
        "seed": seed,
        "n_students": n_students,
        "max_steps": max_steps,
        "simulator_steps": simulator.step_count,
        "teaching_actions": teaching_actions,
        "concept_transitions": transitions,
        "mean_mastery": mean(all_mastery) if all_mastery else 0.0,
        "fraction_student_concepts_below_0_4": mean(all_below) if all_below else 0.0,
        "fraction_concepts_below_threshold": mean(concept_below) if concept_below else 0.0,
        "mean_class_mastery_by_concept": mean(concept_mastery) if concept_mastery else 0.0,
        "mean_engagement": final_summary.mean_engagement,
        "mean_participation": final_summary.mean_participation,
        "participation_gini": final_summary.participation_gini,
        "confidently_wrong_students": len(final_summary.confidently_wrong_students),
        # Final-estimate mastery criterion is computed for both modes, unlike
        # planner progression count, which only exists for the adaptive mode.
        "concepts_meeting_mastery_criterion": concepts_meeting_mastery_criterion,
        "course_completion_fraction": concepts_meeting_mastery_criterion / len(catalog.concepts),
        "planner_completed_concepts": planner_completed,
        "action_sequence": "|".join(actions),
    }


def evaluate_policies(
    *,
    scenarios: Iterable[str] = SCENARIOS,
    seeds: Iterable[int] = (0, 1, 2),
    n_students: int = 24,
    max_steps: int = 60,
    config: PlannerConfig | None = None,
) -> list[dict[str, Any]]:
    """Evaluate both policies over the same set of scenarios and random seeds."""
    rows: list[dict[str, Any]] = []
    scenarios = tuple(scenarios)
    seeds = tuple(seeds)
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unsupported scenario {scenario!r}; expected one of {SCENARIOS}")
        for seed in seeds:
            for mode in ("fixed", "adaptive"):
                rows.append(
                    run_episode(
                        mode=mode,
                        scenario=scenario,
                        seed=seed,
                        n_students=n_students,
                        max_steps=max_steps,
                        config=config,
                    )
                )
    return rows


def aggregate_results(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Average the core evaluation metrics grouped by policy mode."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["mode"])].append(row)

    metrics = (
        "mean_mastery",
        "fraction_student_concepts_below_0_4",
        "fraction_concepts_below_threshold",
        "mean_engagement",
        "mean_participation",
        "participation_gini",
        "confidently_wrong_students",
        "teaching_actions",
        "concept_transitions",
        "course_completion_fraction",
        "concepts_meeting_mastery_criterion",
    )
    return {
        mode: {metric: mean(float(row[metric]) for row in values) for metric in metrics}
        for mode, values in buckets.items()
    }


def write_csv(rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    """Write raw episode metrics to CSV without requiring pandas."""
    rows = list(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No evaluation rows to write")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main() -> None:
    """CLI entry point for a quick reproducible comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS), help="Comma-separated scenario names")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated integer seeds")
    parser.add_argument("--students", type=int, default=24, help="Students per simulated class")
    parser.add_argument("--max-steps", type=int, default=60, help="Simulator action budget per episode")
    parser.add_argument("--output", default="module4_planner/evaluation_results.csv", help="CSV output path")
    args = parser.parse_args()

    scenarios = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    rows = evaluate_policies(
        scenarios=scenarios,
        seeds=seeds,
        n_students=args.students,
        max_steps=args.max_steps,
    )
    path = write_csv(rows, args.output)
    print(f"Wrote {len(rows)} episode(s) to {path}")
    print(json.dumps(aggregate_results(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
