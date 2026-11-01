"""Runs the full classroom digital twin: every module in one loop.

    python main.py --scenario widespread_misconception
    python main.py --policy rl --scenario small_group_behind
    python main.py --compare

Each step:
  1. The planner (Module 2 planning) chooses the next teaching action from the
     student model's estimates, using the chosen policy (plan, heuristic or rl).
  2. Peer and TA actions are split into groups by the classroom CSP.
  3. The simulator applies the action and returns classroom events.
  4. Student questions are re-labelled by the NLU pipeline.
  5. The student model updates its estimates from the events.
  6. For struggling students, knowledge-graph search ranks likely root causes
     and suggests resources.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from module1_knowledge_graph.catalog import load_course_catalog, match_misconceptions
from module1_knowledge_graph.student_graph import diagnose_student
from module2_csp.adapter import GroupScheduler
from module3_student_model import ClassroomSimulator, StudentModelEngine, make_classroom
from module3_student_model.engine import EngineConfig
from module3_student_model.profiles import SCENARIOS
from module4_planner.executor import AdaptiveExecutor
from module4_planner.nlu_adapter import nlu_result_to_event
from module4_planner.plan_policy import PlanningTeachingPolicy
from module4_planner.planner import AdaptivePlanner
from module4_planner.policy import HeuristicTeachingPolicy
from module5_nlu.nlu.pipeline import analyse_question
from module5_rl.actions import time_cost
from module5_rl.metrics import compute_metrics, known_matrix
from module5_rl.policies import FixedLecturePlanPolicy, RLTeachingPolicy, ThresholdRepeatPolicy
from shared.schema import ActionType, CourseCatalog, Event, EventType, Intent

RL_DIR = Path(__file__).resolve().parent / "module5_rl" / "results"
DIAGNOSE_ACTIONS = {ActionType.TA_INTERVENTION, ActionType.REVISION, ActionType.EXAMPLE}


def make_policy(name: str, catalog: CourseCatalog, time_budget: float):
    if name == "plan":
        return PlanningTeachingPolicy(catalog)
    if name == "heuristic":
        return HeuristicTeachingPolicy()
    if name == "rl":
        return RLTeachingPolicy.load(
            str(RL_DIR / "q_table_q_learning.npy"), str(RL_DIR / "visits_q_learning.npy"), time_budget=time_budget
        )
    if name == "fixed":
        return FixedLecturePlanPolicy()
    if name == "threshold":
        return ThresholdRepeatPolicy()
    raise ValueError(f"Unknown policy: {name}")


def make_nlu_hook(catalog: CourseCatalog, log: list[str] | None):
    """Re-labels question and discussion events from their text with the NLU pipeline."""

    def hook(event: Event) -> Event:
        if event.type not in (EventType.QUESTION, EventType.DISCUSSION) or not event.text:
            return event
        result = analyse_question(event.text)
        mapped = nlu_result_to_event(event.student_id, result, catalog, step=event.step, fallback_concept_id=event.concept_id)
        concept_ms = {m.id for m in catalog.misconceptions_for(event.concept_id)}
        signals = [m for m, _ in match_misconceptions(event.text, [event.concept_id]) if m in concept_ms]
        if mapped is not None and mapped.concept_id == event.concept_id:
            signals = list(dict.fromkeys(mapped.misconception_signals + signals))
            intent, confidence = mapped.intent, mapped.nlu_confidence
        else:
            intent, confidence = Intent.UNCLEAR, 0.3
        if signals:
            intent, confidence = Intent.MISCONCEPTION_SIGNAL, max(confidence or 0.0, 0.8)
        new = event.model_copy(update={"intent": intent, "nlu_confidence": confidence, "misconception_signals": signals})
        if log is not None:
            log.append(f"      NLU {event.student_id}: \"{event.text}\" -> {intent.value} ({confidence:.2f}) {signals or ''}")
        return new

    return hook


def run_episode(
    scenario: str = "mixed",
    n_students: int = 24,
    seed: int = 0,
    policy_name: str = "plan",
    time_budget: float = 80.0,
    use_csp: bool = True,
    use_nlu: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    catalog = load_course_catalog()
    profiles = make_classroom(catalog, scenario, n_students, seed=seed)
    sim = ClassroomSimulator(catalog, profiles, seed=seed)
    engine = StudentModelEngine(catalog, sim.student_ids(), dataclasses.replace(EngineConfig(), seed=seed))
    policy = make_policy(policy_name, catalog, time_budget)
    planner = AdaptivePlanner(catalog, policy=policy)
    if hasattr(policy, "reset"):
        policy.reset()
    notes: list[str] | None = [] if verbose else None
    executor = AdaptiveExecutor(
        group_scheduler=GroupScheduler(catalog, sim.student_ids(), seed=seed) if use_csp else None,
        event_hook=make_nlu_hook(catalog, notes) if use_nlu else None,
    )
    known0 = known_matrix(sim, catalog)

    if verbose:
        print(f"Scenario {scenario}, {n_students} students, policy {policy_name}, time budget {time_budget}")
        print(f"Course: {' -> '.join(catalog.concept_ids())}\n")

    time_left, steps = time_budget, 0
    while time_left > 1e-9:
        if hasattr(policy, "set_time_left"):
            policy.set_time_left(time_left)
        decision = planner.plan_next(engine)
        if decision is None:
            break
        result = executor.execute(decision, sim, engine, planner)
        time_left -= time_cost(decision.action)
        if decision.action != ActionType.NEXT_CONCEPT:
            steps += 1
        if not verbose:
            continue

        targets = f"{len(decision.target_student_ids)} students" if decision.target_student_ids else "class"
        print(f"[{steps:3d}] {decision.action.value:16} {decision.concept_id:15} {targets:12} {decision.rationale}")
        if result.group_plan is not None:
            for g in result.group_plan.groups:
                print(f"      group {g.name}: {g.kind} on {g.concept_id}, {g.ta} at {g.slot}: {', '.join(g.student_ids)}")
            for line in result.group_plan.explanation[:3]:
                print(f"      CSP: {line}")
        for line in result.notes:
            print(f"      {line}")
        for line in notes[:4]:
            print(line)
        notes.clear()
        if decision.action in DIAGNOSE_ACTIONS and decision.target_student_ids:
            sid = decision.target_student_ids[0]
            diagnosis = diagnose_student(engine.get_student(sid), decision.concept_id, top_k=2)
            top = diagnosis["ranked_causes"][0]
            print(f"      diagnosis {sid}: likely cause {top['name']} ({'; '.join(top['reasons'])}); {diagnosis['next_step']}")

    metrics = compute_metrics(sim, engine, catalog, len(planner.completed_concepts), known0, time_budget - max(time_left, 0.0), steps)
    if verbose:
        summary = engine.get_class_summary()
        print("\nFinal estimates")
        for cid, cs in summary.concepts.items():
            print(f"  {cid:15} mastery {cs.mean_mastery:.2f}  below threshold {cs.fraction_below_threshold:.0%}")
        print(f"\nCompleted {len(planner.completed_concepts)}/{len(catalog.concepts)} concepts")
        print("Outcome (simulator truth): " + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
    return metrics


def compare(policies: list[str], seeds: list[int], n_students: int, time_budget: float) -> None:
    keys = ["known", "left_behind", "coverage", "misconceptions_left", "participation_balance", "equity_gap"]
    print(f"{'scenario':26}{'policy':11}" + "".join(f"{k:>22}" for k in keys))
    for scenario in SCENARIOS:
        for name in policies:
            runs = [run_episode(scenario, n_students, s, name, time_budget, verbose=False) for s in seeds]
            cells = [f"{np.mean([r[k] for r in runs]):.3f} ± {np.std([r[k] for r in runs]):.3f}" for k in keys]
            print(f"{scenario:26}{name:11}" + "".join(f"{c:>22}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    parser.add_argument("--students", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["plan", "heuristic", "rl", "fixed", "threshold"], default="plan")
    parser.add_argument("--time-budget", type=float, default=80.0)
    parser.add_argument("--no-csp", action="store_true")
    parser.add_argument("--no-nlu", action="store_true")
    parser.add_argument("--compare", action="store_true", help="compare policies on every scenario")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if args.compare:
        compare(["plan", "heuristic", "rl", "fixed", "threshold"], list(range(args.seeds)), args.students, args.time_budget)
    else:
        run_episode(args.scenario, args.students, args.seed, args.policy, args.time_budget, not args.no_csp, not args.no_nlu)


if __name__ == "__main__":
    main()
