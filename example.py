"""End-to-end example that checks every module of the classroom digital twin.

    python example.py

Each check prints PASS or FAIL. The script exits with code 1 if any check fails.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

from main import run_episode
from module1_knowledge_graph.astar_examples import STUDENT_TEXT, run_all
from module1_knowledge_graph.catalog import load_course_catalog, match_misconceptions
from module1_knowledge_graph.knowledge_graph import diagnose, explain
from module1_knowledge_graph.student_graph import concept_status, study_plan
from module2_csp.adapter import GroupScheduler
from module2_csp.classroom_csp import make_demo_classroom, solve_classroom
from module3_student_model import ClassroomSimulator, StudentModelEngine, make_classroom
from module4_planner.action_model import plan_concept
from module4_planner.nlu_adapter import nlu_result_to_event
from module4_planner.planner import AdaptivePlanner
from module5_nlu.nlu.pipeline import analyse_question
from module5_rl.policies import RLTeachingPolicy
from shared.schema import ActionType, Event, EventType, Intent

RL_DIR = Path(__file__).resolve().parent / "module5_rl" / "results"
RESULTS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else "")
    )


def quiz(
    sid: str, concept: str, correct: bool, chosen: str | None = None, **kw
) -> Event:
    return Event(
        student_id=sid,
        concept_id=concept,
        type=EventType.QUIZ_ANSWER,
        correct=correct,
        chosen_misconception=chosen,
        **kw,
    )


def main() -> None:
    catalog = load_course_catalog()

    print("\n1. Knowledge graph and search (module1_knowledge_graph)")
    check(
        "course catalog loads",
        len(catalog.concepts) == 11 and len(catalog.misconceptions) == 22,
        f"{len(catalog.concepts)} concepts, {len(catalog.misconceptions)} misconceptions",
    )
    runs = run_all()
    check(
        "A* finds the optimal path; the student-error variants do not",
        runs["correct"]["cost"] == 4
        and all(runs[v]["cost"] > 4 for v in ("h_only", "fifo", "inadmissible")),
        ", ".join(f"{k}={v['cost']}" for k, v in runs.items()),
    )
    result = diagnose("Astar", STUDENT_TEXT["h_only"])
    top = result["ranked_causes"][0]
    check(
        "diagnosis finds the misconception in a student's words",
        top["misconception_id"] == "astar_uses_only_h",
        f"top cause {top['name']}",
    )
    print(
        "\n"
        + "\n".join("     " + line for line in explain(result).splitlines()[:6])
        + "\n"
    )

    print("2. Classroom CSP (module2_csp)")
    cfg = make_demo_classroom()
    res = solve_classroom(cfg)
    check("valid groups for a normal class", res.feasible, f"{res.num_groups} groups")
    overloaded = copy.deepcopy(cfg)
    for ta in overloaded.tas:
        ta.max_groups = 1
    res = solve_classroom(overloaded)
    check(
        "overloaded TAs are infeasible and explained",
        not res.feasible and any("[ta_workload]" in line for line in res.explanation),
        res.explanation[1] if len(res.explanation) > 1 else "",
    )

    print("\n3. Student model (module3_student_model)")
    engine = StudentModelEngine(catalog)
    for c in [False, False, False, True, True]:
        engine.update(quiz("improving", "Heuristic", c))
    for c in [True, True, False, False, False]:
        engine.update(quiz("declining", "Heuristic", c))
    p_up = engine.get_student("improving").concepts["Heuristic"].p_mastery
    p_down = engine.get_student("declining").concepts["Heuristic"].p_mastery
    check(
        "same 2/5 score, different histories, different mastery",
        p_up > p_down + 0.2,
        f"improving {p_up:.2f}, declining {p_down:.2f}",
    )
    for _ in range(4):
        engine.update(quiz("lacks", "Path_Cost", False))
        engine.update(quiz("misconceived", "Path_Cost", False, "g_is_remaining_cost"))
    lacks = engine.get_student("lacks")
    mis = engine.get_student("misconceived")
    check(
        "'not learned' is kept separate from 'misconception'",
        concept_status(lacks, "Path_Cost") == "not_learned"
        and concept_status(mis, "Path_Cost") == "misconception",
        f"P(misconception) {lacks.concepts['Path_Cost'].misconceptions['g_is_remaining_cost']:.2f} "
        f"vs {mis.concepts['Path_Cost'].misconceptions['g_is_remaining_cost']:.2f}",
    )
    check(
        "student-knowledge graph gives a study plan",
        "Path_Cost" in study_plan(mis, "Astar"),
        " -> ".join(study_plan(mis, "Astar")),
    )

    print("\n4. NLU (module5_nlu)")
    text = "I don't understand how A* search works."
    nlu = analyse_question(text)
    event = nlu_result_to_event("s01", nlu, catalog)
    check(
        "question intent and concept are recognised",
        event is not None
        and event.intent == Intent.CONFUSION
        and event.concept_id == "Astar",
        f"{nlu['intent']}, {nlu['primary_concept']}, confidence {nlu['nlu_confidence']}",
    )
    statement = "Path cost is the estimated remaining cost to the goal."
    check(
        "misconception found in a student's statement",
        match_misconceptions(statement)[0][0] == "g_is_remaining_cost",
        statement,
    )
    vague = analyse_question("What should I do?")
    check(
        "vague question gets low confidence",
        vague["nlu_confidence"] < 0.5,
        f"{vague['intent']}, confidence {vague['nlu_confidence']}",
    )

    print("\n5. Planning (module4_planner)")
    plan = plan_concept(frozenset())
    check(
        "new concept gets a lecture-quiz-discussion sequence",
        plan[:2] == [ActionType.LECTURE, ActionType.DIAGNOSTIC_QUIZ]
        and ActionType.PEER_DISCUSSION in plan,
        " -> ".join(a.value for a in plan),
    )
    replan = plan_concept(frozenset({"introduced", "evidence", "misconception"}))
    check(
        "plan changes when a quiz reveals a misconception",
        replan[0] == ActionType.TA_INTERVENTION,
        " -> ".join(a.value for a in replan),
    )
    check(
        "plan starts with revision when a prerequisite is weak",
        plan_concept(frozenset({"prereq_gap"}))[0] == ActionType.REVISION,
    )

    print("\n6. Reinforcement learning (module5_rl)")
    policy = RLTeachingPolicy.load(
        str(RL_DIR / "q_table_q_learning.npy"),
        str(RL_DIR / "visits_q_learning.npy"),
        time_budget=80,
    )
    profiles = make_classroom(catalog, "mixed", 12, seed=0)
    sim = ClassroomSimulator(catalog, profiles, seed=0)
    eng = StudentModelEngine(catalog, sim.student_ids())
    planner = AdaptivePlanner(catalog, policy=policy)
    planner.plan_next(eng)
    decision = planner.plan_next(eng)
    check(
        "trained Q-table loads and picks an action",
        decision is not None and decision.action is not None,
        f"{decision.action.value} on {decision.concept_id}",
    )

    print("\n7. CSP groups from live estimates")
    sim = ClassroomSimulator(
        catalog, make_classroom(catalog, "small_group_behind", 18, seed=1), seed=1
    )
    eng = StudentModelEngine(catalog, sim.student_ids())
    for cid in catalog.concept_ids()[:4]:
        for action in (
            ActionType.LECTURE,
            ActionType.EXAMPLE,
            ActionType.EXAMPLE,
            ActionType.DIAGNOSTIC_QUIZ,
        ):
            eng.update_many(sim.step(action, cid))
    groups = GroupScheduler(catalog, sim.student_ids(), seed=1).schedule(
        eng, "DFS", sim.student_ids(), "peer"
    )
    placed = sorted([s for g in groups.groups for s in g.student_ids] + groups.unplaced)
    check(
        "every student is placed in a group or reported",
        placed == sorted(sim.student_ids()),
        f"{len(groups.groups)} groups, {len(groups.unplaced)} unplaced",
    )
    for g in groups.groups[:3]:
        print(
            f"     {g.name}: {g.kind} on {g.concept_id}, {g.ta} at {g.slot}: {', '.join(g.student_ids)}"
        )

    print("\n8. Full system on every evaluation scenario (all modules together)")
    scenarios = [
        "mixed",
        "widespread_misconception",
        "small_group_behind",
        "dominant_talkers",
        "confident_wrong",
        "quiet_knowers",
    ]
    known_wins = behind_wins = 0
    for scenario in scenarios:
        adaptive = run_episode(
            scenario,
            n_students=24,
            seed=0,
            policy_name="plan",
            time_budget=80,
            verbose=False,
        )
        fixed = run_episode(
            scenario,
            n_students=24,
            seed=0,
            policy_name="fixed",
            time_budget=80,
            verbose=False,
        )
        known_wins += adaptive["known"] > fixed["known"]
        behind_wins += adaptive["left_behind"] <= fixed["left_behind"]
        print(
            f"     {scenario:25} adaptive: known {adaptive['known']:.2f}, left behind {adaptive['left_behind']:.2f}, "
            f"coverage {adaptive['coverage']:.2f}   |   fixed plan: known {fixed['known']:.2f}, "
            f"left behind {fixed['left_behind']:.2f}, coverage {fixed['coverage']:.2f}"
        )
    check(
        "adaptive teaching leaves fewer or equal students behind than the fixed plan",
        behind_wins >= 5,
        f"{behind_wins}/6 scenarios",
    )
    check(
        "adaptive teaching gives higher final mastery than the fixed plan",
        known_wins >= 5,
        f"{known_wins}/6 scenarios",
    )

    passed = sum(ok for _, ok in RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
