"""Tests for the RL teaching-policy module.  Run from the repository root:  pytest module5_rl"""

from __future__ import annotations

import numpy as np

from module3_student_model.engine import StudentModelEngine
from module3_student_model.profiles import make_classroom
from module3_student_model.sample_catalog import astar_catalog
from module3_student_model.simulator import ClassroomSimulator
from module4_planner.models import PlannerConfig
from module4_planner.tests.helpers import make_students, make_summary
from module5_rl.actions import (
    N_ACTIONS,
    RLAction,
    build_decision,
    time_cost,
    valid_actions_mask,
)
from module5_rl.env import TeachingEnv
from module5_rl.policies import (
    AdvancingPlanner,
    FixedLecturePlanPolicy,
    RandomPolicy,
    RLTeachingPolicy,
    ThresholdRepeatPolicy,
    whole_class_decision,
)
from module5_rl.runner import run_policy_episode
from module5_rl.state import STATE_SHAPE, describe_state, encode_state, time_pace_level
from module5_rl.training import eps_greedy, td_update, train
from shared.schema import ActionType


def _raises(exc_type, fn) -> bool:
    try:
        fn()
    except exc_type:
        return True
    return False


def _classroom(scenario="mixed", n=12, seed=3):
    catalog = astar_catalog()
    sim = ClassroomSimulator(catalog, make_classroom(catalog, scenario, n, seed=seed), seed=seed)
    return catalog, sim, StudentModelEngine(catalog, sim.student_ids())


# ----------------------------------------------------------------------------- state / actions
def test_state_encoding_is_in_range_and_describable() -> None:
    students = make_students(["s1", "s2", "s3"])
    summary = make_summary(students)
    state = encode_state(summary, "topic", time_left=20.0, time_budget=45.0)
    assert len(state) == len(STATE_SHAPE)
    assert all(0 <= v < n for v, n in zip(state, STATE_SHAPE))
    assert "mastery=" in describe_state(state) and "time_pace=" in describe_state(state)


def test_time_pace_levels() -> None:
    # 6 concepts, 4 still to go: a fair share of the 45 units would be 30
    assert time_pace_level(10.0, 45.0, 4, 6) == 0  # behind schedule
    assert time_pace_level(30.0, 45.0, 4, 6) == 1  # on track
    assert time_pace_level(45.0, 45.0, 4, 6) == 2  # ahead
    assert time_pace_level(0.0, 45.0, 4, 6) == 0


def test_every_action_builds_a_valid_decision() -> None:
    ids = ["s1", "s2", "s3", "s4"]
    students = make_students(ids, mastery={"s1": 0.2, "s2": 0.3, "s3": 0.8, "s4": 0.9})
    summary = make_summary(students)
    for action in RLAction:
        d = build_decision(action, summary, students, "topic", PlannerConfig())
        assert set(d.target_student_ids) <= set(ids)
        assert len(d.target_student_ids) < len(ids)  # whole class is encoded as ()
        assert d.metadata["rl_action"] == action.name
    adv = build_decision(RLAction.ADVANCE, summary, students, "topic", PlannerConfig())
    assert adv.action == ActionType.NEXT_CONCEPT and adv.metadata["rule"] == "policy_advance"


def test_ta_support_targets_the_prevalent_misconception() -> None:
    ids = ["s1", "s2", "s3", "s4"]
    students = make_students(
        ids,
        misconceptions={"s1": {"confuses_g_and_h": 0.9}, "s2": {"confuses_g_and_h": 0.8}, "s3": {"confuses_g_and_h": 0.1}},
    )
    summary = make_summary(students, misconception_prevalence={"confuses_g_and_h": 0.5})
    d = build_decision(RLAction.TA_SUPPORT, summary, students, "topic", PlannerConfig())
    assert d.action == ActionType.TA_INTERVENTION
    assert d.target_misconception == "confuses_g_and_h"
    assert set(d.target_student_ids) == {"s1", "s2"}


def test_peer_discussion_targets_quiet_students_when_unbalanced() -> None:
    students = make_students(["s1", "s2", "s3"], participation={"s1": 0.05, "s2": 0.4, "s3": 0.9})
    unbalanced = make_summary(students, participation_gini=0.6)
    balanced = make_summary(students, participation_gini=0.1)
    cfg = PlannerConfig()
    assert build_decision(RLAction.PEER_DISCUSSION, unbalanced, students, "topic", cfg).target_student_ids == ("s1",)
    assert build_decision(RLAction.PEER_DISCUSSION, balanced, students, "topic", cfg).target_student_ids == ()


def test_advance_is_masked_before_any_teaching() -> None:
    assert not valid_actions_mask({})[RLAction.ADVANCE]
    assert not valid_actions_mask(None)[RLAction.ADVANCE]
    assert valid_actions_mask({"lecture": 1})[RLAction.ADVANCE]
    assert valid_actions_mask({"lecture": 1}).sum() == N_ACTIONS


# ----------------------------------------------------------------------------- learning rules
def test_td_targets_differ_between_q_learning_and_sarsa() -> None:
    def fresh():
        Q = np.zeros((2, 2))
        Q[1] = [10.0, 2.0]  # greedy next action is 0, but the behaviour policy takes 1
        return Q, np.zeros((2, 2), dtype=np.int64)

    Qq, Nq = fresh()
    err_q = td_update(Qq, Nq, 0, 0, r=1.0, s2=1, a2=1, done=False, gamma=0.9, algo="q_learning")
    Qs, Ns = fresh()
    err_s = td_update(Qs, Ns, 0, 0, r=1.0, s2=1, a2=1, done=False, gamma=0.9, algo="sarsa")
    assert abs(err_q - (1.0 + 0.9 * 10.0)) < 1e-9  # off-policy: max over next actions
    assert abs(err_s - (1.0 + 0.9 * 2.0)) < 1e-9  # on-policy: the action actually taken
    assert Nq[0, 0] == 1


def test_td_update_masks_invalid_next_actions() -> None:
    Q = np.zeros((2, 2))
    Q[1] = [10.0, 2.0]
    N = np.zeros((2, 2), dtype=np.int64)
    err = td_update(Q, N, 0, 0, 1.0, 1, None, False, 0.9, "q_learning", mask2=np.array([False, True]))
    assert abs(err - (1.0 + 0.9 * 2.0)) < 1e-9


def test_both_algorithms_solve_a_toy_chain_mdp() -> None:
    """s0 --go--> s1 --go--> end (+10);  'quit' ends immediately for +1. Optimal: always 'go'."""
    for algo in ("q_learning", "sarsa"):
        rng = np.random.default_rng(0)
        Q, N = np.zeros((2, 2)), np.zeros((2, 2), dtype=np.int64)
        mask = np.array([True, True])
        for _ in range(1500):
            s, a = 0, eps_greedy(Q[0], mask, 0.3, rng)
            while True:
                if a == 1:  # quit
                    td_update(Q, N, s, a, 1.0, None, None, True, 0.9, algo)
                    break
                if s == 1:  # go at the last state
                    td_update(Q, N, s, a, 10.0, None, None, True, 0.9, algo)
                    break
                s2 = 1
                a2 = eps_greedy(Q[s2], mask, 0.3, rng)
                td_update(Q, N, s, a, 0.0, s2, a2, False, 0.9, algo, mask)
                s, a = s2, a2
        assert Q[0].argmax() == 0 and Q[1].argmax() == 0, algo
        assert Q[1, 0] > 8.0, algo  # close to the true value of 10 (SARSA is a bit lower: eps-greedy)


def test_short_training_run_updates_the_table() -> None:
    Q, N, returns = train("q_learning", episodes=4, seed=1, n_students=6, log_every=1000)
    assert np.isfinite(Q).all() and N.sum() > 0 and len(returns) == 4
    assert (Q[N == 0] == 0).all()  # only visited (state, action) pairs are changed


# ----------------------------------------------------------------------------- policy interface
def _random_table(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, STATE_SHAPE + (N_ACTIONS,)), np.ones(STATE_SHAPE + (N_ACTIONS,), dtype=np.int64)


def test_policy_runs_inside_the_real_planner_loop() -> None:
    catalog, sim, engine = _classroom()
    Q, V = _random_table()
    planner = AdvancingPlanner(catalog, policy=RLTeachingPolicy(Q, V))
    seen = set()
    for _ in range(25):
        d = planner.plan_next(engine)
        if d is None:
            break
        events = sim.step(d.action, d.concept_id, list(d.target_student_ids) or None, d.target_misconception)
        engine.update_many(events)
        planner.record_execution(d)
        seen.add(d.action)
        assert d.rationale
    assert len(seen) > 1


def test_unseen_states_fall_back_to_the_rule_based_policy() -> None:
    students = make_students(["s1", "s2", "s3"])
    summary = make_summary(students)
    policy = RLTeachingPolicy(np.zeros(STATE_SHAPE + (N_ACTIONS,)), np.zeros(STATE_SHAPE + (N_ACTIONS,), dtype=np.int64))
    d = policy.choose_action(summary, students, "topic", {})
    assert d.rationale.startswith("[unseen state") and d.metadata["fallback"] is True
    assert policy.fallback_rate == 1.0


def test_policy_picks_the_highest_valued_tried_action_and_explains_it() -> None:
    students = make_students(["s1", "s2", "s3"])
    summary = make_summary(students)
    policy = RLTeachingPolicy(np.zeros(STATE_SHAPE + (N_ACTIONS,)), np.zeros(STATE_SHAPE + (N_ACTIONS,), dtype=np.int64))
    state = encode_state(summary, "topic", policy.time_left, policy.time_budget)
    policy.Q[state][RLAction.EXAMPLE] = 5.0
    policy.Q[state][RLAction.LECTURE] = 2.0
    policy.visits[state][[RLAction.EXAMPLE, RLAction.LECTURE]] = 3
    d = policy.choose_action(summary, students, "topic", {"lecture": 1})
    assert d.action == ActionType.EXAMPLE
    assert "EXAMPLE" in d.rationale and "LECTURE" in d.rationale and "mastery=" in d.rationale
    assert d.metadata["fallback"] is False and 0.3 <= d.priority <= 0.95


def test_policy_clock_advances_with_chosen_actions() -> None:
    students = make_students(["s1", "s2", "s3"])
    summary = make_summary(students)
    Q, V = _random_table(1)
    policy = RLTeachingPolicy(Q, V, time_budget=30.0)
    d = policy.choose_action(summary, students, "topic", {"lecture": 1})
    assert abs(policy.time_left - (30.0 - time_cost(d.action))) < 1e-9
    policy.reset()
    assert policy.time_left == 30.0


# ----------------------------------------------------------------------------- planner integration
def test_policy_can_advance_the_planner_to_the_next_concept() -> None:
    class AlwaysAdvance:
        config = FixedLecturePlanPolicy().config

        def choose_action(self, summary, students, concept_id, action_counts=None):
            return whole_class_decision(RLAction.ADVANCE, concept_id, "move on", "test")

    catalog, sim, engine = _classroom()
    planner = AdvancingPlanner(catalog, policy=AlwaysAdvance())
    first = planner.plan_next(engine)
    assert first.action == ActionType.NEXT_CONCEPT and first.concept_id == "graph_search"
    second = planner.plan_next(engine)  # the policy asks to advance -> planner transitions
    assert second.metadata["rule"] == "concept_transition" and second.concept_id != "graph_search"
    assert "graph_search" in planner.completed_concepts


def test_fixed_plan_follows_the_plan_and_never_self_advances() -> None:
    catalog, sim, engine = _classroom("mixed", n=8)
    planner = AdvancingPlanner(catalog, policy=FixedLecturePlanPolicy())
    actions = []
    for _ in range(14):
        d = planner.plan_next(engine)
        if d.action == ActionType.NEXT_CONCEPT:
            actions.append(("NEXT", d.concept_id))
        else:
            actions.append(d.action)
        engine.update_many(sim.step(d.action, d.concept_id, None, None))
        planner.record_execution(d)
    plan = [ACTION_FOR[a] for a in FixedLecturePlanPolicy.PLAN]
    assert actions[0] == ("NEXT", "graph_search")
    assert actions[1:6] == plan
    assert actions[6][0] == "NEXT" and actions[6][1] != "graph_search"


ACTION_FOR = {
    RLAction.LECTURE: ActionType.LECTURE,
    RLAction.EXAMPLE: ActionType.EXAMPLE,
    RLAction.EXERCISE: ActionType.EXERCISE,
    RLAction.PEER_DISCUSSION: ActionType.PEER_DISCUSSION,
    RLAction.DIAGNOSTIC_QUIZ: ActionType.DIAGNOSTIC_QUIZ,
}


def test_threshold_policy_repeats_until_the_average_is_high_enough() -> None:
    students = make_students(["s1", "s2"])
    pol = ThresholdRepeatPolicy(threshold=0.6, max_lectures=3)
    low = make_summary(students, mean_mastery=0.3)
    high = make_summary(students, mean_mastery=0.8)
    assert pol.choose_action(low, students, "topic", {}).action == ActionType.LECTURE
    assert pol.choose_action(low, students, "topic", {"lecture": 1}).action == ActionType.DIAGNOSTIC_QUIZ
    assert pol.choose_action(low, students, "topic", {"lecture": 1, "diagnostic_quiz": 1}).action == ActionType.LECTURE
    assert pol.choose_action(high, students, "topic", {"lecture": 1, "diagnostic_quiz": 1}).action == ActionType.NEXT_CONCEPT
    capped = {"lecture": 3, "diagnostic_quiz": 3}
    assert pol.choose_action(low, students, "topic", capped).action == ActionType.NEXT_CONCEPT


# ----------------------------------------------------------------------------- environment
def _play(env, scenario, seed, actions_rng_seed=0):
    rng = np.random.default_rng(actions_rng_seed)
    env.reset(scenario, seed)
    rewards = []
    while True:
        a = int(rng.choice(np.flatnonzero(env.valid_mask())))
        _, r, done, _ = env.step(a)
        rewards.append(r)
        if done:
            return rewards


def test_env_is_deterministic_for_a_seed() -> None:
    a = _play(TeachingEnv(n_students=6), "mixed", 11)
    b = _play(TeachingEnv(n_students=6), "mixed", 11)
    assert a == b


def test_env_terminates_and_metrics_are_valid() -> None:
    env = TeachingEnv(n_students=6)
    _play(env, "small_group_behind", 4)
    m = env.metrics()
    assert 0.0 <= m["known"] <= 1.0 and 0.0 <= m["left_behind"] <= 1.0 and 0.0 <= m["coverage"] <= 1.0
    assert m["time_used"] <= env.T + 1e-9
    assert _raises(RuntimeError, lambda: env.step(RLAction.LECTURE))


def test_env_rejects_advancing_before_teaching() -> None:
    env = TeachingEnv(n_students=6)
    env.reset("mixed", 0)
    assert _raises(ValueError, lambda: env.step(RLAction.ADVANCE))


def test_training_env_and_production_planner_path_agree() -> None:
    """The policy sees identical states and makes identical choices in both paths."""
    Q, V = _random_table(5)
    policy = RLTeachingPolicy(Q, V, time_budget=45.0)
    for scenario, seed in [("mixed", 5), ("confident_wrong", 6)]:
        metrics, log = run_policy_episode(policy, scenario, seed, n_students=8, trace=True)
        env = TeachingEnv(n_students=8)
        s = env.reset(scenario, seed)
        chosen = []
        while True:
            a = int(np.argmax(np.where(env.valid_mask(), Q[s], -np.inf)))
            chosen.append(RLAction(a).name)
            s, _, done, _ = env.step(a)
            if done:
                break
        executed = [d.metadata["rl_action"] for d in log if d.metadata.get("rule") == "rl_policy"]
        assert executed == [c for c in chosen if c != "ADVANCE"]
        assert abs(metrics["known"] - env.metrics()["known"]) < 1e-12


def test_random_policy_and_baselines_complete_an_episode() -> None:
    for pol in (RandomPolicy(0), FixedLecturePlanPolicy(), ThresholdRepeatPolicy()):
        m = run_policy_episode(pol, "dominant_talkers", 2, n_students=8)
        assert 0.0 <= m["known"] <= 1.0 and m["time_used"] <= 45.0 + 1e-9
