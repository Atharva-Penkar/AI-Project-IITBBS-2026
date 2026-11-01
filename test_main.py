import pytest

from main import run_episode


@pytest.mark.parametrize("policy", ["plan", "heuristic", "rl"])
def test_full_loop_runs_with_every_module(policy):
    metrics = run_episode("small_group_behind", n_students=12, seed=0, policy_name=policy, time_budget=20, verbose=False)
    assert 0.0 <= metrics["known"] <= 1.0
    assert metrics["steps"] > 0
