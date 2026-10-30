"""Tabular Q-learning and SARSA for the teaching policy.

    python -m module5_rl.training --algo both --episodes 2500 --out module5_rl/results

Each training episode is one simulated lesson series for a randomly drawn classroom
scenario (the six scenarios defined by Team 3) and a fresh random seed. Both algorithms are
epsilon-greedy with a linearly decaying epsilon, a per-(state, action) learning rate that
shrinks with the number of visits, and invalid actions masked out.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from module3_student_model.profiles import SCENARIOS
from module5_rl.actions import N_ACTIONS
from module5_rl.env import TeachingEnv
from module5_rl.state import STATE_SHAPE

ALPHA_MIN = 0.05


def learning_rate(visit_count: int) -> float:
    return max(ALPHA_MIN, 1.0 / (1.0 + visit_count) ** 0.7)


def eps_greedy(q_row: np.ndarray, mask: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    valid = np.flatnonzero(mask)
    if rng.random() < eps:
        return int(rng.choice(valid))
    masked = np.where(mask, q_row, -np.inf)
    best = np.flatnonzero(masked == masked.max())
    return int(rng.choice(best))


def td_update(
    Q: np.ndarray,
    N: np.ndarray,
    s,
    a: int,
    r: float,
    s2,
    a2: int | None,
    done: bool,
    gamma: float,
    algo: str,
    mask2: np.ndarray | None = None,
) -> float:
    """One temporal-difference update. Returns the TD error.

    Q-learning (off-policy): target = r + gamma * max_a' Q(s', a')
    SARSA      (on-policy) : target = r + gamma * Q(s', a'), a' = the action actually taken next
    """
    if done:
        target = r
    elif algo == "q_learning":
        q2 = Q[s2] if mask2 is None else np.where(mask2, Q[s2], -np.inf)
        target = r + gamma * float(np.max(q2))
    elif algo == "sarsa":
        target = r + gamma * float(Q[s2][a2])
    else:
        raise ValueError(f"unknown algorithm {algo!r}")
    td_error = target - Q[s][a]
    Q[s][a] += learning_rate(int(N[s][a])) * td_error
    N[s][a] += 1
    return float(td_error)


def train(
    algo: str,
    episodes: int,
    seed: int = 0,
    n_students: int = 12,
    time_budget: float = 45.0,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    decay_frac: float = 0.7,
    log_every: int = 100,
    out_dir: str | None = None,
    checkpoint_every: int = 250,
):
    if algo not in ("q_learning", "sarsa"):
        raise ValueError(algo)
    rng = np.random.default_rng(seed + 17)
    env = TeachingEnv(n_students=n_students, time_budget=time_budget)
    Q = np.zeros(STATE_SHAPE + (N_ACTIONS,))
    N = np.zeros(STATE_SHAPE + (N_ACTIONS,), dtype=np.int64)
    returns: list[float] = []
    t0 = time.time()

    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (decay_frac * episodes))
        scenario = SCENARIOS[int(rng.integers(len(SCENARIOS)))]
        s = env.reset(scenario, seed=seed * 1_000_003 + ep)
        a = eps_greedy(Q[s], env.valid_mask(), eps, rng)
        total = 0.0
        while True:
            s2, r, done, _ = env.step(a)
            total += r
            a2, mask2 = None, None
            if not done:
                mask2 = env.valid_mask()
                a2 = eps_greedy(Q[s2], mask2, eps, rng)
            td_update(Q, N, s, a, r, s2, a2, done, gamma, algo, mask2)
            if done:
                break
            s, a = s2, a2
        returns.append(total)

        if (ep + 1) % log_every == 0:
            visited = int((N.sum(axis=-1) > 0).sum())
            print(
                f"[{algo}] ep {ep + 1}/{episodes}  avg return (last {log_every}) "
                f"{np.mean(returns[-log_every:]):6.2f}  eps {eps:.2f}  states visited {visited}  "
                f"{time.time() - t0:5.0f}s",
                flush=True,
            )
        if out_dir and ((ep + 1) % checkpoint_every == 0 or ep + 1 == episodes):
            save(out_dir, algo, Q, N, returns, dict(
                algo=algo, episodes_done=ep + 1, seed=seed, n_students=n_students,
                time_budget=time_budget, gamma=gamma, eps_end=eps_end, decay_frac=decay_frac,
            ))
    return Q, N, np.array(returns)


def save(out_dir: str, algo: str, Q, N, returns, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"q_table_{algo}.npy"), Q)
    np.save(os.path.join(out_dir, f"visits_{algo}.npy"), N)
    np.save(os.path.join(out_dir, f"train_returns_{algo}.npy"), np.asarray(returns))
    with open(os.path.join(out_dir, f"train_meta_{algo}.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["q_learning", "sarsa", "both"], default="both")
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--students", type=int, default=12)
    ap.add_argument("--time-budget", type=float, default=45.0)
    ap.add_argument("--out", default="module5_rl/results")
    args = ap.parse_args()
    for algo in (["q_learning", "sarsa"] if args.algo == "both" else [args.algo]):
        train(algo, args.episodes, args.seed, args.students, args.time_budget, out_dir=args.out)


if __name__ == "__main__":
    main()
