"""Sanity check: can a trivial fixed sequence match the learned policy?

    python -m module5_rl.ablation --out module5_rl/results

"Example xN" = N worked examples per concept, then move on (whole class, no adaptation).
If such a policy matches the RL policy, the simulator mostly rewards one good action and the
RL result says little about adaptivity. Read this table before claiming that RL "adapts".
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from module3_student_model.profiles import SCENARIOS
from module5_rl.actions import RLAction
from module5_rl.evaluate import EVAL_SEED0
from module5_rl.policies import MANUAL_PROGRESSION, RLTeachingPolicy, whole_class_decision
from module5_rl.runner import run_policy_episode


class FixedSequence:
    def __init__(self, seq, name: str) -> None:
        self.seq, self.name, self.config = list(seq), name, MANUAL_PROGRESSION

    def reset(self) -> None: ...

    def choose_action(self, summary, students, concept_id, action_counts=None):
        k = sum((action_counts or {}).values())
        action = self.seq[k] if k < len(self.seq) else RLAction.ADVANCE
        return whole_class_decision(action, concept_id, "fixed sequence", "ablation")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="module5_rl/results")
    ap.add_argument("--episodes", type=int, default=10, help="per scenario")
    args = ap.parse_args()
    E, P, X = RLAction.EXAMPLE, RLAction.PEER_DISCUSSION, RLAction.EXERCISE
    pols = {
        "Example x3": FixedSequence([E] * 3, "e3"),
        "Example x4": FixedSequence([E] * 4, "e4"),
        "Example x6": FixedSequence([E] * 6, "e6"),
        "Example x7": FixedSequence([E] * 7, "e7"),
        "E,E,E,X,E,E": FixedSequence([E, E, E, X, E, E], "mix1"),
        "E,E,P,E,E": FixedSequence([E, E, P, E, E], "mix2"),
    }
    ref_path = os.path.join(args.out, "q_table_q_learning.npy")
    pols["Q-learning (reference)"] = RLTeachingPolicy.load(
        ref_path, os.path.join(args.out, "visits_q_learning.npy"), time_budget=45.0, name="q_learning"
    )
    res = {}
    lines = [f"{'policy':26s}{'known':>8s}{'left_behind':>13s}{'coverage':>10s}{'time_used':>11s}"]
    for name, p in pols.items():
        runs = [run_policy_episode(p, sc, EVAL_SEED0 + i, 24, 45.0) for sc in SCENARIOS for i in range(args.episodes)]
        res[name] = runs
        lines.append(
            f"{name:26s}{np.mean([r['known'] for r in runs]):8.3f}{np.mean([r['left_behind'] for r in runs]):13.3f}"
            f"{np.mean([r['coverage'] for r in runs]):10.2f}{np.mean([r['time_used'] for r in runs]):11.1f}"
        )
    lines.append("\nPaired difference in final mastery, Q-learning minus each fixed sequence (mean +/- 95% CI):")
    ref = res["Q-learning (reference)"]
    for name in list(pols)[:-1]:
        d = np.array([a["known"] - b["known"] for a, b in zip(ref, res[name])])
        lines.append(f"  vs {name:24s}{d.mean():+.3f} +/- {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "ablation.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
