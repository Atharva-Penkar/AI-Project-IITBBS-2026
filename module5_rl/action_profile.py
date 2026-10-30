"""Which actions does a policy choose in each scenario?  (evidence of adaptivity)

    python -m module5_rl.action_profile --out module5_rl/results
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

from module3_student_model.profiles import SCENARIOS
from module5_rl.evaluate import EVAL_SEED0, build_policies
from module5_rl.runner import run_policy_episode
from shared.schema import ActionType


def action_profile(policy, episodes: int = 10, n_students: int = 24, time_budget: float = 45.0) -> dict:
    out = {}
    for sc in SCENARIOS:
        c: Counter = Counter()
        targeted = total = 0
        for i in range(episodes):
            _, log = run_policy_episode(policy, sc, EVAL_SEED0 + i, n_students, time_budget, trace=True)
            for d in log:
                if d.action == ActionType.NEXT_CONCEPT:
                    continue
                c[d.action.value] += 1
                total += 1
                targeted += bool(d.target_student_ids)
        out[sc] = {"share": {k: v / total for k, v in c.items()}, "targeted_share": targeted / total}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="module5_rl/results")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    policies = build_policies(args.out, 45.0)
    kinds = [a.value for a in ActionType if a != ActionType.NEXT_CONCEPT]
    with open(os.path.join(args.out, "action_profile.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "scenario", "targeted_share", *kinds])
        for name in ("Q-learning", "SARSA"):
            prof = action_profile(policies[name], args.episodes)
            print(f"\n{name}")
            print(f"{'scenario':26s}{'targeted':>9s}" + "".join(f"{k[:9]:>10s}" for k in kinds))
            for sc, d in prof.items():
                row = [d["share"].get(k, 0.0) for k in kinds]
                print(f"{sc:26s}{d['targeted_share']:>9.2f}" + "".join(f"{x:>10.2f}" for x in row))
                w.writerow([name, sc, f"{d['targeted_share']:.3f}", *[f"{x:.3f}" for x in row]])


if __name__ == "__main__":
    main()
