"""Compare the learned policies with fixed and rule-based strategies.

    python -m module5_rl.evaluate --episodes 25 --budgets 45 90 --out module5_rl/results

Every policy is run through ``runner.run_policy_episode`` (the real AdaptivePlanner path) on the
same classrooms (same scenario + seed), so differences are paired. Evaluation classes have 24
students; training used 12, so this also checks generalisation to a different class size.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from module3_student_model.profiles import SCENARIOS
from module4_planner.policy import HeuristicTeachingPolicy
from module5_rl.policies import FixedLecturePlanPolicy, RandomPolicy, RLTeachingPolicy, ThresholdRepeatPolicy
from module5_rl.runner import run_policy_episode

EVAL_SEED0 = 5_000_000  # training seeds are < 3_000_000, so evaluation classrooms are unseen
METRICS = [
    "known", "gain", "left_behind", "coverage", "misconceptions_left", "engagement",
    "participation_balance", "time_used", "gain_low", "gain_high", "equity_gap",
]


def build_policies(results_dir: str, time_budget: float) -> dict:
    policies: dict = {}
    for label, algo in [("Q-learning", "q_learning"), ("SARSA", "sarsa")]:
        q_path = os.path.join(results_dir, f"q_table_{algo}.npy")
        if os.path.exists(q_path):
            policies[label] = RLTeachingPolicy.load(
                q_path, os.path.join(results_dir, f"visits_{algo}.npy"), time_budget=time_budget, name=algo
            )
    heuristic = HeuristicTeachingPolicy()
    heuristic.name = "Heuristic"
    policies["Heuristic"] = heuristic
    policies["FixedPlan"] = FixedLecturePlanPolicy()
    policies["ThresholdRepeat"] = ThresholdRepeatPolicy()
    policies["Random"] = RandomPolicy(seed=0)
    return policies


def evaluate(policies: dict, episodes: int, n_students: int, time_budget: float) -> dict:
    raw: dict = {}
    for name, pol in policies.items():
        raw[name] = {}
        for sc in SCENARIOS:
            raw[name][sc] = [
                run_policy_episode(pol, sc, EVAL_SEED0 + i, n_students=n_students, time_budget=time_budget)
                for i in range(episodes)
            ]
            if isinstance(pol, RandomPolicy):
                pol.rng = np.random.default_rng(1)
        print(f"  evaluated {name}", flush=True)
    return raw


def summarise(raw: dict) -> dict:
    out: dict = {}
    for name, per_sc in raw.items():
        out[name] = {}
        for sc, runs in per_sc.items():
            out[name][sc] = {}
            for k in runs[0]:
                x = np.array([r[k] for r in runs])
                out[name][sc][k] = (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0)
        pooled = [r for sc in per_sc.values() for r in sc]
        out[name]["ALL"] = {
            k: (float(np.mean([r[k] for r in pooled])), float(np.std([r[k] for r in pooled], ddof=1) / np.sqrt(len(pooled))))
            for k in pooled[0]
        }
    return out


def paired_difference(raw: dict, a: str, b: str, metric: str) -> tuple[float, float]:
    """Mean and 95% half-width of (a - b) over all paired episodes (all scenarios pooled)."""
    d = np.array(
        [ra[metric] - rb[metric] for sc in SCENARIOS for ra, rb in zip(raw[a][sc], raw[b][sc])]
    )
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))


def format_tables(summary: dict, metrics: list[str]) -> str:
    names = list(summary)
    lines = []
    for m in metrics:
        lines.append(f"\n{m.upper()}  (mean over episodes)")
        lines.append(f"{'scenario':28s}" + "".join(f"{n:>17s}" for n in names))
        for sc in [*SCENARIOS, "ALL"]:
            lines.append(f"{sc:28s}" + "".join(f"{summary[n][sc][m][0]:>17.3f}" for n in names))
    return "\n".join(lines)


def save_results(raw: dict, summary: dict, out_dir: str, tag: str) -> None:
    with open(os.path.join(out_dir, f"evaluation_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, f"evaluation_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "scenario", "metric", "mean", "std_error"])
        for n, per_sc in summary.items():
            for sc, ms in per_sc.items():
                for m, (mu, se) in ms.items():
                    w.writerow([n, sc, m, f"{mu:.4f}", f"{se:.4f}"])


def make_plots(summary: dict, out_dir: str, tag: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary)
    cols = [*SCENARIOS, "ALL"]
    for metric, title in [
        ("known", "Final mean mastery over the whole syllabus (higher is better)"),
        ("left_behind", "Share of students left behind (lower is better)"),
        ("coverage", "Syllabus coverage (concepts completed)"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 4.6))
        bw = 0.8 / len(names)
        x = np.arange(len(cols))
        for i, n in enumerate(names):
            ax.bar(x + i * bw, [summary[n][c][metric][0] for c in cols], bw,
                   yerr=[summary[n][c][metric][1] for c in cols], label=n, capsize=2)
        ax.set_xticks(x + bw * (len(names) - 1) / 2)
        ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=8)
        ax.set_title(f"{title}  [{tag}]")
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"compare_{metric}_{tag}.png"), dpi=140)
        plt.close(fig)


def plot_learning_curves(results_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    w = 100
    for label, algo in [("Q-learning", "q_learning"), ("SARSA", "sarsa")]:
        p = os.path.join(results_dir, f"train_returns_{algo}.npy")
        if os.path.exists(p):
            r = np.load(p)
            if len(r) > w:
                ma = np.convolve(r, np.ones(w) / w, mode="valid")
                ax.plot(np.arange(len(ma)) + w, ma, label=label)
    ax.set_xlabel("training episode")
    ax.set_ylabel(f"episode return ({w}-episode moving average)")
    ax.set_title("Learning curves (exploration decays during training)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "learning_curves.png"), dpi=140)
    plt.close(fig)


def write_decision_trace(policy, out_path: str, scenario: str, seed: int, time_budget: float, n_students: int) -> None:
    _, log = run_policy_episode(policy, scenario, seed, n_students=n_students, time_budget=time_budget, trace=True)
    with open(out_path, "w") as f:
        f.write(f"Decision trace: {policy.name}, scenario={scenario}, seed={seed}, budget={time_budget}\n\n")
        for i, d in enumerate(log, 1):
            tgt = f"{len(d.target_student_ids)} students" if d.target_student_ids else "whole class"
            mis = f", misconception={d.target_misconception}" if d.target_misconception else ""
            f.write(f"{i:2d}. [{d.concept_id}] {d.action.value} -> {tgt}{mis}\n    {d.rationale}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=25, help="episodes per scenario")
    ap.add_argument("--students", type=int, default=24)
    ap.add_argument("--budgets", type=float, nargs="+", default=[45.0, 90.0])
    ap.add_argument("--out", default="module5_rl/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_learning_curves(args.out)

    for budget in args.budgets:
        tag = f"budget{int(budget)}"
        print(f"\n=== evaluation, time budget {budget} ===")
        policies = build_policies(args.out, budget)
        raw = evaluate(policies, args.episodes, args.students, budget)
        with open(os.path.join(args.out, f"raw_episodes_{tag}.json"), "w") as f:
            json.dump(raw, f)  # keep the raw per-episode results in case formatting fails
        summary = summarise(raw)
        text = format_tables(summary, ["known", "left_behind", "coverage", "misconceptions_left",
                                       "engagement", "participation_balance", "equity_gap", "time_used"])
        rl = [n for n in ("Q-learning", "SARSA") if n in raw]
        text += "\n\nPAIRED DIFFERENCES (all scenarios pooled; mean +/- 95% CI)\n"
        for a in rl:
            for b in [n for n in raw if n != a]:
                row = [f"{m}: {paired_difference(raw, a, b, m)[0]:+.3f} +/- {paired_difference(raw, a, b, m)[1]:.3f}"
                       for m in ("known", "left_behind", "coverage")]
                text += f"{a} - {b:16s} " + " | ".join(row) + "\n"
        if "Q-learning" in policies:
            text += "\nRL fallback rate to rule-based policy (unseen states): " + ", ".join(
                f"{n}={np.mean([r['fallback_rate'] for sc in SCENARIOS for r in raw[n][sc]]):.3f}" for n in rl
            ) + "\n"
        print(text)
        with open(os.path.join(args.out, f"evaluation_{tag}.txt"), "w") as f:
            f.write(text)
        save_results(raw, summary, args.out, tag)
        make_plots(summary, args.out, tag)
        if budget == args.budgets[0] and "Q-learning" in policies:
            write_decision_trace(policies["Q-learning"], os.path.join(args.out, "decision_trace_small_group.txt"),
                                 "small_group_behind", EVAL_SEED0, budget, args.students)
            write_decision_trace(policies["Q-learning"], os.path.join(args.out, "decision_trace_confident_wrong.txt"),
                                 "confident_wrong", EVAL_SEED0, budget, args.students)


if __name__ == "__main__":
    main()
