# module5_rl: Reinforcement-learning teaching policy (Team 5, Person A)

A tabular **Q-learning** and **SARSA** teaching policy that plugs into Team 4's `AdaptivePlanner`
through the existing `TeachingPolicy` interface, trained and evaluated on Team 3's classroom
simulator and student-model engine. No existing file is modified; the module only adds a new folder.

## Formulation

**Finite-horizon MDP** (not a bandit): actions change the future state (prerequisite repair, engagement,
time budget, progress through the syllabus), so rewards are delayed and an action's value depends on what follows.

| Item | Definition |
|---|---|
| State (7 features, 1,458 states) | Team 3's `discretise` (mastery, misconception prevalence, engagement, participation balance, share left behind; each 0-2) + `uncertainty` (0/1) + `time_pace` (behind / on track / ahead of schedule). Built only from the engine's *estimates*. |
| Actions (9) | lecture, example, diagnostic quiz, peer discussion, exercise, revision, **TA support** (targeted at misconception holders / weakest), **peer group** (targeted at students below mastery threshold), **advance** (ask the planner to move on). `select_targets` chooses *who* an action is aimed at. |
| Reward (uses the simulator's TRUE state) | 10 x learning gain + 2 x misconceptions removed + 4 x engagement change + 2 x participation-balance change - 0.1 x class time; +1 per concept completed, - 3 x share of students not yet mastering a concept being left; terminal: + 10 x final mean mastery - 5 x share left behind. Weights are in `env.RewardWeights`. |
| Episode | One simulated course (6 A* concepts) with a class-time budget (45 units by default; actions cost 0-1.5 units). |
| Algorithms | epsilon-greedy Q-learning and SARSA, linearly decaying epsilon, visit-count learning rate, invalid actions masked (`ADVANCE` is illegal before anything is taught). |
| Safety | A state never seen in training falls back to Team 4's rule-based policy (fallback rate in evaluation: 0.2% / 0%). |
| Explainability | Every decision's `rationale` states the class state, the chosen action's Q-value and the runner-up's. |

## Files

| File | Purpose |
|---|---|
| `actions.py`, `state.py` | action space, targeting, time costs; state encoding |
| `env.py` | training environment (simulator + engine + planner), reward |
| `policies.py` | `RLTeachingPolicy`, `AdvancingPlanner`, baselines (`FixedLecturePlanPolicy`, `ThresholdRepeatPolicy`, `RandomPolicy`) |
| `runner.py` | runs *any* `TeachingPolicy` through the real planner path (used for all evaluation) |
| `training.py`, `evaluate.py`, `action_profile.py`, `ablation.py` | train, compare, inspect behaviour, sanity-check |
| `tests/test_rl.py` | 22 tests |
| `results/` | trained Q-tables, tables (`evaluation_budget45/90.txt|csv|json`), plots, decision traces, `ablation.txt`, `action_profile.csv` |

## Run (from the repository root)

```bash
pytest module5_rl
python -m module5_rl.training --algo both --episodes 2500 --out module5_rl/results   # ~25 min on one core
python -m module5_rl.evaluate --episodes 20 --budgets 45 90 --out module5_rl/results
python -m module5_rl.action_profile --out module5_rl/results
python -m module5_rl.ablation --out module5_rl/results
```

Use it in the planner:

```python
from module4_planner.planner import AdaptivePlanner
from module5_rl.policies import AdvancingPlanner, RLTeachingPolicy

policy = RLTeachingPolicy.load("module5_rl/results/q_table_q_learning.npy",
                               "module5_rl/results/visits_q_learning.npy", time_budget=45.0)
planner = AdvancingPlanner(catalog, policy=policy)   # AdaptivePlanner + "policy may advance early"
decision = planner.plan_next(engine)                  # ActionDecision with rationale
```
`AdvancingPlanner` is needed only because the stock planner advances solely on its mastery criterion.
`RLTeachingPolicy` keeps its own clock of class time (the interface has no time argument); call
`policy.set_time_left(t)` if the controller knows the real time.

## Results (evaluation classes of 24 students, unseen seeds; training used 12 students)

20 episodes x 6 scenarios = 120 paired episodes per policy. "known" = final share of (student, concept)
pairs truly mastered; "left behind" = share of students who mastered < 40% of the syllabus.

| Policy | known (45) | left behind (45) | coverage (45) | known (90) | left behind (90) | coverage (90) |
|---|---|---|---|---|---|---|
| Q-learning | **0.687** | 0.161 | 0.81 | **0.754** | **0.117** | 1.00 |
| SARSA | 0.656 | **0.157** | 0.65 | 0.726 | 0.124 | 0.99 |
| Team 4 heuristic | 0.517 | 0.359 | 0.36 | 0.616 | 0.235 | 0.67 |
| Fixed lecture plan | 0.557 | 0.332 | 1.00 | 0.557 | 0.332 | 1.00 |
| Threshold-repeat | 0.520 | 0.383 | 1.00 | 0.520 | 0.383 | 1.00 |
| Random | 0.595 | 0.273 | 0.81 | 0.614 | 0.254 | 1.00 |

Paired difference in `known`, Q-learning minus fixed plan: +0.130 +/- 0.010 (budget 45), +0.197 +/- 0.010 (budget 90).
The policy was trained only at budget 45 and still generalises to 90 and to a larger class.

## Limitations (please read before writing the report)

1. **A trivial policy matches the learned one.** "Give 7 examples per concept, then move on" (no adaptation)
   scores known 0.717 / left-behind 0.14 / full coverage, versus Q-learning 0.688 / 0.16 / 0.78 coverage
   (`results/ablation.txt`, difference -0.030 +/- 0.011). In this simulator, worked examples are the strongest
   single action, and the standard baselines do not exploit that. So the table above shows the learned policy beats the
   *given* baselines; it does **not** show that adaptive teaching beats the best fixed strategy.
2. **Limited scenario-specific behaviour.** `results/action_profile.csv`: Q-learning gives examples 59-77% of the time
   in every scenario, SARSA about 40-55% examples + about 40% peer discussion; targeted actions are only 6-16% of moves.
   Differences between scenarios exist (e.g. more quizzes with dominant/quiet students) but are modest.
   The Q-value gaps behind many choices are small (see `decision_trace_*.txt`).
3. **Under-covers the syllabus at 45 units** (Q-learning 0.81, SARSA 0.65): it spends time per concept like the
   examples-only policy but does not learn to advance quickly enough.
4. **Participation balance is not improved**: Team 4's heuristic is higher (0.74 vs 0.70), notably in the dominant-talkers scenario.
5. The reward weights and simulator are design choices; results are only as realistic as Team 3's simulator.
6. SARSA's saved checkpoint is at 2,250 of 2,500 planned episodes (the run was interrupted after epsilon had reached its floor).
   One training seed per algorithm; SARSA vs Q-learning (+0.031 +/- 0.012) should not be over-interpreted.
7. All code was run with Python 3.12 in an environment without network access: the 22 tests were run with a minimal
   stand-in for `pydantic` and a plain test runner, not real `pytest`. Please run `pytest module5_rl` on the project's Python 3.13 setup.

## Suggested next steps

* Ask Team 3 whether the dominance of EXAMPLE in the simulator is intended; adding diminishing returns for repeated
  examples would force genuine adaptivity.
* Add action-history and per-group (not class-average) features to the state; add a stronger penalty for slow coverage.
* Evaluate with several training seeds; add the examples-only policy as a standard baseline in the report.
* Optional: feed Team 5 Person B's NLU question analysis into the misconception-prevalence feature.
