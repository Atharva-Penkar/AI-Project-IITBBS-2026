# Reinforcement-learning teaching policy

Tabular Q-learning and SARSA policies that choose the next teaching action. They plug into the planner through the same `TeachingPolicy` interface as the rule-based policies, and are trained and evaluated on the classroom simulator and student-model engine.

## Formulation

We treat this as a finite-horizon MDP rather than a bandit, because actions change what happens later: fixing a prerequisite, keeping engagement up, and using up class time all matter, so rewards are delayed.

| Item | Definition |
|---|---|
| State (1,458 states) | Five 3-level features from the student model's class summary (mastery, misconception prevalence, engagement, participation balance, share of students left behind), plus whether mastery is still uncertain (2 levels) and whether the class is behind, on track or ahead of schedule (3 levels). Built only from the model's estimates, never from the simulator's truth |
| Actions (9) | lecture, example, diagnostic quiz, peer discussion, exercise, revision, TA support (aimed at students holding a misconception, or the weakest), peer group (aimed at students below the mastery threshold), advance (move to the next concept) |
| Reward | Uses the simulator's true state. 10 x learning gain, 2 x misconceptions removed, 4 x engagement change, 2 x participation-balance change, minus 0.1 x class time; plus 1 per concept completed and minus 3 x the share of students who haven't mastered a concept being left behind; at the end, plus 10 x final mean mastery and minus 5 x the share left behind. Weights are in `env.RewardWeights` |
| Episode | One simulated course with a class-time budget (45 units by default; actions cost 0 to 1.5) |
| Learning | Epsilon-greedy with linearly decaying epsilon and a visit-count learning rate. Actions that make no sense (advancing before anything is taught) are masked |
| Safety | A state never seen in training falls back to the rule-based policy. This happened 0% of the time at budget 45 and 0.2% at budget 90 |
| Explanations | Each decision's rationale states the class state, the chosen action's Q-value and the runner-up's |

## Files

| File | Purpose |
|---|---|
| `actions.py`, `state.py` | Action space, who an action targets, time costs; state encoding |
| `env.py` | Training environment (simulator, engine and planner) and the reward |
| `policies.py` | `RLTeachingPolicy`, `AdvancingPlanner`, and the baselines `FixedLecturePlanPolicy`, `ThresholdRepeatPolicy` and `RandomPolicy` |
| `runner.py` | Runs any policy through the real planner path; all evaluation uses this |
| `training.py`, `evaluate.py`, `action_profile.py`, `ablation.py` | Train, compare, inspect behaviour, and test against simple fixed sequences |
| `tests/test_rl.py` | 22 tests |
| `results/` | The trained Q-tables, evaluation tables, plots, decision traces, ablation and action profile that the numbers below come from |

## Running it

From the project folder:

```bash
python -m pytest module5_rl
```

The trained Q-tables are already in `results/`, and `main.py --policy rl` uses them. You only need to train again if you change the environment or reward.

The commands below write to a separate folder, `module5_rl/my_run`, so they don't overwrite the committed results. To reproduce the tables, train on the six-concept A* catalog, which is what the evaluation uses:

```bash
python -m module5_rl.training --algo both --episodes 2500 --catalog sample --out module5_rl/my_run
python -m module5_rl.evaluate --episodes 20 --budgets 45 90 --out module5_rl/my_run
python -m module5_rl.action_profile --out module5_rl/my_run
python -m module5_rl.ablation --out module5_rl/my_run
```

Training takes roughly 15 to 25 minutes depending on the machine, evaluation 10 to 15, the action profile about a minute and the ablation about 4. Each script reads the Q-tables from the `--out` folder, so run them in this order with the same folder.

For a quick check that everything runs (about 2 minutes), add `--episodes 40` to the training command and `--episodes 2` to the other three, and use `--budgets 45` in evaluation. The numbers it produces are not meaningful.

Training with `--catalog kg` instead uses the 11-concept course that `main.py` runs; the evaluation scripts always use the six-concept one.

Using a policy directly:

```python
from module1_knowledge_graph.catalog import load_course_catalog
from module3_student_model import ClassroomSimulator, StudentModelEngine, make_classroom
from module5_rl.policies import AdvancingPlanner, RLTeachingPolicy

catalog = load_course_catalog()
sim = ClassroomSimulator(catalog, make_classroom(catalog, "mixed", 12, seed=0), seed=0)
engine = StudentModelEngine(catalog, sim.student_ids())

policy = RLTeachingPolicy.load("module5_rl/results/q_table_q_learning.npy",
                               "module5_rl/results/visits_q_learning.npy", time_budget=45.0)
planner = AdvancingPlanner(catalog, policy=policy)
decision = planner.plan_next(engine)      # an ActionDecision with a rationale
```

`AdvancingPlanner` is the stock planner plus the ability for a policy to move on early; the stock planner only advances on its own mastery criterion. The policy keeps its own clock of remaining class time, since the interface doesn't pass one; call `policy.set_time_left(t)` if you know the real value.

## Results

Evaluation classes of 24 students on unseen seeds (training used 12). 20 episodes in each of 6 scenarios gives 120 paired episodes per policy. "Known" is the final share of student-concept pairs truly mastered, "left behind" is the share of students who mastered under 40% of the syllabus, and "coverage" is the share of concepts completed.

| Policy | Known (45) | Left behind (45) | Coverage (45) | Known (90) | Left behind (90) | Coverage (90) |
|---|---|---|---|---|---|---|
| Q-learning | 0.687 | 0.161 | 0.81 | 0.754 | 0.117 | 1.00 |
| SARSA | 0.656 | 0.157 | 0.65 | 0.726 | 0.124 | 0.99 |
| Rule-based heuristic | 0.517 | 0.359 | 0.36 | 0.616 | 0.235 | 0.67 |
| Fixed lecture plan | 0.557 | 0.332 | 1.00 | 0.557 | 0.332 | 1.00 |
| Threshold-repeat | 0.520 | 0.383 | 1.00 | 0.520 | 0.383 | 1.00 |
| Random | 0.595 | 0.273 | 0.81 | 0.614 | 0.254 | 1.00 |

The number in brackets is the time budget. Q-learning beats the fixed plan on knowledge by 0.130 (budget 45) and 0.197 (budget 90), with a 95% interval of about 0.010. It was trained only at budget 45 and still does well at 90 and with the larger class.

## Limitations

1. **A trivial policy matches the learned one.** Giving seven worked examples per concept and then moving on, with no adaptation, scores known 0.717, left behind 0.14 and full coverage, against 0.688, 0.16 and 0.78 coverage for Q-learning (`results/ablation.txt`; Q-learning is lower by 0.030 +/- 0.011). Worked examples are the strongest single action in our simulator, and the standard baselines don't exploit that. So the table shows the learned policy beats the given baselines, not that adaptive teaching beats the best fixed strategy.
2. **Behaviour barely varies between scenarios.** Per `results/action_profile.csv`, Q-learning gives an example 59 to 77% of the time in every scenario. SARSA uses roughly 40 to 55% examples and about 40% peer discussion. Targeted actions are only 5 to 16% of moves. The Q-value gaps behind many choices are small (see `decision_trace_*.txt`).
3. **It under-covers the syllabus at 45 units** (0.81 for Q-learning, 0.65 for SARSA). It spends time per concept like the examples-only policy but doesn't learn to move on quickly enough.
4. **Participation balance isn't improved.** The rule-based heuristic scores higher (0.74 against 0.70), most clearly in the dominant-talkers scenario.
5. The reward weights and the simulator are our design choices, and the results are only as realistic as the simulator.
6. SARSA's saved checkpoint is from episode 2,250 of the 2,500 planned, by which point epsilon had already reached its floor. Each algorithm was trained with one seed, so the gap between SARSA and Q-learning (+0.031 +/- 0.012 in Q-learning's favour) shouldn't be over-interpreted.
