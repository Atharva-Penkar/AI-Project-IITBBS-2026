# Adaptive teaching planner and execution loop

This module decides what to teach next and runs the lesson. It looks at the student model's estimates, chooses a teaching action, has the simulator carry it out, feeds the resulting events back into the student model, and repeats. When a quiz reveals a misconception or a prerequisite turns out to be weak, the plan changes.

## Teaching actions as a planning problem

`action_model.py` describes each action STRIPS-style, with preconditions, expected effects and a duration. The state of the class on the current concept is a set of facts computed from the student model: `introduced`, `evidence` (a quiz has been taken and uncertainty is low), `misconception`, `low_mastery`, `low_engagement`, `uneven_participation`, `prereq_gap`, `discussed`, `practised`, `assessed`.

| Action | Needs | Effect |
|---|---|---|
| Lecture | not yet introduced, no prerequisite gap | introduces the concept |
| Diagnostic quiz | introduced | produces evidence |
| TA intervention | evidence, misconception | removes the misconception |
| Example | evidence, low mastery, no misconception | raises mastery |
| Peer discussion | evidence, no misconception or low mastery | raises engagement, evens out participation |
| Exercise | discussed, no misconception or low mastery | practised |
| Assessment | practised, no blockers | confirms mastery |
| Revision | prerequisite gap | clears the gap |

The goal is `introduced`, `evidence`, `discussed`, `practised` and `assessed` with no blockers left. `plan_concept` finds the cheapest action sequence that gets there using uniform-cost search.

`plan_policy.py` wraps this as `PlanningTeachingPolicy`, the default in `main.py`. It re-observes the class and replans at every step, so a fresh class gets lecture, quiz, discussion, exercise, assessment, while a class where the quiz exposed a misconception gets a TA intervention first. When the plan changes, the decision's rationale says what changed.

## The pieces

| File | Purpose |
|---|---|
| `action_model.py` | The actions above, fact observation and the planner |
| `plan_policy.py` | The replanning policy |
| `policy.py` | `HeuristicTeachingPolicy`, a simpler rule-based alternative (below) |
| `planner.py` | `AdaptivePlanner`: moves through the syllabus in prerequisite order, delegates each choice to a policy, and marks a concept complete |
| `executor.py` | `AdaptiveExecutor`: runs a decision on the simulator, passes the events to the student model, optionally calls the group scheduler and the language-analysis hook |
| `nlu_adapter.py` | Translates the language-analysis output into the shared `Event` format |
| `models.py` | `ActionDecision` and `PlannerConfig` (all thresholds) |
| `evaluation.py` | Compares the adaptive policy with a fixed lesson on identical seeded classes |
| `demo.py` | Prints a lesson step by step |

A concept counts as complete when the class's mean mastery reaches 0.72 and no more than 20% of students are below the threshold, or after 15 actions on it, whichever comes first. A concept is not started until its prerequisites are complete. Every `ActionDecision` carries a plain-language `rationale`, and an empty target list means the whole class while a non-empty one means a subset.

### The rule-based policy

`HeuristicTeachingPolicy` checks these in order and takes the first that applies: intervene on a common misconception; correct students who are confidently wrong; give a short diagnostic quiz if evidence is thin; target a small group below the mastery threshold; run a discussion if participation is lopsided or engagement is low; escalate to a TA if progress stalls; otherwise use examples or revision for low mastery, then exercises.

## Running it

From the project folder:

```bash
python -m module4_planner.demo --students 12 --steps 40 --seed 7
python -m pytest module4_planner
```

To compare the heuristic policy with a fixed lesson cycle on identical classes:

```bash
python -m module4_planner.evaluation --scenarios mixed,widespread_misconception,small_group_behind --seeds 0,1,2 --students 24 --max-steps 60 --output my_planner_results.csv
```

This prints averages and writes per-run numbers to the CSV you name. The fixed lesson cycles lecture, example, exercise and revision in order without looking at the class state. Both modes get the same step budget.

For the full system including the replanning policy, use `python main.py` (see the main README).

## Language analysis adapter

The NLU module and the student model use different labels. NLU says `concept_confusion`, `astar_search` and `ASTAR_G_VS_H`; the student model expects `Intent.CONFUSION`, `Astar` and `g_is_remaining_cost`. `nlu_result_to_event` maps one to the other and returns `None` when no compatible concept can be determined. Pass `fallback_concept_id` only when the lesson itself tells you which concept is being discussed. A missing confidence becomes the neutral value 0.5.

## Limitations

- All thresholds in `PlannerConfig` and `FactThresholds` are starting values chosen by us, not tuned against real classrooms.
- Expected effects are descriptions used for planning; the real effect on each student comes from the simulator.
- The `NEXT_CONCEPT` action produces no events. That is intended.
- Results are from simulation. Compare policies over several seeds and look at the traces before drawing conclusions.
