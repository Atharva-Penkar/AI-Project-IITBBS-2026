# Team 4: Adaptive Teaching Planner and Execution

## Purpose

`module4_planner` chooses the next teaching action from the current probabilistic
class model, respects concept prerequisites, executes the action through Team
3's `ClassroomSimulator`, and feeds every returned event back to
`StudentModelEngine`. The package starts with a transparent deterministic
heuristic policy. It exposes a clear policy interface so a future policy can be
plugged in without duplicating the student model or simulator.

The repository's visible Team 4 assignment brief was not present when this
module was prepared. The implementation therefore follows the shared schema
and Team 3/Team 5 contracts found in the repository, rather than claiming to
cover unstated grading requirements.

## Files

- `models.py`: the immutable `ActionDecision` record and validated policy/progression thresholds.
- `policy.py`: ordered rules for diagnosis, misconception-specific support, confidently-wrong learners, small-group support, participation balance, engagement, and mastery consolidation.
- `planner.py`: prerequisite validation, topological progression, mastery-based concept completion, and action-history tracking.
- `executor.py`: calls the simulator and sends returned events to the student model.
- `nlu_adapter.py`: maps Team 5 NLU labels and concept/misconception IDs into Team 3's shared `Event` schema.
- `evaluation.py`: compares the adaptive policy with a fixed lesson baseline on identical seeded scenarios and writes raw metrics to CSV.
- `demo.py`: end-to-end trace of decisions and class summaries.
- `tests/`: unit tests plus integration tests that use the project's existing Team 3 simulator/model.

## Architecture

```text
StudentModelEngine class/student estimates
                |
                v
        AdaptivePlanner
   - prerequisite guard
   - concept completion check
                |
                v
     HeuristicTeachingPolicy
   - choose action/concept target
                |
                v
        AdaptiveExecutor
                |
                v
     ClassroomSimulator.step(...)
                |
             events
                |
                v
      StudentModelEngine.update_many(events)
                |
          next planning cycle
```

The `NEXT_CONCEPT` action is an explicit transition marker. Team 3's simulator
returns no events for it; this is expected, not an error. The planner begins
with concepts whose prerequisites are empty and does not unlock a dependent
concept until each declared prerequisite is marked complete by the configured
mastery criterion.

## Policy rules

The baseline considers these needs in priority order:

1. Intervene on a misconception whose prevalence is high enough in the current concept estimates.
2. Give corrective support to learners who are confidently wrong and have low estimated mastery.
3. Use a bounded diagnostic quiz when evidence is sparse or uncertainty is high.
4. Target peer learning or a TA intervention to a small group below the mastery threshold.
5. Use structured peer discussion when participation is unequal or class engagement is low.
6. Escalate stalled progress to a TA intervention after repeated actions.
7. Use worked examples or revision for low mastery, then exercises to consolidate.

The first matching applicable rule wins. `ActionDecision.rationale` and
`metadata["rule"]` record why the action was selected. An empty target list
means the entire class; a populated target list means a focused action.

## Integration requirements

Run commands from the project root, where these existing modules are available:

- `shared.schema` (`ActionType`, `Event`, `CourseCatalog`, estimates and summaries)
- `module3_student_model` (`StudentModelEngine`, `ClassroomSimulator`, sample catalog)
- Team 5's NLU pipeline, when using `nlu_adapter.py`

No new third-party dependencies are introduced. The project already requires
NumPy and Pydantic. The module does not implement BKT, misconception tracking,
NLU classification, knowledge-graph search, classroom scheduling, or RL.

### Why the NLU adapter exists

Team 5 uses labels such as `concept_confusion`, `astar_search`, and
`ASTAR_G_VS_H`; Team 3 expects `Intent.CONFUSION`, `astar`, and
`confuses_g_and_h`. `nlu_adapter.nlu_result_to_event(...)` performs explicit
mapping. It returns `None` if the input is invalid or no compatible concept
can be determined. Supply `fallback_concept_id` only when the surrounding
classroom context independently identifies the current concept. A missing NLU
confidence is assigned the schema's neutral value of 0.5, not a fabricated
classifier confidence score.

## Run the demo

```bash
python -m module4_planner.demo --students 12 --steps 40 --seed 7
```

The output shows action, concept, target group, generated event count, and the
policy rationale. The seed makes the simulation reproducible in the same
runtime/dependency versions.

## Evaluate against a fixed lesson

```bash
python -m module4_planner.evaluation \
  --scenarios mixed,widespread_misconception,small_group_behind \
  --seeds 0,1,2 \
  --students 24 \
  --max-steps 60 \
  --output module4_planner/evaluation_results.csv
```

The harness holds scenario, seed, roster size, and simulator-step budget
constant for both modes. The fixed baseline cycles through lecture, example,
exercise, and revision in catalog order, without adapting to model state. The
CSV keeps per-episode metrics; the CLI also prints aggregate metrics. In
particular, compare mastery, the fraction of student-concept estimates below
0.4, participation Gini, action count, concept transitions, and course
completion. Because an adaptive transition uses a simulator step without a
teaching event, transition overhead is reported separately.

This is simulation-based evaluation, not real classroom validation. Run enough
seeds and held-out scenarios, inspect action traces, and report variability
before making any claim that one policy outperforms another.

## Tests

From the repository root:

```bash
python -m pytest module4_planner/tests -v
```

The integration tests require the repository's Team 3 module and existing
project dependencies. Unit tests exercise the policy, graph validation,
execution semantics, NLU mapping, and safe handling of no-op concept transitions.

## Resetting state

Call `planner.reset()` to clear concept progression and action history. Reset
the simulator with the same or a new seed and create/reset the student model
separately; each object owns its own state.
