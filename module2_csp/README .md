# Team 2 — Constraint Satisfaction and Classroom Scheduling

**Project:** Agentic Classroom Digital Twin (Group 8, Artificial Intelligence)
**Module:** 1 · Part B — Constraint Satisfaction Problem (CSP)
**Package:** `module2_csp`

## 1. Objective

Automatically organise students, teaching assistants (TAs), groups, activities and time slots while respecting classroom constraints. If no valid arrangement exists, the scheduler says **which constraints caused the infeasibility**.

**Deliverable:** a working classroom scheduler that produces

- valid student groups,
- discussion / activity allocations,
- TA assignments and time slots,
- a human-readable explanation when the classroom cannot be scheduled.

## 2. Files

| File | Purpose |
|---|---|
| `classroom_csp.py` | Problem model: data classes, constraint builder, global propagators, infeasibility diagnosis, schedule validator, result formatter, demo-classroom generator. |
| `csp_solver.py` | Generic, problem-independent CSP solver (backtracking + MRV/degree + LCV + forward checking + AC-3/MAC + pluggable global propagators). |
| `adapter.py` | `GroupScheduler`: builds classroom CSPs from live student-model estimates (Module 3 output) and solves them. This is the integration point for the rest of the project. |
| `demo.py` | One valid arrangement and three infeasible ones, each with an explanation. |
| `test_csp.py` | pytest suite (class sizes, infeasibility explanations, live student-model integration). |

## 3. Problem formulation

### 3.1 Inputs

| Class | Fields |
|---|---|
| `Student` | `sid`, `ability` ∈ [0,1], `mastery` {concept → P(mastery)}, `available_slots` |
| `TeachingAssistant` | `tid`, `available_slots`, `max_groups` (workload cap), `expertise` (concepts; empty = any) |
| `Activity` | `aid`, `name`, `kind` (revision / exercise / discussion / peer / ta), `concept` |
| `ClassroomConfig` | students, TAs, activities, time slots, concept prerequisite graph, and the tunable parameters in §3.4 |

### 3.2 Variables

For a fixed number of groups `k`:

| Variable | Domain | Meaning |
|---|---|---|
| `grp[sid]` (one per student) | `{0..k-1}` | which group the student belongs to |
| `slot[Gi]` | time slots at which some TA is free | when group *i* meets |
| `act[Gi]` | activities some TA can cover | what group *i* works on |
| `ta[Gi]` | TAs | who supervises group *i* |

The solver tries `k` from the smallest possible value (`ceil(n / max_group_size)`) up to `n // min_group_size`, and returns the **smallest feasible `k`**. A symmetry-breaking constraint (canonical group labelling) avoids searching equivalent relabellings.

### 3.3 Constraints

Every constraint belongs to a named **category**, which is what the explanation engine reports.

| Category | Rule |
|---|---|
| `group_size` | each group has `min_group_size`…`max_group_size` students |
| `ability_mix` | weak students are at most `max_weak_ratio` of a group, and every group with a weak student has at least one strong peer |
| `prerequisite` | every member must have mastery ≥ `mastery_threshold` on all (transitive) prerequisites of the group's activity |
| `activity_need` | the group's mean mastery of the activity's concept must be ≤ `need_ceiling` (do not teach what they already know) |
| `student_availability` | the group's slot must be in each member's `available_slots` |
| `ta_availability` | the TA must be free in the group's slot |
| `ta_expertise` | the TA must cover the activity's concept (if the TA lists expertise) |
| `ta_conflict` | one TA cannot supervise two groups in the same slot |
| `ta_workload` | a TA supervises at most `max_groups` groups; optional `ta_balance_tolerance` bounds the load spread between TAs |
| `room_capacity` | at most `max_parallel_groups` groups in one slot |

Weak / strong are defined by `weak_threshold` and `strong_threshold` on a student's `ability`.

### 3.4 Tunable parameters (`ClassroomConfig`)

`min_group_size=3`, `max_group_size=5`, `max_parallel_groups=2`, `mastery_threshold=0.5`, `weak_threshold=0.4`, `strong_threshold=0.7`, `max_weak_ratio=0.5`, `need_ceiling=0.9`, `ta_balance_tolerance=None`.

## 4. Solving approach

`CSPSolver` (in `csp_solver.py`) is a complete backtracking search with:

- **Variable ordering:** MRV (minimum remaining values), ties broken by degree (`mrv_degree`).
- **Value ordering:** LCV (least constraining value).
- **Propagation:** forward checking plus **MAC** (AC-3 after each assignment) on binary constraints.
- **Global propagators** (in `classroom_propagators`) for group size, ability mix, activity need, TA workload, room capacity and TA conflict. These prune the many n-ary constraints that arc consistency cannot see.
- **Limits:** `max_nodes` and `time_limit` (applied per group count). Hitting a limit yields status `UNKNOWN`, never a false "infeasible".

Every schedule the solver returns is re-checked by `validate_schedule`, an independent checker that does not use the solver's data structures.

## 5. Explaining infeasibility

When no arrangement exists, `solve_classroom` produces explanations in two stages for each group count tried:

1. **Static diagnosis** (`static_diagnosis`): cheap necessary conditions. Any message is a proof of infeasibility. Examples: total TA capacity is below `k`; a student is never free when a TA is; a student satisfies the prerequisites of no activity; there are weak students but no strong peers.
2. **Conflict-core search** (`_conflict_core`): if the static checks pass but search still fails, the solver deletes constraint categories one at a time and re-solves. It reports the set of categories that is still infeasible on its own and becomes satisfiable if any one of them is removed. If a relaxation hits the limits, the output says the set may not be minimal.

Result status is one of `SOLVED`, `INFEASIBLE` (proven), `UNKNOWN` (limits hit), or `INVALID_INPUT` (config validation failed; the problems are listed).

## 6. Usage

### 6.1 Run the demo

```bash
python -m module2_csp.demo
```

Output (abridged):

```
=== Demo 1: normal classroom ===
Valid arrangement with 4 groups (32 search nodes)
G0: slot=Tue-10  TA=TA3  activity=Arrays refresher [revision, arrays]
     students: S01(M), S02(M), S03(M), S06(W), S07(S)  mean ability=0.49
...
=== Demo 2: overloaded TAs (each TA may supervise only 1 group) ===
NO VALID CLASSROOM ARRANGEMENT FOUND
Status: INFEASIBLE (proven)
--- 4 groups: infeasible ---
[ta_workload] 4 groups need a TA each but total TA capacity is 3 groups.
```

Tags after student IDs: **W** weak, **M** medium, **S** strong.

### 6.2 Standalone use

```python
from module2_csp.classroom_csp import (
    make_demo_classroom, solve_classroom, validate_schedule, format_result)

cfg = make_demo_classroom(n_students=24, seed=1)
cfg.max_parallel_groups = 3

res = solve_classroom(cfg, max_nodes=200_000, time_limit=5.0)
print(format_result(cfg, res))

assert res.feasible
assert validate_schedule(cfg, res.solution) == []   # [] means no violations
```

`res.solution` maps group name to `{"slot", "activity", "ta", "students"}`.

### 6.3 Integrated with the student model (`adapter.py`)

```python
from module2_csp.adapter import GroupScheduler

scheduler = GroupScheduler(catalog, student_ids, seed=0)
plan = scheduler.schedule(engine, "path_cost", student_ids, kind="peer")

for g in plan.groups:
    print(g.name, g.student_ids, g.ta, g.slot, g.concept_id, g.kind)
print(plan.unplaced, plan.explanation)
```

- **Ability** of each student is the mean `p_mastery` over the target concept and all its transitive prerequisites, taken from `StudentModelEngine`.
- **`kind="peer"`:** students who meet the concept's prerequisites are split into mixed-ability peer groups. The rest go to TA-led revision groups for their *first missing prerequisite* (earliest in teaching order).
- **`kind="ta"`:** every student goes into a small TA-led group (1–4 students), on the concept or on their first missing prerequisite.
- Groups scheduled earlier in one call consume TA time and rooms, so later groups see only the remaining capacity.
- Timetables and TAs are not part of the student model, so by default they are generated from `seed` (about 85% slot availability per student, three TAs). Pass `slots`, `availability` and `tas` to use real data.
- `GroupPlan` fields: `feasible`, `groups`, `unplaced`, `explanation`, `text`.

## 7. Independent testing

Run the automated tests (requires `pytest` and the other modules on the path for the adapter test):

```bash
pytest module2_csp/test_csp.py -v
```

| Test | What it checks |
|---|---|
| `test_valid_arrangement_for_different_class_sizes` | 9, 16 and 24 students: feasible, passes `validate_schedule`, every student placed exactly once |
| `test_too_few_ta_slots_is_infeasible_and_explained` | TA workload cap of 1 gives `INFEASIBLE` with a `[ta_workload]` explanation |
| `test_student_never_free_when_a_ta_is_is_explained` | a student free only when no TA is available gives a `[student_availability]` explanation naming that student |
| `test_all_weak_class_cannot_be_mixed` | all-weak class with a single strong student gives an `[ability_mix]` explanation |
| `test_scheduler_uses_live_student_model_estimates` | the adapter schedules a simulated 18-student class from live estimates and places every student |

### 7.1 Manual experiments (the "try different…" scenarios)

Modify a config from `make_demo_classroom` and call `solve_classroom`:

| Scenario | How | Expected |
|---|---|---|
| Different class sizes | `make_demo_classroom(n_students=N)` | smallest feasible `k` found; larger classes need more groups, TAs and slots |
| Fewer / more TAs | change the `cfg.tas` list or `t.max_groups` | too little capacity gives `[ta_workload]` / `[ta_conflict]` |
| Ability distributions | set `s.ability` (e.g. all 0.2, or all 0.9) | all-weak gives `[ability_mix]`; all-strong is fine unless `need_ceiling` blocks it |
| Conflicting time slots | restrict `s.available_slots` or `t.available_slots` | `[student_availability]` / `[ta_availability]` messages |
| Missing prerequisites | lower `s.mastery`, or remove a revision activity | `[prerequisite]` messages (see Demo 3) |
| Rooms | lower `cfg.max_parallel_groups` | `[room_capacity]` messages |
| Strict TA fairness | `cfg.ta_balance_tolerance = 0` | only group counts divisible by the number of TAs work (see Demo 4) |

**Observed in a quick scaling check** (`seed=1`, `max_parallel_groups=3`, demo TAs):

| Students | Status | Groups | Search nodes |
|---|---|---|---|
| 9 | SOLVED | 2 | 16 |
| 16 | SOLVED | 4 | 28 |
| 24 | SOLVED | 5 | 67 |
| 40 | INFEASIBLE | — | 0 (rejected statically) |

The 40-student case is infeasible by design: the default demo TAs can supervise only 6 groups, and 40 students need at least 8 groups of ≤5. Adding TAs or raising `max_groups` makes it solvable.

## 8. Limitations and notes

- The number of groups `k` is searched incrementally, so very large classes with tight limits may return `UNKNOWN`; raise `max_nodes` / `time_limit`.
- Each group has exactly one activity and one TA in one slot; one session per student per call.
- `GroupScheduler` defaults (`max_nodes=5000`, `time_limit=2.0`) are deliberately small for interactive use.
- Conflict-core explanations name constraint **categories**, not individual students, apart from the static diagnosis which does name students.
- In `adapter.py`, `_with_remaining_capacity` approximates the parallel-room limit across slots by using the tightest remaining slot.
