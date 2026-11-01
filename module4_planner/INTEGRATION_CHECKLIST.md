# Team 4 integration checklist

This checklist reflects the files/interfaces visible in the current repository snapshot. The official Team 4 assignment brief and implementations for Teams 1, 2, and Team 5 Person A were not present in that snapshot, so those integrations cannot be implemented against verified interfaces yet.

## Current integration boundaries

- [x] **Team 3 student model:** reads `StudentModelEngine.student_ids()`, `get_student()`, and `get_class_summary()`; sends simulator events to `update_many()`.
- [x] **Team 3 simulator:** uses `ClassroomSimulator.step(action, concept_id, target_ids, target_misconception)` and treats `NEXT_CONCEPT` returning no events as expected behavior.
- [x] **Shared schema:** decisions use shared `ActionType`; NLU events use `Event`, `EventType`, and `Intent` from `shared.schema`.
- [x] **Team 5 Person B NLU:** `nlu_adapter.py` maps the visible NLU taxonomy labels into supported shared-schema IDs and declines to fabricate an event when no concept maps.
- [x] **Test/evaluation contract:** fixed and adaptive modes can be run on the same scenario/seed/roster/step budget; raw per-episode metrics are exported as CSV.
- [x] **Extensible policy hook:** any rule-based or learned class that implements `TeachingPolicy.choose_action(...)` can replace the default heuristic. No RL algorithm is duplicated in Team 4.

## Pending once upstream team modules/contracts are available

- [ ] **Team 1 knowledge graph:** obtain the graph's canonical concept IDs, prerequisite API, and misconception IDs. Build a small adapter that creates/validates `CourseCatalog`; do not create a second graph inside this module.
- [ ] **Team 2 classroom scheduling/CSP:** obtain its action-feasibility and time/capacity contract. Add a planner constraint hook that rejects infeasible actions before execution and records the selected schedule slot. No such API was visible, so this package does not invent one.
- [ ] **Team 5 Person A reinforcement learning:** if their policy is in scope, adapt its state/action/reward interface to `TeachingPolicy`; keep course prerequisite enforcement and execution safety in `AdaptivePlanner`/`AdaptiveExecutor`.
- [ ] **Official rubric:** compare required metrics, artefacts, and interaction protocol against the actual assignment brief before final submission.

## Before merging into the main repository

1. Copy the `module4_planner/` directory into the repository root.
2. Run `python -m pytest module4_planner/tests -v` from the root.
3. Run the demo and evaluation commands in `README.md`.
4. Review action traces and evaluation results; change thresholds only with recorded rationale.
5. Confirm the final shared identifiers against Team 1's graph and Team 5's final taxonomy.
