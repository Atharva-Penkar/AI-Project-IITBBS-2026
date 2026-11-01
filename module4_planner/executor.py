"""Execute planner decisions against Team 3's simulator and student model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.schema import ActionType, Event

from module4_planner.models import ActionDecision


@dataclass(frozen=True)
class ExecutionResult:
    """Record what was executed and which events were returned by the simulator."""

    decision: ActionDecision
    events: tuple[Event, ...]
    simulator_step: int
    targeted_student_count: int

    @property
    def event_count(self) -> int:
        return len(self.events)


class AdaptiveExecutor:
    """Bridge a decision to ``ClassroomSimulator.step`` and update the engine.

    This layer is intentionally thin. It does not modify Team 3's simulator or
    infer student state itself; it forwards events to ``engine.update_many``.
    """

    def execute(
        self,
        decision: ActionDecision,
        simulator: Any,
        engine: Any,
        planner: Any | None = None,
    ) -> ExecutionResult:
        """Execute one decision and feed every returned event into the model.

        ``NEXT_CONCEPT`` is a documented no-op in Team 3's simulator. The
        planner uses it to represent course progression, so an empty event list
        for that action is valid and is not treated as an execution failure.
        """
        # Fail fast when a planner has a stale concept or target list.
        simulator.catalog.concept(decision.concept_id)
        roster = set(simulator.student_ids())
        unknown_targets = set(decision.target_student_ids) - roster
        if unknown_targets:
            raise ValueError(f"Decision targets unknown student(s): {sorted(unknown_targets)}")

        # An empty tuple means all students, per ActionDecision's contract.
        if not decision.target_student_ids or set(decision.target_student_ids) == roster:
            target_ids = None
            targeted_count = len(roster)
        else:
            target_ids = list(decision.target_student_ids)
            targeted_count = len(target_ids)

        events = simulator.step(
            action=decision.action,
            concept_id=decision.concept_id,
            target_ids=target_ids,
            target_misconception=decision.target_misconception,
        )
        engine.update_many(events)

        if planner is not None:
            planner.record_execution(decision)

        return ExecutionResult(
            decision=decision,
            events=tuple(events),
            simulator_step=simulator.step_count,
            targeted_student_count=targeted_count,
        )
