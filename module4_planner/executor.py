"""Execute planner decisions against Team 3's simulator and student model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
    group_plan: Any = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def event_count(self) -> int:
        return len(self.events)


GROUP_ACTIONS = {ActionType.PEER_DISCUSSION: "peer", ActionType.PEER_LEARNING: "peer", ActionType.TA_INTERVENTION: "ta"}


class AdaptiveExecutor:
    """Bridge a decision to ``ClassroomSimulator.step`` and update the engine.

    Optional hooks:
    - ``group_scheduler``: an object with ``schedule(engine, concept_id, student_ids, kind)``
      returning a plan with ``groups`` (each with ``student_ids``, ``concept_id``, ``kind``)
      and ``unplaced``. Used for peer and TA actions, so groups, TAs and slots come
      from the classroom CSP.
    - ``event_hook``: called on each event before the engine sees it; returns the
      event to use (for example, one re-labelled by the NLU pipeline).
    """

    def __init__(
        self,
        group_scheduler: Any | None = None,
        event_hook: Callable[[Event], Event] | None = None,
    ) -> None:
        self.group_scheduler = group_scheduler
        self.event_hook = event_hook

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

        group_plan = None
        notes: list[str] = []
        kind = GROUP_ACTIONS.get(decision.action)
        if self.group_scheduler is not None and kind is not None:
            ids = target_ids or sorted(roster)
            group_plan = self.group_scheduler.schedule(engine, decision.concept_id, ids, kind)
            events = []
            for group in group_plan.groups:
                action = ActionType.REVISION if group.kind == "revision" else decision.action
                events += simulator.step(
                    action=action,
                    concept_id=group.concept_id,
                    target_ids=list(group.student_ids),
                    target_misconception=decision.target_misconception,
                )
            if group_plan.unplaced:
                notes.append(f"No valid group for {len(group_plan.unplaced)} student(s); taught them together instead.")
                events += simulator.step(
                    action=decision.action,
                    concept_id=decision.concept_id,
                    target_ids=list(group_plan.unplaced),
                    target_misconception=decision.target_misconception,
                )
        else:
            events = simulator.step(
                action=decision.action,
                concept_id=decision.concept_id,
                target_ids=target_ids,
                target_misconception=decision.target_misconception,
            )

        if self.event_hook is not None:
            events = [self.event_hook(e) for e in events]
        engine.update_many(events)

        if planner is not None:
            planner.record_execution(decision)

        return ExecutionResult(
            decision=decision,
            events=tuple(events),
            simulator_step=simulator.step_count,
            targeted_student_count=targeted_count,
            group_plan=group_plan,
            notes=tuple(notes),
        )
