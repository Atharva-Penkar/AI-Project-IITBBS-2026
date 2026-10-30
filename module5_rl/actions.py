"""RL action space and the translation of an RL action into an ``ActionDecision``.

The learner chooses *what kind* of teaching move to make. Who the move is aimed
at (the whole class, the students holding a misconception, the quiet students...)
is decided by ``select_targets`` from the current student estimates. This keeps
the Q-table small while still producing targeted, auditable decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum

import numpy as np

from module4_planner.models import ActionDecision, PlannerConfig
from shared.schema import ActionType, GroupSummary, StudentEstimate


class RLAction(IntEnum):
    LECTURE = 0
    EXAMPLE = 1
    DIAGNOSTIC_QUIZ = 2
    PEER_DISCUSSION = 3
    EXERCISE = 4
    REVISION = 5
    TA_SUPPORT = 6  # targeted: students holding a misconception / weakest students
    PEER_GROUP = 7  # targeted: students below the mastery threshold
    ADVANCE = 8  # ask the planner to move to the next concept


N_ACTIONS = len(RLAction)

ACTION_TYPE: dict[RLAction, ActionType] = {
    RLAction.LECTURE: ActionType.LECTURE,
    RLAction.EXAMPLE: ActionType.EXAMPLE,
    RLAction.DIAGNOSTIC_QUIZ: ActionType.DIAGNOSTIC_QUIZ,
    RLAction.PEER_DISCUSSION: ActionType.PEER_DISCUSSION,
    RLAction.EXERCISE: ActionType.EXERCISE,
    RLAction.REVISION: ActionType.REVISION,
    RLAction.TA_SUPPORT: ActionType.TA_INTERVENTION,
    RLAction.PEER_GROUP: ActionType.PEER_LEARNING,
    RLAction.ADVANCE: ActionType.NEXT_CONCEPT,
}

# Class-time units. Moving on is free; a quiz is short; discussions and exercises are long.
ACTION_TIME_COST: dict[ActionType, float] = {
    ActionType.LECTURE: 1.0,
    ActionType.EXAMPLE: 1.0,
    ActionType.DIAGNOSTIC_QUIZ: 0.5,
    ActionType.ASSESSMENT: 0.5,
    ActionType.PEER_DISCUSSION: 1.5,
    ActionType.PEER_LEARNING: 1.5,
    ActionType.EXERCISE: 1.5,
    ActionType.REVISION: 1.0,
    ActionType.TA_INTERVENTION: 1.0,
    ActionType.NEXT_CONCEPT: 0.0,
}


def time_cost(action: ActionType) -> float:
    return ACTION_TIME_COST[action]


def valid_actions_mask(action_counts: Mapping[str, int] | None) -> np.ndarray:
    """ADVANCE is illegal before anything has been taught or tested on the concept."""
    mask = np.ones(N_ACTIONS, dtype=bool)
    if not action_counts or sum(action_counts.values()) == 0:
        mask[RLAction.ADVANCE] = False
    return mask


def _compact(ids: Sequence[str], all_ids: Sequence[str], config: PlannerConfig) -> tuple[str, ...]:
    """Whole class is represented as an empty tuple (the convention used by Team 4)."""
    allowed = set(all_ids)
    targets = tuple(sorted(set(ids) & allowed))
    if len(targets) < config.target_min_students or set(targets) == allowed:
        return ()
    return targets


def _bottom_quarter(students: Sequence[StudentEstimate], concept_id: str) -> list[str]:
    ranked = sorted(students, key=lambda s: (s.concepts[concept_id].p_mastery, s.student_id))
    return [s.student_id for s in ranked[: max(1, len(students) // 4)]]


def select_targets(
    action: RLAction,
    summary: GroupSummary,
    students: Sequence[StudentEstimate],
    concept_id: str,
    config: PlannerConfig,
) -> tuple[tuple[str, ...], str | None]:
    """Return ``(target_student_ids, target_misconception)`` for an RL action."""
    all_ids = [s.student_id for s in students]
    cs = summary.concepts[concept_id]

    if action == RLAction.DIAGNOSTIC_QUIZ:
        uncertain = [
            s.student_id
            for s in students
            if s.concepts[concept_id].n_observations == 0
            or s.concepts[concept_id].uncertainty >= config.uncertainty_threshold
        ]
        return _compact(uncertain, all_ids, config), None

    if action == RLAction.PEER_DISCUSSION:
        if summary.participation_gini >= config.high_participation_gini:
            quiet = [s.student_id for s in students if s.participation < config.low_participation_threshold]
            return _compact(quiet, all_ids, config), None
        return (), None

    if action == RLAction.TA_SUPPORT:
        for m, prevalence in sorted(cs.misconception_prevalence.items(), key=lambda kv: (-kv[1], kv[0])):
            if prevalence < config.misconception_prevalence_threshold:
                continue
            affected = [
                s.student_id
                for s in students
                if s.concepts[concept_id].misconceptions.get(m, 0.0) >= config.individual_misconception_threshold
            ]
            if affected:
                return _compact(affected, all_ids, config), m
        wrong = [
            s.student_id
            for s in students
            if s.confidently_wrong and s.concepts[concept_id].p_mastery < config.low_mastery_threshold
        ]
        if wrong and len(wrong) < len(all_ids):
            return _compact(wrong, all_ids, config), None
        low = [s.student_id for s in students if s.concepts[concept_id].p_mastery < config.low_mastery_threshold]
        if low and len(low) < len(all_ids):
            return _compact(low, all_ids, config), None
        return _compact(_bottom_quarter(students, concept_id), all_ids, config), None

    if action == RLAction.PEER_GROUP:
        low = [s.student_id for s in students if s.concepts[concept_id].p_mastery < config.low_mastery_threshold]
        if not low or len(low) == len(all_ids):
            low = _bottom_quarter(students, concept_id)
        return _compact(low, all_ids, config), None

    return (), None


def build_decision(
    action: RLAction,
    summary: GroupSummary,
    students: Sequence[StudentEstimate],
    concept_id: str,
    config: PlannerConfig,
    rationale: str = "",
    priority: float = 0.5,
    metadata: dict | None = None,
) -> ActionDecision:
    """Turn an RL action into the ``ActionDecision`` the planner/executor understand."""
    action = RLAction(action)
    meta = {"rule": "rl_policy", "rl_action": action.name, **(metadata or {})}
    if action == RLAction.ADVANCE:
        meta["rule"] = "policy_advance"
        return ActionDecision(
            action=ActionType.NEXT_CONCEPT,
            concept_id=concept_id,
            rationale=rationale or "Policy requests moving on to the next concept.",
            priority=priority,
            metadata=meta,
        )
    targets, misconception = select_targets(action, summary, students, concept_id, config)
    return ActionDecision(
        action=ACTION_TYPE[action],
        concept_id=concept_id,
        target_student_ids=targets,
        target_misconception=misconception,
        rationale=rationale,
        priority=priority,
        metadata=meta,
    )
