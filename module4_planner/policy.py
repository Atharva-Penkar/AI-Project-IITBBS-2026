"""Transparent baseline policy for selecting the next teaching action.

The policy is intentionally rule-based: it is deterministic, auditable, easy
for teammates to test, and provides a stable baseline before any learned policy
is plugged into the same decision interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from shared.schema import ActionType, GroupSummary, StudentEstimate

from module4_planner.models import ActionDecision, PlannerConfig


class TeachingPolicy(Protocol):
    """Interface a deterministic, learned, or externally supplied policy must satisfy."""

    config: PlannerConfig

    def choose_action(
        self,
        summary: GroupSummary,
        students: Sequence[StudentEstimate],
        concept_id: str,
        action_counts: Mapping[str, int] | None = None,
    ) -> ActionDecision: ...


class HeuristicTeachingPolicy:
    """Choose an action using mastery, uncertainty, misconceptions, and behaviour.

    Rule order matters. Specific, evidence-backed needs (a diagnosed
    misconception or confidently wrong learner) take precedence over generic
    whole-class instruction. Diagnostic quizzes are bounded to avoid repeatedly
    testing students without following up on what the results show.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def choose_action(
        self,
        summary: GroupSummary,
        students: Sequence[StudentEstimate],
        concept_id: str,
        action_counts: Mapping[str, int] | None = None,
    ) -> ActionDecision:
        """Return one action for ``concept_id`` based on the current model state."""
        if concept_id not in summary.concepts:
            raise KeyError(f"Concept {concept_id!r} is missing from the group summary")
        if not students:
            raise ValueError("The policy needs at least one student estimate")

        counts = action_counts or {}
        concept_summary = summary.concepts[concept_id]
        all_ids = tuple(s.student_id for s in students)
        total_actions = sum(counts.values())
        n_observations = [s.concepts[concept_id].n_observations for s in students]
        mean_observations = sum(n_observations) / len(n_observations)
        diagnostics = counts.get(ActionType.DIAGNOSTIC_QUIZ.value, 0)

        # Rule 1: address a misconception that appears consistently in the
        # estimates. Prevalence is a fraction of the class, not a probability.
        candidates = sorted(
            concept_summary.misconception_prevalence.items(),
            key=lambda pair: (-pair[1], pair[0]),
        )
        for misconception_id, prevalence in candidates:
            if prevalence < self.config.misconception_prevalence_threshold:
                continue
            affected = [
                s.student_id
                for s in students
                if s.concepts[concept_id].misconceptions.get(misconception_id, 0.0)
                >= self.config.individual_misconception_threshold
            ]
            targets = self._compact_targets(affected, all_ids)
            return ActionDecision(
                action=ActionType.TA_INTERVENTION,
                concept_id=concept_id,
                target_student_ids=targets,
                target_misconception=misconception_id,
                rationale=(
                    f"Target misconception {misconception_id!r}: model prevalence "
                    f"is {prevalence:.0%} for this concept."
                ),
                priority=min(1.0, 0.65 + prevalence),
                metadata={"rule": "prevalent_misconception", "prevalence": prevalence},
            )

        # Rule 2: confidently wrong students need corrective support rather than
        # simply being moved forward because they sound confident.
        confidently_wrong = [
            s.student_id
            for s in students
            if s.confidently_wrong
            and s.confidence >= 0.65
            and s.concepts[concept_id].p_mastery < self.config.low_mastery_threshold
        ]
        if confidently_wrong:
            targets = self._compact_targets(confidently_wrong, all_ids)
            return ActionDecision(
                action=ActionType.TA_INTERVENTION,
                concept_id=concept_id,
                target_student_ids=targets,
                rationale=(
                    f"Provide corrective feedback to {len(confidently_wrong)} "
                    "confident but low-mastery learner(s)."
                ),
                priority=0.88,
                metadata={"rule": "confidently_wrong", "target_count": len(confidently_wrong)},
            )

        # Rule 3: collect evidence when the model has little data or uncertainty
        # remains high. Cap quiz actions, then act on the available estimates.
        if diagnostics < self.config.max_diagnostic_actions_per_concept and (
            mean_observations < self.config.diagnostic_observation_target
            or concept_summary.mean_uncertainty >= self.config.uncertainty_threshold
        ):
            uncertain_students = [
                s.student_id
                for s in students
                if s.concepts[concept_id].n_observations == 0
                or s.concepts[concept_id].uncertainty >= self.config.uncertainty_threshold
            ]
            targets = self._compact_targets(uncertain_students, all_ids)
            reason = (
                f"Collect diagnostic evidence: mean observations={mean_observations:.2f}, "
                f"mean uncertainty={concept_summary.mean_uncertainty:.2f}."
            )
            return ActionDecision(
                action=ActionType.DIAGNOSTIC_QUIZ,
                concept_id=concept_id,
                target_student_ids=targets,
                rationale=reason,
                priority=0.78,
                metadata={"rule": "diagnostic_evidence", "mean_observations": mean_observations},
            )

        # Rule 4: support a minority that is behind without forcing the entire
        # class to repeat material it already understands.
        low_mastery_students = [
            s.student_id
            for s in students
            if s.concepts[concept_id].p_mastery < self.config.low_mastery_threshold
        ]
        low_fraction = len(low_mastery_students) / len(students)
        if low_mastery_students and low_fraction <= self.config.small_group_fraction:
            if len(low_mastery_students) == 1:
                action = ActionType.TA_INTERVENTION
                rationale = "One learner is below the mastery threshold; provide targeted support."
            else:
                action = ActionType.PEER_LEARNING
                rationale = (
                    f"Use a focused peer-learning activity for {len(low_mastery_students)} "
                    "learners below the mastery threshold."
                )
            return ActionDecision(
                action=action,
                concept_id=concept_id,
                target_student_ids=self._compact_targets(low_mastery_students, all_ids),
                rationale=rationale,
                priority=0.72,
                metadata={"rule": "small_group_behind", "target_fraction": low_fraction},
            )

        # Rule 5: make participation less unequal. Target low-participation
        # students when the estimates identify a meaningful subset.
        quiet_students = [
            s.student_id
            for s in students
            if s.participation < self.config.low_participation_threshold
        ]
        if summary.participation_gini >= self.config.high_participation_gini or (
            summary.mean_participation < self.config.low_participation_threshold
        ):
            targets = self._compact_targets(quiet_students, all_ids)
            return ActionDecision(
                action=ActionType.PEER_DISCUSSION,
                concept_id=concept_id,
                target_student_ids=targets,
                rationale=(
                    f"Use structured peer discussion to improve participation; "
                    f"Gini={summary.participation_gini:.2f}, "
                    f"mean participation={summary.mean_participation:.2f}."
                ),
                priority=0.64,
                metadata={"rule": "participation_balance"},
            )

        # Rule 6: low engagement calls for interaction rather than another long
        # lecture. The simulator models discussion as an engagement-positive action.
        if summary.mean_engagement < self.config.low_engagement_threshold:
            return ActionDecision(
                action=ActionType.PEER_DISCUSSION,
                concept_id=concept_id,
                rationale=(
                    f"Re-engage the class with peer discussion; mean engagement="
                    f"{summary.mean_engagement:.2f}."
                ),
                priority=0.62,
                metadata={"rule": "low_engagement"},
            )

        # Rule 7: if progress is poor after several actions, escalate to a TA
        # intervention instead of repeating the same generic activity forever.
        if total_actions >= self.config.escalation_after_actions and (
            concept_summary.mean_mastery < self.config.low_mastery_threshold
        ):
            return ActionDecision(
                action=ActionType.TA_INTERVENTION,
                concept_id=concept_id,
                target_student_ids=self._compact_targets(low_mastery_students, all_ids),
                rationale=(
                    f"Escalate after {total_actions} actions because mean mastery is "
                    f"still {concept_summary.mean_mastery:.2f}."
                ),
                priority=0.83,
                metadata={"rule": "stalled_progress"},
            )

        # Rule 8: choose content support proportional to difficulty. Use worked
        # examples for severe/initial difficulty and revision for persistent gaps.
        if concept_summary.mean_mastery < self.config.very_low_mastery_threshold:
            return ActionDecision(
                action=ActionType.EXAMPLE,
                concept_id=concept_id,
                rationale=(
                    f"Mean mastery is very low ({concept_summary.mean_mastery:.2f}); "
                    "show a worked example before more practice."
                ),
                priority=0.68,
                metadata={"rule": "very_low_mastery_example"},
            )

        if concept_summary.mean_mastery < self.config.low_mastery_threshold:
            prior_support = counts.get(ActionType.EXAMPLE.value, 0) + counts.get(ActionType.REVISION.value, 0)
            action = ActionType.REVISION if prior_support >= 2 else ActionType.EXAMPLE
            return ActionDecision(
                action=action,
                concept_id=concept_id,
                rationale=(
                    f"Mean mastery remains low ({concept_summary.mean_mastery:.2f}); "
                    f"use {action.value.replace('_', ' ')}."
                ),
                priority=0.65,
                metadata={"rule": "low_mastery_support"},
            )

        # Rule 9: once the class is not in crisis, use practice to consolidate
        # learning and provide new evidence for the student model.
        return ActionDecision(
            action=ActionType.EXERCISE,
            concept_id=concept_id,
            rationale=(
                f"Consolidate learning with practice; mean mastery="
                f"{concept_summary.mean_mastery:.2f}."
            ),
            priority=0.45,
            metadata={"rule": "consolidation_practice"},
        )

    def _compact_targets(self, target_ids: Sequence[str], all_ids: Sequence[str]) -> tuple[str, ...]:
        """Represent the whole-class target as an empty tuple.

        ``ClassroomSimulator.step`` treats ``target_ids=None`` as all students.
        The executor uses the same convention. This helper also removes stale
        or duplicate IDs by intersecting with the current class roster.
        """
        allowed = set(all_ids)
        targets = tuple(sorted(set(target_ids) & allowed))
        if len(targets) < self.config.target_min_students or set(targets) == allowed:
            return ()
        return targets
