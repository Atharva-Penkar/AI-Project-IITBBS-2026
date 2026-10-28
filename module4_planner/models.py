"""Small value objects shared by the planner, policy, executor, and evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shared.schema import ActionType


@dataclass(frozen=True)
class ActionDecision:
    """One complete, auditable teaching decision.

    ``target_student_ids`` being empty means the action applies to the entire
    class. A non-empty tuple means that the action is intended for that subset.
    ``rationale`` is human-readable so a teacher or evaluator can inspect why
    the policy chose this action instead of treating the planner as a black box.
    """

    action: ActionType
    concept_id: str
    target_student_ids: tuple[str, ...] = ()
    target_misconception: str | None = None
    rationale: str = ""
    priority: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject malformed decisions early, close to the source of the error."""
        if not self.concept_id.strip():
            raise ValueError("concept_id must be a non-empty string")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must be in the interval [0, 1]")
        if len(set(self.target_student_ids)) != len(self.target_student_ids):
            raise ValueError("target_student_ids must not contain duplicates")


@dataclass(frozen=True)
class PlannerConfig:
    """Thresholds for progression and the baseline heuristic policy.

    Defaults are starting values for simulation, not universal educational
    cut-offs. Tune them on held-out scenarios before making classroom claims.
    """

    # A concept is complete when the class reaches the mastery target and the
    # proportion below the student-model's 0.4 mastery threshold is small.
    mastery_target: float = 0.72
    max_fraction_below_threshold_for_completion: float = 0.20
    prerequisite_mastery_threshold: float = 0.60

    # Action-selection thresholds.
    low_mastery_threshold: float = 0.45
    very_low_mastery_threshold: float = 0.35
    uncertainty_threshold: float = 0.18
    misconception_prevalence_threshold: float = 0.15
    individual_misconception_threshold: float = 0.45
    high_participation_gini: float = 0.45
    low_participation_threshold: float = 0.28
    low_engagement_threshold: float = 0.38
    small_group_fraction: float = 0.35

    # Guardrails against repeatedly quizzing the class or letting one rule
    # dominate indefinitely when the model is not improving.
    max_diagnostic_actions_per_concept: int = 2
    diagnostic_observation_target: int = 1
    escalation_after_actions: int = 5

    # If a targeted subset is empty after filtering, fall back to the class.
    # This avoids accidental no-op teaching actions caused by stale estimates.
    target_min_students: int = 1

    def __post_init__(self) -> None:
        """Validate intervals and counts so bad config is caught at startup."""
        unit_values = {
            "mastery_target": self.mastery_target,
            "max_fraction_below_threshold_for_completion": self.max_fraction_below_threshold_for_completion,
            "prerequisite_mastery_threshold": self.prerequisite_mastery_threshold,
            "low_mastery_threshold": self.low_mastery_threshold,
            "very_low_mastery_threshold": self.very_low_mastery_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "misconception_prevalence_threshold": self.misconception_prevalence_threshold,
            "individual_misconception_threshold": self.individual_misconception_threshold,
            "high_participation_gini": self.high_participation_gini,
            "low_participation_threshold": self.low_participation_threshold,
            "low_engagement_threshold": self.low_engagement_threshold,
            "small_group_fraction": self.small_group_fraction,
        }
        for name, value in unit_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in the interval [0, 1]")
        if self.very_low_mastery_threshold > self.low_mastery_threshold:
            raise ValueError("very_low_mastery_threshold must be <= low_mastery_threshold")
        if self.max_diagnostic_actions_per_concept < 0:
            raise ValueError("max_diagnostic_actions_per_concept cannot be negative")
        if self.diagnostic_observation_target < 0:
            raise ValueError("diagnostic_observation_target cannot be negative")
        if self.escalation_after_actions < 1:
            raise ValueError("escalation_after_actions must be at least 1")
        if self.target_min_students < 1:
            raise ValueError("target_min_students must be at least 1")
