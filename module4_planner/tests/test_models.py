"""Validation tests for planner records and configuration."""

import pytest

from shared.schema import ActionType
from module4_planner.models import ActionDecision, PlannerConfig


def test_action_decision_rejects_bad_priority() -> None:
    with pytest.raises(ValueError):
        ActionDecision(action=ActionType.EXAMPLE, concept_id="topic", priority=1.1)


def test_action_decision_rejects_duplicate_targets() -> None:
    with pytest.raises(ValueError):
        ActionDecision(
            action=ActionType.TA_INTERVENTION,
            concept_id="topic",
            target_student_ids=("s01", "s01"),
        )


def test_planner_config_validates_thresholds() -> None:
    with pytest.raises(ValueError):
        PlannerConfig(very_low_mastery_threshold=0.6, low_mastery_threshold=0.4)
