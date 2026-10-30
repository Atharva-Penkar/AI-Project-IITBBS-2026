"""Reinforcement-learning teaching policy (Team 5, Person A).

A tabular Q-learning / SARSA policy that plugs into Team 4's ``AdaptivePlanner``
through the existing ``TeachingPolicy`` interface.
"""

from module5_rl.policies import (
    AdvancingPlanner,
    FixedLecturePlanPolicy,
    RandomPolicy,
    RLTeachingPolicy,
    ThresholdRepeatPolicy,
)

__all__ = [
    "AdvancingPlanner",
    "FixedLecturePlanPolicy",
    "RLTeachingPolicy",
    "RandomPolicy",
    "ThresholdRepeatPolicy",
]
