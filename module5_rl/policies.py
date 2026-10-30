"""Teaching policies that plug into Team 4's ``AdaptivePlanner``.

* ``RLTeachingPolicy``      - the learned Q-learning / SARSA policy
* ``FixedLecturePlanPolicy``- follows the original lecture plan, ignoring the class state
* ``ThresholdRepeatPolicy`` - repeats the topic while the average score is below a threshold
* ``RandomPolicy``          - sanity-check lower bound

All of them implement ``module4_planner.policy.TeachingPolicy``. ``AdvancingPlanner`` is a
thin subclass of Team 4's planner that lets a policy ask to move on to the next concept
(the stock planner only advances when its mastery criterion is met).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from module4_planner.models import ActionDecision, PlannerConfig
from module4_planner.planner import AdaptivePlanner
from module4_planner.policy import HeuristicTeachingPolicy
from module5_rl.actions import (
    ACTION_TYPE,
    N_ACTIONS,
    RLAction,
    build_decision,
    time_cost,
    valid_actions_mask,
)
from module5_rl.state import describe_state, encode_state
from shared.schema import ActionType, GroupSummary, StudentEstimate

# Progression is controlled entirely by the policy: the planner's own completion
# criterion (mean mastery >= 1.0) can never be met by a probabilistic estimate.
MANUAL_PROGRESSION = PlannerConfig(mastery_target=1.0, max_fraction_below_threshold_for_completion=0.0)


class AdvancingPlanner(AdaptivePlanner):
    """``AdaptivePlanner`` + support for a policy that asks to move on early."""

    def advance_active(self) -> None:
        """Mark the active concept as finished (the policy decided to move on)."""
        if self._active_concept_id is not None:
            self._completed_concepts.add(self._active_concept_id)
            self._active_concept_id = None

    def plan_next(self, engine: object) -> ActionDecision | None:
        decision = super().plan_next(engine)
        if (
            decision is not None
            and decision.action == ActionType.NEXT_CONCEPT
            and decision.metadata.get("rule") != "concept_transition"
        ):
            self.advance_active()
            return super().plan_next(engine)  # transition into the next concept (or None)
        return decision


def whole_class_decision(action: RLAction, concept_id: str, rationale: str, rule: str) -> ActionDecision:
    """Un-targeted decision used by the fixed baselines (they ignore who needs what)."""
    return ActionDecision(
        action=ACTION_TYPE[action],
        concept_id=concept_id,
        rationale=rationale,
        priority=0.5,
        metadata={"rule": rule if action != RLAction.ADVANCE else "policy_advance"},
    )


class RLTeachingPolicy:
    """Greedy policy over a learned Q-table.

    * Unseen states (never updated during training) fall back to Team 4's rule-based
      ``HeuristicTeachingPolicy`` so the planner never receives an arbitrary action.
    * ``choose_action`` keeps a small clock of class time so the ``time_pace`` feature is
      available even though the ``TeachingPolicy`` interface has no time argument. A
      controller that knows the real time can call ``set_time_left``.
    """

    def __init__(
        self,
        q_table: np.ndarray,
        visits: np.ndarray | None = None,
        config: PlannerConfig | None = None,
        time_budget: float = 45.0,
        min_visits: int = 1,
        name: str = "rl",
    ) -> None:
        if q_table.shape[-1] != N_ACTIONS:
            raise ValueError(f"Q-table must end with {N_ACTIONS} actions, got shape {q_table.shape}")
        self.Q = q_table
        self.visits = visits if visits is not None else (q_table != 0).astype(np.int64)
        self.config = config or PlannerConfig()
        self.time_budget = time_budget
        self.min_visits = min_visits
        self.name = name
        self._fallback = HeuristicTeachingPolicy(self.config)
        self.time_left = time_budget
        self.fallback_count = 0
        self.decision_count = 0

    @classmethod
    def load(cls, q_path: str, visits_path: str | None = None, **kwargs) -> "RLTeachingPolicy":
        q = np.load(q_path)
        v = np.load(visits_path) if visits_path else None
        return cls(q, v, **kwargs)

    def reset(self) -> None:
        self.time_left = self.time_budget
        self.fallback_count = 0
        self.decision_count = 0

    def set_time_left(self, time_left: float) -> None:
        self.time_left = time_left

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.decision_count if self.decision_count else 0.0

    def choose_action(
        self,
        summary: GroupSummary,
        students: Sequence[StudentEstimate],
        concept_id: str,
        action_counts: Mapping[str, int] | None = None,
    ) -> ActionDecision:
        state = encode_state(summary, concept_id, self.time_left, self.time_budget)
        mask = valid_actions_mask(action_counts)
        q, v = self.Q[state], self.visits[state]
        tried = (v >= self.min_visits) & mask
        self.decision_count += 1

        if not tried.any():
            self.fallback_count += 1
            decision = self._fallback.choose_action(summary, students, concept_id, action_counts)
            decision = ActionDecision(
                action=decision.action,
                concept_id=decision.concept_id,
                target_student_ids=decision.target_student_ids,
                target_misconception=decision.target_misconception,
                rationale=f"[unseen state -> rule-based fallback] {decision.rationale}",
                priority=decision.priority,
                metadata={**decision.metadata, "fallback": True, "state": state},
            )
        else:
            masked = np.where(tried, q, -np.inf)
            order = np.argsort(-masked, kind="stable")
            best = RLAction(int(order[0]))
            second = RLAction(int(order[1])) if tried.sum() > 1 else None
            margin = float(masked[best] - masked[second]) if second is not None else 0.0
            why = (
                f"RL policy ({self.name}) in state [{describe_state(state)}] chose {best.name} "
                f"(Q={q[best]:.2f})"
            )
            if second is not None:
                why += f" over {second.name} (Q={q[second]:.2f})"
            decision = build_decision(
                best,
                summary,
                students,
                concept_id,
                self.config,
                rationale=why + ".",
                priority=float(np.clip(0.5 + 0.05 * margin, 0.3, 0.95)),
                metadata={
                    "state": state,
                    "q_values": {RLAction(i).name: float(q[i]) for i in range(N_ACTIONS) if tried[i]},
                    "fallback": False,
                },
            )
        self.time_left -= time_cost(decision.action)
        return decision


class FixedLecturePlanPolicy:
    """Baseline 1: the original lecture plan, followed for every concept regardless of the class."""

    PLAN = (
        RLAction.LECTURE,
        RLAction.EXAMPLE,
        RLAction.EXERCISE,
        RLAction.PEER_DISCUSSION,
        RLAction.DIAGNOSTIC_QUIZ,
    )

    def __init__(self) -> None:
        self.config = MANUAL_PROGRESSION
        self.name = "FixedPlan"

    def reset(self) -> None: ...

    def choose_action(self, summary, students, concept_id, action_counts=None) -> ActionDecision:
        step = sum((action_counts or {}).values())
        if step >= len(self.PLAN):
            return whole_class_decision(RLAction.ADVANCE, concept_id, "Lecture plan for this concept is finished.", "fixed_plan")
        action = self.PLAN[step]
        return whole_class_decision(action, concept_id, f"Lecture plan step {step + 1}/{len(self.PLAN)}.", "fixed_plan")


class ThresholdRepeatPolicy:
    """Baseline 2: lecture, quiz, and repeat the lecture while the class average is below a threshold."""

    def __init__(self, threshold: float = 0.6, max_lectures: int = 4) -> None:
        self.config = MANUAL_PROGRESSION
        self.threshold = threshold
        self.max_lectures = max_lectures
        self.name = "ThresholdRepeat"

    def reset(self) -> None: ...

    def choose_action(self, summary, students, concept_id, action_counts=None) -> ActionDecision:
        counts = action_counts or {}
        lectures = counts.get(ActionType.LECTURE.value, 0)
        quizzes = counts.get(ActionType.DIAGNOSTIC_QUIZ.value, 0)
        if lectures == 0:
            return whole_class_decision(RLAction.LECTURE, concept_id, "First lecture on the concept.", "threshold_repeat")
        if quizzes < lectures:
            return whole_class_decision(RLAction.DIAGNOSTIC_QUIZ, concept_id, "Quiz after lecture.", "threshold_repeat")
        score = summary.concepts[concept_id].mean_mastery
        if score < self.threshold and lectures < self.max_lectures:
            return whole_class_decision(
                RLAction.LECTURE, concept_id, f"Average {score:.2f} < {self.threshold}: repeat the topic.", "threshold_repeat"
            )
        return whole_class_decision(
            RLAction.ADVANCE, concept_id, f"Average {score:.2f} >= {self.threshold} (or repeat limit): move on.", "threshold_repeat"
        )


class RandomPolicy:
    """Lower bound: uniformly random valid action (uses the same targeting as the RL policy)."""

    def __init__(self, seed: int = 0, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self.name = "Random"
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self._seed if seed is None else seed)

    def choose_action(self, summary, students, concept_id, action_counts=None) -> ActionDecision:
        valid = np.flatnonzero(valid_actions_mask(action_counts))
        action = RLAction(int(self.rng.choice(valid)))
        return build_decision(action, summary, students, concept_id, self.config, rationale="Uniformly random action.")
