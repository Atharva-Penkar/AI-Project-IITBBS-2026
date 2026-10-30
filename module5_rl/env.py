"""Training environment built from the project's real components.

    ClassroomSimulator (Team 3, hidden true state)
      -> events -> StudentModelEngine (Team 3, BKT + misconception beliefs)
      -> GroupSummary -> RL state -> RL action -> ActionDecision -> ClassroomSimulator ...

Progression through the syllabus is still owned by Team 4's ``AdaptivePlanner``: it moves on
automatically when its mastery criterion is met, and the RL policy may also ask to move on
early (``RLAction.ADVANCE``). The reward uses the simulator's TRUE state; the policy only ever
sees the engine's estimates.

``reset`` / ``step`` follow the usual Gym convention but there is no gym dependency.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from module3_student_model.engine import EngineConfig, StudentModelEngine
from module3_student_model.profiles import make_classroom
from module3_student_model.sample_catalog import astar_catalog
from module3_student_model.simulator import ClassroomSimulator
from module4_planner.models import ActionDecision, PlannerConfig
from module5_rl.actions import RLAction, build_decision, time_cost, valid_actions_mask
from module5_rl.metrics import (
    compute_metrics,
    known_matrix,
    left_behind_fraction,
    mean_engagement,
    misconception_count,
)
from module5_rl.policies import AdvancingPlanner
from module5_rl.state import encode_state
from shared.schema import ActionType, CourseCatalog


@dataclass(frozen=True)
class RewardWeights:
    """Reward = learning gain + misconception reduction + engagement + participation balance
    - class time, plus bonuses/penalties when a concept is completed and at the end."""

    gain: float = 10.0  # per newly-mastered (student, concept) pair, averaged over students
    misconception: float = 2.0  # per misconception removed, averaged over students
    engagement: float = 4.0  # per unit change in mean true engagement
    balance: float = 2.0  # per unit change in participation balance (1 - Gini)
    time: float = 0.1  # per class-time unit spent
    completion: float = 1.0  # syllabus coverage: bonus when a concept is completed
    leave_penalty: float = 3.0  # x share of students who have NOT mastered the concept being left
    terminal_known: float = 10.0  # x final mean mastery over the whole syllabus
    terminal_left_behind: float = 5.0  # x share of students left behind at the end


class _CapturePolicy:
    """Stub policy: lets the planner run its own checks, and records what it saw."""

    def __init__(self, env: "TeachingEnv") -> None:
        self.env = env
        self.config = env.planner_config

    def choose_action(self, summary, students, concept_id, action_counts=None) -> ActionDecision:
        self.env.ctx = (summary, students, concept_id, dict(action_counts or {}))
        return ActionDecision(action=ActionType.LECTURE, concept_id=concept_id, metadata={"rule": "defer"})


class TeachingEnv:
    def __init__(
        self,
        n_students: int = 12,
        time_budget: float = 45.0,
        catalog: CourseCatalog | None = None,
        engine_config: EngineConfig | None = None,
        planner_config: PlannerConfig | None = None,
        weights: RewardWeights | None = None,
    ) -> None:
        self.n_students = n_students
        self.T = time_budget
        self.catalog = catalog or astar_catalog()
        self.engine_config = engine_config or EngineConfig()
        self.planner_config = planner_config or PlannerConfig()
        self.w = weights or RewardWeights()
        self.cids = self.catalog.concept_ids()

    # ------------------------------------------------------------------ episode control
    def reset(self, scenario: str = "mixed", seed: int = 0) -> tuple[int, ...]:
        profiles = make_classroom(self.catalog, scenario, self.n_students, seed=seed)
        self.sim = ClassroomSimulator(self.catalog, profiles, seed=seed)
        self.engine = StudentModelEngine(
            self.catalog, self.sim.student_ids(), dataclasses.replace(self.engine_config, seed=seed)
        )
        self.planner = AdvancingPlanner(self.catalog, policy=_CapturePolicy(self), config=self.planner_config)
        self.scenario, self.seed = scenario, seed
        self.time_left, self.steps, self.done = self.T, 0, False
        self.known0 = known_matrix(self.sim, self.catalog)
        self._rewarded: set[str] = set()
        self._settle()
        return self.state

    def valid_mask(self) -> np.ndarray:
        return valid_actions_mask(self.ctx[3])

    def metrics(self) -> dict[str, float]:
        return compute_metrics(
            self.sim, self.engine, self.catalog, len(self.planner.completed_concepts),
            self.known0, self.T - max(self.time_left, 0.0), self.steps,
        )

    # ------------------------------------------------------------------ internals
    def _execute(self, d: ActionDecision) -> None:
        targets = list(d.target_student_ids) or None
        events = self.sim.step(d.action, d.concept_id, targets, d.target_misconception)
        self.engine.update_many(events)
        self.planner.record_execution(d)

    def _settle(self) -> bool:
        """Let the planner perform free transitions until it is ready to ask the policy again."""
        while True:
            d = self.planner.plan_next(self.engine)
            if d is None:
                self.course_complete = True
                return True
            if d.metadata.get("rule") == "concept_transition":
                self._execute(d)
                continue
            break
        summary, _students, concept_id, _counts = self.ctx
        self.state = encode_state(summary, concept_id, self.time_left, self.T)
        self.balance = 1.0 - summary.participation_gini
        self.course_complete = False
        return False

    # ------------------------------------------------------------------ transition
    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, dict]:
        action = RLAction(action)
        if self.done:
            raise RuntimeError("Episode finished; call reset()")
        if not self.valid_mask()[action]:
            raise ValueError(f"{action.name} is not valid before the concept has been taught")
        summary, students, concept_id, _ = self.ctx
        k0, m0, e0, b0 = known_matrix(self.sim, self.catalog), misconception_count(self.sim), mean_engagement(self.sim), self.balance

        if action == RLAction.ADVANCE:
            self.planner.advance_active()
            spent = 0.0
        else:
            d = build_decision(action, summary, students, concept_id, self.planner_config)
            self._execute(d)
            spent = time_cost(d.action)
            self.steps += 1
        self.time_left -= spent

        done = self.time_left <= 1e-9
        if not done:
            done = self._settle()
        self.done = done

        k1 = known_matrix(self.sim, self.catalog)
        w = self.w
        reward = (
            w.gain * float((k1 - k0).sum(axis=1).mean())
            + w.misconception * (m0 - misconception_count(self.sim))
            + w.engagement * (mean_engagement(self.sim) - e0)
            + w.balance * ((self.balance - b0) if not (done and self.time_left <= 1e-9) else 0.0)
            - w.time * spent
        )
        for cid in self.planner.completed_concepts - self._rewarded:
            self._rewarded.add(cid)
            not_known = 1.0 - float(k1[:, self.cids.index(cid)].mean())
            reward += w.completion - w.leave_penalty * not_known
        if done:
            reward += w.terminal_known * float(k1.mean()) - w.terminal_left_behind * left_behind_fraction(k1)
        return (None if done and self.course_complete else self.state), float(reward), done, {"action": action.name}
