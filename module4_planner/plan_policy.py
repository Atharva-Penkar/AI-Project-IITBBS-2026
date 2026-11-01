"""Teaching policy that plans an action sequence for the active concept and replans every step.

    policy = PlanningTeachingPolicy(catalog)
    planner = AdaptivePlanner(catalog, policy=policy)

Each call observes the current facts, plans the cheapest sequence to the
concept's goal with `action_model.plan_concept`, and returns the first action.
When the plan differs from the previous one, the decision records what
changed (for example, a quiz revealed a misconception).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from module4_planner.action_model import ACTION_MODEL, FactThresholds, observe_facts, plan_concept, suspected_share
from module4_planner.models import ActionDecision, PlannerConfig
from shared.schema import ActionType, CourseCatalog, GroupSummary, StudentEstimate


class PlanningTeachingPolicy:
    def __init__(
        self,
        catalog: CourseCatalog,
        config: PlannerConfig | None = None,
        thresholds: FactThresholds | None = None,
    ) -> None:
        self.catalog = catalog
        self.config = config or PlannerConfig()
        self.thresholds = thresholds or FactThresholds()
        self._last: dict[str, tuple[frozenset[str], tuple[ActionType, ...]]] = {}

    def choose_action(
        self,
        summary: GroupSummary,
        students: Sequence[StudentEstimate],
        concept_id: str,
        action_counts: Mapping[str, int] | None = None,
    ) -> ActionDecision:
        counts = dict(action_counts or {})
        facts = observe_facts(summary, self.catalog, concept_id, counts, self.thresholds, students)
        plan = plan_concept(facts)
        previous = self._last.get(concept_id)
        self._last[concept_id] = (facts, tuple(plan or ()))

        replanned, why = False, ""
        if previous is not None and facts != previous[0] and previous[1][1:] != tuple(plan or ()):
            replanned = True
            gained = sorted(facts - previous[0])
            lost = sorted(previous[0] - facts)
            why = f" Replanned: new facts {gained or 'none'}, cleared {lost or 'none'}."

        if not plan:
            reason = "Goal reached" if plan is not None else "No action sequence reaches the goal"
            return ActionDecision(
                action=ActionType.NEXT_CONCEPT,
                concept_id=concept_id,
                rationale=f"{reason} for {concept_id!r} (facts: {sorted(facts)}); move to the next concept.",
                priority=0.4,
                metadata={"rule": "plan_complete", "facts": sorted(facts), "plan": []},
            )

        action = plan[0]
        spec = ACTION_MODEL[action]
        target_concept, targets, misconception = self._targets(action, summary, students, concept_id)
        sequence = " -> ".join(a.value for a in plan)
        return ActionDecision(
            action=action,
            concept_id=target_concept,
            target_student_ids=targets,
            target_misconception=misconception,
            rationale=(
                f"Plan for {concept_id!r}: {sequence}. Doing {action.value} now "
                f"(expected effect: {spec.expected_effect}).{why}"
            ),
            priority=0.7,
            metadata={
                "rule": f"plan:{action.value}",
                "plan": [a.value for a in plan],
                "facts": sorted(facts),
                "replanned": replanned,
                "duration": spec.duration,
                "active_concept": concept_id,
            },
        )

    def _targets(
        self,
        action: ActionType,
        summary: GroupSummary,
        students: Sequence[StudentEstimate],
        concept_id: str,
    ) -> tuple[str, tuple[str, ...], str | None]:
        """Who the action is aimed at. An empty tuple means the whole class."""
        all_ids = {s.student_id for s in students}

        def compact(ids: list[str]) -> tuple[str, ...]:
            chosen = tuple(sorted(set(ids) & all_ids))
            return () if not chosen or set(chosen) == all_ids else chosen

        if action == ActionType.REVISION:
            prereqs = self.catalog.concept(concept_id).prerequisites
            weakest = min(prereqs, key=lambda p: summary.concepts[p].mean_mastery)
            weak = [s.student_id for s in students
                    if s.concepts[weakest].p_mastery < self.thresholds.prereq_mastery]
            return weakest, compact(weak), None

        if action == ActionType.TA_INTERVENTION:
            cut = self.thresholds.suspected_misconception
            share = suspected_share(students, concept_id, cut)
            misconception = max(share, key=share.get)
            holders = [s.student_id for s in students
                       if s.concepts[concept_id].misconceptions.get(misconception, 0.0) >= cut]
            return concept_id, compact(holders), misconception

        if action == ActionType.EXAMPLE:
            low = [s.student_id for s in students
                   if s.concepts[concept_id].p_mastery < self.thresholds.low_mastery]
            return concept_id, compact(low), None

        if action == ActionType.DIAGNOSTIC_QUIZ:
            unsure = [s.student_id for s in students
                      if s.concepts[concept_id].uncertainty >= self.thresholds.uncertainty
                      or s.concepts[concept_id].n_observations == 0]
            return concept_id, compact(unsure), None

        return concept_id, (), None
