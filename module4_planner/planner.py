"""Course-aware planner that sequences concepts and requests one teaching action."""

from __future__ import annotations

from collections import Counter, defaultdict
from shared.schema import ActionType, CourseCatalog, GroupSummary, StudentEstimate

from module4_planner.models import ActionDecision, PlannerConfig
from module4_planner.policy import HeuristicTeachingPolicy, TeachingPolicy


class AdaptivePlanner:
    """Maintain course progression and delegate within-concept choices to a policy.

    The planner enforces prerequisite order and advances only when the class
    estimate meets the configured mastery criterion. `NEXT_CONCEPT` is emitted
    as an explicit transition marker because Team 3's simulator intentionally
    returns no events for that action.
    """

    def __init__(
        self,
        catalog: CourseCatalog,
        policy: TeachingPolicy | None = None,
        config: PlannerConfig | None = None,
    ) -> None:
        self.catalog = catalog
        inherited_config = getattr(policy, "config", None) if policy is not None else None
        self.config = config or inherited_config or PlannerConfig()
        self.policy = policy or HeuristicTeachingPolicy(self.config)
        self._concept_order = [concept.id for concept in catalog.concepts]
        self._prerequisites = {concept.id: tuple(concept.prerequisites) for concept in catalog.concepts}
        self._validate_prerequisites()

        self._active_concept_id: str | None = None
        self._completed_concepts: set[str] = set()
        # Counts are maintained separately for every concept. A diagnostic
        # quiz limit is meaningful per concept, not across the entire course.
        self._concept_action_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._decisions_made = 0

    @property
    def active_concept_id(self) -> str | None:
        """The concept being taught now, or ``None`` between concepts."""
        return self._active_concept_id

    @property
    def completed_concepts(self) -> frozenset[str]:
        """An immutable view of concepts the planner has accepted as mastered."""
        return frozenset(self._completed_concepts)

    @property
    def action_counts(self) -> dict[str, int]:
        """Return a copy of executed non-transition action counts by action value."""
        total: Counter[str] = Counter()
        for counts in self._concept_action_counts.values():
            total.update(counts)
        return dict(total)

    def action_counts_for(self, concept_id: str) -> dict[str, int]:
        """Return executed action counts for one concept (useful for audit/tests)."""
        return dict(self._concept_action_counts.get(concept_id, Counter()))

    @property
    def decisions_made(self) -> int:
        return self._decisions_made

    @property
    def is_complete(self) -> bool:
        return len(self._completed_concepts) == len(self._concept_order)

    def plan_next(self, engine: object) -> ActionDecision | None:
        """Plan the next action from a ``StudentModelEngine``-compatible object.

        The duck-typed ``engine`` argument keeps unit tests simple, while the
        production interface is Team 3's StudentModelEngine. Returns ``None``
        when no students are registered or the catalog has been completed.
        """
        student_ids = list(engine.student_ids())  # type: ignore[attr-defined]
        if not student_ids:
            return None

        summary: GroupSummary = engine.get_class_summary()  # type: ignore[attr-defined]

        # A completed concept stays completed. The currently active concept is
        # checked after every observation batch, not merely after a fixed lesson.
        if self._active_concept_id is not None and self._meets_completion_criterion(
            summary, self._active_concept_id
        ):
            self._completed_concepts.add(self._active_concept_id)
            self._active_concept_id = None

        if self._active_concept_id is None:
            next_concept = self._next_eligible_concept()
            if next_concept is None:
                # With a valid acyclic graph, no eligible concept means either
                # everything is complete or the catalog is inconsistent.
                if not self.is_complete:
                    raise RuntimeError("No concept is eligible; check prerequisite configuration")
                return None

            self._active_concept_id = next_concept
            self._decisions_made += 1
            return ActionDecision(
                action=ActionType.NEXT_CONCEPT,
                concept_id=next_concept,
                rationale=(
                    f"Begin concept {next_concept!r}; all declared prerequisites "
                    "have been completed by the planner."
                ),
                priority=0.35,
                metadata={
                    "rule": "concept_transition",
                    "completed_prerequisites": list(self._prerequisites[next_concept]),
                },
            )

        student_estimates: list[StudentEstimate] = [engine.get_student(sid) for sid in student_ids]  # type: ignore[attr-defined]
        decision = self.policy.choose_action(
            summary=summary,
            students=student_estimates,
            concept_id=self._active_concept_id,
            action_counts=self._concept_action_counts[self._active_concept_id],
        )
        self._decisions_made += 1
        return decision

    def record_execution(self, decision: ActionDecision) -> None:
        """Update decision-history counters after an action actually executes."""
        if decision.action != ActionType.NEXT_CONCEPT:
            self._concept_action_counts[decision.concept_id][decision.action.value] += 1

    def reset(self) -> None:
        """Reset course progression while retaining the immutable catalog/policy."""
        self._active_concept_id = None
        self._completed_concepts.clear()
        self._concept_action_counts.clear()
        self._decisions_made = 0

    def _meets_completion_criterion(self, summary: GroupSummary, concept_id: str) -> bool:
        concept = summary.concepts[concept_id]
        return (
            concept.mean_mastery >= self.config.mastery_target
            and concept.fraction_below_threshold
            <= self.config.max_fraction_below_threshold_for_completion
        )

    def _next_eligible_concept(self) -> str | None:
        """Choose the first not-yet-completed concept with completed prerequisites."""
        for concept_id in self._concept_order:
            if concept_id in self._completed_concepts:
                continue
            if set(self._prerequisites[concept_id]).issubset(self._completed_concepts):
                return concept_id
        return None

    def _validate_prerequisites(self) -> None:
        """Reject unknown prerequisites and cycles before the first lesson starts."""
        known = set(self._concept_order)
        for concept_id, prerequisites in self._prerequisites.items():
            missing = set(prerequisites) - known
            if missing:
                raise ValueError(
                    f"Concept {concept_id!r} references unknown prerequisite(s): {sorted(missing)}"
                )
            if concept_id in prerequisites:
                raise ValueError(f"Concept {concept_id!r} cannot be its own prerequisite")

        # Kahn's algorithm detects cycles independently of the order in which
        # concepts happen to be listed in CourseCatalog.
        remaining = {cid: set(reqs) for cid, reqs in self._prerequisites.items()}
        ready = [cid for cid, reqs in remaining.items() if not reqs]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for cid, reqs in remaining.items():
                if current in reqs:
                    reqs.remove(current)
                    if not reqs:
                        ready.append(cid)
        if visited != len(remaining):
            cyclic = sorted(cid for cid, reqs in remaining.items() if reqs)
            raise ValueError(f"Course prerequisites contain a cycle involving: {cyclic}")
