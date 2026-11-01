"""STRIPS-style teaching actions: preconditions, expected effects and duration.

The classroom state for the active concept is described by a set of facts
computed from the student model (see `observe_facts`). Each action can run
when its preconditions hold, and is expected to add and remove facts.
`plan_concept` searches for the cheapest action sequence (by duration) that
reaches the goal, using uniform-cost search.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shared.schema import ActionType, CourseCatalog, GroupSummary, StudentEstimate


@dataclass(frozen=True)
class ActionSpec:
    action: ActionType
    duration: float
    requires: frozenset[str] = frozenset()
    forbids: frozenset[str] = frozenset()
    adds: frozenset[str] = frozenset()
    removes: frozenset[str] = frozenset()
    expected_effect: str = ""

    def applicable(self, facts: frozenset[str]) -> bool:
        return self.requires <= facts and not (self.forbids & facts)

    def apply(self, facts: frozenset[str]) -> frozenset[str]:
        return (facts - self.removes) | self.adds


def _f(*names: str) -> frozenset[str]:
    return frozenset(names)


ACTION_MODEL: dict[ActionType, ActionSpec] = {
    ActionType.REVISION: ActionSpec(
        ActionType.REVISION, 1.0, requires=_f("prereq_gap"), removes=_f("prereq_gap"),
        expected_effect="raises mastery of a weak prerequisite for the students missing it",
    ),
    ActionType.LECTURE: ActionSpec(
        ActionType.LECTURE, 1.0, forbids=_f("introduced", "prereq_gap"), adds=_f("introduced"),
        expected_effect="first exposure to the concept; small mastery gain, slight engagement drop",
    ),
    ActionType.DIAGNOSTIC_QUIZ: ActionSpec(
        ActionType.DIAGNOSTIC_QUIZ, 0.5, requires=_f("introduced"), forbids=_f("evidence"), adds=_f("evidence"),
        expected_effect="reduces uncertainty and reveals misconceptions",
    ),
    ActionType.TA_INTERVENTION: ActionSpec(
        ActionType.TA_INTERVENTION, 1.0, requires=_f("evidence", "misconception"), removes=_f("misconception"),
        expected_effect="removes a targeted misconception for the students who hold it",
    ),
    ActionType.EXAMPLE: ActionSpec(
        ActionType.EXAMPLE, 0.75, requires=_f("introduced", "evidence", "low_mastery"),
        forbids=_f("misconception"), removes=_f("low_mastery"),
        expected_effect="worked example; raises mastery for students who have not learned the concept",
    ),
    ActionType.PEER_DISCUSSION: ActionSpec(
        ActionType.PEER_DISCUSSION, 1.0, requires=_f("introduced", "evidence"),
        forbids=_f("misconception", "low_mastery"), adds=_f("discussed"),
        removes=_f("low_engagement", "uneven_participation"),
        expected_effect="raises engagement and gives quiet students a turn",
    ),
    ActionType.EXERCISE: ActionSpec(
        ActionType.EXERCISE, 1.0, requires=_f("introduced", "evidence", "discussed"),
        forbids=_f("misconception", "low_mastery"), adds=_f("practised"),
        expected_effect="consolidates mastery and produces new evidence",
    ),
    ActionType.ASSESSMENT: ActionSpec(
        ActionType.ASSESSMENT, 0.5, requires=_f("practised"),
        forbids=_f("misconception", "low_mastery", "low_engagement", "uneven_participation"),
        adds=_f("assessed"),
        expected_effect="confirms mastery before moving on",
    ),
}

GOAL = _f("introduced", "evidence", "discussed", "practised", "assessed")
BLOCKERS = _f("prereq_gap", "misconception", "low_mastery", "low_engagement", "uneven_participation")


@dataclass(frozen=True)
class FactThresholds:
    uncertainty: float = 0.18
    misconception_prevalence: float = 0.15
    suspected_misconception: float = 0.35
    low_mastery: float = 0.6
    low_engagement: float = 0.4
    uneven_participation_gini: float = 0.45
    prereq_mastery: float = 0.5
    prereq_gap_fraction: float = 0.25


def suspected_share(students: Sequence[StudentEstimate], concept_id: str, cut: float) -> dict[str, float]:
    """Share of students whose belief in each misconception is at least `cut`."""
    if not students:
        return {}
    ids = sorted({m for s in students for m in s.concepts[concept_id].misconceptions})
    return {
        m: sum(s.concepts[concept_id].misconceptions.get(m, 0.0) >= cut for s in students) / len(students)
        for m in ids
    }


def observe_facts(
    summary: GroupSummary,
    catalog: CourseCatalog,
    concept_id: str,
    counts: Mapping[str, int],
    t: FactThresholds = FactThresholds(),
    students: Sequence[StudentEstimate] = (),
) -> frozenset[str]:
    """Facts that hold now, from the class summary and this concept's action history.

    A misconception counts as present when enough students are above the
    `suspected_misconception` belief, so one quiz is enough to trigger support.
    """
    cs = summary.concepts[concept_id]
    facts: set[str] = set()
    taught = sum(counts.get(a.value, 0) for a in (ActionType.LECTURE, ActionType.EXAMPLE, ActionType.REVISION))
    if taught:
        facts.add("introduced")
    if counts.get(ActionType.DIAGNOSTIC_QUIZ.value, 0) and cs.mean_uncertainty < t.uncertainty:
        facts.add("evidence")
    elif counts.get(ActionType.DIAGNOSTIC_QUIZ.value, 0) >= 2:
        facts.add("evidence")
    suspected = suspected_share(students, concept_id, t.suspected_misconception)
    if max([*cs.misconception_prevalence.values(), *suspected.values()], default=0.0) >= t.misconception_prevalence:
        facts.add("misconception")
    if "evidence" in facts and cs.mean_mastery < t.low_mastery:
        facts.add("low_mastery")
    if summary.mean_engagement < t.low_engagement:
        facts.add("low_engagement")
    if summary.participation_gini >= t.uneven_participation_gini:
        facts.add("uneven_participation")
    if counts.get(ActionType.PEER_DISCUSSION.value, 0) or counts.get(ActionType.PEER_LEARNING.value, 0):
        facts.add("discussed")
    if counts.get(ActionType.EXERCISE.value, 0):
        facts.add("practised")
    if counts.get(ActionType.ASSESSMENT.value, 0):
        facts.add("assessed")
    for prereq in catalog.concept(concept_id).prerequisites:
        if summary.concepts[prereq].fraction_below_threshold >= t.prereq_gap_fraction or (
            summary.concepts[prereq].mean_mastery < t.prereq_mastery
        ):
            facts.add("prereq_gap")
            break
    return frozenset(facts)


def plan_concept(facts: frozenset[str], goal: frozenset[str] = GOAL, max_depth: int = 12) -> list[ActionType] | None:
    """Cheapest sequence of actions from `facts` to a state with all goal facts and no blockers."""
    counter = itertools.count()
    frontier = [(0.0, next(counter), facts, ())]
    best: dict[frozenset[str], float] = {facts: 0.0}
    while frontier:
        cost, _, state, path = heapq.heappop(frontier)
        if goal <= state and not (state & BLOCKERS):
            return list(path)
        if len(path) >= max_depth or cost > best.get(state, float("inf")):
            continue
        for spec in ACTION_MODEL.values():
            if not spec.applicable(state):
                continue
            nxt = spec.apply(state)
            new_cost = cost + spec.duration
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                heapq.heappush(frontier, (new_cost, next(counter), nxt, path + (spec.action,)))
    return None
