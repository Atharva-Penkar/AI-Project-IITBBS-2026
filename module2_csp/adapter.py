"""Builds classroom CSPs from live student-model estimates and solves them.

    scheduler = GroupScheduler(catalog, student_ids, seed=0)
    plan = scheduler.schedule(engine, concept_id, student_ids, kind="peer")
    for group in plan.groups:
        ...   # group.student_ids, group.ta, group.slot, group.concept_id, group.kind

kind="peer": students who meet the concept's prerequisites are split into
mixed-ability groups for the concept; students who do not are placed in
TA-led revision groups for the prerequisites they are missing.
kind="ta": every listed student is placed in a small TA-led group, either
on the concept or on their first missing prerequisite.
Groups scheduled earlier in a call use up TA time and rooms for later ones.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from module2_csp.classroom_csp import (
    Activity,
    ClassroomConfig,
    Student,
    TeachingAssistant,
    format_result,
    solve_classroom,
)
from shared.schema import CourseCatalog, StudentEstimate


@dataclass(frozen=True)
class ScheduledGroup:
    name: str
    student_ids: tuple[str, ...]
    ta: str
    slot: str
    concept_id: str
    kind: str


@dataclass
class GroupPlan:
    feasible: bool
    groups: list[ScheduledGroup] = field(default_factory=list)
    unplaced: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    text: str = ""


def _transitive_prereqs(catalog: CourseCatalog, concept_id: str) -> list[str]:
    seen: list[str] = []
    stack = list(catalog.concept(concept_id).prerequisites)
    while stack:
        c = stack.pop()
        if c not in seen:
            seen.append(c)
            stack.extend(catalog.concept(c).prerequisites)
    return seen


class GroupScheduler:
    """Turns StudentEstimates into ClassroomConfigs and solves them.

    Timetables and TAs are not part of the student model, so they are created
    here from a seed. Pass `slots`, `availability` and `tas` to use real ones.
    """

    def __init__(
        self,
        catalog: CourseCatalog,
        student_ids: list[str],
        seed: int = 0,
        slots: list[str] | None = None,
        availability: dict[str, set[str]] | None = None,
        tas: list[TeachingAssistant] | None = None,
        max_parallel_groups: int = 3,
        mastery_threshold: float = 0.5,
        max_nodes: int = 5_000,
        time_limit: float = 2.0,
    ) -> None:
        self.catalog = catalog
        self.slots = slots or ["Slot-1", "Slot-2", "Slot-3"]
        rng = random.Random(seed)
        if availability is None:
            availability = {}
            for sid in student_ids:
                free = {s for s in self.slots if rng.random() < 0.85}
                availability[sid] = free or {rng.choice(self.slots)}
        self.availability = availability
        self.tas = tas or [
            TeachingAssistant("TA1", set(self.slots), 3),
            TeachingAssistant("TA2", set(self.slots), 3),
            TeachingAssistant("TA3", set(self.slots[1:]), 2),
        ]
        self.max_parallel_groups = max_parallel_groups
        self.mastery_threshold = mastery_threshold
        self.max_nodes = max_nodes
        self.time_limit = time_limit

    def _students(self, estimates: list[StudentEstimate], concept_id: str) -> list[Student]:
        scope = [concept_id, *_transitive_prereqs(self.catalog, concept_id)]
        return [
            Student(
                sid=s.student_id,
                ability=round(sum(s.concepts[c].p_mastery for c in scope) / len(scope), 3),
                mastery={c: round(est.p_mastery, 3) for c, est in s.concepts.items()},
                available_slots=set(self.availability.get(s.student_id, self.slots)),
            )
            for s in estimates
        ]

    def peer_config(self, students: list[Student], concept_id: str) -> ClassroomConfig:
        n = len(students)
        abilities = sorted(s.ability for s in students)
        weak_cut = abilities[int(0.3 * (n - 1))]
        strong_cut = abilities[int(0.7 * (n - 1))]
        return ClassroomConfig(
            students, self.tas,
            [Activity(f"act_{concept_id}", f"Peer work on {concept_id}", "peer", concept_id)],
            self.slots, {c.id: list(c.prerequisites) for c in self.catalog.concepts},
            min_group_size=min(3, n), max_group_size=5, max_parallel_groups=self.max_parallel_groups,
            mastery_threshold=self.mastery_threshold, need_ceiling=1.0,
            weak_threshold=min(weak_cut + 1e-9, strong_cut), strong_threshold=strong_cut,
        )

    def ta_config(self, students: list[Student], concept_id: str, kind: str = "ta") -> ClassroomConfig:
        """Small TA-led groups, all working on one concept."""
        return ClassroomConfig(
            students, self.tas,
            [Activity(f"{kind}_{concept_id}", f"{kind} on {concept_id}", kind, concept_id)],
            self.slots, {c.id: list(c.prerequisites) for c in self.catalog.concepts},
            min_group_size=1, max_group_size=4, max_parallel_groups=self.max_parallel_groups,
            mastery_threshold=self.mastery_threshold, weak_threshold=0.0, strong_threshold=0.0,
            need_ceiling=1.0,
        )

    def first_missing(self, student: Student, concept_id: str) -> str | None:
        """The earliest prerequisite (in teaching order) the student has not mastered, or None."""
        prereqs = set(_transitive_prereqs(self.catalog, concept_id))
        for c in self.catalog.concept_ids():
            if c in prereqs and student.mastery.get(c, 0.0) < self.mastery_threshold:
                return c
        return None

    def _with_remaining_capacity(self, cfg: ClassroomConfig, used: list[ScheduledGroup]) -> ClassroomConfig:
        """Removes TA time and rooms already taken by groups scheduled earlier in this call."""
        busy = {(g.ta, g.slot) for g in used}
        load = {t.tid: sum(1 for g in used if g.ta == t.tid) for t in self.tas}
        rooms = {slot: sum(1 for g in used if g.slot == slot) for slot in self.slots}
        free_slots = [slot for slot in self.slots if rooms[slot] < self.max_parallel_groups]
        cfg.tas = [
            TeachingAssistant(t.tid, {x for x in t.available_slots if (t.tid, x) not in busy and x in free_slots},
                              max(0, t.max_groups - load[t.tid]), set(t.expertise))
            for t in self.tas
        ]
        cfg.slots = free_slots or list(self.slots)
        if free_slots:
            cfg.max_parallel_groups = min(self.max_parallel_groups - rooms[slot] for slot in free_slots)
        return cfg

    def _solve(self, cfg: ClassroomConfig, plan: GroupPlan, prefix: str) -> None:
        cfg = self._with_remaining_capacity(cfg, plan.groups)
        res = solve_classroom(cfg, max_nodes=self.max_nodes, time_limit=self.time_limit)
        plan.text += format_result(cfg, res) + "\n"
        if not res.feasible:
            plan.feasible = False
            plan.unplaced += [s.sid for s in cfg.students]
            plan.explanation += res.explanation
            return
        activities = {a.aid: a for a in cfg.activities}
        for name, g in res.solution.items():
            act = activities[g["activity"]]
            plan.groups.append(ScheduledGroup(f"{prefix}{name}", tuple(g["students"]), g["ta"], g["slot"], act.concept, act.kind))

    def _revision_groups(self, students: list[Student], concept_id: str, plan: GroupPlan, kind: str) -> None:
        """Each student revises their first missing prerequisite; students with none work on the concept."""
        buckets: dict[str, list[Student]] = {}
        for s in students:
            target = self.first_missing(s, concept_id)
            buckets.setdefault(target or concept_id, []).append(s)
        for target in sorted(buckets, key=self.catalog.concept_ids().index):
            activity_kind = kind if target == concept_id else "revision"
            self._solve(self.ta_config(buckets[target], target, activity_kind), plan, f"TA-{target}-")

    def schedule(self, engine, concept_id: str, student_ids: list[str], kind: str = "peer") -> GroupPlan:
        students = self._students([engine.get_student(sid) for sid in student_ids], concept_id)
        plan = GroupPlan(True)
        if kind == "ta":
            self._revision_groups(students, concept_id, plan, "ta")
            return plan

        probe = self.peer_config(students, concept_id)
        activity = probe.activities[0]
        ready = [s for s in students if probe.student_ready(s, activity)]
        not_ready = [s for s in students if not probe.student_ready(s, activity)]
        for s in not_ready:
            plan.explanation.append(
                f"[prerequisite] {s.sid} is missing {probe.missing_prereqs(s, activity)} "
                f"and revises {self.first_missing(s, concept_id)} with a TA."
            )
        if len(ready) >= 2:
            self._solve(self.peer_config(ready, concept_id), plan, "P-")
        else:
            not_ready += ready
        if not_ready:
            self._revision_groups(not_ready, concept_id, plan, "ta")
        return plan
