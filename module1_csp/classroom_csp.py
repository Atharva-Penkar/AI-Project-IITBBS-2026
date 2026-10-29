

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from csp_solver import CSPSolver, Propagator

Assignment = Dict[str, Any]


@dataclass
class Student:
    sid: str
    ability: float                    
    mastery: Dict[str, float]         
    available_slots: Set[str]


@dataclass
class TeachingAssistant:
    tid: str
    available_slots: Set[str]
    max_groups: int                   
    expertise: Set[str] = field(default_factory=set)   


@dataclass
class Activity:
    aid: str
    name: str
    kind: str                         
    concept: str                      


@dataclass
class ClassroomConfig:
    students: List[Student]
    tas: List[TeachingAssistant]
    activities: List[Activity]
    slots: List[str]
    concept_prereqs: Dict[str, List[str]]   
    min_group_size: int = 3
    max_group_size: int = 5
    max_parallel_groups: int = 2        
    mastery_threshold: float = 0.5          
    weak_threshold: float = 0.4             
    strong_threshold: float = 0.7           
    max_weak_ratio: float = 0.5
    need_ceiling: float = 0.9
    ta_balance_tolerance: Optional[int] = None  

    def is_weak(self, s: Student) -> bool:
        return s.ability < self.weak_threshold

    def is_strong(self, s: Student) -> bool:
        return s.ability >= self.strong_threshold

    def required_concepts(self, act: Activity) -> Set[str]:
        seen: Set[str] = set()
        stack = list(self.concept_prereqs.get(act.concept, []))
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            stack.extend(self.concept_prereqs.get(c, []))
        return seen

    def missing_prereqs(self, s: Student, act: Activity) -> List[str]:
        return sorted(c for c in self.required_concepts(act)
                      if s.mastery.get(c, 0.0) < self.mastery_threshold)

    def student_ready(self, s: Student, act: Activity) -> bool:
        return not self.missing_prereqs(s, act)


def validate_config(cfg: ClassroomConfig) -> List[str]:
    problems: List[str] = []

    def duplicates(ids: List[str]) -> List[str]:
        return sorted({x for x in ids if ids.count(x) > 1})

    if not cfg.students:
        problems.append("no students to schedule.")
    for label, ids in (("student", [s.sid for s in cfg.students]),
                       ("TA", [t.tid for t in cfg.tas]),
                       ("activity", [a.aid for a in cfg.activities]),
                       ("slot", list(cfg.slots))):
        dup = duplicates(ids)
        if dup:
            problems.append(f"duplicate {label} ids: {dup}.")
    for s in cfg.students:
        if not 0.0 <= s.ability <= 1.0:
            problems.append(f"{s.sid}: ability {s.ability} is outside [0, 1].")
        bad = sorted(c for c, p in s.mastery.items() if not 0.0 <= p <= 1.0)
        if bad:
            problems.append(f"{s.sid}: mastery outside [0, 1] for {bad}.")
    for t in cfg.tas:
        if t.max_groups < 0:
            problems.append(f"{t.tid}: max_groups must be >= 0.")
    if cfg.min_group_size < 1:
        problems.append("min_group_size must be >= 1.")
    if cfg.max_group_size < cfg.min_group_size:
        problems.append("max_group_size must be >= min_group_size.")
    if cfg.max_parallel_groups < 1:
        problems.append("max_parallel_groups must be >= 1.")
    if cfg.weak_threshold > cfg.strong_threshold:
        problems.append("weak_threshold must be <= strong_threshold.")
    if not 0.0 <= cfg.max_weak_ratio <= 1.0:
        problems.append("max_weak_ratio must be in [0, 1].")
    if not 0.0 <= cfg.mastery_threshold <= 1.0:
        problems.append("mastery_threshold must be in [0, 1].")
    if cfg.ta_balance_tolerance is not None and cfg.ta_balance_tolerance < 0:
        problems.append("ta_balance_tolerance must be None or >= 0.")
    return problems


@dataclass
class Constraint:
    name: str
    category: str
    scope: Tuple[str, ...]
    fn: Callable[[Assignment], bool]   # must return True when under-determined
    description: str = ""


class CSP:
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]],
                 constraints: List[Constraint]):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.constraints_of: Dict[str, List[Constraint]] = {v: [] for v in variables}
        for c in constraints:
            for v in c.scope:
                self.constraints_of[v].append(c)

    def violations(self, assignment: Assignment) -> List[str]:
        return [c.name for c in self.constraints if not c.fn(assignment)]


def v_member(sid: str) -> str:
    return f"grp[{sid}]"


def v_slot(g: int) -> str:
    return f"slot[G{g}]"


def v_act(g: int) -> str:
    return f"act[G{g}]"


def v_ta(g: int) -> str:
    return f"ta[G{g}]"


CATEGORY_INFO = {
    "group_size": "group size limits (min/max students per group)",
    "ability_mix": "ability mix (weak ratio cap, weak students need a strong peer)",
    "prerequisite": "prerequisite completion (members must have mastered required concepts)",
    "activity_need": "activity relevance (group should not already know the concept)",
    "student_availability": "student time-slot availability",
    "ta_availability": "TA time-slot availability",
    "ta_expertise": "TA expertise for the activity concept",
    "ta_conflict": "a TA cannot supervise two groups in the same slot",
    "ta_workload": "TA workload cap / balance",
    "room_capacity": "number of parallel groups (rooms) per slot",
}
ALL_CATEGORIES = list(CATEGORY_INFO.keys())


def _eligible_domains(cfg: ClassroomConfig, disabled: Set[str]
                      ) -> Tuple[List[str], List[str], List[str]]:
    slots, acts, tas = list(cfg.slots), list(cfg.activities), list(cfg.tas)
    if "ta_availability" not in disabled:
        slots = [s for s in slots if any(s in t.available_slots for t in tas)]
        tas = [t for t in tas if t.available_slots & set(cfg.slots)]
    if "ta_expertise" not in disabled:
        acts = [a for a in acts
                if any(not t.expertise or a.concept in t.expertise for t in tas)]
        tas = [t for t in tas
               if not t.expertise or any(a.concept in t.expertise for a in acts)]
    return slots, [a.aid for a in acts], [t.tid for t in tas]


def build_csp(cfg: ClassroomConfig, k: int,
              disabled: Optional[Set[str]] = None) -> CSP:
    disabled = disabled or set()
    students = cfg.students
    n = len(students)
    member_vars = [v_member(s.sid) for s in students]
    act_by_id = {a.aid: a for a in cfg.activities}
    ta_by_id = {t.tid: t for t in cfg.tas}
    groups = list(range(k))

    slot_dom, act_dom, ta_dom = _eligible_domains(cfg, disabled)
    variables: List[str] = list(member_vars)
    domains: Dict[str, List[Any]] = {v: list(groups) for v in member_vars}
    for g in groups:
        variables += [v_slot(g), v_act(g), v_ta(g)]
        domains[v_slot(g)] = list(slot_dom)
        domains[v_act(g)] = list(act_dom)
        domains[v_ta(g)] = list(ta_dom)

    constraints: List[Constraint] = []

    def add(name, cat, scope, fn, desc=""):
        if cat in disabled:
            return
        constraints.append(Constraint(name, cat, tuple(scope), fn, desc))

    def symmetry(a: Assignment) -> bool:
        mx = -1
        for v in member_vars:
            if v not in a:
                break
            if a[v] > mx + 1:
                return False
            mx = max(mx, a[v])
        return True

    constraints.append(Constraint("symmetry", "symmetry", tuple(member_vars), symmetry,
                                  "canonical group labelling"))

    def members_of(a: Assignment, g: int) -> Tuple[List[Student], bool]:
        mem = [s for s, v in zip(students, member_vars) if v in a and a[v] == g]
        complete = all(v in a for v in member_vars)
        return mem, complete

    def make_size(g: int):
        def fn(a: Assignment) -> bool:
            assigned = [a[v] for v in member_vars if v in a]
            cnt = sum(1 for x in assigned if x == g)
            remaining = n - len(assigned)
            if cnt > cfg.max_group_size:
                return False
            if cnt + remaining < cfg.min_group_size:
                return False
            return True
        return fn

    for g in groups:
        add(f"group_size(G{g})", "group_size", member_vars, make_size(g),
            f"G{g}: {cfg.min_group_size}..{cfg.max_group_size} students")

    weak_cap_partial = int(math.floor(cfg.max_weak_ratio * cfg.max_group_size + 1e-9))

    def make_mix(g: int):
        def fn(a: Assignment) -> bool:
            mem, complete = members_of(a, g)
            weak = sum(1 for s in mem if cfg.is_weak(s))
            strong = sum(1 for s in mem if cfg.is_strong(s))
            if weak > weak_cap_partial:
                return False
            if complete and mem:
                if weak > cfg.max_weak_ratio * len(mem) + 1e-9:
                    return False
                if weak > 0 and strong == 0:
                    return False
            return True
        return fn

    for g in groups:
        add(f"ability_mix(G{g})", "ability_mix", member_vars, make_mix(g),
            f"G{g}: weak <= {cfg.max_weak_ratio:.0%}, weak needs a strong peer")

    def make_prereq(s: Student, g: int):
        mv, av = v_member(s.sid), v_act(g)

        def fn(a: Assignment) -> bool:
            if mv not in a or av not in a or a[mv] != g:
                return True
            return cfg.student_ready(s, act_by_id[a[av]])
        return fn

    def make_student_avail(s: Student, g: int):
        mv, sv = v_member(s.sid), v_slot(g)

        def fn(a: Assignment) -> bool:
            if mv not in a or sv not in a or a[mv] != g:
                return True
            return a[sv] in s.available_slots
        return fn

    for s in students:
        for g in groups:
            add(f"prerequisite({s.sid},G{g})", "prerequisite",
                (v_member(s.sid), v_act(g)), make_prereq(s, g),
                f"{s.sid} must satisfy prerequisites of G{g}'s activity")
            add(f"student_availability({s.sid},G{g})", "student_availability",
                (v_member(s.sid), v_slot(g)), make_student_avail(s, g),
                f"{s.sid} must be free in G{g}'s slot")

    def make_need(g: int):
        av = v_act(g)

        def fn(a: Assignment) -> bool:
            if av not in a or not all(v in a for v in member_vars):
                return True
            mem, _ = members_of(a, g)
            if not mem:
                return True
            concept = act_by_id[a[av]].concept
            avg = sum(s.mastery.get(concept, 0.0) for s in mem) / len(mem)
            return avg <= cfg.need_ceiling
        return fn

    for g in groups:
        add(f"activity_need(G{g})", "activity_need", member_vars + [v_act(g)],
            make_need(g), f"G{g}: mean mastery of target concept <= {cfg.need_ceiling}")

    def make_ta_avail(g: int):
        tv, sv = v_ta(g), v_slot(g)

        def fn(a: Assignment) -> bool:
            if tv not in a or sv not in a:
                return True
            return a[sv] in ta_by_id[a[tv]].available_slots
        return fn

    def make_ta_expertise(g: int):
        tv, av = v_ta(g), v_act(g)

        def fn(a: Assignment) -> bool:
            if tv not in a or av not in a:
                return True
            exp = ta_by_id[a[tv]].expertise
            return (not exp) or act_by_id[a[av]].concept in exp
        return fn

    for g in groups:
        add(f"ta_availability(G{g})", "ta_availability", (v_ta(g), v_slot(g)),
            make_ta_avail(g), f"TA of G{g} must be free in G{g}'s slot")
        add(f"ta_expertise(G{g})", "ta_expertise", (v_ta(g), v_act(g)),
            make_ta_expertise(g), f"TA of G{g} must cover G{g}'s concept")

    def make_conflict(g1: int, g2: int):
        t1, t2, s1, s2 = v_ta(g1), v_ta(g2), v_slot(g1), v_slot(g2)

        def fn(a: Assignment) -> bool:
            if not all(v in a for v in (t1, t2, s1, s2)):
                return True
            return not (a[t1] == a[t2] and a[s1] == a[s2])
        return fn

    for g1 in groups:
        for g2 in groups:
            if g1 < g2:
                add(f"ta_conflict(G{g1},G{g2})", "ta_conflict",
                    (v_ta(g1), v_ta(g2), v_slot(g1), v_slot(g2)),
                    make_conflict(g1, g2), "same TA cannot be in two places at once")

    ta_vars = [v_ta(g) for g in groups]

    def make_workload(t: TeachingAssistant):
        def fn(a: Assignment) -> bool:
            return sum(1 for v in ta_vars if v in a and a[v] == t.tid) <= t.max_groups
        return fn

    for t in cfg.tas:
        add(f"ta_workload({t.tid})", "ta_workload", ta_vars, make_workload(t),
            f"{t.tid} supervises at most {t.max_groups} groups")

    if cfg.ta_balance_tolerance is not None:
        def balance(a: Assignment) -> bool:
            if not all(v in a for v in ta_vars):
                return True
            loads = [sum(1 for v in ta_vars if a[v] == t.tid) for t in cfg.tas]
            return max(loads) - min(loads) <= cfg.ta_balance_tolerance
        add("ta_balance", "ta_workload", ta_vars, balance,
            f"TA loads differ by at most {cfg.ta_balance_tolerance}")

    slot_vars = [v_slot(g) for g in groups]

    def make_room(slot: str):
        def fn(a: Assignment) -> bool:
            return sum(1 for v in slot_vars if v in a and a[v] == slot) \
                <= cfg.max_parallel_groups
        return fn

    for slot in cfg.slots:
        add(f"room_capacity({slot})", "room_capacity", slot_vars, make_room(slot),
            f"<= {cfg.max_parallel_groups} parallel groups in {slot}")

    return CSP(variables, domains, constraints)


def _drop(domains: Dict[str, List[Any]], var: str, value: Any,
          changed: List[str]) -> bool:
    d = domains[var]
    if value in d:
        domains[var] = [x for x in d if x != value]
        changed.append(var)
        return bool(domains[var])
    return True


def _force(domains: Dict[str, List[Any]], var: str, value: Any,
           changed: List[str]) -> None:
    if domains[var] != [value]:
        domains[var] = [value]
        changed.append(var)


def classroom_propagators(cfg: ClassroomConfig, k: int,
                          disabled: Optional[Set[str]] = None) -> List[Propagator]:
    """Global propagators; each only removes values no solution of its category can use.
    A one-value domain counts as forced."""
    disabled = disabled or set()
    students = cfg.students
    members = [v_member(s.sid) for s in students]
    groups = list(range(k))
    act_by_id = {a.aid: a for a in cfg.activities}
    weak_vars = [v for s, v in zip(students, members) if cfg.is_weak(s)]
    strong_vars = [v for s, v in zip(students, members) if cfg.is_strong(s)]
    # weak students need a strong peer in the group, so at most size-1 are weak
    weak_cap = min(int(math.floor(cfg.max_weak_ratio * cfg.max_group_size + 1e-9)),
                   cfg.max_group_size - 1)
    ta_vars = [v_ta(g) for g in groups]
    slot_vars = [v_slot(g) for g in groups]

    def group_size_propagator(assignment, domains):
        changed: List[str] = []
        for g in groups:
            forced = [v for v in members if domains[v] == [g]]
            possible = [v for v in members if g in domains[v]]
            if len(forced) > cfg.max_group_size or len(possible) < cfg.min_group_size:
                return False, changed
            if len(forced) == cfg.max_group_size:
                for v in possible:
                    if domains[v] != [g] and not _drop(domains, v, g, changed):
                        return False, changed
                possible = forced
            if len(possible) == cfg.min_group_size:
                for v in possible:
                    _force(domains, v, g, changed)
        return True, changed

    def ability_mix_propagator(assignment, domains):
        changed: List[str] = []
        for g in groups:
            size_possible = sum(1 for v in members if g in domains[v])
            weak_forced = [v for v in weak_vars if domains[v] == [g]]
            strong_possible = [v for v in strong_vars if g in domains[v]]
            if len(weak_forced) > weak_cap:
                return False, changed
            if weak_forced:
                if (not strong_possible or
                        len(weak_forced) > cfg.max_weak_ratio * size_possible + 1e-9):
                    return False, changed
                if len(strong_possible) == 1:
                    _force(domains, strong_possible[0], g, changed)
            if len(weak_forced) >= weak_cap or not strong_possible:
                for v in weak_vars:
                    if domains[v] != [g] and not _drop(domains, v, g, changed):
                        return False, changed
        return True, changed

    def activity_need_propagator(assignment, domains):
        changed: List[str] = []
        for g in groups:
            av = v_act(g)
            forced = [i for i, v in enumerate(members) if domains[v] == [g]]
            cand = [i for i, v in enumerate(members)
                    if g in domains[v] and domains[v] != [g]]
            lo = max(cfg.min_group_size, len(forced))
            hi = min(cfg.max_group_size, len(forced) + len(cand))
            if lo > hi:
                continue
            keep = []
            for aid in domains[av]:
                concept = act_by_id[aid].concept
                base = sum(students[i].mastery.get(concept, 0.0) for i in forced)
                extra = sorted(students[i].mastery.get(concept, 0.0) for i in cand)
                best = min((base + sum(extra[:m - len(forced)])) / m
                           for m in range(lo, hi + 1))
                if best <= cfg.need_ceiling + 1e-9:
                    keep.append(aid)
            if len(keep) < len(domains[av]):
                domains[av] = keep
                changed.append(av)
                if not keep:
                    return False, changed
        return True, changed

    def _cap_values(domains, variables, value, cap, changed):
        forced = [v for v in variables if domains[v] == [value]]
        if len(forced) > cap:
            return False
        if len(forced) == cap:
            for v in variables:
                if domains[v] != [value] and not _drop(domains, v, value, changed):
                    return False
        return True

    def ta_workload_propagator(assignment, domains):
        changed: List[str] = []
        for t in cfg.tas:
            if not _cap_values(domains, ta_vars, t.tid, t.max_groups, changed):
                return False, changed
        return True, changed

    def room_capacity_propagator(assignment, domains):
        changed: List[str] = []
        for s in cfg.slots:
            if not _cap_values(domains, slot_vars, s, cfg.max_parallel_groups, changed):
                return False, changed
        return True, changed

    def ta_conflict_propagator(assignment, domains):
        changed: List[str] = []
        for g in groups:
            if len(domains[v_ta(g)]) != 1 or len(domains[v_slot(g)]) != 1:
                continue
            ta, slot = domains[v_ta(g)][0], domains[v_slot(g)][0]
            for h in groups:
                if h == g:
                    continue
                if domains[v_ta(h)] == [ta]:
                    if not _drop(domains, v_slot(h), slot, changed):
                        return False, changed
                elif domains[v_slot(h)] == [slot]:
                    if not _drop(domains, v_ta(h), ta, changed):
                        return False, changed
        return True, changed

    by_category = [
        ("group_size", group_size_propagator),
        ("ability_mix", ability_mix_propagator),
        ("activity_need", activity_need_propagator),
        ("ta_workload", ta_workload_propagator),
        ("room_capacity", room_capacity_propagator),
        ("ta_conflict", ta_conflict_propagator),
    ]
    return [p for cat, p in by_category if cat not in disabled]


@dataclass
class SolveResult:
    feasible: bool
    num_groups: Optional[int]
    solution: Optional[Dict[str, Any]]
    nodes: int
    timed_out: bool                   # some group count was not decided within the limits
    explanation: List[str]
    status: str = "UNKNOWN"           # SOLVED | INFEASIBLE | UNKNOWN | INVALID_INPUT
    k_status: Dict[int, str] = field(default_factory=dict)   # status per group count tried


def extract_solution(cfg: ClassroomConfig, k: int, a: Assignment) -> Dict[str, Any]:
    groups = {f"G{g}": {"slot": a[v_slot(g)], "activity": a[v_act(g)],
                        "ta": a[v_ta(g)], "students": []} for g in range(k)}
    for s in cfg.students:
        groups[f"G{a[v_member(s.sid)]}"]["students"].append(s.sid)
    return groups


def validate_schedule(cfg: ClassroomConfig, solution: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    students = {s.sid: s for s in cfg.students}
    acts = {a.aid: a for a in cfg.activities}
    tas = {t.tid: t for t in cfg.tas}
    placed: Dict[str, int] = {}
    ta_load = {t.tid: 0 for t in cfg.tas}
    ta_slots: Dict[Tuple[str, str], int] = {}
    slot_load: Dict[str, int] = {}

    for name, g in solution.items():
        slot, act_id, ta_id = g.get("slot"), g.get("activity"), g.get("ta")
        ids = list(g.get("students", []))
        known = True
        if slot not in cfg.slots:
            errors.append(f"[slot] {name}: unknown slot {slot!r}.")
            known = False
        if act_id not in acts:
            errors.append(f"[activity] {name}: unknown activity {act_id!r}.")
            known = False
        if ta_id not in tas:
            errors.append(f"[ta] {name}: unknown TA {ta_id!r}.")
            known = False
        for sid in ids:
            if sid not in students:
                errors.append(f"[students] {name}: unknown student {sid!r}.")
                known = False
            placed[sid] = placed.get(sid, 0) + 1

        if not cfg.min_group_size <= len(ids) <= cfg.max_group_size:
            errors.append(f"[group_size] {name}: {len(ids)} students, allowed "
                          f"{cfg.min_group_size}..{cfg.max_group_size}.")
        if not known:
            continue

        mem = [students[sid] for sid in ids]
        act, ta = acts[act_id], tas[ta_id]
        weak = sum(1 for s in mem if cfg.is_weak(s))
        strong = sum(1 for s in mem if cfg.is_strong(s))
        if mem and weak > cfg.max_weak_ratio * len(mem) + 1e-9:
            errors.append(f"[ability_mix] {name}: {weak} weak of {len(mem)} exceeds "
                          f"ratio {cfg.max_weak_ratio}.")
        if weak > 0 and strong == 0:
            errors.append(f"[ability_mix] {name}: weak students without a strong peer.")
        for s in mem:
            missing = cfg.missing_prereqs(s, act)
            if missing:
                errors.append(f"[prerequisite] {name}: {s.sid} lacks {missing} for "
                              f"'{act.name}'.")
            if slot not in s.available_slots:
                errors.append(f"[student_availability] {name}: {s.sid} is not "
                              f"available in {slot}.")
        if mem:
            mean = sum(s.mastery.get(act.concept, 0.0) for s in mem) / len(mem)
            if mean > cfg.need_ceiling:
                errors.append(f"[activity_need] {name}: mean mastery {mean:.2f} of "
                              f"'{act.concept}' exceeds {cfg.need_ceiling}.")
        if slot not in ta.available_slots:
            errors.append(f"[ta_availability] {name}: {ta.tid} is not available "
                          f"in {slot}.")
        if ta.expertise and act.concept not in ta.expertise:
            errors.append(f"[ta_expertise] {name}: {ta.tid} lacks expertise in "
                          f"'{act.concept}'.")
        ta_load[ta.tid] += 1
        ta_slots[(ta.tid, slot)] = ta_slots.get((ta.tid, slot), 0) + 1
        slot_load[slot] = slot_load.get(slot, 0) + 1

    for sid in students:
        if placed.get(sid, 0) != 1:
            errors.append(f"[assignment] {sid} is placed in {placed.get(sid, 0)} "
                          f"groups (expected exactly 1).")
    for (tid, slot), cnt in sorted(ta_slots.items()):
        if cnt > 1:
            errors.append(f"[ta_conflict] {tid} supervises {cnt} groups in {slot}.")
    for t in cfg.tas:
        if ta_load[t.tid] > t.max_groups:
            errors.append(f"[ta_workload] {t.tid} supervises {ta_load[t.tid]} groups, "
                          f"cap {t.max_groups}.")
    if cfg.ta_balance_tolerance is not None and cfg.tas:
        spread = max(ta_load.values()) - min(ta_load.values())
        if spread > cfg.ta_balance_tolerance:
            errors.append(f"[ta_workload] TA load spread {spread} exceeds "
                          f"{cfg.ta_balance_tolerance}.")
    for slot, cnt in sorted(slot_load.items()):
        if cnt > cfg.max_parallel_groups:
            errors.append(f"[room_capacity] {cnt} groups in {slot}, rooms "
                          f"{cfg.max_parallel_groups}.")
    return errors


def verify_assignment(cfg: ClassroomConfig, k: int, a: Assignment) -> List[str]:
    names = [v_member(s.sid) for s in cfg.students]
    for g in range(k):
        names += [v_slot(g), v_act(g), v_ta(g)]
    missing = [v for v in names if v not in a]
    if missing:
        return [f"incomplete assignment: {v} is unassigned." for v in missing]
    bad = [v_member(s.sid) for s in cfg.students if a[v_member(s.sid)] not in range(k)]
    if bad:
        return [f"{v} has a group index outside 0..{k - 1}." for v in bad]
    return validate_schedule(cfg, extract_solution(cfg, k, a))


def _solver_for(cfg: ClassroomConfig, k: int, max_nodes: int,
                time_limit: Optional[float],
                disabled: Optional[Set[str]] = None) -> CSPSolver:
    return CSPSolver(build_csp(cfg, k, disabled), var_order="mrv_degree",
                     propagation="mac", lcv=True, max_nodes=max_nodes,
                     time_limit=time_limit,
                     propagators=classroom_propagators(cfg, k, disabled))


def solve_classroom(cfg: ClassroomConfig, max_nodes: int = 200_000,
                    explain: bool = True,
                    time_limit: Optional[float] = None) -> SolveResult:
    """Smallest feasible group count wins. max_nodes/time_limit apply per group count."""
    problems = validate_config(cfg)
    if problems:
        return SolveResult(False, None, None, 0, False,
                           ["Invalid configuration:"] + [f"  {p}" for p in problems],
                           "INVALID_INPUT")

    n = len(cfg.students)
    lo = max(1, math.ceil(n / cfg.max_group_size))
    hi = n // cfg.min_group_size
    if lo > hi:
        msg = (f"[group_size] {n} students cannot be split into groups of "
               f"{cfg.min_group_size}..{cfg.max_group_size}.")
        return SolveResult(False, None, None, 0, False, [msg], "INFEASIBLE")

    total_nodes = 0
    k_status: Dict[int, str] = {}
    notes: Dict[int, List[str]] = {}
    for k in range(lo, hi + 1):
        static = static_diagnosis(cfg, k)
        if static:
            k_status[k] = "INFEASIBLE"
            notes[k] = static
            continue
        solver = _solver_for(cfg, k, max_nodes, time_limit)
        assignment = solver.solve()
        total_nodes += solver.stats.nodes
        k_status[k] = solver.status
        if assignment is not None:
            solution = extract_solution(cfg, k, assignment)
            violations = validate_schedule(cfg, solution)
            if violations:
                raise RuntimeError("solver returned a schedule that fails validation: "
                                   + "; ".join(violations))
            return SolveResult(True, k, solution, total_nodes, False, [],
                               "SOLVED", k_status)
        if solver.status == "INFEASIBLE" and explain:
            notes[k] = _search_proof_lines(cfg, k, solver, max_nodes, time_limit)
        elif solver.status == "UNKNOWN":
            notes[k] = [_inconclusive_line(k)]

    undecided = any(s == "UNKNOWN" for s in k_status.values())
    explanation: List[str] = []
    if explain:
        for k in sorted(notes):
            title = "inconclusive" if k_status[k] == "UNKNOWN" else "infeasible"
            explanation.append(f"--- {k} groups: {title} ---")
            explanation += notes[k]
    return SolveResult(False, None, None, total_nodes, undecided, explanation,
                       "UNKNOWN" if undecided else "INFEASIBLE", k_status)


def static_diagnosis(cfg: ClassroomConfig, k: int) -> List[str]:
    """Necessary conditions: any message returned proves k groups infeasible."""
    msgs: List[str] = []
    n = len(cfg.students)
    slots = set(cfg.slots)
    ta_slots = {t.tid: t.available_slots & slots for t in cfg.tas}

    if k * cfg.min_group_size > n or k * cfg.max_group_size < n:
        msgs.append(f"[group_size] {n} students cannot fill {k} groups of "
                    f"{cfg.min_group_size}..{cfg.max_group_size}.")

    cap = sum(t.max_groups for t in cfg.tas)
    distinct_cap = sum(min(t.max_groups, len(ta_slots[t.tid])) for t in cfg.tas)
    if k > cap:
        msgs.append(f"[ta_workload] {k} groups need a TA each but total TA capacity "
                    f"is {cap} groups.")
    elif k > distinct_cap:
        msgs.append(f"[ta_conflict] a TA cannot run two groups in one slot, so the "
                    f"TAs can supervise at most {distinct_cap} groups (workload "
                    f"cap and available slots); {k} are needed.")

    room_cap = cfg.max_parallel_groups * len(cfg.slots)
    slot_cap = sum(min(cfg.max_parallel_groups,
                       sum(1 for t in cfg.tas if s in ta_slots[t.tid]))
                   for s in cfg.slots)
    if k > room_cap:
        msgs.append(f"[room_capacity] {k} groups exceed room capacity "
                    f"({cfg.max_parallel_groups} per slot x {len(cfg.slots)} slots).")
    elif k > slot_cap:
        msgs.append(f"[room_capacity] with rooms and TA availability, at most "
                    f"{slot_cap} groups can run across all slots; {k} are needed.")

    viable = {s for s in cfg.slots if any(s in ta_slots[t.tid] for t in cfg.tas)}
    for st in cfg.students:
        free = st.available_slots & viable
        if not free:
            msgs.append(f"[student_availability] {st.sid} is never free when a TA is "
                        f"available.")
            continue
        ready = [a for a in cfg.activities if cfg.student_ready(st, a)]
        if not ready:
            if cfg.activities:
                best = min(cfg.activities,
                           key=lambda a: len(cfg.missing_prereqs(st, a)))
                msgs.append(f"[prerequisite] {st.sid} satisfies the prerequisites of no "
                            f"activity (closest: '{best.name}', missing "
                            f"{cfg.missing_prereqs(st, best)}).")
            else:
                msgs.append("[prerequisite] there are no activities to assign.")
            continue
        fits = any(free & ta_slots[t.tid] and (not t.expertise or a.concept in t.expertise)
                   for a in ready for t in cfg.tas)
        if not fits:
            msgs.append(f"[prerequisite+ta_expertise+ta_availability+student_availability] "
                        f"no combination of activity, TA and time slot works for "
                        f"{st.sid}: ready for {[a.aid for a in ready]}, free in "
                        f"{sorted(free)}.")

    weak = [s for s in cfg.students if cfg.is_weak(s)]
    strong = [s for s in cfg.students if cfg.is_strong(s)]
    if weak:
        cap_weak = min(int(math.floor(cfg.max_weak_ratio * cfg.max_group_size + 1e-9)),
                       cfg.max_group_size - 1)
        if not strong:
            msgs.append("[ability_mix] there are weak students but no strong peers.")
        elif cap_weak == 0:
            msgs.append("[ability_mix] no group may contain a weak student under the "
                        "current ratio and group size limits.")
        else:
            if len(weak) > k * cap_weak:
                msgs.append(f"[ability_mix] {len(weak)} weak students exceed what {k} "
                            f"groups can hold ({cap_weak} weak per group).")
            need_groups = math.ceil(len(weak) / cap_weak)
            if len(strong) < need_groups:
                msgs.append(f"[ability_mix] {len(weak)} weak students need at least "
                            f"{need_groups} groups, each with its own strong peer, "
                            f"but there are {len(strong)} strong students.")
    return msgs


def _inconclusive_line(k: int) -> str:
    return (f"search for {k} groups stopped at the node/time limit before finishing; "
            f"infeasibility is not proven (raise max_nodes or time_limit).")


def _conflict_core(cfg: ClassroomConfig, k: int, max_nodes: int,
                   time_limit: Optional[float]) -> Tuple[List[str], bool]:
    active = list(ALL_CATEGORIES)
    minimal = True
    for cat in ALL_CATEGORIES:
        trial = [c for c in active if c != cat]
        solver = _solver_for(cfg, k, max_nodes, time_limit,
                             disabled=set(ALL_CATEGORIES) - set(trial))
        solver.solve()
        if solver.status == "INFEASIBLE":
            active = trial
        elif solver.status == "UNKNOWN":
            minimal = False
    return active, minimal


def _search_proof_lines(cfg: ClassroomConfig, k: int, solver: CSPSolver,
                        max_nodes: int, time_limit: Optional[float]) -> List[str]:
    lines = [f"exhaustive search found no valid arrangement for {k} groups."]
    if solver.stats.root_failure:
        lines.append(f"Propagation: {solver.stats.root_failure}.")
    core, minimal = _conflict_core(cfg, k, max_nodes, time_limit)
    if minimal:
        lines.append("Conflicting constraint categories (with only these active the "
                     "problem is infeasible; removing any one of them makes it "
                     "satisfiable):")
    else:
        lines.append("Conflicting constraint categories (infeasible with only these "
                     "active; some relaxations were inconclusive, so the set may "
                     "not be minimal):")
    lines += [f"  * [{cat}] {CATEGORY_INFO[cat]}" for cat in core]
    return lines


def explain_infeasibility(cfg: ClassroomConfig, k: int, max_nodes: int = 50_000,
                          time_limit: Optional[float] = None) -> List[str]:
    lines = static_diagnosis(cfg, k)
    if lines:
        return lines
    solver = _solver_for(cfg, k, max_nodes, time_limit)
    if solver.solve() is not None:
        return ["a valid arrangement exists for this group count."]
    if solver.status == "UNKNOWN":
        return [_inconclusive_line(k)]
    return _search_proof_lines(cfg, k, solver, max_nodes, time_limit)


def format_result(cfg: ClassroomConfig, res: SolveResult) -> str:
    out: List[str] = []
    if not res.feasible:
        out.append("NO VALID CLASSROOM ARRANGEMENT FOUND")
        out.append(f"Status: {res.status}"
                   + (" (proven)" if res.status == "INFEASIBLE" else ""))
        out += res.explanation
        return "\n".join(out)

    sm = {s.sid: s for s in cfg.students}
    am = {a.aid: a for a in cfg.activities}
    out.append(f"Valid arrangement with {res.num_groups} groups "
               f"({res.nodes} search nodes)")
    for gname, g in res.solution.items():
        act = am[g["activity"]]
        abilities = [sm[x].ability for x in g["students"]]
        tags = []
        for x in g["students"]:
            tag = "W" if cfg.is_weak(sm[x]) else ("S" if cfg.is_strong(sm[x]) else "M")
            tags.append(f"{x}({tag})")
        out.append(f"{gname}: slot={g['slot']}  TA={g['ta']}  "
                   f"activity={act.name} [{act.kind}, {act.concept}]")
        out.append(f"     students: {', '.join(tags)}  "
                   f"mean ability={sum(abilities) / len(abilities):.2f}")
    return "\n".join(out)


def make_demo_classroom(n_students: int = 16, seed: int = 7) -> ClassroomConfig:
    rng = random.Random(seed)
    slots = ["Mon-10", "Tue-10", "Wed-10", "Thu-10", "Fri-10"]
    concept_prereqs = {
        "queues": ["arrays"],
        "graphs_basics": ["arrays"],
        "priority_queues": ["queues"],
        "heuristics": ["graphs_basics"],
        "path_cost": ["graphs_basics"],
        "a_star": ["heuristics", "path_cost", "priority_queues"],
    }
    offsets = {"arrays": 0.25, "queues": 0.15, "graphs_basics": 0.15,
               "priority_queues": 0.05, "heuristics": 0.0, "path_cost": 0.0,
               "a_star": -0.15}

    def clamp(x: float) -> float:
        return round(min(1.0, max(0.0, x)), 2)

    students = []
    for i in range(n_students):
        ability = clamp(rng.gauss(0.55, 0.2))
        mastery = {c: clamp(ability + off + rng.gauss(0, 0.08))
                   for c, off in offsets.items()}
        avail = {s for s in slots if rng.random() < 0.9}
        if len(avail) < 2:
            avail = set(rng.sample(slots, 2))
        students.append(Student(f"S{i + 1:02d}", ability, mastery, avail))

    tas = [
        TeachingAssistant("TA1", {"Mon-10", "Tue-10", "Wed-10", "Thu-10"}, 2,
                          {"arrays", "queues", "priority_queues"}),
        TeachingAssistant("TA2", {"Tue-10", "Wed-10", "Thu-10", "Fri-10"}, 2,
                          {"graphs_basics", "heuristics", "path_cost", "a_star"}),
        TeachingAssistant("TA3", set(slots), 2),
    ]
    activities = [
        Activity("act_arrays", "Arrays refresher", "revision", "arrays"),
        Activity("act_queues", "Queues revision", "revision", "queues"),
        Activity("act_graph", "Graph basics revision", "revision", "graphs_basics"),
        Activity("act_pq", "Priority queue exercise", "exercise", "priority_queues"),
        Activity("act_heur", "Heuristics discussion", "discussion", "heuristics"),
        Activity("act_cost", "Path cost vs heuristic peer task", "peer", "path_cost"),
        Activity("act_astar", "A* worked exercise", "exercise", "a_star"),
    ]
    return ClassroomConfig(students, tas, activities, slots, concept_prereqs,
                           min_group_size=3, max_group_size=5,
                           max_parallel_groups=2)


if __name__ == "__main__":
    cfg = make_demo_classroom()

    print("=== Demo 1: normal classroom ===")
    res = solve_classroom(cfg)
    print(format_result(cfg, res))

    print("\n=== Demo 2: overloaded TAs (each TA may supervise only 1 group) ===")
    bad = copy.deepcopy(cfg)
    for t in bad.tas:
        t.max_groups = 1
    print(format_result(bad, solve_classroom(bad)))

    print("\n=== Demo 3: no revision of arrays, and S01 has weak prerequisites ===")
    bad2 = copy.deepcopy(cfg)
    bad2.activities = [a for a in bad2.activities if a.aid != "act_arrays"]
    bad2.students[0].mastery = {c: 0.1 for c in bad2.students[0].mastery}
    print(format_result(bad2, solve_classroom(bad2)))

    print("\n=== Demo 4: TA loads must be exactly equal (needs a multiple of 3 groups) ===")
    bad3 = copy.deepcopy(cfg)
    bad3.ta_balance_tolerance = 0
    print(format_result(bad3, solve_classroom(bad3)))
