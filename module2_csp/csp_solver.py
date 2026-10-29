

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from classroom_csp import CSP, Constraint

Assignment = Dict[str, Any]
Domains = Dict[str, List[Any]]
Propagator = Callable[[Assignment, Domains], Tuple[bool, List[str]]]


def _bump(table: Dict[str, int], key: str, amount: int = 1) -> None:
    table[key] = table.get(key, 0) + amount


def _total_size(domains: Domains) -> int:
    return sum(len(d) for d in domains.values())


@dataclass
class SolverStats:
    nodes: int = 0
    backtracks: int = 0
    prunings: int = 0
    revisions: int = 0
    elapsed: float = 0.0
    timed_out: bool = False
    conflicts: Dict[str, int] = field(default_factory=dict)
    wipeouts: Dict[str, int] = field(default_factory=dict)
    propagator_failures: Dict[str, int] = field(default_factory=dict)
    root_failure: Optional[str] = None

    def top(self, table: Dict[str, int], n: int = 3) -> List[Tuple[str, int]]:
        return sorted(table.items(), key=lambda kv: -kv[1])[:n]


class CSPSolver:
    VAR_ORDERS = ("static", "mrv", "mrv_degree")
    PROPAGATION = ("none", "fc", "mac")

    def __init__(self, csp: CSP, var_order: str = "mrv_degree",
                 propagation: str = "mac", lcv: bool = True,
                 max_nodes: int = 200_000, time_limit: Optional[float] = None,
                 propagators: Optional[List[Propagator]] = None,
                 seed: Optional[int] = None):
        if var_order not in self.VAR_ORDERS:
            raise ValueError(f"var_order must be one of {self.VAR_ORDERS}")
        if propagation not in self.PROPAGATION:
            raise ValueError(f"propagation must be one of {self.PROPAGATION}")
        self.csp = csp
        self.var_order = var_order
        self.propagation = propagation
        self.lcv = lcv
        self.max_nodes = max_nodes
        self.time_limit = time_limit
        self.propagators = list(propagators or [])
        self.rng = random.Random(seed) if seed is not None else None
        self.nvars = len(csp.variables)
        self.stats = SolverStats()
        self.status = "UNKNOWN"
        self._t0 = 0.0
        self._at_root = False

        self.arcs: Dict[str, List[Tuple[Constraint, str]]] = {v: [] for v in csp.variables}
        for c in csp.constraints:
            if len(c.scope) == 2 and c.scope[0] != c.scope[1]:
                x, y = c.scope
                self.arcs[x].append((c, y))
                self.arcs[y].append((c, x))

    def solve(self) -> Optional[Assignment]:
        self.stats = SolverStats()
        self.status = "UNKNOWN"
        self._t0 = time.perf_counter()
        domains: Domains = {v: list(d) for v, d in self.csp.domains.items()}
        if self.rng is not None:
            for v in domains:
                self.rng.shuffle(domains[v])
        try:
            if not self._root_consistent(domains):
                self.status = "INFEASIBLE"
                return None
            result = self._search({}, domains)
            if result is not None:
                self.status = "SOLVED"
            elif self.stats.timed_out:
                self.status = "UNKNOWN"
            else:
                self.status = "INFEASIBLE"
            return result
        finally:
            self.stats.elapsed = time.perf_counter() - self._t0

    def _root_consistent(self, domains: Domains) -> bool:
        empty = [v for v, d in domains.items() if not d]
        if empty:
            self.stats.root_failure = f"variable {empty[0]} has an empty domain"
            return False
        if self.propagation == "none":
            return True
        self._at_root = True
        try:
            return self._fixpoint({}, domains, None)
        finally:
            self._at_root = False

    def _limit_hit(self) -> bool:
        if self.stats.nodes >= self.max_nodes:
            self.stats.timed_out = True
        elif (self.time_limit is not None
              and time.perf_counter() - self._t0 > self.time_limit):
            self.stats.timed_out = True
        return self.stats.timed_out

    def _search(self, assignment: Assignment, domains: Domains) -> Optional[Assignment]:
        if len(assignment) == self.nvars:
            return dict(assignment)
        var = self._select_var(assignment, domains)
        for val in self._order_values(var, assignment, domains):
            if self._limit_hit():
                return None
            self.stats.nodes += 1
            assignment[var] = val
            child = dict(domains)
            child[var] = [val]
            bad = self._first_violated(var, assignment)
            if bad is None:
                if self._propagate(var, assignment, child):
                    res = self._search(assignment, child)
                    if res is not None:
                        return res
            else:
                _bump(self.stats.conflicts, bad)
            del assignment[var]
            self.stats.backtracks += 1
        return None

    def _first_violated(self, var: str, assignment: Assignment) -> Optional[str]:
        for c in self.csp.constraints_of[var]:
            if not c.fn(assignment):
                return c.name
        return None

    def _select_var(self, assignment: Assignment, domains: Domains) -> str:
        unassigned = [v for v in self.csp.variables if v not in assignment]
        if self.var_order == "static":
            return unassigned[0]
        smallest = min(len(domains[v]) for v in unassigned)
        ties = [v for v in unassigned if len(domains[v]) == smallest]
        if self.var_order == "mrv" or len(ties) == 1:
            return ties[0]
        return max(ties, key=lambda v: self._open_degree(v, assignment))

    def _open_degree(self, var: str, assignment: Assignment) -> int:
        return sum(1 for c in self.csp.constraints_of[var]
                   if any(u != var and u not in assignment for u in c.scope))

    def _order_values(self, var: str, assignment: Assignment,
                      domains: Domains) -> List[Any]:
        vals = list(domains[var])
        if not self.lcv or len(vals) <= 1:
            return vals

        def cost(val: Any) -> int:
            total = 0
            for c, other in self.arcs[var]:
                if other in assignment:
                    continue
                for vo in domains[other]:
                    if not c.fn({var: val, other: vo}):
                        total += 1
            return total

        return sorted(vals, key=cost)

    def _record_wipeout(self, var: str, c: Constraint) -> None:
        _bump(self.stats.wipeouts, c.name)
        if self._at_root and self.stats.root_failure is None:
            self.stats.root_failure = (
                f"propagation at the root emptied the domain of {var} "
                f"through constraint {c.name}")

    def _propagate(self, var: str, assignment: Assignment, domains: Domains) -> bool:
        if self.propagation == "none":
            return True
        changed = self._forward_check(var, assignment, domains)
        if changed is None:
            return False
        return self._fixpoint(assignment, domains, [var] + changed)

    def _forward_check(self, var: str, assignment: Assignment,
                       domains: Domains) -> Optional[List[str]]:
        changed: List[str] = []
        for c in self.csp.constraints_of[var]:
            unassigned = [v for v in c.scope if v not in assignment]
            if len(unassigned) != 1:
                continue
            u = unassigned[0]
            kept = []
            for val in domains[u]:
                assignment[u] = val
                ok = c.fn(assignment)
                del assignment[u]
                if ok:
                    kept.append(val)
            if len(kept) < len(domains[u]):
                self.stats.prunings += len(domains[u]) - len(kept)
                domains[u] = kept
                changed.append(u)
                if not kept:
                    self._record_wipeout(u, c)
                    return None
        return changed

    def _fixpoint(self, assignment: Assignment, domains: Domains,
                  seeds: Optional[List[str]]) -> bool:
        while True:
            before = _total_size(domains)
            touched: List[str] = []
            for p in self.propagators:
                ok, ch = p(assignment, domains)
                if not ok:
                    name = getattr(p, "__name__", "propagator")
                    _bump(self.stats.propagator_failures, name)
                    if self._at_root and self.stats.root_failure is None:
                        self.stats.root_failure = f"propagator {name} failed at the root"
                    return False
                touched.extend(ch)
            self.stats.prunings += before - _total_size(domains)
            if self.propagation == "mac":
                ac_seeds = None if seeds is None else seeds + touched
                if not self._ac3(assignment, domains, ac_seeds):
                    return False
            if _total_size(domains) == before:
                return True
            seeds = []

    def _ac3(self, assignment: Assignment, domains: Domains,
             seeds: Optional[List[str]]) -> bool:
        queue: Deque[Tuple[Constraint, str, str]] = deque()
        queued = set()

        def push(c: Constraint, x: str, y: str) -> None:
            key = (id(c), x, y)
            if key not in queued:
                queued.add(key)
                queue.append((c, x, y))

        if seeds is None:
            for y in self.csp.variables:
                for c, x in self.arcs[y]:
                    push(c, x, y)
        else:
            for y in dict.fromkeys(seeds):
                for c, x in self.arcs[y]:
                    if x not in assignment:
                        push(c, x, y)

        while queue:
            c, x, y = queue.popleft()
            queued.discard((id(c), x, y))
            if x in assignment:
                continue
            dx, dy = domains[x], domains[y]
            self.stats.revisions += 1
            new = [vx for vx in dx if any(c.fn({x: vx, y: vy}) for vy in dy)]
            if len(new) < len(dx):
                self.stats.prunings += len(dx) - len(new)
                domains[x] = new
                if not new:
                    self._record_wipeout(x, c)
                    return False
                for c2, w in self.arcs[x]:
                    if c2 is not c and w not in assignment:
                        push(c2, w, x)
        return True
