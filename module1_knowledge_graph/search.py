"""
Graph search: prerequisite tracing, study plans, and the A* family.

  bfs_ancestors / dfs_ancestors   trace prerequisite dependencies
  missing_prerequisites           study plan: unmastered prerequisites, foundations first
  search                          generic weighted search: A*, greedy, uniform-cost,
                                  and a FIFO-frontier variant that models a common student error
  is_admissible                   checks a heuristic against true costs (used in tests)

Standard library only.
"""
import heapq
import itertools
from collections import deque


def bfs_ancestors(prereqs, start):
    """BFS over prerequisite edges. Returns {concept: depth}, start at depth 0.
    Depth is the fewest prerequisite steps from the start."""
    depth = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for p in prereqs[cur]:
            if p not in depth:
                depth[p] = depth[cur] + 1
                queue.append(p)
    return depth


def dfs_ancestors(prereqs, start):
    """DFS over prerequisite edges. Returns the set of all ancestors (excluding start)."""
    seen = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        for p in prereqs[cur]:
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def missing_prerequisites(prereqs, target, mastered):
    """Unmastered prerequisites of target. The BFS does not expand past a mastered
    concept, so a student's known foundations cut off their own prerequisite trees.
    Returned in prerequisite-first order, so foundations are studied before the concepts built on them."""
    mastered = set(mastered)
    if target in mastered:
        return []
    depth = {target: 0}
    missing = {}
    queue = deque([target])
    while queue:
        cur = queue.popleft()
        for p in prereqs[cur]:
            if p in depth or p in mastered:
                continue
            depth[p] = depth[cur] + 1
            missing[p] = depth[p]
            queue.append(p)
    # Put the set in prerequisite-first order (depth alone is not a valid learning order).
    missing_set = set(missing)
    order, seen = [], set()

    def visit(c):
        seen.add(c)
        for p in prereqs[c]:
            if p in missing_set and p not in seen:
                visit(p)
        order.append(c)

    for c in sorted(missing_set, key=lambda c: (-missing[c], c)):
        if c not in seen:
            visit(c)
    return order


def search(adj, start, goal, h=None, key="f", frontier="priority"):
    """
    Generic search over a weighted directed graph.

      adj       {node: [(neighbour, edge_cost), ...]}
      h         {node: estimated remaining cost to goal}; missing nodes count as 0
      key       "f" -> g + h   (A*)
                "h" -> h only  (greedy best-first)
                "g" -> g only  (uniform-cost search)
      frontier  "priority" -> heap ordered by key; goal tested on expansion (correct)
                "fifo"     -> plain queue; cost is ignored when choosing what to expand
                              (a common student misconception about priority queues)

    Returns {"found", "path", "cost", "expanded"}.
    """
    h = h or {}

    def score(g, node):
        hv = h.get(node, 0)
        return {"f": g + hv, "h": hv, "g": g}[key]

    expanded = 0
    if frontier == "priority":
        counter = itertools.count()
        heap = [(score(0, start), next(counter), 0, start, (start,))]
        best_g = {start: 0}
        while heap:
            _, _, g, node, path = heapq.heappop(heap)
            if g > best_g[node]:
                continue  # stale entry, a cheaper route was already found
            if node == goal:
                return {"found": True, "path": path, "cost": g, "expanded": expanded}
            expanded += 1
            for nbr, cost in adj.get(node, []):
                ng = g + cost
                if ng < best_g.get(nbr, float("inf")):
                    best_g[nbr] = ng
                    heapq.heappush(heap, (score(ng, nbr), next(counter), ng, nbr, path + (nbr,)))
    elif frontier == "fifo":
        queue = deque([(0, start, (start,))])
        while queue:
            g, node, path = queue.popleft()
            if node == goal:
                return {"found": True, "path": path, "cost": g, "expanded": expanded}
            expanded += 1
            for nbr, cost in adj.get(node, []):
                if nbr not in path:  # avoid cycles, but no cost-based ordering
                    queue.append((g + cost, nbr, path + (nbr,)))
    else:
        raise ValueError(f"Unknown frontier: {frontier}")
    return {"found": False, "path": None, "cost": None, "expanded": expanded}


def a_star(adj, start, goal, h):
    return search(adj, start, goal, h=h, key="f", frontier="priority")


def greedy_best_first(adj, start, goal, h):
    return search(adj, start, goal, h=h, key="h", frontier="priority")


def uniform_cost(adj, start, goal):
    return search(adj, start, goal, h=None, key="g", frontier="priority")


def true_costs_to_goal(adj, goal):
    """Exact remaining cost from every node to goal (Bellman-Ford style relaxation)."""
    nodes = set(adj) | {v for edges in adj.values() for v, _ in edges}
    dist = {n: float("inf") for n in nodes}
    dist[goal] = 0
    changed = True
    while changed:
        changed = False
        for u, edges in adj.items():
            for v, cost in edges:
                if dist[v] + cost < dist[u]:
                    dist[u] = dist[v] + cost
                    changed = True
    return dist


def is_admissible(adj, goal, h):
    """True if h never overestimates the true remaining cost."""
    true = true_costs_to_goal(adj, goal)
    return all(h.get(n, 0) <= true[n] for n in true)
