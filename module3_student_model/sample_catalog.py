"""Sample A* syllabus used by the demo and tests."""

from shared.schema import Concept, CourseCatalog, Misconception


def astar_catalog() -> CourseCatalog:
    concepts = [
        Concept(id="graph_search", name="Graph search basics"),
        Concept(id="priority_queue", name="Priority queues", prerequisites=["graph_search"]),
        Concept(id="path_cost", name="Path cost g(n)", prerequisites=["graph_search"]),
        Concept(id="heuristic", name="Heuristic h(n)", prerequisites=["graph_search"]),
        Concept(id="admissibility", name="Admissible heuristics", prerequisites=["heuristic"]),
        Concept(
            id="astar",
            name="A* search",
            prerequisites=["priority_queue", "path_cost", "heuristic", "admissibility"],
        ),
    ]
    misconceptions = [
        Misconception(
            id="pq_as_stack",
            concept_id="priority_queue",
            description="Uses a stack or FIFO queue instead of ordering by priority",
        ),
        Misconception(
            id="g_is_edge_cost",
            concept_id="path_cost",
            description="Thinks g(n) is only the last edge cost, not the total path cost",
        ),
        Misconception(
            id="h_must_be_exact",
            concept_id="heuristic",
            description="Thinks the heuristic must equal the true remaining cost",
        ),
        Misconception(
            id="admissible_means_overestimate",
            concept_id="admissibility",
            description="Thinks an admissible heuristic may overestimate",
        ),
        Misconception(
            id="confuses_g_and_h",
            concept_id="astar",
            description="Swaps g(n) and h(n) when computing f(n)",
        ),
        Misconception(
            id="astar_orders_by_h_only",
            concept_id="astar",
            description="Orders the frontier by h(n) alone (greedy best-first)",
        ),
    ]
    return CourseCatalog(concepts=concepts, misconceptions=misconceptions)