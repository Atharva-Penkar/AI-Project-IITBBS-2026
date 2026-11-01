"""
A toy weighted graph where A* fails in the ways students confuse it.
Start S, goal G. The optimal route S-B-G costs 4; the tempting route S-A-G costs 6.
"""
from search import search, is_admissible

ADJ = {
    "S": [("A", 1), ("B", 2)],
    "A": [("G", 5)],
    "B": [("G", 2)],
    "G": [],
}
START, GOAL = "S", "G"
H_ADMISSIBLE = {"S": 4, "A": 1, "B": 2, "G": 0}
H_OVERESTIMATE = {"S": 4, "A": 1, "B": 10, "G": 0}  # h(B)=10 but true cost is 2

VARIANTS = {
    "correct":      dict(frontier="priority", key="f", h=H_ADMISSIBLE),
    "h_only":       dict(frontier="priority", key="h", h=H_ADMISSIBLE),
    "fifo":         dict(frontier="fifo",     key="f", h=H_ADMISSIBLE),
    "inadmissible": dict(frontier="priority", key="f", h=H_OVERESTIMATE),
}

# What a student holding each error would say when asked to explain their wrong answer.
STUDENT_TEXT = {
    "h_only": "I thought A* uses only the heuristic h(n), not path cost.",
    "fifo": "A priority queue always removes the earliest inserted item.",
    "inadmissible": "I think an overestimating heuristic still guarantees an optimal A* path.",
}


def run_variant(name):
    v = VARIANTS[name]
    return search(ADJ, START, GOAL, h=v["h"], key=v["key"], frontier=v["frontier"])


def run_all():
    return {name: run_variant(name) for name in VARIANTS}
