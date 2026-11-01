"""Builds the shared CourseCatalog from knowledge_graph.json.

Every module should get concept and misconception IDs from here, so there is
one source of truth:

    from module1_knowledge_graph.catalog import load_course_catalog
    catalog = load_course_catalog()
"""

from __future__ import annotations

from module1_knowledge_graph.knowledge_graph import nodes, prereqs, tokens, best_misconception_detail, MATCH_THRESHOLD
from shared.schema import Concept, CourseCatalog, Misconception


def load_course_catalog() -> CourseCatalog:
    """Concepts in a valid teaching order (every prerequisite before the concepts that need it)."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(cid: str) -> None:
        if cid in seen:
            return
        seen.add(cid)
        for p in prereqs[cid]:
            visit(p)
        order.append(cid)

    for cid in nodes:
        visit(cid)

    concepts = [Concept(id=c, name=nodes[c]["name"], prerequisites=list(prereqs[c])) for c in order]
    misconceptions = [
        Misconception(id=mid, concept_id=c, description=text)
        for c in order
        for text, mid in nodes[c].get("misconception_ids", {}).items()
    ]
    return CourseCatalog(concepts=concepts, misconceptions=misconceptions)


def match_misconceptions(text: str, concept_ids: list[str] | None = None) -> list[tuple[str, float]]:
    """Misconception IDs whose wording (or a stored paraphrase) matches the text, best first.

    A match only counts if at least one shared word is not part of the concept's
    own name, so "Can you explain priority queues?" does not match the
    misconception "A priority queue and FIFO queue are interchangeable."
    """
    text_tokens = tokens(text)
    out = []
    for cid in concept_ids or list(nodes):
        score, stated, words = best_misconception_detail(text_tokens, nodes[cid])
        name_words = tokens(nodes[cid]["name"]) | tokens(cid.replace("_", " "))
        if stated is not None and score >= MATCH_THRESHOLD and words - name_words:
            out.append((nodes[cid]["misconception_ids"][stated], score))
    return sorted(out, key=lambda pair: -pair[1])
