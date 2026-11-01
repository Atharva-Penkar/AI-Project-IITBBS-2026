"""Student-knowledge graph: one student's estimates laid over the concept graph.

The estimates come from the shared student model (StudentEstimate); this file
does not keep its own copy of mastery.

    graph = student_knowledge_graph(engine.get_student("s07"))
    plan = study_plan(engine.get_student("s07"), "Astar")
"""

from __future__ import annotations

from module1_knowledge_graph.knowledge_graph import diagnose, nodes, prereqs
from module1_knowledge_graph.search import missing_prerequisites
from shared.schema import StudentEstimate

MASTERED = 0.9
MISCONCEPTION = 0.5
MIN_OBSERVATIONS = 1


def concept_status(student: StudentEstimate, concept_id: str) -> str:
    """unknown, mastered, misconception or not_learned."""
    est = student.concepts[concept_id]
    if est.n_observations < MIN_OBSERVATIONS and not any(p >= MISCONCEPTION for p in est.misconceptions.values()):
        return "unknown"
    if any(p >= MISCONCEPTION for p in est.misconceptions.values()):
        return "misconception"
    if est.p_mastery >= MASTERED:
        return "mastered"
    return "not_learned"


def student_knowledge_graph(student: StudentEstimate) -> dict:
    """Nodes with the student's status, mastery and uncertainty; edges from the concept graph."""
    return {
        "student_id": student.student_id,
        "nodes": [
            {
                "id": cid,
                "name": nodes[cid]["name"],
                "status": concept_status(student, cid),
                "mastery": round(student.concepts[cid].p_mastery, 3),
                "uncertainty": round(student.concepts[cid].uncertainty, 3),
                "misconceptions": {m: round(p, 3) for m, p in student.concepts[cid].misconceptions.items() if p >= MISCONCEPTION},
            }
            for cid in nodes
            if cid in student.concepts
        ],
        "edges": [{"source": p, "target": c} for c in nodes for p in prereqs[c]],
    }


def mastery_map(student: StudentEstimate) -> dict[str, float]:
    return {cid: est.p_mastery for cid, est in student.concepts.items() if cid in nodes}


def study_plan(student: StudentEstimate, target: str) -> list[str]:
    """Unmastered prerequisites of target, foundations first."""
    mastered = {cid for cid in nodes if cid in student.concepts and concept_status(student, cid) == "mastered"}
    return missing_prerequisites(prereqs, target, mastered)


def diagnose_student(student: StudentEstimate, concept_id: str, text: str = "", top_k: int = 5) -> dict:
    """Ranked likely causes of a student's difficulty, using their live mastery estimates."""
    return diagnose(concept_id, text, mastery=mastery_map(student), top_k=top_k)
