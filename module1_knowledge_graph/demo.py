"""Demo: graph, prerequisite tracing, A* failures, diagnosis, and one student's knowledge graph.

    python -m module1_knowledge_graph.demo
"""
from module1_knowledge_graph.astar_examples import STUDENT_TEXT, run_all
from module1_knowledge_graph.catalog import load_course_catalog
from module1_knowledge_graph.knowledge_graph import diagnose, explain, nodes, prerequisite_chain
from module1_knowledge_graph.student_graph import diagnose_student, student_knowledge_graph, study_plan
from module3_student_model.engine import StudentModelEngine
from shared.schema import Event, EventType


def main():
    print("== Concepts ==")
    for cid, n in nodes.items():
        print(f"  {cid}: {n['name']}")

    print("\n== Study order for A* Search (prerequisites first) ==")
    print("  " + ", then ".join(nodes[c]["name"] for c in prerequisite_chain("Astar")))

    print("\n== A* family on the toy graph (optimal cost is 4) ==")
    for name, r in run_all().items():
        path = "-".join(r["path"]) if r["path"] else "none"
        print(f"  {name:13} cost={r['cost']}  path={path}  expanded={r['expanded']}")

    print("\n== Diagnosis for each student error ==")
    for name, text in STUDENT_TEXT.items():
        print(f"\n[{name}] student says: \"{text}\"")
        print(explain(diagnose("Astar", text)))

    print("\n== Student-knowledge graph: student 'Ravi' ==")
    engine = StudentModelEngine(load_course_catalog(), ["Ravi"])
    answers = [("Heuristic", True, None)] * 3 + [
        ("Path_Cost", False, "g_is_remaining_cost"),
        ("Path_Cost", False, None),
        ("Astar", False, "astar_uses_only_h"),
    ]
    for concept, correct, chosen in answers:
        engine.update(Event(student_id="Ravi", concept_id=concept, type=EventType.QUIZ_ANSWER,
                            correct=correct, chosen_misconception=chosen))
    ravi = engine.get_student("Ravi")
    print(f"  {'concept':34}{'status':15}{'mastery':9}uncertainty")
    for row in student_knowledge_graph(ravi)["nodes"]:
        if row["status"] != "unknown":
            print(f"  {row['name']:34}{row['status']:15}{row['mastery']:<9}{row['uncertainty']}")
    print("\n  Study plan for A* Search:")
    for cid in study_plan(ravi, "Astar"):
        print(f"    - {nodes[cid]['name']}")
    print("\n" + explain(diagnose_student(ravi, "Astar", STUDENT_TEXT["h_only"])))


if __name__ == "__main__":
    main()
