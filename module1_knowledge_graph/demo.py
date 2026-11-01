"""Demo: graph, prerequisite tracing, A* failures, diagnosis, and one student's knowledge state."""
from knowledge_graph import nodes, prerequisite_chain, diagnose, explain
from astar_examples import run_all, STUDENT_TEXT
from student_model import StudentModel


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
    s = StudentModel("Ravi")
    for _ in range(3):
        s.observe("Heuristic", correct=True)
    s.observe("Path_Cost", correct=False, text="Path cost is the estimated remaining cost to the goal.")
    s.observe("Path_Cost", correct=False)
    s.observe("Queue_PQ", kind="hint")
    s.observe("Astar", correct=False, text=STUDENT_TEXT["h_only"])
    s.observe("Astar", kind="ambiguous", text="it just does the thing")
    print(f"  {'concept':34}{'status':15}{'mastery':9}{'uncertainty'}")
    for row in s.table():
        if row["status"] != "unknown":
            print(f"  {row['name']:34}{row['status']:15}{row['mastery']:<9}{row['uncertainty']}")
    print("\n  Study plan for A* Search:")
    for cid, name, status in s.study_plan("Astar"):
        print(f"    - {name} ({status})")
    print("\n" + explain(s.diagnose_for_student("Astar")))


if __name__ == "__main__":
    main()
