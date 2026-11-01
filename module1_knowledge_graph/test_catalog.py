from module1_knowledge_graph.catalog import load_course_catalog, match_misconceptions
from module1_knowledge_graph.knowledge_graph import nodes
from module1_knowledge_graph.student_graph import concept_status, diagnose_student, study_plan
from module3_student_model.engine import StudentModelEngine
from shared.schema import Event, EventType


def test_catalog_has_every_concept_and_misconception():
    cat = load_course_catalog()
    assert set(cat.concept_ids()) == set(nodes)
    assert len(cat.misconceptions) == sum(len(n["misconceptions"]) for n in nodes.values())


def test_catalog_order_puts_prerequisites_first():
    cat = load_course_catalog()
    pos = {c: i for i, c in enumerate(cat.concept_ids())}
    for c in cat.concepts:
        for p in c.prerequisites:
            assert pos[p] < pos[c.id]


def test_text_matches_misconception_id():
    assert match_misconceptions("Path cost is the estimated remaining cost to the goal.")[0][0] == "g_is_remaining_cost"


def test_student_graph_uses_shared_student_model():
    engine = StudentModelEngine(load_course_catalog(), ["s"])
    for _ in range(3):
        engine.update(Event(student_id="s", concept_id="Path_Cost", type=EventType.QUIZ_ANSWER,
                            correct=False, chosen_misconception="g_is_remaining_cost"))
    student = engine.get_student("s")
    assert concept_status(student, "Path_Cost") == "misconception"
    assert "Path_Cost" in study_plan(student, "Astar")
    top = diagnose_student(student, "Astar")["ranked_causes"][0]
    assert top["concept_id"] != "Astar" or top["score"] > 0
