import numpy as np
import pytest

from module3_student_model.engine import StudentModelEngine
from module3_student_model.profiles import SCENARIOS, make_classroom
from module3_student_model.sample_catalog import astar_catalog
from module3_student_model.simulator import ClassroomSimulator
from shared.schema import ActionType, EventType

CAT = astar_catalog()
LESSON = [
    ActionType.DIAGNOSTIC_QUIZ,
    ActionType.LECTURE,
    ActionType.EXAMPLE,
    ActionType.DIAGNOSTIC_QUIZ,
    ActionType.PEER_DISCUSSION,
    ActionType.EXERCISE,
    ActionType.DIAGNOSTIC_QUIZ,
]


def run(scenario, seed=1):
    sim = ClassroomSimulator(CAT, make_classroom(CAT, scenario, 24, seed=seed), seed=seed)
    eng = StudentModelEngine(CAT, sim.student_ids())
    for cid in CAT.concept_ids():
        for action in LESSON:
            eng.update_many(sim.step(action, cid))
    return sim, eng


def test_same_seed_is_reproducible():
    a = ClassroomSimulator(CAT, make_classroom(CAT, "mixed", 10, seed=3), seed=3)
    b = ClassroomSimulator(CAT, make_classroom(CAT, "mixed", 10, seed=3), seed=3)
    assert a.step(ActionType.DIAGNOSTIC_QUIZ, "astar") == b.step(ActionType.DIAGNOSTIC_QUIZ, "astar")


def test_events_do_not_reveal_true_state():
    sim = ClassroomSimulator(CAT, make_classroom(CAT, "mixed", 10, seed=0), seed=0)
    for e in sim.step(ActionType.DIAGNOSTIC_QUIZ, "astar"):
        assert e.type in (EventType.QUIZ_ANSWER, EventType.NO_RESPONSE)
        assert "known" not in e.model_dump()


def test_next_concept_produces_no_events():
    sim = ClassroomSimulator(CAT, make_classroom(CAT, "mixed", 5), seed=0)
    assert sim.step(ActionType.NEXT_CONCEPT, "astar") == []


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_engine_tracks_true_mastery(scenario):
    sim, eng = run(scenario)
    errors = [
        abs(eng.get_student(s).concepts[c].p_mastery - sim.true_mastery(s, c))
        for s in sim.student_ids()
        for c in CAT.concept_ids()
    ]
    assert np.mean(errors) < 0.3


def test_misconception_detection_precision_and_recall():
    tp = fp = fn = 0
    for seed in (1, 2, 3, 4):
        sim, eng = run("widespread_misconception", seed)
        for s in sim.student_ids():
            truth = sim.true_misconceptions(s)
            for c in CAT.concept_ids():
                for m, p in eng.get_student(s).concepts[c].misconceptions.items():
                    flagged = p >= 0.5
                    tp += flagged and m in truth
                    fp += flagged and m not in truth
                    fn += (not flagged) and m in truth
    assert tp / (tp + fp) > 0.75
    assert tp / (tp + fn) > 0.6


def test_confident_wrong_students_are_found():
    sim, eng = run("confident_wrong")
    flagged = set(eng.get_class_summary().confidently_wrong_students)
    planted = {s for s, p in sim.profiles.items() if p.archetype == "confident_wrong"}
    assert len(flagged & planted) / len(planted) >= 0.75


def test_quiet_knowers_have_high_mastery_but_low_participation():
    sim, eng = run("quiet_knowers")
    quiet = [s for s, p in sim.profiles.items() if p.archetype == "quiet_knower"]
    others = [s for s, p in sim.profiles.items() if p.archetype != "quiet_knower"]
    quiet_part = np.mean([eng.get_student(s).participation for s in quiet])
    other_part = np.mean([eng.get_student(s).participation for s in others])
    quiet_mastery = np.mean([np.mean([c.p_mastery for c in eng.get_student(s).concepts.values()]) for s in quiet])
    assert quiet_part < other_part
    assert quiet_mastery > 0.6