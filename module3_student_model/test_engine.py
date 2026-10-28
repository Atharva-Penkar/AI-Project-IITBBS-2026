import pytest

from module3_student_model.engine import StudentModelEngine
from module3_student_model.sample_catalog import astar_catalog
from shared.schema import ActionType, Event, EventType, Intent

CAT = astar_catalog()


def quiz(sid, correct, concept="heuristic", **kw):
    return Event(student_id=sid, concept_id=concept, type=EventType.QUIZ_ANSWER, correct=correct, **kw)


def test_same_score_different_histories_give_different_estimates():
    """Same 2/5 score, different answer order, gives different estimates."""
    eng = StudentModelEngine(CAT)
    for c in [False, False, False, True, True]:
        eng.update(quiz("improving", c))
    for c in [True, True, False, False, False]:
        eng.update(quiz("declining", c))
    for c in [False, False, False, True, True]:
        eng.update(quiz("hinted", c, hint_used=c))

    p = {s: eng.get_student(s).concepts["heuristic"].p_mastery for s in ("improving", "declining", "hinted")}
    assert p["improving"] > p["declining"] + 0.2
    assert p["hinted"] < p["improving"]


def test_mastery_and_misconception_are_kept_separate():
    eng = StudentModelEngine(CAT)
    for _ in range(4):
        eng.update(quiz("lacks", False, chosen_misconception=None))
        eng.update(quiz("misconceived", False, chosen_misconception="h_must_be_exact"))
    lacks = eng.get_student("lacks").concepts["heuristic"]
    mis = eng.get_student("misconceived").concepts["heuristic"]
    assert lacks.p_mastery < 0.2 and mis.p_mastery < 0.2
    assert lacks.misconceptions["h_must_be_exact"] < 0.15
    assert mis.misconceptions["h_must_be_exact"] > 0.8
    assert "h_must_be_exact" in mis.explanation


def test_vague_question_moves_estimates_less_than_clear_one():
    eng = StudentModelEngine(CAT)
    for sid, conf, intent in (("clear", 0.9, Intent.MISCONCEPTION_SIGNAL), ("vague", 0.3, Intent.UNCLEAR)):
        eng.update(Event(student_id=sid, concept_id="astar", type=EventType.QUESTION, voluntary=True,
                         intent=intent, nlu_confidence=conf, misconception_signals=["confuses_g_and_h"]))
    clear = eng.get_student("clear").concepts["astar"].misconceptions["confuses_g_and_h"]
    vague = eng.get_student("vague").concepts["astar"].misconceptions["confuses_g_and_h"]
    assert clear > vague > 0.15


def test_instruction_raises_mastery_without_counting_as_observation():
    eng = StudentModelEngine(CAT, ["s"])
    before = eng.get_student("s").concepts["astar"]
    eng.update(Event(student_id="s", concept_id="astar", type=EventType.INSTRUCTION, action=ActionType.TA_INTERVENTION))
    after = eng.get_student("s").concepts["astar"]
    assert after.p_mastery > before.p_mastery
    assert after.n_observations == 0


def test_summaries_and_discretised_state():
    eng = StudentModelEngine(CAT)
    for sid in ("a", "b", "c", "d"):
        for _ in range(4):
            eng.update(quiz(sid, sid != "d", concept="astar"))
    summary = eng.get_class_summary()
    assert summary.student_ids == ["a", "b", "c", "d"]
    assert summary.concepts["astar"].fraction_below_threshold == pytest.approx(0.25)
    group = eng.get_group_summary(["a", "b"])
    assert group.concepts["astar"].mean_mastery > summary.concepts["astar"].mean_mastery
    state = eng.discretised_state("astar")
    assert len(state) == 5 and all(v in (0, 1, 2) for v in state)


def test_unknown_concept_and_student_raise():
    eng = StudentModelEngine(CAT)
    with pytest.raises(KeyError):
        eng.update(quiz("s", True, concept="not_a_concept"))
    with pytest.raises(KeyError):
        eng.get_student("nobody")


def test_answer_event_without_result_is_rejected():
    eng = StudentModelEngine(CAT)
    with pytest.raises(ValueError):
        eng.update(Event(student_id="s", concept_id="astar", type=EventType.QUIZ_ANSWER))