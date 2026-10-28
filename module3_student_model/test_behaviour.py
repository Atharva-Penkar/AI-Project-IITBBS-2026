from module3_student_model.behaviour import BehaviourTracker
from shared.schema import ActionType, Event, EventType


def ev(**kw):
    return Event(student_id=kw.pop("sid", "s"), concept_id="c", **kw)


def test_confidently_wrong_student_is_flagged():
    t = BehaviourTracker()
    for _ in range(4):
        t.update(ev(type=EventType.QUIZ_ANSWER, correct=False, stated_confidence=0.9))
    assert t.confidently_wrong("s")
    assert t.overconfidence("s") > 0.25


def test_calibrated_student_is_not_flagged():
    t = BehaviourTracker()
    for correct in (True, True, False, True):
        t.update(ev(type=EventType.QUIZ_ANSWER, correct=correct, stated_confidence=0.75))
    assert not t.confidently_wrong("s")


def test_participation_counts_contributions_per_opportunity():
    t = BehaviourTracker()
    for i in range(10):
        t.update(ev(sid="talker", type=EventType.INSTRUCTION, action=ActionType.PEER_DISCUSSION))
        t.update(ev(sid="talker", type=EventType.DISCUSSION, voluntary=True))
        t.update(ev(sid="quiet", type=EventType.INSTRUCTION, action=ActionType.PEER_DISCUSSION))
    assert t.participation("talker") > 0.8
    assert t.participation("quiet") < 0.15


def test_no_response_lowers_engagement():
    t = BehaviourTracker()
    start = t.engagement("s")
    t.update(ev(type=EventType.NO_RESPONSE))
    assert t.engagement("s") < start