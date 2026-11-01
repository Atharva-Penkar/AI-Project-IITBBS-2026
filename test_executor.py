"""Tests for event forwarding and simulator no-op semantics."""

from shared.schema import ActionType, Concept, CourseCatalog, Event, EventType
from module4_planner.executor import AdaptiveExecutor
from module4_planner.models import ActionDecision


class FakeSimulator:
    def __init__(self, events):
        self.catalog = CourseCatalog(concepts=[Concept(id="topic", name="Topic")])
        self._events = events
        self._students = ["s01", "s02", "s03"]
        self.step_count = 0
        self.last_call = None

    def student_ids(self):
        return list(self._students)

    def step(self, action, concept_id, target_ids=None, target_misconception=None):
        self.step_count += 1
        self.last_call = (action, concept_id, target_ids, target_misconception)
        return list(self._events)


class FakeEngine:
    def __init__(self):
        self.updated = []

    def update_many(self, events):
        self.updated.extend(events)


class FakePlanner:
    def __init__(self):
        self.executed = []

    def record_execution(self, decision):
        self.executed.append(decision)


def test_executor_forwards_targeted_action_and_updates_model() -> None:
    event = Event(
        student_id="s02", concept_id="topic", type=EventType.INSTRUCTION,
        action=ActionType.TA_INTERVENTION,
    )
    simulator = FakeSimulator([event])
    engine = FakeEngine()
    planner = FakePlanner()
    decision = ActionDecision(
        action=ActionType.TA_INTERVENTION,
        concept_id="topic",
        target_student_ids=("s02",),
        target_misconception="mis_1",
        rationale="test targeting",
    )

    result = AdaptiveExecutor().execute(decision, simulator, engine, planner=planner)

    assert simulator.last_call == (ActionType.TA_INTERVENTION, "topic", ["s02"], "mis_1")
    assert engine.updated == [event]
    assert result.targeted_student_count == 1
    assert result.event_count == 1
    assert planner.executed == [decision]


def test_next_concept_may_validly_return_no_events() -> None:
    simulator = FakeSimulator([])
    engine = FakeEngine()
    decision = ActionDecision(action=ActionType.NEXT_CONCEPT, concept_id="topic")

    result = AdaptiveExecutor().execute(decision, simulator, engine)

    assert result.event_count == 0
    assert engine.updated == []
    assert simulator.step_count == 1


def test_executor_rejects_unknown_target_student() -> None:
    simulator = FakeSimulator([])
    engine = FakeEngine()
    decision = ActionDecision(
        action=ActionType.EXAMPLE, concept_id="topic", target_student_ids=("unknown",)
    )

    try:
        AdaptiveExecutor().execute(decision, simulator, engine)
    except ValueError as exc:
        assert "unknown student" in str(exc)
    else:
        raise AssertionError("Expected invalid target to raise ValueError")
