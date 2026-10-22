"""Engagement, participation and confidence estimates."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.schema import Event, EventType


@dataclass(frozen=True)
class BehaviourParams:
    engagement_start: float = 0.7
    engagement_rate: float = 0.2
    confidence_start: float = 0.5
    confidence_rate: float = 0.3
    participation_prior: tuple[float, float] = (1.0, 1.0)
    overconfidence_flag: float = 0.25
    min_answers_for_flag: int = 3


@dataclass
class StudentBehaviour:
    engagement: float
    confidence: float
    opportunities: int = 0
    contributions: int = 0
    calibration_sum: float = 0.0
    calibration_n: int = 0


@dataclass
class BehaviourTracker:
    params: BehaviourParams = field(default_factory=BehaviourParams)
    _students: dict[str, StudentBehaviour] = field(default_factory=dict)

    def _get(self, student_id: str) -> StudentBehaviour:
        if student_id not in self._students:
            self._students[student_id] = StudentBehaviour(
                engagement=self.params.engagement_start, confidence=self.params.confidence_start
            )
        return self._students[student_id]

    def _nudge_engagement(self, b: StudentBehaviour, signal: float) -> None:
        r = self.params.engagement_rate
        b.engagement = (1 - r) * b.engagement + r * signal

    def update(self, event: Event) -> None:
        b = self._get(event.student_id)

        if event.type == EventType.INSTRUCTION:
            b.opportunities += 1
        elif event.type == EventType.NO_RESPONSE:
            self._nudge_engagement(b, 0.0)
        elif event.type in (EventType.QUIZ_ANSWER, EventType.EXERCISE):
            self._nudge_engagement(b, 0.7)
        elif event.type in (EventType.QUESTION, EventType.DISCUSSION):
            self._nudge_engagement(b, 1.0)
            if event.voluntary:
                b.contributions += 1
        elif event.type == EventType.HINT:
            self._nudge_engagement(b, 0.8)

        if event.stated_confidence is not None:
            r = self.params.confidence_rate
            b.confidence = (1 - r) * b.confidence + r * event.stated_confidence
            outcome = event.score if event.score is not None else (1.0 if event.correct else 0.0)
            if event.correct is not None or event.score is not None:
                b.calibration_sum += event.stated_confidence - outcome
                b.calibration_n += 1

    def engagement(self, student_id: str) -> float:
        return self._get(student_id).engagement

    def participation(self, student_id: str) -> float:
        b = self._get(student_id)
        a, c = self.params.participation_prior
        return min(1.0, (b.contributions + a) / (b.opportunities + a + c))

    def confidence(self, student_id: str) -> float:
        return self._get(student_id).confidence

    def overconfidence(self, student_id: str) -> float:
        """Mean of (stated confidence - actual result); positive means overconfident."""
        b = self._get(student_id)
        return b.calibration_sum / b.calibration_n if b.calibration_n else 0.0

    def confidently_wrong(self, student_id: str) -> bool:
        b = self._get(student_id)
        return (
            b.calibration_n >= self.params.min_answers_for_flag
            and self.overconfidence(student_id) > self.params.overconfidence_flag
        )