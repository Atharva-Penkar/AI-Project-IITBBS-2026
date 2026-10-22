"""Student-state engine: the entry point for updating and reading student estimates.

    engine = StudentModelEngine(catalog)
    engine.update(event)                 # after every classroom event
    engine.get_student("s07")            # StudentEstimate
    engine.get_group_summary(ids)        # GroupSummary
    engine.get_class_summary()           # GroupSummary
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from module3_student_model.aggregate import SummaryThresholds, discretise, summarise
from module3_student_model.behaviour import BehaviourParams, BehaviourTracker
from module3_student_model.bkt import ACTION_LEARNING_WEIGHT, BKTParams
from module3_student_model.misconceptions import MisconceptionParams, MisconceptionTracker
from module3_student_model.uncertainty import EnsembleTracker
from shared.schema import (
    ConceptEstimate,
    CourseCatalog,
    Event,
    EventType,
    GroupSummary,
    Intent,
    StudentEstimate,
)

QUESTION_EVIDENCE: dict[Intent, float] = {
    Intent.CONFUSION: -0.3,
    Intent.MISCONCEPTION_SIGNAL: -0.2,
    Intent.CLARIFICATION: 0.1,
    Intent.EXAMPLE_REQUEST: 0.0,
}
"""Soft mastery evidence from a question, as a shift from the neutral score 0.5.

Scaled by the NLU confidence. UNCLEAR questions give no mastery evidence.
"""


@dataclass
class EngineConfig:
    bkt: BKTParams = field(default_factory=BKTParams)
    bkt_per_concept: dict[str, BKTParams] = field(default_factory=dict)
    ensemble_size: int = 15
    seed: int = 0
    misconception: MisconceptionParams = field(default_factory=MisconceptionParams)
    misconception_priors: dict[str, float] = field(default_factory=dict)
    behaviour: BehaviourParams = field(default_factory=BehaviourParams)
    thresholds: SummaryThresholds = field(default_factory=SummaryThresholds)
    default_instruction_strength: float = 0.5


class StudentModelEngine:
    def __init__(
        self,
        catalog: CourseCatalog,
        student_ids: Iterable[str] | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.config = config or EngineConfig()
        self.mastery = EnsembleTracker(
            default_params=self.config.bkt,
            concept_params=dict(self.config.bkt_per_concept),
            n_members=self.config.ensemble_size,
            seed=self.config.seed,
        )
        self.misconceptions = MisconceptionTracker(
            params=self.config.misconception, priors=dict(self.config.misconception_priors)
        )
        self.behaviour = BehaviourTracker(params=self.config.behaviour)
        self._students: list[str] = []
        self._concept_ms = {c.id: [m.id for m in catalog.misconceptions_for(c.id)] for c in catalog.concepts}
        for sid in student_ids or []:
            self.register(sid)

    def register(self, student_id: str) -> None:
        if student_id not in self._students:
            self._students.append(student_id)

    def student_ids(self) -> list[str]:
        return list(self._students)

    def update(self, event: Event) -> None:
        """Apply one classroom event. Students are registered on first sight."""
        if event.concept_id not in self._concept_ms:
            raise KeyError(f"Unknown concept: {event.concept_id}")
        self.register(event.student_id)
        self.behaviour.update(event)

        sid, cid = event.student_id, event.concept_id
        concept_ms = self._concept_ms[cid]

        if event.type == EventType.INSTRUCTION:
            strength = (
                ACTION_LEARNING_WEIGHT[event.action]
                if event.action is not None
                else self.config.default_instruction_strength
            )
            self.mastery.expose(sid, cid, strength)
            self.misconceptions.apply_instruction(sid, concept_ms, event.target_misconception, strength)

        elif event.type in (EventType.QUIZ_ANSWER, EventType.EXERCISE):
            if event.correct is None and event.score is None:
                raise ValueError("Answer events need `correct` or `score`")
            correct = event.correct if event.correct is not None else event.score >= 0.5
            prm = self.config.bkt_per_concept.get(cid, self.config.bkt)
            guess = prm.p_guess_hint if event.hint_used else prm.p_guess
            self.misconceptions.update_from_answer(
                sid,
                concept_ms,
                correct=correct,
                chosen_misconception=event.chosen_misconception,
                p_mastery=self.mastery.mean(sid, cid),
                guess=guess,
                slip=prm.p_slip,
                available_misconceptions=event.available_misconceptions,
                n_distractors=event.n_distractors,
            )
            if event.correct is not None:
                self.mastery.observe(sid, cid, correct=event.correct, hint_used=event.hint_used)
            else:
                self.mastery.observe(sid, cid, score=event.score, hint_used=event.hint_used)

        elif event.type in (EventType.QUESTION, EventType.DISCUSSION):
            c = event.nlu_confidence if event.nlu_confidence is not None else 0.5
            signal_c = c * (0.5 if event.intent == Intent.UNCLEAR else 1.0)
            for m in event.misconception_signals:
                if m in concept_ms:
                    self.misconceptions.update_from_signal(sid, m, signal_c)
            if event.intent in QUESTION_EVIDENCE:
                self.mastery.observe(sid, cid, score=0.5 + QUESTION_EVIDENCE[event.intent] * c)

    def update_many(self, events: Iterable[Event]) -> None:
        for e in events:
            self.update(e)

    def _concept_estimate(self, sid: str, cid: str) -> ConceptEstimate:
        p = self.mastery.mean(sid, cid)
        u = self.mastery.std(sid, cid)
        n = self.mastery.n_observations(sid, cid)
        ms = self.misconceptions.for_concept(sid, self._concept_ms[cid])
        parts = [f"mastery {p:.2f} ± {u:.2f} from {n} observation{'s' if n != 1 else ''}"]
        for m, pm in sorted(ms.items(), key=lambda kv: -kv[1]):
            if pm >= 0.3:
                picks, tested, signals = self.misconceptions.evidence(sid, m)
                parts.append(
                    f"P({m}) = {pm:.2f} (picked its wrong option {picks} of {tested} times, {signals} NLU signals)"
                )
        return ConceptEstimate(
            concept_id=cid,
            p_mastery=p,
            uncertainty=u,
            n_observations=n,
            misconceptions=ms,
            explanation="; ".join(parts),
        )

    def get_student(self, student_id: str) -> StudentEstimate:
        if student_id not in self._students:
            raise KeyError(f"Unknown student: {student_id}")
        return StudentEstimate(
            student_id=student_id,
            concepts={cid: self._concept_estimate(student_id, cid) for cid in self._concept_ms},
            engagement=self.behaviour.engagement(student_id),
            participation=self.behaviour.participation(student_id),
            confidence=self.behaviour.confidence(student_id),
            overconfidence=self.behaviour.overconfidence(student_id),
            confidently_wrong=self.behaviour.confidently_wrong(student_id),
        )

    def get_group_summary(self, student_ids: Iterable[str]) -> GroupSummary:
        return summarise(
            [self.get_student(s) for s in student_ids], list(self._concept_ms), self.config.thresholds
        )

    def get_class_summary(self) -> GroupSummary:
        return self.get_group_summary(self._students)

    def discretised_state(self, concept_id: str, student_ids: Iterable[str] | None = None) -> tuple[int, ...]:
        """State tuple for tabular RL; see aggregate.discretise."""
        ids = list(student_ids) if student_ids is not None else self._students
        return discretise(self.get_group_summary(ids), concept_id)