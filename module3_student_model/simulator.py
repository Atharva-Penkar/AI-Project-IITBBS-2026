"""Rule-based classroom simulator.

Holds each student's TRUE state (what they really know and believe) and
produces noisy events when a teaching action is applied. The student model
engine never sees the true state; it must infer it from the events.

A student who holds a misconception about a concept has not mastered it, and
cannot learn it until the misconception is removed by teaching.

Fast and seeded, so it can run many episodes for training and testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from module3_student_model.profiles import StudentProfile
from shared.schema import ActionType, CourseCatalog, Event, EventType, Intent

TEACHING_LEARN_RATE: dict[ActionType, float] = {
    ActionType.LECTURE: 0.12,
    ActionType.EXAMPLE: 0.18,
    ActionType.PEER_DISCUSSION: 0.15,
    ActionType.PEER_LEARNING: 0.20,
    ActionType.TA_INTERVENTION: 0.30,
    ActionType.REVISION: 0.20,
}
ANSWER_ACTIONS = {ActionType.DIAGNOSTIC_QUIZ, ActionType.ASSESSMENT, ActionType.EXERCISE}
PRACTICE_LEARN_RATE = {ActionType.DIAGNOSTIC_QUIZ: 0.03, ActionType.ASSESSMENT: 0.03, ActionType.EXERCISE: 0.12}
ENGAGEMENT_EFFECT: dict[ActionType, float] = {
    ActionType.LECTURE: -0.04,
    ActionType.EXAMPLE: 0.02,
    ActionType.PEER_DISCUSSION: 0.05,
    ActionType.PEER_LEARNING: 0.05,
    ActionType.TA_INTERVENTION: 0.04,
    ActionType.REVISION: -0.02,
    ActionType.DIAGNOSTIC_QUIZ: -0.01,
    ActionType.ASSESSMENT: -0.02,
    ActionType.EXERCISE: 0.0,
}
MISCONCEPTION_EXPRESS = 0.7
UNLEARN_TARGETED = 0.5
UNLEARN_GENERAL = 0.06


@dataclass
class TrueState:
    known: dict[str, bool]
    misconceptions: set[str]
    engagement: float


class ClassroomSimulator:
    def __init__(
        self,
        catalog: CourseCatalog,
        profiles: list[StudentProfile],
        seed: int = 0,
        n_distractors: int = 3,
    ) -> None:
        self.catalog = catalog
        self.profiles = {p.student_id: p for p in profiles}
        self.n_distractors = n_distractors
        self._seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self._seed if seed is None else seed)
        self.step_count = 0
        self.state: dict[str, TrueState] = {}
        for sid, p in self.profiles.items():
            known = {c.id: bool(self.rng.random() < p.prior_knowledge.get(c.id, 0.2)) for c in self.catalog.concepts}
            for m in p.misconceptions:
                known[self.catalog.misconception(m).concept_id] = False
            self.state[sid] = TrueState(known=known, misconceptions=set(p.misconceptions), engagement=p.engagement)

    def student_ids(self) -> list[str]:
        return list(self.profiles)

    def true_mastery(self, student_id: str, concept_id: str) -> bool:
        return self.state[student_id].known[concept_id]

    def true_misconceptions(self, student_id: str) -> set[str]:
        return set(self.state[student_id].misconceptions)

    def true_engagement(self, student_id: str) -> float:
        return self.state[student_id].engagement

    def step(
        self,
        action: ActionType,
        concept_id: str,
        target_ids: list[str] | None = None,
        target_misconception: str | None = None,
    ) -> list[Event]:
        """Apply one teaching action to the targeted students and return the events."""
        self.catalog.concept(concept_id)
        self.step_count += 1
        if action == ActionType.NEXT_CONCEPT:
            return []
        targets = target_ids if target_ids is not None else self.student_ids()
        events: list[Event] = []
        for sid in targets:
            if action in ANSWER_ACTIONS:
                events.extend(self._answer(sid, concept_id, action))
            else:
                events.extend(self._teach(sid, concept_id, action, target_misconception))
            self._update_engagement(sid, action)
        return events

    def _concept_misconceptions(self, concept_id: str) -> list[str]:
        return [m.id for m in self.catalog.misconceptions_for(concept_id)]

    def _prereq_factor(self, sid: str, concept_id: str) -> float:
        prereqs = self.catalog.concept(concept_id).prerequisites
        if not prereqs:
            return 1.0
        frac = sum(self.state[sid].known[p] for p in prereqs) / len(prereqs)
        return 0.3 + 0.7 * frac

    def _learn(self, sid: str, concept_id: str, rate: float) -> None:
        st, prof = self.state[sid], self.profiles[sid]
        if st.known[concept_id]:
            return
        if st.misconceptions & set(self._concept_misconceptions(concept_id)):
            return
        p = rate * prof.learning_speed * (0.4 + 0.6 * st.engagement) * self._prereq_factor(sid, concept_id)
        if self.rng.random() < min(p, 0.95):
            st.known[concept_id] = True

    def _teach(self, sid: str, concept_id: str, action: ActionType, target_m: str | None) -> list[Event]:
        st, prof = self.state[sid], self.profiles[sid]
        self._learn(sid, concept_id, TEACHING_LEARN_RATE[action])
        for m in list(st.misconceptions & set(self._concept_misconceptions(concept_id))):
            rate = UNLEARN_TARGETED if m == target_m else UNLEARN_GENERAL
            if self.rng.random() < rate * (0.4 + 0.6 * st.engagement):
                st.misconceptions.discard(m)

        events = [
            Event(
                student_id=sid,
                concept_id=concept_id,
                type=EventType.INSTRUCTION,
                step=self.step_count,
                action=action,
                target_misconception=target_m,
            )
        ]
        if self.rng.random() < prof.participation_tendency * st.engagement * 0.6:
            events.append(self._contribution(sid, concept_id, action))
        return events

    def _contribution(self, sid: str, concept_id: str, action: ActionType) -> Event:
        st = self.state[sid]
        etype = (
            EventType.DISCUSSION
            if action in (ActionType.PEER_DISCUSSION, ActionType.PEER_LEARNING)
            else EventType.QUESTION
        )
        held = sorted(st.misconceptions & set(self._concept_misconceptions(concept_id)))
        signals: list[str] = []
        if self.rng.random() < 0.15:
            intent, conf = Intent.UNCLEAR, float(self.rng.uniform(0.2, 0.5))
        elif held and self.rng.random() < 0.6:
            intent, conf = Intent.MISCONCEPTION_SIGNAL, float(self.rng.uniform(0.5, 0.95))
            signals = [held[int(self.rng.integers(len(held)))]]
        elif not st.known[concept_id]:
            intent, conf = Intent.CONFUSION, float(self.rng.uniform(0.5, 0.95))
        else:
            intent = Intent.CLARIFICATION if self.rng.random() < 0.5 else Intent.EXAMPLE_REQUEST
            conf = float(self.rng.uniform(0.5, 0.95))
        return Event(
            student_id=sid,
            concept_id=concept_id,
            type=etype,
            step=self.step_count,
            voluntary=True,
            intent=intent,
            nlu_confidence=conf,
            misconception_signals=signals,
        )

    def _answer(self, sid: str, concept_id: str, action: ActionType) -> list[Event]:
        st, prof = self.state[sid], self.profiles[sid]
        if self.rng.random() < (1.0 - st.engagement) * 0.5:
            return [Event(student_id=sid, concept_id=concept_id, type=EventType.NO_RESPONSE, step=self.step_count)]

        concept_ms = self._concept_misconceptions(concept_id)
        available = concept_ms[: self.n_distractors]
        held = sorted(st.misconceptions & set(available))
        known = st.known[concept_id]
        hint = action == ActionType.EXERCISE and not known and self.rng.random() < 0.5

        chosen: str | None = None
        if held and self.rng.random() < MISCONCEPTION_EXPRESS:
            correct = False
            chosen = held[int(self.rng.integers(len(held)))]
        else:
            if known:
                correct = bool(self.rng.random() > prof.slip)
            else:
                correct = bool(self.rng.random() < (0.5 if hint else prof.guess))
            if not correct:
                options: list[str | None] = list(available) + [None] * (self.n_distractors - len(available))
                chosen = options[int(self.rng.integers(len(options)))]

        conf = prof.confidence + (0.1 if known else -0.05) + self.rng.normal(0, 0.08)
        event = Event(
            student_id=sid,
            concept_id=concept_id,
            type=EventType.EXERCISE if action == ActionType.EXERCISE else EventType.QUIZ_ANSWER,
            step=self.step_count,
            action=action,
            correct=correct,
            chosen_misconception=chosen,
            available_misconceptions=available,
            n_distractors=self.n_distractors,
            hint_used=hint,
            stated_confidence=float(np.clip(conf, 0.0, 1.0)),
        )
        self._learn(sid, concept_id, PRACTICE_LEARN_RATE[action])
        return [event]

    def _update_engagement(self, sid: str, action: ActionType) -> None:
        st, prof = self.state[sid], self.profiles[sid]
        e = st.engagement + ENGAGEMENT_EFFECT.get(action, 0.0)
        e += 0.1 * (prof.engagement - e)
        e += self.rng.normal(0, 0.02)
        st.engagement = float(np.clip(e, 0.05, 1.0))