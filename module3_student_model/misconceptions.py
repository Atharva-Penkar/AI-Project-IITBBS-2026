"""Misconception beliefs.

Each (student, misconception) has its own probability, kept separate from
mastery. A student can lack knowledge without holding any misconception.

Answer model used for the likelihoods:
- A student who holds misconception M answers with M's wrong option with
  probability `express`; otherwise they behave like a student without M.
- A student without M answers correctly with probability
  q = P(known)(1 - slip) + (1 - P(known)) guess, and otherwise picks one of
  the wrong options at random.

So picking M's wrong option is strong evidence for M, and a correct answer or
a generic wrong option is evidence against M. Picking a DIFFERENT
misconception's wrong option leaves M unchanged, because students can hold
several misconceptions at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _clip(p: float) -> float:
    return min(max(p, 1e-6), 1.0 - 1e-6)


@dataclass(frozen=True)
class MisconceptionParams:
    prior: float = 0.15
    express: float = 0.7
    signal_if_held: float = 0.6
    signal_if_not_held: float = 0.1
    unlearn_targeted: float = 0.35
    unlearn_general: float = 0.05


@dataclass
class MisconceptionTracker:
    params: MisconceptionParams = field(default_factory=MisconceptionParams)
    priors: dict[str, float] = field(default_factory=dict)
    _p: dict[tuple[str, str], float] = field(default_factory=dict)
    _picks: dict[tuple[str, str], int] = field(default_factory=dict)
    _tested: dict[tuple[str, str], int] = field(default_factory=dict)
    _signals: dict[tuple[str, str], int] = field(default_factory=dict)

    def get(self, student_id: str, misconception_id: str) -> float:
        key = (student_id, misconception_id)
        if key not in self._p:
            self._p[key] = self.priors.get(misconception_id, self.params.prior)
        return self._p[key]

    def _bayes(self, student_id: str, misconception_id: str, like_held: float, like_not: float) -> float:
        p = self.get(student_id, misconception_id)
        num = p * like_held
        p_new = _clip(num / (num + (1.0 - p) * like_not))
        self._p[(student_id, misconception_id)] = p_new
        return p_new

    def update_from_answer(
        self,
        student_id: str,
        concept_misconceptions: list[str],
        correct: bool,
        chosen_misconception: str | None,
        p_mastery: float,
        guess: float,
        slip: float,
        available_misconceptions: list[str] | None = None,
        n_distractors: int = 3,
    ) -> None:
        """Update every misconception the question can reveal.

        `p_mastery` must be the mastery estimate before this answer was applied.
        """
        e = self.params.express
        nd = max(n_distractors, 1)
        q = p_mastery * (1.0 - slip) + (1.0 - p_mastery) * guess
        w = 1.0 - q
        for m in concept_misconceptions:
            if available_misconceptions is not None and m not in available_misconceptions:
                continue
            key = (student_id, m)
            self._tested[key] = self._tested.get(key, 0) + 1
            if correct:
                like_held, like_not = (1.0 - e) * q, q
            elif chosen_misconception == m:
                like_held, like_not = e + (1.0 - e) * w / nd, w / nd
                self._picks[key] = self._picks.get(key, 0) + 1
            elif chosen_misconception is not None:
                continue
            else:
                like_held, like_not = (1.0 - e) * w / nd, w / nd
            self._bayes(student_id, m, like_held, like_not)

    def update_from_signal(self, student_id: str, misconception_id: str, nlu_confidence: float) -> None:
        """A question or discussion contribution that hints at a misconception.

        The update is scaled by the NLU confidence: at confidence 0 nothing changes.
        """
        c = nlu_confidence
        like_held = c * self.params.signal_if_held + (1.0 - c)
        like_not = c * self.params.signal_if_not_held + (1.0 - c)
        self._bayes(student_id, misconception_id, like_held, like_not)
        key = (student_id, misconception_id)
        self._signals[key] = self._signals.get(key, 0) + 1

    def apply_instruction(
        self,
        student_id: str,
        concept_misconceptions: list[str],
        target_misconception: str | None,
        strength: float,
    ) -> None:
        """Teaching can remove misconceptions; targeted teaching removes them faster."""
        if strength <= 0:
            return
        for m in concept_misconceptions:
            rate = self.params.unlearn_targeted if m == target_misconception else self.params.unlearn_general
            p = self.get(student_id, m)
            self._p[(student_id, m)] = _clip(p * (1.0 - rate * strength))

    def for_concept(self, student_id: str, concept_misconceptions: list[str]) -> dict[str, float]:
        return {m: self.get(student_id, m) for m in concept_misconceptions}

    def evidence(self, student_id: str, misconception_id: str) -> tuple[int, int, int]:
        """(times its wrong option was picked, times it was tested, NLU signals)."""
        key = (student_id, misconception_id)
        return self._picks.get(key, 0), self._tested.get(key, 0), self._signals.get(key, 0)