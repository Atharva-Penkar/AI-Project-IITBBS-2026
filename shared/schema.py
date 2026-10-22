"""Data formats shared by all modules."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    """Teaching actions that can be applied to students."""

    LECTURE = "lecture"
    EXAMPLE = "example"
    DIAGNOSTIC_QUIZ = "diagnostic_quiz"
    PEER_DISCUSSION = "peer_discussion"
    PEER_LEARNING = "peer_learning"
    TA_INTERVENTION = "ta_intervention"
    EXERCISE = "exercise"
    REVISION = "revision"
    ASSESSMENT = "assessment"
    NEXT_CONCEPT = "next_concept"


class EventType(str, Enum):
    """Things that happen to one student during a session."""

    INSTRUCTION = "instruction"
    QUIZ_ANSWER = "quiz_answer"
    EXERCISE = "exercise"
    HINT = "hint"
    QUESTION = "question"
    DISCUSSION = "discussion"
    NO_RESPONSE = "no_response"


class Intent(str, Enum):
    """Intents assigned to student questions."""

    CONFUSION = "confusion"
    CLARIFICATION = "clarification"
    MISCONCEPTION_SIGNAL = "misconception_signal"
    EXAMPLE_REQUEST = "example_request"
    UNCLEAR = "unclear"


class Concept(BaseModel):
    id: str
    name: str
    prerequisites: list[str] = Field(default_factory=list)


class Misconception(BaseModel):
    id: str
    concept_id: str
    description: str


class CourseCatalog(BaseModel):
    """Concepts and the misconceptions linked to them."""

    concepts: list[Concept]
    misconceptions: list[Misconception] = Field(default_factory=list)

    def concept_ids(self) -> list[str]:
        return [c.id for c in self.concepts]

    def concept(self, concept_id: str) -> Concept:
        for c in self.concepts:
            if c.id == concept_id:
                return c
        raise KeyError(f"Unknown concept: {concept_id}")

    def misconceptions_for(self, concept_id: str) -> list[Misconception]:
        return [m for m in self.misconceptions if m.concept_id == concept_id]

    def misconception(self, misconception_id: str) -> Misconception:
        for m in self.misconceptions:
            if m.id == misconception_id:
                return m
        raise KeyError(f"Unknown misconception: {misconception_id}")


class Event(BaseModel):
    """One observation about one student.

    - INSTRUCTION: the student received a teaching action (set `action`).
    - QUIZ_ANSWER / EXERCISE: set `correct`, or `score` for partial credit.
      If the student picked a wrong option linked to a misconception, set
      `chosen_misconception`. `available_misconceptions` lists the
      misconceptions the question's wrong options test (None = all of the
      concept's misconceptions); `n_distractors` is the number of wrong options.
    - QUESTION / DISCUSSION: set `intent`, `nlu_confidence` and
      `misconception_signals` from the NLU module.
    - NO_RESPONSE: the student was prompted and did not respond.
    """

    student_id: str
    concept_id: str
    type: EventType
    step: int = 0

    action: ActionType | None = None
    target_misconception: str | None = None

    correct: bool | None = None
    score: float | None = None
    chosen_misconception: str | None = None
    available_misconceptions: list[str] | None = None
    n_distractors: int = 3
    hint_used: bool = False
    stated_confidence: float | None = None

    voluntary: bool = False
    intent: Intent | None = None
    nlu_confidence: float | None = None
    misconception_signals: list[str] = Field(default_factory=list)

    text: str | None = None

    @field_validator("score", "stated_confidence", "nlu_confidence")
    @classmethod
    def _unit_interval(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0 and 1")
        return v


class ConceptEstimate(BaseModel):
    concept_id: str
    p_mastery: float
    uncertainty: float
    n_observations: int
    misconceptions: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class StudentEstimate(BaseModel):
    student_id: str
    concepts: dict[str, ConceptEstimate]
    engagement: float
    participation: float
    confidence: float
    overconfidence: float
    confidently_wrong: bool


class ConceptSummary(BaseModel):
    concept_id: str
    mean_mastery: float
    std_mastery: float
    mean_uncertainty: float
    fraction_below_threshold: float
    misconception_prevalence: dict[str, float] = Field(default_factory=dict)


class GroupSummary(BaseModel):
    student_ids: list[str]
    concepts: dict[str, ConceptSummary]
    mean_engagement: float
    mean_participation: float
    participation_gini: float
    students_left_behind: list[str]
    students_high_uncertainty: list[str]
    confidently_wrong_students: list[str]