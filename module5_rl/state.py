"""State encoding for the tabular RL policy.

Starts from Team 3's ``discretise`` (mastery, misconception prevalence, engagement,
participation balance, share left behind; each 0/1/2) and adds two features that the
teaching decision depends on:

* ``uncertainty``: is the class-level mastery estimate still uncertain? (0/1) - this
  is what makes a diagnostic quiz worth its time.
* ``time_pace``: remaining time relative to the share of the course still to cover
  (0 = behind schedule, 1 = on track, 2 = ahead) - this is what trades depth for coverage.
"""

from __future__ import annotations

from module3_student_model.aggregate import discretise
from shared.schema import GroupSummary

STATE_DIMS = (
    "mastery",
    "misconceptions",
    "engagement",
    "participation",
    "left_behind",
    "uncertainty",
    "time_pace",
)
STATE_SHAPE = (3, 3, 3, 3, 3, 2, 3)
STATE_LABELS = (
    ("low", "medium", "high"),
    ("rare", "some", "widespread"),
    ("low", "medium", "high"),
    ("dominated", "uneven", "balanced"),
    ("few", "some", "many"),
    ("low", "high"),
    ("behind schedule", "on track", "ahead of schedule"),
)

UNCERTAINTY_CUT = 0.15
PACE_CUTS = (0.7, 1.2)


def time_pace_level(time_left: float, time_budget: float, concepts_remaining: int, n_concepts: int) -> int:
    if time_budget <= 0 or concepts_remaining <= 0:
        return 1
    pace = (max(time_left, 0.0) / time_budget) / (concepts_remaining / n_concepts)
    return 0 if pace < PACE_CUTS[0] else (1 if pace < PACE_CUTS[1] else 2)


def encode_state(
    summary: GroupSummary, concept_id: str, time_left: float, time_budget: float
) -> tuple[int, ...]:
    concept_ids = list(summary.concepts)
    n = len(concept_ids)
    remaining = n - concept_ids.index(concept_id)
    base = discretise(summary, concept_id)
    uncertainty = int(summary.concepts[concept_id].mean_uncertainty >= UNCERTAINTY_CUT)
    return (*base, uncertainty, time_pace_level(time_left, time_budget, remaining, n))


def describe_state(state: tuple[int, ...]) -> str:
    return ", ".join(f"{name}={STATE_LABELS[i][state[i]]}" for i, name in enumerate(STATE_DIMS))
