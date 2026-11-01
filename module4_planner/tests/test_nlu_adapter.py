"""Contract tests for Team 5 NLU -> Team 3 Event schema mapping."""

from shared.schema import Concept, CourseCatalog, Intent, Misconception
from module4_planner.nlu_adapter import nlu_result_to_event


def _catalog() -> CourseCatalog:
    return CourseCatalog(
        concepts=[
            Concept(id="astar", name="A* search"),
            Concept(id="heuristic", name="Heuristic"),
        ],
        misconceptions=[
            Misconception(
                id="confuses_g_and_h", concept_id="astar",
                description="Confuses accumulated path cost and estimated remaining cost",
            )
        ],
    )


def test_team5_intent_and_concept_are_mapped_to_shared_schema() -> None:
    event = nlu_result_to_event(
        "s01",
        {
            "input_valid": True,
            "question": "I do not understand A*.",
            "intent": "concept_confusion",
            "primary_concept": "astar_search",
            "misconceptions": [],
        },
        _catalog(),
    )

    assert event is not None
    assert event.concept_id == "astar"
    assert event.intent == Intent.CONFUSION
    assert event.nlu_confidence == 0.5


def test_nlu_misconception_is_mapped_to_its_canonical_owner() -> None:
    event = nlu_result_to_event(
        "s02",
        {
            "input_valid": True,
            "question": "A* cost question",
            "intent": "misconception_signal",
            "primary_concept": "heuristic",
            "misconceptions": [{"misconception_id": "ASTAR_G_VS_H"}],
        },
        _catalog(),
    )

    assert event is not None
    assert event.concept_id == "astar"
    assert event.misconception_signals == ["confuses_g_and_h"]


def test_invalid_or_unsupported_input_does_not_fabricate_an_event() -> None:
    assert nlu_result_to_event(
        "s01", {"input_valid": False, "intent": "other"}, _catalog()
    ) is None
    assert nlu_result_to_event(
        "s01",
        {"input_valid": True, "intent": "other", "primary_concept": "minimax", "misconceptions": []},
        _catalog(),
    ) is None
