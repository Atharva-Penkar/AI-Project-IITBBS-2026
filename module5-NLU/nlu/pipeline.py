
"""Integrated NLU pipeline for the classroom digital twin."""

from nlu.intent_classifier import classify_intent
from nlu.concept_identifier import identify_concepts
from nlu.misconception_detector import detect_misconceptions


def analyse_question(question: str) -> dict[str, object]:
    """Run all NLU components and return one structured result."""

    intent_result = classify_intent(question)
    concept_result = identify_concepts(question)
    misconception_result = detect_misconceptions(question)

    # Invalid input should be represented consistently across the pipeline.
    if not isinstance(question, str) or not question.strip():
        return {
            "question": question if isinstance(question, str) else None,
            "input_valid": False,
            "intent": intent_result["intent"],
            "primary_concept": concept_result["primary_concept"],
            "related_concepts": concept_result["related_concepts"],
            "misconceptions": misconception_result["detections"],
            "needs_clarification": True,
        }

    # If a known misconception was detected, make sure its related concepts
    # are available in the integrated result even if the question did not
    # explicitly mention every one of them.
    related_ids = list(concept_result["related_concepts"])
    primary_id = concept_result["primary_concept"]

    for detection in misconception_result["detections"]:
        for concept_id in detection["related_concepts"]:
            if concept_id != primary_id and concept_id not in related_ids:
                related_ids.append(concept_id)

    needs_clarification = (
        intent_result["needs_clarification"]
        or concept_result["needs_clarification"]
    )

    return {
        "question": question,
        "input_valid": True,
        "intent": intent_result["intent"],
        "intent_rule": intent_result["matched_rule"],
        "primary_concept": primary_id,
        "related_concepts": related_ids,
        "misconceptions": misconception_result["detections"],
        "has_potential_misconception": (
            misconception_result["has_potential_misconception"]
        ),
        "needs_clarification": needs_clarification,
    }


if __name__ == "__main__":
    examples = [
        "I don't understand how A* search works.",
        "Is the heuristic the actual cost of reaching the goal?",
        "Does BFS always expand the deepest node first?",
        "What should I do?",
    ]

    for question in examples:
        print(f"\nQuestion: {question}")
        print(analyse_question(question))
