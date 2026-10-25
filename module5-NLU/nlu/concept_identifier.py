
"""Rule-based identification of concepts in student questions.

This module identifies concepts explicitly mentioned in a question.
It does not classify intent, diagnose mastery, or infer misconceptions.
"""

import re

from nlu.taxonomy import CONCEPTS


def _find_alias_matches(question: str) -> list[dict[str, object]]:
    """Find concept aliases in the question.

    Longer aliases are considered first when matches overlap, so that
    'heuristic function' is preferred over the shorter 'heuristic'.
    Matching is case-insensitive.
    """
    candidates = []

    for concept_id, concept in CONCEPTS.items():
        for alias in concept["aliases"]:
            # Escape special regex characters such as the '*' in 'A*'.
            # Word-character boundaries avoid matching aliases inside
            # unrelated words.
            pattern = re.compile(
                rf"(?<!\w){re.escape(alias)}(?!\w)",
                re.IGNORECASE,
            )

            for match in pattern.finditer(question):
                candidates.append({
                    "concept_id": concept_id,
                    "alias": alias,
                    "start": match.start(),
                    "end": match.end(),
                    "matched_text": match.group(0),
                })

    # At the same starting position, prefer the longest matching alias.
    candidates.sort(
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
        )
    )

    selected = []

    for candidate in candidates:
        overlaps = any(
            candidate["start"] < existing["end"]
            and existing["start"] < candidate["end"]
            for existing in selected
        )

        if not overlaps:
            selected.append(candidate)

    # Report matches in the order in which they appear in the question.
    selected.sort(key=lambda item: item["start"])
    return selected


def identify_concepts(question: str) -> dict[str, object]:
    """Identify explicitly mentioned concepts in a student question.

    Returns:
      primary_concept: first distinct concept mentioned, or None
      related_concepts: other distinct concepts, in mention order
      matches: matched concept IDs, aliases, and text spans
      needs_clarification: True if no supported concept was found

    This is deterministic phrase matching, not a probability estimate.
    """
    if not isinstance(question, str) or not question.strip():
        return {
            "primary_concept": None,
            "related_concepts": [],
            "matches": [],
            "needs_clarification": True,
        }

    matches = _find_alias_matches(question)

    # A concept may be mentioned multiple times. Keep its first occurrence
    # for the primary/related concept lists while retaining match evidence.
    concept_ids = list(dict.fromkeys(
        match["concept_id"] for match in matches
    ))

    primary_concept = concept_ids[0] if concept_ids else None
    related_concepts = concept_ids[1:]

    return {
        "primary_concept": primary_concept,
        "related_concepts": related_concepts,
        "matches": matches,
        "needs_clarification": not bool(concept_ids),
    }


if __name__ == "__main__":
    demo_questions = [
        "I don't understand A* search.",
        "What is the difference between BFS and DFS?",
        "Is g(n) the estimated remaining cost and h(n) the cost so far?",
        "How do I implement BFS using a queue?",
        "What should I do?",
    ]

    for question in demo_questions:
        result = identify_concepts(question)
        print(f"Question: {question}")
        print(f"Result:   {result}")
        print()
