
"""Rule-based detection of possible student misconceptions.

This module reports textual evidence of known misconception patterns.
It does not determine whether a student truly holds a misconception,
estimate mastery, or update probabilistic student models.
"""

import re

from nlu.taxonomy import MISCONCEPTIONS


# Each rule maps to an existing canonical misconception ID.
# Keep these patterns explicit and conservative.
MISCONCEPTION_RULES = {
    "ASTAR_HEURISTIC_VS_PATH_COST": re.compile(
        r"\bis\s+(?:the\s+)?heuristic\b.{0,80}"
        r"\b(?:actual|real)\s+cost\b",
        re.IGNORECASE,
    ),

    "ASTAR_G_VS_H": re.compile(
        r"\bis\s+g\s*\(\s*n\s*\)\s+the\s+estimated\s+remaining\s+cost\b"
        r".{0,80}"
        r"\bh\s*\(\s*n\s*\)\s+the\s+cost\s+so\s+far\b",
        re.IGNORECASE,
    ),

    "BFS_USES_DEPTH_PRIORITY": re.compile(
        r"\b(?:bfs|breadth[- ]first search)\b.{0,80}"
        r"\b(?:deepest node|deepest)\b.{0,20}\bfirst\b",
        re.IGNORECASE,
    ),

    "DFS_GUARANTEES_SHORTEST_PATH": re.compile(
        r"\b(?:dfs|depth[- ]first search)\b.{0,80}"
        r"\b(?:always|guarantee|guarantees)\b.{0,40}"
        r"\bshortest path\b",
        re.IGNORECASE,
    ),

    "ALPHA_BETA_CHANGES_MINIMAX_RESULT": re.compile(
        r"\balpha[- ]beta pruning\b.{0,80}"
        r"\b(?:change|changes|alter|alters)\b.{0,40}"
        r"\b(?:optimal minimax result|minimax result|result)\b",
        re.IGNORECASE,
    ),

    "RL_REWARD_EQUALS_POLICY": re.compile(
        r"\breward\b.{0,50}"
        r"\b(?:is|equals|equal to|the same as)\b.{0,30}"
        r"\bpolicy\b",
        re.IGNORECASE,
    ),
}


def detect_misconceptions(question: str) -> dict[str, object]:
    """Detect known misconception patterns in a student question.

    Returns:
      has_potential_misconception: whether any rule matched
      detections: list of matched misconception IDs, descriptions,
                  related concepts, rules, and textual evidence

    An empty detection list means that none of the current rules matched.
    It does not prove that the student has no misconception.
    """
    if not isinstance(question, str) or not question.strip():
        return {
            "has_potential_misconception": False,
            "detections": [],
            "input_valid": False,
        }

    text = re.sub(r"\s+", " ", question).strip()
    detections = []

    for misconception_id, pattern in MISCONCEPTION_RULES.items():
        match = pattern.search(text)

        if match is None:
            continue

        definition = MISCONCEPTIONS[misconception_id]

        detections.append({
            "misconception_id": misconception_id,
            "description": definition["description"],
            "related_concepts": definition["related_concepts"],
            "matched_rule": misconception_id,
            "evidence": match.group(0),
            "start": match.start(),
            "end": match.end(),
        })

    return {
        "has_potential_misconception": bool(detections),
        "detections": detections,
        "input_valid": True,
    }


if __name__ == "__main__":
    demo_questions = [
        "Is the heuristic the actual cost of reaching the goal?",
        "Is g(n) the estimated remaining cost and h(n) the cost so far?",
        "Does BFS always expand the deepest node first?",
        "Does DFS always find the shortest path?",
        "Does alpha-beta pruning change the optimal minimax result?",
        "Is reward the same as policy?",
        "Why does alpha-beta pruning give the same result as minimax?",
        "What should I do?",
    ]

    for question in demo_questions:
        result = detect_misconceptions(question)
        print(f"Question: {question}")
        print(f"Result:   {result}")
        print()
