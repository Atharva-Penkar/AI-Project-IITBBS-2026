
"""Rule-based classification of student-question intent.

This module classifies intent only. It does not identify concepts,
estimate student mastery, or update student misconception beliefs.
"""

import re

from nlu.taxonomy import INTENTS


# These patterns look for explicit evidence of a possible misconception.
# They are intentionally specific to reduce accidental false positives.
MISCONCEPTION_PATTERNS = [
    (
        "heuristic_as_actual_cost",
        re.compile(
            r"\bis\s+the\s+heuristic\b.*\b(actual|real)\s+cost\b",
            re.IGNORECASE,
        ),
    ),
    (
        "g_and_h_reversed",
        re.compile(
            r"\bis\s+g\s*\(\s*n\s*\)\s+the\s+estimated\s+remaining\s+cost\b.*"
            r"\bh\s*\(\s*n\s*\)\s+the\s+cost\s+so\s+far\b",
            re.IGNORECASE,
        ),
    ),
    (
        "bfs_deepest_first",
        re.compile(
            r"\b(?:bfs|breadth[- ]first search)\b.*"
            r"\b(?:deepest|deepest node)\b.*\bfirst\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dfs_shortest_path",
        re.compile(
            r"\b(?:dfs|depth[- ]first search)\b.*"
            r"\b(?:always|guarantee|guarantees)\b.*\bshortest path\b",
            re.IGNORECASE,
        ),
    ),
    (
        "alpha_beta_changes_result",
        re.compile(
            r"\balpha[- ]beta pruning\b.*"
            r"\b(?:change|changes|alter|alters)\b.*"
            r"\b(?:minimax|optimal|minimax result|result)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reward_equals_policy",
        re.compile(
            r"\breward\b.*\b(?:is|equals|equal to|the same as)\b.*"
            r"\bpolicy\b",
            re.IGNORECASE,
        ),
    ),
    (
    "h_n_confused_with_accumulated_cost",
    re.compile(
        r"\bh\s*\(\s*n\s*\).*"
        r"\b(?:amount of cost already paid|"
        r"cost already paid|cost so far)\b",
        re.IGNORECASE,
    ),
),
(
    "dfs_cheapest_solution",
    re.compile(
        r"\b(?:dfs|depth[- ]first search)\b.*"
        r"\b(?:guarantee|guarantees|always find|find)\b.*"
        r"\b(?:cheapest solution|least-cost solution)\b",
        re.IGNORECASE,
    ),
),
(
    "alpha_beta_changes_best_move",
    re.compile(
        r"\balpha[- ]beta\b.*"
        r"\b(?:change|changes|can change|alter|alters)\b.*"
        r"\b(?:best minimax move|optimal move|minimax result)\b",
        re.IGNORECASE,
    ),
),
(
    "reward_confused_with_policy",
    re.compile(
        r"\breward\b.*"
        r"\b(?:tells|determines|dictates)\b.*"
        r"\bpolicy\b",
        re.IGNORECASE,
    ),
),
    (
        "general_always_claim",
        re.compile(
            r"\b(?:does|do|can)\b.+\balways\b.+\?",
            re.IGNORECASE,
        ),
    ),
        (
        "heuristic_confused_with_travelled_cost",
        re.compile(
            r"\bheuristic(?:\s+value)?\b.*"
            r"\b(?:exact|actual|real)\s+cost\b.*"
            r"\b(?:travelled|traveled|already|so far)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "h_n_confused_with_path_cost",
        re.compile(
            r"\bh\s*\(\s*n\s*\).*"
            r"\b(?:cost from the start|cost so far|"
            r"cost already travelled|cost already traveled)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reward_function_confused_with_policy",
        re.compile(
            r"\breward(?:\s+function)?\b.*"
            r"\b(?:same thing as|same as|equals|equal to)\b.*"
            r"\b(?:an?\s+)?(?:rl\s+)?policy\b",
            re.IGNORECASE,
        ),
    ),
    (
        "bfs_confused_with_depth_first_traversal",
        re.compile(
            r"\b(?:bfs|breadth[- ]first search)\b.*"
            r"\b(?:entire branch|whole branch|one branch)\b.*"
            r"\b(?:before|then)\b.*"
            r"\b(?:other branches|other nodes|next branch)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "heuristic_confused_with_path_cost",
        re.compile(
            r"\b(?:low|lower)\s+heuristic\b.*"
            r"\b(?:path cost|cost so far)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "alpha_beta_different_optimal_move",
        re.compile(
            r"\balpha[- ]beta pruning\b.*"
            r"\b(?:different|change|changes|alter|alters)\b.*"
            r"\b(?:minimax|optimal move|result)\b",
            re.IGNORECASE,
        ),
    ),
        (
        "bfs_uses_heuristic_priority",
        re.compile(
            r"\b(?:bfs|breadth[- ]first search)\b.*"
            r"\b(?:smallest|lowest|minimum|min)\b.*"
            r"\bheuristic\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dfs_guarantees_shortest_path",
        re.compile(
            r"\b(?:dfs|depth[- ]first search)\b.*"
            r"\b(?:must|always|guarantee|guarantees|"
            r"guaranteed|will)\b.*"
            r"\b(?:shortest|cheapest|least[- ]cost)\s+path\b",
            re.IGNORECASE,
        ),
    ),
    (
        "h_n_confused_with_path_cost",
        re.compile(
            r"\bh\s*\(\s*n\s*\)\b.*"
            r"\b(?:cost accumulated|accumulated cost|"
            r"cost from the start|cost so far)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "alpha_beta_changes_minimax_value",
        re.compile(
            r"\balpha[- ]beta(?: pruning)?\b.*"
            r"\b(?:can|could|does|may|might|will)\b.*"
            r"\bchange\b.*"
            r"\b(?:minimax value|game value|optimal move|"
            r"minimax result)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reward_directly_specifies_policy",
        re.compile(
            r"\breward\b.*"
            r"\b(?:directly specifies|specifies|determines|"
            r"tells|dictates)\b.*"
            r"\bpolicy\b",
            re.IGNORECASE,
        ),
    ),
    (
        "heuristic_may_overestimate",
        re.compile(
            r"\b(?:admissible heuristic|heuristic)\b.*"
            r"\b(?:allowed|can|may|might)\b.*"
            r"\boverestimate\b",
            re.IGNORECASE,
        ),
    ),
    (
        "csp_constraint_confused_with_domain",
        re.compile(
            r"\bconstraint\b.*"
            r"\b(?:tells us|specifies|defines|gives)\b.*"
            r"\bpossible values\b",
            re.IGNORECASE,
        ),
    ),
    (
        "bayes_ignores_prior",
        re.compile(
            r"\b(?:bayes'? rule|bayesian updating|"
            r"bayesian reasoning)\b.*"
            r"\b(?:ignore|ignores|discard|discards|"
            r"forget|forgets)\b.*"
            r"\bprior\b",
            re.IGNORECASE,
        ),
    ),
    (
        "astar_priority_confused_with_depth",
        re.compile(
            r"\b(?:priority queue|a[- ]star|a\\*)\b.*"
            r"\b(?:orders|order|sorts|sorted|selects)\b.*"
            r"\b(?:only by|solely by|based only on)\b.*"
            r"\bdepth\b",
            re.IGNORECASE,
        ),
    ),
    (
    "astar_h_value_confused_with_path_cost",
    re.compile(
        r"\ba[- ]star\b.*\bh value\b.*"
        r"\b(?:records|represents|stores)\b.*"
        r"\bcost accumulated from the start\b",
        re.IGNORECASE,
    ),
),
(
    "immediate_reward_guarantees_optimal_policy",
    re.compile(
        r"\bhigh immediate reward\b.*"
        r"\bguarantees?\b.*"
        r"\bpolicy\b.*\boptimal\b",
        re.IGNORECASE,
    ),
),
]

PREREQUISITE_PATTERNS = [
    re.compile(
        r"\b(?:because|since|as)\b.*"
        r"\b(?:don't|do not|can't|cannot)\s+"
        r"(?:yet\s+)?"
        r"(?:understand|know|follow|grasp)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:because|since|as)\b.*"
        r"\b(?:unclear|not clear|unfamiliar|confusing)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:prerequisite|prerequisites|basic concept|basics)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:could follow|difficulty with|stuck on)\b.*"
        r"\b(?:if i first understood|comes from not knowing|"
        r"because .* unfamiliar)\b",
        re.IGNORECASE,
    ),
        re.compile(
        r"\b(?:need|have|want)\s+to\s+"
        r"(?:learn|understand|know|grasp|master)\b.*"
        r"\b(?:before|first|until)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:before|until)\b.*"
        r"\b(?:i can|i could|i understand|"
        r"i follow|studying|learning)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:not understood|not understanding|"
        r"still unfamiliar|unfamiliar with)\b",
        re.IGNORECASE,
    ),
    re.compile(
    r"\bunderstand\b.*\bbefore\b.*\btrying to\b",
    re.IGNORECASE,
),
re.compile(
    r"\bunderstanding\b.*\bfirst\b.*\bwould help me learn\b",
    re.IGNORECASE,
),
re.compile(
    r"\bcannot make progress on\b.*"
    r"\bwithout first understanding\b",
    re.IGNORECASE,
),
]
CONCEPT_CONFUSION_PATTERNS = [
    re.compile(
        r"\b(?:don't understand|do not understand|"
        r"cannot understand|can't understand|"
        r"cannot grasp|can't grasp|"
        r"find .{0,40} difficult to understand|"
        r"struggling to understand|"
        r"struggling to make sense of|"
        r"having trouble understanding|"
        r"confused about|confused by|"
        r"unclear to me|"
        r"still unclear|"
        r"is confusing)\b",
        # r"\b(?:not clear on|unclear about|not clear to me)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:not clear on|unclear about|not clear to me)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:does not make sense to me|"
        r"doesn't make sense to me|"
        r"cannot make sense of|"
        r"can't make sense of)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:keep mixing up|mix up|confuse|confusing)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cannot figure out|can't figure out|"
        r"don't see how|do not see how)\b",
        re.IGNORECASE,
    ),
    # re.compile(
    #     r"\b(?:distinction between|difference between)\b",
    #     re.IGNORECASE,
    # ),
    re.compile(
        r"\b(?:distinction between|difference between)\b.*"
        r"\b(?:path cost|estimated remaining cost|"
        r"heuristic|g\s*\(\s*n\s*\)|h\s*\(\s*n\s*\)|"
        r"a[- ]star|a\*)\b",
        re.IGNORECASE,
    ),
    
]

# EXAMPLE_PATTERNS = [
#     re.compile(
#         r"\b(?:give|show|provide|walk me through|"
#         r"walk through|demonstrate|illustrate)\b.*"
#         r"\b(?:example|worked example|demonstration|"
#         r"game tree|map.colou?ring|update with numbers)\b",
#         re.IGNORECASE,
#     ),
#     re.compile(
#         # r"\b(?:worked example|sample problem|illustration)\b",
#         r"\b(?:example|worked example|demonstration|"
#         r"game tree|map.colou?ring|update with numbers|episode)\b",
#         re.IGNORECASE,
#     ),
# ]
EXAMPLE_PATTERNS = [
    re.compile(
        r"\b(?:give|show|provide|walk me through|walk through|"
        r"demonstrate|illustrate)\b.*"
        r"\b(?:example|worked example|demonstration|game tree|"
        r"map.colou?ring|update with numbers|episode)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:worked example|sample problem|"
        r"illustration)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:could we trace|please illustrate)\b",
        re.IGNORECASE,
    ),
        re.compile(
        r"\b(?:can we|could we|can you|could you)\b.*"
        r"\b(?:use numbers|work through|show a case|"
        r"walk me through|illustrate|demonstrate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:show me|show|give me|provide)\b.*"
        r"\b(?:simple graph|small example|tiny game tree|"
        r"concrete example|worked example)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bplease walk me through\b",
        re.IGNORECASE,
    ),
]
PROCEDURE_PATTERNS = [
    re.compile(
        r"\bhow\s+(?:do|can|should)\s+i\s+"
        r"(?:implement|write|code|use|execute|apply|solve|"
        r"construct|update|compute|calculate|insert|trace|"
        r"formulate|determine)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat procedure is used to\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:steps to|what steps should i follow to|"
        r"procedure for|implementation of)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow to\s+(?:implement|write|code|use|execute|"
        r"apply|solve|construct|update|compute|calculate|"
        r"insert|trace|formulate|determine)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat steps should i use to\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat procedure should i follow to\b",
        re.IGNORECASE,
    ),
        re.compile(
        r"\bwhat (?:sequence|process|steps)\b.*"
        r"\b(?:use|follow|for|to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow can i\b.*"
        r"\b(?:systematically solve|check whether|"
        r"determine|reconstruct|calculate|update)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:describe|explain)\s+the\s+procedure\b",
        re.IGNORECASE,
    ),
    re.compile(
    r"\bwhat is the process for\b",
    re.IGNORECASE,
),
re.compile(
    r"\bsteps for performing\b",
    re.IGNORECASE,
),
]

CLARIFICATION_PATTERNS = [
    re.compile(
        r"\bwhat\s+is\b.*"
        r"\b(?:search|heuristic|path cost|minimax|"
        r"alpha[- ]beta|constraint satisfaction problem|"
        r"bayesian reasoning|reinforcement learning|"
        r"priority queue)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+does\b.*\bmean\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdefine\b", re.IGNORECASE),
    re.compile(r"\bexplain\b", re.IGNORECASE),
    re.compile(
        r"\bwhy\s+(?:does|do|is|are|can)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdifference\s+between\b", re.IGNORECASE),
    re.compile(
        r"\bhow\s+is\b.*\bdifferent\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:could you|can you)\s+clarify\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat information does\b.*\bmaintain\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat does\b.*"
        r"\b(?:algorithm|heuristic|path cost|minimax|"
        r"alpha[- ]beta|q-learning|priority queue)\b.*"
        r"\b(?:calculate|compute|return|do)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat information does\b.*\brepresent\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow does\b.*\b(?:update|work|choose|operate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat is meant by\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat exactly counts as\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat exactly does\b.*\bmean\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow should i interpret\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat is the evidence in\b",
        re.IGNORECASE,
    ),
]

def _matches_any(question: str, patterns: list[re.Pattern]) -> bool:
    """Return True if any pattern matches the question."""
    return any(pattern.search(question) for pattern in patterns)


def classify_intent(question: str) -> dict[str, object]:
    """Classify a student's question using ordered, transparent rules.

    Returns a dictionary containing:
      - intent: one of the canonical labels in nlu.taxonomy.INTENTS
      - matched_rule: a short identifier explaining the decision
      - needs_clarification: whether the question is too vague or unsupported

    Rule-based decisions are not calibrated probabilities. This baseline
    deliberately returns the first matching rule.
    """
    if not isinstance(question, str) or not question.strip():
        return {
            "intent": "other",
            "matched_rule": "empty_or_invalid_input",
            "needs_clarification": True,
        }

    text = re.sub(r"\s+", " ", question).strip()

    # Order matters: more specific signals should be checked first.
    for rule_name, pattern in MISCONCEPTION_PATTERNS:
        if pattern.search(text):
            return {
                "intent": "misconception_signal",
                "matched_rule": rule_name,
                "needs_clarification": False,
            }

    if _matches_any(text, PREREQUISITE_PATTERNS):
        return {
            "intent": "prerequisite_confusion",
            "matched_rule": "prerequisite_language",
            "needs_clarification": False,
        }

    if _matches_any(text, EXAMPLE_PATTERNS):
        return {
            "intent": "example_request",
            "matched_rule": "example_request_language",
            "needs_clarification": False,
        }

    if _matches_any(text, PROCEDURE_PATTERNS):
        return {
            "intent": "procedure_question",
            "matched_rule": "procedure_language",
            "needs_clarification": False,
        }

    if _matches_any(text, CONCEPT_CONFUSION_PATTERNS):
        return {
            "intent": "concept_confusion",
            "matched_rule": "difficulty_understanding_language",
            "needs_clarification": False,
        }

    if _matches_any(text, CLARIFICATION_PATTERNS):
        return {
            "intent": "clarification_request",
            "matched_rule": "clarification_language",
            "needs_clarification": False,
        }

    return {
        "intent": "other",
        "matched_rule": "no_supported_rule_matched",
        "needs_clarification": True,
    }


if __name__ == "__main__":
    demo_questions = [
        "I don't understand how A* search works.",
        "What does the heuristic function mean?",
        "Can you give me a worked example of A* search?",
        "Does BFS always expand the deepest node first?",
        "I cannot understand A* because I do not understand path cost.",
        "How do I implement breadth-first search using a queue?",
        "What should I do?",
    ]

    for question in demo_questions:
        result = classify_intent(question)
        print(f"Question: {question}")
        print(f"Result:   {result}")
        print()
