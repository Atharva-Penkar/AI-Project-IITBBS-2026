
"""Canonical labels and definitions for the NLU baseline.

This file defines supported categories only. It does not classify questions.
"""

INTENTS = {
    "concept_confusion": {
        "description": "The student explicitly expresses difficulty understanding a concept.",
    },
    "clarification_request": {
        "description": "The student asks for clarification of a term, definition, or explanation.",
    },
    "example_request": {
        "description": "The student asks for an example, demonstration, or worked illustration.",
    },
    "misconception_signal": {
        "description": "The question provides evidence of a potentially incorrect conceptual belief.",
    },
    "prerequisite_confusion": {
        "description": "The student appears to struggle with a prerequisite concept.",
    },
    "procedure_question": {
        "description": "The student asks how to execute an algorithm or solve a problem.",
    },
    "other": {
        "description": "The question is unsupported, unrelated, or insufficiently informative.",
    },
}


CONCEPTS = {
    "breadth_first_search": {
        "name": "Breadth-First Search",
        "aliases": ["BFS", "breadth first search"],
        "description": "Search that explores nodes level by level.",
    },
    "depth_first_search": {
        "name": "Depth-First Search",
        "aliases": ["DFS", "depth first search"],
        "description": "Search that explores a branch deeply before backtracking.",
    },
    "astar_search": {
        "name": "A* Search",
        "aliases": ["A*", "A star", "A-star"],
        "description": "Informed search using path cost and heuristic estimates.",
    },
    "heuristic": {
        "name": "Heuristic",
        "aliases": ["heuristic", "heuristic function", "h(n)"],
        "description": "An estimate of the remaining cost to a goal.",
    },
    "path_cost": {
        "name": "Path Cost",
        "aliases": ["path cost", "g(n)", "cost so far", "actual cost so far"],
        "description": "The accumulated cost along a path to the current node.",
    },
    "priority_queue": {
        "name": "Priority Queue",
        "aliases": ["priority queue", "priority queues"],
        "description": "A data structure that retrieves an element by priority.",
    },
    "minimax": {
        "name": "Minimax",
        "aliases": ["minimax", "min-max"],
        "description": "A decision rule for adversarial search.",
    },
    "alpha_beta_pruning": {
        "name": "Alpha-Beta Pruning",
        "aliases": ["alpha-beta pruning", "alpha beta pruning", "alpha-beta"],
        "description": "Prunes minimax branches that cannot affect the result.",
    },
    "constraint_satisfaction_problem": {
        "name": "Constraint Satisfaction Problem",
        "aliases": ["CSP", "constraint satisfaction problem", "constraint satisfaction"],
        "description": "A problem defined by variables, domains, and constraints.",
    },
    "bayesian_reasoning": {
        "name": "Bayesian Reasoning",
        "aliases": ["Bayes rule", "Bayes theorem", "Bayesian reasoning"],
        "description": "Updating beliefs using evidence and probabilities.",
    },
    "reinforcement_learning": {
        "name": "Reinforcement Learning",
        "aliases": ["reinforcement learning", "Q-learning", "SARSA"],
        "description": "Learning behaviour through interaction and rewards.",
    },
}


MISCONCEPTIONS = {
    "ASTAR_HEURISTIC_VS_PATH_COST": {
        "description": (
            "Confusing estimated remaining cost with accumulated path cost."
        ),
        "related_concepts": ["astar_search", "heuristic", "path_cost"],
    },
    "ASTAR_G_VS_H": {
        "description": (
            "Treating g(n), the cost accumulated so far, as equivalent "
            "to h(n), the estimated remaining cost."
        ),
        "related_concepts": ["astar_search", "heuristic", "path_cost"],
    },
    "BFS_USES_DEPTH_PRIORITY": {
        "description": (
            "Believing breadth-first search prioritises the deepest node."
        ),
        "related_concepts": ["breadth_first_search"],
    },
    "DFS_GUARANTEES_SHORTEST_PATH": {
        "description": (
            "Believing depth-first search generally guarantees a shortest path."
        ),
        "related_concepts": ["depth_first_search"],
    },
    "ALPHA_BETA_CHANGES_MINIMAX_RESULT": {
        "description": (
            "Believing alpha-beta pruning changes the minimax-optimal result."
        ),
        "related_concepts": ["alpha_beta_pruning", "minimax"],
    },
    "RL_REWARD_EQUALS_POLICY": {
        "description": (
            "Confusing a reward signal with the policy that selects actions."
        ),
        "related_concepts": ["reinforcement_learning"],
    },
}
