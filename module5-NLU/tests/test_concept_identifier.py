
import unittest

from nlu.concept_identifier import identify_concepts
from nlu.taxonomy import CONCEPTS


class ConceptIdentifierTests(unittest.TestCase):

    def test_identifies_astar(self):
        result = identify_concepts("I don't understand A* search.")
        self.assertEqual(result["primary_concept"], "astar_search")
        self.assertEqual(result["related_concepts"], [])
        self.assertFalse(result["needs_clarification"])

    def test_matching_is_case_insensitive(self):
        result = identify_concepts("How does bfs work?")
        self.assertEqual(
            result["primary_concept"],
            "breadth_first_search",
        )

    def test_identifies_multiple_concepts(self):
        result = identify_concepts(
            "What is the difference between BFS and DFS?"
        )
        self.assertEqual(
            result["primary_concept"],
            "breadth_first_search",
        )
        self.assertEqual(
            result["related_concepts"],
            ["depth_first_search"],
        )

    def test_identifies_g_and_h_aliases(self):
        result = identify_concepts(
            "Is g(n) the estimated remaining cost and h(n) the cost so far?"
        )
        self.assertEqual(
            result["primary_concept"],
            "path_cost",
        )
        self.assertIn("heuristic", result["related_concepts"])

    def test_prefers_longer_alias(self):
        result = identify_concepts(
            "What does the heuristic function mean?"
        )
        heuristic_matches = [
            match for match in result["matches"]
            if match["concept_id"] == "heuristic"
        ]
        self.assertEqual(len(heuristic_matches), 1)
        self.assertEqual(
            heuristic_matches[0]["matched_text"],
            "heuristic function",
        )

    def test_repeated_concept_is_not_duplicated(self):
        result = identify_concepts(
            "BFS is breadth-first search, and BFS uses a queue."
        )
        self.assertEqual(
            result["primary_concept"],
            "breadth_first_search",
        )
        self.assertEqual(result["related_concepts"], [])

    def test_identifies_csp(self):
        result = identify_concepts(
            "What is a constraint satisfaction problem?"
        )
        self.assertEqual(
            result["primary_concept"],
            "constraint_satisfaction_problem",
        )

    def test_identifies_reinforcement_learning_alias(self):
        result = identify_concepts(
            "How is Q-learning different from SARSA?"
        )
        self.assertEqual(
            result["primary_concept"],
            "reinforcement_learning",
        )

    def test_no_concept_returns_clarification(self):
        result = identify_concepts("What should I do?")
        self.assertIsNone(result["primary_concept"])
        self.assertEqual(result["related_concepts"], [])
        self.assertTrue(result["needs_clarification"])

    def test_empty_input(self):
        result = identify_concepts("   ")
        self.assertIsNone(result["primary_concept"])
        self.assertTrue(result["needs_clarification"])

    def test_invalid_input(self):
        result = identify_concepts(None)
        self.assertIsNone(result["primary_concept"])
        self.assertTrue(result["needs_clarification"])

    def test_all_identified_concepts_are_in_taxonomy(self):
        result = identify_concepts(
            "A* uses a heuristic. BFS and DFS are search algorithms."
        )
        identified = [result["primary_concept"]]
        identified.extend(result["related_concepts"])

        for concept_id in identified:
            with self.subTest(concept_id=concept_id):
                self.assertIn(concept_id, CONCEPTS)

    def test_match_evidence_contains_offsets(self):
        question = "Explain BFS please."
        result = identify_concepts(question)

        self.assertTrue(result["matches"])
        match = result["matches"][0]
        self.assertEqual(
            question[match["start"]:match["end"]],
            match["matched_text"],
        )


if __name__ == "__main__":
    unittest.main()
