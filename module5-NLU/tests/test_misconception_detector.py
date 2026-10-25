
import unittest

from nlu.misconception_detector import (
    MISCONCEPTION_RULES,
    detect_misconceptions,
)
from nlu.taxonomy import MISCONCEPTIONS, CONCEPTS


class MisconceptionDetectorTests(unittest.TestCase):

    def assert_detects(self, question, expected_id):
        result = detect_misconceptions(question)

        self.assertTrue(result["input_valid"])
        self.assertTrue(result["has_potential_misconception"])

        detected_ids = [
            item["misconception_id"]
            for item in result["detections"]
        ]
        self.assertIn(expected_id, detected_ids)

    def test_heuristic_actual_cost(self):
        self.assert_detects(
            "Is the heuristic the actual cost of reaching the goal?",
            "ASTAR_HEURISTIC_VS_PATH_COST",
        )

    def test_g_and_h_reversed(self):
        self.assert_detects(
            "Is g(n) the estimated remaining cost and h(n) the cost so far?",
            "ASTAR_G_VS_H",
        )

    def test_bfs_depth_priority(self):
        self.assert_detects(
            "Does BFS always expand the deepest node first?",
            "BFS_USES_DEPTH_PRIORITY",
        )

    def test_dfs_shortest_path(self):
        self.assert_detects(
            "Does DFS always find the shortest path?",
            "DFS_GUARANTEES_SHORTEST_PATH",
        )

    def test_alpha_beta_changes_result(self):
        self.assert_detects(
            "Does alpha-beta pruning change the optimal minimax result?",
            "ALPHA_BETA_CHANGES_MINIMAX_RESULT",
        )

    def test_reward_equals_policy(self):
        self.assert_detects(
            "Is reward the same as policy?",
            "RL_REWARD_EQUALS_POLICY",
        )

    def test_normal_clarification_does_not_trigger_rule(self):
        result = detect_misconceptions(
            "Why does alpha-beta pruning give the same result as minimax?"
        )
        self.assertFalse(result["has_potential_misconception"])
        self.assertEqual(result["detections"], [])

    def test_unrelated_question_has_no_detection(self):
        result = detect_misconceptions("What should I do?")
        self.assertFalse(result["has_potential_misconception"])
        self.assertEqual(result["detections"], [])
        self.assertTrue(result["input_valid"])

    def test_empty_input(self):
        result = detect_misconceptions("   ")
        self.assertFalse(result["has_potential_misconception"])
        self.assertEqual(result["detections"], [])
        self.assertFalse(result["input_valid"])

    def test_invalid_input(self):
        result = detect_misconceptions(None)
        self.assertFalse(result["has_potential_misconception"])
        self.assertEqual(result["detections"], [])
        self.assertFalse(result["input_valid"])

    def test_evidence_matches_question(self):
        question = "Does BFS always expand the deepest node first?"
        result = detect_misconceptions(question)

        self.assertTrue(result["detections"])
        detection = result["detections"][0]

        self.assertEqual(
            question[detection["start"]:detection["end"]],
            detection["evidence"],
        )

    def test_rule_ids_match_taxonomy(self):
        self.assertEqual(
            set(MISCONCEPTION_RULES),
            set(MISCONCEPTIONS),
        )

    def test_related_concepts_exist(self):
        questions = [
            "Is the heuristic the actual cost of reaching the goal?",
            "Is g(n) the estimated remaining cost and h(n) the cost so far?",
            "Does BFS always expand the deepest node first?",
            "Does DFS always find the shortest path?",
            "Does alpha-beta pruning change the optimal minimax result?",
            "Is reward the same as policy?",
        ]

        for question in questions:
            result = detect_misconceptions(question)

            for detection in result["detections"]:
                for concept_id in detection["related_concepts"]:
                    with self.subTest(
                        question=question,
                        concept_id=concept_id,
                    ):
                        self.assertIn(concept_id, CONCEPTS)

    def test_matching_is_case_insensitive(self):
        self.assert_detects(
            "does bfs always expand the deepest node first?",
            "BFS_USES_DEPTH_PRIORITY",
        )


if __name__ == "__main__":
    unittest.main()
