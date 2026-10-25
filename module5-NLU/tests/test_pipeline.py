
import unittest

from nlu.pipeline import analyse_question
from nlu.taxonomy import INTENTS, CONCEPTS, MISCONCEPTIONS


class PipelineTests(unittest.TestCase):

    def test_confusion_question(self):
        result = analyse_question("I don't understand how A* search works.")

        self.assertTrue(result["input_valid"])
        self.assertEqual(result["intent"], "concept_confusion")
        self.assertEqual(result["primary_concept"], "astar_search")
        self.assertFalse(result["has_potential_misconception"])

    def test_misconception_question(self):
        result = analyse_question(
            "Does BFS always expand the deepest node first?"
        )

        self.assertEqual(result["intent"], "misconception_signal")
        self.assertEqual(
            result["primary_concept"],
            "breadth_first_search",
        )
        self.assertTrue(result["has_potential_misconception"])
        self.assertIn(
            "BFS_USES_DEPTH_PRIORITY",
            [
                item["misconception_id"]
                for item in result["misconceptions"]
            ],
        )

    # def test_g_and_h_question_has_related_concepts(self):
    #     result = analyse_question(
    #         "Is g(n) the estimated remaining cost and h(n) the cost so far?"
    #     )

    #     self.assertTrue(result["has_potential_misconception"])
    #     self.assertIn("path_cost", result["related_concepts"])
    #     self.assertIn("heuristic", result["related_concepts"])
    def test_g_and_h_question_identifies_both_concepts(self):
        result = analyse_question(
            "Is g(n) the estimated remaining cost and h(n) the cost so far?"
        )

        self.assertTrue(result["has_potential_misconception"])

        # path_cost is the primary concept identified from g(n).
        self.assertEqual(result["primary_concept"], "path_cost")

        # heuristic is also identified in the question.
        self.assertIn("heuristic", result["related_concepts"])

        # The misconception detector associates both concepts with the error.
        misconception_ids = [
            item["misconception_id"]
            for item in result["misconceptions"]
        ]
        self.assertIn("ASTAR_G_VS_H", misconception_ids)

        detected_concepts = set()
        for item in result["misconceptions"]:
            detected_concepts.update(item["related_concepts"])

        self.assertIn("path_cost", detected_concepts)
        self.assertIn("heuristic", detected_concepts)
    def test_vague_question(self):
        result = analyse_question("What should I do?")

        self.assertEqual(result["intent"], "other")
        self.assertIsNone(result["primary_concept"])
        self.assertTrue(result["needs_clarification"])
        self.assertFalse(result["has_potential_misconception"])

    def test_empty_input(self):
        result = analyse_question("")

        self.assertFalse(result["input_valid"])
        self.assertTrue(result["needs_clarification"])

    def test_invalid_input(self):
        result = analyse_question(None)

        self.assertFalse(result["input_valid"])
        self.assertIsNone(result["question"])

    def test_all_intents_are_canonical(self):
        questions = [
            "I don't understand A*.",
            "What is a heuristic?",
            "Give me an example of BFS.",
            "Does DFS always find the shortest path?",
            "I don't understand A* because I don't understand path cost.",
            "How do I implement BFS?",
            "What should I do?",
        ]

        for question in questions:
            result = analyse_question(question)
            with self.subTest(question=question):
                self.assertIn(result["intent"], INTENTS)

    def test_all_returned_concepts_are_valid(self):
        result = analyse_question(
            "Is the heuristic the actual cost of reaching the goal?"
        )

        concept_ids = [result["primary_concept"]]
        concept_ids.extend(result["related_concepts"])

        for concept_id in concept_ids:
            if concept_id is not None:
                self.assertIn(concept_id, CONCEPTS)

    def test_all_returned_misconceptions_are_valid(self):
        result = analyse_question(
            "Does BFS always expand the deepest node first?"
        )

        for item in result["misconceptions"]:
            self.assertIn(item["misconception_id"], MISCONCEPTIONS)

    def test_result_is_json_serializable(self):
        import json

        result = analyse_question(
            "Does BFS always expand the deepest node first?"
        )
        encoded = json.dumps(result)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["intent"], result["intent"])


if __name__ == "__main__":
    unittest.main()
