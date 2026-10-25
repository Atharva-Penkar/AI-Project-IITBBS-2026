
import unittest

from nlu.intent_classifier import classify_intent
from nlu.taxonomy import INTENTS


class IntentClassifierTests(unittest.TestCase):

    def assert_intent(self, question, expected_intent):
        result = classify_intent(question)

        self.assertEqual(result["intent"], expected_intent)
        self.assertIn(result["intent"], INTENTS)
        self.assertTrue(result["matched_rule"])
        self.assertIsInstance(result["needs_clarification"], bool)

    def test_concept_confusion(self):
        self.assert_intent(
            "I don't understand how A* search works.",
            "concept_confusion",
        )

    def test_clarification_request(self):
        self.assert_intent(
            "What does the heuristic function mean?",
            "clarification_request",
        )

    def test_example_request(self):
        self.assert_intent(
            "Can you give me a worked example of A* search?",
            "example_request",
        )

    def test_heuristic_misconception(self):
        self.assert_intent(
            "Is the heuristic the actual cost of reaching the goal?",
            "misconception_signal",
        )

    def test_reversed_g_and_h_misconception(self):
        self.assert_intent(
            "Is g(n) the estimated remaining cost and h(n) the cost so far?",
            "misconception_signal",
        )

    def test_prerequisite_confusion(self):
        self.assert_intent(
            "I cannot understand A* because I do not understand path cost.",
            "prerequisite_confusion",
        )

    def test_procedure_question(self):
        self.assert_intent(
            "How do I implement breadth-first search using a queue?",
            "procedure_question",
        )

    def test_bfs_misconception(self):
        self.assert_intent(
            "Does BFS always expand the deepest node first?",
            "misconception_signal",
        )

    def test_dfs_misconception(self):
        self.assert_intent(
            "Does DFS always find the shortest path?",
            "misconception_signal",
        )

    def test_alpha_beta_clarification(self):
        self.assert_intent(
            "Why does alpha-beta pruning give the same result as minimax?",
            "clarification_request",
        )

    def test_alpha_beta_misconception(self):
        self.assert_intent(
            "Does alpha-beta pruning change the optimal minimax result?",
            "misconception_signal",
        )

    def test_unrelated_vague_question(self):
        result = classify_intent("What should I do?")
        self.assertEqual(result["intent"], "other")
        self.assertTrue(result["needs_clarification"])

    def test_empty_input(self):
        result = classify_intent("   ")
        self.assertEqual(result["intent"], "other")
        self.assertTrue(result["needs_clarification"])

    def test_non_string_input(self):
        result = classify_intent(None)
        self.assertEqual(result["intent"], "other")
        self.assertTrue(result["needs_clarification"])
    
    def test_alternative_concept_confusion_phrases(self):
        questions = [
            "I am struggling to understand how depth-first search works.",
            "I cannot grasp how Bayesian reasoning updates beliefs.",
            "I find breadth-first search difficult to understand.",
            "I am struggling to make sense of reinforcement learning.",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assert_intent(question, "concept_confusion")

    def test_alternative_example_request_phrases(self):
        questions = [
            "Can you walk through a simple depth-first search example?",
            "Could you demonstrate alpha-beta pruning on a game tree?",
            "Can you illustrate a constraint satisfaction problem with map colouring?",
            "Can you demonstrate one Q-learning update with numbers?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assert_intent(question, "example_request")

    def test_alternative_procedure_question_phrases(self):
        questions = [
            "What steps should I follow to calculate an A-star node score?",
            "How can I construct a game tree for minimax?",
            "How do I update alpha and beta while traversing a tree?",
            "How do I compute the posterior probability using Bayes' rule?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assert_intent(question, "procedure_question")

    def test_unrelated_what_is_question(self):
        result = classify_intent("What is the capital of Japan?")
        self.assertEqual(result["intent"], "other")
        self.assertTrue(result["needs_clarification"])



if __name__ == "__main__":
    unittest.main()
