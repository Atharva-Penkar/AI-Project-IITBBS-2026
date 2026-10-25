
import json
import unittest
from pathlib import Path

from nlu.taxonomy import CONCEPTS, INTENTS, MISCONCEPTIONS


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "seed_questions.jsonl"


class TaxonomyTests(unittest.TestCase):

    def test_intent_taxonomy_contains_expected_labels(self):
        expected = {
            "concept_confusion",
            "clarification_request",
            "example_request",
            "misconception_signal",
            "prerequisite_confusion",
            "procedure_question",
            "other",
        }
        self.assertEqual(set(INTENTS), expected)

    def test_concept_ids_are_valid(self):
        self.assertGreaterEqual(len(CONCEPTS), 10)

        for concept_id, concept in CONCEPTS.items():
            with self.subTest(concept_id=concept_id):
                self.assertTrue(concept_id)
                self.assertTrue(concept["name"])
                self.assertIsInstance(concept["aliases"], list)
                self.assertTrue(concept["description"])

    def test_misconception_references_are_valid(self):
        self.assertGreaterEqual(len(MISCONCEPTIONS), 5)

        for misconception_id, item in MISCONCEPTIONS.items():
            with self.subTest(misconception_id=misconception_id):
                self.assertTrue(item["description"])
                self.assertTrue(item["related_concepts"])

                for concept_id in item["related_concepts"]:
                    self.assertIn(
                        concept_id,
                        CONCEPTS,
                        f"{misconception_id} references unknown concept "
                        f"{concept_id}",
                    )

    def test_seed_dataset_is_valid(self):
        self.assertTrue(DATASET_PATH.exists())

        with DATASET_PATH.open(encoding="utf-8") as file:
            rows = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        self.assertGreaterEqual(len(rows), 10)

        question_ids = set()

        for row in rows:
            with self.subTest(question_id=row.get("question_id")):
                self.assertTrue(row["question_id"])
                self.assertNotIn(row["question_id"], question_ids)
                question_ids.add(row["question_id"])

                self.assertTrue(row["question_text"].strip())
                self.assertIn(row["intent"], INTENTS)

                primary = row["primary_concept"]
                if primary is not None:
                    self.assertIn(primary, CONCEPTS)

                for concept_id in row["related_concepts"]:
                    self.assertIn(concept_id, CONCEPTS)

                for misconception_id in row["misconception_ids"]:
                    self.assertIn(misconception_id, MISCONCEPTIONS)

                self.assertIsInstance(
                    row["needs_clarification"], bool
                )

                # Every annotated misconception must be relevant
                # to at least one concept identified in the example.
                mentioned_concepts = set(row["related_concepts"])
                if primary is not None:
                    mentioned_concepts.add(primary)

                for misconception_id in row["misconception_ids"]:
                    required_concepts = set(
                        MISCONCEPTIONS[misconception_id][
                            "related_concepts"
                        ]
                    )
                    self.assertTrue(
                        mentioned_concepts.intersection(required_concepts),
                        f"{row['question_id']}: misconception "
                        f"{misconception_id} has no related concept",
                    )

        # Each intent should have at least one seed example.
        observed_intents = {row["intent"] for row in rows}
        self.assertEqual(observed_intents, set(INTENTS))


if __name__ == "__main__":
    unittest.main()
