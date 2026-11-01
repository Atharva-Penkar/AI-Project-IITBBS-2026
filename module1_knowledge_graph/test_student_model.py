"""Run with: python -m unittest test_student_model -v"""
import unittest

from knowledge_graph import prereqs, diagnose
from student_model import StudentModel, DEFAULTS
from astar_examples import STUDENT_TEXT


class BKTTests(unittest.TestCase):
    def test_correct_answer_raises_mastery(self):
        s = StudentModel("t")
        before = s.mastery["Heuristic"]
        s.observe("Heuristic", correct=True)
        self.assertGreater(s.mastery["Heuristic"], before)

    def test_wrong_answer_lowers_posterior_before_transition(self):
        s = StudentModel("t", params={"p_transit": 0.0})
        s.observe("Heuristic", correct=False)
        self.assertLess(s.mastery["Heuristic"], DEFAULTS["p_init"])

    def test_hint_only_applies_transition(self):
        s = StudentModel("t")
        s.observe("Heuristic", kind="hint")
        self.assertAlmostEqual(s.mastery["Heuristic"], 0.2 + 0.8 * 0.15)

    def test_ambiguous_response_does_not_change_mastery(self):
        s = StudentModel("t")
        s.observe("Heuristic", kind="ambiguous")
        self.assertEqual(s.mastery["Heuristic"], DEFAULTS["p_init"])
        self.assertEqual(s.ambiguous["Heuristic"], 1)

    def test_repeated_correct_answers_reach_mastery(self):
        s = StudentModel("t")
        for _ in range(4):
            s.observe("Heuristic", correct=True)
        self.assertEqual(s.status("Heuristic"), "mastered")

    def test_uncertainty_falls_with_evidence(self):
        s = StudentModel("t")
        self.assertEqual(s.uncertainty("Heuristic"), 1.0)
        for _ in range(5):
            s.observe("Heuristic", correct=True)
        self.assertEqual(s.uncertainty("Heuristic"), 0.0)


class MisconceptionLifecycleTests(unittest.TestCase):
    def test_paraphrased_misconception_is_recorded(self):
        s = StudentModel("t")
        s.observe("Queue_PQ", correct=False, text="A priority queue works first in first out")
        self.assertEqual(list(s.misconceptions["Queue_PQ"]), ["A priority queue always removes the earliest inserted item."])

    def test_misconception_is_resolved_after_mastery(self):
        s = StudentModel("t")
        s.observe("Queue_PQ", correct=False, text=STUDENT_TEXT["fifo"])
        self.assertEqual(s.status("Queue_PQ"), "misconception")
        for _ in range(8):
            s.observe("Queue_PQ", correct=True)
        self.assertEqual(s.status("Queue_PQ"), "mastered")
        self.assertEqual(s.misconceptions["Queue_PQ"], {})
        self.assertTrue(s.resolved["Queue_PQ"])
        self.assertIn("Queue_PQ", s.mastered())

    def test_misconception_stays_while_it_keeps_appearing(self):
        s = StudentModel("t")
        for _ in range(8):
            s.observe("Queue_PQ", correct=True, text=STUDENT_TEXT["fifo"])
        self.assertEqual(s.status("Queue_PQ"), "misconception")
        self.assertFalse(s.resolved["Queue_PQ"])

    def test_misconception_not_cleared_before_mastery(self):
        s = StudentModel("t")
        s.observe("Queue_PQ", correct=False, text=STUDENT_TEXT["fifo"])
        s.observe("Queue_PQ", correct=True)
        self.assertEqual(s.status("Queue_PQ"), "misconception")


class StudentStateTests(unittest.TestCase):
    def test_not_learned_vs_misconception(self):
        s = StudentModel("t")
        s.observe("Path_Cost", correct=False)
        self.assertEqual(s.status("Path_Cost"), "not_learned")
        s.observe("Astar", correct=False, text=STUDENT_TEXT["h_only"])
        self.assertEqual(s.status("Astar"), "misconception")

    def test_misconception_text_is_recorded(self):
        s = StudentModel("t")
        s.observe("Queue_PQ", correct=False, text=STUDENT_TEXT["fifo"])
        self.assertTrue(s.misconceptions["Queue_PQ"])

    def test_to_mastery_only_has_touched_concepts(self):
        s = StudentModel("t")
        s.observe("Heuristic", correct=True)
        self.assertEqual(set(s.to_mastery()), {"Heuristic"})

    def test_study_plan_skips_mastered_prerequisites(self):
        s = StudentModel("t")
        for _ in range(4):
            s.observe("Heuristic", correct=True)
        plan = [c for c, _, _ in s.study_plan("Astar")]
        self.assertNotIn("Heuristic", plan)
        self.assertIn("Path_Cost", plan)
        self.assertEqual(plan[-1], "Astar")

    def test_study_plan_is_foundations_first(self):
        s = StudentModel("t")
        plan = [c for c, _, _ in s.study_plan("Astar")]
        pos = {c: i for i, c in enumerate(plan)}
        for c in plan:
            for p in prereqs[c]:
                if p in pos:
                    self.assertLess(pos[p], pos[c])

    def test_student_diagnosis_uses_their_low_mastery(self):
        s = StudentModel("t")
        for _ in range(3):
            s.observe("Path_Cost", correct=False)
        r = s.diagnose_for_student("Astar")
        self.assertEqual(r["ranked_causes"][0]["concept_id"], "Path_Cost")

    def test_unknown_concept_raises(self):
        with self.assertRaises(KeyError):
            StudentModel("t").observe("Nope", correct=True)

    def test_bad_kind_raises(self):
        with self.assertRaises(ValueError):
            StudentModel("t").observe("Heuristic", kind="guess", correct=True)


if __name__ == "__main__":
    unittest.main()
