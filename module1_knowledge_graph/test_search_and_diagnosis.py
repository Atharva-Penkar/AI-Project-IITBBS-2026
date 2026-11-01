"""Run with: python -m unittest test_search_and_diagnosis -v"""
import unittest

import csv
import json
from pathlib import Path

from knowledge_graph import nodes, prereqs, prerequisite_chain, diagnose, explain, explained_away
from export_csv import build_tables
from search import (bfs_ancestors, dfs_ancestors, missing_prerequisites,
                    search, is_admissible)
from astar_examples import (ADJ, START, GOAL, H_ADMISSIBLE, H_OVERESTIMATE,
                            run_variant, STUDENT_TEXT)


class GraphTests(unittest.TestCase):
    def test_prerequisite_chain_is_prerequisite_first(self):
        for c in nodes:
            chain = prerequisite_chain(c)
            pos = {x: i for i, x in enumerate(chain)}
            for x in chain:
                for p in prereqs[x]:
                    self.assertLess(pos[p], pos[x])

    def test_astar_direct_prerequisites(self):
        self.assertEqual(set(prereqs["Astar"]),
                         {"Path_Cost", "Heuristic", "Queue_PQ", "UCS", "Admissibility"})

    def test_bfs_and_dfs_find_the_same_ancestors(self):
        for c in nodes:
            self.assertEqual(set(bfs_ancestors(prereqs, c)) - {c}, dfs_ancestors(prereqs, c))

    def test_bfs_depth_is_shortest_path_length(self):
        d = bfs_ancestors(prereqs, "Astar")
        self.assertEqual(d["AI_Foundations"], 2)  # via Heuristic, not via Graph_Basics

    def test_missing_prerequisites_stops_at_mastered(self):
        self.assertNotIn("Heuristic", missing_prerequisites(prereqs, "Astar", {"Heuristic"}))
        self.assertEqual(missing_prerequisites(prereqs, "Astar",
                                               {"Path_Cost", "Heuristic", "Queue_PQ", "UCS", "Admissibility"}), [])

    def test_unknown_concept_raises(self):
        with self.assertRaises(KeyError):
            prerequisite_chain("Nope")


class SearchTests(unittest.TestCase):
    def test_correct_astar_finds_optimal_path(self):
        r = run_variant("correct")
        self.assertTrue(r["found"])
        self.assertEqual(r["cost"], 4)
        self.assertEqual(r["path"], ("S", "B", "G"))

    def test_uniform_cost_is_optimal_on_toy_graph(self):
        self.assertEqual(search(ADJ, START, GOAL, key="g")["cost"], 4)

    def test_greedy_heuristic_only_fails(self):
        r = run_variant("h_only")
        self.assertEqual(r["cost"], 6)
        self.assertEqual(r["path"], ("S", "A", "G"))

    def test_fifo_frontier_fails(self):
        self.assertEqual(run_variant("fifo")["cost"], 6)

    def test_inadmissible_heuristic_fails(self):
        self.assertEqual(run_variant("inadmissible")["cost"], 6)

    def test_admissibility_check(self):
        self.assertTrue(is_admissible(ADJ, GOAL, H_ADMISSIBLE))
        self.assertFalse(is_admissible(ADJ, GOAL, H_OVERESTIMATE))

    def test_unreachable_goal(self):
        self.assertFalse(search({"S": []}, "S", "G")["found"])


class DiagnosisTests(unittest.TestCase):
    def test_h_only_error_points_at_astar_misconception(self):
        self.assertEqual(diagnose("Astar", STUDENT_TEXT["h_only"])["ranked_causes"][0]["concept_id"], "Astar")

    def test_fifo_error_points_at_priority_queue(self):
        self.assertEqual(diagnose("Astar", STUDENT_TEXT["fifo"])["ranked_causes"][0]["concept_id"], "Queue_PQ")

    def test_inadmissible_error_points_at_admissibility(self):
        self.assertEqual(diagnose("Astar", STUDENT_TEXT["inadmissible"])["ranked_causes"][0]["concept_id"],
                         "Admissibility")

    def test_path_cost_definition_error(self):
        r = diagnose("Astar", "Path cost is the estimated remaining cost to the goal.")
        self.assertEqual(r["ranked_causes"][0]["concept_id"], "Path_Cost")

    def test_low_mastery_alone_is_enough(self):
        r = diagnose("Astar", mastery={"Path_Cost": 0.1, "Heuristic": 0.9})
        self.assertEqual(r["ranked_causes"][0]["concept_id"], "Path_Cost")
        self.assertEqual(r["ranked_causes"][0]["confidence"], "evidence")

    def test_no_evidence_is_marked_unverified(self):
        r = diagnose("Astar")
        self.assertTrue(all(c["confidence"] == "unverified" for c in r["ranked_causes"]))
        self.assertIn("No clear evidence", r["next_step"])

    def test_causes_come_with_examples_and_exercises(self):
        cause = diagnose("Astar", STUDENT_TEXT["fifo"])["ranked_causes"][0]
        self.assertTrue(cause["examples"] or cause["exercises"])

    def test_bad_inputs_raise(self):
        with self.assertRaises(KeyError):
            diagnose("Nope")
        with self.assertRaises(ValueError):
            diagnose("Astar", mastery={"Path_Cost": 1.5})


class ParaphraseTests(unittest.TestCase):
    """Students rarely repeat the stored sentence word for word."""

    def top(self, target, text):
        return diagnose(target, text)["ranked_causes"][0]

    def test_paraphrase_of_bfs_misconception(self):
        c = self.top("BFS", "BFS always finds the cheapest path in a weighted graph")
        self.assertEqual((c["concept_id"], c["confidence"]), ("BFS", "evidence"))
        self.assertIn("BFS is always the lowest-cost algorithm.", c["reasons"][0])   # reports the stored statement

    def test_paraphrases_of_astar_errors(self):
        self.assertEqual(self.top("Astar", "I thought A* only uses path cost and ignores the heuristic")["concept_id"], "Astar")
        self.assertEqual(self.top("Astar", "Priority queue works first in first out")["concept_id"], "Queue_PQ")
        self.assertEqual(self.top("Astar", "g(n) is just the cost of the last edge")["concept_id"], "Path_Cost")
        self.assertEqual(self.top("Astar", "I used a heuristic that overestimates but A* should still find the best path")["concept_id"],
                         "Admissibility")

    def test_exact_statement_still_matches_fully(self):
        c = self.top("Astar", "A* uses only g(n).")
        self.assertEqual(c["concept_id"], "Astar")
        self.assertIn("100%", c["reasons"][0])
        self.assertIn("A* uses only g(n).", c["reasons"][0])

    def test_neutral_questions_do_not_match(self):
        for text in ["My A* found a longer path and I don't know why", "I don't understand the heuristic part",
                     "How do I implement A* in Python?", "The queue was empty at the end",
                     "Can you explain the path cost again?", "Why is my graph search slow?"]:
            r = diagnose("Astar", text)
            self.assertTrue(all(c["confidence"] == "unverified" for c in r["ranked_causes"]), text)

    def test_every_misconception_has_paraphrases_and_they_match_their_own_concept(self):
        for n in nodes.values():
            for m in n["misconceptions"]:
                paras = n["misconception_paraphrases"].get(m, [])
                self.assertGreaterEqual(len(paras), 2, (n["id"], m))
                for ph in paras:
                    c = diagnose(n["id"], ph)["ranked_causes"][0]
                    self.assertEqual((c["concept_id"], c["confidence"]), (n["id"], "evidence"), ph)


class ExplainingAwayTests(unittest.TestCase):
    def test_overestimating_example_has_one_cause_not_two(self):
        causes = {c["concept_id"]: c for c in diagnose("Astar", STUDENT_TEXT["inadmissible"])["ranked_causes"]}
        self.assertEqual(diagnose("Astar", STUDENT_TEXT["inadmissible"])["ranked_causes"][0]["concept_id"], "Admissibility")
        self.assertEqual(causes["Admissibility"]["score"], 5.0)
        self.assertEqual(causes["Astar"]["score"], 0.0)
        self.assertEqual(causes["Astar"]["confidence"], "unverified")
        self.assertIn("already explained", causes["Astar"]["reasons"][0])

    def test_two_independent_misconceptions_are_both_kept(self):
        text = STUDENT_TEXT["fifo"] + " " + STUDENT_TEXT["h_only"]
        causes = {c["concept_id"]: c for c in diagnose("Astar", text)["ranked_causes"]}
        self.assertEqual(causes["Queue_PQ"]["confidence"], "evidence")
        self.assertEqual(causes["Astar"]["confidence"], "evidence")

    def test_explained_away_helper(self):
        matches = {"A": (1.0, frozenset({"x", "y", "z"})), "B": (0.6, frozenset({"x", "y"})), "C": (0.6, frozenset({"q", "x"}))}
        self.assertEqual(explained_away(matches), {"B": "A"})
        self.assertEqual(explained_away({"A": (1.0, frozenset({"x"}))}), {})


class OutputAndDataTests(unittest.TestCase):
    def test_explain_calls_it_study_order_not_a_chain_of_dependencies(self):
        text = explain(diagnose("Astar"))
        self.assertIn("Study order (prerequisites first)", text)
        self.assertNotIn("Prerequisite chain", text)

    def test_scope_matches_the_graph(self):
        data = json.loads((Path(__file__).with_name("knowledge_graph.json")).read_text(encoding="utf-8"))
        self.assertNotIn("Constraint", data["scope"])

    def test_no_team_or_person_labels_in_the_files(self):
        import re
        folder = Path(__file__).parent
        for f in folder.iterdir():
            if f.suffix in (".py", ".md", ".json", ".csv") and f.name != Path(__file__).name:
                text = f.read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"\bTeam\b|\bPerson [AB]\b", text), f.name)

    def test_no_localised_datacamp_links(self):
        text = Path(__file__).with_name("resource_links.json").read_text(encoding="utf-8")
        self.assertNotIn("datacamp.com/ro/", text)

    def test_csv_files_are_in_sync_with_json(self):
        for name, (header, rows) in build_tables().items():
            with open(Path(__file__).with_name(name), newline="", encoding="utf-8") as f:
                got = list(csv.reader(f))
            self.assertEqual(got, [header] + rows, f"{name} is out of date: run python export_csv.py")


if __name__ == "__main__":
    unittest.main()
