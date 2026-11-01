# Knowledge Graph and Diagnostic Search (AI Search)

Covers Module 1, Part A (the knowledge graph and the diagnostic search), plus the student-knowledge graph.

**Review status:** the misconceptions are drafts and need instructor review before use with students. The resource links come from web search results. Check each one before sharing it with students.

## Files

- **Graph:** `knowledge_graph.json` (source of truth, 11 concepts, 17 prerequisite edges, misconceptions with alternative wordings, resources). The CSVs are generated from it by `export_csv.py`. `resource_links.json` holds real links for each concept.
- **Search and diagnosis:** `search.py` (BFS/DFS, study plans, A*, greedy, uniform-cost), `knowledge_graph.py` (`diagnose`, which ranks likely root causes and attaches links), `astar_examples.py` (toy graph for independent testing).
- **Student-knowledge graph:** `student_model.py` (per-student Bayesian Knowledge Tracing for mastery, misconception flags, uncertainty, and study plans).
- **Other:** `demo.py`, `export_csv.py` (regenerates the CSVs from the JSON), `draw_graph.py` (regenerates the PNG), `test_search_and_diagnosis.py`, `test_student_model.py`.

## Run

Requires Python 3.9+. Standard library only.

    python demo.py
    python -m unittest test_search_and_diagnosis test_student_model -v
    python export_csv.py        # only after editing knowledge_graph.json

Run everything from inside this folder (the modules import each other by plain name).

Regenerate the image (optional, needs matplotlib): `python draw_graph.py`

## Using it

    from knowledge_graph import diagnose, explain
    from student_model import StudentModel

    s = StudentModel("student-1")
    s.observe("Path_Cost", correct=False, text="Path cost is the estimated remaining cost to the goal.")
    s.observe("Heuristic", correct=True)
    print(explain(s.diagnose_for_student("Astar")))
    print(s.study_plan("Astar"))

## Student-knowledge graph

Each student has, for every concept:

- **Mastery** (Bayesian Knowledge Tracing, a standalone model for this folder). Correct answers raise it, wrong answers lower it, and the model accounts for guessing and slips. Hints apply only the learning transition.
- **Status:** `unknown` (no evidence), `not_learned` (evidence of difficulty, no stored misconception matched), `misconception` (the student's own words match a stored misconception), or `mastered` (mastery at or above 0.9). A recorded misconception is moved to `resolved` once the student answers correctly, reaches mastery, and no longer shows it; until then it keeps the status at `misconception`.
- **Uncertainty:** 1.0 with no evidence, falling to 0 after 5 scored answers.
- **Ambiguous responses** are counted but do not change mastery, so an incomplete answer does not look like a mistake.

## Independent testing (A* failures)

`astar_examples.py` reproduces the failures on a 4-node graph (optimal cost 4):

| Student error | Variant | Result | Diagnosed concept |
|---|---|---|---|
| Uses only h(n) | greedy (`key="h"`) | cost 6 | A* Search |
| Priority queue = FIFO | FIFO frontier | cost 6 | Queues and Priority Queues |
| Overestimating heuristic is fine | h(B)=10 | cost 6 | Admissible and Consistent Heuristics |
| Path cost = remaining cost | n/a | n/a | Path Cost g(n) |

## How the diagnosis works

Each prerequisite of the target (and the target itself) gets a score:

- `4 * (1 - mastery)` when a mastery estimate is supplied.
- `5 * match` when the student's text covers at least half of the words of one of the concept's misconceptions (and shares at least 2 words with it). Each misconception is compared with its stored sentence and with its alternative wordings (`misconception_paraphrases` in the JSON), so students do not have to repeat the stored sentence. The reason always shows the stored statement. Notation like `h(n)` and `path cost` is mapped to words first.

**Explaining away:** if every word that matched one concept's misconception was also matched by a stronger match on another concept, the weaker match is not counted as separate evidence. Example: "an overestimating heuristic still guarantees an optimal A* path" matches the admissibility misconception at 100%; its words "A*", "optimal" and "heuristic" also touch the weaker A* misconception, which would otherwise add a false second cause (score 3.0). Two independent misconceptions in one answer are both kept.

Concepts with no evidence are still listed, marked `unverified`, so the teacher can check them with a diagnostic question.

## Changes in this version

- Real learning links for every concept, taken from search results (`resource_links.json`). The DataCamp links now point to the English page (`/tutorial/...`, not the Romanian `/ro/` version).
- Student-knowledge graph (`student_model.py`) with tests.
- Study plans are in prerequisite order. The earlier depth-based order could put a concept before its own prerequisite.
- Graph image redrawn with better ordering. Some edges still pass behind boxes.
- Review status recorded in the data and here.
- **Matching:** 2 alternative wordings per misconception (44 in total, `misconception_paraphrases.csv`), plus a rule that at least 2 words must be shared. Before, a paraphrase such as "BFS always finds the cheapest path in a weighted graph" scored 0 against "BFS is always the lowest-cost algorithm."
- **Display:** "Prerequisite chain" is now "Study order (prerequisites first)". The arrows suggested that each concept depends on the previous one, which is not true for neighbours in the list.
- **Scope:** the JSON no longer claims "Search and Constraint Satisfaction"; this graph covers search only.
- **Misconceptions can be resolved** in the student model (see above) instead of blocking `mastered` forever.
- **Explaining away** (see above) removes the false secondary cause on A* in the overestimating-heuristic example.
- **Labels removed:** the files no longer carry team or person labels, and the main test file is now `test_search_and_diagnosis.py`.
- **`export_csv.py`** regenerates the CSVs from the JSON; a test checks that the CSVs match.

## Limitations

- Misconceptions and their alternative wordings are drafts and have not been reviewed by an instructor. Two wordings per misconception will not cover every phrasing; add more from real student answers.
- The links were found through search. Only the DataCamp page was confirmed to exist; I did not open the others, so check them before sharing.
- Diagnosis is rule-based word matching. "A* uses only g(n)" and "A* uses only h(n)" are still hard to tell apart when the student mentions both.
- The BKT parameters (guess 0.25, slip 0.1, transition 0.15) are standard starting values, not fitted to data.
- This folder is standalone: concept IDs (`Astar`, `Queue_PQ`, ...) and the plain-name imports are its own. The wider project repository has other modules (including its own student model) with different IDs, and nothing here converts between them. Do not use `student_model.py` as a second student model alongside the project's.
