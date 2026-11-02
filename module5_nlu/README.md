# Question analysis (natural language understanding)

Reads a student's question and works out what kind of help they are asking for, which concept it is about, and whether it reveals a known misconception. It is rule-based: keyword and pattern matching, with no machine-learning model and no external services.

## What it returns

```python
from module5_nlu.nlu.pipeline import analyse_question

analyse_question("Isn't it true that DFS always finds the shortest path?")
```

gives a dictionary with the fields below (excerpt):

| Field | Example | Meaning |
|---|---|---|
| `intent` | `misconception_signal` | What the student wants |
| `primary_concept` | `depth_first_search` | The concept the question is mainly about |
| `related_concepts` | `[...]` | Other concepts involved |
| `misconceptions` | `DFS_GUARANTEES_SHORTEST_PATH` | Matched misconceptions, with details of each match |
| `needs_clarification` | `False` | True when the question is too vague to act on |
| `nlu_confidence` | `0.9` | A rule-based confidence in [0, 1] |

Confidence is 0.3 when the intent is `other`, 0.9 when a misconception is matched, 0.5 when the concept is missing or the question needs clarification, and 0.8 otherwise. These values are set by hand, not calibrated.

## Labels

- **Intents (7):** `concept_confusion`, `clarification_request`, `example_request`, `misconception_signal`, `prerequisite_confusion`, `procedure_question`, `other`.
- **Concepts (11):** breadth-first search, depth-first search, A*, heuristic, path cost, priority queue, minimax, alpha-beta pruning, constraint satisfaction, Bayesian reasoning, reinforcement learning.
- **Misconceptions (6):** A* heuristic versus path cost, A* g versus h, BFS using depth priority, DFS guaranteeing the shortest path, alpha-beta changing the minimax result, and RL reward equalling policy.

These are defined in `nlu/taxonomy.py`. The labels are wider than the 11-concept course the rest of the project teaches, and use different names. `module4_planner/nlu_adapter.py` translates them, and `main.py` also runs the course's own misconception matcher on question text, which catches paraphrases this module misses.

## Files

| File | Purpose |
|---|---|
| `nlu/taxonomy.py` | The intent, concept and misconception definitions |
| `nlu/intent_classifier.py` | Rule-based intent classification |
| `nlu/concept_identifier.py` | Finds the concepts a question mentions |
| `nlu/misconception_detector.py` | Pattern rules for the six misconceptions |
| `nlu/pipeline.py` | `analyse_question`, which combines the three |
| `nlu/evaluate.py`, `evaluate_heldout.py`, `evaluate_final.py` | Score the intent classifier on a labelled set |
| `data/` | Labelled questions: 15 seed examples and three evaluation sets |
| `tests/` | Unit tests for each component |

## Running it

From the project folder:

```bash
python -m pytest module5_nlu
python -m module5_nlu.nlu.evaluate
python -m module5_nlu.nlu.evaluate_heldout
python -m module5_nlu.nlu.evaluate_final
```

The evaluators print per-intent precision, recall and F1, a confusion matrix and any misclassified questions.

## How well it does

Intent accuracy on the three labelled sets:

| Set | Questions | Result |
|---|---|---|
| `evaluation_questions.jsonl` | 70 | all correct |
| `heldout_questions.jsonl` | 35 | all correct |
| `final_evaluation_questions.jsonl` | 70 | 69 correct (macro F1 98.6%); the one miss labels "What is the difference between a laptop and a tablet?" as a clarification request instead of `other` |

## Limitations

- The questions in the evaluation sets are short and fairly regular, and the rules are hand-written, so these scores say little about how it would do on real student text.
- Only intent is scored on labelled data. Concept and misconception detection are covered by unit tests, not by an accuracy measurement.
- Paraphrases are missed. "I thought A* only uses h(n), is that right?" currently comes out as `other` with no misconception, because no rule matches that wording.
- The concept and misconception lists are fixed. Anything outside them is reported as unclear.
