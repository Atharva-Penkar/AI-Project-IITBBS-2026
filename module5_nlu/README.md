
# Agentic Classroom Digital Twin — NLU

## Purpose

This component analyses simulated student questions for Group 8's
Agentic Classroom Digital Twin project.

It will classify question intent, identify relevant AI concepts, and
detect possible misconceptions using a rule-based baseline.

## Project boundaries

The NLU component does not implement:

- Knowledge-graph search
- Bayesian Knowledge Tracing or student mastery estimation
- Teaching planning
- Reinforcement learning
- Generative AI agents

Those responsibilities belong to other project components.

## Current milestone

Step 1 defines the canonical taxonomy and labelled seed examples.

No question-classification algorithm has been implemented yet.

## Taxonomy

- Intents: `nlu/taxonomy.py`
- Concepts: `nlu/taxonomy.py`
- Misconceptions: `nlu/taxonomy.py`
- Seed examples: `data/seed_questions.jsonl`
- Consistency tests: `tests/test_taxonomy.py`

## Requirements

Python 3.10 or newer is recommended.

This milestone uses only the Python standard library.

## Run tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```
