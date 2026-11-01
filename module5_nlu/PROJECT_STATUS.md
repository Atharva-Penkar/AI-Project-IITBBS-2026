<!-- 
# Agentic Classroom Digital Twin — Project Status

## Official assignment

Group 8, AI course: Agentic Classroom Digital Twin.

All modules except Module 5 (Generative AI) are in scope for the group.
The official Module 4 brief requires NLU to analyse student questions,
classify intent, and identify related concepts or misconceptions.

## My responsibility

Team 5, Person B: Natural Language Understanding (NLU).

My teammate, Person A, owns reinforcement learning.

My component must classify student-question intent, identify relevant
concepts, detect possible misconceptions, represent uncertainty, and
provide structured outputs that can later integrate with other modules.

## Project constraints

- Start with a rule-based baseline.
- Do not use generative AI or paid external APIs.
- Keep the component independently runnable.
- Use mock data until other teams agree on interfaces.
- Evaluate with labelled questions, tests, and measurable metrics.
- Add lightweight ML only if evaluation shows a worthwhile benefit.
- Do not implement graph search, BKT, planning, or RL inside NLU.

## Agreed initial intent labels

concept_confusion
clarification_request
example_request
misconception_signal
prerequisite_confusion
procedure_question
other

## Initial implementation milestones

1. Define taxonomy and labelled seed examples.
2. Implement intent classification.
3. Implement concept identification.
4. Implement misconception detection.
5. Integrate the pipeline and validate structured output.
6. Expand the labelled dataset and evaluate performance.
7. Compare against optional TF-IDF and Logistic Regression.
8. Integrate using agreed interfaces.

## Current milestone

Step 1: taxonomy and seed dataset.

Planned files:
- nlu/taxonomy.py
- data/seed_questions.jsonl
- tests/test_taxonomy.py
- README.md

Step 1 is complete only after the files are created and the consistency
tests pass. Do not assume that local implementation or tests have been
completed until confirmed.

## Integration boundaries

- Team 1: knowledge graph and diagnostic search.
- Team 2: CSP and classroom scheduling.
- Team 3: probabilistic student modelling.
- Team 4: adaptive teaching planner and execution.
- Team 5 Person A: reinforcement learning.
- Team 5 Person B: NLU.

Canonical IDs and interfaces remain provisional until agreed with
the relevant teams. -->

# Agentic Classroom Digital Twin — Project Status

## My responsibility
Team 5, Person B: Natural Language Understanding (NLU).

## Project constraints
- Rule-based baseline first.
- No generative AI or paid external APIs.
- Independently runnable Python component.
- Use labelled data and tests to measure correctness.
- Do not implement graph search, student mastery/BKT, planning, or RL.

## Completed milestones
- Step 1: Defined canonical intent, concept, and misconception labels.
- Step 1: Created labelled seed questions and taxonomy consistency tests.
- Step 1: Four taxonomy tests passed.
- Step 2: Implemented rule-based intent classification.
- Step 2: Added intent-classifier tests; confirm test results after running.

## Current files
- `nlu/taxonomy.py`
- `nlu/intent_classifier.py`
- `data/seed_questions.jsonl`
- `tests/test_taxonomy.py`
- `tests/test_intent_classifier.py`

## Current milestone
Step 2: Verify that all tests pass and inspect the classifier output.

## Remaining milestones
3. Implement concept identification.
4. Implement misconception detection.
5. Combine the components into a structured NLU pipeline.
6. Expand labelled data and evaluate performance.
7. Compare with optional TF-IDF + Logistic Regression.
8. Integrate using interfaces agreed with the other teams.

## Integration boundaries
- Team 1: Knowledge graph and diagnostic search.
- Team 2: CSP and classroom scheduling.
- Team 3: Probabilistic student modelling.
- Team 4: Adaptive teaching planner and execution.
- Team 5 Person A: Reinforcement learning.
- Team 5 Person B: NLU.

Canonical IDs and interfaces remain provisional until agreed with the relevant teams.

## Completed milestones
- Step 1: Taxonomy and labelled seed examples — 4 tests passed.
- Step 2: Rule-based intent classifier — 14 classifier tests passed.
- Combined test suite: 18 tests passed.
- Demo questions produced the expected intent labels.

## Current milestone
Step 3: Implement concept identification.

- Step 3: Rule-based concept identification implemented.
- Step 3: Concept aliases sourced from the canonical taxonomy.
- Step 3: Match evidence includes alias text and character offsets.
- Step 3: Concept-identifier tests passed (confirm after running).
- Step 3: Rule-based concept identification implemented.
- Step 3: Alias matching, overlapping phrases, duplicate handling,
  and match evidence tested.
- Combined test suite: 31 tests passed.

- Step 4: Rule-based misconception detection implemented.
- Step 4: Six taxonomy-defined misconception patterns supported.
- Step 4: Detection output includes rule IDs and textual evidence.
- Step 4: Negative cases and taxonomy consistency tested.
- Combined test suite: 45 tests passed.

## Current milestone
Step 4 complete. Begin Step 5: integrate the NLU pipeline.