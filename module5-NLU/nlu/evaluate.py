
"""Evaluate the rule-based intent classifier on labelled JSONL data.

Run from the project root:
    python -m nlu.evaluate
"""

import json
from collections import Counter
from pathlib import Path

from nlu.intent_classifier import classify_intent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_questions.jsonl"

LABELS = [
    "concept_confusion",
    "clarification_request",
    "example_request",
    "misconception_signal",
    "prerequisite_confusion",
    "procedure_question",
    "other",
]


def load_dataset(path: Path) -> list[dict[str, str]]:
    """Load and validate labelled questions from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            required = ("question_id", "question_text", "intent")
            if not all(field in row for field in required):
                raise ValueError(
                    f"Line {line_number} is missing a required field."
                )

            if row["intent"] not in LABELS:
                raise ValueError(
                    f"Unknown intent on line {line_number}: {row['intent']}"
                )

            if not isinstance(row["question_text"], str):
                raise ValueError(
                    f"question_text must be a string on line {line_number}."
                )

            rows.append(row)

    if not rows:
        raise ValueError("The evaluation dataset is empty.")

    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs found.")

    return rows


def calculate_metrics(
    expected: list[str],
    predicted: list[str],
    labels: list[str],
) -> dict:
    """Calculate accuracy, per-class metrics, and a confusion matrix."""
    total = len(expected)
    correct = sum(
        actual == prediction
        for actual, prediction in zip(expected, predicted)
    )
    accuracy = correct / total if total else 0.0

    matrix = {
        actual: {prediction: 0 for prediction in labels}
        for actual in labels
    }

    for actual, prediction in zip(expected, predicted):
        matrix[actual][prediction] += 1

    per_class = {}

    for label in labels:
        true_positive = matrix[label][label]
        false_positive = sum(
            matrix[actual][label]
            for actual in labels
            if actual != label
        )
        false_negative = sum(
            matrix[label][prediction]
            for prediction in labels
            if prediction != label
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }

    macro_precision = sum(
        per_class[label]["precision"] for label in labels
    ) / len(labels)
    macro_recall = sum(
        per_class[label]["recall"] for label in labels
    ) / len(labels)
    macro_f1 = sum(
        per_class[label]["f1"] for label in labels
    ) / len(labels)

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_class": per_class,
        "matrix": matrix,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def evaluate() -> None:
    """Run evaluation and print the results."""
    rows = load_dataset(DATASET_PATH)

    expected = []
    predicted = []
    errors = []

    for row in rows:
        result = classify_intent(row["question_text"])
        actual = row["intent"]
        guess = result["intent"]

        expected.append(actual)
        predicted.append(guess)

        if actual != guess:
            errors.append(
                {
                    "id": row["question_id"],
                    "question": row["question_text"],
                    "expected": actual,
                    "predicted": guess,
                    "rule": result["matched_rule"],
                }
            )

    metrics = calculate_metrics(expected, predicted, LABELS)

    print("=" * 72)
    print("RULE-BASED INTENT CLASSIFIER EVALUATION")
    print("=" * 72)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Total questions: {metrics['total']}")
    print(f"Correct predictions: {metrics['correct']}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")

    print("\nPER-INTENT METRICS")
    print(
        f"{'Intent':<28} {'Precision':>10} {'Recall':>10} "
        f"{'F1-score':>10} {'Support':>9}"
    )
    print("-" * 72)

    for label in LABELS:
        item = metrics["per_class"][label]
        print(
            f"{label:<28} "
            f"{item['precision']:>9.2%} "
            f"{item['recall']:>9.2%} "
            f"{item['f1']:>9.2%} "
            f"{item['support']:>9}"
        )

    print("\nMACRO AVERAGES")
    print(f"Macro precision: {metrics['macro_precision']:.2%}")
    print(f"Macro recall:    {metrics['macro_recall']:.2%}")
    print(f"Macro F1-score:  {metrics['macro_f1']:.2%}")

    print("\nCONFUSION MATRIX")
    print("Rows = expected; columns = predicted")
    print(f"{'Expected / Predicted':<28}", end="")
    for label in LABELS:
        print(f"{label[:8]:>10}", end="")
    print()

    for actual in LABELS:
        print(f"{actual:<28}", end="")
        for prediction in LABELS:
            print(f"{metrics['matrix'][actual][prediction]:>10}", end="")
        print()

    print("\nMISCLASSIFIED QUESTIONS")
    if not errors:
        print("No misclassifications.")
    else:
        for error in errors:
            print(f"\n{error['id']}: {error['question']}")
            print(f"  Expected:  {error['expected']}")
            print(f"  Predicted: {error['predicted']}")
            print(f"  Rule:      {error['rule']}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    evaluate()
