
import json
from collections import Counter

from nlu.intent_classifier import classify_intent


# DATASET_PATH = "data/heldout_questions.jsonl"
DATASET_PATH = "data/final_evaluation_questions.jsonl"
VALID_INTENTS = {
    "concept_confusion",
    "clarification_request",
    "example_request",
    "misconception_signal",
    "prerequisite_confusion",
    "procedure_question",
    "other",
}


def load_dataset(path):
    questions = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            required_fields = {
                "question_id",
                "question_text",
                "intent",
            }

            missing = required_fields - record.keys()
            if missing:
                raise ValueError(
                    f"Line {line_number}: missing fields {missing}"
                )

            if record["intent"] not in VALID_INTENTS:
                raise ValueError(
                    f"Line {line_number}: invalid intent "
                    f"{record['intent']}"
                )

            questions.append(record)

    ids = [q["question_id"] for q in questions]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs found.")

    return questions


def main():
    questions = load_dataset(DATASET_PATH)

    confusion = {
        actual: Counter()
        for actual in sorted(VALID_INTENTS)
    }

    correct = 0
    errors = []

    for record in questions:
        actual = record["intent"]
        predicted = classify_intent(
            record["question_text"]
        )["intent"]

        confusion[actual][predicted] += 1

        if actual == predicted:
            correct += 1
        else:
            errors.append(
                (
                    record["question_id"],
                    record["question_text"],
                    actual,
                    predicted,
                )
            )

    total = len(questions)
    accuracy = correct / total if total else 0.0

    print("=" * 72)
    print("FINAL INTENT CLASSIFIER EVALUATION")
    print("=" * 72)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Total questions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nPER-INTENT METRICS")
    print(
        f"{'Intent':<28}"
        f"{'Precision':>11}"
        f"{'Recall':>10}"
        f"{'F1-score':>11}"
        f"{'Support':>10}"
    )
    print("-" * 70)

    precision_values = []
    recall_values = []
    f1_values = []

    for intent in sorted(VALID_INTENTS):
        tp = confusion[intent][intent]

        predicted_count = sum(
            confusion[actual][intent]
            for actual in VALID_INTENTS
        )
        actual_count = sum(confusion[intent].values())

        precision = (
            tp / predicted_count
            if predicted_count else 0.0
        )
        recall = (
            tp / actual_count
            if actual_count else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        print(
            f"{intent:<28}"
            f"{precision:>10.2%}"
            f"{recall:>10.2%}"
            f"{f1:>10.2%}"
            f"{actual_count:>10}"
        )

    print("\nMACRO AVERAGES")
    print(
        f"Macro precision: "
        f"{sum(precision_values) / len(precision_values):.2%}"
    )
    print(
        f"Macro recall: "
        f"{sum(recall_values) / len(recall_values):.2%}"
    )
    print(
        f"Macro F1-score: "
        f"{sum(f1_values) / len(f1_values):.2%}"
    )

    intents = sorted(VALID_INTENTS)

    print("\nCONFUSION MATRIX")
    print("Rows = expected; columns = predicted")
    print(f"{'Expected / Predicted':<28}", end="")

    for intent in intents:
        print(f"{intent[:8]:>10}", end="")
    print()

    for actual in intents:
        print(f"{actual:<28}", end="")
        for predicted in intents:
            print(
                f"{confusion[actual][predicted]:>10}",
                end="",
            )
        print()

    print("\nMISCLASSIFIED QUESTIONS")

    if not errors:
        print("No misclassifications.")
    else:
        for qid, question, actual, predicted in errors:
            print(f"\n{qid}: {question}")
            print(f"  Expected:  {actual}")
            print(f"  Predicted: {predicted}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
