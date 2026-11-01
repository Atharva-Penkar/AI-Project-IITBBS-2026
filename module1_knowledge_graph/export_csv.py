"""Regenerates the CSV files from knowledge_graph.json (the source of truth).

    python export_csv.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent


def build_tables(data=None):
    data = data or json.loads((HERE / "knowledge_graph.json").read_text(encoding="utf-8"))
    nodes = data["nodes"]
    return {
        "concepts.csv": (["id", "name", "type", "description"],
                         [[n["id"], n["name"], n["type"], n["description"]] for n in nodes]),
        "prerequisite_edges.csv": (["source", "target", "relation"],
                                   [[e["source"], e["target"], e["relation"]] for e in data["edges"]]),
        "misconceptions.csv": (["concept_id", "concept", "misconception"],
                               [[n["id"], n["name"], m] for n in nodes for m in n["misconceptions"]]),
        "misconception_paraphrases.csv": (["concept_id", "misconception", "paraphrase"],
                                          [[n["id"], m, p] for n in nodes
                                           for m, ps in n.get("misconception_paraphrases", {}).items() for p in ps]),
        "learning_resources.csv": (["concept_id", "concept", "resource"],
                                   [[n["id"], n["name"], r] for n in nodes for r in n["resources"]]),
    }


def write_all():
    for name, (header, rows) in build_tables().items():
        with open(HERE / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


if __name__ == "__main__":
    write_all()
    print("CSV files regenerated from knowledge_graph.json")
