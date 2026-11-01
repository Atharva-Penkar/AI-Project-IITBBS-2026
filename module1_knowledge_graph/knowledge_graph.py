"""
AI Search knowledge graph and diagnostic engine.

  prerequisite_chain(c)                -> all prerequisites of c, in study order (prerequisites first)
  diagnose(c, text, mastery)           -> ranked possible causes with examples and exercises
  explain(result)                      -> readable text for a diagnosis

Standard library only. Run demo.py to see it in action.
"""
import json
import re
from pathlib import Path

from module1_knowledge_graph.search import bfs_ancestors

DATA_PATH = Path(__file__).with_name("knowledge_graph.json")
LINKS_PATH = Path(__file__).with_name("resource_links.json")

W_MASTERY = 4.0         # weight applied to (1 - mastery) for a concept
W_MISCONCEPTION = 5.0   # weight applied to a misconception match
MATCH_THRESHOLD = 0.5   # fraction of a misconception's keywords the student text must contain
MIN_SHARED_TOKENS = 2   # ...and at least this many words must be shared (blocks one-word false matches)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "is", "are", "was", "in", "it", "its",
    "that", "this", "for", "on", "with", "as", "be", "by", "from", "at", "do", "does",
    "can", "so", "but", "i", "my", "think", "thought", "just", "how", "what", "which",
    "when", "if", "then", "there", "their", "they",
}
# Notation is mapped to words so "h(n)" and "heuristic" match each other.
_NOTATION = [("path cost", " pathcost "), ("g(n)", " pathcost "),
             ("h(n)", " heuristic "), ("f(n)", " evaluation "), ("a*", " astar ")]


def load_graph(path=DATA_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    prereqs = {k: [] for k in nodes}
    dependents = {k: [] for k in nodes}
    for edge in data["edges"]:
        if edge["source"] not in nodes or edge["target"] not in nodes:
            raise ValueError(f"Edge refers to an unknown concept: {edge}")
        prereqs[edge["target"]].append(edge["source"])
        dependents[edge["source"]].append(edge["target"])
    seen_ids = set()
    for n in nodes.values():
        for stated in n.get("misconception_paraphrases", {}):
            if stated not in n.get("misconceptions", []):
                raise ValueError(f"{n['id']}: paraphrases given for unknown misconception: {stated!r}")
        ids = n.get("misconception_ids", {})
        if set(ids) != set(n.get("misconceptions", [])):
            raise ValueError(f"{n['id']}: every misconception needs exactly one entry in misconception_ids")
        for mid in ids.values():
            if mid in seen_ids:
                raise ValueError(f"Duplicate misconception id: {mid}")
            seen_ids.add(mid)
    _assert_acyclic(prereqs)
    return nodes, prereqs, dependents


def _assert_acyclic(prereqs):
    done, active = set(), set()

    def visit(n):
        if n in done:
            return
        if n in active:
            raise ValueError(f"Prerequisite cycle through {n}")
        active.add(n)
        for p in prereqs[n]:
            visit(p)
        active.remove(n)
        done.add(n)

    for n in prereqs:
        visit(n)


nodes, prereqs, dependents = load_graph()


def _load_links(path=LINKS_PATH):
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8")).get("links", {})


RESOURCE_LINKS = _load_links()


def _check(concept_id):
    if concept_id not in nodes:
        raise KeyError(f"Unknown concept: {concept_id}")


def prerequisite_chain(concept_id):
    """All prerequisites of concept_id, ordered so every concept comes after its own prerequisites."""
    _check(concept_id)
    seen, order = set(), []

    def visit(cur):
        for p in prereqs[cur]:
            if p not in seen:
                seen.add(p)
                visit(p)
                order.append(p)

    visit(concept_id)
    return order


def classify_resource(text):
    t = text.lower()
    if t.startswith(("exercise", "practice")):
        return "exercises"
    if any(w in t for w in ("worked", "trace", "example", "simulation", "calculat")):
        return "examples"
    return "notes"


def tokens(text):
    """Lowercase content words of a text, with notation mapped and simple plurals removed."""
    t = text.lower()
    for raw, repl in _NOTATION:
        t = t.replace(raw, repl)
    out = set()
    for w in re.findall(r"[a-z]+", t):
        if w in STOPWORDS or len(w) < 2:
            continue
        if w.endswith("s") and len(w) > 3:
            w = w[:-1]
        out.add(w)
    return out


def misconception_match(text_tokens, misconception):
    """Fraction of the misconception's keywords that appear in the student's text.
    Returns 0 unless at least MIN_SHARED_TOKENS words are shared."""
    m = tokens(misconception)
    if not m:
        return 0.0
    shared = len(text_tokens & m)
    if shared < MIN_SHARED_TOKENS:
        return 0.0
    return shared / len(m)


def best_misconception_detail(text_tokens, node):
    """Best (score, misconception, matched_words) for a concept node.

    Each stored misconception is compared with its own wording and with the
    paraphrases listed under "misconception_paraphrases", so a student does not
    have to repeat the stored sentence word for word. The returned misconception
    is always the canonical stored statement; matched_words are the words the
    student's text shares with the wording that matched best."""
    paraphrases = node.get("misconception_paraphrases", {})
    best, best_mis, best_words = 0.0, None, frozenset()
    for mis in node.get("misconceptions", []):
        for wording in [mis] + paraphrases.get(mis, []):
            score = misconception_match(text_tokens, wording)
            if score > best:
                best, best_mis, best_words = score, mis, frozenset(text_tokens & tokens(wording))
    return best, best_mis, best_words


def best_misconception(text_tokens, node):
    """Best (score, misconception) for a concept node (see best_misconception_detail)."""
    best, mis, _ = best_misconception_detail(text_tokens, node)
    return best, mis


def explained_away(matches):
    """Concepts whose misconception match is only a side effect of a stronger match elsewhere.

    matches: {concept_id: (score, matched_words)} for matches at or above MATCH_THRESHOLD.
    A weaker match is explained away when every word it matched was also matched by a
    stronger match on another concept (e.g. "an overestimating heuristic still guarantees
    an optimal A* path" matches the admissibility misconception strongly; its words
    "A*", "optimal", "heuristic" also touch a weaker A* misconception, but add no new evidence).
    Returns {explained_concept: concept_that_explains_it}."""
    out = {}
    for c, (score, words) in matches.items():
        for other, (o_score, o_words) in matches.items():
            if other != c and o_score > score and words <= o_words:
                if c not in out or matches[out[c]][0] < o_score:
                    out[c] = other
    return out


def diagnose(concept_id, difficulty_text="", mastery=None, top_k=5):
    """
    Rank the concepts that could be causing a difficulty with concept_id.

    Each prerequisite (and the target itself) scores:
      + W_MASTERY * (1 - mastery)          if the student's mastery estimate is given
      + W_MISCONCEPTION * match            if the student's text matches one of its misconceptions
    Candidates with no evidence are still listed (marked 'unverified') so the
    teacher can check them with a diagnostic question.

    mastery: {concept_id: value in [0, 1]}
    """
    _check(concept_id)
    mastery = dict(mastery or {})
    for c, m in mastery.items():
        if c not in nodes:
            raise KeyError(f"Unknown concept in mastery: {c}")
        if not 0.0 <= m <= 1.0:
            raise ValueError(f"Mastery for {c} must be between 0 and 1")

    depths = bfs_ancestors(prereqs, concept_id)
    text_tokens = tokens(difficulty_text)
    found = {}
    for concept in depths:
        b, m, words = best_misconception_detail(text_tokens, nodes[concept])
        if b >= MATCH_THRESHOLD:
            found[concept] = (b, m, words)
    explained = explained_away({c: (b, words) for c, (b, m, words) in found.items()})
    causes = []
    for concept, depth in depths.items():
        node = nodes[concept]
        score, reasons = 0.0, []
        matched_misconception = None

        if concept in mastery:
            m = mastery[concept]
            score += W_MASTERY * (1.0 - m)
            reasons.append(f"mastery estimate is {m:.0%}")

        if concept in found and concept in explained:
            by = nodes[explained[concept]]["name"]
            reasons.append(f"its misconception match is already explained by a stronger match on {by}")
        elif concept in found:
            best, best_mis, _ = found[concept]
            score += W_MISCONCEPTION * best
            matched_misconception = node.get("misconception_ids", {}).get(best_mis)
            reasons.append(f"difficulty text matches misconception ({best:.0%}): \"{best_mis}\"")

        if not reasons:
            reasons.append("prerequisite of the target; no evidence yet")

        by_kind = {"examples": [], "exercises": [], "notes": []}
        for r in node.get("resources", []):
            by_kind[classify_resource(r)].append(r)

        causes.append({
            "concept_id": concept,
            "name": node["name"],
            "depth": depth,
            "score": round(score, 3),
            "misconception_id": matched_misconception,
            "confidence": "evidence" if score > 0 else "unverified",
            "reasons": reasons,
            "examples": by_kind["examples"],
            "exercises": by_kind["exercises"],
            "notes": by_kind["notes"],
            "links": RESOURCE_LINKS.get(concept, []),
        })

    causes.sort(key=lambda c: (-c["score"], c["depth"], c["name"]))
    top_causes = causes[:top_k]

    target = nodes[concept_id]
    if not top_causes or top_causes[0]["confidence"] == "unverified":
        next_step = (f"No clear evidence yet. Ask a diagnostic question on "
                     f"{top_causes[0]['name'] if top_causes else target['name']} first.")
    elif top_causes[0]["concept_id"] == concept_id:
        next_step = f"Re-teach {target['name']} with a worked example, then check its misconception."
    else:
        next_step = f"Revise {top_causes[0]['name']} before returning to {target['name']}."

    return {
        "target": {"id": concept_id, "name": target["name"]},
        "prerequisite_chain": [{"id": c, "name": nodes[c]["name"]} for c in prerequisite_chain(concept_id)],
        "ranked_causes": top_causes,
        "next_step": next_step,
        "note": "Rule-based baseline. Validate with more student responses and instructor review. Misconceptions are drafts.",
    }


def explain(result):
    lines = [f"Target: {result['target']['name']}"]
    chain = result["prerequisite_chain"]
    lines.append("Study order (prerequisites first): " + (", then ".join(c["name"] for c in chain) if chain else "(none)"))
    lines.append("Possible causes:")
    for i, cause in enumerate(result["ranked_causes"], 1):
        lines.append(f"  {i}. {cause['name']}  (score {cause['score']:.1f}, {cause['confidence']})")
        for reason in cause["reasons"]:
            lines.append(f"       - {reason}")
        if cause["examples"]:
            lines.append("       examples: " + "; ".join(cause["examples"]))
        if cause["exercises"]:
            lines.append("       exercises: " + "; ".join(cause["exercises"]))
    lines.append("Next step: " + result["next_step"])
    lines.append("Note: " + result["note"])
    return "\n".join(lines)
