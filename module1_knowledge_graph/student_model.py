"""
Student-knowledge graph for the AI Search unit.

Each student gets a Bayesian Knowledge Tracing (BKT) estimate of mastery for
every concept in the graph, plus:
  - misconception evidence (a student's own words matched to a stored misconception),
    kept separate from "not yet learned"
  - an uncertainty value that falls as evidence accumulates

Evidence kinds for observe():
  "answer" / "exercise"  correct or incorrect; full BKT update
  "hint"                 the student received a hint; learning transition only
  "ambiguous"            incomplete or unclear response; counted, no mastery update

Standard library only; uses the graph and matcher from knowledge_graph.py.
"""
from knowledge_graph import nodes, prereqs, best_misconception, tokens, MATCH_THRESHOLD, diagnose
from search import missing_prerequisites

DEFAULTS = {"p_init": 0.2, "p_transit": 0.15, "p_slip": 0.1, "p_guess": 0.25}
MASTERY_THRESHOLD = 0.9        # mastery at or above this counts as mastered
FULL_CONFIDENCE_OBS = 5        # observations needed before uncertainty reaches 0


class StudentModel:
    def __init__(self, student_id, params=None):
        self.student_id = student_id
        self.params = {**DEFAULTS, **(params or {})}
        p0 = self.params["p_init"]
        self.mastery = {c: p0 for c in nodes}
        self.observations = {c: 0 for c in nodes}
        self.ambiguous = {c: 0 for c in nodes}
        self.touched = set()
        self.misconceptions = {c: {} for c in nodes}   # concept -> {misconception text: count}, still active
        self.resolved = {c: {} for c in nodes}         # misconceptions cleared after the student mastered the concept
        self.log = []

    def _transit(self, m):
        return m + (1 - m) * self.params["p_transit"]

    def observe(self, concept_id, kind="answer", correct=None, text=None):
        if concept_id not in nodes:
            raise KeyError(f"Unknown concept: {concept_id}")
        p = self.params
        m = self.mastery[concept_id]

        if kind == "ambiguous" or (kind in ("answer", "exercise") and correct is None):
            self.ambiguous[concept_id] += 1
        elif kind == "hint":
            self.mastery[concept_id] = self._transit(m)
        elif kind in ("answer", "exercise"):
            self.observations[concept_id] += 1
            if correct:
                num = m * (1 - p["p_slip"])
                den = num + (1 - m) * p["p_guess"]
            else:
                num = m * p["p_slip"]
                den = num + (1 - m) * (1 - p["p_guess"])
            self.mastery[concept_id] = self._transit(num / den)
        else:
            raise ValueError(f"Unknown evidence kind: {kind}")

        matched_now = False
        if text:
            best, best_mis = best_misconception(tokens(text), nodes[concept_id])
            if best >= MATCH_THRESHOLD:
                counts = self.misconceptions[concept_id]
                counts[best_mis] = counts.get(best_mis, 0) + 1
                matched_now = True

        # A misconception is not permanent: once the student answers correctly and has reached
        # mastery without showing it again, move it to `resolved` so the concept can read as mastered.
        if (not matched_now and kind in ("answer", "exercise") and correct
                and self.mastery[concept_id] >= MASTERY_THRESHOLD and self.misconceptions[concept_id]):
            for mis, n in self.misconceptions[concept_id].items():
                self.resolved[concept_id][mis] = self.resolved[concept_id].get(mis, 0) + n
            self.misconceptions[concept_id] = {}

        self.touched.add(concept_id)
        self.log.append((concept_id, kind, correct, round(self.mastery[concept_id], 3)))
        return self.mastery[concept_id]

    def status(self, concept_id):
        if self.misconceptions[concept_id]:
            return "misconception"
        if self.mastery[concept_id] >= MASTERY_THRESHOLD:
            return "mastered"
        if concept_id not in self.touched:
            return "unknown"
        return "not_learned"

    def uncertainty(self, concept_id):
        """0 = well-evidenced, 1 = no evidence yet."""
        return 1 - min(1.0, self.observations[concept_id] / FULL_CONFIDENCE_OBS)

    def mastered(self):
        return {c for c in nodes if self.status(c) == "mastered"}

    def to_mastery(self):
        """Mastery dict for diagnose(): only concepts with some evidence."""
        return {c: self.mastery[c] for c in self.touched}

    def study_plan(self, target):
        """Unmastered prerequisites of target, foundations first, then target if still needed."""
        plan = [c for c in missing_prerequisites(prereqs, target, self.mastered())]
        if target not in self.mastered():
            plan.append(target)
        return [(c, nodes[c]["name"], self.status(c)) for c in plan]

    def diagnose_for_student(self, target, text=""):
        return diagnose(target, text, mastery=self.to_mastery())

    def table(self):
        rows = []
        for c in nodes:
            rows.append({
                "id": c,
                "name": nodes[c]["name"],
                "status": self.status(c),
                "mastery": round(self.mastery[c], 3),
                "uncertainty": round(self.uncertainty(c), 2),
                "misconceptions": list(self.misconceptions[c]),
                "resolved_misconceptions": list(self.resolved[c]),
            })
        return rows
