"""Adapter from Team 5's rule-based NLU output to the shared Event schema.

The Team 5 taxonomy and Team 3 shared schema intentionally use different
labels. This module is the single explicit boundary between them. Unknown NLU
labels are not passed through as if they were canonical IDs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.schema import CourseCatalog, Event, EventType, Intent

# Team 5 intent labels -> the compatible shared-schema intent labels.
INTENT_MAP: dict[str, Intent] = {
    "concept_confusion": Intent.CONFUSION,
    "prerequisite_confusion": Intent.CONFUSION,
    "clarification_request": Intent.CLARIFICATION,
    "example_request": Intent.EXAMPLE_REQUEST,
    "misconception_signal": Intent.MISCONCEPTION_SIGNAL,
    "procedure_question": Intent.CLARIFICATION,
    "other": Intent.UNCLEAR,
}

# Cross-module concept aliases. A concept is only emitted if it exists in the
# actual catalog passed by the caller, so this adapter also works on smaller
# course catalogs used by tests or later project modules.
CONCEPT_MAP: dict[str, str] = {
    "astar_search": "astar",
    "a_star_search": "astar",
    "a_star": "astar",
    "bfs": "breadth_first_search",
    "dfs": "depth_first_search",
}

# NLU taxonomy IDs -> nearest compatible Team 3 misconception IDs. The ASTAR
# label is not the same identifier as Team 3's `confuses_g_and_h`, so explicit
# mapping here avoids silently losing the signal.
MISCONCEPTION_MAP: dict[str, str] = {
    "ASTAR_G_VS_H": "confuses_g_and_h",
    "ASTAR_HEURISTIC_VS_PATH_COST": "confuses_g_and_h",
}


def nlu_result_to_event(
    student_id: str,
    result: Mapping[str, Any],
    catalog: CourseCatalog,
    *,
    step: int = 0,
    fallback_concept_id: str | None = None,
    confidence_if_missing: float = 0.5,
) -> Event | None:
    """Convert one Team 5 ``analyse_question`` result into a QUESTION event.

    Return ``None`` when the input is invalid or no supported concept can be
    mapped. A fallback concept is allowed when the caller has independent
    context (for example, the concept currently being taught). Missing NLU
    confidence defaults to 0.5, the neutral confidence value used by the shared
    schema, rather than pretending the classifier produced a score.
    """
    if not result.get("input_valid", True):
        return None

    available_concepts = set(catalog.concept_ids())
    raw_intent = str(result.get("intent", "other"))
    intent = INTENT_MAP.get(raw_intent, Intent.UNCLEAR)

    raw_primary = result.get("primary_concept")
    concept_id = _supported_concept(str(raw_primary) if raw_primary else None, available_concepts)

    # A detected misconception is more specific than a broad concept guess.
    # Use its canonical owner when that owner exists in the current catalog.
    raw_detections = result.get("misconceptions", [])
    if isinstance(raw_detections, list):
        for detection in raw_detections:
            if not isinstance(detection, Mapping):
                continue
            raw_mid = detection.get("misconception_id", detection.get("id"))
            if not raw_mid:
                continue
            mapped_mid = MISCONCEPTION_MAP.get(str(raw_mid), str(raw_mid))
            try:
                owner = catalog.misconception(mapped_mid).concept_id
            except KeyError:
                continue
            # This sets a concept that actually contains the signal. The model
            # otherwise ignores misconception IDs not attached to the event's
            # concept, by design.
            concept_id = owner
            break

    if concept_id is None and fallback_concept_id in available_concepts:
        concept_id = fallback_concept_id
    if concept_id is None:
        return None

    valid_misconceptions = {m.id for m in catalog.misconceptions_for(concept_id)}
    signals: list[str] = []
    if isinstance(raw_detections, list):
        for detection in raw_detections:
            if not isinstance(detection, Mapping):
                continue
            raw_mid = detection.get("misconception_id", detection.get("id"))
            if not raw_mid:
                continue
            mapped_mid = MISCONCEPTION_MAP.get(str(raw_mid), str(raw_mid))
            if mapped_mid in valid_misconceptions and mapped_mid not in signals:
                signals.append(mapped_mid)

    confidence_value = result.get("nlu_confidence", confidence_if_missing)
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = confidence_if_missing
    # Keep invalid values out of Pydantic validation. A malformed classifier
    # response is treated neutrally, not allowed to break the class session.
    if not 0.0 <= confidence <= 1.0:
        confidence = confidence_if_missing

    question = result.get("question")
    return Event(
        student_id=student_id,
        concept_id=concept_id,
        type=EventType.QUESTION,
        step=step,
        intent=intent,
        nlu_confidence=confidence,
        misconception_signals=signals,
        text=question if isinstance(question, str) else None,
    )


def _supported_concept(raw_concept: str | None, available: set[str]) -> str | None:
    """Resolve a raw NLU concept only if the shared catalog supports it."""
    if not raw_concept:
        return None
    mapped = CONCEPT_MAP.get(raw_concept, raw_concept)
    if mapped in available:
        return mapped
    # Accept a case-insensitive canonical ID without inventing a synonym.
    lower_lookup = {candidate.lower(): candidate for candidate in available}
    return lower_lookup.get(mapped.lower())
