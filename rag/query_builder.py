from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Scene context heuristics
# These map groups of co-occurring labels to a descriptive scene phrase,
# which enriches the query and helps the retriever surface more relevant chunks.
# Extend this dict freely when adding new classes or scene types.
# ---------------------------------------------------------------------------

_SCENE_HINTS: list[tuple[frozenset, str]] = [
    # Electronics / computing
    (frozenset({"laptop", "keyboard", "mouse", "monitor"}), "full desktop workstation setup"),
    (frozenset({"laptop", "monitor"}),                      "laptop docked to external display"),
    (frozenset({"keyboard", "mouse", "monitor"}),           "desktop computer workstation"),
    (frozenset({"controller", "monitor", "headphones"}),    "gaming setup"),
    (frozenset({"controller", "keyboard", "monitor"}),      "gaming and computing setup"),
    (frozenset({"camera", "laptop", "monitor"}),            "content creation or photography workspace"),

    # Study and academic
    (frozenset({"notebook", "pen", "calculator"}),          "mathematics or science study session"),
    (frozenset({"notebook", "pen", "ruler", "eraser"}),     "structured academic study session"),
    (frozenset({"notebook", "laptop", "pen"}),              "hybrid digital and handwritten study session"),
    (frozenset({"notebook", "calculator", "ruler"}),        "STEM coursework study session"),
    (frozenset({"paper", "pen", "ruler"}),                  "writing or technical drawing session"),
    (frozenset({"backpack", "notebook", "laptop"}),         "student at a study location"),

    # Personal / everyday carry
    (frozenset({"wallet", "keys", "phone"}),                "personal items of someone settled into their space"),
    (frozenset({"wallet", "watch", "phone"}),               "personal everyday carry items on a desk"),
    (frozenset({"keys", "backpack"}),                       "person arriving or preparing to leave"),

    # Food and drink at desk
    (frozenset({"cup", "laptop"}),                          "working or studying with a hot beverage"),
    (frozenset({"can", "controller"}),                      "gaming session with an energy or soft drink"),
    (frozenset({"chopsticks", "laptop"}),                   "eating a meal while working at a desk"),
    (frozenset({"cup", "notebook", "pen"}),                 "studying with a beverage"),
    (frozenset({"bottle", "backpack"}),                     "student or commuter with hydration"),

    # Audio
    (frozenset({"headphones", "laptop"}),                   "focused work or study with audio isolation"),
    (frozenset({"earbuds", "phone"}),                       "mobile audio listening"),
    (frozenset({"speaker", "monitor"}),                     "desktop audio listening setup"),
]


@dataclass
class QueryContext:
    query: str
    labels: list[str]
    scene_hint: Optional[str]
    top_labels: list[str]


def build_query(
    labels: list[str],
    scores: Optional[list[float]] = None,
    max_labels: int = 8,
) -> QueryContext:
    if not labels:
        return QueryContext(
            query="desk workspace scene with personal items",
            labels=[],
            scene_hint=None,
            top_labels=[],
        )

    # Sort by score if provided, otherwise keep original order
    if scores and len(scores) == len(labels):
        paired = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
        sorted_labels = [l for l, _ in paired]
    else:
        sorted_labels = list(labels)

    # Deduplicate while preserving order
    seen = set()
    unique_labels = []
    for label in sorted_labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)

    # Cap at max_labels
    top_labels = unique_labels[:max_labels]
    label_set = frozenset(top_labels)

    # Find the best matching scene hint
    best_hint = None
    best_overlap = 0
    for hint_set, hint_text in _SCENE_HINTS:
        overlap = len(hint_set & label_set)
        coverage = overlap / len(hint_set)
        # Require at least 2 matching labels and >50% coverage of hint set
        if overlap >= 2 and coverage > 0.5 and overlap > best_overlap:
            best_hint = hint_text
            best_overlap = overlap

    # Compose query
    label_phrase = ", ".join(top_labels)

    if best_hint:
        query = f"{label_phrase} — {best_hint}"
    else:
        # Fallback: generic desk/workspace framing
        query = f"desk or personal workspace scene containing {label_phrase}"

    return QueryContext(
        query=query,
        labels=top_labels,
        scene_hint=best_hint,
        top_labels=top_labels,
    )


def build_query_from_detections(detections: list[dict]) -> QueryContext:
    labels = [d["label"] for d in detections]
    scores = [d.get("score", 0.0) for d in detections]
    return build_query(labels, scores)