"""Input parser for plain text and OCR-simulated input."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedInput:
    """Structured representation of incoming query text."""

    raw_query: str
    normalized_query: str
    dish_name: str
    input_type: str


def parse_input(query: str) -> ParsedInput:
    """Parse user input and extract probable dish name.

    OCR simulation convention:
    - If query starts with "ocr:", treat input type as OCR text.
    """

    q = (query or "").strip()
    input_type = "text"

    if q.lower().startswith("ocr:"):
        input_type = "ocr"
        q = q[4:].strip()

    dish = _extract_dish_name(q)

    return ParsedInput(
        raw_query=query,
        normalized_query=q,
        dish_name=dish,
        input_type=input_type,
    )


def _extract_dish_name(query: str) -> str:
    """Best-effort dish extraction from freeform query."""

    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", query).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    stop_words = {
        "spicy",
        "level",
        "macro",
        "macros",
        "nutrition",
        "summary",
        "about",
        "tell",
        "me",
        "what",
        "is",
        "the",
        "for",
        "of",
        "dish",
        "menu",
        "translate",
    }

    tokens = [t for t in cleaned.split(" ") if t.lower() not in stop_words]
    if not tokens:
        return cleaned.title() if cleaned else "Unknown Dish"

    return " ".join(tokens).title()
