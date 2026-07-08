"""Split a text corpus into analyzable segments (pure; no ``dspy``).

Sentiment statistics need a *distribution* of scores, so the corpus is broken
into units that are scored independently. Prefer paragraph boundaries; fall back
to sentence boundaries when the text is a single block. The unit count is capped
by merging adjacent units (never dropping content) so downstream LLM cost stays
bounded regardless of input size.
"""

from __future__ import annotations

import math
import re

_PARAGRAPH = re.compile(r"\n\s*\n")
# Split after ., !, or ? followed by whitespace — good enough for scoring units.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _sentences(paragraph: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE.split(paragraph) if s.strip()]
    return parts if len(parts) >= 2 else [paragraph.strip()]


def _merge(units: list[str], max_segments: int) -> list[str]:
    """Merge adjacent units into at most ``max_segments`` groups (no content lost)."""
    size = math.ceil(len(units) / max_segments)
    return [" ".join(units[i : i + size]) for i in range(0, len(units), size)]


def split_segments(text: str, *, max_segments: int) -> list[str]:
    """Return 1..``max_segments`` non-empty segments covering all of ``text``."""
    paragraphs = [p.strip() for p in _PARAGRAPH.split(text) if p.strip()]
    if len(paragraphs) >= 2:
        units = paragraphs
    elif paragraphs:
        units = _sentences(paragraphs[0])
    else:
        return []

    units = [u for u in units if u]
    if not units:
        return []
    cap = max(1, max_segments)
    if len(units) <= cap:
        return units
    return _merge(units, cap)
