"""Documents for a label run, cut into segments the student sees whole.

Documents come from ``ingest/`` (files, directories, URLs) or JSONL rows. A segment is
capped by **student tokens**, not characters: its state ``{input_field: text}`` must
fit the smallest state room the questions leave in laya's layout, so the student is
never trained on a truncated state. :func:`segment_text` packs paragraphs, falls back
to sentences, then words, then characters, and drops nothing.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aiagent.config import Settings
from aiagent.distill.splits import document_id
from aiagent.exceptions import DataLoadError, DistillError, SourceError
from aiagent.ingest.sources import SourceDoc, fetch_source, read_file
from aiagent.system1.sequence import InternalQuestion, SequenceTokenizer

logger = logging.getLogger(__name__)

DOC_SUFFIXES: Final = frozenset({".txt", ".md", ".html", ".htm", ".xhtml", ".pdf"})

_PARAGRAPH: Final = re.compile(r"\n\s*\n")
_SENTENCE: Final = re.compile(r"(?<=[.!?])\s+")  # as in core/segment.py
_JOIN: Final = "\n\n"


@dataclass(frozen=True)
class Sources:
    """Where a label run reads documents from."""

    files: tuple[Path, ...] = ()
    dirs: tuple[Path, ...] = ()
    urls: tuple[str, ...] = ()
    jsonl: tuple[Path, ...] = ()


def _dir_documents(directory: Path) -> list[SourceDoc]:
    if not directory.is_dir():
        raise SourceError(f"directory not found: {directory}")
    docs = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in DOC_SUFFIXES:
            continue
        try:
            docs.append(read_file(path))
        except SourceError as exc:
            logger.warning("skipping %s: %s", path, exc)
    return docs


def _jsonl_documents(path: Path, input_field: str) -> list[SourceDoc]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"{path}: cannot read: {exc}") from exc
    docs = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataLoadError(f"{where}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise DataLoadError(f"{where}: not a JSON object")
        text = row.get(input_field)
        if not isinstance(text, str) or not text.strip():
            raise DataLoadError(f"{where}: needs a non-empty str {input_field!r} field")
        docs.append(SourceDoc(origin=where, text=text.strip()))
    return docs


def collect_documents(
    sources: Sources, *, input_field: str, settings: Settings
) -> list[SourceDoc]:
    """Read every source into SourceDocs, deduplicated by document_id, in order."""
    docs = [read_file(path) for path in sources.files]
    for directory in sources.dirs:
        docs.extend(_dir_documents(directory))
    docs.extend(fetch_source(url, settings=settings) for url in sources.urls)
    for path in sources.jsonl:
        docs.extend(_jsonl_documents(path, input_field))
    unique: dict[str, SourceDoc] = {}
    for doc in docs:  # the first occurrence of each text, in source order
        unique.setdefault(document_id(doc.text), doc)
    if not unique:
        raise DistillError("no documents with text in the given sources")
    return list(unique.values())


def question_room(
    tokenizer: SequenceTokenizer,
    questions: Mapping[str, InternalQuestion],
    *,
    max_len: int,
    head_max_len: int,
) -> int:
    """The smallest state room over all questions."""
    return min(
        tokenizer.room(q, max_len=max_len, head_max_len=head_max_len)
        for q in questions.values()
    )


@dataclass(frozen=True)
class _StateFits:
    tokenizer: SequenceTokenizer
    input_field: str
    room: int

    def __call__(self, text: str) -> bool:
        state = {self.input_field: text}
        return len(self.tokenizer.state_ids(state)) <= self.room


def state_fits(
    tokenizer: SequenceTokenizer, *, input_field: str, room: int
) -> Callable[[str], bool]:
    """text -> len(tokenizer.state_ids({input_field: text})) <= room."""
    return _StateFits(tokenizer, input_field, room)


def _cut(word: str, fits: Callable[[str], bool]) -> list[str]:
    """The longest character prefixes of `word` that fit (binary search)."""
    pieces = []
    rest = word
    while rest:
        if not fits(rest[:1]):
            room = getattr(fits, "room", None)
            where = "" if room is None else f" (room={room})"
            raise DistillError(f"questions leave no room for the state{where}")
        lo, hi = 1, len(rest)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(rest[:mid]):
                lo = mid
            else:
                hi = mid - 1
        pieces.append(rest[:lo])
        rest = rest[lo:]
    return pieces


def _words(sentence: str, fits: Callable[[str], bool]) -> list[str]:
    """Pack whitespace-separated words greedily into units that fit."""
    units: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}" if current else word
        if fits(candidate):
            current = candidate
            continue
        if current:
            units.append(current)
        pieces = [word] if fits(word) else _cut(word, fits)
        units.extend(pieces[:-1])
        current = pieces[-1]
    return [*units, current]


def _units(paragraph: str, fits: Callable[[str], bool]) -> list[str]:
    if fits(paragraph):
        return [paragraph]
    units: list[str] = []
    for sentence in _SENTENCE.split(paragraph):  # stripped: the split eats the spaces
        units.extend([sentence] if fits(sentence) else _words(sentence, fits))
    return units


def segment_text(text: str, *, fits: Callable[[str], bool]) -> list[str]:
    """Split text into segments that each fit, in order, dropping nothing."""
    segments: list[str] = []
    for paragraph in (p.strip() for p in _PARAGRAPH.split(text)):
        if not paragraph:
            continue
        for unit in _units(paragraph, fits):
            joined = f"{segments[-1]}{_JOIN}{unit}" if segments else ""
            if joined and fits(joined):
                segments[-1] = joined
            else:
                segments.append(unit)
    return segments
