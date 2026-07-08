"""Resolve local files and URLs into analyzable text documents.

Format is chosen by file extension (files) or ``Content-Type`` (URLs): PDF and
HTML are converted to plain text; anything else is decoded as UTF-8. Each source
yields a :class:`SourceDoc` carrying its origin label for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiagent.config import Settings
from aiagent.exceptions import SourceError
from aiagent.ingest.extract_text import html_to_text, pdf_to_text
from aiagent.ingest.fetch import fetch_url

_HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})
_PDF_SUFFIXES = frozenset({".pdf"})


@dataclass(frozen=True)
class SourceDoc:
    """Text extracted from a single source, tagged with where it came from."""

    origin: str
    text: str


def _require_text(text: str, origin: str) -> SourceDoc:
    cleaned = text.strip()
    if not cleaned:
        raise SourceError(f"no extractable text in {origin}")
    return SourceDoc(origin=origin, text=cleaned)


def read_file(path: Path) -> SourceDoc:
    """Read and extract text from a local file (``.pdf``/``.html``/other=text)."""
    if not path.is_file():
        raise SourceError(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in _PDF_SUFFIXES:
        text = pdf_to_text(path.read_bytes())
    elif suffix in _HTML_SUFFIXES:
        text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return _require_text(text, str(path))


def fetch_source(url: str, *, settings: Settings) -> SourceDoc:
    """Fetch a URL through the proxy and extract text by content type."""
    result = fetch_url(url, settings=settings)
    content_type = result.content_type
    if "pdf" in content_type:
        text = pdf_to_text(result.content)
    elif "html" in content_type or "xml" in content_type:
        text = html_to_text(result.content.decode("utf-8", errors="replace"))
    else:
        text = result.content.decode("utf-8", errors="replace")
    return _require_text(text, url)
