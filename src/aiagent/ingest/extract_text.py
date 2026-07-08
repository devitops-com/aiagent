"""Extract plain text from HTML and PDF payloads (pure; stdlib + pypdf).

HTML uses the stdlib ``html.parser`` (no extra dependency): it drops
non-content elements (script/style/head/…) and keeps visible text. PDF uses
``pypdf`` (pure-Python, torch-free), imported lazily so non-PDF paths pay
nothing for it.
"""

from __future__ import annotations

import io
import re
from html.parser import HTMLParser

_SKIP_TAGS = frozenset({"script", "style", "head", "noscript", "template", "svg"})
_BLANK_RUN = re.compile(r"\n{3,}")


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text, skipping the content of non-rendered elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._chunks.append(stripped)

    def text(self) -> str:
        return "\n".join(self._chunks)


def _collapse(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> str:
    """Return the visible text of an HTML document."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    return _collapse(parser.text())


def pdf_to_text(data: bytes) -> str:
    """Return the extractable text of a PDF document."""
    from pypdf import PdfReader  # lazy: only imported when a PDF is processed

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _collapse("\n\n".join(pages))
