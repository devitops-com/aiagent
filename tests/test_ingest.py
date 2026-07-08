"""Tests for the data-source ingestion layer (files, URLs, text extraction)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from aiagent.config import load_settings
from aiagent.exceptions import SourceError
from aiagent.ingest.extract_text import html_to_text, pdf_to_text
from aiagent.ingest.fetch import fetch_url
from aiagent.ingest.sources import fetch_source, read_file


def make_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF whose text pypdf can extract."""
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    stream = b"BT /F1 24 Tf 72 700 Td (" + text.encode("latin-1") + b") Tj ET"
    objs.append(b"<</Length %d>>stream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj" % i + body + b"endobj\n"
    xref_pos = len(out)
    n = len(objs) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % n
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (n, xref_pos)
    return bytes(out)


# --- text extraction ------------------------------------------------------


def test_html_to_text_drops_scripts_and_styles() -> None:
    html = (
        "<html><head><style>.x{color:red}</style></head>"
        "<body><h1>Title</h1><script>evil()</script>"
        "<p>Hello world.</p></body></html>"
    )
    text = html_to_text(html)
    assert "Title" in text
    assert "Hello world." in text
    assert "evil" not in text
    assert "color:red" not in text


def test_pdf_to_text_extracts_body() -> None:
    text = pdf_to_text(make_pdf("The service was excellent and I am happy."))
    assert "excellent" in text


# --- read_file dispatch ---------------------------------------------------


def test_read_file_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("A perfectly fine day.", encoding="utf-8")
    doc = read_file(path)
    assert doc.origin == str(path)
    assert doc.text == "A perfectly fine day."


def test_read_file_html(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text("<p>Nice <b>work</b></p>", encoding="utf-8")
    assert "Nice" in read_file(path).text


def test_read_file_pdf(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_pdf("Terrible experience overall."))
    assert "Terrible" in read_file(path).text


def test_read_file_missing(tmp_path: Path) -> None:
    with pytest.raises(SourceError):
        read_file(tmp_path / "nope.txt")


def test_read_file_empty_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(SourceError):
        read_file(path)


# --- URL fetch (httpx mocked) ---------------------------------------------


class _Resp:
    def __init__(self, content: bytes, content_type: str, status: int = 200) -> None:
        self.content = content
        self.headers = {"content-type": content_type}
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=None, response=None)  # type: ignore[arg-type]


def _client_factory(handler: Any) -> type:
    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            self.kwargs = k

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def get(self, url: str) -> _Resp:
            return handler(url)

    return _Client


def test_fetch_url_uses_configured_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            captured.update(k)

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def get(self, url: str) -> _Resp:
            return _Resp(b"hello", "text/plain")

    monkeypatch.setattr(httpx, "Client", _Client)
    settings = load_settings()  # default proxy_url = pipelock
    result = fetch_url("https://example.com", settings=settings)
    assert result.content == b"hello"
    assert captured["proxy"] == "http://devai-pipelock:8888"


def test_fetch_source_html(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"<html><body><p>Loved it.</p></body></html>"
    monkeypatch.setattr(
        httpx, "Client", _client_factory(lambda _u: _Resp(body, "text/html"))
    )
    doc = fetch_source("https://example.com", settings=load_settings())
    assert "Loved it." in doc.text
    assert doc.origin == "https://example.com"


def test_fetch_source_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = make_pdf("Fetched PDF sentiment sample.")
    monkeypatch.setattr(
        httpx, "Client", _client_factory(lambda _u: _Resp(pdf, "application/pdf"))
    )
    doc = fetch_source("https://example.com/x.pdf", settings=load_settings())
    assert "Fetched" in doc.text


def test_fetch_url_rejects_bad_scheme() -> None:
    with pytest.raises(SourceError):
        fetch_url("file:///etc/passwd", settings=load_settings())


def test_fetch_url_wraps_connect_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_url: str) -> _Resp:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "Client", _client_factory(boom))
    with pytest.raises(SourceError):
        fetch_url("https://example.com", settings=load_settings())


def test_proxy_disabled_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            captured.update(k)

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def get(self, url: str) -> _Resp:
            return _Resp(b"x", "text/plain")

    monkeypatch.setenv("AIAGENT_PROXY_URL", "")
    monkeypatch.setattr(httpx, "Client", _Client)
    fetch_url("https://example.com", settings=load_settings())
    assert captured["proxy"] is None
