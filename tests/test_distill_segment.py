"""Segments that fit the student's token budget, and the documents they come from."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

import pytest

import aiagent.distill.segment as segment_mod
from aiagent.config import Settings, load_settings
from aiagent.distill.segment import (
    DOC_SUFFIXES,
    Sources,
    collect_documents,
    question_room,
    segment_text,
    state_fits,
)
from aiagent.distill.splits import document_id
from aiagent.exceptions import DataLoadError, DistillError, SourceError
from aiagent.ingest.sources import SourceDoc
from aiagent.system1.sequence import SequenceTokenizer, prepare
from system1_helpers import FIXTURE, FIXTURE_HEAD_MAX_LEN, FIXTURE_MAX_LEN

LIMITS = {"max_len": FIXTURE_MAX_LEN, "head_max_len": FIXTURE_HEAD_MAX_LEN}
POLARITY = {
    "type": "choice",
    "instructions": "Overall sentiment polarity of the passage.",
    "criteria": ["negative", "neutral", "mixed", "positive"],
}


def chars(limit: int) -> Callable[[str], bool]:
    return lambda text: len(text) <= limit


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def assert_segments(text: str, segments: list[str], limit: int) -> None:
    assert all(0 < len(s) <= limit for s in segments), segments
    assert "".join(squash(s) for s in segments) == squash(text)  # nothing dropped, in order


# --------------------------------------------------------------------------- segment_text


def test_empty_text_has_no_segments() -> None:
    assert segment_text("", fits=chars(10)) == []
    assert segment_text(" \n\n \n", fits=chars(10)) == []


def test_paragraphs_are_packed() -> None:
    text = "aaa bbb.\n\nccc ddd.\n \neee"
    segments = segment_text(text, fits=chars(20))
    assert segments == ["aaa bbb.\n\nccc ddd.", "eee"]
    assert_segments(text, segments, 20)


def test_a_whole_text_that_fits_is_one_segment() -> None:
    text = "  one.\n\ntwo.  "
    assert segment_text(text, fits=chars(100)) == ["one.\n\ntwo."]


def test_sentence_fallback() -> None:
    text = "One two three. Four five six! Seven?"
    segments = segment_text(text, fits=chars(22))
    assert segments == ["One two three.", "Four five six!\n\nSeven?"]
    assert_segments(text, segments, 22)


def test_word_fallback() -> None:
    text = "alpha beta gamma delta epsilon zeta"
    segments = segment_text(text, fits=chars(12))
    assert segments == ["alpha beta", "gamma delta", "epsilon zeta"]
    assert_segments(text, segments, 12)


def test_character_fallback() -> None:
    text = "abcdefghijklmnopqrstuvwxyz ok. Short one."
    segments = segment_text(text, fits=chars(10))
    assert segments == ["abcdefghij", "klmnopqrst", "uvwxyz ok.", "Short one."]
    assert_segments(text, segments, 10)


def test_mixed_text_keeps_everything_in_order() -> None:
    text = (
        "A short opening paragraph.\n\n"
        + "This sentence is fine. " * 3
        + "Supercalifragilisticexpialidocious-and-then-some words follow here.\n\n"
        + "Tail."
    )
    for limit in (8, 15, 30, 60, 1000):
        assert_segments(text, segment_text(text, fits=chars(limit)), limit)


def test_no_room_raises() -> None:
    with pytest.raises(DistillError, match="questions leave no room for the state"):
        segment_text("x", fits=chars(0))


# --------------------------------------------------------------------------- student tokens


@pytest.fixture(scope="module")
def tok() -> SequenceTokenizer:
    return SequenceTokenizer.from_dir(FIXTURE / "tokenizer")


def test_question_room_is_the_smallest_room(tok: SequenceTokenizer) -> None:
    many = {"type": "choice", "instructions": "Pick one.", "criteria": [f"option {i}" for i in range(8)]}
    questions = prepare({"polarity": POLARITY, "many": many})
    rooms = [tok.room(q, **LIMITS) for q in questions.values()]
    assert rooms[0] != rooms[1]
    assert question_room(tok, questions, **LIMITS) == min(rooms)


def test_every_segment_fits_the_student(tok: SequenceTokenizer) -> None:
    questions = prepare({"polarity": POLARITY})
    room = question_room(tok, questions, **LIMITS)
    fits = state_fits(tok, input_field="text", room=room)
    paragraph = " ".join(f"word{i} good service here." for i in range(40))
    # "a.a.a..." is one whitespace word of 300 tokens: only the character fallback cuts it
    text = "\n\n".join([paragraph, "Short.", "a." * 150, paragraph])
    segments = segment_text(text, fits=fits)
    assert len(segments) > 4
    for seg in segments:
        assert fits(seg)
        assert tok.fits({"text": seg}, questions, **LIMITS)
    assert "".join(squash(s) for s in segments) == squash(text)


def test_state_fits_counts_the_serialized_state(tok: SequenceTokenizer) -> None:
    fits = state_fits(tok, input_field="text", room=len(tok.state_ids({"text": "good service"})))
    assert fits("good service")
    assert not fits("good service here")


def test_no_room_names_the_room(tok: SequenceTokenizer) -> None:
    with pytest.raises(DistillError, match=r"room=3\b"):
        segment_text("good", fits=state_fits(tok, input_field="text", room=3))


# --------------------------------------------------------------------------- collect_documents


@pytest.fixture
def settings() -> Settings:
    return load_settings()


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_doc_suffixes() -> None:
    assert DOC_SUFFIXES == frozenset({".txt", ".md", ".html", ".htm", ".xhtml", ".pdf"})


def test_files(tmp_path: Path, settings: Settings) -> None:
    a = write(tmp_path / "a.txt", "  Alpha document.  \n")
    b = write(tmp_path / "b.html", "<html><body><p>Beta page.</p></body></html>")
    docs = collect_documents(Sources(files=(a, b)), input_field="text", settings=settings)
    assert docs == [SourceDoc(str(a), "Alpha document."), SourceDoc(str(b), "Beta page.")]


def test_a_missing_or_empty_file_is_an_error(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(SourceError, match="file not found"):
        collect_documents(Sources(files=(tmp_path / "nope.txt",)), input_field="text", settings=settings)
    empty = write(tmp_path / "empty.txt", "  \n")
    with pytest.raises(SourceError, match="no extractable text"):
        collect_documents(Sources(files=(empty,)), input_field="text", settings=settings)


def test_dirs_recurse_sorted_filter_and_skip_empty(
    tmp_path: Path, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "corpus"
    write(root / "sub" / "c.md", "Charlie notes.")
    write(root / "d.txt", "Delta text.")
    write(root / "a.PDF.bak", "not a document")
    write(root / "data.json", '{"text": "skipped"}')
    write(root / "g.TXT", "Golf.")
    write(root / "sub" / "deeper" / "e.htm", "<p>Echo.</p>")
    empty = write(root / "b_empty.txt", "\n")
    with caplog.at_level(logging.WARNING, logger="aiagent.distill.segment"):
        docs = collect_documents(Sources(dirs=(root,)), input_field="text", settings=settings)
    assert [d.text for d in docs] == ["Delta text.", "Golf.", "Charlie notes.", "Echo."]
    assert [d.origin for d in docs] == [
        str(root / "d.txt"), str(root / "g.TXT"), str(root / "sub" / "c.md"),
        str(root / "sub" / "deeper" / "e.htm"),
    ]
    assert str(empty) in caplog.text


def test_a_missing_dir_is_an_error(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(SourceError, match="directory not found"):
        collect_documents(Sources(dirs=(tmp_path / "nope",)), input_field="text", settings=settings)


def test_jsonl(tmp_path: Path, settings: Settings) -> None:
    path = write(
        tmp_path / "reviews.jsonl",
        json.dumps({"text": " Great value. ", "stars": 5}) + "\n\n"
        + json.dumps({"text": "Late delivery, čudno."}, ensure_ascii=False) + "\n",
    )
    docs = collect_documents(Sources(jsonl=(path,)), input_field="text", settings=settings)
    assert docs == [
        SourceDoc(f"{path}:1", "Great value."),
        SourceDoc(f"{path}:3", "Late delivery, čudno."),
    ]


@pytest.mark.parametrize(
    ("line", "problem"),
    [
        ('{"body": "x"}', "'text'"),
        ('{"text": 3}', "'text'"),
        ('{"text": "   "}', "'text'"),
        ('["text"]', "JSON object"),
        ("{not json", "invalid JSON"),
    ],
)
def test_jsonl_errors_name_path_and_line(
    tmp_path: Path, settings: Settings, line: str, problem: str
) -> None:
    path = write(tmp_path / "bad.jsonl", '{"text": "fine"}\n' + line + "\n")
    with pytest.raises(DataLoadError, match=re.escape(f"{path}:2: ") + ".*" + re.escape(problem)):
        collect_documents(Sources(jsonl=(path,)), input_field="text", settings=settings)


def test_a_missing_jsonl_is_an_error(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(DataLoadError, match="nope.jsonl"):
        collect_documents(Sources(jsonl=(tmp_path / "nope.jsonl",)), input_field="text", settings=settings)


def test_urls_use_fetch_source(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    seen: list[tuple[str, Settings]] = []

    def fake_fetch(url: str, *, settings: Settings) -> SourceDoc:
        seen.append((url, settings))
        return SourceDoc(origin=url, text=f"Page at {url}.")

    monkeypatch.setattr(segment_mod, "fetch_source", fake_fetch)
    docs = collect_documents(
        Sources(urls=("https://a.example/x", "https://b.example/y")), input_field="text", settings=settings
    )
    assert [d.origin for d in docs] == ["https://a.example/x", "https://b.example/y"]
    assert seen == [("https://a.example/x", settings), ("https://b.example/y", settings)]


def test_source_order_and_dedupe(tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        segment_mod, "fetch_source", lambda url, *, settings: SourceDoc(url, "Same text.")
    )
    a = write(tmp_path / "a.txt", "Same text.")
    corpus = tmp_path / "dir"
    write(corpus / "b.txt", "Dir text.")
    write(corpus / "c.txt", "Same text.\n")
    jsonl = write(tmp_path / "x.jsonl", '{"text": "Dir text."}\n{"text": "Json text."}\n')
    sources = Sources(files=(a,), dirs=(corpus,), urls=("https://u.example",), jsonl=(jsonl,))
    docs = collect_documents(sources, input_field="text", settings=settings)
    assert docs == [
        SourceDoc(str(a), "Same text."),
        SourceDoc(str(corpus / "b.txt"), "Dir text."),
        SourceDoc(f"{jsonl}:2", "Json text."),
    ]
    assert len({document_id(d.text) for d in docs}) == 3


def test_no_documents_is_an_error(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(DistillError, match="no documents with text in the given sources"):
        collect_documents(Sources(), input_field="text", settings=settings)
    write(tmp_path / "only" / "empty.md", " ")
    with pytest.raises(DistillError, match="no documents"):
        collect_documents(Sources(dirs=(tmp_path / "only",)), input_field="text", settings=settings)
