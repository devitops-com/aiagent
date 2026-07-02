"""Unit tests for the resumable chat session store."""

from __future__ import annotations

import pytest

from aiagent.cli.chat_session import ChatSession, session_path
from aiagent.config import load_settings
from aiagent.exceptions import AiagentError


def test_append_persists_and_reloads() -> None:
    settings = load_settings()  # sessions_dir points at a temp dir (conftest)
    convo = ChatSession.load(settings, "demo")
    assert convo.turns == []
    convo.append("hi", "hello")
    convo.append("bye", "goodbye")

    reloaded = ChatSession.load(settings, "demo")
    assert reloaded.turns == [
        {"text": "hi", "answer": "hello"},
        {"text": "bye", "answer": "goodbye"},
    ]


def test_clear_empties_the_session() -> None:
    settings = load_settings()
    convo = ChatSession.load(settings, "demo")
    convo.append("hi", "hello")
    convo.clear()
    assert ChatSession.load(settings, "demo").turns == []


@pytest.mark.parametrize("name", ["../escape", "a/b", "", "with space", ".hidden/../x"])
def test_unsafe_session_name_rejected(name: str) -> None:
    with pytest.raises(AiagentError, match="invalid session name"):
        session_path(load_settings(), name)


def test_corrupt_session_file_raises() -> None:
    settings = load_settings()
    path = session_path(settings, "broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json}", encoding="utf-8")
    with pytest.raises(AiagentError, match="unreadable"):
        ChatSession.load(settings, "broken")
