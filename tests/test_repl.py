"""Unit tests for the chat REPL reader (history, completion, fzf pickers)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aiagent.cli import repl
from aiagent.cli.repl import PromptReader, complete_command, fzf_available, fzf_pick

_COMMANDS = (":quit", ":q", ":reset", ":history", ":help")


def test_complete_command_prefix_matches() -> None:
    assert complete_command(":h", _COMMANDS) == [":history", ":help"]
    assert complete_command(":q", _COMMANDS) == [":quit", ":q"]
    assert complete_command(":zzz", _COMMANDS) == []


def test_fzf_available_reflects_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repl.shutil, "which", lambda _name: "/usr/bin/fzf")
    assert fzf_available() is True
    monkeypatch.setattr(repl.shutil, "which", lambda _name: None)
    assert fzf_available() is False


def test_fzf_pick_returns_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout="second\n")

    monkeypatch.setattr(repl.subprocess, "run", fake_run)
    assert fzf_pick(["first", "second"]) == "second"


def test_fzf_pick_empty_and_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    # No candidates -> never shells out.
    assert fzf_pick([]) is None

    # Non-zero exit (Esc/no-match) -> None.
    monkeypatch.setattr(
        repl.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 130, stdout=""),
    )
    assert fzf_pick(["x"]) is None


def test_fzf_pick_missing_binary_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(repl.subprocess, "run", boom)
    assert fzf_pick(["x"]) is None


def test_history_persists_and_seeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hist = tmp_path / "history"
    r1 = PromptReader(history_path=hist, commands=_COMMANDS, use_fzf=False)
    r1.remember("first question")
    r1.remember("second question")
    r1.save()

    assert hist.read_text(encoding="utf-8").splitlines() == [
        "first question",
        "second question",
    ]

    # A new reader loads persisted history; the fzf picker sees most-recent first.
    r2 = PromptReader(history_path=hist, commands=_COMMANDS, use_fzf=True)
    captured: list[str] = []

    def fake_pick(candidates: list[str], **_k: object) -> str | None:
        captured.extend(candidates)
        return candidates[0]

    monkeypatch.setattr(repl, "fzf_pick", fake_pick)
    assert r2.pick_history() == "second question"
    assert captured == ["second question", "first question"]


def test_history_dedups_keeping_order(tmp_path: Path) -> None:
    r = PromptReader(history_path=tmp_path / "history", commands=_COMMANDS, use_fzf=False)
    r.remember("a")
    r.remember("b")
    r.remember("a")  # repeat -> not duplicated; first-seen order preserved
    r.save()
    assert (tmp_path / "history").read_text(encoding="utf-8").splitlines() == ["a", "b"]


def test_pickers_return_none_without_fzf(tmp_path: Path) -> None:
    r = PromptReader(history_path=tmp_path / "history", commands=_COMMANDS, use_fzf=False)
    r.remember("x")
    assert r.fzf_enabled is False
    assert r.pick_history() is None
    assert r.pick_command() is None


def test_read_returns_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    r = PromptReader(history_path=tmp_path / "history", commands=_COMMANDS, use_fzf=False)
    monkeypatch.setattr("builtins.input", lambda _prompt: "typed line")
    assert r.read("you> ") == "typed line"


def test_save_is_best_effort_on_bad_path(tmp_path: Path) -> None:
    # history_path under a file (not a dir) can't be created; save must not raise.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    r = PromptReader(history_path=blocker / "history", commands=_COMMANDS, use_fzf=False)
    r.remember("y")
    r.save()  # no exception
