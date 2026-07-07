"""CLI tests for the LLM-driven commands, using DSPy's DummyLM (no network)."""

from __future__ import annotations

from pathlib import Path

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from typer.testing import CliRunner

from aiagent.cli.app import app

runner = CliRunner()

_EXTRACT = {"reasoning": "r", "merchant": "Cafe", "date": "2026-02-14", "amount": "4.95"}
_CHAT = {"answer": "hello there"}


def _install_dummy(monkeypatch: pytest.MonkeyPatch, module: str, response: dict) -> None:
    """Replace a command's configure_lm with one that installs a DummyLM."""

    def fake(_settings: object, _model: object) -> None:
        dspy.configure(lm=DummyLM([dict(response) for _ in range(800)]))

    monkeypatch.setattr(f"{module}.configure_lm", fake)


def test_run_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.run", _EXTRACT)
    result = runner.invoke(app, ["run", "extract", "--text", "Cafe 4.95 on 2026-02-14"])
    assert result.exit_code == 0
    assert "merchant: Cafe" in result.stdout


def test_run_route(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.run", _EXTRACT)
    result = runner.invoke(app, ["run", "extract", "--route", "--text", "x", "--json"])
    assert result.exit_code == 0
    assert '"merchant"' in result.stdout


def test_run_requires_input() -> None:
    result = runner.invoke(app, ["run", "extract"])
    assert result.exit_code == 1


def test_run_unknown_skill() -> None:
    result = runner.invoke(app, ["run", "nope", "--text", "x"])
    assert result.exit_code == 1


def test_eval_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.eval_cmd", _EXTRACT)
    result = runner.invoke(app, ["eval", "extract", "--json"])
    assert result.exit_code == 0
    assert '"score"' in result.stdout


def test_eval_no_devset() -> None:
    result = runner.invoke(app, ["eval", "chat"])
    assert result.exit_code == 1


def test_run_verbose_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.run", _EXTRACT)
    result = runner.invoke(
        app, ["run", "extract", "--text", "Cafe 4.95 on 2026-02-14", "-v"]
    )
    assert result.exit_code == 0
    assert "[-v] skill=extract model=dummy" in result.stderr
    assert "[-v] elapsed=" in result.stderr
    assert "calls=1" in result.stderr


def test_run_verbose_dspy_level(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.run", _EXTRACT)
    result = runner.invoke(
        app, ["run", "extract", "--text", "Cafe 4.95 on 2026-02-14", "-vv"]
    )
    assert result.exit_code == 0
    assert "System message:" in result.stderr
    assert "User message:" in result.stderr
    assert "Cafe 4.95 on 2026-02-14" in result.stderr


def test_run_verbose_wire_level(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.run", _EXTRACT)
    result = runner.invoke(
        app, ["run", "extract", "--text", "Cafe 4.95 on 2026-02-14", "-vvv"]
    )
    assert result.exit_code == 0
    assert "[-vvv] response_model=" in result.stderr


def test_run_no_verbose_flag_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.run", _EXTRACT)
    result = runner.invoke(app, ["run", "extract", "--text", "x"])
    assert result.exit_code == 0
    assert result.stderr == ""


def test_eval_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.eval_cmd", _EXTRACT)
    result = runner.invoke(app, ["eval", "extract", "-vv"])
    assert result.exit_code == 0
    assert "[-v] skill=extract" in result.stderr
    assert "System message:" in result.stderr


def test_optimize_verbose(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.optimize_cmd", _EXTRACT)
    out = tmp_path / "extract.json"
    result = runner.invoke(app, ["optimize", "extract", "--out", str(out), "-v"])
    assert result.exit_code == 0
    assert "[-v] skill=extract" in result.stderr
    assert "[-v] elapsed=" in result.stderr


def test_optimize_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.optimize_cmd", _EXTRACT)
    out = tmp_path / "extract.json"
    result = runner.invoke(app, ["optimize", "extract", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "saved" in result.stdout


def test_optimize_no_trainset() -> None:
    result = runner.invoke(app, ["optimize", "chat"])
    assert result.exit_code == 1


def _feed(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    """Drive the chat REPL by feeding ``lines`` to PromptReader.read."""
    answers = iter(lines)
    # A non-empty prefill simulates the user accepting the pre-filled line as-is.
    monkeypatch.setattr(
        "aiagent.cli.repl.PromptReader.read",
        lambda self, prompt, prefill="": prefill or next(answers),
    )


def test_chat_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.chat", _CHAT)
    _feed(monkeypatch, ["hi", ":reset", "again", ":quit"])
    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 0
    assert "bot> hello there" in result.stdout


def test_chat_history_picker_prefills(monkeypatch: pytest.MonkeyPatch) -> None:
    # ':history' routes through the fzf picker; the selection is submitted next.
    _install_dummy(monkeypatch, "aiagent.cli.chat", _CHAT)
    _feed(monkeypatch, [":history", ":quit"])
    monkeypatch.setattr(
        "aiagent.cli.repl.PromptReader.pick_history", lambda self: "picked prompt"
    )
    monkeypatch.setattr(
        "aiagent.cli.repl.PromptReader.fzf_enabled", property(lambda self: True)
    )
    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 0
    assert "bot> hello there" in result.stdout


def test_chat_history_picker_hint_without_fzf(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.chat", _CHAT)
    _feed(monkeypatch, [":history", ":quit"])
    monkeypatch.setattr(
        "aiagent.cli.repl.PromptReader.fzf_enabled", property(lambda self: False)
    )
    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 0
    assert "fzf not found" in result.stdout


def test_run_chat_single_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.run", _CHAT)
    result = runner.invoke(app, ["run", "chat", "--text", "hi"])
    assert result.exit_code == 0
    assert "answer: hello there" in result.stdout


def test_chat_session_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy(monkeypatch, "aiagent.cli.chat", _CHAT)

    # First run: answer one turn, then quit -> persists to session 'work'.
    _feed(monkeypatch, ["hello", ":quit"])
    r1 = runner.invoke(app, ["chat", "--session", "work"])
    assert r1.exit_code == 0

    # Second run: the same session resumes with the prior turn.
    _feed(monkeypatch, [":quit"])
    r2 = runner.invoke(app, ["chat", "--session", "work"])
    assert r2.exit_code == 0
    assert "resumed session 'work': 1 turns" in r2.stdout

    # --new discards the resumed history.
    _feed(monkeypatch, [":quit"])
    r3 = runner.invoke(app, ["chat", "--session", "work", "--new"])
    assert r3.exit_code == 0
    assert "resumed session" not in r3.stdout
