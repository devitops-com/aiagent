"""Tests for skills list and the devai shell entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import aiagent.cli.shell as shell_mod
from aiagent.cli.app import app

runner = CliRunner()


def test_skills_list_table() -> None:
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0
    assert "extract" in result.stdout
    assert "chat" in result.stdout


def test_skills_list_json() -> None:
    result = runner.invoke(app, ["skills", "list", "--json"])
    assert result.exit_code == 0
    assert '"extract"' in result.stdout


def test_skills_list_source_builtin() -> None:
    result = runner.invoke(app, ["skills", "list", "--source", "builtin"])
    assert result.exit_code == 0
    assert "extract" in result.stdout


def test_skills_list_warns_on_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sd = tmp_path / "skills"
    (sd / "broken").mkdir(parents=True)
    (sd / "broken" / "SKILL.md").write_text("not a manifest")
    monkeypatch.setenv("AIAGENT_SKILLS_DIR", str(sd))
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0
    combined = result.output
    try:
        combined += result.stderr
    except (ValueError, AttributeError):
        pass  # stderr merged into output on older Click
    assert "warning" in combined and "broken" in combined


def test_shell_prints_banner_and_execs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_execvp(path: str, args: list[str]) -> None:
        calls["path"] = path
        calls["args"] = args

    monkeypatch.setattr(shell_mod.os, "execvp", fake_execvp)
    result = runner.invoke(app, ["shell", "--model", "llama3.2:3b"])
    assert result.exit_code == 0
    assert "devai agent shell" in result.stdout
    assert "path" in calls
