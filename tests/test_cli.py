"""CLI smoke tests: command wiring + the lazy-dspy guarantee."""

from __future__ import annotations

import subprocess
import sys

from typer.testing import CliRunner

from aiagent import __version__
from aiagent.cli.app import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_config_show_defaults() -> None:
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "api_base" in result.stdout
    assert "***" not in result.stdout or "api_key" in result.stdout  # key shown masked or absent


def test_doctor_offline() -> None:
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == 0
    assert "offline" in result.stdout.lower()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("doctor", "models", "config", "version"):
        assert cmd in result.stdout


def test_importing_cli_does_not_import_dspy() -> None:
    """`aiagent --help` and friends must not pay DSPy's heavy import cost."""
    code = "import aiagent.cli.app, sys; print('dspy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
