"""CLI smoke tests: command wiring + the lazy-dspy guarantee."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from aiagent import __version__
from aiagent.cli.app import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _plain(text: str) -> str:
    """Help output with ANSI escapes removed.

    Rich colours its output whenever it detects GitHub Actions
    (``GITHUB_ACTIONS=1``), even though CliRunner's stream is not a terminal, so
    under CI a help line begins ``\x1b[1m`` rather than its text. Anything
    asserting on the *structure* of help (line starts, prefixes) has to strip
    escapes first or it passes locally and fails only in CI.
    """
    return _ANSI.sub("", text)


@pytest.mark.parametrize("cmd", ["run", "eval", "optimize"])
def test_arg_command_help_renders_metavar(cmd: str) -> None:
    """Help for a command with a positional argument must render its metavar.

    A pre-Click-8.2 Typer overrides ``make_metavar`` without ``ctx`` and crashes
    the usage/help render for any arg-bearing command; this guards that pairing
    can't regress (issue #1).

    The spelling is Typer's to choose and it changed in 0.27.0 (breaking,
    fastapi/typer#1863): ``SKILL`` became ``{skill}`` once metavars stopped
    being upper-cased. Both are fine — assert only that the argument is named
    in the usage line, and match case-insensitively so either spelling passes.
    Scoping to that line matters: ``run`` renders the word "SKILL" in an
    unrelated ``--route`` option description, so a whole-output check passed
    even when its usage metavar had gone.
    """
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0
    usage = next(
        (ln for ln in _plain(result.stdout).splitlines() if ln.strip().startswith("Usage:")),
        "",
    )
    assert "skill" in usage.lower(), f"no skill metavar in usage line: {usage!r}"


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_tracks_package_metadata() -> None:
    # `__version__` must derive from the installed package metadata, never a
    # hand-maintained literal that can drift from the release (issue #5).
    from importlib.metadata import version

    assert __version__ == version("aiagent")


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


@pytest.mark.parametrize(
    "argv", [["nosuchcmd"], ["run"]], ids=["unknown-command", "missing-argument"]
)
def test_usage_error_prints_help_not_traceback(argv: list[str]) -> None:
    """A usage error must print help and exit 2 — never dump a traceback.

    ``main()`` catches Typer's *vendored* Click exceptions, which live outside
    the stdlib ``click`` hierarchy. When Typer 0.27.2 moved ``Abort`` out of
    ``typer._click.exceptions`` (fastapi/typer#1942) the combined import of
    ``Abort``+``UsageError`` started raising ``ImportError``, so the vendored
    ``UsageError`` was dropped from ``_USAGE_ERRORS`` too and every usage error
    escaped ``main()`` as an unhandled traceback.

    The other CLI tests invoke ``app`` through ``CliRunner`` and so never reach
    ``main()``'s handlers; this drives the module entry point end to end, which
    is the only path that exercises them. By design this CLI answers a usage
    error with the context-appropriate help text rather than a one-line message,
    so the assertions cover exit code and help — not any error wording.
    """
    out = subprocess.run(
        [sys.executable, "-m", "aiagent.cli.app", *argv],
        capture_output=True,
        text=True,
    )
    combined = _plain(out.stdout + out.stderr)
    assert "Traceback" not in combined, f"usage error dumped a traceback:\n{combined}"
    assert out.returncode == 2, f"expected exit 2, got {out.returncode}:\n{combined}"
    assert "Usage:" in combined, f"help text not printed:\n{combined}"
