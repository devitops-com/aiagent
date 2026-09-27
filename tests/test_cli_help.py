"""Every command's --help: it exits 0 and ends with examples that really run.

The examples are epilogs built by ``aiagent.cli._common.examples``. These tests walk
the Click tree of the real app, so a new command without examples, an example that
names an option its command lacks, or one that Rich would mangle fails here.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterator
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from aiagent.cli._common import CLI_CONTEXT_SETTINGS, examples
from aiagent.cli.app import app
from aiagent.config import Settings

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
# Typer pads the epilog by one column on each side: an 80-column terminal shows 78.
_MAX_EXAMPLE_WIDTH = 78
_MIN_EXAMPLES, _MAX_EXAMPLES = 2, 5
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_ENV_PREFIX = "AIAGENT_"
_CONTROL_OPERATORS = {"|", "||", "&&", ";", "&"}
_REDIRECTIONS = {"<", ">", ">>"}
_HELP_OPTIONS = set(CLI_CONTEXT_SETTINGS["help_option_names"])


def _walk(cmd: Any, path: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], Any]]:
    yield path, cmd
    for name, sub in sorted(getattr(cmd, "commands", {}).items()):
        yield from _walk(sub, (*path, name))


ROOT = typer.main.get_command(app)
COMMANDS = list(_walk(ROOT, ()))
IDS = [" ".join(("aiagent", *path)) for path, _ in COMMANDS]


def _lines(cmd: Any) -> list[str]:
    """The epilog's lines (comments and commands), without the header and blanks."""
    paragraphs = (cmd.epilog or "").split("\n\n")
    assert paragraphs[0] == "Examples:", f"no Examples epilog: {cmd.epilog!r}"
    return [p for p in paragraphs[1:] if p]


def _command_lines(cmd: Any) -> list[str]:
    return [line for line in _lines(cmd) if not line.startswith("#")]


def _argv(line: str) -> list[str]:
    """The aiagent arguments of an example: leading VAR=value assignments dropped, cut
    at the first control operator, redirections and their targets removed.

    A dropped AIAGENT_* name must name a setting: Settings ignores unknown ones."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    tokens = list(lexer)
    while tokens and _ENV_ASSIGNMENT.match(tokens[0]):
        name = tokens.pop(0).split("=", 1)[0]
        if name.startswith(_ENV_PREFIX):
            field = name.removeprefix(_ENV_PREFIX).lower()
            assert field in Settings.model_fields, f"{name} is no setting: {line!r}"
    assert tokens[:1] == ["aiagent"], f"example does not run aiagent: {line!r}"
    argv: list[str] = []
    rest = iter(tokens[1:])
    for token in rest:
        if token in _CONTROL_OPERATORS:
            break
        if token in _REDIRECTIONS:
            next(rest, None)
            continue
        argv.append(token)
    return argv


def _resolve(argv: list[str]) -> tuple[tuple[str, ...], Any, list[str]]:
    """The command an example's arguments name, and the arguments left for it."""
    path: tuple[str, ...] = ()
    cmd = ROOT
    args = list(argv)
    while args and args[0] in getattr(cmd, "commands", {}):
        name = args.pop(0)
        path, cmd = (*path, name), cmd.commands[name]
    return path, cmd, args


def _options(args: list[str]) -> set[str]:
    """Option names in ``args``: ``--opt=value`` counts as ``--opt``, ``-vv`` as ``-v``."""
    names: set[str] = set()
    for arg in args:
        if arg == "--":
            break
        if not arg.startswith("-") or arg == "-":
            continue
        name = arg.split("=", 1)[0]
        if re.fullmatch(r"-([A-Za-z])\1+", name):
            name = name[:2]
        names.add(name)
    return names


def _declared(cmd: Any) -> set[str]:
    return {o for p in cmd.params for o in (*p.opts, *p.secondary_opts)} | _HELP_OPTIONS


@pytest.mark.parametrize("columns", ["80", "100"])
@pytest.mark.parametrize(("path", "cmd"), COMMANDS, ids=IDS)
def test_help_exits_0_and_shows_every_example_on_its_own_line(
    path: tuple[str, ...], cmd: Any, columns: str
) -> None:
    result = runner.invoke(app, [*path, "--help"], env={"COLUMNS": columns})
    assert result.exit_code == 0, result.output
    shown = [line.strip() for line in _ANSI.sub("", result.stdout).splitlines()]
    assert "Examples:" in shown
    for line in _lines(cmd):
        assert line in shown, f"{line!r} not shown whole in:\n{result.stdout}"


@pytest.mark.parametrize(("path", "cmd"), COMMANDS, ids=IDS)
def test_examples_run_real_commands_with_real_options(
    path: tuple[str, ...], cmd: Any
) -> None:
    comments = [line for line in _lines(cmd) if line.startswith("#")]
    assert _MIN_EXAMPLES <= len(comments) <= _MAX_EXAMPLES, comments
    targets = []
    for line in _command_lines(cmd):
        assert len(line) <= _MAX_EXAMPLE_WIDTH, f"too wide for 80 columns: {line!r}"
        target_path, target, args = _resolve(_argv(line))
        unknown = _options(args) - _declared(target)
        assert not unknown, f"{line!r}: {sorted(unknown)} not options of {target.name}"
        if not _options(args) & _HELP_OPTIONS:
            # Click parses it as the shell would hand it over: values, types and
            # Click's own ranges. A handler's own checks (label's --k 1-32, eval's
            # targets) do not run here.
            target.make_context(" ".join(("aiagent", *target_path)), list(args))
        targets.append(target_path)
    assert any(t[: len(path)] == path for t in targets), (
        f"no example runs aiagent {' '.join(path)}: {targets}"
    )


def test_example_blocks_render_as_one_paragraph_per_line() -> None:
    epilog = examples(("First", "aiagent version"), ("Second", "aiagent a", "aiagent b"))
    assert epilog.split("\n\n") == [
        "Examples:",
        "",
        "# First",
        "aiagent version",
        "",
        "# Second",
        "aiagent a",
        "aiagent b",
    ]


@pytest.mark.parametrize(
    ("line", "argv"),
    [
        ("aiagent config show --json | jq -r .x", ["config", "show", "--json"]),
        ("aiagent run p --jsonl a.jsonl > b.jsonl", ["run", "p", "--jsonl", "a.jsonl"]),
        ("""A_B='{"k":"v"}' aiagent run p -t "x y\"""", ["run", "p", "-t", "x y"]),
        ("aiagent version; command -v aiagent", ["version"]),
    ],
)
def test_example_argv_stops_where_the_shell_would(line: str, argv: list[str]) -> None:
    assert _argv(line) == argv


def test_example_env_names_must_be_settings() -> None:
    assert _argv("AIAGENT_REQUEST_TIMEOUT_S=1 aiagent version") == ["version"]
    with pytest.raises(AssertionError, match="AIAGENT_SYSTEM1_MOD is no setting"):
        _argv("AIAGENT_SYSTEM1_MOD='{}' aiagent version")


def test_option_names_fold_repeats_and_values() -> None:
    assert _options(["-vv", "--k=3", "-", "x", "--", "--not-an-option"]) == {"-v", "--k"}
