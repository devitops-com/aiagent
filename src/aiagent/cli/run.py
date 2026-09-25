"""``aiagent run <skill>`` — run a skill on one input, or on many with ``--jsonl``."""

from __future__ import annotations

import contextvars
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import typer

from aiagent.cli._common import echo_err, get_settings, print_json
from aiagent.cli._runtime import configure_lm, prediction_to_dict
from aiagent.cli._verbosity import VERBOSE_OPTION, verbosity_scope
from aiagent.exceptions import AiagentError
from aiagent.skills.loader import build_module
from aiagent.skills.registry import load_registry
from aiagent.skills.router import route

_DEFAULT_CONCURRENCY = 4  # devai's teacher serves 4 requests at once
_MAX_CONCURRENCY = 16


def run(
    skill: str = typer.Argument(..., help="Skill name (or free text with --route)."),
    text: str | None = typer.Option(None, "--text", "-t", help="Input text."),
    input_file: Path | None = typer.Option(
        None, "--input", "-i", help="JSON object of inputs."
    ),
    model: str | None = typer.Option(None, "--model", help="Model override."),
    use_route: bool = typer.Option(
        False, "--route", help="Treat SKILL as free text and route to a skill."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
    jsonl: Path | None = typer.Option(
        None,
        "--jsonl",
        help="Many inputs: one JSON object per line ('-' reads stdin). Prints one "
        "JSON prediction per line, in input order.",
    ),
    concurrency: int = typer.Option(
        _DEFAULT_CONCURRENCY,
        "--concurrency",
        min=1,
        max=_MAX_CONCURRENCY,
        help="With --jsonl: inputs in flight at once.",
    ),
    verbose: int = VERBOSE_OPTION,
) -> None:
    """Run a skill and print its prediction. Requires a reachable router."""
    settings = get_settings()
    registry, _ = load_registry(settings)
    target = route(skill, registry).skill if use_route else registry.get(skill)

    if jsonl is not None:
        if text is not None or input_file is not None:
            raise AiagentError("--jsonl cannot be combined with --text or --input")
        rows = _read_jsonl(jsonl)
    else:
        inputs = _resolve_inputs(text, input_file)
    configure_lm(settings, model)
    module = build_module(target)
    if settings.system1_mode.get(target.name, "off") != "off":
        from aiagent.system1.cascade import apply_system1  # lazy: numpy, ORT

        apply_system1(module, target, settings)
    with verbosity_scope(verbose=verbose, skill=target.name):
        if jsonl is not None:
            failed = _run_batch(module, rows, concurrency)
        else:
            data = prediction_to_dict(module(**inputs))

    if jsonl is not None:
        if failed:
            echo_err(f"{failed} of {len(rows)} inputs failed")
            raise typer.Exit(1)
        return

    if as_json:
        print_json(data)
    else:
        for key in sorted(data):
            typer.echo(f"{key}: {data[key]}")


def _resolve_inputs(text: str | None, input_file: Path | None) -> dict[str, Any]:
    if text is not None:
        return {"text": text}
    if input_file is not None:
        raw = json.loads(input_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise AiagentError("--input must contain a JSON object")
        return raw
    raise AiagentError("provide --text, --input or --jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The non-blank lines of ``path`` ('-' is stdin), each a JSON object of inputs."""
    try:
        raw = sys.stdin.buffer.read() if str(path) == "-" else path.read_bytes()
        text = raw.decode("utf-8-sig")
    except OSError as exc:
        raise AiagentError(f"cannot read --jsonl {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise AiagentError(
            f"--jsonl {path}: not UTF-8 ({exc.reason} at byte {exc.start})"
        ) from exc
    rows: list[dict[str, Any]] = []
    # Only "\n" ends a JSONL line: str.splitlines() would also break on the U+2028,
    # U+2029 and U+0085 that json.dumps(..., ensure_ascii=False) leaves in strings.
    for number, line in enumerate(text.split("\n"), start=1):
        line = line.removesuffix("\r")
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AiagentError(
                f"--jsonl line {number}: not valid JSON ({exc.msg})"
            ) from exc
        if not isinstance(row, dict):
            raise AiagentError(f"--jsonl line {number}: not a JSON object")
        rows.append(row)
    if not rows:
        raise AiagentError("--jsonl: no inputs")
    return rows


def _run_batch(module: Any, rows: list[dict[str, Any]], concurrency: int) -> int:
    """Run ``module`` on every row, ``concurrency`` at a time; print one JSON line per
    row in input order (``{"error": ...}`` for a row that failed); return the failures.
    """

    def one(inputs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        try:
            return prediction_to_dict(module(**inputs)), False
        except Exception as exc:  # reported in the row's place; the batch goes on
            return {"error": f"{type(exc).__name__}: {exc}"}, True

    failed = 0
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        # dspy keeps its context overrides in contextvars: give each task a copy.
        futures = [
            pool.submit(contextvars.copy_context().run, one, row) for row in rows
        ]
        for future in futures:
            data, bad = future.result()
            failed += bad
            line = json.dumps(data, default=str, sort_keys=True, ensure_ascii=False)
            typer.echo(line)
    except KeyboardInterrupt:
        echo_err("interrupted: waiting for the inputs already in flight to finish")
        raise
    finally:
        pool.shutdown(cancel_futures=True)  # Ctrl-C: drop what has not started
    return failed

