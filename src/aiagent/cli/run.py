"""``aiagent run <skill>`` — run a skill once on a single input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from aiagent.cli._common import get_settings, print_json
from aiagent.cli._runtime import configure_lm, prediction_to_dict
from aiagent.cli._verbosity import VERBOSE_OPTION, verbosity_scope
from aiagent.exceptions import AiagentError
from aiagent.skills.loader import build_module
from aiagent.skills.registry import load_registry
from aiagent.skills.router import route


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
    verbose: int = VERBOSE_OPTION,
) -> None:
    """Run a skill once and print its prediction. Requires a reachable router."""
    settings = get_settings()
    registry, _ = load_registry(settings)
    target = route(skill, registry).skill if use_route else registry.get(skill)

    inputs = _resolve_inputs(text, input_file)
    configure_lm(settings, model)
    module = build_module(target)
    with verbosity_scope(verbose=verbose, skill=target.name):
        data = prediction_to_dict(module(**inputs))

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
    raise AiagentError("provide --text or --input")
