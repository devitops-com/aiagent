"""``aiagent eval <skill>`` — score a skill over its dev set."""

from __future__ import annotations

from pathlib import Path

import typer

from aiagent.cli._common import get_settings, print_json
from aiagent.cli._runtime import configure_lm
from aiagent.cli._verbosity import VERBOSE_OPTION, verbosity_scope
from aiagent.exceptions import AiagentError
from aiagent.skills.loader import build_metric, build_module, dataset_path
from aiagent.skills.registry import load_registry


def eval_skill(
    skill: str = typer.Argument(..., help="Skill name."),
    devset: Path | None = typer.Option(None, "--devset", help="Dev JSONL override."),
    compiled: Path | None = typer.Option(
        None, "--compiled", help="Compiled program JSON to load before evaluating."
    ),
    model: str | None = typer.Option(None, "--model", help="Model override."),
    num_threads: int | None = typer.Option(None, "--num-threads"),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
    verbose: int = VERBOSE_OPTION,
) -> None:
    """Evaluate a skill over its dev set. Requires a reachable router."""
    from aiagent.core.evaluate import evaluate
    from aiagent.data.loader import load_expense_set
    from aiagent.optimize.harness import load_compiled

    settings = get_settings()
    registry, _ = load_registry(settings)
    sk = registry.get(skill)
    dev_path = devset or dataset_path(sk, "devset")
    if dev_path is None:
        raise AiagentError(f"skill {skill!r} has no devset; pass --devset")

    configure_lm(settings, model)
    module = build_module(sk)
    if compiled is not None:
        load_compiled(module, compiled)
    metric = build_metric(sk)
    examples = load_expense_set(dev_path)
    threads = num_threads if num_threads is not None else settings.num_threads
    with verbosity_scope(verbose=verbose, skill=skill):
        result = evaluate(module, examples, metric, num_threads=threads)

    if as_json:
        print_json({"skill": skill, "score": result.score, "n": result.n})
    else:
        typer.echo(f"skill : {skill}")
        typer.echo(f"n     : {result.n}")
        typer.echo(f"score : {result.score:.2f}")
