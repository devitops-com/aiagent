"""``aiagent optimize <skill>`` — compile a skill and (optionally) save it."""

from __future__ import annotations

from pathlib import Path

import typer

from aiagent.cli._common import examples, get_settings
from aiagent.cli._runtime import configure_lm
from aiagent.cli._verbosity import VERBOSE_OPTION, verbosity_scope
from aiagent.exceptions import AiagentError
from aiagent.skills.loader import build_metric, build_module, dataset_path
from aiagent.skills.registry import load_registry

OPTIMIZE_EXAMPLES = examples(
    (
        "BootstrapFewShot on the bundled sets; save it for eval --compiled",
        "aiagent optimize extract --out compiled/extract.json",
    ),
    (
        "MIPROv2 instead: instructions and demos, many more LLM calls",
        "aiagent optimize extract --optimizer mipro --out compiled/extract.json",
    ),
    (
        "Your own train and dev sets",
        "aiagent optimize extract --trainset my_train.jsonl --devset my_dev.jsonl",
    ),
)


def optimize_skill(
    skill: str = typer.Argument(..., help="Skill name."),
    trainset: Path | None = typer.Option(
        None, "--trainset", help="Train JSONL (default: the skill's own trainset)."
    ),
    devset: Path | None = typer.Option(
        None,
        "--devset",
        help="Dev JSONL for the before/after scores (default: the skill's own).",
    ),
    optimizer: str = typer.Option(
        "bootstrap",
        "--optimizer",
        help="bootstrap (BootstrapFewShot) or mipro (MIPROv2).",
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Save the compiled program (state-only JSON) here."
    ),
    model: str | None = typer.Option(None, "--model", help="Model override."),
    num_threads: int | None = typer.Option(
        None, "--num-threads", help="Parallel threads (default: num_threads)."
    ),
    verbose: int = VERBOSE_OPTION,
) -> None:
    """Optimize a skill against its metric. Requires a reachable router.

    Compiles the skill with a DSPy optimizer on the train set. With a dev set it
    prints the baseline and after scores and the lift; --out saves the program
    as state-only JSON. Rows are JSONL objects in extract's shape: text,
    merchant, date and amount. Demo and round counts come from the settings.
    """
    from aiagent.data.loader import load_expense_set
    from aiagent.optimize.harness import optimize

    settings = get_settings()
    registry, _ = load_registry(settings)
    sk = registry.get(skill)
    train_path = trainset or dataset_path(sk, "trainset")
    if train_path is None:
        raise AiagentError(f"skill {skill!r} has no trainset; pass --trainset")
    dev_path = devset or dataset_path(sk, "devset")

    configure_lm(settings, model)
    module = build_module(sk)
    metric = build_metric(sk)
    train = load_expense_set(train_path)
    dev = load_expense_set(dev_path) if dev_path is not None else None
    threads = num_threads if num_threads is not None else settings.num_threads

    with verbosity_scope(verbose=verbose, skill=skill):
        result = optimize(
            module,
            train,
            metric,
            method=optimizer,
            devset=dev,
            save_path=out,
            num_threads=threads,
            max_bootstrapped_demos=settings.max_bootstrapped_demos,
            max_labeled_demos=settings.max_labeled_demos,
            max_rounds=settings.max_rounds,
        )

    typer.echo(f"optimizer : {optimizer}")
    if result.baseline_score is not None and result.after_score is not None:
        lift = result.after_score - result.baseline_score
        typer.echo(f"baseline  : {result.baseline_score:.2f}")
        typer.echo(f"after     : {result.after_score:.2f}")
        typer.echo(f"lift      : {lift:+.2f}")
    if result.save_path:
        typer.echo(f"saved     : {result.save_path}")
