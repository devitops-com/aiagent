"""``aiagent chat`` — a basic multi-turn REPL (minor convenience feature)."""

from __future__ import annotations

import typer

from aiagent.cli._common import get_settings
from aiagent.cli._runtime import configure_lm
from aiagent.skills.loader import build_module
from aiagent.skills.registry import load_registry


def chat(
    skill: str = typer.Option("chat", "--skill", help="Chat skill name."),
    model: str | None = typer.Option(None, "--model", help="Model override."),
) -> None:
    """Interactive chat against the configured model. Requires a reachable router."""
    import dspy

    settings = get_settings()
    registry, _ = load_registry(settings)
    sk = registry.get(skill)
    configure_lm(settings, model)
    module = build_module(sk)

    typer.echo("aiagent chat — type :quit to exit, :reset to clear history.")
    messages: list[dict[str, str]] = []
    while True:
        try:
            question = typer.prompt("you", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break
        q = question.strip()
        if not q:
            continue
        if q in (":quit", ":q"):
            break
        if q == ":reset":
            messages = []
            typer.echo("(history cleared)")
            continue
        history = dspy.History(messages=list(messages))
        answer = str(getattr(module(history=history, question=q), "answer", "")).strip()
        typer.echo(f"bot> {answer}")
        messages.append({"question": q, "answer": answer})
