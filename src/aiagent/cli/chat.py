"""``aiagent chat`` — a basic, resumable multi-turn Q&A prompt loop.

Each answer is produced with the prior turns as context, and the conversation is
persisted to a named session so it can be resumed later. A minor convenience
feature; aiagent is not a chat framework.
"""

from __future__ import annotations

import typer

from aiagent.cli._common import get_settings
from aiagent.cli._runtime import configure_lm
from aiagent.cli.chat_session import ChatSession
from aiagent.skills.loader import build_module
from aiagent.skills.registry import load_registry


def chat(
    session: str = typer.Option(
        "default", "--session", "-s", help="Session to resume or start."
    ),
    new: bool = typer.Option(
        False, "--new", help="Start fresh, discarding the session's history."
    ),
    skill: str = typer.Option("chat", "--skill", help="Chat skill name."),
    model: str | None = typer.Option(None, "--model", help="Model override."),
) -> None:
    """Ask questions and get answers, with resumable multi-turn history.

    Requires a reachable router.
    """
    import dspy

    settings = get_settings()
    registry, _ = load_registry(settings)
    sk = registry.get(skill)
    configure_lm(settings, model)
    module = build_module(sk)

    convo = ChatSession.load(settings, session)
    if new:
        convo.clear()

    typer.echo("aiagent chat — type :quit to exit, :reset to clear this session.")
    if convo.turns:
        typer.echo(f"(resumed session {session!r}: {len(convo.turns)} turns)")

    while True:
        try:
            message = typer.prompt("you", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break
        text = message.strip()
        if not text:
            continue
        if text in (":quit", ":q"):
            break
        if text == ":reset":
            convo.clear()
            typer.echo("(session cleared)")
            continue
        history = dspy.History(messages=[dict(turn) for turn in convo.turns])
        answer = str(getattr(module(text=text, history=history), "answer", "")).strip()
        typer.echo(f"bot> {answer}")
        convo.append(text, answer)
