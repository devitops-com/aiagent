"""``aiagent chat`` — a basic, resumable multi-turn Q&A prompt loop.

Each answer is produced with the prior turns as context, and the conversation is
persisted to a named session so it can be resumed later. A minor convenience
feature; aiagent is not a chat framework.
"""

from __future__ import annotations

from collections.abc import Callable

import typer

from aiagent.cli._common import get_settings
from aiagent.cli._runtime import configure_lm
from aiagent.cli.chat_session import ChatSession
from aiagent.cli.repl import PromptReader
from aiagent.skills.loader import build_module
from aiagent.skills.registry import load_registry

# REPL directives, offered by Tab completion and the fzf command palette.
_COMMANDS = (":quit", ":q", ":reset", ":history", ":help")


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

    reader = PromptReader(
        history_path=settings.sessions_dir / "history",
        commands=_COMMANDS,
        seed=[turn["text"] for turn in convo.turns],
    )
    hint = "type :quit to exit, :reset to clear, :history to search past prompts."
    typer.echo(f"aiagent chat — {hint}")
    if convo.turns:
        typer.echo(f"(resumed session {session!r}: {len(convo.turns)} turns)")

    prefill = ""
    try:
        while True:
            try:
                message = reader.read("you> ", prefill=prefill)
            except (EOFError, KeyboardInterrupt):
                typer.echo()
                break
            prefill = ""
            text = message.strip()
            if not text:
                continue
            if text in (":quit", ":q"):
                break
            if text == ":reset":
                convo.clear()
                typer.echo("(session cleared)")
                continue
            if text in (":history", ":h"):
                prefill = _pick(reader.pick_history, reader.fzf_enabled, "history")
                continue
            if text in (":help", ":?"):
                prefill = _pick(reader.pick_command, reader.fzf_enabled, "commands")
                continue
            reader.remember(text)
            history = dspy.History(messages=[dict(turn) for turn in convo.turns])
            answer = str(
                getattr(module(text=text, history=history), "answer", "")
            ).strip()
            typer.echo(f"bot> {answer}")
            convo.append(text, answer)
    finally:
        reader.save()


def _pick(picker: Callable[[], str | None], fzf_enabled: bool, what: str) -> str:
    """Run an fzf picker, returning the selection to pre-fill (empty on miss).

    Falls back to a hint when fzf is unavailable so the feature is discoverable
    without it (Ctrl-R still searches history via readline).
    """
    if not fzf_enabled:
        typer.echo(
            f"(fzf not found — install it to browse {what}; "
            "Ctrl-R searches history inline)"
        )
        return ""
    return picker() or ""
