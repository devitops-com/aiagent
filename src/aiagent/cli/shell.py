"""``aiagent shell`` — the devai picker entrypoint.

The devai model-picker exports ``AIAGENT_*`` env (router URL + chosen model) and
then runs ``aiagent shell``. This prints a short banner and execs an interactive
shell so the user can run ``aiagent run|optimize|eval|chat`` with the env already
wired. Imports no ``dspy``.
"""

from __future__ import annotations

import os
import shutil

import typer

from aiagent.cli._common import examples, get_settings

SHELL_EXAMPLES = examples(
    ("A banner with the router and model, then your $SHELL", "aiagent shell"),
    (
        "Pin the model for everything run in that shell",
        "aiagent shell --model qwen3.5:9b-q8_0",
    ),
)


def shell(
    model: str | None = typer.Option(
        None, "--model", help="Pin the model (AIAGENT_MODEL in the new shell)."
    ),
) -> None:
    """Open a shell pre-configured for this devai agent session.

    Prints the router, the model and a few commands to try, then replaces itself
    with $SHELL (else bash, else /bin/sh). devai's model picker runs it when you
    pick aiagent.
    """
    settings = get_settings()
    if model:
        os.environ["AIAGENT_MODEL"] = model
    effective = (model or settings.model) or f"(alias: {settings.default_alias})"

    typer.echo("aiagent — devai agent shell")
    typer.echo(f"  router : {settings.api_base}")
    typer.echo(f"  model  : {effective}")
    typer.echo("  try    : aiagent doctor")
    typer.echo("           aiagent run extract --text '<note>'")
    typer.echo("           aiagent eval extract")
    typer.echo("           aiagent optimize extract --out compiled/extract.json")
    typer.echo("           aiagent chat")

    sh = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
    os.execvp(sh, [sh])  # nosec B606 - sh is from $SHELL/PATH, never user input
