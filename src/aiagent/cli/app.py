"""aiagent root CLI — Typer dispatcher.

Every top-level command is a native Typer command. ``main()`` wraps Typer's
invocation so that *usage* errors (unknown command, missing option, bad flag)
print the context-appropriate help text rather than a one-line error, while
*runtime* errors (``AiagentError`` raised by a handler) print a clean message
and keep their exit code.

Commands are registered incrementally as the layers land; only import-light
commands are wired here so ``--help`` never pays the cost of importing ``dspy``.
"""

from __future__ import annotations

import sys

import click
import typer

from aiagent import __version__
from aiagent.cli._common import CLI_CONTEXT_SETTINGS
from aiagent.cli.chat import chat as chat_cmd
from aiagent.cli.config_cmd import config_app
from aiagent.cli.doctor import doctor as doctor_cmd
from aiagent.cli.eval_cmd import eval_skill
from aiagent.cli.models import models_app
from aiagent.cli.optimize_cmd import optimize_skill
from aiagent.cli.run import run as run_cmd
from aiagent.cli.sentiment import sentiment as sentiment_cmd
from aiagent.cli.shell import shell as shell_cmd
from aiagent.cli.skills_cmd import skills_app
from aiagent.exceptions import AiagentError

app = typer.Typer(
    name="aiagent",
    help=(
        "aiagent — a programmatic DSPy agent: optimize prompts, run goal-reaching "
        "loops, and process data autonomously over local LLMs."
    ),
    no_args_is_help=True,
    add_completion=False,
    context_settings=CLI_CONTEXT_SETTINGS,
)


@app.callback()
def _root() -> None:
    """aiagent — programmatic DSPy agent for optimization and data processing."""
    # Presence of a callback forces Typer to keep sub-command structure even
    # when only one command is registered.


@app.command()
def version() -> None:
    """Print the aiagent version."""
    click.echo(__version__)


# Connectivity / inspection commands (import-light; no dspy).
app.command("doctor", context_settings=CLI_CONTEXT_SETTINGS)(doctor_cmd)
app.add_typer(models_app)
app.add_typer(config_app)
app.add_typer(skills_app)

# Runtime commands. These touch DSPy, but import it lazily inside their handlers,
# so registering them here keeps `import aiagent.cli.app` dspy-free.
app.command("run", context_settings=CLI_CONTEXT_SETTINGS)(run_cmd)
app.command("sentiment", context_settings=CLI_CONTEXT_SETTINGS)(sentiment_cmd)
app.command("eval", context_settings=CLI_CONTEXT_SETTINGS)(eval_skill)
app.command("optimize", context_settings=CLI_CONTEXT_SETTINGS)(optimize_skill)
app.command("chat", context_settings=CLI_CONTEXT_SETTINGS)(chat_cmd)
app.command("shell", context_settings=CLI_CONTEXT_SETTINGS)(shell_cmd)


# Typer (>= ~0.16) vendors its own Click, so usage errors raised under
# ``standalone_mode=False`` are not always subclasses of the stdlib ``click``
# exceptions. Catch both the stdlib and (when present) the vendored variants.
_USAGE_ERRORS: tuple[type[BaseException], ...] = (click.exceptions.UsageError,)
_ABORT_ERRORS: tuple[type[BaseException], ...] = (click.exceptions.Abort,)
try:  # pragma: no cover - exercised only on Typer builds that vendor Click
    from typer._click.exceptions import Abort as _TyperAbort
    from typer._click.exceptions import UsageError as _TyperUsageError

    _USAGE_ERRORS = (*_USAGE_ERRORS, _TyperUsageError)
    _ABORT_ERRORS = (*_ABORT_ERRORS, _TyperAbort)
except ImportError:  # pragma: no cover - older Typer uses stdlib Click
    pass


def _print_help_for(exc: BaseException) -> None:
    """Print help for the deepest context Click reached for a usage error."""
    if type(exc).__name__ == "NoArgsIsHelpError":
        # Typer already emitted the help text before raising.
        return
    ctx = getattr(exc, "ctx", None)
    if ctx is None:
        root_cmd = typer.main.get_command(app)
        ctx = root_cmd.make_context("aiagent", [], resilient_parsing=True)
    click.echo(ctx.get_help())


def main() -> None:
    """aiagent CLI entry point.

    Usage errors produce help text; ``AiagentError`` handlers produce a clean
    stderr message with exit code 1; aborts exit 1. Click with
    ``standalone_mode=False`` returns the handler's exit code, which we re-raise
    so runtime exit codes propagate to the shell.
    """
    try:
        rv = app(standalone_mode=False, prog_name="aiagent")
    except _USAGE_ERRORS as exc:
        _print_help_for(exc)
        sys.exit(2)
    except _ABORT_ERRORS:
        click.echo("Aborted.", err=True)
        sys.exit(1)
    except AiagentError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    if isinstance(rv, int) and rv != 0:
        sys.exit(rv)


if __name__ == "__main__":
    main()
