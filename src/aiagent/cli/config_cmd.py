"""``aiagent config show`` — print resolved settings (api_key redacted)."""

from __future__ import annotations

import typer

from aiagent.cli._common import CLI_CONTEXT_SETTINGS, examples, get_settings, print_json

_SHOW_EXAMPLES = examples(
    ("Every resolved setting, one per line", "aiagent config show"),
    (
        "One setting in a script, e.g. the artifacts directory",
        "aiagent config show --json | jq -r .artifacts_dir",
    ),
    (
        "See what an AIAGENT_* variable resolves to",
        "AIAGENT_REQUEST_TIMEOUT_S=1800 aiagent config show",
    ),
)

config_app = typer.Typer(
    name="config",
    help="Inspect resolved configuration.",
    epilog=_SHOW_EXAMPLES,
    no_args_is_help=True,
    add_completion=False,
    context_settings=CLI_CONTEXT_SETTINGS,
)


@config_app.command("show", epilog=_SHOW_EXAMPLES)
def show(
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Show resolved settings (env > TOML > devai-env > defaults), api_key masked.

    Highest first: AIAGENT_* variables, ~/.config/aiagent/config.toml, the
    variables devai exports (OPENAI_BASE_URL, OPENAI_MODEL, ...), defaults.
    """
    data = get_settings().redacted()
    if as_json:
        print_json(data)
        return
    for key in sorted(data):
        typer.echo(f"{key:<22} = {data[key]}")
