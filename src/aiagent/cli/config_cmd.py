"""``aiagent config show`` — print resolved settings (api_key redacted)."""

from __future__ import annotations

import typer

from aiagent.cli._common import CLI_CONTEXT_SETTINGS, get_settings, print_json

config_app = typer.Typer(
    name="config",
    help="Inspect resolved configuration.",
    no_args_is_help=True,
    add_completion=False,
    context_settings=CLI_CONTEXT_SETTINGS,
)


@config_app.command("show")
def show(
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Show resolved settings (env > TOML > devai-env > defaults), api_key masked."""
    data = get_settings().redacted()
    if as_json:
        print_json(data)
        return
    for key in sorted(data):
        typer.echo(f"{key:<22} = {data[key]}")
