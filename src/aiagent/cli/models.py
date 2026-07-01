"""``aiagent models list`` — show registry aliases and router-advertised models."""

from __future__ import annotations

import typer

from aiagent.cli._common import CLI_CONTEXT_SETTINGS, get_settings, print_json
from aiagent.llm.registry import get_registry, list_model_aliases

models_app = typer.Typer(
    name="models",
    help="Inspect configured model aliases and router-advertised models.",
    no_args_is_help=True,
    add_completion=False,
    context_settings=CLI_CONTEXT_SETTINGS,
)


@models_app.command("list")
def list_models(
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List alias -> model strings, and (online) what the router advertises."""
    settings = get_settings()
    registry = get_registry(settings.registry_overrides)
    aliases = list_model_aliases(registry, settings.default_reasoning)

    advertised: list[str] = []
    error: str | None = None
    import httpx  # local import keeps module load light

    try:
        with httpx.Client(timeout=settings.request_timeout_s) as client:
            resp = client.get(settings.models_url())
            if resp.status_code == 200:
                advertised = [
                    item.get("id", "") for item in resp.json().get("data", [])
                ]
            else:
                error = f"HTTP {resp.status_code}"
    except (httpx.HTTPError, ValueError) as exc:
        error = str(exc)

    if as_json:
        print_json(
            {
                "aliases": [{"alias": a, "model": m} for a, m in aliases],
                "advertised": advertised,
                "error": error,
            }
        )
        return

    typer.echo("Aliases:")
    for alias, model in aliases:
        typer.echo(f"  {alias:<12} -> {model}")
    typer.echo("\nRouter-advertised models:")
    if advertised:
        for mid in advertised:
            typer.echo(f"  {mid}")
    else:
        typer.echo(f"  (none — {error or 'router returned no models'})")
