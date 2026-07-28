"""``aiagent models list`` — show registry aliases and router-advertised models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from aiagent.cli._common import CLI_CONTEXT_SETTINGS, get_settings, print_json
from aiagent.llm.registry import get_registry, list_model_aliases

if TYPE_CHECKING:  # import-light: discovery pulls httpx, so only for typing
    from aiagent.llm.discovery import EndpointReport

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
    """List alias -> model strings, and (online) what each endpoint advertises."""
    settings = get_settings()
    registry = get_registry(settings.registry_overrides, settings.model)
    aliases = list_model_aliases(
        registry, settings.default_reasoning, settings.context_tokens
    )

    from aiagent.llm.discovery import probe  # local import keeps module load light

    reports = probe(settings.endpoints(), settings.request_timeout_s)
    primary = reports[0]

    if as_json:
        print_json(
            {
                "aliases": [
                    {
                        "alias": alias,
                        "model": model,
                        "api_base": registry[alias].api_base or settings.api_base,
                    }
                    for alias, model in aliases
                ],
                # `advertised` / `error` describe `api_base`, as they always
                # have; `endpoints` carries every probed endpoint.
                "advertised": list(primary.models),
                "error": primary.error,
                "endpoints": [report.as_dict() for report in reports],
            }
        )
        return

    typer.echo("Aliases:")
    for alias, model in aliases:
        endpoint = registry[alias].api_base
        suffix = f"  [{endpoint}]" if endpoint else ""
        typer.echo(f"  {alias:<12} -> {model}{suffix}")

    typer.echo("\nRouter-advertised models:")
    if len(reports) == 1:
        _echo_models(primary, indent="  ")
        return
    for report in reports:
        typer.echo(f"  {report.endpoint}")
        _echo_models(report, indent="    ")


def _echo_models(report: EndpointReport, indent: str) -> None:
    if report.models:
        for mid in report.models:
            typer.echo(f"{indent}{mid}")
        return
    typer.echo(f"{indent}(none — {report.error or 'router returned no models'})")
