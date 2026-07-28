"""``aiagent doctor`` — verify connectivity to the devai router.

Online: probes ``GET /health`` and ``GET /v1/models`` on every configured
endpoint (``api_base`` plus any ``discover_endpoints``). Offline (``--offline``):
config sanity only (no network), for use in build/CI environments with no router.
Imports no ``dspy`` — stays fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from aiagent.cli._common import get_settings, print_json
from aiagent.exceptions import AiagentConfigError

if TYPE_CHECKING:  # import-light: discovery pulls httpx, so only for typing
    from aiagent.llm.discovery import EndpointReport

# Exit codes: 0 healthy, 1 unreachable, 2 config error.
_EXIT_UNREACHABLE = 1
_EXIT_CONFIG = 2

_COLD_START_HINT = (
    "devai's vLLM/SGLang backends (ports 11435/11436) are recreated on demand; "
    "the first request to a cold backend can take many minutes. Raise the timeout "
    "with AIAGENT_REQUEST_TIMEOUT (seconds) if a call appears to hang."
)


def doctor(
    offline: bool = typer.Option(
        False, "--offline", "-O", help="Skip all network; check config only."
    ),
    timeout: float | None = typer.Option(
        None, "--timeout", help="Per-probe timeout in seconds (default: configured)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Check configuration and (unless --offline) reach every configured endpoint."""
    try:
        settings = get_settings()
    except AiagentConfigError as exc:
        typer.echo(f"config: ERROR — {exc}", err=True)
        raise typer.Exit(_EXIT_CONFIG) from exc

    report: dict[str, object] = {
        "api_base": settings.api_base,
        "model": settings.model or f"(default alias: {settings.default_alias})",
        "offline": offline,
    }

    if offline:
        report["status"] = "ok (offline: config valid)"
        _emit(report, as_json)
        return

    probe_timeout = timeout if timeout is not None else settings.request_timeout_s

    from aiagent.llm.discovery import probe  # local import keeps module load light

    reports = probe(settings.endpoints(), probe_timeout, health=True)
    primary = reports[0]

    # The top-level keys describe `api_base` (always the first endpoint), as they
    # always have; `endpoints` carries the per-endpoint detail.
    primary_dict = primary.as_dict()
    report["health"] = primary_dict["health"]
    report["models_endpoint"] = primary_dict["models_endpoint"]
    report["models"] = list(primary.models)
    report["endpoints"] = [r.as_dict() for r in reports]
    report["status"] = _overall_status(reports)
    if primary.error is not None:
        report["error"] = primary.error
    report["hint"] = _COLD_START_HINT

    _emit(report, as_json)
    if report["status"] != "ok":
        raise typer.Exit(_EXIT_UNREACHABLE)


def _overall_status(reports: list[EndpointReport]) -> str:
    """``ok`` only when every endpoint is; ``unreachable`` when none answered."""
    statuses = {r.status for r in reports}
    if statuses == {"ok"}:
        return "ok"
    if statuses == {"unreachable"}:
        return "unreachable"
    return "degraded"


def _emit(report: dict[str, object], as_json: bool) -> None:
    if as_json:
        print_json(report)
        return
    typer.echo(f"api_base : {report['api_base']}")
    typer.echo(f"model    : {report['model']}")
    typer.echo(f"status   : {report['status']}")
    if report.get("models"):
        models = report["models"]
        assert isinstance(models, list)
        typer.echo(f"models   : {', '.join(models) if models else '(none advertised)'}")
    if report.get("error"):
        typer.echo(f"error    : {report['error']}")
    endpoints = report.get("endpoints")
    if isinstance(endpoints, list) and len(endpoints) > 1:
        typer.echo("endpoints:")
        for entry in endpoints:
            assert isinstance(entry, dict)
            detail = ", ".join(entry["models"]) or entry["error"] or "(none advertised)"
            typer.echo(f"  {entry['endpoint']:<40} {entry['status']:<12} {detail}")
    if report.get("hint") and report.get("status") != "ok":
        typer.echo(f"hint     : {report['hint']}")
