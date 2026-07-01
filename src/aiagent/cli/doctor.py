"""``aiagent doctor`` — verify connectivity to the devai router.

Online: probes ``GET /health`` and ``GET /v1/models``. Offline (``--offline``):
config sanity only (no network), for use in build/CI environments with no router.
Imports no ``dspy`` — stays fast.
"""

from __future__ import annotations

import typer

from aiagent.cli._common import get_settings, print_json
from aiagent.exceptions import AiagentConfigError

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
    """Check configuration and (unless --offline) reach the devai router."""
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
    import httpx  # local import keeps module load light

    health_ok = False
    models: list[str] = []
    error: str | None = None
    try:
        with httpx.Client(timeout=probe_timeout) as client:
            h = client.get(settings.health_url())
            health_ok = h.status_code == 200
            report["health"] = {"url": settings.health_url(), "status": h.status_code}
            m = client.get(settings.models_url())
            report["models_endpoint"] = {
                "url": settings.models_url(),
                "status": m.status_code,
            }
            if m.status_code == 200:
                data = m.json()
                models = [item.get("id", "") for item in data.get("data", [])]
    except (httpx.HTTPError, ValueError) as exc:
        error = str(exc)

    report["models"] = models
    if error is not None:
        report["status"] = "unreachable"
        report["error"] = error
        report["hint"] = _COLD_START_HINT
        _emit(report, as_json)
        raise typer.Exit(_EXIT_UNREACHABLE)

    report["status"] = "ok" if health_ok else "degraded"
    report["hint"] = _COLD_START_HINT
    _emit(report, as_json)
    if not health_ok:
        raise typer.Exit(_EXIT_UNREACHABLE)


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
    if report.get("hint") and report.get("status") != "ok":
        typer.echo(f"hint     : {report['hint']}")
