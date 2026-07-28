"""Probe OpenAI-compatible endpoints for the models they advertise.

Shared by ``aiagent models list`` and ``aiagent doctor``, which each probe every
endpoint in ``settings.endpoints()`` (issue #11). Pure ``httpx`` — no ``dspy``,
so both commands stay fast; the CLI still imports this module lazily, inside the
command body, to keep ``--help`` off the ``httpx`` import path.

One unreachable endpoint does not sink the run: the failure is recorded on that
endpoint's report and the rest are still probed, so a partially-up stack lists
what it can.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from aiagent.config import health_url_for, models_url_for


@dataclass(frozen=True)
class EndpointReport:
    """What one endpoint answered, or why it could not."""

    endpoint: str
    models: tuple[str, ...] = ()
    error: str | None = None  # why /models yielded nothing (transport or HTTP)
    reachable: bool = True  # False only when the transport itself failed
    health_status: int | None = None  # None when /health was not probed
    models_status: int | None = None

    @property
    def status(self) -> str:
        """``ok`` / ``degraded`` / ``unreachable`` for this endpoint alone.

        ``degraded`` means the endpoint answered but one of its probes did not
        return 200 — reachable, yet not serving properly.
        """
        if not self.reachable:
            return "unreachable"
        if self.health_status is not None and self.health_status != 200:
            return "degraded"
        if self.models_status is not None and self.models_status != 200:
            return "degraded"
        return "ok"

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view for ``--json`` output."""
        out: dict[str, Any] = {
            "endpoint": self.endpoint,
            "models": list(self.models),
            "error": self.error,
            "status": self.status,
            "models_endpoint": {
                "url": models_url_for(self.endpoint),
                "status": self.models_status,
            },
        }
        if self.health_status is not None or not self.reachable:
            out["health"] = {
                "url": health_url_for(self.endpoint),
                "status": self.health_status,
            }
        return out


def probe(
    endpoints: list[str], timeout: float, *, health: bool = False
) -> list[EndpointReport]:
    """Query ``/models`` (and optionally ``/health``) on each endpoint, in order."""
    with httpx.Client(timeout=timeout) as client:
        return [_probe_one(client, endpoint, health) for endpoint in endpoints]


def _probe_one(client: httpx.Client, endpoint: str, health: bool) -> EndpointReport:
    health_status: int | None = None
    try:
        if health:
            health_status = client.get(health_url_for(endpoint)).status_code
        resp = client.get(models_url_for(endpoint))
        if resp.status_code != 200:
            return EndpointReport(
                endpoint,
                error=f"HTTP {resp.status_code}",
                health_status=health_status,
                models_status=resp.status_code,
            )
        models = tuple(item.get("id", "") for item in resp.json().get("data", []))
    except (httpx.HTTPError, ValueError) as exc:
        return EndpointReport(
            endpoint, error=str(exc), reachable=False, health_status=health_status
        )
    return EndpointReport(
        endpoint,
        models=models,
        health_status=health_status,
        models_status=resp.status_code,
    )
