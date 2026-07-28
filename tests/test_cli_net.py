"""Tests for the network-probing commands (doctor, models) with httpx mocked."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from aiagent.cli.app import app

runner = CliRunner()


class _Resp:
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def _client_factory(handler: Any) -> type:
    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def get(self, url: str) -> _Resp:
            return handler(url)

    return _Client


def _ok(url: str) -> _Resp:
    if url.endswith("/health"):
        return _Resp(200, {"status": "ok"})
    return _Resp(200, {"data": [{"id": "qwen3.5:9b-q8_0"}]})


def test_doctor_offline() -> None:
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == 0
    assert "offline" in result.stdout.lower()


def test_doctor_online_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _client_factory(_ok))
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert "qwen3.5:9b-q8_0" in result.stdout


def test_doctor_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_url: str) -> _Resp:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "Client", _client_factory(boom))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "unreachable" in result.stdout


def test_models_list_online(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _client_factory(_ok))
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "default" in result.stdout  # alias
    assert "qwen3.5:9b-q8_0" in result.stdout  # advertised


def test_models_list_json_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_url: str) -> _Resp:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "Client", _client_factory(boom))
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0
    assert '"advertised"' in result.stdout


# --- multi-endpoint (issue #11) ---------------------------------------------

_SECOND = "http://devai-router:11435/v1"
_THIRD = "http://devai-router:11436/v1"


def _per_port(url: str) -> _Resp:
    """Each port serves its own model list; :11436 is down."""
    if ":11436" in url:
        raise httpx.ConnectError("refused")
    if url.endswith("/health"):
        return _Resp(200, {"status": "ok"})
    model = "Qwen3.5-9B-NVFP4" if ":11435" in url else "qwen3.5:9b-q8_0"
    return _Resp(200, {"data": [{"id": model}]})


@pytest.fixture
def three_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIAGENT_DISCOVER_ENDPOINTS", f"{_SECOND},{_THIRD}")
    monkeypatch.setattr(httpx, "Client", _client_factory(_per_port))


def test_models_list_groups_by_endpoint(three_endpoints: None) -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "qwen3.5:9b-q8_0" in result.stdout  # :11434
    assert "Qwen3.5-9B-NVFP4" in result.stdout  # :11435
    assert _THIRD in result.stdout  # listed even though it is down
    assert "refused" in result.stdout


def test_models_list_json_reports_every_endpoint(three_endpoints: None) -> None:
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    endpoints = {e["endpoint"]: e for e in payload["endpoints"]}
    assert len(endpoints) == 3
    assert endpoints[_SECOND]["models"] == ["Qwen3.5-9B-NVFP4"]
    assert endpoints[_THIRD]["status"] == "unreachable"
    # The single-endpoint keys still describe api_base.
    assert payload["advertised"] == ["qwen3.5:9b-q8_0"]


def test_models_list_shows_per_alias_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AIAGENT_REGISTRY_OVERRIDES",
        json.dumps({"vllm": {"model": "Qwen3.5-9B-NVFP4", "api_base": _SECOND}}),
    )
    monkeypatch.setattr(httpx, "Client", _client_factory(_ok))
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert _SECOND in result.stdout


def test_doctor_reports_each_endpoint(three_endpoints: None) -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    # One endpoint down is a degraded report, not a clean bill of health.
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "degraded"
    statuses = {e["endpoint"]: e["status"] for e in payload["endpoints"]}
    assert statuses == {
        "http://devai-router:11434/v1": "ok",
        _SECOND: "ok",
        _THIRD: "unreachable",
    }
    # api_base is still described by the top-level keys.
    assert payload["models"] == ["qwen3.5:9b-q8_0"]


def test_doctor_lists_endpoints_in_text_output(three_endpoints: None) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "endpoints:" in result.stdout
    assert _THIRD in result.stdout


def test_doctor_degraded_when_health_is_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def unhealthy(url: str) -> _Resp:
        if url.endswith("/health"):
            return _Resp(503)
        return _Resp(200, {"data": [{"id": "qwen3.5:9b-q8_0"}]})

    monkeypatch.setattr(httpx, "Client", _client_factory(unhealthy))
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "degraded"


def test_doctor_degraded_when_models_endpoint_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A healthy root but a failing /models is not a clean bill of health.
    def bad_models(url: str) -> _Resp:
        return _Resp(200, {"status": "ok"}) if url.endswith("/health") else _Resp(502)

    monkeypatch.setattr(httpx, "Client", _client_factory(bad_models))
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "degraded"


def test_models_list_reports_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _client_factory(lambda _url: _Resp(502)))
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "HTTP 502" in result.stdout


def test_bad_registry_override_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # A mistyped key used to be dropped without a word (issue #11).
    monkeypatch.setenv(
        "AIAGENT_REGISTRY_OVERRIDES",
        json.dumps({"typo": {"model": "m", "api_bse": _SECOND}}),
    )
    monkeypatch.setattr(httpx, "Client", _client_factory(_ok))
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 1
    assert "registry_overrides.typo" in str(result.exception)
    assert "api_bse" in str(result.exception)
