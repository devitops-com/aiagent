"""Tests for the network-probing commands (doctor, models) with httpx mocked."""

from __future__ import annotations

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
