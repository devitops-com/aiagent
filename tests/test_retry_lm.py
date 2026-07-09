"""Tests for the status-aware LM retry policy (issue #10)."""

from __future__ import annotations

import asyncio

import litellm
import pytest

from aiagent.config import load_settings
from aiagent.llm.lm import build_lm
from aiagent.llm.retry_lm import RetryAwareLM, is_retryable, run_with_retry


def _err(cls: type[Exception]) -> Exception:
    """Construct a litellm HTTP error (works for BadRequestError/InternalServerError)."""
    return cls("boom", model="m", llm_provider="openai")  # type: ignore[call-arg]


class _StatusError(Exception):
    """A minimal exception carrying an HTTP ``status_code``, like litellm's."""

    def __init__(self, status: int) -> None:
        super().__init__(f"status {status}")
        self.status_code = status


# --- is_retryable ---------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status: int) -> None:
    assert is_retryable(_StatusError(status)) is False


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_server_and_ratelimit_errors_are_retryable(status: int) -> None:
    assert is_retryable(_StatusError(status)) is True


@pytest.mark.parametrize("cls", [litellm.Timeout, litellm.APIConnectionError])
def test_connection_and_timeout_types_are_retryable(cls: type[Exception]) -> None:
    # These are matched by type, not status; build without invoking __init__ so
    # the test doesn't depend on each class's constructor signature.
    assert is_retryable(cls.__new__(cls)) is True


def test_unknown_exception_is_not_retryable() -> None:
    assert is_retryable(ValueError("nope")) is False


# --- run_with_retry -------------------------------------------------------


def test_succeeds_without_retry() -> None:
    calls = 0

    def call() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert run_with_retry(call, max_retries=2, sleep=lambda _s: None) == "ok"
    assert calls == 1


def test_non_retryable_fails_immediately() -> None:
    calls = 0
    sleeps: list[float] = []

    def call() -> str:
        nonlocal calls
        calls += 1
        raise _StatusError(400)

    with pytest.raises(_StatusError):
        run_with_retry(call, max_retries=5, sleep=sleeps.append)
    assert calls == 1  # tried once, no retries
    assert sleeps == []


def test_transient_then_success() -> None:
    calls = 0
    sleeps: list[float] = []

    def call() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _StatusError(500)
        return "recovered"

    assert run_with_retry(call, max_retries=3, sleep=sleeps.append) == "recovered"
    assert calls == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff before each retry


def test_transient_exhausts_retries_then_raises() -> None:
    calls = 0

    def call() -> str:
        nonlocal calls
        calls += 1
        raise _StatusError(503)

    with pytest.raises(_StatusError):
        run_with_retry(call, max_retries=2, sleep=lambda _s: None)
    assert calls == 3  # initial + 2 retries


# --- RetryAwareLM ---------------------------------------------------------


def test_build_lm_returns_retry_aware_with_litellm_retries_disabled() -> None:
    lm = build_lm(settings=load_settings())
    assert isinstance(lm, RetryAwareLM)
    assert lm._max_retries == 2  # from settings.num_retries
    assert lm.num_retries == 0  # litellm's own blind retry is off


def test_forward_does_not_retry_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_forward(_self: object, *a: object, **k: object) -> str:
        nonlocal calls
        calls += 1
        raise _err(litellm.BadRequestError)

    monkeypatch.setattr("dspy.LM.forward", fake_forward)
    lm = build_lm(settings=load_settings())  # max_retries=2
    with pytest.raises(litellm.BadRequestError):
        lm.forward(messages=[{"role": "user", "content": "x"}])
    assert calls == 1  # the 400 is surfaced once, not hammered


def test_forward_retries_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_forward(_self: object, *a: object, **k: object) -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise _err(litellm.InternalServerError)
        return "ok"

    monkeypatch.setattr("dspy.LM.forward", fake_forward)
    monkeypatch.setattr("aiagent.llm.retry_lm.time.sleep", lambda _s: None)
    lm = build_lm(settings=load_settings())
    assert lm.forward(messages=[{"role": "user", "content": "x"}]) == "ok"
    assert calls == 2


def test_aforward_retries_transient_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_aforward(_self: object, *a: object, **k: object) -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise _err(litellm.InternalServerError)
        return "ok"

    async def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("dspy.LM.aforward", fake_aforward)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    lm = build_lm(settings=load_settings())
    result = asyncio.run(lm.aforward(messages=[{"role": "user", "content": "x"}]))
    assert result == "ok"
    assert calls == 2


def test_aforward_does_not_retry_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_aforward(_self: object, *a: object, **k: object) -> str:
        nonlocal calls
        calls += 1
        raise _err(litellm.BadRequestError)

    monkeypatch.setattr("dspy.LM.aforward", fake_aforward)
    lm = build_lm(settings=load_settings())
    with pytest.raises(litellm.BadRequestError):
        asyncio.run(lm.aforward(messages=[{"role": "user", "content": "x"}]))
    assert calls == 1
