"""Status-aware retry wrapper around ``dspy.LM`` (issue #10).

DSPy hands ``num_retries`` straight to ``litellm.completion(..., retry_strategy=
"exponential_backoff_retry")``, whose retry loop is **not** status-aware: it
retries *any* exception, so a permanent client error (HTTP 400/404/422) is
re-issued ``num_retries`` times — a latency storm that, against a model router,
repeatedly reloads a backend that will always fail.

This subclass disables litellm's blind retry (``num_retries=0`` on the parent)
and applies our own retry only for genuinely transient failures — connection
errors, timeouts, and 5xx/429 — while surfacing 4xx client errors immediately,
matching how Codex/Claude Code/OpenCode behave. Cold-start resilience (the
generous ``request_timeout_s`` plus retries on transient errors) is preserved.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import dspy
import litellm

# 408 request timeout, 409 conflict, 429 rate limit — plus any 5xx — are worth a
# retry. Every other 4xx is a permanent client error and must fail fast (this
# also covers the reported ``x-should-retry: false`` 400 case by status alone).
_RETRYABLE_STATUS = frozenset({408, 409, 429})
_BACKOFF_BASE_S = 1.0
_BACKOFF_MAX_S = 10.0


def is_retryable(exc: BaseException) -> bool:
    """True only for transient failures (connection / timeout / 429 / 5xx)."""
    if isinstance(exc, (litellm.Timeout, litellm.APIConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS or status >= 500
    return False


def _backoff(attempt: int) -> float:
    """Exponential backoff (1s, 2s, 4s, …) capped at 10s."""
    return min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * 2.0 ** (attempt - 1))


def run_with_retry[T](
    call: Callable[[], T],
    *,
    max_retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Invoke ``call``; retry only transient failures up to ``max_retries`` times."""
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:
            if attempt >= max_retries or not is_retryable(exc):
                raise
            attempt += 1
            sleep(_backoff(attempt))


class RetryAwareLM(dspy.LM):  # type: ignore[misc]  # dspy ships no stubs
    """``dspy.LM`` that retries only transient errors (see module docstring)."""

    def __init__(self, model: str, *, max_retries: int, **kwargs: Any) -> None:
        # num_retries=0 disables litellm's non-status-aware retry; we own retries.
        super().__init__(model, num_retries=0, **kwargs)
        # Retry budget lives on the instance, not in dump_state(). aiagent sets the
        # LM only via dspy.configure()/context() (never per-predictor), so
        # module.save(save_program=False) never serializes it. If per-predictor LM
        # overrides are ever adopted, override dump_state/load_state or the policy
        # is lost on reload (reconstructed as a bare dspy.LM with num_retries=0).
        self._max_retries = max_retries

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return run_with_retry(
            lambda: super(RetryAwareLM, self).forward(*args, **kwargs),
            max_retries=self._max_retries,
        )

    async def aforward(self, *args: Any, **kwargs: Any) -> Any:
        import asyncio

        attempt = 0
        while True:
            try:
                return await super().aforward(*args, **kwargs)
            except Exception as exc:
                if attempt >= self._max_retries or not is_retryable(exc):
                    raise
                attempt += 1
                await asyncio.sleep(_backoff(attempt))
