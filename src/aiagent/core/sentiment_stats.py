"""Aggregate per-segment sentiment samples into summary statistics (pure).

Given a list of per-segment score samples (each segment scored ``resample``
times), this computes the overall sentiment plus the statistical properties the
skill reports:

* **volatility** — sample standard deviation of the per-segment mean scores; how
  much sentiment swings *across the content*.
* **model_uncertainty** — mean within-segment standard deviation across the
  resamples; how much the model *disagrees with itself* on the same passage.
* **significance** — a one-sample Student's t-test of the segment means against a
  neutral mean of 0, reported as a t-statistic and two-sided p-value, plus a
  qualitative confidence label.

Depends only on ``math`` (no numpy/scipy), so it is trivial to unit-test and
safe for the torch-free bundle. The t-distribution p-value uses the regularized
incomplete beta function (Numerical Recipes continued-fraction form).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from aiagent.exceptions import SourceError

SCALE_MIN = -10.0
SCALE_MAX = 10.0
_Z_95 = 1.959963984540054  # normal-approx 95% CI multiplier


@dataclass(frozen=True)
class SentimentStats:
    """Immutable summary of an aggregate sentiment measurement."""

    n_segments: int
    n_samples: int
    mean: float
    volatility: float
    model_uncertainty: float
    std_error: float
    t_statistic: float | None
    p_value: float | None
    ci_low: float | None
    ci_high: float | None
    polarity: str
    confidence: str


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    var = math.fsum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def _betacf(a: float, b: float, x: float) -> float:
    max_iter, eps, fpmin = 200, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def two_sided_t_p(t: float, df: float) -> float:
    """Two-sided p-value for a t-statistic with ``df`` degrees of freedom."""
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _polarity(mean: float) -> str:
    if mean >= 6.0:
        return "very positive"
    if mean >= 2.0:
        return "positive"
    if mean > -2.0:
        return "neutral / mixed"
    if mean > -6.0:
        return "negative"
    return "very negative"


def _confidence(p_value: float | None) -> str:
    if p_value is None:
        return "insufficient-data"
    if p_value < 0.01:
        return "high"
    if p_value < 0.05:
        return "moderate"
    if p_value < 0.10:
        return "low"
    return "not-significant"


def summarize(segment_samples: Sequence[Sequence[float]]) -> SentimentStats:
    """Summarize per-segment score samples into :class:`SentimentStats`."""
    populated = [list(s) for s in segment_samples if s]
    if not populated:
        raise SourceError("no sentiment samples to summarize")

    segment_means = [_mean(s) for s in populated]
    n = len(segment_means)
    n_samples = sum(len(s) for s in populated)
    mean = _mean(segment_means)
    model_uncertainty = _mean([_sample_stdev(s) for s in populated])

    volatility = _sample_stdev(segment_means)
    std_error = volatility / math.sqrt(n) if n >= 2 else 0.0

    t_statistic: float | None
    p_value: float | None
    ci_low: float | None
    ci_high: float | None
    if n < 2:
        # A single segment gives no spread to test against.
        t_statistic = p_value = ci_low = ci_high = None
    elif std_error == 0.0:
        # Every segment agreed exactly.
        if mean == 0.0:
            t_statistic, p_value = 0.0, 1.0
        else:
            t_statistic, p_value = None, 0.0  # unanimous, non-neutral
        ci_low = ci_high = mean
    else:
        t_statistic = mean / std_error
        p_value = two_sided_t_p(t_statistic, n - 1)
        ci_low = mean - _Z_95 * std_error
        ci_high = mean + _Z_95 * std_error

    return SentimentStats(
        n_segments=n,
        n_samples=n_samples,
        mean=mean,
        volatility=volatility,
        model_uncertainty=model_uncertainty,
        std_error=std_error,
        t_statistic=t_statistic,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        polarity=_polarity(mean),
        confidence=_confidence(p_value),
    )
