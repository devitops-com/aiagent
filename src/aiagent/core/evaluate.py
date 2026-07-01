"""Structured wrapper over ``dspy.Evaluate``.

Returns an immutable result carrying the aggregate score plus per-example
records, so callers don't depend on DSPy's result shape directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import dspy


@dataclass(frozen=True)
class EvalRecord:
    """One scored example."""

    example: Any
    prediction: Any
    score: float


@dataclass(frozen=True)
class EvalResult:
    """Aggregate evaluation result."""

    score: float
    records: tuple[EvalRecord, ...]
    n: int


def evaluate(
    program: dspy.Module,
    devset: Sequence[Any],
    metric: Callable[..., Any],
    *,
    num_threads: int = 4,
    display_progress: bool = False,
    display_table: int = 0,
) -> EvalResult:
    """Run ``program`` over ``devset`` and aggregate ``metric`` scores."""
    if not devset:
        raise ValueError("evaluate(): devset is empty")

    evaluator = dspy.Evaluate(
        devset=list(devset),
        metric=metric,
        num_threads=num_threads,
        display_progress=display_progress,
        display_table=display_table,
    )
    result = evaluator(program)

    raw = getattr(result, "results", []) or []
    records = tuple(
        EvalRecord(example=ex, prediction=pred, score=float(score))
        for ex, pred, score in raw
    )
    return EvalResult(score=float(result.score), records=records, n=len(devset))
