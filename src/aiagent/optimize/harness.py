"""Optimize a pipeline against a metric, then persist the compiled program.

Wraps DSPy's prompt optimizers — ``BootstrapFewShot`` (default; works from ~10
examples) and ``MIPROv2`` (opt-in; joint instruction + demo search) — behind a
uniform call, optionally measuring before/after on a dev set. Compiled programs
are saved as state-only JSON (``save_program=False``) and reloaded onto a fresh
instance of the same module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2

from aiagent.core.evaluate import evaluate

Method = str  # "bootstrap" | "mipro"


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of an optimization run."""

    compiled: dspy.Module
    baseline_score: float | None
    after_score: float | None
    save_path: str | None


def save_compiled(module: dspy.Module, path: str | Path) -> None:
    """Persist a compiled program as state-only JSON, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    module.save(str(p), save_program=False)


def load_compiled(module: dspy.Module, path: str | Path) -> dspy.Module:
    """Load saved state into ``module`` (a fresh instance of the same class)."""
    module.load(str(path))
    return module


def optimize(
    student: dspy.Module,
    trainset: Sequence[Any],
    metric: Callable[..., Any],
    *,
    method: Method = "bootstrap",
    devset: Sequence[Any] | None = None,
    save_path: str | Path | None = None,
    num_threads: int = 4,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 8,
    max_rounds: int = 1,
    mipro_auto: str = "light",
) -> OptimizeResult:
    """Compile ``student`` against ``metric`` and optionally measure the lift.

    Sequence: validate trainset -> baseline eval (if devset) -> compile ->
    after eval (if devset) -> save (if path).
    """
    if not trainset:
        raise ValueError("optimize(): trainset is empty")

    baseline = (
        evaluate(student, devset, metric, num_threads=num_threads).score
        if devset
        else None
    )

    if method == "bootstrap":
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
        )
        compiled = optimizer.compile(student=student, trainset=list(trainset))
    elif method == "mipro":
        optimizer = MIPROv2(metric=metric, auto=mipro_auto, num_threads=num_threads)
        valset = list(devset) if devset else list(trainset)
        compiled = optimizer.compile(
            student=student,
            trainset=list(trainset),
            valset=valset,
            requires_permission_to_run=False,
        )
    else:
        raise ValueError(f"unknown optimizer method: {method!r}")

    after = (
        evaluate(compiled, devset, metric, num_threads=num_threads).score
        if devset
        else None
    )

    if save_path is not None:
        save_compiled(compiled, save_path)

    return OptimizeResult(
        compiled=compiled,
        baseline_score=baseline,
        after_score=after,
        save_path=str(save_path) if save_path is not None else None,
    )
