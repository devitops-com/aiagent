"""Field-level extraction metrics for the expense demo.

A single callable serves two roles, per DSPy's metric contract
``metric(example, prediction, trace=None) -> float | bool``:

* ``trace is None`` (Evaluate / scoring): return the **float** fraction of
  fields correct, so the dev-set score is continuous.
* ``trace is not None`` (optimizer bootstrapping): return a **bool** gate — only
  fully-correct traces seed few-shot demos (a partial 0.33 float would otherwise
  count as truthy and pollute the demos).

Pure: depends only on attribute access, so it imports no ``dspy`` and is trivial
to unit-test with plain objects.
"""

from __future__ import annotations

import re
from typing import Any

FIELDS: tuple[str, ...] = ("merchant", "date", "amount")

# Only fully-correct traces become few-shot demonstrations during optimization.
_BOOTSTRAP_GATE = 1.0
_AMOUNT_TOLERANCE = 0.01
_AMOUNT_STRIP = re.compile(r"[^\d.\-]")


def _norm_str(value: Any) -> str:
    """Whitespace-collapse and case-fold a value for tolerant comparison."""
    return " ".join(str(value).split()).casefold()


def _parse_amount(value: Any) -> float | None:
    """Parse an amount tolerant of ``$``, ``USD``, and thousands separators."""
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = _AMOUNT_STRIP.sub("", str(value))
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _amount_ok(gold: Any, pred: Any) -> bool:
    g, p = _parse_amount(gold), _parse_amount(pred)
    return g is not None and p is not None and abs(g - p) <= _AMOUNT_TOLERANCE


def _field_hits(example: Any, prediction: Any) -> int:
    hits = 0
    hits += _norm_str(getattr(example, "merchant", "")) == _norm_str(
        getattr(prediction, "merchant", "")
    )
    hits += _norm_str(getattr(example, "date", "")) == _norm_str(
        getattr(prediction, "date", "")
    )
    hits += _amount_ok(
        getattr(example, "amount", None), getattr(prediction, "amount", None)
    )
    return hits


def field_accuracy(example: Any, prediction: Any, trace: Any = None) -> float | bool:
    """Fraction of ``{merchant, date, amount}`` correct (dual-use; see module doc)."""
    frac = _field_hits(example, prediction) / len(FIELDS)
    if trace is None:
        return frac
    return frac >= _BOOTSTRAP_GATE


def exact_match(example: Any, prediction: Any, trace: Any = None) -> bool:
    """True only when every field is correct (same in scoring and gating modes)."""
    return _field_hits(example, prediction) == len(FIELDS)
