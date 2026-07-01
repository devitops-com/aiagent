"""Unit tests for the optimize harness (edge cases + save/load, no LLM).

The full eval -> optimize -> eval lift is a live test (needs the router); see the
``-m live`` integration test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiagent.core.extract import ExtractExpenseModule
from aiagent.metrics.extraction import field_accuracy
from aiagent.optimize.harness import load_compiled, optimize, save_compiled


def test_empty_trainset_raises() -> None:
    with pytest.raises(ValueError, match="trainset is empty"):
        optimize(ExtractExpenseModule(), [], field_accuracy)


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown optimizer method"):
        optimize(ExtractExpenseModule(), [object()], field_accuracy, method="bogus")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "compiled.json"
    save_compiled(ExtractExpenseModule(), out)
    assert out.exists()  # parent dirs created
    reloaded = load_compiled(ExtractExpenseModule(), out)
    assert reloaded is not None
