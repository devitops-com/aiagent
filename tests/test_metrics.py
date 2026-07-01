"""Unit tests for the dual-use extraction metric (no dspy / no LLM)."""

from __future__ import annotations

from types import SimpleNamespace

from aiagent.metrics.extraction import exact_match, field_accuracy

GOLD = SimpleNamespace(merchant="Whole Foods", date="2026-01-05", amount=1234.56)


def _pred(**kw: object) -> SimpleNamespace:
    base = {"merchant": "Whole Foods", "date": "2026-01-05", "amount": 1234.56}
    base.update(kw)
    return SimpleNamespace(**base)


def test_perfect_scores_one_float() -> None:
    score = field_accuracy(GOLD, _pred(), trace=None)
    assert score == 1.0 and isinstance(score, float)


def test_partial_fraction() -> None:
    assert field_accuracy(GOLD, _pred(merchant="Costco"), trace=None) == 2 / 3


def test_amount_currency_symbol_normalized() -> None:
    assert field_accuracy(GOLD, _pred(amount="$1,234.56"), trace=None) == 1.0


def test_amount_usd_suffix_normalized() -> None:
    assert field_accuracy(GOLD, _pred(amount="1234.56 USD"), trace=None) == 1.0


def test_amount_within_tolerance() -> None:
    assert field_accuracy(GOLD, _pred(amount=1234.565), trace=None) == 1.0


def test_amount_unparseable_counts_wrong() -> None:
    assert field_accuracy(GOLD, _pred(amount="n/a"), trace=None) == 2 / 3


def test_merchant_case_and_whitespace_insensitive() -> None:
    assert field_accuracy(GOLD, _pred(merchant="  whole   foods "), trace=None) == 1.0


def test_trace_gate_is_bool_and_strict() -> None:
    assert field_accuracy(GOLD, _pred(), trace=[]) is True
    assert field_accuracy(GOLD, _pred(merchant="x"), trace=[]) is False


def test_exact_match() -> None:
    assert exact_match(GOLD, _pred()) is True
    assert exact_match(GOLD, _pred(date="2026-01-06")) is False


def test_bool_amount_rejected() -> None:
    # bool is an int subclass; must not be read as 1.0/0.0
    assert field_accuracy(GOLD, _pred(amount=True), trace=None) == 2 / 3
