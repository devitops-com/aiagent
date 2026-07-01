"""Unit tests for the JSONL loader + the bundled extract datasets."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import pytest

from aiagent.data.loader import ExpenseRow, load_expense_set, load_jsonl
from aiagent.exceptions import DataLoadError

_GOOD = '{"text": "Coffee 4.95 on 2026-02-14", "merchant": "Cafe", "date": "2026-02-14", "amount": 4.95}'


def test_loads_good_rows_and_marks_input(tmp_path: Path) -> None:
    f = tmp_path / "d.jsonl"
    f.write_text(_GOOD + "\n\n" + _GOOD + "\n")  # includes a blank line
    rows = load_expense_set(f)
    assert len(rows) == 2
    assert rows[0].merchant == "Cafe"
    assert "text" in rows[0].inputs()


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="file not found"):
        load_expense_set(tmp_path / "nope.jsonl")


def test_empty_file_raises(tmp_path: Path) -> None:
    f = tmp_path / "empty.jsonl"
    f.write_text("\n  \n")
    with pytest.raises(DataLoadError, match="no valid rows"):
        load_expense_set(f)


def test_bad_json_raises_with_line(tmp_path: Path) -> None:
    f = tmp_path / "bad.jsonl"
    f.write_text(_GOOD + "\n{not json}\n")
    with pytest.raises(DataLoadError, match=":2: invalid JSON"):
        load_expense_set(f)


def test_schema_error_raises_with_line(tmp_path: Path) -> None:
    f = tmp_path / "schema.jsonl"
    f.write_text('{"text": "x", "merchant": "m", "date": "d"}\n')  # missing amount
    with pytest.raises(DataLoadError, match=":1: schema error"):
        load_jsonl(f, ExpenseRow, input_keys=("text",))


@pytest.mark.parametrize(("name", "count"), [("trainset.jsonl", 17), ("devset.jsonl", 8)])
def test_bundled_datasets_parse(name: str, count: int) -> None:
    res = files("aiagent.builtin_skills").joinpath("extract", name)
    with as_file(res) as p:
        rows = load_expense_set(p)
    assert len(rows) == count
    assert all(isinstance(r.amount, float) for r in rows)
