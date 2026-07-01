"""Load JSONL datasets into validated ``dspy.Example`` objects.

Each line is one JSON object, validated against a pydantic row model before being
converted to a ``dspy.Example`` with the declared input keys marked. Malformed
rows fail fast with ``path:line`` context rather than corrupting a run.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy
from pydantic import BaseModel, ValidationError

from aiagent.exceptions import DataLoadError


class ExpenseRow(BaseModel):
    """One labeled expense-extraction example."""

    text: str
    merchant: str
    date: str
    amount: float


def load_jsonl[M: BaseModel](
    path: str | Path,
    row_model: type[M],
    input_keys: tuple[str, ...],
) -> list[dspy.Example]:
    """Parse ``path`` (JSONL) into ``dspy.Example`` rows validated by ``row_model``.

    Blank lines are skipped. An all-empty / missing file raises (so an empty
    train/dev set surfaces here, not as a confusing downstream error).
    """
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"{p}: file not found")

    rows: list[dspy.Example] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DataLoadError(f"{p}:{i}: invalid JSON: {exc.msg}") from exc
        try:
            model = row_model.model_validate(raw)
        except ValidationError as exc:
            raise DataLoadError(f"{p}:{i}: schema error: {exc.errors()}") from exc
        example = dspy.Example(**model.model_dump()).with_inputs(*input_keys)
        rows.append(example)

    if not rows:
        raise DataLoadError(f"{p}: no valid rows")
    return rows


def load_expense_set(path: str | Path) -> list[dspy.Example]:
    """Load an expense JSONL file (``text`` is the only input field)."""
    return load_jsonl(path, ExpenseRow, input_keys=("text",))
