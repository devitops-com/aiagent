"""Live integration tests against a real devai router (opt-in).

These are skipped by default (the suite runs ``-m "not live"``). Run inside the
devai network with::

    pytest -m live

They exercise the real eval -> optimize -> eval loop end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from aiagent.cli.app import app

pytestmark = pytest.mark.live

runner = CliRunner()


def test_doctor_reaches_router() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output


def test_run_extract_live() -> None:
    result = runner.invoke(
        app, ["run", "extract", "--text", "Lunch at Chipotle 3/14/2026 $14.27"]
    )
    assert result.exit_code == 0, result.output
    assert "merchant" in result.stdout


def test_eval_optimize_eval_lift(tmp_path: Path) -> None:
    base = runner.invoke(app, ["eval", "extract", "--json"])
    assert base.exit_code == 0, base.output

    out = tmp_path / "extract.json"
    opt = runner.invoke(app, ["optimize", "extract", "--out", str(out)])
    assert opt.exit_code == 0, opt.output
    assert out.exists()

    after = runner.invoke(
        app, ["eval", "extract", "--compiled", str(out), "--json"]
    )
    assert after.exit_code == 0, after.output
