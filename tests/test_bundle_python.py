"""check-python.sh: the build gate on the staged interpreter (exactly the pinned CPython).

build-binary.sh stages uv's python-build-standalone CPython for the X.Y.Z in .python-version.
The gate fails unless the staged ``bin/python3.X`` runs and reports exactly that version.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from helpers_scripts import PINNED_PYTHON as PINNED
from helpers_scripts import ROOT, SYSTEM_PATH, run_script, write_program

CHECK_PYTHON = ROOT / "tools" / "package" / "check-python.sh"
MINOR = PINNED.rsplit(".", 1)[0]


def make_tree(root: Path, version: str = PINNED) -> Path:
    """A staged interpreter tree like the build's, with a fake ``bin/python3.X`` printing
    ``version``."""
    write_program(root / "bin" / f"python{MINOR}", f'echo "{version}"\n')
    (root / "lib" / f"python{MINOR}").mkdir(parents=True)
    return root


def run_check(tree: Path, version: str = PINNED) -> subprocess.CompletedProcess[str]:
    return run_script(["bash", str(CHECK_PYTHON), str(tree), version], {"PATH": SYSTEM_PATH})


def test_check_python_accepts_the_pinned_version(tmp_path: Path) -> None:
    result = run_check(make_tree(tmp_path / "python"))

    assert result.returncode == 0, result.stderr
    assert f"CPython {PINNED}" in result.stdout


@pytest.mark.parametrize("staged", [f"{MINOR}.0", f"{PINNED}1", "3.13.9"])
def test_check_python_refuses_any_other_interpreter_version(staged: str, tmp_path: Path) -> None:
    result = run_check(make_tree(tmp_path / "python", version=staged))

    assert result.returncode == 1
    assert f"staged interpreter is CPython {staged}, .python-version pins {PINNED}" in (
        result.stderr
    )


def test_check_python_refuses_an_interpreter_that_does_not_run(tmp_path: Path) -> None:
    tree = make_tree(tmp_path / "python")
    write_program(tree / "bin" / f"python{MINOR}", "exit 127\n")

    result = run_check(tree)

    assert result.returncode == 1
    assert "cannot run" in result.stderr


@pytest.mark.parametrize(
    "args", [[], ["only-one"], ["dir", "3.14"], ["dir", "3.14.7", "extra"]], ids=str
)
def test_check_python_rejects_bad_usage(args: list[str]) -> None:
    result = run_script(["bash", str(CHECK_PYTHON), *args], {"PATH": SYSTEM_PATH})

    assert result.returncode == 2
    assert "usage: check-python.sh" in result.stderr
