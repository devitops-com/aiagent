"""check-python.sh: the build gate on the staged interpreter (exact version, no libpython).

build-binary.sh stages uv's python-build-standalone CPython for the X.Y.Z in .python-version.
python-build-standalone links libpython statically into ``bin/python3.X``; the shared
``libpython3.X.so`` next to it (32 MB) is for embedding only, so the build drops it. The gate
fails unless the staged ``bin/python3.X`` runs and reports exactly the pinned version, and if a
libpython file is still there or any ELF in the tree NEEDs one. The ELF files here are minimal
hand-made shared objects with just a dynamic section, which is all ``readelf -d`` reads.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from helpers_scripts import PINNED_PYTHON as PINNED
from helpers_scripts import ROOT, SYSTEM_PATH, minimal_elf, run_script, write_program

CHECK_PYTHON = ROOT / "tools" / "package" / "check-python.sh"
MINOR = PINNED.rsplit(".", 1)[0]


def make_tree(root: Path, version: str = PINNED) -> Path:
    """A staged interpreter tree like the build's, minus libpython: a fake ``bin/python3.X``
    printing ``version`` and extension modules that need only libc."""
    write_program(root / "bin" / f"python{MINOR}", f'echo "{version}"\n')
    ext = root / "lib" / f"python{MINOR}" / "site-packages" / "pydantic_core" / "_core.so"
    ext.parent.mkdir(parents=True)
    ext.write_bytes(minimal_elf("libgcc_s.so.1", "libc.so.6"))
    dynload = root / "lib" / f"python{MINOR}" / "lib-dynload" / "_dbm.so"
    dynload.parent.mkdir(parents=True)
    dynload.write_bytes(minimal_elf("libc.so.6"))
    (root / "lib" / f"python{MINOR}" / "os.pyc").write_bytes(b"\xf3\r\r\n" + bytes(12))
    return root


def run_check(tree: Path, version: str = PINNED) -> subprocess.CompletedProcess[str]:
    return run_script(["bash", str(CHECK_PYTHON), str(tree), version], {"PATH": SYSTEM_PATH})


def test_check_python_accepts_the_pinned_version_without_libpython(tmp_path: Path) -> None:
    result = run_check(make_tree(tmp_path / "python"))

    assert result.returncode == 0, result.stderr
    assert f"CPython {PINNED}" in result.stdout
    assert "no libpython" in result.stdout


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
    "name", [f"libpython{MINOR}.so.1.0", f"libpython{MINOR}.so", "libpython3.so"]
)
def test_check_python_refuses_a_libpython_left_in_the_tree(name: str, tmp_path: Path) -> None:
    tree = make_tree(tmp_path / "python")
    (tree / "lib" / name).write_bytes(minimal_elf("libc.so.6"))

    result = run_check(tree)

    assert result.returncode == 1
    assert f"  {tree / 'lib' / name}" in result.stderr.splitlines()


def test_check_python_refuses_a_dangling_libpython_symlink(tmp_path: Path) -> None:
    tree = make_tree(tmp_path / "python")
    (tree / "lib" / f"libpython{MINOR}.so").symlink_to(f"libpython{MINOR}.so.1.0")

    result = run_check(tree)

    assert result.returncode == 1
    assert f"  {tree / 'lib' / f'libpython{MINOR}.so'}" in result.stderr.splitlines()


@pytest.mark.parametrize("renamed", [False, True], ids=["extension", "any-name"])
def test_check_python_refuses_an_elf_that_needs_libpython(renamed: bool, tmp_path: Path) -> None:
    """Found by the ELF magic, not by the file name."""
    tree = make_tree(tmp_path / "python")
    offender = tree / "lib" / f"python{MINOR}" / ("plugin.data" if renamed else "_embed.so")
    offender.write_bytes(minimal_elf("libc.so.6", f"libpython{MINOR}.so.1.0"))

    result = run_check(tree)

    assert result.returncode == 1
    assert f"  {offender}: libpython{MINOR}.so.1.0" in result.stderr.splitlines()


@pytest.mark.parametrize(
    "args", [[], ["only-one"], ["dir", "3.14"], ["dir", "3.14.7", "extra"]], ids=str
)
def test_check_python_rejects_bad_usage(args: list[str]) -> None:
    result = run_script(["bash", str(CHECK_PYTHON), *args], {"PATH": SYSTEM_PATH})

    assert result.returncode == 2
    assert "usage: check-python.sh" in result.stderr
