"""Packaging consistency: the Makefile's lock target and the lock files it writes."""

from __future__ import annotations

from helpers_scripts import ROOT, SYSTEM_PATH, run_script


def test_make_lock_passes_lock_args_to_every_compile() -> None:
    """uv keeps the pins already in a lock file, so a plain `make lock` never moves a package
    past an advisory: that takes e.g. LOCK_ARGS='--upgrade-package anyio'."""
    result = run_script(
        ["make", "-n", "lock", "LOCK_ARGS=--upgrade-package anyio"], {"PATH": SYSTEM_PATH}, ROOT
    )

    assert result.returncode == 0, result.stderr
    compiles = [line for line in result.stdout.splitlines() if "uv pip compile" in line]
    outputs = {line.split(" -o ")[1].split()[0] for line in compiles}
    assert {"requirements.txt", "requirements-dev.txt"} <= outputs
    assert all(line.endswith(" --upgrade-package anyio") for line in compiles)
