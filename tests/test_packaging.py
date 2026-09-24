"""Packaging consistency and the installer build.

The Makefile's lock target and the lock files it writes agree with pyproject.toml, and
tools/package/build-binary.sh, run with fake tools, builds from those locks.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from helpers_scripts import (
    PINNED_PYTHON,
    PYPROJECT,
    ROOT,
    SYSTEM_PATH,
    VERSION,
    run_script,
    write_program,
)

LOCKS = ["requirements.txt", "requirements-dev.txt", "requirements-build.txt"]


# ------------------------------------------------------------------------------------ the locks


def test_make_lock_passes_lock_args_to_every_compile() -> None:
    """uv keeps the pins already in a lock file, so a plain `make lock` never moves a package
    past an advisory: that takes e.g. LOCK_ARGS='--upgrade-package anyio'."""
    result = run_script(
        ["make", "-n", "lock", "LOCK_ARGS=--upgrade-package anyio"], {"PATH": SYSTEM_PATH}, ROOT
    )

    assert result.returncode == 0, result.stderr
    compiles = [line for line in result.stdout.splitlines() if "uv pip compile" in line]
    outputs = {line.split(" -o ")[1].split()[0] for line in compiles}
    assert outputs == set(LOCKS)
    assert all(line.endswith(" --upgrade-package anyio") for line in compiles)


def test_make_lock_pins_the_build_backend_from_pyproject_with_hashes(tmp_path: Path) -> None:
    """requirements-build.txt locks what [build-system] requires: the code that writes the
    shipped wheel."""
    project = tmp_path / "project"
    project.mkdir()
    for name in ("Makefile", "pyproject.toml", ".python-version"):
        shutil.copy2(ROOT / name, project / name)
    log = tmp_path / "uv.log"
    uv = write_program(
        tmp_path / "bin" / "uv",
        f'echo "$*" >> "{log}"\ncase " $* " in *" - "*) sed "s/^/stdin: /" >> "{log}" ;; esac\n',
    )

    result = run_script(["make", "-s", "lock"], {"PATH": f"{uv.parent}:{SYSTEM_PATH}"}, project)

    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    (build,) = [i for i, call in enumerate(calls) if call.endswith("-o requirements-build.txt")]
    args = calls[build].split()
    assert args[:3] == ["pip", "compile", "-"]
    assert "--generate-hashes" in args
    assert args[args.index("--python-version") + 1] == PINNED_PYTHON
    assert calls[build + 1 :] == ["stdin: hatchling"]


def pinned(lock: str) -> dict[str, str]:
    text = (ROOT / lock).read_text(encoding="utf-8")
    return dict(re.findall(r"^([A-Za-z0-9._-]+)==(\S+)", text, re.MULTILINE))


def lock_entries(lock: str) -> list[str]:
    """One string per requirement (continuation lines joined)."""
    text = (ROOT / lock).read_text(encoding="utf-8").replace("\\\n", " ")
    return [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("lock", LOCKS)
def test_every_lock_entry_is_pinned_with_hashes(lock: str) -> None:
    """The build installs requirements.txt --require-hashes, and the wheel's build backend
    from requirements-build.txt the same way."""
    entries = lock_entries(lock)

    assert entries
    for entry in entries:
        assert re.match(r"^[A-Za-z0-9._-]+==\S+ ", entry), entry
        assert "--hash=sha256:" in entry, entry


@pytest.mark.parametrize(
    ("lock", "requirements"),
    [
        ("requirements.txt", PYPROJECT["project"]["dependencies"]),
        (
            "requirements-dev.txt",
            PYPROJECT["project"]["dependencies"]
            + PYPROJECT["project"]["optional-dependencies"]["dev"],
        ),
        ("requirements-build.txt", PYPROJECT["build-system"]["requires"]),
    ],
    ids=["runtime", "dev", "build"],
)
def test_the_locks_satisfy_what_pyproject_declares(lock: str, requirements: list[str]) -> None:
    """A dependency change in pyproject.toml without `make lock` fails here."""
    pins = {canonicalize_name(name): version for name, version in pinned(lock).items()}
    for requirement in map(Requirement, requirements):
        name = canonicalize_name(requirement.name)
        assert name in pins, f"{requirement} is not in {lock}: run make lock"
        assert requirement.specifier.contains(pins[name], prereleases=True), (
            f"{lock} pins {name}=={pins[name]}, pyproject wants {requirement}: run make lock"
        )


# ------------------------------------------------------------- build-binary.sh with fake tools

BUILD_BINARY = ROOT / "tools" / "package" / "build-binary.sh"
WHEEL = f"aiagent-{VERSION}-py3-none-any.whl"


def fake_build_project(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A project with the real build-binary.sh and just enough around it (the locks, makeself,
    x86_64), and a fake uv that logs every call to uv.log, builds an empty wheel and cannot
    install a CPython."""
    project = tmp_path / "project"
    (project / "tools" / "package").mkdir(parents=True)
    shutil.copy2(BUILD_BINARY, project / "tools" / "package" / "build-binary.sh")
    shutil.copy2(ROOT / ".python-version", project / ".python-version")
    (project / "pyproject.toml").write_text(f'[project]\nname = "aiagent"\nversion = "{VERSION}"\n')
    (project / "requirements.txt").write_text("dspy==3.2.1\n")
    (project / "requirements-build.txt").write_text("hatchling==1.32.4\n")
    bin_dir = tmp_path / "bin"
    write_program(bin_dir / "uname", "echo x86_64\n")
    write_program(bin_dir / "makeself", "exit 0\n")
    (bin_dir / "makeself-header.sh").write_text("TMPROOT=\\${TMPDIR:=/tmp}\n")
    write_program(
        bin_dir / "uv",
        f'''echo "$*" >> "{tmp_path / "uv.log"}"
case "$1 $2" in
    "build --wheel") while [ $# -gt 1 ]; do [ "$1" != -o ] || : > "$2/{WHEEL}"; shift; done ;;
    "python install") echo "error: no CPython downloads here" >&2; exit 2 ;;
    "python find") echo "error: No interpreter found" >&2; exit 2 ;;
esac
''',
    )
    return project, {"PATH": f"{bin_dir}:{SYSTEM_PATH}", "HOME": str(tmp_path / "home")}


def run_build(project: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return run_script(["bash", str(project / "tools" / "package" / "build-binary.sh")], env)


def uv_calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "uv.log"
    return [line.split() for line in log.read_text().splitlines()] if log.exists() else []


def test_the_wheel_is_built_by_the_hash_pinned_build_backend(tmp_path: Path) -> None:
    """hatchling writes the shipped wheel: it comes from requirements-build.txt, hash-checked,
    not freshly resolved from the index."""
    project, env = fake_build_project(tmp_path)

    run_build(project, env)

    (build,) = [call for call in uv_calls(tmp_path) if call[:2] == ["build", "--wheel"]]
    assert build[build.index("--build-constraints") + 1] == str(
        project / "requirements-build.txt"
    )
    assert "--require-hashes" in build


def test_the_build_stops_without_the_build_lock(tmp_path: Path) -> None:
    project, env = fake_build_project(tmp_path)
    (project / "requirements-build.txt").unlink()

    result = run_build(project, env)

    assert result.returncode == 1
    assert "requirements-build.txt missing — run 'make lock' first" in result.stderr
    assert uv_calls(tmp_path) == []
