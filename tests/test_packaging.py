"""Packaging consistency and the installer build.

The Makefile's lock target and the lock files it writes agree with pyproject.toml, and
tools/package/build-binary.sh, run with fake tools, builds from those locks.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
import tarfile
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
    minimal_elf,
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


# --------------------------------------------------------------------------- the pinned CPython


def test_python_version_file_pins_an_exact_cpython_on_the_requires_python_floor() -> None:
    """.python-version is the one exact pin (dev venv, CI, the locks, the bundled interpreter);
    the package metadata keeps the X.Y floor."""
    match = re.fullmatch(r"(3\.\d+)\.\d+", PINNED_PYTHON)

    assert match is not None, f".python-version must be a full X.Y.Z, got {PINNED_PYTHON!r}"
    minor = match.group(1)
    assert PYPROJECT["project"]["requires-python"] == f">={minor}"
    assert f"Programming Language :: Python :: {minor}" in PYPROJECT["project"]["classifiers"]


def test_the_suite_runs_on_exactly_the_pinned_cpython() -> None:
    """The tests run on the CPython the installer ships: CI's setup-python reads .python-version,
    locally `make dev-install` makes the venv on it. After a pin bump, re-run dev-install."""
    assert platform.python_version() == PINNED_PYTHON, "stale .venv? run `make dev-install`"


def run_make_dev_install(tmp_path: Path, uv_venv: str) -> subprocess.CompletedProcess[str]:
    """`make dev-install` on a copy of the Makefile and .python-version with a fake uv that logs
    each call to uv.log and runs the shell code ``uv_venv`` for `uv venv`."""
    project = tmp_path / "project"
    project.mkdir()
    for name in ("Makefile", ".python-version"):
        shutil.copy2(ROOT / name, project / name)
    uv = write_program(
        tmp_path / "bin" / "uv",
        f'echo "$*" >> "{tmp_path / "uv.log"}"\n[ "$1" != venv ] || {{ {uv_venv}; }}\n',
    )
    return run_script(
        ["make", "-s", "dev-install"], {"PATH": f"{uv.parent}:{SYSTEM_PATH}"}, cwd=project
    )


def test_make_dev_install_recreates_the_venv_on_exactly_the_pinned_cpython(tmp_path: Path) -> None:
    """A .venv made on another patch (before a pin bump, or on uv's floating 3.X link) must not
    survive: the checks and tests would run on a CPython the installer does not ship."""
    result = run_make_dev_install(tmp_path, uv_venv="exit 0")

    assert result.returncode == 0, result.stderr
    venv, *installs = calls(tmp_path, "uv")
    assert venv[0] == "venv"
    assert "--clear" in venv
    assert venv[venv.index("--python") + 1] == PINNED_PYTHON
    assert installs and all(call[:2] == ["pip", "install"] for call in installs)


def test_make_dev_install_stops_with_uvs_reason_when_it_cannot_make_the_venv(
    tmp_path: Path,
) -> None:
    """E.g. a uv too old to know the pinned CPython, or Python downloads disabled: uv's own error,
    not a later, misleading one from installing into a venv that is not there (or is stale)."""
    result = run_make_dev_install(tmp_path, uv_venv=f'echo "{UV_CANNOT_DOWNLOAD}" >&2; exit 2')

    assert result.returncode != 0
    assert UV_CANNOT_DOWNLOAD in result.stderr.splitlines()
    assert [call[0] for call in calls(tmp_path, "uv")] == ["venv"]


# ------------------------------------------------------------- build-binary.sh with fake tools

BUILD_BINARY = ROOT / "tools" / "package" / "build-binary.sh"
CHECK_PYTHON = ROOT / "tools" / "package" / "check-python.sh"
STARTUP_IN = ROOT / "tools" / "package" / "startup.sh.in"
WHEEL = f"aiagent-{VERSION}-py3-none-any.whl"
MINOR = PINNED_PYTHON.rsplit(".", 1)[0]
PIP_CHECK_OK = 'echo "No broken requirements found."'
ZSTD_VERSION = "1.5.6"
UV_CANNOT_DOWNLOAD = f"error: No download found for request: cpython-{PINNED_PYTHON}-linux-x86_64-gnu"


def managed_python(tmp_path: Path) -> Path:
    """The fake uv-managed CPython of :func:`fake_build_project`."""
    return tmp_path / "uv-python" / f"cpython-{PINNED_PYTHON}-linux-x86_64-gnu"


def fake_build_project(
    tmp_path: Path,
    pip_check: str = PIP_CHECK_OK,
    staged_version: str = PINNED_PYTHON,
    uv_python_install: str = "exit 0",
    smoke_version: str = VERSION,
    zstd_version: str = ZSTD_VERSION,
    zstd_dynamic: bool = False,
) -> tuple[Path, dict[str, str]]:
    """A project with the real build-binary.sh and just enough around it (the locks, a cached
    zstd, x86_64), and fake tools that log to ``tmp_path``:

    - uv (uv.log) builds an empty wheel, runs the shell code ``uv_python_install`` for
      `uv python install` and finds a fake uv-managed CPython (with the shared libpython and
      pkgconfig that python-build-standalone ships)
    - that CPython's python (python.log) prints its prefix, reports ``staged_version`` and fakes
      pip (``pip_check`` is the shell code for `pip check`; pip is gone once the build strips it);
      everything else goes to the interpreter running the tests
    - curl fails, so a build without a usable cached zstd stops there
    - the cached zstd reports ``zstd_version`` and stores and cats instead of compressing;
      readelf (else the real one) sees it as a static ELF, or a dynamic one if ``zstd_dynamic``
    - makeself copies what it packs to ``tmp_path/mkself`` and writes an installer that lays down
      a fake aiagent (its `version` prints ``smoke_version``) and bundled python, which log each
      run and its HOME and AIAGENT_* to smoke.log
    - pip and makeself log the TMPDIR they get to tmpdir.log
    """
    project = tmp_path / "project"
    (project / "tools" / "package").mkdir(parents=True)
    for script in (BUILD_BINARY, CHECK_PYTHON, STARTUP_IN):
        shutil.copy2(script, project / "tools" / "package" / script.name)
    shutil.copy2(ROOT / ".python-version", project / ".python-version")
    (project / "pyproject.toml").write_text(f'[project]\nname = "aiagent"\nversion = "{VERSION}"\n')
    (project / "requirements.txt").write_text("dspy==3.2.1\n")
    (project / "requirements-build.txt").write_text("hatchling==1.32.4\n")
    python = managed_python(tmp_path)
    write_program(
        python / "bin" / f"python{MINOR}",
        f"""echo "$*" >> "{tmp_path / "python.log"}"
here="$(cd "$(dirname "$0")/.." && pwd)"
case "$*" in
    *sys.base_prefix*) echo "$here" ;;
    *"platform.python_version()"*) echo "{staged_version}" ;;
    "-m pip "*)
        echo "pip $TMPDIR" >> "{tmp_path / "tmpdir.log"}"
        [ -d "$here/lib/python{MINOR}/site-packages/pip" ] \\
            || {{ echo "$0: No module named pip" >&2; exit 1; }}
        case "$*" in
            "-m pip check") {pip_check} ;;
            *.whl) printf '#!%s\\nimport aiagent\\n' "$0" > "$here/bin/aiagent" ;;
        esac ;;
    *"import aiagent, dspy"*) ;;
    *) exec "{sys.executable}" "$@" ;;
esac
""",
    )
    (python / "lib" / f"python{MINOR}" / "site-packages" / "pip").mkdir(parents=True)
    (python / "lib" / f"libpython{MINOR}.so.1.0").write_bytes(minimal_elf("libc.so.6"))
    (python / "lib" / f"libpython{MINOR}.so").symlink_to(f"libpython{MINOR}.so.1.0")
    (python / "lib" / "libpython3.so").write_bytes(minimal_elf(f"libpython{MINOR}.so.1.0"))
    (python / "lib" / "pkgconfig").mkdir()
    (python / "lib" / "pkgconfig" / f"python-{MINOR}.pc").write_text("prefix=/install\n")
    dynload = python / "lib" / f"python{MINOR}" / "lib-dynload"
    dynload.mkdir()
    (dynload / f"_dbm.cpython-{MINOR.replace('.', '')}-x86_64-linux-gnu.so").write_bytes(
        minimal_elf("libc.so.6")
    )
    write_program(
        project / ".cache" / "aiagent-build" / f"zstd-{ZSTD_VERSION}-static-x86_64",
        f"""case "$1" in
    --version) echo "*** Zstandard CLI (64-bit) v{zstd_version}, by Yann Collet ***" ;;
    -dc) cat "$2" ;;
    *) for out; do :; done; cat > "$out" ;;
esac
""",
    )
    interpreter = "[Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]"
    program_headers = f'echo "      {interpreter}"' if zstd_dynamic else ":"
    smoke_log = tmp_path / "smoke.log"
    logged = f'echo "$0 $* HOME=$HOME $(env | grep ^AIAGENT_ | sort | tr "\\n" " ")" >> "{smoke_log}"'
    fakes = tmp_path / "installed"
    write_program(
        fakes / "aiagent",
        f"""{logged}
case "$1" in
    version) echo "{smoke_version}" ;;
    run) echo "Usage: aiagent run [OPTIONS] SKILL" ;;
    skills) echo "extract" ;;
esac
""",
    )
    write_program(fakes / f"python{MINOR}", f"{logged}\n")
    installer = write_program(
        tmp_path / "installer.sh",
        f"""TMPROOT=${{TMPDIR:=/var/tmp}}
umask 022
mkdir -p "$AIAGENT_PREFIX/bin" "$AIAGENT_PREFIX/lib/aiagent/bin"
cp "{fakes / "aiagent"}" "$AIAGENT_PREFIX/bin/"
cp "{fakes / f"python{MINOR}"}" "$AIAGENT_PREFIX/lib/aiagent/bin/"
""",
    )
    bin_dir = tmp_path / "bin"
    write_program(bin_dir / "uname", "echo x86_64\n")
    write_program(
        bin_dir / "makeself",
        f"""echo "makeself $TMPDIR" >> "{tmp_path / "tmpdir.log"}"
while [ "${{1#--}}" != "$1" ]; do
    case "$1" in --header|--tar-extra) shift ;; esac
    shift
done
cp -R "$1" "{tmp_path / "mkself"}"
cp "{installer}" "$2"
""",
    )
    (bin_dir / "makeself-header.sh").write_text("TMPROOT=\\${TMPDIR:=/tmp}\n")
    write_program(bin_dir / "curl", 'echo "curl: (6) no network in the tests" >&2; exit 6\n')
    write_program(
        bin_dir / "readelf",
        f"""for file; do :; done
case "$file" in
    */.cache/aiagent-build/zstd-*)
        [ "$1" != -lW ] || {program_headers} ;;
    *) exec /usr/bin/readelf "$@" ;;
esac
""",
    )
    write_program(
        bin_dir / "uv",
        f"""echo "$*" >> "{tmp_path / "uv.log"}"
case "$1 $2" in
    "--version ") echo "uv 0.0.1" ;;
    "build --wheel") while [ $# -gt 1 ]; do [ "$1" != -o ] || : > "$2/{WHEEL}"; shift; done ;;
    "python install") {uv_python_install} ;;
    "python find") echo "{python / "bin" / f"python{MINOR}"}" ;;
    "python dir") echo "{python.parent}" ;;
esac
""",
    )
    return project, {"PATH": f"{bin_dir}:{SYSTEM_PATH}", "HOME": str(tmp_path / "home")}


def run_build(project: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return run_script(["bash", str(project / "tools" / "package" / "build-binary.sh")], env)


def calls(tmp_path: Path, tool: str) -> list[list[str]]:
    log = tmp_path / f"{tool}.log"
    return [line.split() for line in log.read_text().splitlines()] if log.exists() else []


def test_the_wheel_is_built_by_the_hash_pinned_build_backend(tmp_path: Path) -> None:
    """hatchling writes the shipped wheel: it comes from requirements-build.txt, hash-checked,
    not freshly resolved from the index."""
    project, env = fake_build_project(tmp_path)

    run_build(project, env)

    (build,) = [call for call in calls(tmp_path, "uv") if call[:2] == ["build", "--wheel"]]
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
    assert calls(tmp_path, "uv") == []


def test_the_lock_and_the_wheel_install_as_hash_checked_wheels_without_the_resolver(
    tmp_path: Path,
) -> None:
    """Exactly the artifacts pinned in requirements.txt, and nothing pip adds on its own: no
    unpinned dependency, no sdist (whose build dependencies pip fetches unchecked), no cache."""
    project, env = fake_build_project(tmp_path)

    run_build(project, env)

    lock, wheel = [call for call in calls(tmp_path, "python") if call[:3] == ["-m", "pip", "install"]]
    for install in (lock, wheel):
        assert {"--no-deps", "--no-cache-dir", "--no-compile"} <= set(install)
        assert install[install.index("--only-binary") + 1] == ":all:"
        assert "-c" not in install
    assert lock[-3:] == ["--require-hashes", "-r", str(project / "requirements.txt")]
    assert wheel[-1] == str(project / "dist" / WHEEL)
    assert not (project / "dist" / "constraints.txt").exists()


def test_the_build_stops_when_pip_check_finds_the_lock_incomplete(tmp_path: Path) -> None:
    """--no-deps skips pip's resolver: a dependency added to pyproject.toml without `make lock`
    would be missing from the bundle."""
    unmet = "aiagent 0.3.1 requires pyyaml, which is not installed."
    project, env = fake_build_project(tmp_path, pip_check=f'echo "{unmet}"; exit 1')

    result = run_build(project, env)

    assert result.returncode == 1
    assert result.stderr.splitlines()[-2:] == [
        "ERROR: requirements.txt is incomplete (pip check):",
        unmet,
    ]


def test_pip_check_runs_before_the_build_strips_pip_and_the_aws_subtree(tmp_path: Path) -> None:
    """After the strips, litellm's boto3 and huggingface-hub's hf-xet are unmet by design, and
    pip itself is gone."""
    project, env = fake_build_project(tmp_path)

    result = run_build(project, env)

    assert ["-m", "pip", "check"] in calls(tmp_path, "python")
    assert "No module named pip" not in result.stderr
    assert "==> Precompiling all modules" in result.stdout


def test_the_build_refuses_a_python_version_that_is_not_an_exact_x_y_z(tmp_path: Path) -> None:
    project, env = fake_build_project(tmp_path)
    (project / ".python-version").write_text(f"{MINOR}\n")

    result = run_build(project, env)

    assert result.returncode == 1
    assert f"ERROR: .python-version must pin an exact CPython X.Y.Z, got '{MINOR}'" in (
        result.stderr
    )
    assert calls(tmp_path, "uv") == []


def test_the_build_stops_with_uvs_reason_when_uv_cannot_install_the_pinned_cpython(
    tmp_path: Path,
) -> None:
    """A patch bump of .python-version needs a uv that knows the new CPython: the error says so
    instead of a bare 'No interpreter found'."""
    project, env = fake_build_project(
        tmp_path, uv_python_install=f'echo "{UV_CANNOT_DOWNLOAD}" >&2; exit 2'
    )

    result = run_build(project, env)

    assert result.returncode == 1
    assert UV_CANNOT_DOWNLOAD in result.stderr.splitlines()
    assert f"ERROR: uv 0.0.1 cannot install CPython {PINNED_PYTHON} (.python-version): update uv" in (
        result.stderr
    )
    uv = calls(tmp_path, "uv")
    assert ["python", "install", PINNED_PYTHON] in uv
    assert not [call for call in uv if call[:2] == ["python", "find"]]


def test_the_build_stages_a_uv_managed_cpython_never_the_project_venv(tmp_path: Path) -> None:
    """`uv python find` alone returns the project's .venv when there is one, and the build then
    copied that venv's base_prefix: whatever patch the venv was made on."""
    project, env = fake_build_project(tmp_path)

    result = run_build(project, env)

    assert result.returncode == 0, result.stderr
    finds = [call for call in calls(tmp_path, "uv") if call[:2] == ["python", "find"]]
    assert finds == [["python", "find", "--system", "--managed-python", PINNED_PYTHON]]


def test_the_build_refuses_an_interpreter_that_resolves_outside_uvs_python_dir(
    tmp_path: Path,
) -> None:
    project, env = fake_build_project(tmp_path)
    managed = managed_python(tmp_path)
    elsewhere = tmp_path / "elsewhere" / managed.name
    elsewhere.parent.mkdir()
    managed.rename(elsewhere)
    managed.symlink_to(elsewhere)

    result = run_build(project, env)

    assert result.returncode == 1
    assert f"ERROR: {elsewhere} is not a uv-managed CPython" in result.stderr
    assert not (project / "dist" / ".build" / "python").exists()


def test_the_build_refuses_a_staged_interpreter_of_another_patch(tmp_path: Path) -> None:
    project, env = fake_build_project(tmp_path, staged_version=f"{MINOR}.0")

    result = run_build(project, env)

    assert result.returncode == 1
    assert (
        f"ERROR: the staged interpreter is CPython {MINOR}.0, .python-version pins {PINNED_PYTHON}"
        in result.stderr
    )


def test_the_installer_and_the_smoke_test_use_the_x_y_of_the_pinned_cpython(
    tmp_path: Path,
) -> None:
    """The bundle's paths are pythonX.Y (bin/python3.14, lib/python3.14), never pythonX.Y.Z."""
    project, env = fake_build_project(tmp_path)

    result = run_build(project, env)

    assert result.returncode == 0, result.stderr
    startup = (tmp_path / "mkself" / "startup.sh").read_text().splitlines()
    assert startup[:3] == ["#!/bin/sh", f"AIAGENT_VERSION={VERSION}", f"PYVER={MINOR}"]
    smoke = (tmp_path / "smoke.log").read_text().splitlines()
    assert any(f"/lib/aiagent/bin/python{MINOR} -I " in line for line in smoke)


WORK = re.compile(r"/var/tmp/aiagent-build\.[A-Za-z0-9]{6}")


def work_dir(tmp_path: Path) -> Path:
    """The build's private temp dir, from the TMPDIR (WORK/tmp) that pip and makeself got."""
    tmpdirs = {line.split()[1] for line in (tmp_path / "tmpdir.log").read_text().splitlines()}
    (tmpdir,) = tmpdirs
    assert re.fullmatch(rf"{WORK.pattern}/tmp", tmpdir), tmpdir
    return Path(tmpdir).parent


def test_the_build_keeps_its_temp_files_in_a_private_dir_under_var_tmp(tmp_path: Path) -> None:
    """Never /tmp (often a small RAM tmpfs): pip unpacks every wheel into TMPDIR, and makeself
    writes its ~75 MB archive there. The dir is gone when the build ends."""
    project, env = fake_build_project(tmp_path)

    result = run_build(project, {**env, "TMPDIR": str(tmp_path / "caller-tmp")})

    assert result.returncode == 0, result.stderr
    programs = {line.split()[0] for line in (tmp_path / "tmpdir.log").read_text().splitlines()}
    assert programs == {"pip", "makeself"}
    assert not work_dir(tmp_path).exists()


def test_a_failed_smoke_test_leaves_nothing_behind_but_the_build_output(tmp_path: Path) -> None:
    """The smoke install (~290 MB unpacked), the probe and the hostile-env files live in the
    private temp dir, so they go with it also when a check fails."""
    project, env = fake_build_project(tmp_path, smoke_version="0.0.0")

    result = run_build(project, env)

    assert result.returncode == 1
    assert f"ERROR: 'aiagent version' printed '0.0.0', want '{VERSION}'" in result.stderr
    assert sorted(p.name for p in (project / "dist").iterdir()) == [WHEEL, "aiagent-install.sh"]
    assert not work_dir(tmp_path).exists()


def test_the_smoke_test_ignores_the_maintainers_aiagent_config_and_settings(
    tmp_path: Path,
) -> None:
    """~/.config/aiagent (config.toml, user skills) and AIAGENT_* would change what the smoke
    test sees: it runs with HOME in the private temp dir and no AIAGENT_* but the prefix it
    installs to."""
    project, env = fake_build_project(tmp_path)
    config = tmp_path / "home" / ".config" / "aiagent"
    (config / "skills").mkdir(parents=True)
    (config / "config.toml").write_text('model = "from-the-maintainers-config"\n')
    env = {**env, "AIAGENT_MODEL": "from-the-env", "AIAGENT_SKILLS_DIR": str(config / "skills")}

    result = run_build(project, env)

    assert result.returncode == 0, result.stderr
    work = work_dir(tmp_path)
    smoke = (tmp_path / "smoke.log").read_text().splitlines()
    assert len(smoke) >= 9  # verify-versions, --help, version (twice), run/eval --help, ...
    for line in smoke:
        program, *_ = line.split()
        assert program.startswith(f"{work}/prefix/"), line
        assert line.endswith(f" HOME={work}/home "), line
    scripts = [line.split()[2] for line in smoke if line.split()[1] == "-I"]
    assert scripts == [str(project / "tools" / "package" / "verify-versions.py"), f"{work}/probe.py"]


def payload(tmp_path: Path) -> list[str]:
    """The names in the payload tar the build handed to makeself (the fake zstd stores)."""
    with tarfile.open(tmp_path / "mkself" / "bundle.tar.zst") as tar:
        return tar.getnames()


def test_the_bundle_ships_without_libpython(tmp_path: Path) -> None:
    """bin/python3.X has libpython linked in statically: the shared one (32 MB), the libpython3.so
    shim on top of it and their pkgconfig files are for embedding only."""
    project, env = fake_build_project(tmp_path)

    result = run_build(project, env)

    assert result.returncode == 0, result.stderr
    names = payload(tmp_path)
    assert f"python/lib/python{MINOR}/lib-dynload" in names
    assert [name for name in names if "libpython" in name or "pkgconfig" in name] == []
    assert "no libpython and no ELF that needs one" in result.stdout


def test_the_build_fails_when_a_staged_elf_needs_libpython(tmp_path: Path) -> None:
    project, env = fake_build_project(tmp_path)
    embed = managed_python(tmp_path) / "lib" / f"python{MINOR}" / "lib-dynload" / "_embed.so"
    embed.write_bytes(minimal_elf("libc.so.6", f"libpython{MINOR}.so.1.0"))

    result = run_build(project, env)

    assert result.returncode == 1
    staged = project / "dist" / ".build" / "python" / "lib" / f"python{MINOR}" / "lib-dynload"
    assert f"  {staged / '_embed.so'}: libpython{MINOR}.so.1.0" in result.stderr.splitlines()


def test_the_build_reuses_the_cached_static_zstd_of_the_pinned_version(tmp_path: Path) -> None:
    """The cache is versioned (zstd-1.5.6-static-x86_64): a bump never picks up the old binary."""
    project, env = fake_build_project(tmp_path)
    (project / ".cache" / "aiagent-build" / "zstd-static-x86_64").write_text("stale\n")

    result = run_build(project, env)

    assert result.returncode == 0, result.stderr
    assert "Building static zstd" not in result.stdout


@pytest.mark.parametrize(
    ("version", "dynamic"), [("1.5.5", False), (ZSTD_VERSION, True)], ids=["version", "dynamic"]
)
def test_a_cached_zstd_that_is_not_the_pinned_static_one_is_rebuilt(
    version: str, dynamic: bool, tmp_path: Path
) -> None:
    """The installer runs the bundled zstd on hosts without one: it must be the checksummed
    version and static (no program interpreter)."""
    project, env = fake_build_project(tmp_path, zstd_version=version, zstd_dynamic=dynamic)

    result = run_build(project, env)

    assert result.returncode == 6
    assert f"==> Building static zstd {ZSTD_VERSION}" in result.stdout
    assert "curl: (6) no network in the tests" in result.stderr


def test_the_report_gives_the_installers_size_not_its_disk_blocks(tmp_path: Path) -> None:
    """du without --apparent-size counts the blocks XFS preallocated (127M for a 75 MB file)."""
    project, env = fake_build_project(tmp_path)

    result = run_build(project, env)

    assert result.returncode == 0, result.stderr
    out = project / "dist" / "aiagent-install.sh"
    assert f"  {out}  ({out.stat().st_size})" in result.stdout.splitlines()


def test_a_module_that_does_not_compile_fails_the_build_with_the_compiler_error(
    tmp_path: Path,
) -> None:
    """The sourceless step deletes every .py next, so a module that failed to compile would just
    be missing from the bundle."""
    project, env = fake_build_project(tmp_path)
    (managed_python(tmp_path) / "lib" / f"python{MINOR}" / "broken.py").write_text("def (:\n")

    result = run_build(project, env)

    assert result.returncode == 1
    assert "*** Error compiling" in result.stdout
    assert "broken.py" in result.stdout
    assert "SyntaxError" in result.stdout
    assert "==> Dropping .py sources" not in result.stdout
