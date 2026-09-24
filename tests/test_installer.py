"""The installer: the install.sh bootstrap and the makeself startup script.

Hermetic and fast: fake ``uname`` / ``curl`` / ``wget`` / ``zstd`` executables, a plain tar as
the payload and everything else under ``tmp_path``. Nothing reaches GitHub.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from helpers_scripts import (
    INSTALL_SH,
    ROOT,
    SYSTEM_PATH,
    VERSION,
    run_script,
    write_program,
)

STARTUP_IN = ROOT / "tools" / "package" / "startup.sh.in"
RELEASES = "https://github.com/devitops-com/aiagent/releases"


# ----------------------------------------------------------------------------------- install.sh


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    """Fake uname (Linux x86_64) and curl: curl logs its arguments and 'downloads' an installer
    that reports its own path, AIAGENT_PREFIX and TMPDIR."""
    bin_dir = tmp_path / "fake-bin"
    write_program(
        bin_dir / "uname",
        'case "$1" in -s) echo "${FAKE_OS:-Linux}" ;; -m) echo "${FAKE_ARCH:-x86_64}" ;; esac\n',
    )
    write_program(
        bin_dir / "curl",
        'printf "%s\\n" "$@" > "$FAKE_LOG"\n'
        '[ -z "$FAKE_CURL_FAIL" ] || exit 22\n'
        'while [ $# -gt 1 ]; do [ "$1" = -o ] && out="$2"; shift; done\n'
        "cat > \"$out\" <<'EOF'\n"
        'echo "installer=$0"\n'
        'echo "prefix=$AIAGENT_PREFIX"\n'
        'echo "tmpdir=$TMPDIR"\n'
        "EOF\n",
    )
    return bin_dir


def install_env(tmp_path: Path, fake_bin: Path, **extra: str) -> dict[str, str]:
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:{SYSTEM_PATH}",
        "FAKE_LOG": str(tmp_path / "curl.log"),
        "TMPDIR": str(tmp_path),
    }
    return env | extra


def run_install(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return run_script(["sh", str(INSTALL_SH)], env)


def installer_report(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)


def downloads_left(directory: Path) -> list[str]:
    return [p.name for p in directory.iterdir() if p.name.startswith("aiagent-install")]


@pytest.mark.parametrize(
    ("os_name", "arch"), [("Darwin", "arm64"), ("Linux", "aarch64"), ("FreeBSD", "x86_64")]
)
def test_install_sh_refuses_an_unsupported_platform(
    os_name: str, arch: str, tmp_path: Path, fake_bin: Path
) -> None:
    result = run_install(install_env(tmp_path, fake_bin, FAKE_OS=os_name, FAKE_ARCH=arch))

    assert result.returncode == 1
    assert f"unsupported platform {os_name}/{arch}" in result.stderr
    assert not (tmp_path / "curl.log").exists()  # nothing downloaded


@pytest.mark.parametrize(
    ("version", "url"),
    [
        (None, f"{RELEASES}/latest/download/aiagent-install.sh"),
        ("v0.1.0", f"{RELEASES}/download/v0.1.0/aiagent-install.sh"),
    ],
    ids=["latest", "pinned"],
)
def test_install_sh_downloads_the_release_asset_and_runs_it(
    version: str | None, url: str, tmp_path: Path, fake_bin: Path
) -> None:
    extra = {"AIAGENT_PREFIX": str(tmp_path / "prefix")}
    if version is not None:
        extra["AIAGENT_VERSION"] = version

    result = run_install(install_env(tmp_path, fake_bin, **extra))

    assert result.returncode == 0, result.stderr
    assert url in (tmp_path / "curl.log").read_text().splitlines()
    report = installer_report(result.stdout)
    assert report["prefix"] == str(tmp_path / "prefix")
    assert not Path(report["installer"]).exists()  # the downloaded installer is removed


def test_install_sh_fails_when_the_download_fails(tmp_path: Path, fake_bin: Path) -> None:
    result = run_install(install_env(tmp_path, fake_bin, FAKE_CURL_FAIL="1"))

    assert result.returncode == 1
    assert "download failed" in result.stderr
    assert downloads_left(tmp_path) == []


def test_install_sh_stages_the_download_in_tmpdir(tmp_path: Path, fake_bin: Path) -> None:
    result = run_install(install_env(tmp_path, fake_bin))

    assert result.returncode == 0, result.stderr
    report = installer_report(result.stdout)
    assert Path(report["installer"]).parent == tmp_path
    assert Path(report["installer"]).name.startswith("aiagent-install.")
    assert report["tmpdir"] == str(tmp_path)


def test_install_sh_stages_in_var_tmp_never_tmp_without_tmpdir(
    tmp_path: Path, fake_bin: Path
) -> None:
    """/tmp is often a small RAM-backed tmpfs; the installer unpacks ~290 MB."""
    env = install_env(tmp_path, fake_bin)
    del env["TMPDIR"]

    result = run_install(env)

    assert result.returncode == 0, result.stderr
    report = installer_report(result.stdout)
    assert Path(report["installer"]).parent == Path("/var/tmp")
    assert report["tmpdir"] == "/var/tmp"  # the installer extracts there too, not in /tmp
    assert not Path(report["installer"]).exists()


def test_install_sh_downloads_over_https_only_with_curl(tmp_path: Path, fake_bin: Path) -> None:
    """No redirect may downgrade the download to plain HTTP (GitHub redirects to its CDN)."""
    result = run_install(install_env(tmp_path, fake_bin))

    assert result.returncode == 0, result.stderr
    args = (tmp_path / "curl.log").read_text().splitlines()
    assert args[args.index("--proto") + 1] == "=https"
    assert "--tlsv1.2" in args


def test_install_sh_falls_back_to_wget_without_curl(tmp_path: Path, fake_bin: Path) -> None:
    """A PATH without curl: only the tools install.sh needs, and a fake wget."""
    bin_dir = tmp_path / "wget-bin"
    bin_dir.mkdir()
    shutil.copy2(fake_bin / "uname", bin_dir / "uname")
    for tool in ("sh", "mktemp", "rm"):
        (bin_dir / tool).symlink_to(shutil.which(tool) or tool)
    write_program(
        bin_dir / "wget",
        'printf "%s\\n" "$@" > "$FAKE_LOG"\n'
        'while [ $# -gt 0 ]; do case "$1" in -*O) out="$2"; shift ;; esac; shift; done\n'
        'printf \'echo "installer=$0"\\n\' > "$out"\n',
    )
    env = install_env(tmp_path, fake_bin) | {"PATH": str(bin_dir)}

    result = run_install(env)

    assert result.returncode == 0, result.stderr
    assert f"{RELEASES}/latest/download/aiagent-install.sh" in (
        (tmp_path / "curl.log").read_text().splitlines()  # the fake wget logs to the same file
    )
    assert "installer=" in result.stdout


def test_install_sh_names_the_repo_once() -> None:
    code = [
        line
        for line in INSTALL_SH.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]

    assert [line for line in code if "devitops-com/aiagent" in line] == [
        'REPO="devitops-com/aiagent"'
    ]


# ---------------------------------------------------------------- install.sh: AIAGENT_VERIFY=1


def write_fake_gh(bin_dir: Path, rc: int = 0) -> None:
    """A gh that logs its arguments, one per line, and keeps a copy of the file it checked."""
    write_program(
        bin_dir / "gh",
        'printf "%s\\n" "$@" > "$FAKE_GH_LOG"\n'
        '[ ! -f "$3" ] || cp "$3" "$FAKE_GH_LOG.subject"\n'
        f"exit {rc}\n",
    )


def gh_env(tmp_path: Path, bin_dir: Path, **extra: str) -> dict[str, str]:
    return install_env(tmp_path, bin_dir, FAKE_GH_LOG=str(tmp_path / "gh.log"), **extra)


def test_install_sh_verifies_the_attestation_before_running_the_installer(
    tmp_path: Path, fake_bin: Path
) -> None:
    write_fake_gh(fake_bin)

    result = run_install(gh_env(tmp_path, fake_bin, AIAGENT_VERIFY="1"))

    assert result.returncode == 0, result.stderr
    installer = installer_report(result.stdout)["installer"]
    gh_args = (tmp_path / "gh.log").read_text().splitlines()
    assert gh_args == ["attestation", "verify", installer, "--repo", "devitops-com/aiagent"]
    assert 'echo "installer=$0"' in (tmp_path / "gh.log.subject").read_text()  # the download
    assert downloads_left(tmp_path) == []


def test_install_sh_does_not_run_an_installer_that_fails_verification(
    tmp_path: Path, fake_bin: Path
) -> None:
    write_fake_gh(fake_bin, rc=1)

    result = run_install(gh_env(tmp_path, fake_bin, AIAGENT_VERIFY="1"))

    assert result.returncode == 1
    assert "installer=" not in result.stdout  # never executed
    assert "attestation verification failed" in result.stderr
    assert (tmp_path / "gh.log").exists()
    assert downloads_left(tmp_path) == []  # and removed


def test_install_sh_verify_without_gh_fails_before_downloading(
    tmp_path: Path, fake_bin: Path
) -> None:
    """A PATH with only what install.sh needs (the system dirs may hold a real gh)."""
    bin_dir = tmp_path / "no-gh-bin"
    bin_dir.mkdir()
    for fake in ("uname", "curl"):
        shutil.copy2(fake_bin / fake, bin_dir / fake)
    for tool in ("sh", "mktemp", "rm", "cat"):
        (bin_dir / tool).symlink_to(shutil.which(tool) or tool)
    env = install_env(tmp_path, fake_bin, AIAGENT_VERIFY="1") | {"PATH": str(bin_dir)}

    result = run_install(env)

    assert result.returncode == 1
    assert "AIAGENT_VERIFY=1 needs the GitHub CLI (gh)" in result.stderr
    assert "installer=" not in result.stdout
    assert not (tmp_path / "curl.log").exists()  # nothing downloaded
    assert downloads_left(tmp_path) == []


@pytest.mark.parametrize("value", [None, "", "0"], ids=["unset", "empty", "zero"])
def test_install_sh_without_aiagent_verify_never_calls_gh(
    value: str | None, tmp_path: Path, fake_bin: Path
) -> None:
    write_fake_gh(fake_bin, rc=1)
    extra = {} if value is None else {"AIAGENT_VERIFY": value}

    result = run_install(gh_env(tmp_path, fake_bin, **extra))

    assert result.returncode == 0, result.stderr
    assert "installer=" in result.stdout
    assert not (tmp_path / "gh.log").exists()


@pytest.mark.parametrize("value", ["yes", "true", "2"])
def test_install_sh_refuses_an_unknown_aiagent_verify_value(
    value: str, tmp_path: Path, fake_bin: Path
) -> None:
    """Fail closed: a typo must not silently skip the verification that was asked for."""
    write_fake_gh(fake_bin)

    result = run_install(gh_env(tmp_path, fake_bin, AIAGENT_VERIFY=value))

    assert result.returncode == 1
    assert f"AIAGENT_VERIFY must be 1 or 0, got '{value}'" in result.stderr
    assert not (tmp_path / "curl.log").exists()
    assert not (tmp_path / "gh.log").exists()


# ------------------------------------------------------------------------ installer startup

PYVER = "3.14"  # what build-binary.sh bakes in: the X.Y of the bundled CPython
FOREIGN_ID = 4242  # the archive's owner: a uid/gid that means nothing on the installing host
UNSHARE = shutil.which("unshare")


def make_extraction_dir(tmp_path: Path, python: str = "echo bundled python\n") -> Path:
    """What makeself extracts: startup.sh, a zstd and bundle.tar.zst (here: a plain tar).

    Like a careless build, the tar keeps a foreign owner and group-writable modes. ``python``
    is the body of the fake bundled interpreter (the installer runs it once before installing).
    """
    here = tmp_path / "extracted"
    write_program(here / "zstd", 'cat "$2"\n')  # called as: zstd -dc bundle.tar.zst
    members = {
        "python/": None,
        "python/bin/": None,
        f"python/bin/python{PYVER}": f"#!/bin/sh\n{python}",
        "python/bin/aiagent": f"#!/build/host/dist/.build/python/bin/python{PYVER}\nprint('aiagent')\n",
        "python/lib/": None,
        "python/lib/module.pyc": "pyc",
        "doc/": None,
        "doc/README.md": "# aiagent\n",
    }
    with tarfile.open(here / "bundle.tar.zst", "w") as bundle:
        for name, content in members.items():
            info = tarfile.TarInfo(name.rstrip("/"))
            info.uid = info.gid = FOREIGN_ID
            if content is None:
                info.type, info.mode = tarfile.DIRTYPE, 0o775
                bundle.addfile(info)
                continue
            data = content.encode()
            info.size = len(data)
            info.mode = 0o775 if "/bin/" in name else 0o664
            bundle.addfile(info, io.BytesIO(data))
    baked = f"AIAGENT_VERSION={VERSION}\nPYVER={PYVER}\n"  # as build-binary.sh bakes them in
    write_program(here / "startup.sh", baked + STARTUP_IN.read_text())
    return here


def run_startup(
    here: Path, tmp_path: Path, *args: str, wrap: tuple[str, ...] = (), **env: str
) -> subprocess.CompletedProcess[str]:
    """Run startup.sh as makeself does: from ``here``, under umask 077, via ``sh``."""
    base = {"HOME": str(tmp_path / "home"), "PATH": SYSTEM_PATH, "TMPDIR": str(tmp_path)}
    command = [*wrap, "sh", "-c", 'umask 077 && exec sh ./startup.sh "$@"', "sh", *args]
    return run_script(command, base | env, cwd=here)


@pytest.fixture
def short(tmp_path: Path) -> Iterator[Path]:
    """A short path (a symlink to ``tmp_path`` under /var/tmp) for install prefixes.

    The installer refuses a prefix whose launcher shebang exceeds the kernel's 127 bytes, and
    pytest's temp paths can be longer than that allows.
    """
    holder = Path(tempfile.mkdtemp(prefix="aiagent-p.", dir="/var/tmp"))
    link = holder / "t"
    link.symlink_to(tmp_path, target_is_directory=True)
    yield link
    shutil.rmtree(holder)


def assert_installed(prefix: Path) -> None:
    lib = prefix / "lib" / "aiagent"
    assert sorted(p.name for p in (prefix / "bin").iterdir()) == ["aiagent"]
    assert os.readlink(prefix / "bin" / "aiagent") == f"{lib}/bin/aiagent"
    script = (lib / "bin" / "aiagent").read_text().splitlines()
    assert script == [f"#!{lib}/bin/python{PYVER} -I", "print('aiagent')"]
    assert (prefix / "share" / "doc" / "aiagent" / "README.md").is_file()
    assert [p.name for p in prefix.iterdir() if p.name.startswith(".")] == []  # stage removed


def assert_usable_by_everyone_writable_only_by_owner(prefix: Path) -> None:
    for path in [prefix, *prefix.rglob("*")]:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        assert mode & 0o022 == 0, f"group/other-writable: {path} {oct(mode)}"
        assert mode & 0o004, f"not world-readable: {path} {oct(mode)}"
        if path.is_dir():
            assert mode & 0o001, f"not world-traversable: {path} {oct(mode)}"


def test_startup_installs_into_aiagent_prefix_with_only_aiagent_on_path(
    tmp_path: Path, short: Path
) -> None:
    prefix = short / "prefix"

    result = run_startup(make_extraction_dir(tmp_path), tmp_path, AIAGENT_PREFIX=str(prefix))

    assert result.returncode == 0, result.stderr
    assert_installed(prefix)


def test_startup_launcher_runs_the_bundled_python_isolated(tmp_path: Path, short: Path) -> None:
    """-I, not just -s: PYTHONPATH/PYTHONHOME and the script's directory never reach sys.path."""
    prefix = short / "prefix"

    run_startup(make_extraction_dir(tmp_path), tmp_path, AIAGENT_PREFIX=str(prefix))

    shebang = (prefix / "lib" / "aiagent" / "bin" / "aiagent").read_text().splitlines()[0]
    assert shebang.endswith(" -I")


@pytest.mark.parametrize("form", ["separate", "joined"])
def test_startup_prefix_option_wins_over_aiagent_prefix(
    form: str, tmp_path: Path, short: Path
) -> None:
    prefix = short / "chosen"
    args = ["--prefix", str(prefix)] if form == "separate" else [f"--prefix={prefix}"]

    result = run_startup(
        make_extraction_dir(tmp_path), tmp_path, *args, AIAGENT_PREFIX=str(short / "ignored")
    )

    assert result.returncode == 0, result.stderr
    assert_installed(prefix)
    assert not (short / "ignored").exists()


@pytest.mark.parametrize("args", [["--target", "/opt/aiagent"], ["--bogus"], ["--prefix"]])
def test_startup_rejects_an_unknown_or_incomplete_option(
    args: list[str], tmp_path: Path, short: Path
) -> None:
    """--target belongs to makeself (it keeps the raw payload there), so it is not ours."""
    result = run_startup(
        make_extraction_dir(tmp_path), tmp_path, *args, AIAGENT_PREFIX=str(short / "prefix")
    )

    assert result.returncode == 2
    assert "Usage: sh aiagent-install.sh [-- --prefix DIR]" in result.stderr
    assert not (short / "prefix").exists()


def test_startup_refuses_an_empty_prefix(tmp_path: Path, short: Path) -> None:
    result = run_startup(make_extraction_dir(tmp_path), tmp_path, "--prefix=", HOME=str(short))

    assert result.returncode == 2
    assert "empty install prefix" in result.stderr
    assert not (short / ".local").exists()


def test_startup_refuses_a_prefix_whose_shebang_is_too_long(tmp_path: Path) -> None:
    prefix = tmp_path / ("p" * 120)
    prefix.mkdir()

    result = run_startup(make_extraction_dir(tmp_path), tmp_path, AIAGENT_PREFIX=str(prefix))

    assert result.returncode == 1
    assert "install path too long" in result.stderr
    assert "AIAGENT_PREFIX" in result.stderr
    assert list(prefix.iterdir()) == []  # refused before anything was unpacked


@pytest.mark.parametrize("blank", [" ", "\t", "\n"], ids=["space", "tab", "newline"])
def test_startup_refuses_a_prefix_with_whitespace(blank: str, tmp_path: Path, short: Path) -> None:
    """The kernel splits a shebang at whitespace: the launcher could never start."""
    prefix = short / f"my{blank}tools"
    prefix.mkdir()

    result = run_startup(make_extraction_dir(tmp_path), tmp_path, AIAGENT_PREFIX=str(prefix))

    assert result.returncode == 1
    assert "whitespace" in result.stderr
    assert "AIAGENT_PREFIX" in result.stderr
    assert list(prefix.iterdir()) == []  # refused before anything was unpacked


@pytest.mark.parametrize("how", ["env", "option"])
def test_startup_resolves_a_relative_prefix_against_the_callers_directory(
    how: str, tmp_path: Path, short: Path
) -> None:
    """makeself runs startup.sh inside its temp dir and deletes it afterwards; the caller's
    directory is $USER_PWD."""
    caller = short / "caller"
    caller.mkdir()
    args, env = (
        (["--prefix", "rel/pfx"], {}) if how == "option" else ([], {"AIAGENT_PREFIX": "rel/pfx"})
    )

    result = run_startup(
        make_extraction_dir(tmp_path), tmp_path, *args, USER_PWD=str(caller), **env
    )

    assert result.returncode == 0, result.stderr
    assert_installed(caller / "rel" / "pfx")
    assert not (tmp_path / "extracted" / "rel").exists()


def test_startup_expands_a_quoted_tilde_prefix(tmp_path: Path, short: Path) -> None:
    home = short / "home"

    result = run_startup(
        make_extraction_dir(tmp_path), tmp_path, HOME=str(home), AIAGENT_PREFIX="~/apps"
    )

    assert result.returncode == 0, result.stderr
    assert_installed(home / "apps")


def test_startup_installs_a_tree_everyone_can_use_but_only_the_owner_can_write(
    tmp_path: Path, short: Path
) -> None:
    """Under makeself's umask 077, from an archive with group-writable modes."""
    prefix = short / "prefix"

    result = run_startup(make_extraction_dir(tmp_path), tmp_path, AIAGENT_PREFIX=str(prefix))

    assert result.returncode == 0, result.stderr
    assert_usable_by_everyone_writable_only_by_owner(prefix)


def userns_available(*flags: str) -> bool:
    if UNSHARE is None:
        return False
    probe = subprocess.run(
        [UNSHARE, "--user", "--map-root-user", *flags, "true"], capture_output=True, check=False
    )
    return probe.returncode == 0


@pytest.mark.skipif(not userns_available(), reason="needs unprivileged user namespaces")
def test_startup_as_root_gives_the_tree_to_root_not_to_the_archived_owner(
    tmp_path: Path, short: Path
) -> None:
    """As root, tar would restore the archive's owner (uid 4242, unmapped in the namespace, so
    chown fails) and its group-writable modes."""
    prefix = short / "prefix"

    result = run_startup(
        make_extraction_dir(tmp_path),
        tmp_path,
        wrap=(str(UNSHARE), "--user", "--map-root-user"),
        AIAGENT_PREFIX=str(prefix),
    )

    assert result.returncode == 0, result.stderr
    assert_installed(prefix)
    assert {p.lstat().st_uid for p in [prefix, *prefix.rglob("*")]} == {os.getuid()}
    assert_usable_by_everyone_writable_only_by_owner(prefix)


@pytest.mark.skipif(not userns_available("--mount"), reason="needs user + mount namespaces")
def test_startup_works_when_the_extraction_dir_is_mounted_noexec(
    tmp_path: Path, short: Path
) -> None:
    """CIS-hardened hosts mount /tmp and /var/tmp (makeself's TMPDIR) noexec: nothing may be
    executed from there, neither startup.sh nor the bundled zstd."""
    here, noexec, prefix = make_extraction_dir(tmp_path), tmp_path / "noexec", short / "prefix"
    noexec.mkdir()
    script = (
        'mount -t tmpfs -o noexec,mode=0700 tmpfs "$1" && cp -p "$2"/* "$1"/ && cd "$1" '
        "&& umask 077 && exec sh ./startup.sh"
    )
    command = [str(UNSHARE), "--user", "--map-root-user", "--mount", "sh", "-c", script]
    env = {"HOME": str(tmp_path / "home"), "PATH": SYSTEM_PATH, "AIAGENT_PREFIX": str(prefix)}

    result = run_script([*command, "sh", str(noexec), str(here)], env)

    assert result.returncode == 0, result.stderr
    assert_installed(prefix)


def test_startup_refuses_an_interpreter_that_cannot_run_here_and_keeps_the_old_install(
    tmp_path: Path, short: Path
) -> None:
    """E.g. a musl host (the bundled CPython needs glibc) or a noexec prefix."""
    prefix = short / "prefix"
    old = prefix / "lib" / "aiagent" / "bin" / "aiagent"
    write_program(old, "echo old\n")
    here = make_extraction_dir(tmp_path, python="exit 127\n")

    result = run_startup(here, tmp_path, AIAGENT_PREFIX=str(prefix))

    assert result.returncode == 1
    assert "cannot run on this host" in result.stderr
    assert old.read_text() == "#!/bin/sh\necho old\n"
    assert [p.name for p in prefix.iterdir()] == ["lib"]  # no stage left, nothing added


def test_startup_removes_the_bundled_python_link_older_installers_put_on_path(
    tmp_path: Path, short: Path
) -> None:
    """Installers up to 0.3.0 linked the bundled interpreter as $PREFIX/bin/pythonX.Y."""
    prefix = short / "prefix"
    (prefix / "bin").mkdir(parents=True)
    stale = prefix / "bin" / f"python{PYVER}"
    stale.symlink_to(prefix / "lib" / "aiagent" / "bin" / f"python{PYVER}")

    result = run_startup(make_extraction_dir(tmp_path), tmp_path, AIAGENT_PREFIX=str(prefix))

    assert result.returncode == 0, result.stderr
    assert_installed(prefix)
    assert f"{stale}  (older installers put the bundled interpreter on PATH" in result.stdout


def test_startup_keeps_a_python_link_that_does_not_point_into_its_own_lib(
    tmp_path: Path, short: Path
) -> None:
    prefix = short / "prefix"
    (prefix / "bin").mkdir(parents=True)
    own = prefix / "bin" / f"python{PYVER}"
    own.symlink_to(f"/usr/bin/python{PYVER}")

    result = run_startup(make_extraction_dir(tmp_path), tmp_path, AIAGENT_PREFIX=str(prefix))

    assert result.returncode == 0, result.stderr
    assert os.readlink(own) == f"/usr/bin/python{PYVER}"
    assert "Removed:" not in result.stdout
