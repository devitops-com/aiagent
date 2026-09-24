"""Shared by the packaging, installer and release tests: run the project's scripts hermetically."""

from __future__ import annotations

import struct
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
VERSION = PYPROJECT["project"]["version"]
PINNED_PYTHON = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
INSTALL_SH = ROOT / "install.sh"
SYSTEM_PATH = "/usr/bin:/bin"
SCRIPT_TIMEOUT_S = 60.0

DT_NULL, DT_NEEDED, DT_STRTAB, DT_STRSZ = 0, 1, 5, 10
PT_LOAD, PT_DYNAMIC = 1, 2
EHDR_SIZE, PHDR_SIZE = 64, 56


def write_program(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


def run_script(
    command: list[str], env: dict[str, str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run without a terminal (no stdin, no controlling tty), as in CI."""
    return subprocess.run(
        command,
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_S,
        start_new_session=True,
    )


def minimal_elf(*needed: str) -> bytes:
    """An x86-64 ELF shared object whose dynamic section NEEDs ``needed``: all ``readelf -d``
    reads."""
    strtab, offsets = b"\0", []
    for lib in needed:
        offsets.append(len(strtab))
        strtab += lib.encode() + b"\0"
    strtab_at = EHDR_SIZE + 2 * PHDR_SIZE
    dynamic_at = -(-(strtab_at + len(strtab)) // 8) * 8
    entries = [(DT_NEEDED, off) for off in offsets]
    entries += [(DT_STRTAB, strtab_at), (DT_STRSZ, len(strtab)), (DT_NULL, 0)]
    dynamic = b"".join(struct.pack("<qQ", tag, value) for tag, value in entries)
    size = dynamic_at + len(dynamic)
    ident = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)  # 64-bit, little endian, version 1
    header = ident + struct.pack(
        "<HHIQQQIHHHHHH", 3, 62, 1, 0, EHDR_SIZE, 0, 0, EHDR_SIZE, PHDR_SIZE, 2, 64, 0, 0
    )
    load = struct.pack("<IIQQQQQQ", PT_LOAD, 4, 0, 0, 0, size, size, 0x1000)
    dyn = struct.pack(
        "<IIQQQQQQ", PT_DYNAMIC, 6, dynamic_at, dynamic_at, dynamic_at, *[len(dynamic)] * 2, 8
    )
    body = header + load + dyn + strtab
    return body + bytes(dynamic_at - len(body)) + dynamic
