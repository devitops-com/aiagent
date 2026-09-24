"""Shared by the packaging, installer and release tests: run the project's scripts hermetically."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PATH = "/usr/bin:/bin"
SCRIPT_TIMEOUT_S = 60.0


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
