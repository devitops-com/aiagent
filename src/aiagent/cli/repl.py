"""Interactive line editing for the ``aiagent chat`` REPL.

Import-safe (no ``dspy``): wires the stdlib :mod:`readline` for inline editing,
Up/Down history, ``Ctrl-R`` reverse search, and persistent per-session history
(issue #7), and adds Tab completion of the REPL's ``:`` directives plus optional
`fzf <https://github.com/junegunn/fzf>`_ vertical pickers for history and
commands (issue #8).

``fzf`` is optional: when it is not on ``PATH`` the pickers degrade to ``None``
so the caller can fall back to the plain readline path. ``readline`` itself is
imported defensively — a Python built without it (some portable bundles) simply
loses editing/history while ``input()`` keeps working.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - only ever execs the fixed ``fzf`` binary, no shell
from collections.abc import Iterable, Sequence
from pathlib import Path

try:  # pragma: no cover - exercised via the has-readline path in tests
    import readline

    _HAVE_READLINE = True
except ImportError:  # pragma: no cover - portable builds without libreadline
    _HAVE_READLINE = False

# Keep the on-disk history bounded so a long-lived config dir doesn't grow without end.
_MAX_HISTORY = 1000


def fzf_available() -> bool:
    """Whether the ``fzf`` binary is on ``PATH``."""
    return shutil.which("fzf") is not None


def fzf_pick(candidates: Sequence[str], *, prompt: str = "> ") -> str | None:
    """Let the user pick one of ``candidates`` in an ``fzf`` vertical menu.

    Returns the chosen line, or ``None`` when there is nothing to pick, ``fzf``
    is unavailable, or the user aborts (Esc / ``Ctrl-C``). ``fzf`` reads the list
    from stdin and draws its UI on the controlling tty, so only stdout — the
    selection — is captured.
    """
    if not candidates:
        return None
    try:
        completed = subprocess.run(  # nosec B603 B607 - fixed argv, no shell, no user-controlled binary
            ["fzf", "--height=40%", "--reverse", "--prompt", prompt],
            input="\n".join(candidates),
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if completed.returncode != 0:  # 1 = no match, 130 = aborted
        return None
    choice = completed.stdout.strip()
    return choice or None


def complete_command(text: str, commands: Sequence[str]) -> list[str]:
    """Prefix-match ``text`` against the REPL's ``:`` directives."""
    return [c for c in commands if c.startswith(text)]


def _dedup_keep_order(items: Iterable[str]) -> list[str]:
    """De-duplicate ``items`` keeping first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class PromptReader:
    """Reads chat prompts with editing, history, completion, and fzf pickers.

    ``history_path`` persists user prompts across ``chat`` invocations; ``seed``
    (e.g. a resumed session's prompts) is layered on top so Up/Down and the fzf
    history picker surface this session's turns immediately. ``commands`` are the
    ``:`` directives offered by Tab completion and the command palette.
    """

    def __init__(
        self,
        *,
        history_path: Path,
        commands: Sequence[str],
        seed: Sequence[str] = (),
        use_fzf: bool | None = None,
    ) -> None:
        self._history_path = history_path
        self._commands = list(commands)
        self._use_fzf = fzf_available() if use_fzf is None else use_fzf
        self._prompts = _dedup_keep_order([*self._load_persisted(), *seed])
        self._install_readline()

    # -- history -----------------------------------------------------------

    def _load_persisted(self) -> list[str]:
        try:
            text = self._history_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        return [line for line in text.splitlines() if line]

    def remember(self, text: str) -> None:
        """Record a submitted prompt in history (in-memory and readline)."""
        self._prompts = _dedup_keep_order([*self._prompts, text])
        if _HAVE_READLINE:
            readline.add_history(text)

    def save(self) -> None:
        """Persist history (most-recent last, capped) to disk. Best-effort."""
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            recent = self._prompts[-_MAX_HISTORY:]
            self._history_path.write_text(
                "\n".join(recent) + ("\n" if recent else ""), encoding="utf-8"
            )
        except OSError:
            pass

    # -- readline wiring ---------------------------------------------------

    def _install_readline(self) -> None:
        if not _HAVE_READLINE:
            return
        for prompt in self._prompts:
            readline.add_history(prompt)
        # Complete the whole line as one token so ``:`` directives match cleanly.
        readline.set_completer_delims("")
        readline.set_completer(self._readline_completer)
        readline.parse_and_bind("tab: complete")

    def _readline_completer(self, text: str, state: int) -> str | None:
        matches = complete_command(text, self._commands) if text.startswith(":") else []
        return matches[state] if state < len(matches) else None

    # -- reading -----------------------------------------------------------

    def read(self, prompt: str, *, prefill: str = "") -> str:
        """Read a line, optionally pre-filling the editable buffer with ``prefill``."""
        if prefill and not _HAVE_READLINE:
            # No editing available; hand the picked line straight back.
            return prefill
        if prefill:

            def _hook() -> None:
                readline.insert_text(prefill)
                readline.redisplay()

            readline.set_pre_input_hook(_hook)
            try:
                return input(prompt)
            finally:
                readline.set_pre_input_hook(None)
        return input(prompt)

    # -- fzf pickers -------------------------------------------------------

    @property
    def fzf_enabled(self) -> bool:
        """Whether fzf-backed pickers are active."""
        return self._use_fzf

    def pick_history(self) -> str | None:
        """Pick a past prompt via fzf (most-recent first). ``None`` if unavailable."""
        if not self._use_fzf:
            return None
        return fzf_pick(list(reversed(self._prompts)), prompt="history> ")

    def pick_command(self) -> str | None:
        """Pick a ``:`` directive via fzf. ``None`` if unavailable."""
        if not self._use_fzf:
            return None
        return fzf_pick(self._commands, prompt="command> ")
