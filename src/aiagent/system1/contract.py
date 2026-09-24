"""aiagent <-> devai contract primitives: hashing, safe ids, SHA256SUMS (stdlib only).

``H(x)`` (:func:`content_sha256`) is the sha256 of ``x``'s compact JSON: key order kept,
non-ASCII written literally, NaN/inf rejected. Every ``*_sha256`` is 64 lowercase hex
chars with no prefix. Ids and relpaths that become path components are checked here,
before anything joins them into a path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

SHA256_HEX: Final = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SUMS_FILE: Final = "SHA256SUMS"

_CHUNK: Final = 1 << 20  # 1 MiB
_SEPARATORS: Final = ("  ", " *")  # sha256sum's text and binary forms


def canonical_json(obj: object) -> str:
    """Compact JSON (",", ":"), non-ASCII literal, NaN/inf rejected, key order kept."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_sha256(obj: object) -> str:
    """H(x): sha256 hex of canonical_json(x) as UTF-8."""
    return bytes_sha256(canonical_json(obj).encode("utf-8"))


def bytes_sha256(data: bytes) -> str:
    """sha256 hex of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    """sha256 hex of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value: str, what: str) -> str:
    """Return value if it matches SAFE_ID, else raise ValueError naming `what`."""
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe {what} {value!r}: must match {SAFE_ID.pattern}")
    return value


def safe_relpath(value: str) -> PurePosixPath:
    """A relative POSIX path with no '..', empty or absolute parts, else ValueError."""
    parts = value.split("/")
    if (
        not value
        or value.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"unsafe relative path {value!r}")
    return PurePosixPath(value)


def _check_sha256(value: str, what: str) -> str:
    if not SHA256_HEX.fullmatch(value):
        raise ValueError(f"{what}: not a lowercase sha256 hex digest: {value!r}")
    return value


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse '<hex>  <path>' / '<hex> *<path>' lines.

    ValueError on a bad line, bad hex, an unsafe path or a duplicate path.
    """
    entries: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{SUMS_FILE} line {number}"
        if line[64:66] not in _SEPARATORS:
            raise ValueError(f"{where}: expected '<sha256>  <path>', got {line!r}")
        digest, path = _check_sha256(line[:64], where), line[66:]
        try:
            safe_relpath(path)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if path in entries:
            raise ValueError(f"{where}: duplicate path {path!r}")
        entries[path] = digest
    return entries


def format_sha256sums(entries: Mapping[str, str]) -> str:
    """Sorted '<hex>  <path>\\n' lines."""
    lines = []
    for path in sorted(entries):
        safe_relpath(path)
        lines.append(f"{_check_sha256(entries[path], path)}  {path}\n")
    return "".join(lines)


def dataset_id(manifest_sha256: str) -> str:
    """'ds-' + the first 12 hex chars."""
    return "ds-" + _check_sha256(manifest_sha256, "dataset manifest")[:12]


@dataclass(frozen=True)
class Binds:
    """The hashes that bind a trained student to a skill's predictor."""

    signature_sha256: str
    question_set_sha256: str
    skill_source_sha256: str

    def to_json(self) -> dict[str, str]:
        """The three fields, in this order."""
        return {
            "signature_sha256": self.signature_sha256,
            "question_set_sha256": self.question_set_sha256,
            "skill_source_sha256": self.skill_source_sha256,
        }
