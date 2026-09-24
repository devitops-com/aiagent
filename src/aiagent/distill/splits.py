"""Document-hash splits: train / calib / heldout / pool by ``sha256(group_id) mod 100``.

Every segment of a document shares its ``group_id``, so a document never straddles
two splits. Held-out and calib are frozen across rounds: repair copies them verbatim
and never re-splits.
"""

from __future__ import annotations

import hashlib
from typing import Final, Literal

Split = Literal["train", "calib", "heldout", "pool"]
SPLITS: Final[tuple[Split, ...]] = ("train", "calib", "heldout", "pool")
LABELED_SPLITS: Final[tuple[Split, ...]] = ("train", "calib", "heldout")
SPLIT_BOUNDS: Final[tuple[tuple[Split, int], ...]] = (
    ("train", 60),
    ("calib", 70),
    ("heldout", 85),
    ("pool", 100),
)
SPLIT_METHOD: Final = (
    "sha256(group_id) mod 100: train <60, calib <70, heldout <85, pool <100"
)


def document_id(text: str) -> str:
    """'sha256:' + sha256(text UTF-8) hex."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def bucket(group_id: str) -> int:
    """int(sha256(group_id UTF-8) hex, 16) % 100."""
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest(), 16) % 100


def assign_split(group_id: str) -> Split:
    """The first SPLIT_BOUNDS entry whose bound exceeds bucket()."""
    value = bucket(group_id)
    return next(split for split, bound in SPLIT_BOUNDS if value < bound)
