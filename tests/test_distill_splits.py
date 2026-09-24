"""Document-hash splits: ids, buckets, boundaries and the distribution."""

from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from aiagent.distill.splits import (
    LABELED_SPLITS,
    SPLIT_BOUNDS,
    SPLIT_METHOD,
    SPLITS,
    assign_split,
    bucket,
    document_id,
)


def test_constants() -> None:
    assert SPLITS == ("train", "calib", "heldout", "pool")
    assert LABELED_SPLITS == ("train", "calib", "heldout")
    assert SPLIT_BOUNDS == (("train", 60), ("calib", 70), ("heldout", 85), ("pool", 100))
    assert SPLIT_METHOD == "sha256(group_id) mod 100: train <60, calib <70, heldout <85, pool <100"


def test_document_id_format() -> None:
    doc = document_id("Die Lieferung kam zu spät.")
    assert doc == "sha256:" + hashlib.sha256("Die Lieferung kam zu spät.".encode()).hexdigest()
    assert len(doc) == 7 + 64
    assert document_id("a") != document_id("a ")


def test_bucket_is_pinned() -> None:
    group_id = document_id("hello world")
    assert bucket(group_id) == int(hashlib.sha256(group_id.encode()).hexdigest(), 16) % 100
    assert bucket(group_id) == 31  # computed once; the split of every document hangs on it
    assert assign_split(group_id) == "train"


def _id_in_bucket(wanted: int) -> str:
    for i in range(100_000):
        group_id = document_id(f"doc {i}")
        if bucket(group_id) == wanted:
            return group_id
    raise AssertionError(f"no id in bucket {wanted}")  # pragma: no cover


@pytest.mark.parametrize(
    ("value", "split"),
    [(0, "train"), (59, "train"), (60, "calib"), (69, "calib"), (70, "heldout"),
     (84, "heldout"), (85, "pool"), (99, "pool")],
)
def test_boundaries(value: int, split: str) -> None:
    assert assign_split(_id_in_bucket(value)) == split


def test_distribution_is_about_60_10_15_15() -> None:
    counts = Counter(assign_split(document_id(f"document number {i}")) for i in range(10_000))
    for split, share in (("train", 0.60), ("calib", 0.10), ("heldout", 0.15), ("pool", 0.15)):
        assert abs(counts[split] / 10_000 - share) <= 0.02, (split, counts)
