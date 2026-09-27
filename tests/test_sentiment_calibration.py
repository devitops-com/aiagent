"""tools/system1/sentiment_calibration.py: PR 3's analysis of a sentiment shadow log (design §2.9).

A hand-made log with known numbers: LEVEL, its SE, σ_w and σ_b over the accepted neutral
segments, counted per position (a segment repeated within a document too); coverage, the
too_long share and coverage by n_tokens; the pass test ((a) band agreement with its one-sided
Clopper-Pearson bound, (b) the per-document mean drift Δ and its nearest-rank 95th percentile,
at the log's own level or at a given --level); incomplete runs, duplicate documents, unreadable
samples and torn (not JSON) lines, and a first run's log joined to its rerun's as the runbook
does; the τ sweep; the sidecar slices, with (b) per slice, joined on doc_sha256 or on doc_id
"sha256:<hex>"; the model the block pins, from the analysed lines' `model` field (without its
"@<ctx>") or, for those that have none (a 0.7.0 log, alone or joined with a newer one), from
--model, which must be a model string an LM reports; --pin-tau, which calibrates and tests at
its own τ (the block is offered only with it), and with --level the confirmation run's verdict;
and the pasted NEUTRAL block, executed against the real NeutralCalibration and kept within
ruff's line length whatever `measured` holds.

The tool is loaded from its file. Most cases call its ``main(argv)`` in process: it imports
aiagent.core.sentiment (dspy), so a subprocess per case would cost over a second each. One test
runs it as the lab does, a script in a subprocess.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from aiagent.core.sentiment import SCORE_SIGNATURE_SHA256, NeutralCalibration
from aiagent.distill.gates import clopper_pearson_lower
from helpers_scripts import ROOT, run_script

TOOL = ROOT / "tools" / "system1" / "sentiment_calibration.py"
ARTIFACT = "a866e0a4734f66ddb975ac1d2e41780c8961913cf6fa738ef53b6bc7b843e273"
INSTALLED_TAU = 0.894  # what the hand-made would_accept values follow
MODEL = "openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink"  # as NEUTRAL.model names it
SERVED = f"{MODEL}@118784"  # as the log records it: with the context window
THINKING = "openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::think"
MAX_LINE = 88  # ruff's line length, for the pasted block
PASTE_HEADER = "Paste into src/aiagent/core/sentiment.py"
# The installed τ, pinned: on these lines the same acceptance as would_accept.
PIN = ("--pin-tau", str(INSTALLED_TAU))


@pytest.fixture(scope="module")
def tool() -> Iterator[ModuleType]:
    spec = importlib.util.spec_from_file_location("sentiment_calibration", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look up their module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def sha(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def line(
    run: str,
    doc: str,
    index: int,
    n_segments: int,
    samples: Sequence[float | None],
    *,
    student: str | None = "neutral",
    confidence: float | None = 0.95,
    fits: bool = True,
    n_tokens: int = 40,
    artifact: str = ARTIFACT,
    model: str | None = SERVED,
) -> dict[str, Any]:
    """One shadow-log line as sentiment writes it (§2.4); would_accept at INSTALLED_TAU. A
    `model` of None leaves the field out, as 0.7.0 did."""
    if not fits:
        student = confidence = None
    return {
        "ts": f"2026-09-27T10:{index:02d}:00Z",
        "artifact_id": artifact,
        **({} if model is None else {"model": model}),
        "run_id": run,
        "seg_index": index,
        "n_segments": n_segments,
        "doc_sha256": sha(doc),
        "input_sha256": sha(f"{doc}/{index}"),
        "n_tokens": n_tokens,
        "fits": fits,
        "student": student,
        "confidence": confidence,
        "would_accept": bool(
            fits and student == "neutral" and confidence is not None
            and confidence >= INSTALLED_TAU
        ),
        "llm_samples": list(samples),
        "student_ms": 80,
    }


# Three documents, seven segments; four accepted at the installed τ, marked A.
BASE = [
    line("r1", "d1", 0, 3, [0.0, 1.0, 2.0], confidence=0.95, n_tokens=40),  # A: ȳ 1, s² 1
    line("r1", "d1", 1, 3, [5.0, 6.0, 7.0], student="positive", confidence=0.99, n_tokens=70),
    line("r1", "d1", 2, 3, [4.0, None, 4.0], fits=False, n_tokens=1500),  # too long, ȳ 4
    line("r2", "d2", 0, 2, [-1.0, -1.0, -1.0], confidence=0.91, n_tokens=20),  # A: ȳ −1, s² 0
    line("r2", "d2", 1, 2, [3.0, None, None], confidence=0.93, n_tokens=100),  # A: ȳ 3, k 1
    line("r3", "d3", 0, 2, [2.0, 2.0, 5.0], confidence=0.97, n_tokens=50),  # A: ȳ 3, s² 3
    line("r3", "d3", 1, 2, [0.0, 0.0, 1.0], confidence=0.88, n_tokens=30),  # neutral below τ
]
# The accepted ȳ are 1, −1, 3, 3 with k 3, 3, 1, 3.
LEVEL = 1.5
VAR_MEANS = 11 / 3  # (0.25 + 6.25 + 2.25 + 2.25) / 3
SIGMA_W2 = 4 / 3  # mean(1, 0, 3): the three accepted lines with k ≥ 2
MEAN_INV_K = 0.5  # mean(1/3, 1/3, 1, 1/3)
SIGMA_B2 = VAR_MEANS - SIGMA_W2 * MEAN_INV_K  # 3
# Δ = mean with LEVEL at the accepted positions − mean of ȳ:
# d1 (1, 6, 4): 0.5/3; d2 (−1, 3): 0.5; d3 (3, 1/3): (1.5 − 3)/2.
DELTAS = [1 / 6, 0.5, -0.75]


def without_model(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records as aiagent 0.7.0 logged them: no `model` field."""
    return [{k: v for k, v in record.items() if k != "model"} for record in records]


def one_segment_documents(means: Sequence[float], prefix: str = "g") -> list[dict[str, Any]]:
    """One accepted segment per document, all three samples at `mean`."""
    return [line(f"{prefix}{i}", f"{prefix}{i}", 0, 1, [m, m, m]) for i, m in enumerate(means)]


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def analyse(
    tool: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    records: Sequence[dict[str, Any]],
    *args: str,
) -> tuple[int, str, dict[str, Any]]:
    """main() on `records` as a log: (exit code, stdout, the --json summary)."""
    log = write_jsonl(tmp_path / "shadow.jsonl", records)
    out = tmp_path / "summary.json"
    code = tool.main([str(log), "--json", str(out), *args])
    stdout = capsys.readouterr().out
    return code, stdout, json.loads(out.read_text(encoding="utf-8"))


def pasted(stdout: str) -> str:
    """The ready-to-paste block: everything after its header line."""
    head, _, block = stdout.partition(PASTE_HEADER)
    assert block, stdout
    return block.split("\n", 1)[1]


def comment(block: str) -> str:
    """The block's comment as one line of text: unwrapped."""
    return " ".join(row.removeprefix("# ") for row in block.splitlines() if row.startswith("#"))


def test_level_se_and_the_variance_components_of_the_accepted_segments(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, _, summary = analyse(tool, capsys, tmp_path, BASE)

    installed = summary["installed"]
    assert installed["tau"] is None
    assert installed["n"] == 4
    assert installed["level"] == pytest.approx(LEVEL)
    assert installed["var_means"] == pytest.approx(VAR_MEANS)
    assert installed["se"] == pytest.approx(math.sqrt(VAR_MEANS / 4))
    assert installed["n_within"] == 3
    assert installed["sigma_within"] == pytest.approx(math.sqrt(SIGMA_W2))
    assert installed["mean_inv_k"] == pytest.approx(MEAN_INV_K)
    assert installed["sigma_between"] == pytest.approx(math.sqrt(SIGMA_B2))
    assert SIGMA_B2 == pytest.approx(3.0)
    assert summary["artifact_id"] == ARTIFACT
    assert summary["score_signature_sha256"] == SCORE_SIGNATURE_SHA256
    assert summary["installed_tau"] == {"above": 0.88, "at_most": 0.91}
    assert code == 1  # (b) has 3 documents, not 100


def test_coverage_the_too_long_share_and_coverage_by_n_tokens(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, stdout, summary = analyse(tool, capsys, tmp_path, BASE)

    coverage = summary["installed"]["coverage"]
    assert coverage["lines"] == 7
    assert coverage["accepted"] == 4
    assert coverage["coverage"] == pytest.approx(4 / 7)
    assert coverage["too_long"] == 1
    assert coverage["too_long_share"] == pytest.approx(1 / 7)
    bins = {b["bin"]: (b["lines"], b["accepted"], b["too_long"]) for b in coverage["by_n_tokens"]}
    assert bins == {"0-31": (2, 1, 0), "32-63": (2, 2, 0), "64-127": (2, 1, 0), "1024+": (1, 0, 1)}
    assert [b["bin"] for b in coverage["by_n_tokens"]] == ["0-31", "32-63", "64-127", "1024+"]
    assert "coverage 4/7 = 57.1%" in stdout
    assert "too_long 1/7 = 14.3%" in stdout


def test_band_agreement_is_a_one_sided_clopper_pearson_bound(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """ȳ 1 and −1 agree, 3 and 3 do not: 2 of 4, bound Beta.ppf(0.05, 2, 3) = 0.0976 (the
    two-sided bound at α 0.05 would be Beta.ppf(0.025, 2, 3) = 0.0676)."""
    _, stdout, summary = analyse(tool, capsys, tmp_path, BASE)

    agreement = summary["installed"]["agreement"]
    assert (agreement["n"], agreement["agree"]) == (4, 2)
    assert agreement["bound"] == pytest.approx(0.0976, abs=1e-4)
    assert agreement["bound"] == clopper_pearson_lower(2, 4, 0.05)
    assert agreement["passes"] is False
    assert "(a) band agreement: 2/4 in (-2, +2), CP lower bound 0.0976" in stdout


@pytest.mark.parametrize(("n", "passes"), [(29, True), (28, False)])
def test_agreement_needs_29_accepted_segments_without_a_disagreement(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, n: int, passes: bool
) -> None:
    """Design §2.9: with k = n the bound is α^(1/n), 0.9019 at 29 and 0.8985 at 28."""
    _, _, summary = analyse(tool, capsys, tmp_path, one_segment_documents([0.0] * n))

    agreement = summary["installed"]["agreement"]
    assert agreement["agree"] == n
    assert agreement["bound"] == pytest.approx(0.05 ** (1 / n), abs=1e-9)
    assert agreement["passes"] is passes


def test_the_agreement_band_is_open(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    records = one_segment_documents([-2.0, -1.999, 1.999, 2.0])

    _, _, summary = analyse(tool, capsys, tmp_path, records)

    assert summary["installed"]["agreement"]["agree"] == 2


def test_a_segment_repeated_within_a_document_counts_at_each_position(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Sentiment scores a repeated segment once and logs it at each position, with the same
    samples and verdict (design §4, "PR 2 as built"); summarize counts it per position, so
    LEVEL, n and Δ do too: no deduplication by input_sha256."""
    repeated = {"input_sha256": sha("the same segment")}
    records = [
        {**line("p1", "rep", 0, 3, [0.0, 0.0, 0.0]), **repeated},  # A: ȳ 0
        line("p1", "rep", 1, 3, [6.0, 6.0, 6.0], student="positive", confidence=0.99),
        {**line("p1", "rep", 2, 3, [0.0, 0.0, 0.0]), **repeated, "student_ms": 0},  # A: ȳ 0
        line("p2", "one", 0, 1, [3.0, 3.0, 3.0]),  # A: ȳ 3
    ]

    _, _, summary = analyse(tool, capsys, tmp_path, records)

    installed = summary["installed"]
    assert installed["n"] == 3  # ȳ 0, 0, 3: per position
    assert installed["level"] == pytest.approx(1.0)
    assert installed["sigma_within"] == 0.0
    assert installed["sigma_between"] == pytest.approx(math.sqrt(3.0))  # var(0, 0, 3)
    assert installed["coverage"]["accepted"] == 3
    # rep: off (0 + 6 + 0)/3 = 2, gate (1 + 6 + 1)/3; one: off 3, gate 1.
    deltas = {d["run_id"]: d["delta"] for d in installed["drift"]["deltas"]}
    assert deltas == {"p1": pytest.approx(2 / 3), "p2": pytest.approx(-2.0)}


def test_mean_drift_per_document_with_the_pinned_level(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, stdout, summary = analyse(tool, capsys, tmp_path, BASE)

    drift = summary["installed"]["drift"]
    assert drift["level"] == LEVEL
    assert [(d["run_id"], d["doc_sha256"]) for d in drift["deltas"]] == [
        ("r1", sha("d1")),
        ("r2", sha("d2")),
        ("r3", sha("d3")),
    ]
    assert [d["delta"] for d in drift["deltas"]] == pytest.approx(DELTAS)
    assert drift["documents"] == 3
    assert drift["skipped"] == 0
    assert drift["mean_delta"] == pytest.approx(sum(DELTAS) / 3)  # −1/36
    assert drift["p95_abs_delta"] == pytest.approx(0.75)  # rank ⌈0.95·3⌉ = 3
    assert drift["passes"] is False
    assert "(b) mean drift: 3 documents (need ≥ 100), p95 |Δ| 0.7500" in stdout
    assert "mean Δ -0.0278" in stdout


def test_level_computes_the_drift_at_a_given_pin(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """--level 0: Δ with 0 at the accepted positions. d1 (1, 6, 4) -> (0, 6, 4): −1/3; d2
    (−1, 3) -> (0, 0): −1; d3 (3, 1/3) -> (0, 1/3): −1.5. The log's own LEVEL is still
    reported, and nothing is offered to pin: the run evaluates a pin, it calibrates none."""
    code, stdout, summary = analyse(tool, capsys, tmp_path, BASE, "--level", "0")

    installed = summary["installed"]
    assert summary["given_level"] == 0.0
    assert installed["level"] == pytest.approx(LEVEL)
    drift = installed["drift"]
    assert drift["level"] == 0.0
    assert [d["delta"] for d in drift["deltas"]] == pytest.approx([-1 / 3, -1.0, -1.5])
    assert drift["mean_delta"] == pytest.approx(-17 / 18)
    assert drift["p95_abs_delta"] == pytest.approx(1.5)
    sweep = {row["tau"]: row for row in summary["sweep"]}
    assert sweep[0.92]["drift"]["level"] == 0.0  # every τ: the given level, not 2.33
    assert summary["neutral_calibration"] is None
    assert "--level" in summary["cannot_pin"]
    assert "(b) mean drift at the given level 0: 3 documents" in stdout
    assert "cannot pin" in pasted(stdout)
    assert "NeutralCalibration(" not in pasted(stdout)
    assert code == 1


def test_a_review_log_at_the_docs_pin_is_not_fitted_to_itself(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """D3: how much a review that goes through the docs-calibrated gate would shift. Fitted
    to itself, a log whose accepted segments sit at ȳ 1.5 drifts by 0; at the docs pin 0.2
    every document moves by −1.3, and (b) fails."""
    records = one_segment_documents([1.5] * 100)

    fitted_code, _, fitted = analyse(tool, capsys, tmp_path, records)
    code, _, summary = analyse(tool, capsys, tmp_path, records, "--level", "0.2")

    assert fitted["installed"]["drift"]["p95_abs_delta"] == pytest.approx(0.0, abs=1e-12)
    assert fitted_code == 0
    drift = summary["installed"]["drift"]
    assert drift["mean_delta"] == pytest.approx(-1.3)
    assert drift["p95_abs_delta"] == pytest.approx(1.3)
    assert drift["passes"] is False
    assert code == 1


@pytest.mark.parametrize("level", ["x", "nan", "inf", "10.5", "-11"])
def test_level_must_be_a_score(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, level: str
) -> None:
    log = write_jsonl(tmp_path / "shadow.jsonl", BASE)

    with pytest.raises(SystemExit) as exc:
        tool.main([str(log), "--level", level])

    assert exc.value.code == 2
    assert "--level" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("zeros", "ones", "p95", "passes"),
    [(95, 5, 0.05, True), (94, 6, 0.94, False), (99, 0, 0.0, False)],
    ids=["rank-95-is-small", "rank-95-is-an-outlier", "99-documents"],
)
def test_p95_is_the_nearest_rank_over_at_least_100_documents(
    tool: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    zeros: int,
    ones: int,
    p95: float,
    passes: bool,
) -> None:
    """One segment per document: Δ = LEVEL − ȳ. With 95 at ȳ 0 and 5 at ȳ 1, LEVEL is 0.05 and
    the 95th smallest |Δ| is 0.05 (linear interpolation would give 0.095); with 94 and 6 it is
    0.94. 99 documents fail whatever their drift."""
    records = one_segment_documents([0.0] * zeros + [1.0] * ones)

    code, stdout, summary = analyse(tool, capsys, tmp_path, records)

    drift = summary["installed"]["drift"]
    assert drift["documents"] == zeros + ones
    assert drift["p95_abs_delta"] == pytest.approx(p95)
    assert drift["mean_delta"] == pytest.approx(0.0, abs=1e-12)
    assert drift["passes"] is passes
    assert summary["installed"]["agreement"]["passes"] is True
    assert summary["installed"]["passes"] is passes
    assert code == (0 if passes else 1)
    verdict = "PASS" if passes else "FAIL"
    assert any(row.startswith("  (b) mean drift") and row.endswith(verdict) for row in stdout.splitlines())


def test_incomplete_runs_are_reported_and_left_out(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    broken = [
        line("r4", "d4", 0, 3, [9.0, 9.0, 9.0]),  # seg_index 2 missing
        line("r4", "d4", 1, 3, [9.0, 9.0, 9.0]),
        line("r5", "d5", 0, 2, [9.0, 9.0, 9.0]),  # seg_index 0 twice
        line("r5", "d5", 0, 2, [9.0, 9.0, 9.0]),
        line("r8", "d8", 0, 2, [9.0, 9.0, 9.0]),  # one run_id, two documents
        line("r8", "d9", 1, 2, [9.0, 9.0, 9.0]),
    ]

    _, stdout, summary = analyse(tool, capsys, tmp_path, BASE + broken)

    runs = summary["runs"]
    assert (runs["read"], runs["complete"], runs["analysed"]) == (6, 3, 3)
    incomplete = {r["run_id"]: r for r in runs["incomplete"]}
    assert set(incomplete) == {"r4", "r5", "r8"}
    assert (incomplete["r4"]["lines"], incomplete["r4"]["n_segments"]) == (2, 3)
    assert incomplete["r4"]["doc_sha256"] == sha("d4")
    assert "2 of 3" in incomplete["r4"]["problem"]
    assert "repeats" in incomplete["r5"]["problem"]
    assert "disagree" in incomplete["r8"]["problem"]
    assert summary["lines"]["read"] == 13
    assert summary["lines"]["analysed"] == 7
    assert summary["installed"]["level"] == pytest.approx(LEVEL)
    assert summary["installed"]["coverage"]["lines"] == 7
    assert "r4" in stdout and "r5" in stdout


def test_a_duplicate_document_keeps_its_first_complete_run(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A re-run of d2 (r6) is left out; so is an incomplete run of d3 (r0) that came first,
    without making the complete r3 a duplicate."""
    rerun = [
        line("r6", "d2", 0, 2, [9.0, 9.0, 9.0], confidence=0.99),
        line("r6", "d2", 1, 2, [9.0, 9.0, 9.0], confidence=0.99),
    ]
    first_incomplete = [line("r0", "d3", 0, 2, [9.0, 9.0, 9.0])]

    _, stdout, summary = analyse(tool, capsys, tmp_path, first_incomplete + BASE + rerun)

    assert summary["runs"]["duplicates"] == [
        {"doc_sha256": sha("d2"), "run_id": "r6", "kept_run_id": "r2"}
    ]
    assert [r["run_id"] for r in summary["runs"]["incomplete"]] == ["r0"]
    assert summary["runs"]["analysed"] == 3
    assert summary["installed"]["n"] == 4
    assert summary["installed"]["level"] == pytest.approx(LEVEL)
    assert "r6" in stdout


def torn(record: dict[str, Any]) -> str:
    """A record cut short, as a process killed during its write leaves it: no newline."""
    return json.dumps(record)[:150]


def test_an_unreadable_line_is_skipped_and_reported(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A line that is not JSON (torn by a killed process) is left out and named; the rest is
    analysed as usual. Its run lacks that line, so it is incomplete."""
    first, second = line("r4", "d4", 0, 2, [9.0] * 3), line("r4", "d4", 1, 2, [9.0] * 3)
    log = tmp_path / "shadow.jsonl"
    body = [json.dumps(r) for r in BASE[:3]] + [torn(first)] + [json.dumps(r) for r in BASE[3:]]
    log.write_text("\n".join([*body, json.dumps(second)]) + "\n", encoding="utf-8")
    out = tmp_path / "summary.json"

    code = tool.main([str(log), "--json", str(out)])

    stdout = capsys.readouterr().out
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1  # the pass test (3 documents), not bad input
    assert summary["lines"]["unreadable"] == [4]
    assert summary["lines"]["read"] == 8
    assert [r["run_id"] for r in summary["runs"]["incomplete"]] == ["r4"]
    assert summary["installed"]["level"] == pytest.approx(LEVEL)
    assert "1 unreadable" in stdout
    assert "not JSON" in stdout and "line 4" in stdout


def test_a_main_log_and_its_rerun_log_joined_as_the_runbook_does(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """RUNBOOK §7: the first run's log, then the rerun's, each file ending in a newline. The
    first run was killed while writing d4 (a torn last line, no newline), and d2's row had
    failed at its explain call after logging. The rerun logs d4 whole and d2 again, from the
    cache: d2 keeps its first run, d4 comes from the rerun, and the rerun's first line
    survives the torn one."""
    main_log = [json.dumps(r) for r in BASE] + [json.dumps(line("r4", "d4", 0, 2, [0.0] * 3))]
    main_text = "\n".join(main_log) + "\n" + torn(line("r4", "d4", 1, 2, [0.0] * 3))
    rerun = [
        line("r7", "d4", 0, 2, [0.0, 0.0, 0.0]),  # A: ȳ 0
        line("r7", "d4", 1, 2, [0.0, 1.0, 2.0]),  # A: ȳ 1
        line("r6", "d2", 0, 2, [-1.0, -1.0, -1.0], confidence=0.91),  # d2 again, as r2
        line("r6", "d2", 1, 2, [3.0, None, None], confidence=0.93),
    ]
    rerun_text = "".join(json.dumps(r) + "\n" for r in rerun)
    joined = main_text + ("" if main_text.endswith("\n") else "\n") + rerun_text
    log = tmp_path / "docs-all-shadow.jsonl"
    log.write_text(joined, encoding="utf-8")
    out = tmp_path / "summary.json"

    tool.main([str(log), "--json", str(out)])

    capsys.readouterr()
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["lines"]["unreadable"] == [9]
    runs = summary["runs"]
    assert [r["run_id"] for r in runs["incomplete"]] == ["r4"]
    assert runs["duplicates"] == [{"doc_sha256": sha("d2"), "run_id": "r6", "kept_run_id": "r2"}]
    assert runs["analysed"] == 4
    installed = summary["installed"]
    assert installed["n"] == 6  # BASE's 4, and d4's 2 from the rerun
    assert installed["level"] == pytest.approx((1 - 1 + 3 + 3 + 0 + 1) / 6)
    deltas = {d["run_id"]: d["delta"] for d in installed["drift"]["deltas"]}
    assert set(deltas) == {"r1", "r2", "r3", "r7"}
    assert deltas["r2"] == pytest.approx(round(7 / 6, 2) - 1)  # d2: (L, L) against (−1, 3)


def test_lines_without_a_readable_llm_sample(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Null samples are dropped (extra rollouts included); a line with none has no ȳ: it counts
    for coverage, not for LEVEL or (a), and its document has no off-mode mean for (b)."""
    records = [
        line("n1", "e1", 0, 2, [None, None, None, 1.0]),  # accepted: 2nd extra rollout, k 1
        line("n1", "e1", 1, 2, [None, None], student="positive", confidence=0.99),  # no ȳ
        line("n2", "e2", 0, 1, []),  # accepted, no LLM sample (as gate mode logs it)
        line("n3", "e3", 0, 1, [0.0, None, 2.0]),  # accepted: ȳ 1, k 2, s² 2
    ]

    _, stdout, summary = analyse(tool, capsys, tmp_path, records)

    assert summary["lines"]["no_sample"] == 2
    assert summary["lines"]["no_sample_accepted"] == 1
    installed = summary["installed"]
    assert installed["n"] == 2
    assert installed["level"] == pytest.approx(1.0)
    assert installed["sigma_within"] == pytest.approx(math.sqrt(2.0))
    assert installed["mean_inv_k"] == pytest.approx(0.75)
    assert installed["sigma_between"] == 0.0  # max(0, 0 − 2·0.75)
    assert installed["coverage"]["accepted"] == 3
    assert installed["coverage"]["coverage"] == pytest.approx(0.75)
    assert (installed["agreement"]["n"], installed["agreement"]["agree"]) == (2, 2)
    drift = installed["drift"]
    assert (drift["documents"], drift["skipped"]) == (1, 2)
    assert drift["deltas"] == [{"run_id": "n3", "doc_sha256": sha("e3"), "delta": 0.0}]
    assert "2 with no readable LLM sample" in stdout


def test_the_tau_sweep_recomputes_acceptance_from_the_confidence(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, stdout, summary = analyse(tool, capsys, tmp_path, BASE)

    sweep = {row["tau"]: row for row in summary["sweep"]}
    assert list(sweep) == [0.90, 0.92, 0.94, 0.96, 0.98]
    got = {tau: (row["n"], row["level"], row["coverage"]["accepted"]) for tau, row in sweep.items()}
    assert got == {
        0.90: (4, pytest.approx(1.5), 4),
        0.92: (3, pytest.approx(7 / 3), 3),  # ȳ 1, 3, 3
        0.94: (2, pytest.approx(2.0), 2),
        0.96: (1, pytest.approx(3.0), 1),
        0.98: (0, None, 0),
    }
    assert sweep[0.92]["agreement"]["agree"] == 1
    assert sweep[0.92]["drift"]["level"] == 2.33
    assert sweep[0.96]["se"] is None
    assert sweep[0.98]["agreement"] == {
        "n": 0, "agree": 0, "bound": 0.0, "passes": False
    }
    assert sweep[0.98]["drift"]["documents"] == 0
    assert sweep[0.98]["passes"] is False
    rows = [row for row in stdout.splitlines() if row.lstrip().startswith(("0.9", "installed"))]
    assert [row.split()[0] for row in rows] == ["installed", "0.90", "0.92", "0.94", "0.96", "0.98"]
    assert all(row.count("FAIL") == 2 for row in rows)
    assert "(b) FAIL with fewer than 100 documents is the count, not the drift" in stdout


def test_tau_replaces_the_default_sweep_and_is_inclusive(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """0.85 takes the 0.88 line that the installed τ did not; 0.93 takes the 0.93 line."""
    _, _, summary = analyse(tool, capsys, tmp_path, BASE, "--tau", "0.85", "--tau", "0.93")

    sweep = {row["tau"]: row for row in summary["sweep"]}
    assert list(sweep) == [0.85, 0.93]
    assert sweep[0.85]["n"] == 5
    assert sweep[0.85]["level"] == pytest.approx((1 - 1 + 3 + 3 + 1 / 3) / 5)
    assert sweep[0.93]["n"] == 3


@pytest.mark.parametrize("option", ["--tau", "--pin-tau"])
@pytest.mark.parametrize("tau", ["0", "1.5", "nan", "x"])
def test_tau_must_be_in_the_unit_interval(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, tau: str, option: str
) -> None:
    log = write_jsonl(tmp_path / "shadow.jsonl", BASE)

    with pytest.raises(SystemExit) as exc:
        tool.main([str(log), option, tau])

    assert exc.value.code == 2
    assert option in capsys.readouterr().err


@pytest.mark.parametrize("key", ["doc_sha256", "doc_id"])
def test_sidecar_slices_join_on_doc_sha256_or_doc_id(
    tool: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    key: str,
) -> None:
    meta = [("d1", "wiki-en", "en"), ("d2", "wiki-de", "de"), ("d3", "reviews-en", "en")]
    sidecar = write_jsonl(
        tmp_path / "meta.jsonl",
        [
            {"line": i, key: sha(doc) if key == "doc_sha256" else f"sha256:{sha(doc)}",
             "source": source, "lang": lang}
            for i, (doc, source, lang) in enumerate(meta, start=1)
        ],
    )

    _, stdout, summary = analyse(tool, capsys, tmp_path, BASE, "--meta", str(sidecar))

    assert summary["documents_without_sidecar"] == 0
    lang = summary["slices"]["lang"]
    assert set(lang) == {"en", "de"}
    assert (lang["en"]["lines"], lang["en"]["accepted"], lang["en"]["too_long"]) == (5, 2, 1)
    assert lang["en"]["coverage"] == pytest.approx(0.4)
    assert lang["en"]["too_long_share"] == pytest.approx(0.2)
    assert (lang["en"]["agreement"]["n"], lang["en"]["agreement"]["agree"]) == (2, 1)
    en_bins = {b["bin"]: (b["lines"], b["accepted"]) for b in lang["en"]["by_n_tokens"]}
    assert en_bins == {"0-31": (1, 0), "32-63": (2, 2), "64-127": (1, 0), "1024+": (1, 0)}
    assert (lang["de"]["lines"], lang["de"]["accepted"], lang["de"]["coverage"]) == (2, 2, 1.0)
    source = summary["slices"]["source"]
    assert {v: (s["lines"], s["accepted"]) for v, s in source.items()} == {
        "reviews-en": (2, 1),
        "wiki-de": (2, 2),
        "wiki-en": (3, 1),
    }
    assert "reviews-en" in stdout and "wiki-de" in stdout
    # (b) per slice, at the installed LEVEL's pin 1.5: d1 1/6, d2 0.5, d3 −0.75.
    drift = {v: s["drift"] for v, s in lang.items()}
    assert (drift["en"]["documents"], drift["de"]["documents"]) == (2, 1)
    assert drift["en"]["mean_delta"] == pytest.approx((1 / 6 - 0.75) / 2)
    assert drift["en"]["p95_abs_delta"] == pytest.approx(0.75)  # rank ⌈0.95·2⌉ = 2
    assert drift["de"]["mean_delta"] == pytest.approx(0.5)
    assert source["reviews-en"]["drift"]["mean_delta"] == pytest.approx(-0.75)
    assert "(b) drift: 2 documents, p95 |Δ| 0.7500, mean Δ -0.2917" in stdout


def test_slices_give_the_drift_at_a_given_level(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """D3: the review run at the docs pin, read per source (here --level 0)."""
    meta = [("d1", "wiki-en"), ("d2", "reviews-de"), ("d3", "reviews-en")]
    sidecar = write_jsonl(
        tmp_path / "meta.jsonl",
        [{"doc_sha256": sha(doc), "source": source, "lang": "xx"} for doc, source in meta],
    )

    _, stdout, summary = analyse(
        tool, capsys, tmp_path, BASE, "--meta", str(sidecar), "--level", "0"
    )

    source = summary["slices"]["source"]
    assert source["wiki-en"]["drift"]["mean_delta"] == pytest.approx(-1 / 3)
    assert source["reviews-de"]["drift"]["mean_delta"] == pytest.approx(-1.0)
    assert source["reviews-en"]["drift"]["mean_delta"] == pytest.approx(-1.5)
    assert summary["slices"]["lang"]["xx"]["drift"]["documents"] == 3
    assert "(b) at the given level 0" in stdout


def test_documents_without_a_sidecar_line_are_their_own_slice(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    sidecar = write_jsonl(
        tmp_path / "meta.jsonl",
        [{"doc_id": f"sha256:{sha('d1')}", "source": "wiki-en", "lang": "en"},
         {"doc_id": f"sha256:{sha('other')}", "source": "wiki-hr", "lang": "hr"}],
    )

    _, _, summary = analyse(tool, capsys, tmp_path, BASE, "--meta", str(sidecar))

    assert summary["documents_without_sidecar"] == 2
    lang = summary["slices"]["lang"]
    assert set(lang) == {"en", "(no sidecar)"}
    assert (lang["(no sidecar)"]["lines"], lang["(no sidecar)"]["accepted"]) == (4, 3)


def test_no_sidecar_means_no_slices(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, _, summary = analyse(tool, capsys, tmp_path, BASE)

    assert summary["slices"] is None
    assert summary["documents_without_sidecar"] is None


def test_the_pasted_block_builds_the_neutral_calibration(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    measured = (
        "lab 2026-09-27, aiagent 0.7.0 shadow run at --resample 3 on 110 Wikipedia articles "
        "(en/de/hr), polarity a866e0a4, devai teacher Qwen3.8-27B"
    )

    code, stdout, summary = analyse(tool, capsys, tmp_path, BASE, "--measured", measured, *PIN)

    block = pasted(stdout)
    namespace: dict[str, Any] = {"NeutralCalibration": NeutralCalibration}
    exec(block, namespace)  # the block is this tool's own output
    expected = NeutralCalibration(
        artifact_id=ARTIFACT,
        score_signature_sha256=SCORE_SIGNATURE_SHA256,
        model=MODEL,
        tau=INSTALLED_TAU,
        level=1.5,
        se=round(math.sqrt(VAR_MEANS / 4), 4),
        sigma_between=round(math.sqrt(SIGMA_B2), 4),
        sigma_within=round(math.sqrt(SIGMA_W2), 4),
        n=4,
        measured=measured,
    )
    assert namespace["NEUTRAL"] == expected
    assert summary["neutral_calibration"] == {
        "artifact_id": ARTIFACT,
        "score_signature_sha256": SCORE_SIGNATURE_SHA256,
        "model": MODEL,
        "tau": INSTALLED_TAU,
        "level": 1.5,
        "se": 0.9574,
        "sigma_between": 1.7321,
        "sigma_within": 1.1547,
        "n": 4,
        "measured": measured,
    }
    assert summary["cannot_pin"] is None
    assert (summary["model"], summary["model_from"]) == (MODEL, "log")
    assert f"model {MODEL} (from the log)" in stdout
    assert "NEUTRAL: NeutralCalibration | None = NeutralCalibration(" in block
    assert "SCORE_SIGNATURE_SHA256" in block  # the note: the lab must have run this hash
    assert "do not pin" in block  # the pass test fails here
    assert f"Pass test at τ {INSTALLED_TAU}: (a) 2/4" in comment(block)
    assert f"    tau={INSTALLED_TAU}," in block.splitlines()
    assert "--measured" not in block  # it was given
    assert all(len(row) <= MAX_LINE for row in block.splitlines()), block
    assert code == 1


def test_a_measured_with_quotes_and_backslashes_stays_within_the_line_length(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The block goes into src/, where ruff checks 88 columns: the wrap counts the escapes
    that the literals add (\\" and \\\\), not the raw text."""
    measured = 'lab "docs" run: Qwen3.8 – τ 0.8938, résumé \\ C:\\path ' * 4 + '"' * 90

    _, stdout, _ = analyse(tool, capsys, tmp_path, BASE, "--measured", measured, *PIN)

    block = pasted(stdout)
    assert all(len(row) <= MAX_LINE for row in block.splitlines()), block
    namespace: dict[str, Any] = {"NeutralCalibration": NeutralCalibration}
    exec(block, namespace)  # the block is this tool's own output
    assert namespace["NEUTRAL"].measured == measured


def test_a_passing_log_gets_a_block_without_the_warning(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    records = one_segment_documents([0.0] * 95 + [1.0] * 5)

    code, stdout, summary = analyse(tool, capsys, tmp_path, records, *PIN)

    block = pasted(stdout)
    assert code == 0
    assert "do not pin" not in block
    assert summary["neutral_calibration"]["level"] == 0.05
    assert summary["neutral_calibration"]["n"] == 100
    assert summary["neutral_calibration"]["measured"].startswith("shadow.jsonl")
    # The default names neither the --resample nor the corpus.
    assert "--measured" in block


def test_no_block_without_resamples(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """At --resample 1 no segment has two samples, so σ_w and σ_b are unknown."""
    records = [line(f"s{i}", f"s{i}", 0, 1, [0.0]) for i in range(3)]

    _, stdout, summary = analyse(tool, capsys, tmp_path, records)

    assert summary["installed"]["sigma_within"] is None
    assert summary["installed"]["sigma_between"] is None
    assert summary["neutral_calibration"] is None
    assert "sigma_within" in summary["cannot_pin"]
    assert "cannot pin" in pasted(stdout)
    assert "NeutralCalibration(" not in pasted(stdout)


def test_without_pin_tau_nothing_is_pinned(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A calibration pins the τ it was measured and tested at (D9), and the log only brackets
    the installed τ: the block needs --pin-tau. The pass test and the exit code are the
    installed τ's."""
    records = one_segment_documents([0.0] * 95 + [1.0] * 5)

    code, stdout, summary = analyse(tool, capsys, tmp_path, records)

    assert code == 0
    assert (summary["pin_tau"], summary["pinned"], summary["confirmation"]) == (None, None, False)
    assert summary["passes"] is True
    assert summary["neutral_calibration"] is None
    assert "--pin-tau" in summary["cannot_pin"]
    assert "cannot pin" in pasted(stdout) and "--pin-tau" in pasted(stdout)
    assert "NeutralCalibration(" not in pasted(stdout)
    assert "Pass test at the installed τ" in stdout


def test_pin_tau_calibrates_and_tests_at_that_tau(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """At 0.92 BASE's accepted lines are those at 0.95, 0.93 and 0.97 (not 0.91): ȳ 1, 3, 3 with
    k 3, 1, 3. LEVEL 7/3; var(ȳ) 4/3, so SE 2/3; σ_w² mean(1, 3) = 2; mean 1/k 5/9, so σ_b² =
    4/3 − 10/9 = 2/9. (b) at round(7/3, 2) = 2.33: d1 (2.33 − 1)/3, d2 and d3 (2.33 − 3)/2."""
    code, stdout, summary = analyse(
        tool, capsys, tmp_path, BASE, "--pin-tau", "0.92", "--measured", "lab"
    )

    pinned = summary["pinned"]
    assert summary["pin_tau"] == pinned["tau"] == 0.92
    assert pinned["n"] == 3
    assert pinned["level"] == pytest.approx(7 / 3)
    assert pinned["se"] == pytest.approx(2 / 3)
    assert pinned["sigma_within"] == pytest.approx(math.sqrt(2))
    assert pinned["mean_inv_k"] == pytest.approx(5 / 9)
    assert pinned["sigma_between"] == pytest.approx(math.sqrt(2) / 3)
    assert (pinned["coverage"]["lines"], pinned["coverage"]["accepted"]) == (7, 3)
    assert (pinned["agreement"]["n"], pinned["agreement"]["agree"]) == (3, 1)
    assert pinned["agreement"]["bound"] == clopper_pearson_lower(1, 3, 0.05)
    drift = pinned["drift"]
    assert drift["level"] == 2.33
    deltas = [1.33 / 3, -0.67 / 2, -0.67 / 2]
    assert [d["delta"] for d in drift["deltas"]] == pytest.approx(deltas)
    assert drift["p95_abs_delta"] == pytest.approx(1.33 / 3)
    assert summary["installed"]["n"] == 4  # still reported
    assert (summary["passes"], code) == (False, 1)
    assert "Accepted neutral segments at τ 0.92" in stdout
    assert "  LEVEL 2.3333 over n = 3, SE 0.6667" in stdout
    assert "Pass test at τ 0.92" in stdout
    assert "  (a) band agreement: 1/3 in (-2, +2)" in stdout
    assert "  (b) mean drift: 3 documents (need ≥ 100), p95 |Δ| 0.4433" in stdout
    block = pasted(stdout)
    assert "NOT READY: the pass test fails at τ 0.92; do not pin this." in comment(block)
    assert "n = 3 accepted neutral segments at τ 0.92. Pass test at τ 0.92: (a) 1/3" in comment(
        block
    )
    namespace: dict[str, Any] = {"NeutralCalibration": NeutralCalibration}
    exec(block, namespace)  # the block is this tool's own output
    assert namespace["NEUTRAL"] == NeutralCalibration(
        artifact_id=ARTIFACT,
        score_signature_sha256=SCORE_SIGNATURE_SHA256,
        model=MODEL,
        tau=0.92,
        level=2.33,
        se=0.6667,
        sigma_between=0.4714,
        sigma_within=1.4142,
        n=3,
        measured="lab",
    )


def two_segment_documents() -> list[dict[str, Any]]:
    """100 documents: a segment the student calls neutral at 0.97 (samples −1, 0, 1: ȳ 0, s² 1)
    and one at 0.92 (ȳ 3). The installed τ takes both, and (a) fails with 100 of 200; τ 0.96
    takes the first alone: LEVEL 0, and both tests pass."""
    return [
        record
        for i in range(100)
        for record in (
            line(f"t{i}", f"t{i}", 0, 2, [-1.0, 0.0, 1.0], confidence=0.97),
            line(f"t{i}", f"t{i}", 1, 2, [3.0, 3.0, 3.0], confidence=0.92),
        )
    ]


def test_pin_tau_decides_the_verdict_and_the_exit_code(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    records = two_segment_documents()

    installed_code, _, installed = analyse(tool, capsys, tmp_path, records)
    code, stdout, summary = analyse(
        tool, capsys, tmp_path, records, "--pin-tau", "0.96", "--measured", "lab"
    )

    assert installed_code == 1
    assert installed["installed"]["agreement"]["passes"] is False
    assert code == 0
    assert summary["passes"] is True
    assert summary["installed"]["passes"] is False  # reported, but the verdict is τ 0.96's
    pinned = summary["pinned"]
    assert (pinned["n"], pinned["level"], pinned["se"]) == (100, 0.0, 0.0)
    assert (pinned["sigma_within"], pinned["sigma_between"]) == (1.0, 0.0)
    assert pinned["coverage"]["coverage"] == 0.5
    assert pinned["agreement"]["bound"] == pytest.approx(0.05 ** (1 / 100))
    assert (pinned["drift"]["documents"], pinned["drift"]["p95_abs_delta"]) == (100, 0.0)
    assert (
        "  (a) band agreement: 100/100 in (-2, +2), CP lower bound 0.9705 (α = 0.05, need ≥ "
        "0.90): PASS"
    ) in stdout.splitlines()
    rows = [row.split()[0] for row in stdout.splitlines() if row.startswith(("  0.9", "  installed"))]
    assert rows == ["installed", "0.90", "0.92", "0.94", "0.96", "0.98"]  # the sweep stays
    block = pasted(stdout)
    assert "NOT READY" not in block
    namespace: dict[str, Any] = {"NeutralCalibration": NeutralCalibration}
    exec(block, namespace)  # the block is this tool's own output
    assert namespace["NEUTRAL"] == NeutralCalibration(
        artifact_id=ARTIFACT,
        score_signature_sha256=SCORE_SIGNATURE_SHA256,
        model=MODEL,
        tau=0.96,
        level=0.0,
        se=0.0,
        sigma_between=0.0,
        sigma_within=1.0,
        n=100,
        measured="lab",
    )


@pytest.mark.parametrize(("level", "passes"), [("0.2", True), ("1.2", False)])
def test_pin_tau_with_level_is_the_confirmation_run(
    tool: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    level: str,
    passes: bool,
) -> None:
    """D9: a fresh corpus tests the pin fixed in advance, (a) at τ and (b) at τ and the given
    level: Δ = (L − 0)/2 per document, 0.1 passes and 0.6 does not. Nothing is offered to pin."""
    code, stdout, summary = analyse(
        tool, capsys, tmp_path, two_segment_documents(), "--pin-tau", "0.96", "--level", level
    )

    verdict = "PASS" if passes else "FAIL"
    assert code == (0 if passes else 1)
    assert summary["confirmation"] is True
    assert summary["passes"] is passes
    pinned = summary["pinned"]
    assert pinned["agreement"]["passes"] is True
    assert pinned["drift"]["level"] == float(level)
    assert pinned["drift"]["p95_abs_delta"] == pytest.approx(float(level) / 2)
    assert pinned["level"] == 0.0  # this log's own LEVEL, reported only
    assert summary["neutral_calibration"] is None
    assert "--pin-tau" in summary["cannot_pin"] and "--level" in summary["cannot_pin"]
    assert f"(b) mean drift at the given level {level}: 100 documents" in stdout
    assert PASTE_HEADER not in stdout
    assert "NeutralCalibration(" not in stdout
    assert stdout.rstrip().splitlines()[-1] == f"CONFIRMATION at τ 0.96, level {level}: {verdict}"


def test_slices_follow_pin_tau(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """At 0.92 d2 keeps its 0.93 line and loses its 0.91 one; (b) at the pinned 2.33."""
    meta = [("d1", "en"), ("d2", "de"), ("d3", "en")]
    sidecar = write_jsonl(
        tmp_path / "meta.jsonl",
        [{"doc_sha256": sha(doc), "source": "wiki", "lang": lang} for doc, lang in meta],
    )

    _, stdout, summary = analyse(
        tool, capsys, tmp_path, BASE, "--meta", str(sidecar), "--pin-tau", "0.92"
    )

    lang = summary["slices"]["lang"]
    assert (lang["de"]["lines"], lang["de"]["accepted"]) == (2, 1)
    assert (lang["en"]["lines"], lang["en"]["accepted"]) == (5, 2)
    assert lang["de"]["drift"]["mean_delta"] == pytest.approx(-0.67 / 2)
    assert "Slices by lang at τ 0.92" in stdout
    assert "(b) at this log's pinned level 2.33" in stdout


@pytest.mark.parametrize("option", [None, SERVED, MODEL, f"{MODEL}@4096"])
def test_the_pinned_model_is_the_logs_without_its_context_window(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, option: str | None
) -> None:
    """The log's lines carry the full model string, "@<ctx>" included, and may differ in it
    alone (a rerun with another context window); a --model that agrees once both drop their
    "@<ctx>" changes nothing."""
    records = [*BASE[:3], *(line_ | {"model": f"{MODEL}@4096"} for line_ in BASE[3:])]
    args = [*PIN] if option is None else [*PIN, "--model", option]

    _, stdout, summary = analyse(tool, capsys, tmp_path, records, *args)

    assert summary["model"] == MODEL
    assert summary["model_from"] == "log"
    assert summary["neutral_calibration"]["model"] == MODEL
    namespace: dict[str, Any] = {"NeutralCalibration": NeutralCalibration}
    exec(pasted(stdout), namespace)  # the block is this tool's own output
    assert namespace["NEUTRAL"].model == MODEL


@pytest.mark.parametrize(
    ("option", "expected"), [(SERVED, MODEL), (f" {THINKING}@4096 ", THINKING)]
)
def test_a_log_without_a_model_takes_it_from_the_option(
    tool: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    option: str,
    expected: str,
) -> None:
    """aiagent 0.7.0 logs no model: --model names the one the run used, "@<ctx>" dropped. The
    block says so, since the log alone does not show it."""
    _, stdout, summary = analyse(
        tool, capsys, tmp_path, without_model(BASE), "--model", option, *PIN
    )

    assert (summary["model"], summary["model_from"]) == (expected, "--model")
    assert summary["lines"]["no_model"] == 7
    assert summary["neutral_calibration"]["model"] == expected
    assert f"model {expected} (from --model: none of the 7 analysed lines names one)" in stdout
    block = pasted(stdout)
    assert "# model is from --model for the 7 of 7 analysed lines that name none" in block
    assert all(len(row) <= MAX_LINE for row in block.splitlines()), block
    namespace: dict[str, Any] = {"NeutralCalibration": NeutralCalibration}
    exec(block, namespace)  # the block is this tool's own output
    assert namespace["NEUTRAL"].model == expected


def test_a_log_without_a_model_and_no_option_cannot_pin(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    records = without_model(one_segment_documents([0.0] * 95 + [1.0] * 5))

    code, stdout, summary = analyse(tool, capsys, tmp_path, records, *PIN)

    assert code == 0  # the pass test passes; only the pin is missing
    assert summary["model"] is None and summary["model_from"] is None
    assert summary["lines"]["no_model"] == 100
    assert summary["neutral_calibration"] is None
    assert summary["cannot_pin"] == "no model (pass --model)"
    assert "cannot pin: no model (pass --model)" in pasted(stdout)
    assert "NeutralCalibration(" not in pasted(stdout)
    assert "model (no analysed line names one: pass --model)" in stdout


def rerun(records: Sequence[dict[str, Any]], model: str | None) -> list[dict[str, Any]]:
    """The same documents logged again by another run (new run_ids), with `model`."""
    return [
        {**{k: v for k, v in r.items() if k != "model"}, "run_id": f"again-{r['run_id']}",
         **({} if model is None else {"model": model})}
        for r in records
    ]


# A 0.7.0 run's log (no model) joined with a newer run's (with one), as RUNBOOK §7 joins a
# first run's log with its rerun's: the newer run logs d2 again (a duplicate document, so d2
# keeps its model-less first run) and a new document d4.
D4 = [line("r4", "d4", 0, 2, [0.0, 0.0, 0.0]), line("r4", "d4", 1, 2, [0.0, 1.0, 2.0])]
JOINED = [*without_model(BASE), *rerun(BASE[3:5], SERVED), *D4]


def test_a_log_whose_analysed_lines_name_a_model_only_in_part_exits_2(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Without --model, the 7 model-less lines of the 0.7.0 run would be pinned to the newer
    run's model, which nothing says they were scored with."""
    log = write_jsonl(tmp_path / "shadow.jsonl", JOINED)

    code = tool.main([str(log)])

    captured = capsys.readouterr()
    assert code == 2
    assert f"7 of its 9 analysed lines name no model and the others {MODEL}" in captured.err
    assert "--model" in captured.err
    assert captured.out == ""


def test_model_vouches_for_the_analysed_lines_that_name_none(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, stdout, summary = analyse(tool, capsys, tmp_path, JOINED, "--model", SERVED, *PIN)

    assert (summary["model"], summary["model_from"]) == (MODEL, "log and --model")
    assert summary["runs"]["duplicates"] == [
        {"doc_sha256": sha("d2"), "run_id": "again-r2", "kept_run_id": "r2"}
    ]
    assert (summary["lines"]["analysed"], summary["lines"]["no_model"]) == (9, 7)
    assert summary["neutral_calibration"]["model"] == MODEL
    assert summary["installed"]["n"] == 6  # BASE's 4, and d4's 2
    assert (
        f"model {MODEL} (from the log; from --model for the 7 of 9 analysed lines that "
        "name none)"
    ) in stdout
    block = pasted(stdout)
    assert "# model is from --model for the 7 of 9 analysed lines that name none" in block
    assert all(len(row) <= MAX_LINE for row in block.splitlines()), block


def test_model_must_still_equal_the_model_the_other_analysed_lines_name(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """One calibration holds for one model: split such a log instead."""
    log = write_jsonl(tmp_path / "shadow.jsonl", JOINED)

    code = tool.main([str(log), "--model", THINKING])

    captured = capsys.readouterr()
    assert code == 2
    assert f"scored with {MODEL}, but --model is {THINKING}" in captured.err
    assert captured.out == ""


def test_the_model_of_a_duplicate_document_left_out_does_not_count(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The review's case: every document of a 0.7.0 run logged again by a newer run. The
    0.7.0 runs are analysed (the first per document), so the newer run's model is not theirs:
    nothing is pinned without --model, and --model names theirs, whatever the newer run's."""
    records = [*without_model(BASE), *rerun(BASE, SERVED)]

    code, stdout, summary = analyse(tool, capsys, tmp_path, records, *PIN)
    thinking_code, _, thinking = analyse(
        tool, capsys, tmp_path, records, "--model", THINKING, *PIN
    )

    assert code == 1  # the pass test (3 documents), not bad input
    assert len(summary["runs"]["duplicates"]) == 3
    assert summary["model"] is None
    assert summary["neutral_calibration"] is None
    assert "model (no analysed line names one: pass --model)" in stdout
    assert thinking_code == 1
    assert (thinking["model"], thinking["model_from"]) == (THINKING, "--model")
    assert thinking["neutral_calibration"]["model"] == THINKING
    assert thinking["neutral_calibration"]["level"] == LEVEL  # the 0.7.0 runs' own


def test_lines_left_out_do_not_name_the_model(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """An incomplete run and a duplicate document of another model are not analysed, so they
    do not make the log one of several models."""
    other = f"{THINKING}@118784"
    broken = [line("r4", "d4", 0, 3, [9.0, 9.0, 9.0], model=other)]  # 1 of 3 segments

    code, _, summary = analyse(
        tool, capsys, tmp_path, [*BASE, *broken, *rerun(BASE[3:5], other)], *PIN
    )

    assert code == 1  # the pass test (3 documents), not bad input
    assert (summary["model"], summary["model_from"]) == (MODEL, "log")
    assert summary["lines"]["no_model"] == 0
    assert summary["neutral_calibration"]["model"] == MODEL


@pytest.mark.parametrize("model", [THINKING, f"{THINKING}@118784", "openai/other::mtp::nothink"])
def test_a_model_option_that_differs_from_the_logs_exits_2(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, model: str
) -> None:
    log = write_jsonl(tmp_path / "shadow.jsonl", BASE)

    code = tool.main([str(log), "--model", model])

    captured = capsys.readouterr()
    assert code == 2
    assert MODEL in captured.err and model.removesuffix("@118784") in captured.err
    assert captured.out == ""


def test_a_log_of_several_models_exits_2_naming_them(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """"::think" and "::nothink" score differently: one calibration cannot hold for both."""
    records = [*BASE[:3], *(line_ | {"model": f"{THINKING}@118784"} for line_ in BASE[3:])]
    log = write_jsonl(tmp_path / "shadow.jsonl", records)

    code = tool.main([str(log), "--model", MODEL])

    captured = capsys.readouterr()
    assert code == 2
    assert f"several models ({MODEL}, {THINKING})" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "model",
    [
        "",
        "  ",
        "@8192",  # a context window alone: nothing is left once it is dropped
        " @118784 ",
        "Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink",  # no provider, as the model is often named
        "Qwen3.8-27B-MTP-devai-NVFP4",
        "openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp",  # no reasoning
        "openai/::nothink",  # no model
        "/m::nothink",  # no provider
        "openai/m::maybe",
        "openai/m::nothink@8192@118784",  # "@8192" is left in the model's reasoning
    ],
)
def test_model_must_be_a_model_string_an_lm_reports(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path, model: str
) -> None:
    """The guard compares NEUTRAL.model with compose_model_string's
    "<provider>/<model>::<think|nothink>[@<ctx>]", context window aside: any other string would
    never match, and gate would only shadow."""
    log = write_jsonl(tmp_path / "shadow.jsonl", without_model(BASE))

    with pytest.raises(SystemExit) as exc:
        tool.main([str(log), "--model", model])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--model" in err
    assert "<provider>/<model>::<think|nothink>[@<ctx>]" in err


def bad_inputs() -> list[tuple[str, str, str | None, str]]:
    """(id, log text, sidecar text or None, what stderr names)."""
    good = json.dumps(BASE[0]) + "\n"
    other = json.dumps({**BASE[3], "artifact_id": "b" * 64}) + "\n"
    missing = json.dumps({k: v for k, v in BASE[0].items() if k != "would_accept"}) + "\n"
    ill_typed = json.dumps({**BASE[0], "llm_samples": ["4"]}) + "\n"
    ill_typed_model = json.dumps({**BASE[0], "model": 5}) + "\n"
    return [
        ("several-students", good + other, None, "b" * 64),
        ("not-an-object", good + "[1]\n", None, "line 2"),
        ("only-unreadable", "{oops\n", None, "no shadow lines (1 unreadable)"),
        ("missing-field", missing, None, "would_accept"),
        ("ill-typed-samples", ill_typed, None, "llm_samples"),
        ("ill-typed-model", ill_typed_model, None, "model must be a string"),
        ("empty-log", "\n", None, "no shadow lines"),
        ("sidecar-without-key", good, json.dumps({"source": "x"}) + "\n", "doc_id"),
        ("sidecar-bad-doc-id", good, json.dumps({"doc_id": "md5:ab"}) + "\n", "sha256:"),
    ]


@pytest.mark.parametrize(
    ("log_text", "meta_text", "named"),
    [case[1:] for case in bad_inputs()],
    ids=[case[0] for case in bad_inputs()],
)
def test_bad_input_exits_2_naming_the_problem(
    tool: ModuleType,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    log_text: str,
    meta_text: str | None,
    named: str,
) -> None:
    log = tmp_path / "shadow.jsonl"
    log.write_text(log_text, encoding="utf-8")
    argv = [str(log)]
    if meta_text is not None:
        meta = tmp_path / "meta.jsonl"
        meta.write_text(meta_text, encoding="utf-8")
        argv += ["--meta", str(meta)]

    code = tool.main(argv)

    captured = capsys.readouterr()
    assert code == 2
    assert named in captured.err
    assert captured.out == ""


def test_a_missing_log_exits_2(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert tool.main([str(tmp_path / "absent.jsonl")]) == 2
    assert "absent.jsonl" in capsys.readouterr().err


def test_an_unwritable_json_path_exits_2(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    log = write_jsonl(tmp_path / "shadow.jsonl", BASE)

    assert tool.main([str(log), "--json", str(tmp_path)]) == 2  # a directory

    assert f"cannot write {tmp_path}" in capsys.readouterr().err


def test_the_script_runs_standalone(tmp_path: Path) -> None:
    """As the lab runs it: a script under an isolated interpreter, printing the report."""
    log = write_jsonl(tmp_path / "shadow.jsonl", BASE)
    out = tmp_path / "summary.json"
    env = {**os.environ, "HOME": str(tmp_path / "home")}

    result = run_script(
        [sys.executable, "-I", str(TOOL), str(log), "--measured", "lab", "--json", str(out), *PIN],
        env,
    )

    assert result.returncode == 1, result.stderr
    assert f"Pass test at τ {INSTALLED_TAU}" in result.stdout
    assert SCORE_SIGNATURE_SHA256 in result.stdout
    assert f'model="{MODEL}"' in result.stdout
    assert f"tau={INSTALLED_TAU}," in result.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["installed"]["n"] == 4
