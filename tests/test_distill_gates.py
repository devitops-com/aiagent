"""The ship gate: CP bound, calibration metrics, τ fitting, verdicts, reports, scoring."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from aiagent.distill.dataset import build_row
from aiagent.distill.gates import (
    ECE_BINS,
    EXIT_CODES,
    PRECISION_POINTS,
    EvalItem,
    EvalReport,
    GateTargets,
    QuestionReport,
    Scored,
    accuracy,
    brier_score,
    clopper_pearson_lower,
    coverage_at_precision,
    decide,
    expected_calibration_error,
    find_report_for_dataset,
    fit_threshold,
    load_report,
    min_certifiable,
    question_report,
    save_report,
    score_items,
    select_for_repair,
)
from aiagent.distill.splits import document_id
from aiagent.exceptions import DatasetContractError, DistillError
from aiagent.system1.runtime import Answer, System1Runtime
from aiagent.system1.sequence import Encoded, SequenceTokenizer
from system1_helpers import FIXTURE, FIXTURE_HEAD_MAX_LEN, FIXTURE_MAX_LEN

POLARITY = {
    "polarity": {
        "type": "choice",
        "instructions": "Overall sentiment polarity of the passage.",
        "criteria": ["negative", "neutral", "mixed", "positive"],
    }
}
TEXTS = (
    "good service here",
    "the delivery was late and the box was broken",
    "Die Lieferung kam zu spät, aber der Support war hervorragend.",
    "",
    "is it late ?",
)
HEX = "0123456789abcdef" * 4


def answer(conf: float, keys: tuple[str, ...] = ("a", "b")) -> Answer:
    """A two-option choice whose first key wins at probability `conf` (>= 0.5)."""
    return Answer(
        type="choice", keys=keys, probabilities=(conf, 1.0 - conf),
        answer_confidence=conf, confidence=0.0, encoded=Encoded((), (0, 1)),
    )


def scored(conf: float, correct: bool, row_id: str = "r") -> Scored:
    return Scored(row_id=row_id, label="a" if correct else "b", answer=answer(conf))


def many(n: int, conf: float, n_correct: int) -> list[Scored]:
    return [scored(conf, i < n_correct, f"r{conf}-{i:04d}") for i in range(n)]


@pytest.fixture(scope="module")
def runtime() -> System1Runtime:
    return System1Runtime(FIXTURE)


# --------------------------------------------------------------------------- Clopper-Pearson


@pytest.mark.parametrize(
    ("k", "n", "lower"),
    [
        (10, 10, 0.741134),
        (59, 59, 0.950492),
        (58, 59, 0.922102),
        (45, 50, 0.801167),
        (90, 100, 0.836282),
        (190, 200, 0.916665),
        (1, 1, 0.05),
        (0, 10, 0.0),
    ],
)
def test_clopper_pearson_matches_scipy_beta_ppf(k: int, n: int, lower: float) -> None:
    assert clopper_pearson_lower(k, n, 0.05) == pytest.approx(lower, abs=1e-5)


def test_clopper_pearson_edges() -> None:
    assert clopper_pearson_lower(0, 0, 0.05) == 0.0
    with pytest.raises(ValueError):
        clopper_pearson_lower(3, 2, 0.05)


def test_min_certifiable_is_the_smallest_all_correct_n() -> None:
    assert min_certifiable(0.95, 0.05) == 59
    assert clopper_pearson_lower(59, 59, 0.05) >= 0.95
    assert clopper_pearson_lower(58, 58, 0.05) < 0.95
    assert min_certifiable(0.5, 0.05) == 5


# --------------------------------------------------------------------------- metrics


def test_ece_matches_laya() -> None:
    rows = [scored(0.9, True), scored(0.9, False), scored(0.6, True), scored(0.6, True)]
    assert expected_calibration_error(rows) == pytest.approx(0.4)
    assert ECE_BINS == 15


def test_ece_first_bin_includes_zero_and_empty_is_zero() -> None:
    uniform = Answer(type="choice", keys=("a", "b"), probabilities=(0.5, 0.5),
                     answer_confidence=0.0, confidence=0.0, encoded=Encoded((), (0, 1)))
    assert expected_calibration_error([Scored("r", "b", uniform)]) == pytest.approx(0.0)
    assert expected_calibration_error([Scored("r", "a", uniform)]) == pytest.approx(1.0)
    assert expected_calibration_error([]) == 0.0


def test_brier_score_by_hand() -> None:
    three = Answer(type="choice", keys=("a", "b", "c"), probabilities=(0.7, 0.2, 0.1),
                   answer_confidence=0.7, confidence=0.0, encoded=Encoded((), (0, 1, 2)))
    # (0.3^2 + 0.2^2 + 0.1^2) = 0.14 for label a; (0.5^2 + 0.5^2) = 0.5 for label b at 0.5/0.5
    rows = [Scored("r1", "a", three), Scored("r2", "b", answer(0.5))]
    assert brier_score(rows) == pytest.approx(0.32)
    assert brier_score([]) == 0.0


def test_accuracy() -> None:
    assert accuracy([scored(0.9, True), scored(0.8, False), scored(0.7, True)]) == pytest.approx(2 / 3)
    assert accuracy([]) == 0.0


def test_scored_correct_compares_the_answer_key() -> None:
    assert scored(0.9, True).correct
    assert not scored(0.9, False).correct


# --------------------------------------------------------------------------- τ and coverage

# Highest confidence first: precision 1/1, 1/2, 2/3, 3/4, 4/5, 4/6.
NON_MONOTONE = [
    scored(0.99, True), scored(0.9, False), scored(0.8, True),
    scored(0.7, True), scored(0.6, True), scored(0.5, False),
]


def test_fit_threshold_takes_the_lowest_passing_t_not_the_first_failure() -> None:
    assert fit_threshold(NON_MONOTONE, target_precision=0.8) == 0.6
    assert fit_threshold(NON_MONOTONE, target_precision=1.0) == 0.99


def test_fit_threshold_unreachable_is_none() -> None:
    assert fit_threshold([scored(0.9, False), scored(0.8, True)], target_precision=1.0) is None
    assert fit_threshold([], target_precision=0.5) is None


def test_fit_threshold_includes_ties_at_t() -> None:
    rows = [scored(0.9, True), scored(0.9, False), scored(0.8, True)]
    assert fit_threshold(rows, target_precision=0.9) is None  # t=0.9 accepts both 0.9 rows
    assert fit_threshold(rows, target_precision=0.6) == 0.8


def test_coverage_at_precision() -> None:
    assert coverage_at_precision(NON_MONOTONE, 0.8) == pytest.approx(5 / 6)
    assert coverage_at_precision(NON_MONOTONE, 0.95) == pytest.approx(1 / 6)
    assert coverage_at_precision([scored(0.9, False)], 0.5) == 0.0
    assert coverage_at_precision([], 0.5) == 0.0


# --------------------------------------------------------------------------- targets and reports


def test_gate_targets_defaults_and_bounds() -> None:
    targets = GateTargets()
    assert (targets.precision, targets.fit_precision, targets.alpha) == (0.95, 0.98, 0.05)
    assert (targets.min_coverage, targets.epsilon, targets.max_rounds) == (0.20, 0.01, 3)
    assert EXIT_CODES == {"ship": 0, "repair": 3, "stop": 4}
    assert PRECISION_POINTS == (0.90, 0.95, 0.98)
    for bad in (
        {"precision": 0.95, "fit_precision": 0.9},
        {"precision": 1.0, "fit_precision": 1.0},
        {"precision": 0.0},
        {"alpha": 0.0},
        {"alpha": 1.0},
    ):
        with pytest.raises(ValueError):
            GateTargets(**bad)  # type: ignore[arg-type]


def test_question_report_passes() -> None:
    report = question_report("q", many(100, 0.99, 100), many(100, 0.99, 100), GateTargets())
    assert report.qid == "q"
    assert (report.n_calib, report.n_heldout) == (100, 100)
    assert (report.tau, report.accepted, report.correct) == (0.99, 100, 100)
    assert report.precision == 1.0
    assert report.precision_lower == pytest.approx(0.05 ** (1 / 100))
    assert report.coverage == 1.0
    assert report.accuracy == 1.0
    assert list(report.coverage_at_precision) == ["0.90", "0.95", "0.98"]
    assert report.passes


def test_question_report_fails_on_the_cp_bound() -> None:
    report = question_report("q", many(100, 0.99, 100), many(50, 0.99, 50), GateTargets())
    assert report.precision == 1.0
    assert report.precision_lower < 0.95
    assert report.coverage == 1.0
    assert not report.passes


def test_question_report_fails_on_coverage() -> None:
    heldout = many(100, 0.99, 100) + many(900, 0.6, 900)
    report = question_report("q", many(100, 0.99, 100), heldout, GateTargets())
    assert report.accepted == 100
    assert report.precision_lower >= 0.95
    assert report.coverage == pytest.approx(0.1)
    assert not report.passes


def test_question_report_without_tau() -> None:
    report = question_report("q", many(20, 0.9, 0), many(100, 0.99, 100), GateTargets())
    assert report.tau is None
    assert (report.accepted, report.correct, report.precision) == (0, 0, None)
    assert (report.precision_lower, report.coverage) == (0.0, 0.0)
    assert not report.passes


def test_fit_precision_matters() -> None:
    # 200 rows at 0.93 (186 right) and 200 at 0.99 (198 right), used as calib and held-out.
    rows = many(200, 0.93, 186) + many(200, 0.99, 198)

    loose = question_report("q", rows, rows, GateTargets(precision=0.95, fit_precision=0.95))
    assert loose.tau == 0.93
    assert (loose.accepted, loose.correct) == (400, 384)
    assert loose.precision_lower == pytest.approx(0.9399, abs=1e-4)
    assert not loose.passes

    strict = question_report("q", rows, rows, GateTargets(precision=0.95, fit_precision=0.98))
    assert strict.tau == 0.99
    assert (strict.accepted, strict.correct) == (200, 198)
    assert strict.precision_lower == pytest.approx(0.9689, abs=1e-4)
    assert strict.coverage == 0.5
    assert strict.passes


# --------------------------------------------------------------------------- decide


def qreport(*, passes: bool, n_heldout: int = 400, accuracy: float = 0.9, qid: str = "q") -> QuestionReport:
    return QuestionReport(
        qid=qid, n_calib=200, n_heldout=n_heldout, accuracy=accuracy, ece=0.02, brier=0.1,
        coverage_at_precision={"0.90": 0.9, "0.95": 0.6, "0.98": 0.3}, tau=0.9,
        accepted=300, correct=290, precision=290 / 300, precision_lower=0.94, coverage=0.75,
        passes=passes,
    )


def test_decide_ship() -> None:
    verdict, reason = decide([qreport(passes=True)], round=2, pool_remaining=0,
                             previous_accuracy=0.95, targets=GateTargets())
    assert verdict == "ship"
    assert "0.95" in reason


def test_decide_stop_on_a_held_out_too_small_to_certify() -> None:
    verdict, reason = decide([qreport(passes=False, n_heldout=58)], round=0, pool_remaining=100,
                             previous_accuracy=None, targets=GateTargets())
    assert verdict == "stop"
    assert "58" in reason and "59" in reason and "larger corpus" in reason


def test_decide_stop_with_no_rounds_left() -> None:
    verdict, reason = decide([qreport(passes=False)], round=2, pool_remaining=100,
                             previous_accuracy=None, targets=GateTargets())
    assert (verdict, "no rounds left" in reason) == ("stop", True)


def test_decide_stop_on_an_exhausted_pool() -> None:
    verdict, reason = decide([qreport(passes=False)], round=0, pool_remaining=0,
                             previous_accuracy=None, targets=GateTargets())
    assert (verdict, "pool exhausted" in reason) == ("stop", True)


def test_decide_stop_without_improvement() -> None:
    verdict, reason = decide([qreport(passes=False, accuracy=0.9)], round=1, pool_remaining=100,
                             previous_accuracy=0.895, targets=GateTargets())
    assert (verdict, "did not improve" in reason) == ("stop", True)


def test_decide_repair() -> None:
    reports = [qreport(passes=True, qid="a", accuracy=0.95), qreport(passes=False, qid="b", accuracy=0.85)]
    verdict, reason = decide(reports, round=1, pool_remaining=100, previous_accuracy=0.85,
                             targets=GateTargets())
    assert verdict == "repair"
    assert "b" in reason


def test_decide_needs_questions() -> None:
    with pytest.raises(ValueError):
        decide([], round=0, pool_remaining=1, previous_accuracy=None, targets=GateTargets())


# --------------------------------------------------------------------------- EvalReport


def make_report(run: str = "ftjob-1", *, dataset_id: str = "ds-aaaaaaaaaaaa",
                created_at: str = "2026-09-24T20:00:00Z") -> EvalReport:
    return EvalReport(
        run=run, artifact_id="a-0123456789ab", dataset_id=dataset_id, parent_dataset=None,
        campaign="c-20260924T191500Z", round=0, skill="polarity", predictor="classify",
        skill_source_sha256=HEX, source_changed_since_labeling=False,
        targets=GateTargets(fit_precision=0.99), questions=(qreport(passes=False),),
        accuracy=0.9, pool_remaining=12, previous_run=None, previous_accuracy=None,
        files={"manifest.json": HEX, "model.onnx": HEX}, verdict="repair",
        reason="q below the gate", created_at=created_at,
    )


def test_eval_report_json_round_trip() -> None:
    report = make_report()
    data = json.loads(json.dumps(report.to_json()))
    assert data["targets"]["fit_precision"] == 0.99
    assert data["files"] == {"manifest.json": HEX, "model.onnx": HEX}
    assert data["questions"][0]["coverage_at_precision"]["0.95"] == 0.6
    assert EvalReport.from_json(data) == report


def test_save_load_and_find_reports(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    old = make_report("ftjob-old", created_at="2026-09-24T20:00:00Z")
    new = make_report("ftjob-new", created_at="2026-09-25T08:00:00Z")
    other = make_report("ftjob-other", dataset_id="ds-bbbbbbbbbbbb",
                        created_at="2026-09-26T08:00:00Z")

    paths = [save_report(artifacts, report) for report in (new, old, other)]

    assert paths[0] == artifacts / "system1" / "evals" / "ftjob-new.json"
    assert sorted(p.name for p in paths[0].parent.iterdir()) == [
        "ftjob-new.json", "ftjob-old.json", "ftjob-other.json"
    ]  # no temp file left behind
    assert load_report(artifacts, "ftjob-old") == old
    assert load_report(artifacts, "ftjob-absent") is None
    assert find_report_for_dataset(artifacts, "ds-aaaaaaaaaaaa") == new
    assert find_report_for_dataset(artifacts, "ds-cccccccccccc") is None
    assert find_report_for_dataset(tmp_path / "nothing", "ds-aaaaaaaaaaaa") is None

    save_report(artifacts, replace(old, verdict="stop"))  # overwrite in place
    assert load_report(artifacts, "ftjob-old") == replace(old, verdict="stop")


def test_malformed_reports(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    artifacts = tmp_path / "artifacts"
    good = make_report("ftjob-good")
    save_report(artifacts, good)
    evals = artifacts / "system1" / "evals"
    (evals / "ftjob-bad.json").write_text('{"run": "ftjob-bad"}\n', encoding="utf-8")
    data = good.to_json()
    (evals / "ftjob-verdict.json").write_text(json.dumps({**data, "verdict": "maybe"}),
                                             encoding="utf-8")

    with pytest.raises(DistillError, match="ftjob-bad"):
        load_report(artifacts, "ftjob-bad")
    with pytest.raises(DistillError, match="verdict"):
        load_report(artifacts, "ftjob-verdict")
    with caplog.at_level(logging.WARNING):
        assert find_report_for_dataset(artifacts, good.dataset_id) == good
    assert "ftjob-bad" in caplog.text


def test_unsafe_run_ids_are_refused(tmp_path: Path) -> None:
    with pytest.raises(DistillError, match="run"):
        save_report(tmp_path, make_report("../escape"))
    with pytest.raises(DistillError):
        load_report(tmp_path, "a/b")


# --------------------------------------------------------------------------- scoring


def items(texts: tuple[str, ...], *, labels: str | None = "mixed") -> list[EvalItem]:
    tokenizer = SequenceTokenizer.from_dir(FIXTURE / "tokenizer")
    result = []
    for text in texts:
        row = build_row(
            group_id=document_id(text or "empty"), index=0, text=text, input_field="text",
            questions=POLARITY, split="calib", tokenizer=tokenizer,
            max_len=FIXTURE_MAX_LEN, head_max_len=FIXTURE_HEAD_MAX_LEN,
        )
        result.append(EvalItem(
            row_id=row.id, state=row.state,
            labels={"polarity": labels} if labels is not None else {},
            ids_sha256={q: t.ids_sha256 for q, t in row.student_tokens.items()},
        ))
    return result


def test_score_items_with_the_fixture_runtime(runtime: System1Runtime) -> None:
    evals = items(TEXTS)

    result = score_items(runtime, evals, POLARITY)

    assert list(result) == ["polarity"]
    assert [s.row_id for s in result["polarity"]] == [item.row_id for item in evals]
    for item, got in zip(evals, result["polarity"], strict=True):
        want = runtime.predict(item.state, POLARITY)["polarity"]
        assert got.label == "mixed"
        assert got.answer == want
        assert got.correct == (want.key == "mixed")


def test_score_items_rejects_a_wrong_ids_sha256(runtime: System1Runtime) -> None:
    [item] = items(TEXTS[:1])
    bad = replace(item, ids_sha256={"polarity": HEX})

    with pytest.raises(DatasetContractError, match=item.row_id):
        score_items(runtime, [bad], POLARITY)


def test_select_for_repair_least_confident_first(runtime: System1Runtime) -> None:
    pool = items(TEXTS, labels=None)
    twin = replace(pool[0], row_id="000000000000:0000")  # same state: a tie, broken by row id
    pool.append(twin)
    confidence = {
        item.row_id: runtime.predict(item.state, POLARITY)["polarity"].answer_confidence
        for item in pool
    }
    want = sorted(confidence, key=lambda row_id: (confidence[row_id], row_id))

    assert select_for_repair(runtime, pool, POLARITY, max_rows=100) == want
    assert select_for_repair(runtime, pool, POLARITY, max_rows=3) == want[:3]
    assert select_for_repair(runtime, pool, POLARITY, max_rows=0) == []
    tie = [row_id for row_id in want if row_id in (twin.row_id, pool[0].row_id)]
    assert tie == [twin.row_id, pool[0].row_id]
