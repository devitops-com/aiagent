"""The shipped sentiment System 1 calibration: ``NEUTRAL`` as the D9 pre-registration pins it.

Owner decision D9 (design ``docs/design/sentiment-system1.md`` §3): run 1's calibration at
τ 0.96, pinned only because the pre-registered confirmation run on 120 fresh Wikipedia
articles passed (lab 2026-09-27). The pre-registration (``<pilot dir>/sentiment/confirm/
PREREG.md``, "On PASS") fixes every field below. Nothing here is monkeypatched: these tests
read what ships. A different value is a new calibration, which takes a new lab run and pass
test, never an edit. The gate-mode side (another student only shadows) is in
test_sentiment_system1.py.
"""

from __future__ import annotations

from aiagent.core import sentiment as sentiment_mod
from aiagent.core.sentiment import (
    SCORE_SIGNATURE_SHA256,
    NeutralCalibration,
    ScoreSegment,
)
from aiagent.distill.questions import signature_sha256

PREREGISTRATION = (
    "the D9 pre-registration (<pilot dir>/sentiment/confirm/PREREG.md, 'On PASS'; "
    "docs/design/sentiment-system1.md §3)"
)
PREREGISTERED = {
    "artifact_id": "a866e0a4734f66ddb975ac1d2e41780c8961913cf6fa738ef53b6bc7b843e273",
    "score_signature_sha256": "b26dee349163d3219febe678ead8fd8feb1c8e13e3c0469a883f7e3cb7febc96",
    "model": "openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink",
    "tau": 0.96,
    "level": -0.02,
    "se": 0.0332,
    "sigma_between": 0.6832,
    "sigma_within": 0.4039,
    "n": 472,
    # As the pre-registered pin block generated it (confirm/analysis/run1-pin.json,
    # neutral_calibration.measured), verbatim.
    "measured": (
        "lab 2026-09-26: aiagent 0.7.0 shadow run at --resample 3, docs.jsonl 396eef0c "
        "(120 Wikipedia articles, en/de/hr), teacher "
        "openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink@118784; tau 0.96 (D9), "
        "confirmed by lab 2026-09-27: confirm.jsonl 55369be0 (120 fresh Wikipedia "
        "articles, en/de/hr), (a) 455/468, CP bound 0.9562; (b) p95 |delta| 0.4040 "
        "over 117 documents"
    ),
}
# What `measured` must name: run 1 and the confirmation, with the confirmation's figures
# (PREREGISTERED holds the exact string; this list names what an edit dropped).
MEASURED = (
    "lab 2026-09-26",
    "docs.jsonl 396eef0c",
    "teacher openai/Qwen3.8-27B-MTP-devai-NVFP4::mtp::nothink@118784",
    "tau 0.96 (D9)",
    "confirmed by lab 2026-09-27",
    "confirm.jsonl 55369be0",
    "(a) 455/468, CP bound 0.9562",
    "(b) p95 |delta| 0.4040 over 117 documents",
)
INSTALLED_TAU = 0.8938  # a866e0a4's installed τ, 0.89384 (walkthrough, phase 5)
FIXTURE_TAU = 0.0  # the τ test_sentiment_system1.py installs the fixture student at


def shipped() -> NeutralCalibration:
    calibration = sentiment_mod.NEUTRAL
    assert isinstance(calibration, NeutralCalibration), (
        f"sentiment ships no calibration (NEUTRAL is {calibration!r}), but {PREREGISTRATION} "
        "pinned one"
    )
    return calibration


def test_a_calibration_ships() -> None:
    assert isinstance(sentiment_mod.NEUTRAL, NeutralCalibration)


def test_the_shipped_calibration_is_for_the_current_score_segment() -> None:
    calibration = shipped()
    assert (
        calibration.score_signature_sha256
        == signature_sha256(ScoreSegment)
        == SCORE_SIGNATURE_SHA256
    ), (
        "ScoreSegment changed, so the shipped NEUTRAL calibration no longer applies: "
        "recalibrate (docs/design/sentiment-system1.md §2.9), confirm, and pin NEUTRAL and "
        "SCORE_SIGNATURE_SHA256 again"
    )


def test_the_shipped_calibration_is_the_preregistered_one() -> None:
    calibration = shipped()
    actual = {field: getattr(calibration, field) for field in PREREGISTERED}
    assert actual == PREREGISTERED, (
        f"NEUTRAL is not what {PREREGISTRATION} pins: a new value needs a new lab run and "
        "pass test, not an edit"
    )


def test_the_shipped_calibration_names_run_1_and_its_confirmation() -> None:
    measured = shipped().measured
    missing = [part for part in MEASURED if part not in measured]
    assert not missing, f"NEUTRAL.measured does not name {missing} ({PREREGISTRATION})"


def test_the_shipped_tau_is_above_the_students() -> None:
    """The gate takes a segment at the higher of the student's τ and the calibration's, so
    the shipped 0.96 is the one that decides, for the lab's student and the fixture's."""
    assert shipped().tau > max(INSTALLED_TAU, FIXTURE_TAU)
