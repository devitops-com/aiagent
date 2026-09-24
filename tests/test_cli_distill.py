"""``aiagent distill``: the command wiring, output and exit codes over the real campaign.

The seams are those of test_distill_campaign (a DummyLM teacher, a MockTransport trainer, the
fixture student and base); the warm-up is replaced by a recorder.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner, Result

from aiagent.cli import distill_cmd
from aiagent.cli.app import app
from aiagent.config import Settings
from aiagent.distill import campaign
from aiagent.distill import label as label_mod
from aiagent.distill.dataset import DEFAULT_BASE
from aiagent.distill.gates import GateTargets
from aiagent.distill.label import (
    DEFAULT_K,
    DEFAULT_TEMPERATURE,
    MAX_CONCURRENCY,
    LabelConfig,
)
from aiagent.distill.segment import Sources
from test_distill_campaign import (
    BASE,
    COUNTS,
    JOB,
    contrary,
    distill_settings,
    job,
    make_run,
    student_answer,
    use_teacher,
    use_trainer,
    write_corpus,
)

runner = CliRunner()

COMMANDS = ("plan", "label", "train", "status", "eval", "repair", "install")


@pytest.fixture(scope="module")
def oracle() -> str:
    return student_answer()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return distill_settings(tmp_path)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "corpus.jsonl")


@pytest.fixture
def warmed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    models: list[str] = []
    monkeypatch.setattr(label_mod, "warm_up_teacher", lambda model, *, settings: models.append(model))
    return models


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["distill", *args])


def label(corpus: Path, *extra: str) -> tuple[Result, str]:
    """`distill label polarity` over the corpus; the result and the new dataset id."""
    result = invoke("label", "polarity", "--jsonl", str(corpus), "--base", BASE, "--k", "2", *extra)
    assert result.exit_code == 0, result.output
    (dataset,) = [w for w in result.stdout.split() if w.startswith("ds-")][:1]
    return result, dataset


def labeled_run(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, answer: str, **binds: str
) -> None:
    use_teacher(monkeypatch, answer)
    outcome = campaign.label(
        "polarity",
        predictor=None,
        sources=Sources(jsonl=(corpus,)),
        teacher=None,
        config=LabelConfig(k=2),
        base=BASE,
        settings=settings,
    )
    make_run(settings, outcome.dataset, **binds)


def test_distill_help_lists_the_seven_commands() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.stdout


def test_the_option_defaults_mirror_the_library_constants() -> None:
    assert (distill_cmd._DEFAULT_K, distill_cmd._DEFAULT_TEMPERATURE) == (DEFAULT_K, DEFAULT_TEMPERATURE)
    assert distill_cmd._DEFAULT_CONCURRENCY == MAX_CONCURRENCY
    assert distill_cmd._DEFAULT_BASE == DEFAULT_BASE
    assert distill_cmd._DEFAULT_REPAIR_ROWS == campaign.DEFAULT_REPAIR_ROWS
    assert {
        "n_epochs": distill_cmd._DEFAULT_EPOCHS,
        "learning_rate_multiplier": distill_cmd._DEFAULT_LR_MULTIPLIER,
    } == campaign.DEFAULT_HYPERPARAMETERS
    assert GateTargets(
        precision=distill_cmd._DEFAULT_TARGET_PRECISION,
        fit_precision=distill_cmd._DEFAULT_FIT_PRECISION,
        alpha=distill_cmd._DEFAULT_ALPHA,
        min_coverage=distill_cmd._DEFAULT_MIN_COVERAGE,
        epsilon=distill_cmd._DEFAULT_EPSILON,
        max_rounds=distill_cmd._DEFAULT_MAX_ROUNDS,
    ) == GateTargets()


# --------------------------------------------------------------------------- plan


def test_plan_shows_the_questions_hashes_and_heldout_sizing() -> None:
    result = invoke("plan", "polarity")

    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "classify" in out and "qualifies      : yes" in out
    assert "polarity choice [negative, neutral, mixed, positive]" in out
    assert "Overall sentiment polarity of the passage." in out
    assert "≥ 59 accepted held-out rows" in out and "≈181" in out and "≈361" in out


def test_plan_json_carries_the_full_hashes() -> None:
    result = invoke("plan", "polarity", "--predictor", "classify", "--json")

    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    (predictor,) = data["predictors"]
    assert (data["skill"], predictor["name"], predictor["qualifies"]) == ("polarity", "classify", True)
    assert len(predictor["signature_sha256"]) == 64
    assert predictor["questions"]["polarity"]["type"] == "choice"
    assert data["heldout_sizing"]["min_accepted"] == 59


@pytest.mark.parametrize("args", [["extract"], ["polarity", "--predictor", "nope"]])
def test_plan_exits_1_when_nothing_qualifies(args: list[str]) -> None:
    result = invoke("plan", *args)
    assert result.exit_code == 1


def test_plan_names_the_reasons() -> None:
    result = invoke("plan", "extract")
    assert "qualifies      : no" in result.stdout
    assert "output 'merchant': str is not a closed answer set" in result.stdout


# --------------------------------------------------------------------------- label / train / status


def test_label_reports_the_dataset_and_warns_on_stderr(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    teacher = use_teacher(monkeypatch, oracle)

    result, dataset = label(corpus, "--teacher", "big-teacher")

    assert (settings.distill_dir / "inbox" / dataset).is_dir()
    out = result.stdout
    assert "documents : 80 (80 segments, 0 duplicates)" in out
    assert ", ".join(f"{s} {n}" for s, n in COUNTS.items()) in out
    assert f"aiagent distill train {dataset}" in out
    assert "labeled 25/64" in result.stderr and "labeled 64/64" in result.stderr
    assert "warning: held-out has 11 rows" in result.stderr
    assert teacher.calls == [{"alias": "big-teacher", "exact": None}]


def test_label_json(settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str) -> None:
    use_teacher(monkeypatch, oracle)
    result = invoke("label", "polarity", "--jsonl", str(corpus), "--base", BASE, "--k", "2", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["counts"] == COUNTS
    assert data["dataset"].startswith("ds-") and data["warnings"]


@pytest.mark.parametrize("option", [["--k", "0"], ["--temperature", "0"], ["--concurrency", "5"]])
def test_label_rejects_bad_sampling_options(option: list[str], corpus: Path) -> None:
    result = invoke("label", "polarity", "--jsonl", str(corpus), *option)
    assert result.exit_code == 2


def test_train_wait_warms_up_and_points_to_eval(
    settings: Settings,
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    oracle: str,
    warmed: list[str],
) -> None:
    use_teacher(monkeypatch, oracle)
    _, dataset = label(corpus)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json=job("validating_files", training_file=dataset))
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json={"object": "list", "data": []})
        return httpx.Response(200, json=job("succeeded", training_file=dataset))

    requests = use_trainer(monkeypatch, settings, handler)

    result = invoke("train", dataset, "--epochs", "2", "--wait")

    assert result.exit_code == 0, result.output
    assert json.loads(requests[0].content)["hyperparameters"]["n_epochs"] == 2
    assert "batch_size" not in json.loads(requests[0].content)["hyperparameters"]
    assert f"{JOB}: succeeded (api)" in result.stderr
    assert "warm-up   : dummy" in result.stdout
    assert f"next      : aiagent distill eval {JOB}" in result.stdout
    assert warmed == ["dummy"]


def test_train_without_wait_prints_the_job(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    use_teacher(monkeypatch, oracle)
    _, dataset = label(corpus)
    use_trainer(monkeypatch, settings, lambda _r: httpx.Response(200, json=job("queued")))

    result = invoke("train", dataset)

    assert result.exit_code == 0, result.output
    assert "status    : queued" in result.stdout
    assert f"next      : aiagent distill status {JOB} --wait" in result.stdout
    assert json.loads(invoke("train", dataset, "--json").stdout)["status"] == "queued"


def test_train_sends_a_batch_size_only_when_given(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    use_teacher(monkeypatch, oracle)
    _, dataset = label(corpus)
    requests = use_trainer(monkeypatch, settings, lambda _r: httpx.Response(200, json=job("queued")))

    assert invoke("train", dataset).exit_code == 0
    assert invoke("train", dataset, "--batch-size", "16").exit_code == 0

    sent = [json.loads(r.content)["hyperparameters"] for r in requests]
    assert "batch_size" not in sent[0]  # the trainer picks it: it knows the GPU's memory
    assert sent[1]["batch_size"] == 16


def write_job(settings: Settings, status: str, **extra: object) -> None:
    run = settings.distill_dir / "runs" / JOB
    run.mkdir(parents=True)
    (run / "job.json").write_text(json.dumps(job(status, **extra)), encoding="utf-8")
    event = {"created_at": 1_790_000_000, "level": "info", "message": f"Job {status}"}
    undated = {"level": "warning", "message": "no timestamp"}
    (run / "events.jsonl").write_text(
        json.dumps(event) + "\n" + json.dumps(undated) + "\n", encoding="utf-8"
    )


def test_status_shows_the_job_and_its_events(settings: Settings) -> None:
    write_job(settings, "running")

    result = invoke("status", JOB)

    assert result.exit_code == 0, result.output
    assert "status    : running" in result.stdout and "source    : volume" in result.stdout
    assert "2026-09-21T14:13:20Z info Job running" in result.stdout
    assert "- warning no timestamp" in result.stdout


def test_status_json(settings: Settings) -> None:
    write_job(settings, "running")
    result = invoke("status", JOB, "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert (data["status"], data["source"], len(data["events"])) == ("running", "volume", 2)


def test_status_wait_json_on_a_succeeded_job(settings: Settings, warmed: list[str]) -> None:
    write_job(settings, "succeeded")

    result = invoke("status", JOB, "--wait", "--json")

    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert (data["status"], data["next"]) == ("succeeded", f"aiagent distill eval {JOB}")
    assert data["warm-up"] == "skipped (see the warning above)"  # no such dataset
    assert warmed == []


def test_status_wait_exits_1_on_a_failed_job(settings: Settings, warmed: list[str]) -> None:
    write_job(settings, "failed", error={"message": "CUDA out of memory"})

    result = invoke("status", JOB, "--wait", "--no-warm-up")

    assert result.exit_code == 1
    assert "status    : failed" in result.stdout
    assert "CUDA out of memory" in str(result.exception)
    assert warmed == []


# --------------------------------------------------------------------------- eval / install / repair


@pytest.mark.parametrize(
    ("teacher", "args", "code", "verdict"),
    [
        ("oracle", ["--target-precision", "0.5", "--min-coverage", "0"], 0, "ship"),
        ("contrary", ["--target-precision", "0.5", "--min-coverage", "0"], 3, "repair"),
        ("oracle", [], 4, "stop"),
    ],
)
def test_eval_exit_codes(
    teacher: str,
    args: list[str],
    code: int,
    verdict: str,
    settings: Settings,
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    oracle: str,
) -> None:
    labeled_run(settings, corpus, monkeypatch, oracle if teacher == "oracle" else contrary(oracle))

    result = invoke("eval", JOB, *args)

    assert result.exit_code == code, result.output
    assert f"verdict   : {verdict}" in result.stdout
    assert "question polarity" in result.stdout
    assert str(settings.artifacts_dir / "system1" / "evals" / f"{JOB}.json") in result.stdout


def test_eval_json(settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str) -> None:
    labeled_run(settings, corpus, monkeypatch, oracle)
    result = invoke("eval", JOB, "--target-precision", "0.5", "--min-coverage", "0", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert (data["verdict"], data["next"]) == ("ship", f"aiagent distill install {JOB}")


def test_eval_notes_a_skill_changed_since_labeling(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    labeled_run(settings, corpus, monkeypatch, oracle, skill_source_sha256="0" * 64)

    result = invoke("eval", JOB)

    assert result.exit_code == 4, result.output
    assert "note      : the skill's files changed since labeling" in result.stdout
    report = json.loads((settings.artifacts_dir / "system1" / "evals" / f"{JOB}.json").read_text())
    assert report["source_changed_since_labeling"] is True
    assert report["skill_source_sha256"] != "0" * 64  # the current value


def test_eval_fit_precision_below_the_target_is_a_usage_error() -> None:
    result = invoke("eval", JOB, "--target-precision", "0.95", "--fit-precision", "0.9")
    assert result.exit_code == 2


def test_install_after_a_ship_verdict(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    labeled_run(settings, corpus, monkeypatch, oracle)
    assert invoke("eval", JOB, "--target-precision", "0.5", "--min-coverage", "0").exit_code == 0

    result = invoke("install", JOB)

    assert result.exit_code == 0, result.output
    home = settings.artifacts_dir / "system1" / "skills" / "polarity" / "classify"
    assert str(home / "a-0123456789ab") in result.stdout
    assert """AIAGENT_SYSTEM1_MODE='{"polarity":"shadow"}'""" in result.stdout
    assert 'polarity = "shadow"' in result.stdout
    assert json.loads(invoke("install", JOB, "--json").stdout)["artifact_id"] == "a-0123456789ab"


def test_repair_after_a_repair_verdict(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    labeled_run(settings, corpus, monkeypatch, contrary(oracle))
    assert invoke("eval", JOB, "--target-precision", "0.5", "--min-coverage", "0").exit_code == 3

    result = invoke("repair", JOB, "--max-rows", "4", "--concurrency", "2")

    assert result.exit_code == 0, result.output
    assert "labeled   : 4 (0 unlabeled)" in result.stdout
    assert "pool left : 12" in result.stdout
    (dataset,) = [w for w in result.stdout.split() if w.startswith("ds-")][:1]
    assert f"aiagent distill train {dataset}" in result.stdout


def test_repair_json(settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str) -> None:
    labeled_run(settings, corpus, monkeypatch, contrary(oracle))
    invoke("eval", JOB, "--target-precision", "0.5", "--min-coverage", "0")
    data = json.loads(invoke("repair", JOB, "--max-rows", "4", "--json").stdout)
    assert (data["labeled"], data["pool_remaining"]) == (4, 12)
