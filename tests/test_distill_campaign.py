"""The distillation campaign end to end, on the fixture student: no LLM, no trainer, no GPU.

Seams: ``campaign.teacher_lm`` returns a DummyLM keyed by segment text and records its
``alias``/``exact`` arguments; ``campaign.trainer_client`` is a TrainerClient over an
``httpx.MockTransport``. The base checkpoint is the fixture's tokenizer and limits
(``make_base_checkpoint``), so the fixture student is what devai would train from it.

The corpus is 80 one-sentence documents that tokenize alike (the fixture's word-level
vocabulary knows none of the ticket words), so the student gives every one the same answer
at the same confidence: an oracle teacher (the student's own answer) ships, a contrary one
does not. Their document hashes split 48 train / 5 calib / 11 held-out / 16 pool.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from dspy.utils.dummies import DummyLM

from aiagent.config import Settings, load_settings
from aiagent.distill import campaign
from aiagent.distill import label as label_mod
from aiagent.distill.client import FineTuningJob, TrainerClient
from aiagent.distill.dataset import load_dataset, validate_dataset
from aiagent.distill.gates import GateTargets, load_report, save_report
from aiagent.distill.label import LabelConfig
from aiagent.distill.segment import Sources
from aiagent.distill.splits import assign_split, document_id
from aiagent.exceptions import ArtifactError, DatasetContractError, DistillError
from aiagent.system1.artifacts import CURRENT_FILE, artifact_home, load_installed
from aiagent.system1.contract import canonical_json, dataset_id, file_sha256
from aiagent.system1.runtime import System1Runtime
from aiagent.system1.sequence import SequenceTokenizer
from system1_helpers import (
    ARTIFACT_FILES,
    FIXTURE,
    make_base_checkpoint,
    make_run_dir,
    resum,
)

BASE = "laya-multilingual@0000test"
KEYS = ("negative", "neutral", "mixed", "positive")
TEXTS = tuple(f"Ticket x{i:03d} closed." for i in range(80))
COUNTS = {"train": 48, "calib": 5, "heldout": 11, "pool": 16}
QUESTIONS = {
    "polarity": {
        "type": "choice",
        "instructions": "Overall sentiment polarity of the passage.",
        "criteria": list(KEYS),
    }
}
JOB = "ftjob-1"
CONFIG = LabelConfig(k=2, temperature=0.7, concurrency=4)
# min_certifiable(0.5, 0.05) = 5 accepted rows, below the corpus's 11 held-out rows
LOOSE = GateTargets(precision=0.5, min_coverage=0.0)


class Teacher:
    """The teacher_lm seam: a DummyLM giving `answer` for every corpus text but `skip`."""

    def __init__(self, answer: str, skip: tuple[str, ...] = ()) -> None:
        self.answers = {text: {"polarity": answer} for text in TEXTS if text not in skip}
        self.calls: list[dict[str, str | None]] = []

    def __call__(
        self, _settings: Settings, *, alias: str | None = None, exact: str | None = None
    ) -> DummyLM:
        self.calls.append({"alias": alias, "exact": exact})
        return DummyLM(self.answers)


def student_answer() -> str:
    """The fixture student's one answer to every corpus text."""
    runtime = System1Runtime(FIXTURE)
    (key,) = {runtime.predict({"text": t}, QUESTIONS)["polarity"].key for t in TEXTS}
    return key


def contrary(key: str) -> str:
    return KEYS[(KEYS.index(key) + 1) % len(KEYS)]


def write_corpus(path: Path) -> Path:
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS), encoding="utf-8")
    return path


def distill_settings(tmp_path: Path) -> Settings:
    """Settings over a distill volume holding the fixture base (clean_env points there)."""
    (tmp_path / "distill").mkdir()
    make_base_checkpoint(tmp_path / "distill", BASE)
    return load_settings()


def use_teacher(
    monkeypatch: pytest.MonkeyPatch, answer: str, skip: tuple[str, ...] = ()
) -> Teacher:
    teacher = Teacher(answer, skip)
    monkeypatch.setattr(campaign, "teacher_lm", teacher)
    return teacher


def use_trainer(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Route campaign.trainer_client to `handler`; return the requests it got."""
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = TrainerClient(settings.trainer_api_base, transport=httpx.MockTransport(record))
    monkeypatch.setattr(campaign, "trainer_client", lambda _settings: client)
    return requests


def job(status: str, *, training_file: str = "ds-000000000000", **extra: Any) -> dict[str, Any]:
    return {
        "object": "fine_tuning.job",
        "id": JOB,
        "status": status,
        "model": "laya-multilingual",
        "training_file": training_file,
        "fine_tuned_model": None,
        "error": None,
        "created_at": 1_790_000_000,
        "finished_at": None,
        **extra,
    }


def no_http(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected trainer call: {request.method} {request.url}")


def run_label(settings: Settings, corpus: Path, **kwargs: Any) -> campaign.LabelOutcome:
    return campaign.label(
        "polarity",
        predictor=None,
        sources=Sources(jsonl=(corpus,)),
        teacher="teacher-alias",
        config=CONFIG,
        base=BASE,
        settings=settings,
        **kwargs,
    )


def make_run(
    settings: Settings, dataset: campaign.DatasetRef, job_id: str = JOB, **binds: str
) -> Path:
    """runs/<job>/ with the fixture student, bound to polarity/classify and `dataset`
    (`binds` overrides some of the three)."""
    current = campaign.plan("polarity", settings=settings).select(None).binds
    assert current is not None
    return make_run_dir(
        settings.distill_dir,
        job_id,
        binds={**current.to_json(), **binds},
        dataset_manifest_sha256=dataset.manifest_sha256,
    )


@pytest.fixture(scope="module")
def oracle() -> str:
    return student_answer()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return distill_settings(tmp_path)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "corpus.jsonl")


# --------------------------------------------------------------------------- plan


def test_plan_polarity_qualifies_and_extract_does_not(settings: Settings) -> None:
    plan = campaign.plan("polarity", settings=settings)
    chosen = plan.select(None)
    assert [p.name for p in plan.predictors] == ["classify"]
    assert plan.select("classify") is chosen
    assert chosen.derivation.qualifies and chosen.binds is not None
    assert chosen.derivation.questions() == QUESTIONS

    extract = campaign.plan("extract", settings=settings)
    assert [p.binds for p in extract.predictors] == [None]
    with pytest.raises(DistillError, match="no predictor of extract can be distilled.*merchant"):
        extract.select(None)
    with pytest.raises(DistillError, match="cannot be distilled: .*amount"):
        extract.select(extract.predictors[0].name)
    with pytest.raises(DistillError, match="no predictor 'nope' .*classify"):
        plan.select("nope")


TWIN = '''
from typing import Literal

import dspy


class Sig(dspy.Signature):
    """Classify a passage."""

    text: str = dspy.InputField()
    label: Literal["a", "b"] = dspy.OutputField()


class Twin(dspy.Module):
    def __init__(self):
        super().__init__()
        self.first = dspy.Predict(Sig)
        self.second = dspy.Predict(Sig)

    def forward(self, text):
        return self.first(text=text)


def build():
    return Twin()
'''


def test_plan_select_needs_a_name_when_several_predictors_qualify(settings: Settings) -> None:
    skill = settings.skills_dir / "twin"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: twin\ndescription: two predictors\n---\n")
    (skill / "skill.py").write_text(TWIN)

    plan = campaign.plan("twin", settings=settings)

    with pytest.raises(DistillError, match=r"several predictors .*\(first, second\).*--predictor"):
        plan.select(None)
    assert plan.select("second").name == "second"


# --------------------------------------------------------------------------- label


def test_label_writes_a_valid_dataset_from_the_teacher_votes(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    teacher = use_teacher(monkeypatch, oracle)
    progress: list[tuple[int, int]] = []

    outcome = run_label(settings, corpus, progress=lambda done, total: progress.append((done, total)))

    assert outcome.dataset.path == settings.distill_dir / "inbox" / outcome.dataset.id
    validate_dataset(outcome.dataset.path, tokenizer=SequenceTokenizer.from_dir(FIXTURE / "tokenizer"))
    dataset = load_dataset(settings.distill_dir, outcome.dataset.id)
    assert dict(outcome.counts) == COUNTS
    assert (outcome.documents, outcome.segments, outcome.duplicates) == (80, 80, 0)
    assert teacher.calls == [{"alias": "teacher-alias", "exact": None}]
    assert progress[-1] == (64, 64) and len(progress) == 64
    labeled = [r for split in ("train", "calib", "heldout") for r in dataset.rows[split]]
    assert {r.gold["polarity"].label for r in labeled if r.gold} == {oracle}
    assert all(r.gold is None and r.teacher is None for r in dataset.rows["pool"])
    pool_lines = (outcome.dataset.path / "pool.jsonl").read_text(encoding="utf-8").splitlines()
    assert all("gold" not in json.loads(line) for line in pool_lines)

    producer = dataset.manifest.producer
    assert (producer.skill, producer.predictor, producer.input_field) == ("polarity", "classify", "text")
    assert (producer.round, producer.parent_dataset, producer.questions) == (0, None, QUESTIONS)
    assert (producer.teacher.model, producer.teacher.k, producer.teacher.temperature) == ("dummy", 2, 0.7)
    assert producer.campaign == "c-" + producer.created_at.replace("-", "").replace(":", "")
    assert [s.segments for s in producer.sources] == [1] * 80
    assert producer.sources[0].origin == f"{corpus}:1"
    assert outcome.stats.rows == 64 and outcome.stats.unlabeled == 0
    assert outcome.stats.label_histogram["polarity"][oracle] == 64

    (warning,) = outcome.warnings
    assert warning.startswith("held-out has 11 rows; certifying 0.95 at α=0.05 needs ≥ 59 accepted")
    assert "≈181" in warning and "2,000-3,000 segments" in warning


def test_label_skips_repeated_segments_and_drops_rows_the_teacher_never_answers(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    """Every document also yields a shared segment (kept once); the teacher knows only
    the corpus texts, so that one is unlabeled if it lands in a labeled split."""
    use_teacher(monkeypatch, oracle)
    shared = "Ticket shared closed."
    monkeypatch.setattr(campaign, "segment_text", lambda text, *, fits: [text, shared])

    outcome = run_label(settings, corpus)

    assert (outcome.documents, outcome.segments, outcome.duplicates) == (80, 81, 79)
    assert sum(outcome.counts.values()) + outcome.stats.unlabeled == 81


def test_label_drops_the_rows_the_teacher_never_answers(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    skipped = next(t for t in TEXTS if assign_split(document_id(t)) == "train")
    use_teacher(monkeypatch, oracle, skip=(skipped,))

    outcome = run_label(settings, corpus)

    assert outcome.stats.unlabeled == 1
    assert dict(outcome.counts) == {**COUNTS, "train": COUNTS["train"] - 1}
    dataset = load_dataset(settings.distill_dir, outcome.dataset.id)
    assert skipped not in {r.state["text"] for r in dataset.rows["train"]}


def test_the_seams_build_the_configured_teacher_and_trainer(settings: Settings) -> None:
    exact = "openai/teacher-27b::nothink@32768"
    assert campaign.teacher_lm(settings, exact=exact).model == exact
    assert campaign.teacher_lm(settings, alias="teacher").model.startswith("openai/teacher")
    assert campaign.trainer_client(settings)._base == "http://devai-router:11438/v1"


def test_label_needs_the_base_checkpoint(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    teacher = use_teacher(monkeypatch, "mixed")
    with pytest.raises(DistillError, match="base checkpoint not found"):
        run_label(load_settings(), corpus)
    assert teacher.calls == []


# --------------------------------------------------------------------------- train


def test_train_posts_the_job_request(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    use_teacher(monkeypatch, oracle)
    outcome = run_label(settings, corpus)
    requests = use_trainer(
        monkeypatch,
        settings,
        lambda _r: httpx.Response(200, json=job("validating_files", training_file=outcome.dataset.id)),
    )

    created = campaign.train(
        outcome.dataset.id, hyperparameters=campaign.DEFAULT_HYPERPARAMETERS, settings=settings
    )

    (request,) = requests
    assert (request.method, request.url.path) == ("POST", "/v1/fine_tuning/jobs")
    producer = load_dataset(settings.distill_dir, outcome.dataset.id).manifest.producer
    assert json.loads(request.content) == {
        "model": "laya-multilingual",
        "training_file": outcome.dataset.id,
        "hyperparameters": {"n_epochs": 4, "learning_rate_multiplier": 1.0},
        "suffix": "polarity-classify",
        "metadata": {"campaign": producer.campaign, "round": "0"},
    }
    assert (created.id, created.status) == (JOB, "validating_files")


def test_train_finds_a_full_revision_base_in_devais_rev12_directory(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    # devai stages the base as base/<name>@<revision[:12]>/; the manifest keeps the full
    # revision (devai's check_base compares it with its catalog's).
    revision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    make_base_checkpoint(tmp_path / "distill", f"laya-multilingual@{revision[:12]}")
    settings = load_settings()
    use_teacher(monkeypatch, oracle)
    outcome = campaign.label(
        "polarity",
        predictor=None,
        sources=Sources(jsonl=(corpus,)),
        teacher="teacher-alias",
        config=CONFIG,
        base=f"laya-multilingual@{revision}",
        settings=settings,
    )
    dataset = load_dataset(settings.distill_dir, outcome.dataset.id)
    assert dataset.manifest.base_checkpoint.revision == revision
    requests = use_trainer(
        monkeypatch,
        settings,
        lambda _r: httpx.Response(200, json=job("validating_files", training_file=outcome.dataset.id)),
    )

    campaign.train(
        outcome.dataset.id, hyperparameters=campaign.DEFAULT_HYPERPARAMETERS, settings=settings
    )

    assert len(requests) == 1


@pytest.mark.parametrize(
    ("skill", "predictor", "suffix"),
    [
        ("polarity", "classify", "polarity-classify"),
        ("myskill", "classify.predict", "myskill-classify-predict"),  # a ChainOfThought's
        ("my_skill", "classify", "my-skill-classify"),
        ("Ticket__Triage", "--x", "ticket-triage-x"),
        ("a" * 50, "classify", "a" * 40),
        ("_", ".", "distill"),
    ],
)
def test_job_suffix_matches_devais_suffix_rule(skill: str, predictor: str, suffix: str) -> None:
    got = campaign.job_suffix(skill, predictor)
    assert got == suffix
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", got)  # devai laya_trainer/jobs.py SUFFIX_RE


def _retokenized_row(path: Path) -> str:
    """Change a train row's token hash, re-sum and re-id the dataset: only rule 5 catches it."""
    train = path / "train.jsonl"
    lines = train.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["student_tokens"]["polarity"]["ids_sha256"] = "0" * 64
    lines[0] = canonical_json(row)
    train.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    resum(path)
    new_id = dataset_id(file_sha256(path / "manifest.json"))
    path.rename(path.parent / new_id)
    return new_id


def _changed_base(path: Path) -> str:
    base = path.parent.parent / "base" / BASE / "model.safetensors"
    base.write_bytes(b"other weights\n")
    return path.name


@pytest.mark.parametrize(
    ("tamper", "error", "match"),
    [
        (_retokenized_row, DatasetContractError, "does not match the base tokenizer"),
        (_changed_base, DistillError, "base checkpoint changed"),
    ],
    ids=["row", "base"],
)
def test_train_refuses_a_bad_dataset_before_any_http_call(
    tamper: Callable[[Path], str],
    error: type[Exception],
    match: str,
    settings: Settings,
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    oracle: str,
) -> None:
    use_teacher(monkeypatch, oracle)
    outcome = run_label(settings, corpus)
    requests = use_trainer(monkeypatch, settings, no_http)
    dataset = tamper(outcome.dataset.path)

    with pytest.raises(error, match=match):
        campaign.train(dataset, hyperparameters=campaign.DEFAULT_HYPERPARAMETERS, settings=settings)
    assert requests == []


# --------------------------------------------------------------------------- status / wait / warm-up


def test_status_reads_the_volume_first(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = use_trainer(monkeypatch, settings, no_http)
    run = settings.distill_dir / "runs" / JOB
    run.mkdir(parents=True)
    (run / "job.json").write_text(json.dumps(job("running")), encoding="utf-8")
    events = [{"created_at": i, "level": "info", "message": f"step {i}"} for i in range(7)]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    view = campaign.status(JOB, settings=settings)

    assert (view.source, view.job.status) == ("volume", "running")
    assert [e["message"] for e in view.events] == [f"step {i}" for i in range(2, 7)]
    assert requests == []


def test_status_falls_back_to_the_api_with_the_events_oldest_first(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [{"created_at": i, "level": "info", "message": f"step {i}"} for i in (3, 2, 1)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json={"object": "list", "data": events})
        return httpx.Response(200, json=job("queued"))

    requests = use_trainer(monkeypatch, settings, handler)

    view = campaign.status(JOB, settings=settings, events=2)

    assert (view.source, view.job.status) == ("api", "queued")
    assert [e["message"] for e in view.events] == ["step 2", "step 3"]
    assert [r.url.path for r in requests] == [
        "/v1/fine_tuning/jobs/ftjob-1",
        "/v1/fine_tuning/jobs/ftjob-1/events",
    ]


def test_wait_polls_until_the_job_ends(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = iter(["queued", "running", "succeeded"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json={"object": "list", "data": []})
        return httpx.Response(200, json=job(next(statuses)))

    use_trainer(monkeypatch, settings, handler)
    sleeps: list[float] = []
    polls: list[campaign.JobView] = []

    view = campaign.wait(JOB, settings=settings, poll_s=7.0, sleep=sleeps.append, on_poll=polls.append)

    assert view.job.status == "succeeded"
    assert sleeps == [7.0, 7.0]
    assert [p.job.status for p in polls] == ["queued", "running", "succeeded"]


def test_warm_up_restores_the_teacher_the_dataset_names(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    use_teacher(monkeypatch, oracle)
    outcome = run_label(settings, corpus)
    warmed: list[str] = []
    monkeypatch.setattr(label_mod, "warm_up_teacher", lambda model, *, settings: warmed.append(model))
    finished = FineTuningJob(id=JOB, status="succeeded", training_file=outcome.dataset.id)

    assert campaign.warm_up(finished, settings=settings) == "dummy"
    assert warmed == ["dummy"]


@pytest.mark.parametrize("training_file", [None, "ds-000000000000"])
def test_a_failed_warm_up_is_only_a_warning(
    training_file: str | None, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    finished = FineTuningJob(id=JOB, status="failed", training_file=training_file)
    with caplog.at_level(logging.WARNING, logger="aiagent.distill.campaign"):
        assert campaign.warm_up(finished, settings=settings) is None
    assert "teacher warm-up skipped" in caplog.text


# --------------------------------------------------------------------------- evaluate


@pytest.mark.parametrize(
    ("teacher", "targets", "verdict", "reason"),
    [
        ("oracle", LOOSE, "ship", "every question certifies"),
        ("contrary", LOOSE, "repair", "polarity: below the gate"),
        ("contrary", dataclasses.replace(LOOSE, max_rounds=1), "stop", "no rounds left"),
        ("oracle", GateTargets(), "stop", "held-out has 11 rows; certifying 0.95 at α=0.05 needs ≥ 59"),
    ],
    ids=["ship", "repair", "stop-rounds", "stop-heldout"],
)
def test_evaluate_scores_the_student_and_decides(
    teacher: str,
    targets: GateTargets,
    verdict: str,
    reason: str,
    settings: Settings,
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    oracle: str,
) -> None:
    use_teacher(monkeypatch, oracle if teacher == "oracle" else contrary(oracle))
    outcome = run_label(settings, corpus)
    make_run(settings, outcome.dataset)

    result = campaign.evaluate(JOB, targets=targets, settings=settings)

    report = result.report
    assert (report.verdict, report.targets) == (verdict, targets)
    assert reason in report.reason
    assert result.path == settings.artifacts_dir / "system1" / "evals" / f"{JOB}.json"
    assert load_report(settings.artifacts_dir, JOB) == report
    assert (report.dataset_id, report.round, report.pool_remaining) == (outcome.dataset.id, 0, 16)
    assert (report.skill, report.predictor, report.source_changed_since_labeling) == ("polarity", "classify", False)
    assert set(report.files) == {"manifest.json", *ARTIFACT_FILES}
    (question,) = report.questions
    assert (question.n_calib, question.n_heldout) == (5, 11)
    assert question.accuracy == (1.0 if teacher == "oracle" else 0.0)


def test_evaluate_needs_the_run(settings: Settings) -> None:
    with pytest.raises(DistillError, match="run ftjob-1 not found"):
        campaign.evaluate(JOB, targets=LOOSE, settings=settings)
    with pytest.raises(DistillError, match="unsafe run '../x'"):
        campaign.evaluate("../x", targets=LOOSE, settings=settings)


# --------------------------------------------------------------------------- install


def test_install_needs_a_ship_report_and_the_files_eval_verified(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    use_teacher(monkeypatch, oracle)
    outcome = run_label(settings, corpus)
    run_dir = make_run(settings, outcome.dataset)
    home = artifact_home(settings.artifacts_dir, "polarity", "classify")
    with pytest.raises(DistillError, match="aiagent distill eval ftjob-1"):
        campaign.install(JOB, settings=settings)

    report = campaign.evaluate(JOB, targets=LOOSE, settings=settings).report
    weights = run_dir / "model.onnx.data"
    data = weights.read_bytes()
    weights.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ArtifactError, match="model.onnx.data changed since it was verified"):
        campaign.install(JOB, settings=settings)
    assert not (home / CURRENT_FILE).exists()

    weights.write_bytes(data)
    installed = campaign.install(JOB, settings=settings)

    assert installed.path == home / report.artifact_id
    assert dict(installed.thresholds) == {"polarity": report.questions[0].tau}
    assert (installed.run, installed.target_precision) == (JOB, 0.5)
    assert load_installed(settings.artifacts_dir, "polarity", "classify").artifact_id == report.artifact_id


def _as_repair(report: Any, _run_dir: Path) -> Any:
    return dataclasses.replace(report, verdict="repair")


def _source_changed(report: Any, _run_dir: Path) -> Any:
    return dataclasses.replace(report, skill_source_sha256="0" * 64)


def _other_artifact(report: Any, _run_dir: Path) -> Any:
    return dataclasses.replace(report, artifact_id="a-other")


def _no_tau(report: Any, _run_dir: Path) -> Any:
    questions = tuple(dataclasses.replace(q, tau=None) for q in report.questions)
    return dataclasses.replace(report, questions=questions)


def _signature_changed(report: Any, run_dir: Path) -> Any:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["binds"]["signature_sha256"] = "0" * 64
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return report


@pytest.mark.parametrize(
    ("doctor", "match"),
    [
        (_as_repair, "no ship verdict"),
        (_source_changed, "skill polarity changed since eval"),
        (_other_artifact, "artifact a-0123456789ab, not a-other"),
        (_no_tau, "corrupt eval report"),
        (_signature_changed, "signature/questions changed since eval"),
    ],
    ids=["repair", "source", "artifact", "tau", "signature"],
)
def test_install_refuses_what_eval_did_not_certify(
    doctor: Callable[[Any, Path], Any],
    match: str,
    settings: Settings,
    corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    oracle: str,
) -> None:
    use_teacher(monkeypatch, oracle)
    outcome = run_label(settings, corpus)
    run_dir = make_run(settings, outcome.dataset)
    report = campaign.evaluate(JOB, targets=LOOSE, settings=settings).report
    save_report(settings.artifacts_dir, doctor(report, run_dir))

    with pytest.raises(DistillError, match=match):
        campaign.install(JOB, settings=settings)
    assert not artifact_home(settings.artifacts_dir, "polarity", "classify").exists()


# --------------------------------------------------------------------------- repair


def test_repair_labels_the_least_sure_pool_rows_into_a_round_1_dataset(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    teacher = use_teacher(monkeypatch, contrary(oracle))
    outcome = run_label(settings, corpus)
    make_run(settings, outcome.dataset)
    assert campaign.evaluate(JOB, targets=LOOSE, settings=settings).report.verdict == "repair"

    result = campaign.repair(JOB, max_rows=5, concurrency=2, settings=settings)

    assert (result.labeled, result.unlabeled, result.pool_remaining) == (5, 0, 11)
    assert teacher.calls[-1] == {"alias": None, "exact": "dummy"}
    parent = load_dataset(settings.distill_dir, outcome.dataset.id)
    child = load_dataset(settings.distill_dir, result.dataset.id)
    producer = child.manifest.producer
    assert (producer.round, producer.parent_dataset) == (1, outcome.dataset.id)
    assert producer.campaign == parent.manifest.producer.campaign
    assert child.rows["calib"] == parent.rows["calib"]
    assert child.rows["heldout"] == parent.rows["heldout"]
    pool_ids = sorted(r.id for r in parent.rows["pool"])
    added = {r.id for r in child.rows["train"]} - {r.id for r in parent.rows["train"]}
    assert added == set(pool_ids[:5])  # every pool row is as unsure: ties go by row id
    assert sorted(r.id for r in child.rows["pool"]) == pool_ids[5:]

    # the round-1 student is compared with round 0 (same accuracy: no improvement)
    make_run(settings, result.dataset, "ftjob-2")
    second = campaign.evaluate("ftjob-2", targets=LOOSE, settings=settings).report
    assert (second.round, second.previous_run, second.previous_accuracy) == (1, JOB, 0.0)
    assert (second.parent_dataset, second.verdict) == (outcome.dataset.id, "stop")


def test_repair_needs_a_repair_report_and_the_same_base(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch, oracle: str
) -> None:
    use_teacher(monkeypatch, contrary(oracle))
    outcome = run_label(settings, corpus)
    make_run(settings, outcome.dataset)
    with pytest.raises(DistillError, match="no repair verdict"):
        campaign.repair(JOB, max_rows=5, concurrency=2, settings=settings)

    campaign.evaluate(JOB, targets=LOOSE, settings=settings)
    _changed_base(outcome.dataset.path)
    with pytest.raises(DistillError, match="base checkpoint changed"):
        campaign.repair(JOB, max_rows=5, concurrency=2, settings=settings)
