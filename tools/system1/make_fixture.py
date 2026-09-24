"""Generate tests/fixtures/system1: a tiny laya student, its ONNX export, golden rows.

This runs in a laya venv, never in aiagent's: torch, laya and transformers are not
aiagent dependencies. The pinned environment (Python 3.12):

    uv venv --python 3.12
    uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
    uv pip install laya==0.3.20 transformers==5.17.0 onnx==1.23.0 onnxscript==0.7.2 \
        onnxruntime==1.30.0 tokenizers==0.23.2

Run it from the repository root (the golden answers must be fp32: LAYA_CPU_AMP unset):

    env -u LAYA_CPU_AMP TMPDIR=/var/tmp <venv>/bin/python \
        tools/system1/make_fixture.py --out tests/fixtures/system1

Steps: a word-level tokenizer over the CASES' own words (mmBERT-style special tokens);
a 2-layer ModernBERT DecisionModel with every weight matrix drawn from N(0, 0.3), so
answers are not uniform; golden rows from laya's torch Agent (seeds 0..50 until every
answer has a clear top-2 margin); a dynamo ONNX export stripped of its stack-trace
metadata, with external data; an onnxruntime self-check against torch; then
deterministic JSON. The work dir is private under --work (default /var/tmp) and removed
on exit, pass or fail. Not linted by `make lint` (tools/ is outside src/ tests/): run
ruff on it by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import laya
import numpy as np
import onnx
import onnxruntime as ort
import onnxscript
import tokenizers
import torch
import transformers
from laya.agent import Agent
from laya.common import DecisionModel, collate_items, render_options, serialize_state
from laya.common import temp_bucket as laya_temp_bucket
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

LAYA_VERSION = "0.3.20"
LAYA_COMMIT = "23a1752"
COMMAND = "tools/system1/make_fixture.py --out tests/fixtures/system1"
SEEDS = range(51)
INIT_STD = 0.3  # at the default 0.02 every answer is uniform and the argmax flaky
MARGIN = 0.002  # every top-2 gap (choice/score) and every |noul - 0.5|
ORT_TOLERANCE = 1e-4
MAX_BYTES = 1_000_000
SPECIALS = ("<pad>", "<unk>", "<bos>", "<eos>", "<mask>")
MASK = "<mask>"
UNKNOWN_WORD = "zzunknown"  # never in the vocab, so <unk> is exercised
VOCAB_CAP = 400
ERROR_TEXT = "options exceed head_max_len"
INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
DYNAMIC_AXES = {
    "input_ids": {0: "b", 1: "s"},
    "attention_mask": {0: "b", 1: "s"},
    "marker_pos": {0: "b", 1: "k"},
    "marker_mask": {0: "b", 1: "k"},
    "qtype": {0: "b"},
    "logits": {0: "b", 1: "k"},
    "act_logits": {0: "b"},
}
HOST_PATHS = (b"/home/", b"/tmp/", b"scratchpad")
# head_max_len 96, not 48: at 48 an option long enough for the 48-token cap triggers the
# budget squeeze first, so the cap is never observable. The last two option buckets are
# clamped by laya (to 0.5 and 5.0).
AGENT_CONFIG = {
    "encoder": "tiny/offline",
    "head_layers": 2,
    "act_costs": {"escalate": 0.5},
    "max_len": 128,
    "head_max_len": 96,
    "temperature": [1.3, 0.8, 1.1],
    "temperature_by_options": {"choice:2": 2.0, "choice:11+": 0.1, "score:6-10": 7.0},
}

# -------------------------------------------------------------------------- the cases

POLARITY = {
    "type": "choice",
    "instructions": "Overall sentiment polarity of the passage.",
    "criteria": ["negative", "neutral", "mixed", "positive"],
}
LATE = {
    "type": "choice",
    "instructions": "Was the delivery late?",
    "criteria": ["no", "yes"],
}
SATISFIED = {
    "type": "score",
    "instructions": "How satisfied is the customer?",
    "criteria": ["unhappy", "neutral", "happy"],
}
URGENCY = {
    "type": "score",
    "instructions": "How urgent is the request, from 1 to 10?",
    "criteria": [str(v) for v in range(1, 11)],
}
REFUND = {"type": "noul", "instructions": "Does the passage mention a refund?"}
FORMAL = {
    "type": "noul",
    "instructions": "Is the tone formal?",
    "criteria": {"True": "formal register throughout", "false": ""},
    "labels": {"false": "B", "true": "A"},
}
MASKED = {"type": "noul", "instructions": "Is the <mask> customer [MASK] happy?"}
DEPARTMENT = {
    "type": "choice",
    "instructions": "Which department should handle it?",
    "criteria": {
        "billing": "invoices, payments and refunds",
        "shipping": None,
        "support": "",
        "legal": 0,
        "other": {"desc": "anything else", "examples": ["spam", "čćš"]},
    },
}
DEPARTMENT_HR = {
    "type": "choice",
    "instructions": {"zadatak": "Koji odjel treba odgovoriti?", "znakovi": "čćšž đ"},
    "criteria": ["prodaja", "podrška", "računovodstvo"],
}
SUMMARY = {  # "detailed" is capped at 48 tokens; the squeeze does not fire
    "type": "choice",
    "instructions": "Which summary fits best?",
    "criteria": {
        "detailed": " ".join(["a very long and careful description"] * 12),
        "short": "brief",
        "none": None,
    },
}
ASPECT = {  # 6 options of about 20 words: the head-budget squeeze (13 tokens each)
    "type": "choice",
    "instructions": "Which aspect dominates the review?",
    "criteria": {
        "speed": "how fast the parcel travelled from the warehouse to the door and "
        "whether the promised delivery date was kept",
        "courier": "how the courier behaved at the door, whether they rang the bell "
        "and whether they were polite to the customer",
        "price": "whether the price was fair for what arrived, including the shipping "
        "fee and any discount that was promised before",
        "quality": "the quality of the product itself once it was unpacked and used "
        "for a few days by the customer at home",
        "packaging": "how well the product was packed, whether the box was damaged "
        "and whether too much plastic was used inside",
        "service": "how the support team answered questions, how long it took and "
        "whether the problem was solved in the end",
    },
}
CATEGORY = {  # 12 short options: the 11+ temperature bucket
    "type": "choice",
    "instructions": "Which product category?",
    "criteria": [
        "books",
        "music",
        "games",
        "garden",
        "kitchen",
        "toys",
        "tools",
        "sports",
        "beauty",
        "health",
        "office",
        "pets",
    ],
}
# laya places 40 one-word options at max_len 128; with 3-word descriptions it cannot.
TOO_MANY = {
    "type": "choice",
    "instructions": "Pick one",
    "criteria": {f"x{i}": "good service here" for i in range(40)},
}

EN_REVIEW = (
    "The parcel arrived two weeks late, but the support team was excellent and "
    "refunded the shipping fee."
)
DE_REVIEW = (
    "Die Lieferung kam zwei Wochen zu spät, aber der Support war hervorragend. Preis: "
    "25 €, Qualität üblich, Rückgabe über das Portal."
)
HR_REVIEW = (
    "Račun od trgovine stigao je kasno i naplaćen je dvaput. Šteta, ali ćemo još "
    "jednom pokušati."
)
LONG_EN = " ".join(
    [
        "I ordered a kettle in March and it came in April with a broken lid.",
        "The courier left it at the wrong door and nobody answered my emails for days.",
        "When support finally called back they were friendly and sent a new lid at "
        "once.",
        "The kettle itself boils water quickly and quietly, which I really like.",
        "Still, the whole story took far too long and cost me two phone calls.",
        "I would buy from this shop again only if the delivery gets better.",
        "My neighbour had the same trouble with a toaster from the same shop last "
        "year.",
        "We both agree that the products are good but the logistics are a mess.",
    ]
)
LONG_DE = " ".join(
    [
        "Ich habe im März einen Wasserkocher bestellt und er kam im April mit "
        "kaputtem Deckel.",
        "Der Kurier hat ihn beim Nachbarn abgegeben und niemand hat auf meine E-Mails "
        "geantwortet.",
        "Als der Support endlich anrief, war er freundlich und hat sofort einen neuen "
        "Deckel geschickt.",
        "Der Wasserkocher selbst ist schnell und leise, das gefällt mir wirklich.",
        "Trotzdem hat die ganze Geschichte viel zu lange gedauert und zwei Anrufe "
        "gekostet.",
    ]
)
CHAT_LONG = [
    f"{who}: {text}"
    for who, text in [
        ("user", "my order 1234 has not arrived yet"),
        ("agent", "sorry to hear that, let me check the tracking"),
        ("user", "it was due last monday and the tracking has not moved"),
        ("agent", "the parcel is stuck at the regional depot"),
        ("user", "can you send a new one or give me my money back"),
        ("agent", "I can offer a replacement shipped today"),
        ("user", "I would rather have a refund at this point"),
        ("agent", "understood, the refund will reach your card in five days"),
        ("user", "thanks, and please close the old order"),
        ("agent", "done, the old order is cancelled"),
        ("user", "one more thing, the invoice shows the wrong address"),
        ("agent", "I will correct the invoice and email it to you"),
        ("user", "great, that was quick in the end"),
    ]
]


def _case(
    case_id: str, state: Any, *questions: tuple[str, dict[str, Any]]
) -> dict[str, Any]:
    return {"id": case_id, "state": state, "questions": dict(questions)}


CASES: list[dict[str, Any]] = [
    _case("g01", EN_REVIEW, ("polarity", POLARITY)),
    _case(
        "g02",
        {"text": DE_REVIEW},
        ("polarity", POLARITY),
        ("late", LATE),
        ("sat", SATISFIED),
    ),
    # a list cut on the left, then a str cut on the right
    _case("g03", CHAT_LONG, ("polarity", POLARITY), ("refund", REFUND)),
    _case("g04", LONG_EN, ("urgency", URGENCY)),
    _case("g05", "", ("refund", REFUND), ("late", LATE)),
    _case("g06", HR_REVIEW, ("polarity", POLARITY), ("dept", DEPARTMENT)),
    _case(
        "g07",
        {
            "text": "Support <mask> was [MASK] great, and the courier was polite.",
            "meta": {"n": 1, "ok": True},
        },
        ("masked", MASKED),
    ),
    _case("g08", "Trebam račun za narudžbu 4471, molim.", ("odjel", DEPARTMENT_HR)),
    _case("g09", "Short and friendly note about a late parcel.", ("summary", SUMMARY)),
    _case("g10", EN_REVIEW, ("aspect", ASPECT)),
    _case(
        "g11",
        "A set of garden tools and a dog bed for the pets.",
        ("category", CATEGORY),
    ),
    _case(
        "g12",
        {"text": "Dear team, the replacement arrived on time. Kind regards."},
        ("sat", SATISFIED),
        ("urgency", URGENCY),
        ("formal", FORMAL),
    ),
    _case(
        "g13",
        "Dear Sir or Madam, I hereby request the cancellation of my contract.",
        ("formal", FORMAL),
    ),
    _case(
        "g14",
        "Fast delivery, great price, the best shop I know!",
        ("polarity", POLARITY),
    ),
    _case("g15", "The courier came a day after the promised date.", ("late", LATE)),
    {**_case("g16", "good service here", ("pick", TOO_MANY)), "error": ERROR_TEXT},
    _case(
        "g17", "The zzunknown parcel arrived late and damaged.", ("polarity", POLARITY)
    ),
    # a dict cut on the right
    _case("g18", {"text": f"{LONG_DE} {LONG_EN}"}, ("refund", REFUND)),
    _case(
        "g19",
        {"text": DE_REVIEW, "lang": "de"},
        ("polarity", POLARITY),
        ("late", LATE),
        ("sat", SATISFIED),
        ("formal", FORMAL),
    ),
    _case(
        "g20",
        ["user: is the shop open on sunday?", "agent: yes, from 10 to 16"],
        ("polarity", POLARITY),
        ("late", LATE),
    ),
]


# ---------------------------------------------------------------- tokenizer and model


def case_texts(case: dict[str, Any]) -> list[str]:
    """The text laya tokenizes for one case: the state, each head and each option."""
    texts = [serialize_state(case["state"]).replace(MASK, " ")]
    for qdef in case["questions"].values():
        q = Agent._to_internal(qdef)
        texts.append(f"{q['t']} question: {str(q['ins']).replace(MASK, ' ')}")
        texts.extend(" " + option.replace(MASK, " ") for option in render_options(q))
    return texts


def build_vocab() -> dict[str, int]:
    """Specials, then each distinct Whitespace pre-token of the cases, in order."""
    vocab = {token: i for i, token in enumerate(SPECIALS)}
    pre = Whitespace()
    for case in CASES:
        for text in case_texts(case):
            for word, _ in pre.pre_tokenize_str(text):
                if (
                    word != UNKNOWN_WORD
                    and word not in vocab
                    and len(vocab) < VOCAB_CAP
                ):
                    vocab[word] = len(vocab)
    return vocab


def save_tokenizer(vocab: dict[str, int], directory: Path) -> None:
    word_level = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    word_level.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=word_level,
        pad_token="<pad>",
        unk_token="<unk>",
        cls_token="<bos>",
        sep_token="<eos>",
        mask_token="<mask>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast.save_pretrained(str(directory))


def save_checkpoint(work: Path, vocab_size: int, seed: int) -> None:
    """encoder/, model.safetensors and rl_agent_config.json for this seed."""
    torch.manual_seed(seed)
    config = ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        global_attn_every_n_layers=2,
        local_attention=16,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        cls_token_id=2,
        sep_token_id=3,
        reference_compile=False,
    )
    config.save_pretrained(str(work / "encoder"))
    model = DecisionModel(ModernBertModel(config), head_layers=2, n_act=2).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                param.normal_(0.0, INIT_STD)
    save_file(model.state_dict(), str(work / "model.safetensors"))
    write_json(work / "rl_agent_config.json", AGENT_CONFIG)


# ------------------------------------------------------------------------ golden rows


def clear_margin(answer: dict[str, Any]) -> bool:
    if answer["type"] == "noul":
        return abs(answer["noul"] - 0.5) >= MARGIN
    top = sorted(answer["probabilities"].values(), reverse=True)
    return top[0] - top[1] >= MARGIN


def golden_row(agent: Any, case: dict[str, Any]) -> dict[str, Any]:
    """laya's ids, markers and answers for one case, or its error."""
    row = {"id": case["id"], "state": case["state"], "questions": case["questions"]}
    try:
        answers = agent.predict(case["state"], case["questions"])["answers"]
    except ValueError as exc:
        if case.get("error") != ERROR_TEXT or ERROR_TEXT not in str(exc):
            raise
        return {**row, "error": ERROR_TEXT}
    if "error" in case:
        raise SystemExit(f"{case['id']}: laya did not raise {ERROR_TEXT!r}")
    qids = list(case["questions"])
    internal = {qid: Agent._to_internal(case["questions"][qid]) for qid in qids}
    items = agent._encode_state(case["state"], qids, internal)
    expected = {
        qid: {
            "input_ids": item["ids"],
            "markers": item["markers"],
            "answer": {k: v for k, v in answers[qid].items() if k != "action"},
        }
        for qid, item in zip(qids, items, strict=True)
    }
    return {**row, "expected": expected}


def find_seed(work: Path, vocab_size: int) -> tuple[int, Any, list[dict[str, Any]]]:
    """The first seed whose golden answers all have a clear margin."""
    for seed in SEEDS:
        save_checkpoint(work, vocab_size, seed)
        agent = laya.load(str(work), device="cpu")
        rows = [golden_row(agent, case) for case in CASES]
        answers = [
            e["answer"] for row in rows for e in row.get("expected", {}).values()
        ]
        if all(clear_margin(answer) for answer in answers):
            return seed, agent, rows
        print(
            f"seed {seed}: an answer is within {MARGIN} of a tie; trying the next",
            file=sys.stderr,
        )
    raise SystemExit(f"no seed in {SEEDS} gives every answer a clear margin")


# ------------------------------------------------------------------ export and checks


def export_onnx(agent: Any, vocab_size: int, work: Path, out: Path) -> None:
    """Export with dynamo=True (TorchScript bakes the traced length into a Reshape)."""
    ids = torch.randint(len(SPECIALS), vocab_size, (2, 320))
    inputs = (
        ids,
        torch.ones(2, 320, dtype=torch.long),
        torch.tensor([[1, 5, 9], [2, 6, 10]]),
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([0, 1]),
    )
    raw = work / "raw.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            agent.model,
            inputs,
            str(raw),
            opset_version=18,
            dynamo=True,
            input_names=INPUT_NAMES,
            output_names=["logits", "act_logits"],
            dynamic_axes=DYNAMIC_AXES,
        )
    model = onnx.load(str(raw))
    for nodes in [model.graph.node, *(function.node for function in model.functions)]:
        for node in nodes:
            del node.metadata_props[:]  # about 485 KB of stack traces
            node.doc_string = ""
    del model.graph.value_info[:]
    del model.metadata_props[:]
    # The default size_threshold: at 0 the Unsqueeze axes move out and ORT refuses it.
    onnx.save_model(
        model,
        str(out / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
    )


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits.astype(np.float64) / temperature
    p = np.exp(z - z.max())
    return p / p.sum()


def ort_max_difference(agent: Any, model_path: Path) -> float:
    """The largest |p_ort - p_torch| over every golden row, laya-style collation."""
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for case in CASES:
        if "error" in case:
            continue
        qids = list(case["questions"])
        internal = {qid: Agent._to_internal(case["questions"][qid]) for qid in qids}
        items = agent._encode_state(case["state"], qids, internal)
        batch = collate_items([items], agent.tok.pad_token_id)
        with torch.no_grad():
            torch_logits = agent.model(*(batch[name] for name in INPUT_NAMES))[
                0
            ].numpy()
        (ort_logits,) = session.run(
            ["logits"], {name: batch[name].numpy() for name in INPUT_NAMES}
        )
        for r, item in enumerate(items):
            k = len(item["markers"])
            bucket = laya_temp_bucket(item["qtype"], k)
            t = agent.temperature_by_options.get(
                bucket, agent.temperature[item["qtype"]]
            )
            diff = np.abs(
                softmax(torch_logits[r, :k], t) - softmax(ort_logits[r, :k], t)
            )
            worst = max(worst, float(diff.max()))
    return worst


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def canonical_json(obj: object) -> str:
    """aiagent.system1.contract.canonical_json (this venv has no aiagent)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def check_out(out: Path) -> None:
    files = [path for path in out.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    if total >= MAX_BYTES:
        raise SystemExit(
            f"{out} holds {total} bytes; the fixture must stay under {MAX_BYTES}"
        )
    for path in files:
        data = path.read_bytes()
        for needle in HOST_PATHS:
            if needle in data:
                raise SystemExit(
                    f"{path.relative_to(out)} contains {needle.decode()!r}"
                )


# ------------------------------------------------------------------------------- main


def generate(work: Path, stage: Path) -> None:
    vocab = build_vocab()
    save_tokenizer(vocab, work / "tokenizer")
    seed, agent, rows = find_seed(work, len(vocab))
    export_onnx(agent, len(vocab), work, stage)
    worst = ort_max_difference(agent, stage / "model.onnx")
    if worst > ORT_TOLERANCE:
        raise SystemExit(
            f"onnxruntime differs from torch by {worst:.2e} > {ORT_TOLERANCE}"
        )
    write_json(stage / "rl_agent_config.json", AGENT_CONFIG)
    (stage / "tokenizer").mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):  # as laya.load left them
        shutil.copyfile(work / "tokenizer" / name, stage / "tokenizer" / name)
    lines = "".join(canonical_json(row) + "\n" for row in rows)
    (stage / "golden.jsonl").write_text(lines, encoding="utf-8")
    provenance = {
        "command": COMMAND,
        "laya": {"version": laya.__version__, "commit": LAYA_COMMIT},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "onnx": onnx.__version__,
        "onnxscript": onnxscript.__version__,
        "onnxruntime": ort.__version__,
        "tokenizers": tokenizers.__version__,
        "seed": seed,
        "init_std": INIT_STD,
        "vocab_size": len(vocab),
        "golden_rows": len(rows),
        "ort_max_difference": float(f"{worst:.2e}"),
    }
    write_json(stage / "provenance.json", provenance)
    for path in stage.rglob("*"):
        path.chmod(
            0o755 if path.is_dir() else 0o644
        )  # onnx writes the external data 0600
    check_out(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, required=True, help="tests/fixtures/system1"
    )
    parser.add_argument(
        "--work", type=Path, default=Path("/var/tmp"), help="scratch parent"
    )
    args = parser.parse_args()
    if laya.__version__ != LAYA_VERSION:
        print(
            f"laya {laya.__version__} found; the fixture pins laya {LAYA_VERSION}",
            file=sys.stderr,
        )
        return 2
    if "LAYA_CPU_AMP" in os.environ:
        print("unset LAYA_CPU_AMP: the golden answers must be fp32", file=sys.stderr)
        return 2
    work = Path(tempfile.mkdtemp(prefix="aiagent-fixture.", dir=args.work))
    try:
        stage = work / "fixture"
        stage.mkdir()
        generate(work, stage)
        args.out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, args.out, dirs_exist_ok=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    size = sum(path.stat().st_size for path in args.out.rglob("*") if path.is_file())
    print(f"wrote {args.out} ({size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
