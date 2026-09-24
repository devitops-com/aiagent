# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project: aiagent

Project-specific facts a session needs here (the generic guidance above still applies).

## What this is / isn't

A programmatic **DSPy** (pinned `dspy==3.2.1`) agent framework: query/prompt
optimization, goal-reaching loops, and autonomous data processing over an
OpenAI-compatible local-LLM router. **Not a chat UI** — `chat` is a minor feature.
MVP demo = self-optimizing expense extraction (`{merchant, date, amount}`).

## Architecture (src-layout, `src/aiagent/`)

- `config.py` — pydantic-settings; **env-fallback resolution** (see invariants).
- `llm/registry.py` (pure, no dspy) → `compose_model_string` produces
  `openai/<model>::<reasoning>[@<ctx>]` (`@<ctx>` outermost/last, per the devai
  router's right-to-left parse). `llm/lm.py` — `build_lm`/`configure_default`/`routing`.
- `core/` — `pipeline.py` (`Pipeline(dspy.Module)` base), `extract.py`
  (`ExtractExpense` + `ExtractExpenseModule`), `evaluate.py`.
- `data/loader.py`, `metrics/extraction.py` (dual-use metric), `optimize/harness.py`.
- `skills/` = the **engine** (base/discovery/registry/loader/router).
  `builtin_skills/` = shipped skill **content** (`extract`, `chat`), package data.
  **Keep the split** — never merge engine and content dirs.
- `cli/` — Typer; `app.py` dispatcher + `main()`; one command per file.

## Commands

`make dev-install` (fresh `.venv` on exactly `.python-version`) · `make check` (ruff + mypy --strict) ·
`make test` (hermetic) ·
`make test-cov` (gate 85%) · `make lock` (REQUIRED before packaging; writes the hashed
`requirements.txt`, `requirements-dev.txt` and `requirements-build.txt`; keeps existing pins —
`LOCK_ARGS='--upgrade-package X'` moves one) · `make package` ·
`make release` (tag + publish GitHub release; version from pyproject).

## Invariants & gotchas (don't break these)

- **Lazy dspy in the CLI.** `import aiagent.cli.app` must NOT import `dspy`
  (a subprocess test enforces it). `run/eval/optimize/chat` and the online half of
  `doctor`/`models` import dspy / `llm.lm` / `core.evaluate` / `data.loader` /
  `optimize.harness` **inside the function body**, never at module top. `_runtime.py`
  is import-safe (dspy only inside `configure_lm`).
- **dspy ships no type stubs.** `dspy.Module`/`dspy.Signature` subclasses need
  `# type: ignore[misc]`; there's a mypy `dspy.*` override and an `exclude` for
  `builtin_skills/` (exec-loaded plugins with same-named `skill.py`).
- **Config precedence:** `AIAGENT_*` env > TOML (`~/.config/aiagent/config.toml`) >
  devai-injected env (`OPENAI_BASE_URL`/`OLLAMA_HOST`+`/v1`/`OPENAI_API_KEY`/
  `OPENAI_MODEL`/`OLLAMA_DEFAULT_MODEL`/`CONTEXT`/`HTTPS_PROXY`/`HTTP_PROXY`) >
  defaults. `api_key` must be **non-empty** (default `"local"`). `proxy_url`
  (default `http://devai-pipelock:8888`) is the forward proxy for outbound URL
  fetches (empty string = direct); httpx trusts the pipelock MITM CA via the
  system store (`SSL_CERT_FILE`), so no `verify=False`.
- **Adapter:** DSPy default **ChatAdapter**; do NOT rely on native tool-calling or
  server JSON mode (devai backends strip/ignore them).
- **Optimizers:** `from dspy.teleprompt import BootstrapFewShot, MIPROv2`. Persist
  state-only JSON: `save(path, save_program=False)`.
- **Metric contract:** `metric(example, prediction, trace=None) -> float|bool` —
  float when `trace is None`, strict bool gate otherwise.

## Backend (devai)

OpenAI-compatible router at `http://devai-router:11434/v1`, reachable only inside
the external `devai-net` bridge network; no auth; cold starts up to minutes
(`request_timeout_s=900`). `aiagent doctor` = pre-flight; `aiagent models list`
shows the real served tag (registry default is a **placeholder**). aiagent is a
devai agent registered in devai's `model-picker.py`; `aiagent shell` is the picker
entrypoint (banner + `exec $SHELL`).

## Packaging

`make package` → **makeself** self-extractor `dist/aiagent-install.sh`
(**linux-x86_64**, ~63 MB): bundled CPython **exactly** `.python-version` (3.14.7;
X.Y.Z, the single source of truth for the dev venv, CI, the locks'
`--python-version` and the bundle; `requires-python` keeps the 3.14 floor), **no
libpython** (PBS links it statically into `bin/python3.14`; the shared
`libpython3.14.so*`, `libpython3.so` and `lib/pkgconfig` are for embedding only and
dropped — `check-python.sh` fails the build if any `libpython*` is left or any ELF
NEEDs one, `readelf -d`), **sourceless** (`.pyc` only; any compile error fails the
build — the `.py` files are deleted next, so a module that did not compile would
silently be missing),
**zstd -19** payload decompressed by a **bundled static zstd** (target needs no
zstd; built once from checksummed source and cached per version as
`.cache/aiagent-build/zstd-<ver>-static-x86_64`, reused only while it reports that
version and has no program interpreter), SHA256 integrity, `-I` (isolated)
launcher — ignores `PYTHONPATH`,
`PYTHONHOME`, user site and cwd, so a user skill's `<module>:<attr>` metric can't
come from `PYTHONPATH`. `make lock` first. Prefix via `AIAGENT_PREFIX` or
`-- --prefix DIR` (default `~/.local`; `--target` is makeself's own option and is
refused). Keeps numpy/tokenizers/tiktoken for future RAG; drops Tcl/Tk, hf_xet,
and the **AWS/Bedrock subtree** (boto3 + botocore + s3transfer + deps — litellm
makes boto3 a core dep since 1.98 but imports it
lazily, and the local router never takes that path); must stay **torch-free**
(build guards enforce it). The strip set lives once in `STRIP_ABSENT`
(`build-binary.sh`) and feeds both the removal and the audit's allow-list.
Deps install **hash-checked** from `requirements.txt` (`--require-hashes`, wheels only,
`--no-deps`; the aiagent wheel `--no-deps` too) with **`--no-cache-dir`** (always
fresh from the configured index); `pip check` must then report nothing, before pip
and the `STRIP_ABSENT` strips go (those leave boto3/hf-xet unmet on purpose).
The aiagent wheel is built by the hash-pinned backend of `requirements-build.txt`
(`uv build --build-constraints … --require-hashes`), never one freshly resolved.
The build then installs to a temp prefix and **audits every module against
`requirements.txt`** at both the dist-info **and** imported-`__version__` level
(`tools/package/verify-versions.py`, `STRIP_ABSENT` allow-listed), failing on any
stale module — the reproducibility guard for the v0.1.0 metadata/code split.

**No build-host paths in the payload.** The sysconfig data goes back to PBS's
`/install` and the launcher gets a `#!/install/bin/python3.14` placeholder (the
installer rewrites it); `tools/package/check-host-paths.py` (`tests/test_host_paths.py`)
then fails the build when a staged file or symlink target contains the uv CPython's
path, the checkout or `$HOME/`, naming file and pattern. Exempt (only noted): a file
byte-identical to what a wheel pinned in `requirements.txt` shipped (sha256 match in
that distribution's `*.dist-info/RECORD`) — on a runner (`HOME=/home/runner`) five
upstream files name their own CI checkout under `/home/runner/work/`: the SBOMs of
jiter, pydantic_core, rpds_py and tokenizers, and `tokenizers.abi3.so`. Still scanned
although a RECORD hashes them: every file of the aiagent wheel (built from the
checkout; hatchling packs untracked files), pip's `INSTALLER`/`REQUESTED`/
`direct_url.json`, every `../` entry (pip's launchers); a non-UTF-8-CSV RECORD
exempts nothing.

**Exact CPython.** The build stages uv's managed interpreter
(`uv python find --system --managed-python`, which must resolve under `uv python dir`),
never the project `.venv` (plain `uv python find` returns it, with whatever patch it
was made on); `tools/package/check-python.sh` (`tests/test_bundle_python.py`) fails the
build unless it reports exactly `.python-version`. Paths and the baked `PYVER` use X.Y.
`make dev-install` recreates `.venv` on the pin (`uv venv --clear`) and a test fails on
any other interpreter. A patch bump may need a newer uv (it only knows CPython patches
published before it); the build then stops with uv's own error and says so.

**Installer invariants** (`tests/test_installer.py`; the smoke test re-checks the
built one): payload `root:root`, no group/other write (build gate), extracted with
`--no-same-owner` under `umask 022` so a root install is root-owned and
world-readable; relative prefix → `$USER_PWD`, quoted `~` → `$HOME`, whitespace or
over-long (127-byte shebang) prefix refused **before** unpacking; the staged
interpreter must run (`-I -c 'import aiagent'`) before an existing install is
replaced; makeself runs `sh ./startup.sh` and zstd runs from the stage, so a
noexec `$TMPDIR` works; the extraction dir defaults to `/var/tmp` (patched
makeself header), never `/tmp`. Smoke test also checks `aiagent version` ==
pyproject version, with a hostile `PYTHONPATH`/`PYTHONHOME` too, and file modes.

**Build temp files: never `/tmp`.** uv/pip unpacking, makeself's archive, the
static-zstd build and the smoke install, probe and hostile-env files live in one
private `/var/tmp/aiagent-build.XXXXXX` (exported as `TMPDIR`), removed on exit, pass
or fail; `dist/` keeps only the staging trees. The smoke test runs with `HOME` there
and every `AIAGENT_*` unset, so the maintainer's `~/.config/aiagent` (config.toml,
user skills) and env cannot change its result.

**Release/distribution.** `make release` (`tools/release/release.sh`) cuts a
versioned GitHub release: version from pyproject → tag `vX.Y.Z`; guards (on `main`,
clean tree incl. untracked files and assume-unchanged/skip-worktree entries,
in-sync, tag/release absent) → promote CHANGELOG `[Unreleased]` (empty
refused) → rebuild installer → commit → tag → atomic push → `gh release create`
with **two** assets: `aiagent-install.sh` + `install.sh`. `install.sh` is a POSIX
**bootstrap** — a makeself archive can't be piped to `sh` (it seeks within `$0`),
so it downloads the installer to a temp file (auto-removed via `trap`) and runs it.
It downloads HTTPS-only with curl (`--proto '=https' --tlsv1.2`, redirects included;
the wget fallback cannot enforce that) and stages under `$TMPDIR` (default `/var/tmp`).
Repo `devitops-com/aiagent` is **public**; uv-style install:
`curl -fsSL .../releases/latest/download/install.sh | sh` (honors `AIAGENT_PREFIX`,
`AIAGENT_VERSION`). CI/non-interactive: `AIAGENT_RELEASE_ASSUME_YES=1`. The installer
is the **only distribution**: aiagent is not on PyPI, and the `Private :: Do Not Upload`
classifier (pinned by a test) makes PyPI reject an accidental upload.
Scripts: `tools/package/{build-binary.sh, startup.sh.in, check-python.sh, check-host-paths.py}`,
`tools/release/release.sh`, `install.sh`.

## Testing

Hermetic by default. LLM-driven CLI commands use **`dspy.utils.DummyLM`**
(monkeypatch each command's `configure_lm`); `doctor`/`models` use an httpx mock.
`tests/conftest.py::clean_env` (autouse) neutralizes env/TOML/skills-dir. Live
devai tests: `pytest -m live` (run inside `devai-net`).

## Scope (MVP) / not yet

In: extraction demo, sentiment analysis (files/URLs/text via the `ingest` layer),
skills, optimize/eval, chat, packaging. **Deferred:** MCP, RAG/embeddings (libs
kept, not wired), weight finetuning (no torch).

## References

Approved plan: `~/.claude/plans/expense-note-is-ok-tingly-nebula.md`.
Session memory: `~/.claude/projects/-home-sparavec-git-aiagent/memory/`.

