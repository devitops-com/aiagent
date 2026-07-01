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
  `openai/<model>[@<ctx>]::<reasoning>`. `llm/lm.py` — `build_lm`/`configure_default`/`routing`.
- `core/` — `pipeline.py` (`Pipeline(dspy.Module)` base), `extract.py`
  (`ExtractExpense` + `ExtractExpenseModule`), `evaluate.py`.
- `data/loader.py`, `metrics/extraction.py` (dual-use metric), `optimize/harness.py`.
- `skills/` = the **engine** (base/discovery/registry/loader/router).
  `builtin_skills/` = shipped skill **content** (`extract`, `chat`), package data.
  **Keep the split** — never merge engine and content dirs.
- `cli/` — Typer; `app.py` dispatcher + `main()`; one command per file.

## Commands

`make dev-install` · `make check` (ruff + mypy --strict) · `make test` (hermetic) ·
`make test-cov` (gate 85%) · `make lock` (REQUIRED before packaging) · `make package` ·
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
  `OPENAI_MODEL`/`OLLAMA_DEFAULT_MODEL`/`CONTEXT`) > defaults. `api_key` must be
  **non-empty** (default `"local"`).
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
(**linux-x86_64**, ~63 MB): bundled CPython 3.13, **sourceless** (`.pyc` only),
**zstd -19** payload decompressed by a **bundled static zstd** (target needs no
zstd), SHA256 integrity, `-s` hermetic launcher. `make lock` first. Prefix via
`AIAGENT_PREFIX` (default `~/.local`). Keeps numpy/tokenizers/tiktoken for future
RAG; drops Tcl/Tk + hf_xet; must stay **torch-free** (build guards enforce it).

**Release/distribution.** `make release` (`tools/release/release.sh`) cuts a
versioned GitHub release: version from pyproject → tag `vX.Y.Z`; guards (on `main`,
clean tree, in-sync, tag/release absent) → promote CHANGELOG `[Unreleased]` (empty
refused) → rebuild installer → commit → tag → atomic push → `gh release create`
with **two** assets: `aiagent-install.sh` + `install.sh`. `install.sh` is a POSIX
**bootstrap** — a makeself archive can't be piped to `sh` (it seeks within `$0`),
so it downloads the installer to a temp file (auto-removed via `trap`) and runs it.
Repo `devitops-com/aiagent` is **public**; uv-style install:
`curl -fsSL .../releases/latest/download/install.sh | sh` (honors `AIAGENT_PREFIX`,
`AIAGENT_VERSION`). CI/non-interactive: `AIAGENT_RELEASE_ASSUME_YES=1`.
Scripts: `tools/package/{build-binary.sh, startup.sh.in}`, `tools/release/release.sh`, `install.sh`.

## Testing

Hermetic by default. LLM-driven CLI commands use **`dspy.utils.DummyLM`**
(monkeypatch each command's `configure_lm`); `doctor`/`models` use an httpx mock.
`tests/conftest.py::clean_env` (autouse) neutralizes env/TOML/skills-dir. Live
devai tests: `pytest -m live` (run inside `devai-net`).

## Scope (MVP) / not yet

In: extraction demo, skills, optimize/eval, chat, packaging. **Deferred:** MCP,
RAG/embeddings (libs kept, not wired), weight finetuning (no torch).

## References

Approved plan: `~/.claude/plans/expense-note-is-ok-tingly-nebula.md`.
Session memory: `~/.claude/projects/-home-sparavec-git-aiagent/memory/`.

