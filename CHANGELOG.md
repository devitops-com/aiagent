# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **CLI crash on argument-bearing commands** (issue #1): the shipped bundle
  paired a pre-Click-8.2 Typer with Click 8.2+, whose `Parameter.make_metavar`
  gained a required `ctx` argument — mutually incompatible, so rendering usage or
  help for any command with a positional argument (`run`, `eval`, online
  `doctor`) crashed. Dependency floors are pinned to a compatible era
  (`typer>=0.16`, `click>=8.2`).
- **Built-in skills failing to load from the sourceless bundle** (issue #2):
  skill entry modules were imported by `skill.py` file path, but the bundle
  strips `.py` sources (`.pyc` only), so every built-in raised `entry module
  skill.py not found`. Built-ins now load by dotted import
  (`aiagent.builtin_skills.<dir>.<module>`), which resolves the compiled `.pyc`.
- `aiagent run chat --text "..."` crashing with
  `ChatSkill.forward() got an unexpected keyword argument 'text'`.

### Added
- **Resumable multi-turn chat**: `aiagent chat` persists its conversation history
  to a named session (`--session/-s`, `--new`, `:reset`) under
  `<config>/chat-sessions/` (`AIAGENT_SESSIONS_DIR`), so a conversation can be
  resumed across invocations. History is optional, so `run chat --text` also
  works single-shot.

### Changed
- **Bundle build hardening** against the non-reproducible artifact behind the
  v0.1.0 defect: dependencies install with `--no-cache-dir` (always fetched fresh
  from the configured index), and the build now audits every bundled module
  against `requirements.txt` at both the dist-info **and** imported-`__version__`
  level, failing on any stale/mismatched module.

## [0.1.0] - 2026-07-01

### Added
- Initial project scaffold: uv + Python 3.13, hatchling src-layout, Typer CLI,
  ruff/mypy/bandit/pytest tooling, CI workflows.
- **Config + LLM layer**: pydantic-settings with env-fallback resolution
  (`AIAGENT_*` → `OPENAI_*` → `OLLAMA_HOST` → defaults); a model registry that
  composes devai control-surface model strings (`<model>@<ctx>::<reasoning>`),
  `dspy.LM` construction and per-step routing. DSPy's default ChatAdapter.
- **Core**: typed `ExtractExpense` signature + `ChainOfThought` module; JSONL
  loader with pydantic-validated rows; a dual-use field-accuracy metric (float
  for scoring, strict gate for bootstrapping); a structured `Evaluate` wrapper.
- **Optimization**: `optimize()` over `BootstrapFewShot` (default) / `MIPROv2`,
  with state-only JSON save/load.
- **Skills engine**: Claude-Code-style auto-discovery of `SKILL.md` skills
  (built-in package data + user dir), an immutable registry, a free-text router,
  and dynamic module loading (dspy imported lazily).
- **CLI**: `doctor`, `models list`, `config show`, `skills list`, `run`,
  `optimize`, `eval`, `chat`, and `shell` (the devai picker entrypoint).
- **Packaging**: `make package` → a single **makeself** self-extracting
  `aiagent-install.sh` (linux-x86_64) that includes its own CPython 3.13,
  sourceless-precompiled (`.pyc` only), **zstd -19** compressed with a **bundled
  static zstd** (no Python or zstd needed on the target), SHA256 integrity, and a
  hermetic `-s` launcher. ML libs (numpy/tokenizers/tiktoken) kept for future RAG.
  Trims the Tcl/Tk runtime and hf_xet. ~63 MB.
- Built-in `extract` (self-optimizing expense extraction demo) and `chat` skills.
