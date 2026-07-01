# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
