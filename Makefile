SHELL := /bin/bash

# Single source of truth for the Python version (dev venv, CI, and the bundled
# installer all derive from .python-version — edit it there only).
PYTHON_VERSION := $(shell cat .python-version)

# Prefer .venv/bin/* when present (dev-install), else fall back to PATH.
PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTEST := $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,pytest)
RUFF   := $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
MYPY   := $(if $(wildcard .venv/bin/mypy),.venv/bin/mypy,mypy)
BANDIT := $(if $(wildcard .venv/bin/bandit),.venv/bin/bandit,bandit)

.PHONY: help dev-install lock test test-cov lint lint-fix typecheck check package release clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

# --- Setup ---

dev-install: ## Set up .venv with aiagent + dev dependencies (editable)
	uv venv --python $(PYTHON_VERSION) .venv 2>/dev/null || true
	uv pip install --python .venv/bin/python -e ".[dev]"

lock: ## Regenerate requirements.txt and requirements-dev.txt from pyproject.toml
	uv pip compile pyproject.toml --generate-hashes -o requirements.txt
	uv pip compile pyproject.toml --extra dev --generate-hashes -o requirements-dev.txt

# --- Testing ---

test: ## Run tests (hermetic; excludes -m live)
	$(PYTEST) tests/

test-cov: ## Run tests with coverage (term-missing)
	$(PYTEST) tests/ --cov=aiagent --cov-report=term-missing

# --- Static checks ---

lint: ## Run ruff linter
	$(RUFF) check src/ tests/

lint-fix: ## Run ruff with auto-fix
	$(RUFF) check --fix src/ tests/

typecheck: ## Run mypy (strict)
	$(MYPY) src/

check: lint typecheck ## Static checks: ruff + mypy

# --- Packaging ---

package: ## Build the self-contained, precompiled installer -> dist/aiagent-install.sh
	@bash tools/package/build-binary.sh

# --- Release ---

release: ## Cut a release: promote CHANGELOG, rebuild installer, tag + push, publish GitHub release (version from pyproject.toml)
	@bash tools/release/release.sh

# --- Cleanup ---

clean: ## Remove generated and cached files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage .coverage.*
	rm -rf build dist *.egg-info src/*.egg-info
	rm -rf .venv
	@echo "Clean complete"
