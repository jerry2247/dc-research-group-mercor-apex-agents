# =============================================================================
#  apex-agents-bench -- task runner
#  All targets assume the project venv is at ./.venv and is activated by the
#  caller. Use `make setup` from a clean shell to create it.
# =============================================================================

.DEFAULT_GOAL := help

PY        ?= python3.13
VENV      ?= .venv
PIP        = $(VENV)/bin/pip
BIN        = $(VENV)/bin
SHELL     := /bin/bash

# Default per-task output dir. Override with: make smoke MODEL=... OUT=...
OUT       ?= runs

# -----------------------------------------------------------------------------
.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# -----------------------------------------------------------------------------
.PHONY: setup
setup:  ## Create venv, install vendored Archipelago + wrapper + dev tools.
	bash scripts/setup.sh

.PHONY: install
install:  ## Install the wrapper venv and uv-sync the vendored Archipelago subpackages.
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	UV_SKIP_WHEEL_FILENAME_CHECK=1 sh -c 'cd vendor/archipelago/agents  && uv sync --frozen'
	UV_SKIP_WHEEL_FILENAME_CHECK=1 sh -c 'cd vendor/archipelago/grading && uv sync --frozen'

# -----------------------------------------------------------------------------
.PHONY: docker-check
docker-check:  ## Verify Docker daemon is up and the environment image can build.
	bash scripts/docker_check.sh

.PHONY: fetch-dataset
fetch-dataset:  ## Pre-fetch the task/world index from mercor/apex-agents. World zips are pulled per-task.
	bash scripts/fetch_dataset.sh

.PHONY: catalog
catalog:  ## Characterize the dataset; emits data/catalog.json.
	$(BIN)/apex-agents-bench catalog --output data/catalog.json

# -----------------------------------------------------------------------------
.PHONY: smoke
smoke:  ## Smoke-run ONE task end-to-end. Requires: make smoke MODEL=gpt-5.5-medium
	@if [ -z "$(MODEL)" ]; then \
	  echo "error: MODEL is required. Example: make smoke MODEL=gpt-5.5-medium"; \
	  exit 2; \
	fi
	bash scripts/smoke_test.sh "$(MODEL)"

# -----------------------------------------------------------------------------
.PHONY: fmt
fmt:  ## Auto-format the project (ruff format + ruff --fix).
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

.PHONY: lint
lint:  ## Lint without modifying.
	$(BIN)/ruff format --check src tests
	$(BIN)/ruff check src tests

.PHONY: type
type:  ## Static type-check apex_agents_bench.
	$(BIN)/mypy src

.PHONY: test
test:  ## Run pytest (skips docker/network markers by default).
	$(BIN)/pytest -m 'not docker and not network'

.PHONY: test-all
test-all:  ## Run pytest including docker + network markers (requires Docker daemon + keys).
	$(BIN)/pytest

.PHONY: check
check: lint type test  ## Run all checks (lint, type, test).

# -----------------------------------------------------------------------------
.PHONY: clean
clean:  ## Remove caches; preserves venv, data, runs, vendor.
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean  ## Also remove the venv. data/ and runs/ are preserved.
	rm -rf $(VENV)
