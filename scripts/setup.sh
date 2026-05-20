#!/usr/bin/env bash
# =============================================================================
#  apex-agents-bench setup -- create venv, install wrapper, uv-sync vendor.
#  Idempotent: safe to re-run.
# =============================================================================
set -euo pipefail

# Resolve repo root regardless of caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3.13}"
VENV="${VENV:-.venv}"

echo ":: apex-agents-bench setup (root=$ROOT)"

# --- 1. Python version check (Archipelago pins >=3.13,<3.14) -----------------
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)' 2>/dev/null; then
  echo "error: Python 3.13.x required (Archipelago pin). Got: $("$PY" --version 2>&1)" >&2
  echo "       install via: brew install python@3.13   or   pyenv install 3.13" >&2
  exit 2
fi
echo ":: Python OK: $("$PY" --version 2>&1)"

# --- 2. uv check -------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required for managing the vendored Archipelago venvs." >&2
  echo "       install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "       or:      pip install uv" >&2
  exit 2
fi
echo ":: uv OK: $(uv --version 2>&1)"

# --- 3. Docker check (warn but don't fail) -----------------------------------
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo ":: Docker daemon reachable: $(docker --version)"
else
  echo ":: WARNING -- Docker daemon not reachable. The smoke and run commands require Docker."
  echo "             Start Docker Desktop (or `sudo systemctl start docker`) and re-run \`make docker-check\`."
fi

# --- 4. Wrapper venv ---------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  echo ":: creating venv at $VENV (python3.13)"
  "$PY" -m venv "$VENV"
else
  echo ":: venv exists at $VENV"
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo ":: upgrading pip"
pip install --quiet --upgrade pip setuptools wheel

echo ":: installing apex-agents-bench with [dev] extras"
pip install --quiet -e ".[dev]"

# --- 5. Vendor venvs via uv --------------------------------------------------
# UV_SKIP_WHEEL_FILENAME_CHECK=1 works around a known vendor uv.lock bug at the
# pinned commit (3f4a8234): the lockfile pins `litellm==1.83.0` but records the
# 1.81.15 wheel filename. uv's own error message recommends this env var as the
# correct escape hatch. The vendor's actual install resolves to 1.83.0 at run
# time; the lockfile metadata mismatch is cosmetic.
export UV_SKIP_WHEEL_FILENAME_CHECK=1

echo ":: uv sync (vendor/archipelago/agents)"
( cd vendor/archipelago/agents && uv sync --frozen )

echo ":: uv sync (vendor/archipelago/grading)"
( cd vendor/archipelago/grading && uv sync --frozen )

# --- 6. Pre-commit hooks -----------------------------------------------------
if [[ -f .pre-commit-config.yaml ]]; then
  if [[ ! -d .git ]]; then
    echo ":: no .git yet; running git init (initial branch: main)"
    git init -q -b main
  fi
  echo ":: installing pre-commit hooks"
  pre-commit install --install-hooks >/dev/null
fi

# --- 7. .env from .env.example -----------------------------------------------
if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  echo ":: created .env from .env.example (chmod 600). Edit it with real keys before running smoke."
else
  echo ":: .env exists; not overwriting"
fi

# --- 8. Import probe ---------------------------------------------------------
echo ":: import probe"
python -c "
import apex_agents_bench
from apex_agents_bench.config import Settings, DEFAULT_JUDGE_MODEL, RUNS_PER_TASK
from apex_agents_bench.agent_profile import all_profiles
print(f'  apex_agents_bench v{apex_agents_bench.__version__}')
print(f'  judge default: {DEFAULT_JUDGE_MODEL}')
print(f'  runs per task: {RUNS_PER_TASK}')
print(f'  profiles registered: {len(all_profiles())}')
"

echo
echo "setup OK. Next steps:"
echo "  1. \$EDITOR .env       # fill in OPENAI_API_KEY, XAI_API_KEY, HF_TOKEN"
echo "  2. make docker-check   # validate Docker daemon"
echo "  3. make fetch-dataset  # download tasks_and_rubrics.json + world_descriptions.json"
echo "  4. make smoke MODEL=gpt-5.5-medium"
