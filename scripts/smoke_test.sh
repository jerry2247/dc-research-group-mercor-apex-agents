#!/usr/bin/env bash
# =============================================================================
#  scripts/smoke_test.sh MODEL [extra args forwarded to apex-agents-bench smoke]
#  Runs ONE APEX-Agents task end-to-end. Verifies the full pipeline works.
#
#  Usage:
#    make smoke MODEL=gpt-5.5-medium
#    bash scripts/smoke_test.sh gpt-5.5-medium
#    bash scripts/smoke_test.sh grok-4.3-low --domain banking
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL [extra args forwarded to apex-agents-bench smoke]" >&2
  exit 2
fi

MODEL="$1"
shift || true

if [[ ! -d .venv ]]; then
  echo "error: .venv not found. Run 'make setup' first." >&2
  exit 2
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ ! -f .env ]]; then
  echo "error: .env not found. Run 'make setup' and fill in keys." >&2
  exit 2
fi
set -a
# shellcheck disable=SC1091
. ./.env
set +a

if [[ ! -f data/apex-agents/tasks_and_rubrics.json ]]; then
  echo "error: dataset index not found. Run 'make fetch-dataset'." >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "error: Docker daemon not reachable. Start Docker Desktop, then 'make docker-check'." >&2
  exit 2
fi

echo ":: smoke run (agent profile: $MODEL)"
apex-agents-bench smoke --model "$MODEL" "$@"
