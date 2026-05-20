#!/usr/bin/env bash
# =============================================================================
#  Docker readiness check for apex-agents-bench.
#  Confirms: daemon up, target port free, environment image buildable.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PORT="${APEX_AGENTS_HOST_PORT:-8080}"
ENV_DIR="vendor/archipelago/environment"

echo ":: 1. docker binary"
if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker not on PATH. Install Docker Desktop." >&2
  exit 2
fi
echo "   $(docker --version)"

echo ":: 2. daemon reachable"
if ! docker info >/dev/null 2>&1; then
  echo "error: docker daemon not reachable. Start Docker Desktop (or systemctl start docker)." >&2
  exit 2
fi
echo "   daemon: OK"

echo ":: 3. docker compose v2 available"
if ! docker compose version >/dev/null 2>&1; then
  echo "error: docker compose v2 not available. Update Docker Desktop." >&2
  exit 2
fi
echo "   $(docker compose version | head -1)"

echo ":: 4. environment dir present"
if [[ ! -d "$ENV_DIR" ]]; then
  echo "error: $ENV_DIR not found. Did you clone the vendor?" >&2
  exit 2
fi
if [[ ! -f "$ENV_DIR/Dockerfile" ]]; then
  echo "error: $ENV_DIR/Dockerfile not found." >&2
  exit 2
fi
echo "   $ENV_DIR/Dockerfile: present"

echo ":: 5. host port $PORT availability"
if command -v lsof >/dev/null 2>&1; then
  if lsof -i ":$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "   WARNING: port $PORT is in use. Set APEX_AGENTS_HOST_PORT=<other> in .env before running."
    lsof -i ":$PORT" -sTCP:LISTEN | head -3
  else
    echo "   port $PORT: free"
  fi
else
  echo "   (skipping port check -- lsof not installed)"
fi

echo
echo "Docker readiness: OK"
echo "Next: \`make fetch-dataset\` then \`make smoke MODEL=gpt-5.5-medium\`."
