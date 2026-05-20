#!/usr/bin/env bash
# =============================================================================
#  Fetch the APEX-Agents dataset INDEX into ./data/apex-agents/.
#
#  We do NOT pre-download the 18.7 GB of world zips here -- they're fetched
#  per-task on demand by the runner (see src/apex_agents_bench/world.py).
#  This script just gets the two JSON manifests we need to iterate tasks.
#
#  The dataset is CC-BY-4.0 with an explicit eval-only clause. By running
#  this script you acknowledge that the dataset must not be used for
#  training, fine-tuning, or parameter fitting. See docs/REPRODUCIBILITY.md.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

DEST="${DEST:-data/apex-agents}"
mkdir -p "$DEST"

if [[ -f "$DEST/tasks_and_rubrics.json" && -f "$DEST/world_descriptions.json" ]]; then
  echo ":: dataset index already present at $DEST"
  echo "   tasks:  $(jq length "$DEST/tasks_and_rubrics.json" 2>/dev/null || wc -c <"$DEST/tasks_and_rubrics.json") bytes/items"
  echo "   worlds: $(jq length "$DEST/world_descriptions.json" 2>/dev/null || wc -c <"$DEST/world_descriptions.json") bytes/items"
  exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "error: 'hf' CLI not found. Install with: pip install -U huggingface_hub" >&2
  exit 2
fi

# Authentication: dataset is gated; require an HF login.
if ! hf auth whoami >/dev/null 2>&1; then
  echo "error: not logged in to Hugging Face. Run \`hf auth login\` (or set HF_TOKEN in .env)." >&2
  echo "       Then visit https://huggingface.co/datasets/mercor/apex-agents and click Agree." >&2
  exit 2
fi

echo ":: downloading dataset index via hf"
# Positional filenames (not --include) so hf treats both as explicit downloads.
hf download mercor/apex-agents \
    tasks_and_rubrics.json world_descriptions.json \
    --repo-type dataset \
    --local-dir "$DEST"

# --- Verify ------------------------------------------------------------------
TASKS_FILE="$DEST/tasks_and_rubrics.json"
WORLDS_FILE="$DEST/world_descriptions.json"

if [[ ! -f "$TASKS_FILE" ]]; then
  echo "error: expected $TASKS_FILE after fetch; missing." >&2
  exit 3
fi
if [[ ! -f "$WORLDS_FILE" ]]; then
  echo "error: expected $WORLDS_FILE after fetch; missing." >&2
  exit 3
fi

if command -v jq >/dev/null 2>&1; then
  TASKS=$(jq length "$TASKS_FILE")
  WORLDS=$(jq length "$WORLDS_FILE")
  echo ":: dataset index OK"
  echo "   tasks:  $TASKS"
  echo "   worlds: $WORLDS"
  if [[ "$TASKS" != "480" ]]; then
    echo ":: WARNING -- expected 480 tasks, got $TASKS. Upstream may have moved; verify against the dataset card."
  fi
  if [[ "$WORLDS" != "33" ]]; then
    echo ":: WARNING -- expected 33 worlds, got $WORLDS. Upstream may have moved; verify against the dataset card."
  fi
else
  echo ":: dataset index files written to $DEST"
  echo "   install jq for richer verification:  brew install jq"
fi
