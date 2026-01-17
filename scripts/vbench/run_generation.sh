#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/vbench/run_generation.sh [--limit N] [--dry-run]
#
# Notes:
# - Requires: configs/vbench_eval.yaml to exist
# - Outputs: results/vbench/<exp_name>/videos and logs/generation.log

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure project is importable
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG="configs/vbench_eval.yaml"
LIMIT=""
DRY_RUN="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --limit)
      LIMIT="--limit ${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift 1
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

LOG_DIR="results/vbench/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/generation.log"

CMD=(python3 tools/vbench/run_generation.py --config "$CONFIG")
if [[ -n "$LIMIT" ]]; then
  CMD+=($LIMIT)
fi
if [[ "$DRY_RUN" == "true" ]]; then
  CMD+=(--dry_run)
fi

echo "[run_generation] Project: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "[run_generation] Command: ${CMD[*]}" | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[run_generation] Done." | tee -a "$LOG_FILE"


