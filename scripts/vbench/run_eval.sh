#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/vbench/run_eval.sh [--config path] [--limit N] [--dry-run]

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

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
      LIMIT="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

echo "[run_eval] Step 1: build metadata"
BUILD_CMD=(python3 tools/vbench/build_metadata.py --config "$CONFIG")
if [[ -n "$LIMIT" ]]; then
  BUILD_CMD+=(--limit "$LIMIT")
fi
echo "  ${BUILD_CMD[*]}"
"${BUILD_CMD[@]}"

echo "[run_eval] Step 2: run generation"
GEN_ARGS=(--config "$CONFIG")
if [[ -n "$LIMIT" ]]; then
  GEN_ARGS+=(--limit "$LIMIT")
fi
if [[ "$DRY_RUN" == "true" ]]; then
  GEN_ARGS+=(--dry-run)
fi
bash scripts/vbench/run_generation.sh "${GEN_ARGS[@]}"

echo "[run_eval] Step 3: extract frames"
FRAME_CMD=(python3 tools/vbench/extract_frames.py --config "$CONFIG")
if [[ -n "$LIMIT" ]]; then
  FRAME_CMD+=(--limit "$LIMIT")
fi
if [[ "$DRY_RUN" == "true" ]]; then
  FRAME_CMD+=(--dry_run)
fi
echo "  ${FRAME_CMD[*]}"
"${FRAME_CMD[@]}"

echo "[run_eval] Step 4: objective metrics"
METRIC_CMD=(python3 evaluation/vbench/metrics.py --config "$CONFIG")
if [[ -n "$LIMIT" ]]; then
  METRIC_CMD+=(--limit "$LIMIT")
fi
if [[ "$DRY_RUN" == "true" ]]; then
  METRIC_CMD+=(--dry_run)
fi
echo "  ${METRIC_CMD[*]}"
"${METRIC_CMD[@]}"

echo "[run_eval] Step 5: VBench scores"
VBENCH_CMD=(python3 evaluation/vbench/vbench_runner.py --config "$CONFIG")
if [[ -n "$LIMIT" ]]; then
  VBENCH_CMD+=(--limit "$LIMIT")
fi
if [[ "$DRY_RUN" == "true" ]]; then
  VBENCH_CMD+=(--dry_run)
fi
echo "  ${VBENCH_CMD[*]}"
"${VBENCH_CMD[@]}"

echo "[run_eval] Step 6: latency summary"
python3 evaluation/vbench/latency.py --config "$CONFIG"

echo "[run_eval] Step 7: report"
python3 evaluation/vbench/report.py --config "$CONFIG"

echo "[run_eval] Done."

