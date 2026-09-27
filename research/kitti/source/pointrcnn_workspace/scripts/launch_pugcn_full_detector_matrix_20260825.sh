#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/results/pugcn_full_retrain_20260824"
LOG="$RUN_DIR/full_detector_matrix.log"
PID_FILE="$RUN_DIR/full_detector_matrix.pid"

mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Already running: PID $(cat "$PID_FILE")"
  echo "Log: $LOG"
  exit 0
fi

cd "$ROOT"
nohup "$ROOT/scripts/run_pugcn_full_detector_matrix_20260824.sh" \
  >"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "Started PID $pid"
echo "Log: $LOG"
