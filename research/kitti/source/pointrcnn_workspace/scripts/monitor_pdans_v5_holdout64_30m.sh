#!/usr/bin/env bash
set -u

repo="/home/ra87racy/projects/baseline_detectors/PointRCNN"
root="$repo/results/detector_aware_pdans_v5_holdout64_20260805"
run_log="$root/logs/resume_jointsplit_v2.log"
report_log="$root/logs/half_hour_status.log"
run_session="pdans_v5_holdout64_resume"

mkdir -p "$root/logs"

while true; do
  timestamp="$(date -u +%FT%TZ)"
  if tmux has-session -t "$run_session" 2>/dev/null; then
    session_state="running"
  else
    session_state="stopped"
  fi

  {
    printf '=== HOLDOUT64 STATUS %s ===\n' "$timestamp"
    printf 'session=%s\n' "$session_state"
    if [[ -f "$run_log" ]]; then
      grep -E '^(pipeline=|stage=|SPLIT_REGION_SOURCE_PASS|START CenterPoint|END CenterPoint|SKIP completed)' "$run_log" | tail -n 24
      tail -n 8 "$run_log"
    else
      printf 'run_log=missing\n'
    fi
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
    printf '\n'
  } >> "$report_log"

  if grep -q '^pipeline=holdout64_resume_jointsplit_v2 status=completed' "$run_log" 2>/dev/null; then
    exit 0
  fi
  if [[ "$session_state" == "stopped" ]]; then
    exit 1
  fi
  sleep 1800
done
