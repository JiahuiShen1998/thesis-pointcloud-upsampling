#!/usr/bin/env bash
set -u

repo="/home/ra87racy/projects/baseline_detectors/PointRCNN"
run="$repo/results/kitti_patch_punet_causal_ablation_v1_20260731/v5_holdout64_surface_validation/pdans_surface_cover_exact_pr1_c32/line_a_original_x4_up"
log="$repo/results/detector_aware_pdans_v5_holdout64_20260805/logs/health_monitor.log"
mkdir -p "$(dirname "$log")"
while tmux has-session -t pdans_v5_holdout64 2>/dev/null; do
  merged=$(find "$run/merged_raw" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l)
  final=$(find "$run/final_bin" -maxdepth 1 -type f -name '*.bin' 2>/dev/null | wc -l)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,pstate --format=csv,noheader 2>&1 | tr '\n' ' ')
  free=$(df -BG --output=avail "$repo" | tail -1 | tr -d ' ')
  printf 'utc=%s merged=%s/64 final=%s/64 gpu="%s" free=%s\n' "$(date -u +%FT%TZ)" "$merged" "$final" "$gpu" "$free" >> "$log"
  sleep 300
done
printf 'utc=%s session=ended\n' "$(date -u +%FT%TZ)" >> "$log"
