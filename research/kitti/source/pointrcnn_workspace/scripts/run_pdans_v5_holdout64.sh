#!/usr/bin/env bash
set -euo pipefail

repo="/home/ra87racy/projects/baseline_detectors/PointRCNN"
python_bin="/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python"
root="$repo/results/kitti_patch_punet_causal_ablation_v1_20260731"
run_kind="v5_holdout64_surface_validation"
variant="pdans_surface_cover_exact_pr1_c32"
line="line_a_original_x4_up"
run_root="$root/$run_kind/$variant/$line"

cd "$repo"
mkdir -p "$run_root/logs"
printf 'stage=pdans_holdout64 status=started utc=%s\n' "$(date -u +%FT%TZ)" | tee -a "$run_root/logs/pipeline.log"
"$python_bin" -u scripts/run_patch_causal_upsampling.py \
  --run-kind "$run_kind" \
  --frames-file "$root/splits/v5_holdout64.txt" \
  --method pdans \
  --variant-name "$variant" \
  --patch-selection fps_ball_cover_knn_v3 \
  --patch-num-ratio 1 \
  --ball-radius-m 2 \
  --min-ball-points 2048 \
  --cover-min-points 32 \
  --cover-radius-m 6 \
  --lines "$line" \
  --drop-merged-after-final \
  2>&1 | tee -a "$run_root/logs/pipeline.log"
"$python_bin" scripts/verify_patch_causal_strict_x4.py \
  --frames-file "$root/splits/v5_holdout64.txt" \
  --observed-dir "$repo/data/KITTI/object/training/velodyne_original" \
  --predicted-dir "$run_root/final_bin" \
  --output-json "$run_root/reports/strict_x4_independent_summary.json" \
  --output-csv "$run_root/reports/strict_x4_independent_frames.csv" \
  2>&1 | tee -a "$run_root/logs/pipeline.log"
printf 'stage=pdans_holdout64 status=completed utc=%s\n' "$(date -u +%FT%TZ)" | tee -a "$run_root/logs/pipeline.log"
