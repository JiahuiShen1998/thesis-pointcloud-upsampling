#!/usr/bin/env bash
set -euo pipefail

repo="/home/ra87racy/projects/baseline_detectors/PointRCNN"
python_bin="/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python"
root="$repo/results/detector_aware_pdans_v5_holdout64_20260805"
causal="$repo/results/kitti_patch_punet_causal_ablation_v1_20260731"
split="$causal/splits/v5_holdout64.txt"
protocol="$root/holdout64_protocol.json"
observed="$repo/data/KITTI/object/training/velodyne_original_val"
prepared="$root/prepared"
point_compare="$root/detectors/pointrcnn_jointsplit_v2"
center_compare="$root/detectors/centerpoint"
log="$root/logs/resume_jointsplit_v2.log"

mkdir -p "$root/logs"
exec > >(tee -a "$log") 2>&1
cd "$repo"

printf 'pipeline=holdout64_resume_jointsplit_v2 status=started utc=%s\n' "$(date -u +%FT%TZ)"

# Derive one fresh set of boundaries jointly from baseline and V4.  The old
# failure reused baseline-only boundaries, so frame 000020 overflowed after the
# frozen generated tail was appended.  Joint derivation keeps the native 16384
# cap and retains every core point from both sources.
printf 'stage=joint_split_prepare status=started utc=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" scripts/prepare_pointrcnn_split_region_inputs.py \
  --workspace "$point_compare" --split-file "$split" \
  --source "original_baseline=$observed" \
  --source "v4_existing_voxel_pointrcnn=$prepared/v4/inputs/v4_existing_voxel_pointrcnn" \
  --shared-observed-prefix "$observed" \
  --halo-m 3 --halo-cap-policy balanced_sample --halo-voxel-m 0.2 \
  --split-policy density_valley --guide-source original_baseline --boundary-guard-m 1
printf 'stage=joint_split_prepare status=completed utc=%s\n' "$(date -u +%FT%TZ)"

printf 'stage=pointrcnn_baseline status=started utc=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" scripts/run_pointrcnn_split_region_eval.py \
  --workspace "$point_compare" --source original_baseline
printf 'stage=pointrcnn_baseline status=completed utc=%s\n' "$(date -u +%FT%TZ)"

printf 'stage=pointrcnn_v4 status=started utc=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" scripts/run_pointrcnn_split_region_eval.py \
  --workspace "$point_compare" --source v4_existing_voxel_pointrcnn
printf 'stage=pointrcnn_v4 status=completed utc=%s\n' "$(date -u +%FT%TZ)"

printf 'stage=centerpoint_baseline status=started utc=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" scripts/run_patch_causal_centerpoint_eval.py \
  --name holdout64_original_baseline --line line_a --source-mode baseline \
  --protocol-json "$protocol" --result-root "$center_compare" \
  --batch-size 1 --workers 2 --resume
printf 'stage=centerpoint_baseline status=completed utc=%s\n' "$(date -u +%FT%TZ)"

printf 'stage=centerpoint_v5 status=started utc=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" scripts/run_patch_causal_centerpoint_eval.py \
  --name holdout64_v5_car_predicted_existing_voxel_centerpoint --line line_a --source-mode direct \
  --predicted-dir "$prepared/v5/inputs/v5_car_predicted_existing_voxel_centerpoint" \
  --protocol-json "$protocol" --result-root "$center_compare" \
  --batch-size 1 --workers 2 --resume
printf 'stage=centerpoint_v5 status=completed utc=%s\n' "$(date -u +%FT%TZ)"

printf 'pipeline=holdout64_resume_jointsplit_v2 status=completed utc=%s\n' "$(date -u +%FT%TZ)"
