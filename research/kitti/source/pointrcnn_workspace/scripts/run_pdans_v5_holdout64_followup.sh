#!/usr/bin/env bash
set -euo pipefail

repo="/home/ra87racy/projects/baseline_detectors/PointRCNN"
python_bin="/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python"
root="$repo/results/detector_aware_pdans_v5_holdout64_20260805"
causal="$repo/results/kitti_patch_punet_causal_ablation_v1_20260731"
split="$causal/splits/v5_holdout64.txt"
protocol="$root/holdout64_protocol.json"
pdans="$causal/v5_holdout64_surface_validation/pdans_surface_cover_exact_pr1_c32/line_a_original_x4_up"
strict_report="$pdans/reports/strict_x4_independent_summary.json"
observed="$repo/data/KITTI/object/training/velodyne_original_val"
baseline_split="$root/pointrcnn_baseline_split"
point_proposals="$root/internal_proposals/pointrcnn_rpn"
center_proposals="$root/internal_proposals/centerpoint_prenms"
prepared="$root/prepared"
point_compare="$root/detectors/pointrcnn"
center_compare="$root/detectors/centerpoint"

mkdir -p "$root/logs"
exec > >(tee -a "$root/logs/followup.log") 2>&1
cd "$repo"
printf 'stage=wait_pdans status=started utc=%s\n' "$(date -u +%FT%TZ)"
while [[ ! -f "$strict_report" ]]; do
  if ! tmux has-session -t pdans_v5_holdout64 2>/dev/null; then
    printf 'stage=wait_pdans status=failed reason=pdans_session_ended_without_report utc=%s\n' "$(date -u +%FT%TZ)"
    exit 1
  fi
  sleep 60
done
"$python_bin" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="PASS" and p["frames_pass"]==64' "$strict_report"
printf 'stage=wait_pdans status=completed utc=%s\n' "$(date -u +%FT%TZ)"

printf 'stage=baseline_regions status=started utc=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" scripts/prepare_pointrcnn_split_region_inputs.py \
  --workspace "$baseline_split" --split-file "$split" \
  --source "original_baseline=$observed" --source "baseline_shadow=$observed" \
  --halo-m 3 --halo-cap-policy balanced_sample --halo-voxel-m 0.2 \
  --split-policy density_valley --guide-source original_baseline --boundary-guard-m 1
printf 'stage=baseline_regions status=completed utc=%s\n' "$(date -u +%FT%TZ)"

printf 'stage=internal_proposals status=started utc=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" scripts/export_pointrcnn_rpn_proposals.py \
  --source-workspace "$baseline_split" --output "$point_proposals" --limit 0
/home/ra87racy/miniconda3/envs/openpcdet_centerpoint/bin/python \
  scripts/export_centerpoint_prenms_proposals.py \
  --protocol "$protocol" --output "$center_proposals" --limit 0 --workers 2
printf 'stage=internal_proposals status=completed utc=%s\n' "$(date -u +%FT%TZ)"

printf 'stage=prepare_v4_v5 status=started utc=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" scripts/prepare_detector_aware_pdans_v4_inputs.py \
  --workspace "$prepared/v4" --split-file "$split" --observed "$observed" \
  --pdans "$pdans/final_bin" --point-proposals "$point_proposals/merged_kitti" \
  --center-proposals "$center_proposals/npz" --limit 0
"$python_bin" scripts/prepare_detector_aware_pdans_v5_inputs.py \
  --workspace "$prepared/v5" --split-file "$split" --observed "$observed" \
  --pdans "$pdans/final_bin" --proposals "$center_proposals/npz" --limit 0
printf 'stage=prepare_v4_v5 status=completed utc=%s\n' "$(date -u +%FT%TZ)"

printf 'stage=pointrcnn status=started utc=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" scripts/prepare_pointrcnn_split_region_inputs.py \
  --workspace "$point_compare" --split-file "$split" \
  --source "original_baseline=$observed" \
  --source "v4_existing_voxel_pointrcnn=$prepared/v4/inputs/v4_existing_voxel_pointrcnn" \
  --fixed-regions "$baseline_split/manifests/regions.csv" --shared-observed-prefix "$observed" \
  --halo-m 3 --halo-cap-policy balanced_sample --halo-voxel-m 0.2 \
  --split-policy density_valley --guide-source original_baseline --boundary-guard-m 1
"$python_bin" scripts/run_pointrcnn_split_region_eval.py --workspace "$point_compare" --source original_baseline
"$python_bin" scripts/run_pointrcnn_split_region_eval.py --workspace "$point_compare" --source v4_existing_voxel_pointrcnn
printf 'stage=pointrcnn status=completed utc=%s\n' "$(date -u +%FT%TZ)"

printf 'stage=centerpoint status=started utc=%s\n' "$(date -u +%FT%TZ)"
"$python_bin" scripts/run_patch_causal_centerpoint_eval.py \
  --name original_baseline --line line_a --source-mode baseline \
  --protocol-json "$protocol" --result-root "$center_compare" --batch-size 1 --workers 2 --resume
"$python_bin" scripts/run_patch_causal_centerpoint_eval.py \
  --name v5_car_predicted_existing_voxel_centerpoint --line line_a --source-mode direct \
  --predicted-dir "$prepared/v5/inputs/v5_car_predicted_existing_voxel_centerpoint" \
  --protocol-json "$protocol" --result-root "$center_compare" --batch-size 1 --workers 2 --resume
printf 'stage=centerpoint status=completed utc=%s\n' "$(date -u +%FT%TZ)"
printf 'pipeline_status=completed utc=%s\n' "$(date -u +%FT%TZ)"
