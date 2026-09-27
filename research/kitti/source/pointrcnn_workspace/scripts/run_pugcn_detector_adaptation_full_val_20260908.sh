#!/usr/bin/env bash
# Reproducible full-3769 KITTI validation for PU-GCN detector adaptation.
#
# Stages are resumable. Every generator/evaluator validates its own coverage,
# and existing PASS artifacts are reused. No historical result directory is
# overwritten.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT="$ROOT/results/pugcn_detector_adaptation_full_val_20260908"
GEN="$RESULT/generation"
VAL="$ROOT/data/KITTI/ImageSets/val.txt"
ORIGINAL="$ROOT/data/KITTI/object/training/velodyne_original_val"
SPARSE="$ROOT/results/kitti_unified_x4_current_methods_no_detector/downsampled_x4/velodyne_downsampled_x4_val"
OLD_EXP="$ROOT/results/pugcn_full_retrain_20260824"
OFFICIAL_POINT="$ROOT/tools/PointRCNN.pth"
OFFICIAL_CENTER="$ROOT/external/centerpoint_hpc_bundle_20260717/outputs/checkpoints/CenterPoint_KITTI_original_80epoch/checkpoint_epoch_80.pth"
PY="$ROOT/venv_pointrcnn/bin/python"
POINT_PY="/home/ra87racy/miniconda3/envs/pointrcnn_old/bin/python"
CENTER_PY="/home/ra87racy/miniconda3/envs/openpcdet_centerpoint/bin/python"

A_VARIANT="pu_gcn_linea_surface_pr1_c32"
B_VARIANT="pu_gcn_lineb_c2048_r4"
A_PRED="$GEN/val3769_linea/$A_VARIANT/line_a_original_x4_up/final_bin"
B_PRED="$GEN/val3769_lineb/$B_VARIANT/line_b_downsampled_x4_up/final_bin"
A_PROTOCOL="$GEN/val3769_linea/$A_VARIANT/protocol.json"
B_PROTOCOL="$GEN/val3769_lineb/$B_VARIANT/protocol.json"
A_OBS="$RESULT/inputs/line_a_pugcn_observed_first"
B_OBS="$RESULT/inputs/line_b_pugcn_observed_first"

generate_a() {
  "$PY" "$ROOT/scripts/run_patch_causal_upsampling.py" \
    --experiment-root "$GEN" --run-kind val3769_linea --frames-file "$VAL" \
    --method pu_gcn --variant-name "$A_VARIANT" \
    --patch-selection fps_ball_cover_knn_v3 --patch-num-ratio 1 \
    --ball-radius-m 2 --min-ball-points 2048 --cover-min-points 32 --cover-radius-m 6 \
    --normalization-mode legacy_none --lines line_a_original_x4_up \
    --line-a-input "$ORIGINAL" --extract-workers "${EXTRACT_WORKERS:-6}" --frame-batch-size 64 \
    --drop-merged-after-final --cuda-visible-devices 0 --gpu-memory-fraction 0.85 \
    --min-free-gib 60
}

generate_b() {
  "$PY" "$ROOT/scripts/run_patch_causal_upsampling.py" \
    --experiment-root "$GEN" --run-kind val3769_lineb --frames-file "$VAL" \
    --method pu_gcn --variant-name "$B_VARIANT" \
    --patch-selection fps_ball_cover_knn_v3 --patch-num-ratio 1 \
    --ball-radius-m 4 --min-ball-points 2048 --cover-min-points 32 --cover-radius-m 6 \
    --normalization-mode legacy_none --lines line_b_downsampled_x4_up \
    --line-b-input "$SPARSE" --extract-workers "${EXTRACT_WORKERS:-6}" --frame-batch-size 64 \
    --drop-merged-after-final --cuda-visible-devices 0 --gpu-memory-fraction 0.85 \
    --min-free-gib 35
}

audit_and_prepare() {
  mkdir -p "$RESULT/reports" "$RESULT/inputs"
  "$PY" "$ROOT/scripts/verify_patch_causal_strict_x4.py" \
    --frames-file "$VAL" --observed-dir "$ORIGINAL" --predicted-dir "$A_PRED" \
    --output-json "$RESULT/reports/line_a_strict4x_audit.json" \
    --output-csv "$RESULT/reports/line_a_strict4x_frames.csv"
  "$PY" "$ROOT/scripts/verify_patch_causal_strict_x4.py" \
    --frames-file "$VAL" --observed-dir "$SPARSE" --predicted-dir "$B_PRED" \
    --output-json "$RESULT/reports/line_b_strict4x_audit.json" \
    --output-csv "$RESULT/reports/line_b_strict4x_frames.csv"
  "$CENTER_PY" "$ROOT/scripts/prepare_centerpoint_observed_first_train.py" \
    --split-file "$VAL" --observed-dir "$ORIGINAL" --predicted-dir "$A_PRED" \
    --output-dir "$A_OBS" --reference-token original_x4_pu_gcn \
    --manifest "$RESULT/inputs/line_a_pugcn_observed_first_manifest.csv"
  "$CENTER_PY" "$ROOT/scripts/prepare_centerpoint_observed_first_train.py" \
    --split-file "$VAL" --observed-dir "$SPARSE" --predicted-dir "$B_PRED" \
    --output-dir "$B_OBS" --reference-token downsampled_x4_pu_gcn \
    --manifest "$RESULT/inputs/line_b_pugcn_observed_first_manifest.csv"
}

point_eval() {
  local name="$1" lidar="$2" checkpoint="$3"
  "$POINT_PY" "$ROOT/scripts/run_pointrcnn_checkpoint_split_eval.py" \
    --lidar-dir "$lidar" --checkpoint "$checkpoint" \
    --split-file "$VAL" --split-name val \
    --output-dir "$RESULT/evaluations/pointrcnn/$name" \
    --batch-size 1 --workers 0 --seed 20260908 --resume
}

pointrcnn_matrix() {
  point_eval line_a_baseline_official "$ORIGINAL" "$OFFICIAL_POINT"
  point_eval line_a_pugcn_direct_official "$A_PRED" "$OFFICIAL_POINT"
  point_eval line_a_pugcn_direct_adapted "$A_PRED" "$OLD_EXP/pointrcnn/line_a_pugcn/combined_rpn_rcnn_adapted.pth"
  point_eval line_a_pugcn_observed_first_official "$A_OBS" "$OFFICIAL_POINT"
  point_eval line_a_pugcn_observed_first_adapted "$A_OBS" "$OLD_EXP/pointrcnn/line_a_pugcn_observed_first/combined_rpn_rcnn_adapted.pth"
  point_eval line_b_baseline_official "$SPARSE" "$OFFICIAL_POINT"
  point_eval line_b_baseline_adapted "$SPARSE" "$OLD_EXP/pointrcnn/line_b_baseline/combined_rpn_rcnn_adapted.pth"
  point_eval line_b_pugcn_direct_official "$B_PRED" "$OFFICIAL_POINT"
  point_eval line_b_pugcn_direct_adapted "$B_PRED" "$OLD_EXP/pointrcnn/line_b_pugcn/combined_rpn_rcnn_adapted.pth"
  point_eval line_b_pugcn_observed_first_official "$B_OBS" "$OFFICIAL_POINT"
  point_eval line_b_pugcn_observed_first_adapted "$B_OBS" "$OLD_EXP/pointrcnn/line_b_pugcn_observed_first/combined_rpn_rcnn_adapted.pth"
}

center_eval() {
  local name="$1" line="$2" lidar="$3" checkpoint="$4" protocol="$5"
  "$CENTER_PY" "$ROOT/scripts/run_patch_causal_centerpoint_eval.py" \
    --name "fullval3769_20260908_$name" --line "$line" --source-mode direct --predicted-dir "$lidar" \
    --protocol-json "$protocol" --result-root "$RESULT/evaluations/centerpoint" \
    --checkpoint "$checkpoint" --batch-size 1 --workers 0 --resume
}

centerpoint_matrix() {
  center_eval line_a_baseline_official line_a "$ORIGINAL" "$OFFICIAL_CENTER" "$A_PROTOCOL"
  center_eval line_a_pugcn_direct_official line_a "$A_PRED" "$OFFICIAL_CENTER" "$A_PROTOCOL"
  center_eval line_a_pugcn_observed_first_official line_a "$A_OBS" "$OFFICIAL_CENTER" "$A_PROTOCOL"
  center_eval line_a_pugcn_observed_first_adapted line_a "$A_OBS" "$OLD_EXP/centerpoint/line_a_pugcn/openpcdet_output/ckpt/checkpoint_epoch_3.pth" "$A_PROTOCOL"
  center_eval line_b_baseline_official line_b "$SPARSE" "$OFFICIAL_CENTER" "$B_PROTOCOL"
  center_eval line_b_baseline_adapted line_b "$SPARSE" "$OLD_EXP/centerpoint/line_b_baseline/openpcdet_output/ckpt/checkpoint_epoch_3.pth" "$B_PROTOCOL"
  center_eval line_b_pugcn_direct_official line_b "$B_PRED" "$OFFICIAL_CENTER" "$B_PROTOCOL"
  center_eval line_b_pugcn_observed_first_official line_b "$B_OBS" "$OFFICIAL_CENTER" "$B_PROTOCOL"
  center_eval line_b_pugcn_observed_first_adapted line_b "$B_OBS" "$OLD_EXP/centerpoint/line_b_pugcn/openpcdet_output/ckpt/checkpoint_epoch_3.pth" "$B_PROTOCOL"
}

aggregate() {
  "$PY" "$ROOT/scripts/summarize_pugcn_detector_adaptation_full_val_20260908.py"
}

case "${1:-all}" in
  generate_a) generate_a ;;
  generate_b) generate_b ;;
  prepare) audit_and_prepare ;;
  pointrcnn) pointrcnn_matrix ;;
  centerpoint) centerpoint_matrix ;;
  aggregate) aggregate ;;
  all)
    generate_a
    generate_b
    audit_and_prepare
    pointrcnn_matrix
    centerpoint_matrix
    aggregate
    ;;
  *)
    echo "Usage: $0 {generate_a|generate_b|prepare|pointrcnn|centerpoint|aggregate|all}" >&2
    exit 2
    ;;
esac

echo "PU_GCN_FULL_VAL_STAGE_PASS stage=${1:-all} result=$RESULT"
