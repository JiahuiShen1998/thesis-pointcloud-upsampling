#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/ra87racy/miniconda3/envs/pointrcnn_old/bin/python
REPORT_PY=/home/ra87racy/miniconda3/envs/pointrcnn_old/bin/python
TRAIN_SPLIT="$ROOT/data/KITTI/ImageSets/train.txt"
VAL_SPLIT="$ROOT/data/KITTI/ImageSets/val.txt"
OFFICIAL="$ROOT/tools/PointRCNN.pth"
TRAIN_INPUT_ROOT="$ROOT/results/pugcn_full_retrain_20260824/centerpoint_inputs"
VAL_INPUT_ROOT="$ROOT/results/pugcn_detector_adaptation_full_val_20260908/inputs"
RUN_TAG="${RUN_TAG:-pugcn_detector_convergence_20260914}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/$RUN_TAG}"
POINT_ROOT="$RUN_ROOT/pointrcnn"
EVAL_ROOT="$RUN_ROOT/evaluations/pointrcnn"
REPORT_ROOT="$RUN_ROOT/reports"
SCRATCH_ROOT="${SCRATCH_ROOT:-/tmp/$RUN_TAG}"
EPOCHS="${EPOCHS:-12}"
MIN_DELTA="${MIN_DELTA:-0.2}"
PATIENCE="${PATIENCE:-3}"

mkdir -p "$POINT_ROOT" "$EVAL_ROOT" "$REPORT_ROOT" "$SCRATCH_ROOT"

latest_checkpoint() {
  local ckpt_dir=$1
  if [[ ! -d "$ckpt_dir" ]]; then
    return 0
  fi
  find "$ckpt_dir" -maxdepth 1 -type f -name 'checkpoint_epoch_*.pth' -printf '%f\t%p\n' \
    | sort -V \
    | tail -n 1 \
    | cut -f 2-
}

safe_remove_scratch() {
  local target=$1
  case "$target" in
    "$SCRATCH_ROOT"/*) rm -rf -- "$target" ;;
    *) echo "REFUSING_UNSAFE_SCRATCH_REMOVAL target=$target" >&2; exit 90 ;;
  esac
}

train_stage() {
  local mode=$1
  local lidar_dir=$2
  local output_dir=$3
  local init_ckpt=$4
  local roi_dir=${5:-}
  local feature_dir=${6:-}
  local target="$output_dir/ckpt/checkpoint_epoch_${EPOCHS}.pth"
  local log="$output_dir/runner_stdout.log"
  if [[ -f "$target" ]]; then
    echo "TRAIN_SKIP_COMPLETE mode=$mode output=$output_dir"
    return
  fi
  mkdir -p "$output_dir"
  local -a command=(
    "$PY" "$ROOT/scripts/run_pointrcnn_full_train_stage.py"
    --mode "$mode"
    --lidar-dir "$lidar_dir"
    --split-file "$TRAIN_SPLIT"
    --output-dir "$output_dir"
    --init-ckpt "$init_ckpt"
    --epochs "$EPOCHS"
    --ckpt-save-interval 1
    --workers 0
    --batch-size 1
  )
  local resume_ckpt
  resume_ckpt="$(latest_checkpoint "$output_dir/ckpt")"
  if [[ -n "$resume_ckpt" ]]; then
    command+=(--resume-ckpt "$resume_ckpt")
    echo "TRAIN_RESUME mode=$mode checkpoint=$resume_ckpt"
  else
    echo "TRAIN_START_CLEAN mode=$mode init=$init_ckpt"
  fi
  if [[ "$mode" == "rcnn_offline" ]]; then
    command+=(--rcnn-training-roi-dir "$roi_dir" --rcnn-training-feature-dir "$feature_dir")
  fi
  "${command[@]}" >> "$log" 2>&1
  if [[ ! -f "$target" ]]; then
    echo "TRAIN_MISSING_TARGET mode=$mode target=$target" >&2
    exit 91
  fi
  for epoch in $(seq 1 "$EPOCHS"); do
    [[ -s "$output_dir/ckpt/checkpoint_epoch_${epoch}.pth" ]] || {
      echo "TRAIN_MISSING_EPOCH mode=$mode epoch=$epoch output=$output_dir" >&2
      exit 92
    }
  done
  echo "TRAIN_PASS mode=$mode target=$target"
}

combine_checkpoint() {
  local rpn_ckpt=$1
  local rcnn_ckpt=$2
  local output=$3
  local log=$4
  if [[ -s "$output" ]]; then
    return
  fi
  mkdir -p "$(dirname "$output")"
  local -a command=(
    "$PY" "$ROOT/scripts/combine_pointrcnn_rpn_with_official_rcnn.py"
    --official "$OFFICIAL"
    --rpn "$rpn_ckpt"
  )
  if [[ -n "$rcnn_ckpt" ]]; then
    command+=(--rcnn "$rcnn_ckpt")
  fi
  command+=(--output "$output")
  "${command[@]}" >> "$log" 2>&1
  [[ -s "$output" ]] || { echo "COMBINE_FAILED output=$output" >&2; exit 93; }
}

evaluate_checkpoint() {
  local arm=$1
  local phase=$2
  local epoch=$3
  local lidar_dir=$4
  local checkpoint=$5
  local output_dir="$EVAL_ROOT/${arm}_${phase}_epoch_${epoch}"
  local marker="$output_dir/run_complete.json"
  local log="$output_dir/runner_stdout.log"
  mkdir -p "$output_dir"
  if [[ -f "$marker" ]] && jq -e \
      --arg checkpoint "$checkpoint" \
      '.status == "PASS" and .frame_count == 3769 and .actual_consumed_frame_count == 3769 and .checkpoint == $checkpoint' \
      "$marker" >/dev/null; then
    echo "EVAL_SKIP_PASS arm=$arm phase=$phase epoch=$epoch"
    return
  fi
  echo "EVAL_START arm=$arm phase=$phase epoch=$epoch"
  "$PY" "$ROOT/scripts/run_pointrcnn_checkpoint_split_eval.py" \
    --lidar-dir "$lidar_dir" \
    --checkpoint "$checkpoint" \
    --split-file "$VAL_SPLIT" \
    --split-name val \
    --output-dir "$output_dir" \
    --batch-size 1 \
    --workers 0 \
    --seed 20260908 \
    --resume \
    --cleanup-heavy-artifacts >> "$log" 2>&1
  jq -e \
    --arg checkpoint "$checkpoint" \
    '.status == "PASS" and .frame_count == 3769 and .actual_consumed_frame_count == 3769 and .checkpoint == $checkpoint and .heavy_artifacts_retained == false' \
    "$marker" >/dev/null
  local ap
  ap="$(jq -r '.metrics_percent.Car."3d_ap_r40".moderate' "$marker")"
  echo "EVAL_PASS arm=$arm phase=$phase epoch=$epoch car_3d_moderate=$ap"
}

summarize_stage() {
  local arm=$1
  local phase=$2
  "$REPORT_PY" "$ROOT/scripts/summarize_pugcn_pointrcnn_convergence_20260914.py" stage \
    --run-root "$RUN_ROOT" \
    --arm "$arm" \
    --phase "$phase" \
    --expected-epochs "$EPOCHS" \
    --min-delta "$MIN_DELTA" \
    --patience "$PATIENCE" > "$REPORT_ROOT/pointrcnn_${arm}_${phase}_summary_stdout.json"
  local summary="$REPORT_ROOT/pointrcnn_${arm}_${phase}_convergence.json"
  local status
  status="$(jq -r '.status' "$summary")"
  echo "STAGE_SUMMARY arm=$arm phase=$phase status=$status"
  if [[ "$status" != "CONVERGED" ]]; then
    echo "STAGE_NEEDS_LONGER_SCHEDULE arm=$arm phase=$phase summary=$summary" >&2
    exit 42
  fi
}

export_selected_rpn() {
  local arm=$1
  local lidar_dir=$2
  local rpn_epoch=$3
  local rpn_ckpt=$4
  local output_dir="$SCRATCH_ROOT/$arm/rpn_export_epoch_${rpn_epoch}"
  local marker="$output_dir/export_complete.json"
  local log="$POINT_ROOT/$arm/rpn_export_epoch_${rpn_epoch}.log"
  if [[ -f "$marker" ]] && jq -e \
      --arg ckpt "$rpn_ckpt" --arg ep_label "$rpn_epoch" \
      '.status == "PASS" and .frames == 3712 and .rois == 3712 and .rpn_ckpt == $ckpt and .epoch_label == $ep_label' \
      "$marker" >/dev/null; then
    echo "$output_dir"
    return
  fi
  if [[ -e "$output_dir" ]]; then
    safe_remove_scratch "$output_dir"
  fi
  mkdir -p "$output_dir"
  echo "RPN_EXPORT_START arm=$arm rpn_epoch=$rpn_epoch" >&2
  "$PY" "$ROOT/scripts/run_pointrcnn_rpn_feature_export.py" \
    --lidar-dir "$lidar_dir" \
    --split-file "$TRAIN_SPLIT" \
    --rpn-ckpt "$rpn_ckpt" \
    --output-dir "$output_dir" \
    --epoch-label "$rpn_epoch" \
    --workers 0 >> "$log" 2>&1
  jq -e \
    --arg ckpt "$rpn_ckpt" --arg ep_label "$rpn_epoch" \
    '.status == "PASS" and .frames == 3712 and .rois == 3712 and .rpn_ckpt == $ckpt and .epoch_label == $ep_label' \
    "$marker" >/dev/null
  echo "RPN_EXPORT_PASS arm=$arm rpn_epoch=$rpn_epoch" >&2
  echo "$output_dir"
}

run_arm() {
  local arm=$1
  local train_lidar="$TRAIN_INPUT_ROOT/$arm"
  local val_lidar="$VAL_INPUT_ROOT/$arm"
  local arm_root="$POINT_ROOT/$arm"
  local rpn_dir="$arm_root/rpn"
  local combine_log="$arm_root/combine.log"
  mkdir -p "$arm_root"

  echo "ARM_START arm=$arm"
  train_stage rpn "$train_lidar" "$rpn_dir" "$OFFICIAL"

  for epoch in $(seq 1 "$EPOCHS"); do
    local rpn_ckpt="$rpn_dir/ckpt/checkpoint_epoch_${epoch}.pth"
    local combined="$arm_root/combined_rpn_curve/checkpoint_epoch_${epoch}.pth"
    combine_checkpoint "$rpn_ckpt" "" "$combined" "$combine_log"
    evaluate_checkpoint "$arm" rpn "$epoch" "$val_lidar" "$combined"
  done
  summarize_stage "$arm" rpn

  local rpn_summary="$REPORT_ROOT/pointrcnn_${arm}_rpn_convergence.json"
  local selected_rpn_epoch
  selected_rpn_epoch="$(jq -r '.global_best_epoch' "$rpn_summary")"
  local selected_rpn_ckpt="$rpn_dir/ckpt/checkpoint_epoch_${selected_rpn_epoch}.pth"
  local selected_init="$arm_root/selected_rpn_epoch_${selected_rpn_epoch}_official_rcnn.pth"
  combine_checkpoint "$selected_rpn_ckpt" "" "$selected_init" "$combine_log"

  local export_root
  export_root="$(export_selected_rpn "$arm" "$train_lidar" "$selected_rpn_epoch" "$selected_rpn_ckpt")"
  local export_data="$export_root/eval/epoch_${selected_rpn_epoch}/train"
  local feature_dir="$export_data/features"
  local roi_dir="$export_data/detections/data"
  local rcnn_dir="$arm_root/rcnn_from_rpn_epoch_${selected_rpn_epoch}"
  train_stage rcnn_offline "$train_lidar" "$rcnn_dir" "$selected_init" "$roi_dir" "$feature_dir"

  # The 21 GB export is reproducible from the selected RPN checkpoint and is
  # no longer needed once all offline RCNN checkpoints have been written.
  safe_remove_scratch "$export_root"
  echo "RPN_EXPORT_SCRATCH_CLEANED arm=$arm"

  for epoch in $(seq 1 "$EPOCHS"); do
    local rcnn_ckpt="$rcnn_dir/ckpt/checkpoint_epoch_${epoch}.pth"
    local combined="$arm_root/combined_rcnn_curve_rpn_epoch_${selected_rpn_epoch}/checkpoint_epoch_${epoch}.pth"
    combine_checkpoint "$selected_rpn_ckpt" "$rcnn_ckpt" "$combined" "$combine_log"
    evaluate_checkpoint "$arm" rcnn "$epoch" "$val_lidar" "$combined"
  done
  summarize_stage "$arm" rcnn
  echo "ARM_PASS arm=$arm selected_rpn_epoch=$selected_rpn_epoch"
}

run_arm line_a_pugcn_observed_first
run_arm line_b_pugcn_observed_first

"$REPORT_PY" "$ROOT/scripts/summarize_pugcn_pointrcnn_convergence_20260914.py" aggregate \
  --run-root "$RUN_ROOT" \
  --expected-epochs "$EPOCHS" > "$REPORT_ROOT/pointrcnn_convergence_summary_stdout.json"

echo "PU_GCN_POINTRCNN_CONVERGENCE_PASS result=$RUN_ROOT"
