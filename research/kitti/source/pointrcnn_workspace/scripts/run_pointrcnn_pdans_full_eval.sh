#!/usr/bin/env bash
set -euo pipefail

# Run full-validation PointRCNN evaluation (default settings) + offline KITTI AP
# computation for a PDANS-upsampled KITTI line (Original-Up or Downsampled50-Up).
#
# Usage:
#   scripts/run_pointrcnn_pdans_full_eval.sh <line_name> <bin_dir> <result_root> ["<method label>"]
#
# Example (Line B):
#   scripts/run_pointrcnn_pdans_full_eval.sh \
#     pdans_downsampled50 \
#     results/pdans_main_kitti_pipeline/pdans_downsampled50_up_bin \
#     results/pdans_main_kitti_pipeline/pointrcnn_pdans_downsampled50_eval \
#     "PDANS Downsampled50-Up"
#
# Mirrors the smoke-test protocol validated in
# results/pdans_main_kitti_pipeline/smoke_test/pdans_smoke_test_report.md:
# default RPN.NUM_POINTS=16384, default TEST.RPN_DISTANCE_BASED_PROPOSE=True,
# only override is RPN.LOC_XZ_FINE False, TEST.SPLIT val.
#
# Temporarily symlinks data/KITTI/object/training/velodyne -> <bin_dir> and restores
# the previous state (absent, symlink, or directory) on exit, following the
# restore_mode pattern from results/tulip_original_full_validation_pointrcnn.

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <line_name> <bin_dir> <result_root> [\"<method label>\"]" >&2
  exit 1
fi

LINE_NAME="$1"
BIN_DIR_REL="$2"
RESULT_ROOT_REL="$3"
METHOD_LABEL="${4:-PDANS ${LINE_NAME}}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/ra87racy/projects/baseline_detectors/PointRCNN}"
POINT_RCNN_PY="${POINT_RCNN_PY:-${PROJECT_ROOT}/venv_pointrcnn/bin/python}"
VAL_SPLIT="${VAL_SPLIT:-${PROJECT_ROOT}/data/KITTI/ImageSets/val.txt}"
TRAINING_DIR="${TRAINING_DIR:-${PROJECT_ROOT}/data/KITTI/object/training}"
BIN_DIR="${PROJECT_ROOT}/${BIN_DIR_REL}"
ACTIVE_VELODYNE="${TRAINING_DIR}/velodyne"
RESULT_ROOT="${PROJECT_ROOT}/${RESULT_ROOT_REL}"
TIME_FILE="${RESULT_ROOT}/time.txt"

mkdir -p "${RESULT_ROOT}"

# 1. Verify full .bin coverage for the val split before touching the velodyne symlink.
"${POINT_RCNN_PY}" - <<PY
from pathlib import Path
val = [x.strip() for x in Path("${VAL_SPLIT}").read_text().splitlines() if x.strip()]
bin_dir = Path("${BIN_DIR}")
missing = [frame_id for frame_id in val if not (bin_dir / f"{frame_id}.bin").exists()]
print(f"val_frames={len(val)}")
print(f"${LINE_NAME}_bin_coverage={len(val)-len(missing)}/{len(val)}")
if missing:
    print("missing_first_50=" + ",".join(missing[:50]))
    raise SystemExit("Full .bin coverage for ${LINE_NAME} is required before PointRCNN evaluation.")
PY

# 2. Save current velodyne state and plan how to restore it on exit.
RESTORE_MODE="absent"
RESTORE_TARGET=""
BACKUP_PATH="${RESULT_ROOT}/velodyne_backup_before_eval"

restore_velodyne() {
  set +e
  rm -f "${ACTIVE_VELODYNE}"
  if [[ "${RESTORE_MODE}" == "symlink" ]]; then
    ln -s "${RESTORE_TARGET}" "${ACTIVE_VELODYNE}"
  elif [[ "${RESTORE_MODE}" == "directory" ]]; then
    if [[ -d "${BACKUP_PATH}" ]]; then
      mv "${BACKUP_PATH}" "${ACTIVE_VELODYNE}"
    fi
  fi
}
trap restore_velodyne EXIT

if [[ -L "${ACTIVE_VELODYNE}" ]]; then
  RESTORE_MODE="symlink"
  RESTORE_TARGET="$(readlink "${ACTIVE_VELODYNE}")"
elif [[ -d "${ACTIVE_VELODYNE}" ]]; then
  RESTORE_MODE="directory"
  if [[ -e "${BACKUP_PATH}" ]]; then
    echo "Backup path already exists, refusing to move active velodyne: ${BACKUP_PATH}" >&2
    exit 1
  fi
  mv "${ACTIVE_VELODYNE}" "${BACKUP_PATH}"
elif [[ -e "${ACTIVE_VELODYNE}" ]]; then
  echo "Active velodyne exists but is neither symlink nor directory: ${ACTIVE_VELODYNE}" >&2
  exit 1
fi

ln -s "${BIN_DIR}" "${ACTIVE_VELODYNE}"
readlink -f "${ACTIVE_VELODYNE}" > "${RESULT_ROOT}/velodyne_during.txt"
{
  echo "restore_mode=${RESTORE_MODE}"
  echo "restore_target=${RESTORE_TARGET}"
} > "${RESULT_ROOT}/velodyne_restore_plan.txt"

# 3. Run PointRCNN full validation (default settings, no TULIP-style overrides).
cd "${PROJECT_ROOT}/tools"
NUMBA_ENABLE_CUDASIM=1 \
PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/tools:${PROJECT_ROOT}/lib/net:${PYTHONPATH:-}" \
/usr/bin/time -f 'wall_clock_sec=%e\nmax_rss_kb=%M' -o "${TIME_FILE}" \
"${POINT_RCNN_PY}" eval_rcnn.py \
  --cfg_file cfgs/default.yaml \
  --ckpt PointRCNN.pth \
  --batch_size 1 \
  --workers 0 \
  --eval_mode rcnn \
  --save_result \
  --test \
  --output_dir "${RESULT_ROOT}/inference" \
  --set RPN.LOC_XZ_FINE False TEST.SPLIT val \
  2>&1 | tee "${RESULT_ROOT}/full_log_stdout_stderr.log"

cd "${PROJECT_ROOT}"
DETECTION_DIR="${RESULT_ROOT}/inference/eval/epoch_no_number/val/test_mode/final_result/data"

# 4. Offline KITTI AP computation (CPU rotate-iou, works around GPU/numba requirement).
"${POINT_RCNN_PY}" scripts/run_pointrcnn_full_validation_pugcn_cap100k_ap_eval.py \
  --result-root "${RESULT_ROOT}" \
  --full-split "${VAL_SPLIT}" \
  --label-dir "${TRAINING_DIR}/label_2" \
  --detection-dir "${DETECTION_DIR}" \
  --method "${METHOD_LABEL}" \
  --detector-variant "PointRCNN default settings (RPN.NUM_POINTS=16384, TEST.RPN_DISTANCE_BASED_PROPOSE=True), RPN.LOC_XZ_FINE False" \
  --summary-prefix "${LINE_NAME}_ap_eval" \
  2>&1 | tee "${RESULT_ROOT}/ap_eval_stdout_stderr.log"

# 5. Write parsed_ap_results.csv in the same format used by other methods' reports.
"${POINT_RCNN_PY}" - <<PY
import csv
from pathlib import Path
root = Path("${RESULT_ROOT}")
src = root / "${LINE_NAME}_ap_eval_summary.csv"
dst = root / "parsed_ap_results.csv"
rows = list(csv.DictReader(src.open()))
if not rows:
    raise SystemExit(f"No rows in {src}")
row = rows[0]
if row.get("status") != "success":
    raise SystemExit(f"AP summary status is not success: {row.get('status')}")
with dst.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "easy", "moderate", "hard", "note"])
    writer.writerow(["bbox_AP", row["bbox_AP_easy"], row["bbox_AP_moderate"], row["bbox_AP_hard"], "offline KITTI AP recovery, Car AP@0.70/0.70/0.70"])
    writer.writerow(["bev_AP", row["bev_AP_easy"], row["bev_AP_moderate"], row["bev_AP_hard"], "offline KITTI AP recovery, Car AP@0.70/0.70/0.70"])
    writer.writerow(["3d_AP", row["3d_AP_easy"], row["3d_AP_moderate"], row["3d_AP_hard"], "offline KITTI AP recovery, Car AP@0.70/0.70/0.70"])
print(f"wrote {dst}")
PY

echo "PointRCNN full validation + AP evaluation for ${LINE_NAME} completed: ${RESULT_ROOT}"
