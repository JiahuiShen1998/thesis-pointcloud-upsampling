#!/usr/bin/env bash
# Line B evaluation under the c2048/r4 patch configuration, on the split-region track.
#
# This is the first time Line B is measured on a patch configuration that is not
# ~65% duplicate rows: every patch here has 2048 unique source points, zero repeat
# fill, and 99.96% source coverage.  Parameters mirror the Line A surface-c32
# workspace (pointrcnn_split_region_surface256_v1_20260804) so the two lines are
# read off the same protocol; only the inputs differ.
#
# Reference points for reading the result:
#   downsampled baseline   65.75   (full-val E2 number, context only)
#   original point cloud   82.26   (Line B's oracle: the removed 3M real points)
# The paired split-region baseline produced here is the denominator that matters;
# split-region numbers must never be subtracted from native full-frame numbers.
set -u

cd /home/ra87racy/projects/baseline_detectors/PointRCNN || exit 1
PY=/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python
GEN=results/kitti_patch_punet_causal_ablation_v1_20260731/c2048_r4_pilot256_v1
WS=results/pointrcnn_split_region_lineb_c2048_r4_pilot256_v1
SPLIT=data/KITTI/ImageSets/patch_causal_pilot256.txt
DS=results/kitti_unified_x4_current_methods_no_detector/downsampled_x4/velodyne_downsampled_x4_val

SOURCES=(--source "original_baseline=$DS")
for m in pu_gcn pu_net_fixed pu_edgeformer pdans; do
  d=$GEN/${m}_c2048_r4/line_b_downsampled_x4_up/final_bin
  n=$(ls "$d" 2>/dev/null | wc -l)
  if [ "$n" -eq 256 ]; then
    SOURCES+=(--source "c2048_${m}=$d")
    echo "EVAL_SOURCE_OK $m ($n frames)"
  else
    echo "EVAL_SOURCE_SKIP $m (only $n/256 frames)"
  fi
done

echo "EVAL_PREP_START $(date -u +%H:%M:%S)"
if ! $PY scripts/prepare_pointrcnn_split_region_inputs.py \
      --workspace "$WS" --split-file "$SPLIT" "${SOURCES[@]}"; then
  echo "EVAL_PREP_FAIL"; exit 1
fi
echo "EVAL_PREP_DONE $(date -u +%H:%M:%S)"

# Detection: one source at a time, each retried because launching many fresh CUDA
# subprocesses segfaults on teardown roughly once every dozen slots.
for src in $($PY -c "
import json;print(' '.join(json.load(open('$WS/protocol.json'))['sources']))"); do
  echo "EVAL_DET_START $src $(date -u +%H:%M:%S)"
  ok=0
  for a in 1 2 3 4 5 6; do
    if $PY scripts/run_pointrcnn_split_region_eval.py --workspace "$WS" --source "$src"; then
      echo "EVAL_DET_DONE $src attempts=$a $(date -u +%H:%M:%S)"; ok=1; break
    fi
    echo "EVAL_DET_RETRY $src $a"; sleep 5
  done
  [ "$ok" -eq 1 ] || echo "EVAL_DET_FAIL $src"
done

echo "EVAL_ALL_DONE $(date -u +%H:%M:%S)"
