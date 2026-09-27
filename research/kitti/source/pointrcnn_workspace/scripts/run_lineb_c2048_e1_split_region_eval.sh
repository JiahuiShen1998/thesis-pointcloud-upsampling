#!/usr/bin/env bash
# Line B / c2048-r4 re-evaluated with the exact-4N protocol actually applied:
# each input is `M observed + 3M generated`, every observed point preserved.
#
# The previous pass fed `final_bin` straight into the split-region track, and
# final_bin holds 4M points drawn entirely from the method's merged_raw pool —
# zero observed points survive.  So that table compared a baseline holding all M
# real measurements against methods holding none.  This run fixes only that.
#
# Pre-registered reading, fixed before any AP is seen: the paired baseline is the
# same original_baseline source as before; the question is whether any method now
# reaches or exceeds it.  Single evaluation, no configuration sweep — picking a
# winner out of several variants by AP would be selecting inputs on test labels.
set -u

cd /home/ra87racy/projects/baseline_detectors/PointRCNN || exit 1
PY=/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python
E1=results/kitti_patch_punet_causal_ablation_v1_20260731/c2048_r4_pilot256_v1/e1_observed_preserved
WS=results/pointrcnn_split_region_lineb_c2048_e1_pilot256_v1
SPLIT=data/KITTI/ImageSets/patch_causal_pilot256.txt
DS=results/kitti_unified_x4_current_methods_no_detector/downsampled_x4/velodyne_downsampled_x4_val

SOURCES=(--source "original_baseline=$DS")
for m in pu_gcn pu_net_fixed pu_edgeformer pdans; do
  n=$(ls "$E1/$m" 2>/dev/null | wc -l)
  [ "$n" -eq 256 ] && SOURCES+=(--source "e1_${m}=$E1/$m") || echo "EVAL_SOURCE_SKIP $m ($n/256)"
done

echo "EVAL_PREP_START $(date -u +%H:%M:%S)"
$PY scripts/prepare_pointrcnn_split_region_inputs.py \
    --workspace "$WS" --split-file "$SPLIT" "${SOURCES[@]}" || { echo "EVAL_PREP_FAIL"; exit 1; }
echo "EVAL_PREP_DONE $(date -u +%H:%M:%S)"

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
