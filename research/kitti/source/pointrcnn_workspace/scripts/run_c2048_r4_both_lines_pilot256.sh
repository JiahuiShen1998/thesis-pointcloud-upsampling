#!/usr/bin/env bash
# pilot256 generation for the unique-2048 / zero-repeat patch configuration at r=4 m.
#
# Line B is queued first: it is the line with real headroom (downsampled baseline
# 65.75 vs original 82.26) and the line whose every previous result used a patch
# configuration with ~65% duplicate rows.  Line A follows only so that both lines
# share one radius; Line A already has an equivalent run at r=2 m (surface-c32),
# which is strictly more local because its density supports the tighter ball.
#
# Retries: PDANS launches one CUDA subprocess per frame and segfaults on teardown
# roughly every ~13 frames, after its output is already written.  The runner keeps
# a frame-level merged_raw cache, so a restart only redoes unfinished frames and
# retrying is nearly free.  Batch methods launch once and rarely need more than
# one attempt; PU-EdgeFormer fails deterministically (its conda env segfaults on
# `import tensorflow`) and is capped low so it cannot burn the queue.
set -u

cd /home/ra87racy/projects/baseline_detectors/PointRCNN || exit 1
ROOT=results/kitti_patch_punet_causal_ablation_v1_20260731
FRAMES=$ROOT/splits/pilot256.txt
RUN_KIND=c2048_r4_pilot256_v1
PY=/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python

BASE=(
  --run-kind "$RUN_KIND"
  --frames-file "$FRAMES"
  --patch-selection fps_ball_cover_knn_v3
  --patch-num-ratio 1
  --ball-radius-m 4
  --min-ball-points 2048
  --cover-min-points 32
  --cover-radius-m 6
)

# method  tag  max_attempts
METHODS=(
  "pu_gcn pu_gcn 3"
  "pu_net pu_net_fixed 3"
  "pu_edgeformer pu_edgeformer 2"
  "pdans pdans 60"
)

for line in line_b_downsampled_x4_up line_a_original_x4_up; do
  short=${line:5:1}
  for spec in "${METHODS[@]}"; do
    set -- $spec; method=$1; tag=$2; max=$3
    extra=()
    [ "$method" = "pu_net" ] && extra=(--normalization-mode unit_sphere_v1)
    label="line${short}_${tag}"
    echo "C2048_START $label max_attempts=$max $(date -u +%H:%M:%S)"
    ok=0
    for attempt in $(seq 1 "$max"); do
      if "$PY" scripts/run_patch_causal_upsampling.py "${BASE[@]}" \
          --lines "$line" --method "$method" \
          --variant-name "${tag}_c2048_r4" "${extra[@]}"; then
        echo "C2048_DONE $label attempts=$attempt $(date -u +%H:%M:%S)"
        ok=1
        break
      fi
      echo "C2048_RETRY $label attempt=$attempt/$max $(date -u +%H:%M:%S)"
      sleep 5
    done
    [ "$ok" -eq 1 ] || echo "C2048_FAIL $label exhausted=$max $(date -u +%H:%M:%S)"
  done
  echo "C2048_LINE_DONE line${short} $(date -u +%H:%M:%S)"
done

echo "C2048_ALL_DONE $(date -u +%H:%M:%S)"
