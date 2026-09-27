#!/usr/bin/env bash
# Line B patch dilemma: locality and unique support cannot both hold at N/4 density.
#
#              | repeat fill        | 2048 unique
#   r = 2 m    | B1                 | impossible (densest 2 m ball holds 1383 pts)
#   r = 4 m    | B3                 | B2
#
#   B1 vs B3 isolates radius; B3 vs B2 isolates uniqueness; B1 vs B2 is the dilemma.
# Cover parameters are held fixed across arms so only the two factors vary.
set -u

cd /home/ra87racy/projects/baseline_detectors/PointRCNN || exit 1
ROOT=results/kitti_patch_punet_causal_ablation_v1_20260731
FRAMES=$ROOT/splits/geometry20.txt
RUN_KIND=lineb20_dilemma_v1
PY=/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python

BASE=(
  --run-kind "$RUN_KIND"
  --frames-file "$FRAMES"
  --patch-selection fps_ball_cover_knn_v3
  --patch-num-ratio 1
  --cover-min-points 32
  --cover-radius-m 6
  --lines line_b_downsampled_x4_up
)

# arm  radius  min-unique
ARMS=("B1 2 256" "B3 4 256" "B2 4 2048")
METHODS=("pu_gcn pu_gcn" "pu_edgeformer pu_edgeformer" "pu_net pu_net_fixed")

for arm_spec in "${ARMS[@]}"; do
  set -- $arm_spec; arm=$1; radius=$2; minpts=$3
  for method_spec in "${METHODS[@]}"; do
    set -- $method_spec; method=$1; tag=$2
    extra=()
    [ "$method" = "pu_net" ] && extra=(--normalization-mode unit_sphere_v1)
    label="${arm}_${tag}"
    echo "DILEMMA_START $label r=${radius} min_unique=${minpts} $(date -u +%H:%M:%S)"
    if "$PY" scripts/run_patch_causal_upsampling.py "${BASE[@]}" \
        --ball-radius-m "$radius" --min-ball-points "$minpts" \
        --method "$method" --variant-name "${arm}_${tag}_r${radius}_u${minpts}" \
        "${extra[@]}"; then
      echo "DILEMMA_DONE $label"
    else
      echo "DILEMMA_FAIL $label rc=$?"
    fi
  done
done

echo "DILEMMA_ALL_DONE"
