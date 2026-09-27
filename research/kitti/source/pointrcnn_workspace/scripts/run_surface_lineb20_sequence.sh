#!/usr/bin/env bash
# Generate Line B surface-c32 on the frozen 20-frame geometry split.
#
# Line B is the only line where generated-point correctness is measurable: the
# reference is the original scan, so a candidate is right or wrong by whether it
# lands on a voxel the full scan actually occupies. Line A has no such reference.
#
# Patch settings are identical to the frozen Line A surface-c32 run, except that
# --reuse-patch-base is dropped: Line B has different input points, so patches
# must be extracted from the downsampled cloud rather than reused.
set -u

cd /home/ra87racy/projects/baseline_detectors/PointRCNN || exit 1
ROOT=results/kitti_patch_punet_causal_ablation_v1_20260731
FRAMES=$ROOT/splits/geometry20.txt
RUN_KIND=surface_lineb20_v1

COMMON=(
  --run-kind "$RUN_KIND"
  --frames-file "$FRAMES"
  --patch-selection fps_ball_cover_knn_v3
  --patch-num-ratio 1
  --ball-radius-m 2
  --min-ball-points 2048
  --cover-min-points 32
  --cover-radius-m 6
  --lines line_b_downsampled_x4_up
)
PY=/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python

# Cheap methods first so a PDANS failure cannot block the rest.
run_one() {
  local label="$1"; shift
  echo "LINEB20_START $label $(date -u +%H:%M:%S)"
  if "$PY" scripts/run_patch_causal_upsampling.py "${COMMON[@]}" "$@"; then
    echo "LINEB20_DONE $label $(date -u +%H:%M:%S)"
  else
    echo "LINEB20_FAIL $label rc=$?"
  fi
}

run_one PU-GCN        --method pu_gcn        --variant-name pu_gcn_surface_lineb_c32
run_one PU-EdgeFormer --method pu_edgeformer --variant-name pu_edgeformer_surface_lineb_c32
run_one PU-Net-fixed  --method pu_net        --variant-name pu_net_fixed_surface_lineb_c32 \
                      --normalization-mode unit_sphere_v1
run_one PDANS         --method pdans         --variant-name pdans_surface_lineb_c32

echo "LINEB20_ALL_DONE"
