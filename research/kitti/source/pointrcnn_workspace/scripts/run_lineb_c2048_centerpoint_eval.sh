#!/usr/bin/env bash
# Line B / c2048-r4 on the second frozen detector, CenterPoint.
#
# PointRCNN showed that consensus selection does not help: it raised novel-voxel
# precision by 15-21 points yet moved detection inconsistently (PDANS -4.18,
# PU-EdgeFormer -5.20, PU-GCN +0.94, PU-Net +8.92), with the gain confined to the
# method whose candidate pool was dirtiest.  The reading was that spatial coverage
# matters more to the detector than pointwise accuracy -- consensus halves
# removed-voxel recall (19.8% -> 8.6%) to buy its precision.
#
# CenterPoint voxelises rather than sampling points, so it stresses coverage
# differently.  Running both selections here tests whether that reading is a
# property of the input or an artefact of one detector.  Both selections are
# already fixed; nothing is tuned in response to what PointRCNN reported.
set -u

cd /home/ra87racy/projects/baseline_detectors/PointRCNN || exit 1
PY=/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python
GEN=results/kitti_patch_punet_causal_ablation_v1_20260731/c2048_r4_pilot256_v1
ROOT=results/centerpoint_lineb_c2048_pilot256_v1

run_one() {
  local name=$1; shift
  echo "CP_START $name $(date -u +%H:%M:%S)"
  for a in 1 2 3 4; do
    if $PY scripts/run_patch_causal_centerpoint_eval.py \
        --name "$name" --line line_b --result-root "$ROOT" "$@"; then
      echo "CP_DONE $name attempts=$a $(date -u +%H:%M:%S)"; return 0
    fi
    echo "CP_RETRY $name $a"; sleep 5
  done
  echo "CP_FAIL $name"; return 1
}

run_one baseline --source-mode baseline

for m in pdans pu_gcn pu_edgeformer pu_net_fixed; do
  run_one "rand_${m}" --source-mode direct --predicted-dir "$GEN/e1_observed_preserved/$m"
  run_one "cons_${m}" --source-mode direct --predicted-dir "$GEN/e1_consensus_selected/$m"
done

echo "CP_ALL_DONE $(date -u +%H:%M:%S)"
