# PU-EdgeFormer ModelNet40 Lab-Params FULL Audit

- Generated: `2026-07-16 18:22:20 UTC`
- overall: **PASS**
- scope: upsampled datasets only (no geometry metrics, no PointNet++, no detector, no KITTI AP)

## Line B (256 → 1024, direct)

- root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4_up/pu_edgeformer_labparams_20260716`
- status: **PASS**
- train/test/total: 9843 / 2468 / 12311 (expected 9843 / 2468 / 12311)
- counts_ok: True
- exact 1024×3: True
- dtype float32: True
- any NaN/Inf: False
- raw shortage rows: 0
- crop actions: {'exact': 12311}
- padding/jitter/interpolation/fake points: NO
- manifest pass/fail: 12311 / 0

## Line A (1024 → 4096, A1_direct)

- root: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original_x4_up/pu_edgeformer_labparams_20260716`
- status: **PASS**
- train/test/total: 9843 / 2468 / 12311 (expected 9843 / 2468 / 12311)
- counts_ok: True
- exact 4096×3: True
- dtype float32: True
- any NaN/Inf: False
- raw shortage rows: 0
- crop actions: {'exact': 12311}
- padding/jitter/interpolation/fake points: NO
- manifest pass/fail: 12311 / 0

## Safety flags

- DETECTOR_EVAL_STARTED=NO
- POINTNET_CLASSIFIER_STARTED=NO
- KITTI_AP_EVAL_STARTED=NO
- GEOMETRY_METRICS_STARTED=NO
