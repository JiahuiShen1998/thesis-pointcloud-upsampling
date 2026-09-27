# PU-EdgeFormer ModelNet40 Pipeline Integration Audit

- Generated: `2026-07-16 18:36:34 UTC`
- overall: **PASS**
- method: symlink train/test only; metadata class maps copied; no data rewrite

## lineB
- target: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineB_downsampled_x4_up/strict_N/pu_edgeformer`
- source: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_downsampled_x4_up/pu_edgeformer_labparams_20260716`
- train/test/total: 9843 / 2468 / 12311
- expected points: 1024×3 float32
- train/test symlink ok: True / True
- points to authentic source: True / True
- bad_shape / bad_dtype / nan_inf: 0 / 0 / 0
- metadata: `class_to_idx.json|idx_to_class.json` (ok=True, no manifests=True)
- status: **PASS**

## lineA
- target: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/lineA_original_up/strict_4N/pu_edgeformer`
- source: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original_x4_up/pu_edgeformer_labparams_20260716`
- train/test/total: 9843 / 2468 / 12311
- expected points: 4096×3 float32
- train/test symlink ok: True / True
- points to authentic source: True / True
- bad_shape / bad_dtype / nan_inf: 0 / 0 / 0
- metadata: `class_to_idx.json|idx_to_class.json` (ok=True, no manifests=True)
- status: **PASS**

## Safety
- POINTNET_CLASSIFIER_STARTED=NO
- DETECTOR_EVAL_STARTED=NO
- KITTI_AP_EVAL_STARTED=NO
- GEOMETRY_METRICS_STARTED=YES
