# PU-EdgeFormer Smoke Leakage Audit

- Generated: `2026-07-16 18:29:10 UTC`

## Line B
- overlapping compared: 10
- content_equal: 10
- hash_equal: 10
- full mtime after smoke: 10
- copy-suspect (hash equal & full mtime not after smoke): 0
- status: **PASS**

## Line A
- smoke_base: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/datasets/modelnet40_original_x4_up/pu_edgeformer_labparams_smoke_20260716/outputs/lineA_A1_direct_1024_to_4096`
- overlapping compared: 10
- content_equal: 10
- hash_equal: 10
- full mtime after smoke: 10
- copy-suspect: 0
- status: **PASS**

## Interpretation
- Content equality with newer full mtime is consistent with deterministic re-inference, not smoke file copy.
- FAIL only if overlapping files look like unchanged copies (same hash, not newer than smoke).

## Overall: **PASS**
