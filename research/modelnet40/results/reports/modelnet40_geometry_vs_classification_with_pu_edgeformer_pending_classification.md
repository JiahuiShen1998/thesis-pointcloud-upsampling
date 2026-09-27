# Geometry vs Classification (PU-EdgeFormer classification pending)

- Generated: `2026-07-16 19:20:00 UTC`
- Comparison logic source: `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`

## Comparison rules (do not mix)

1. **Geometry delta (Line B):** vs **Original baseline 1024**
   - `delta_metric_vs_original = metric(LineB method) − metric(Original baseline)`
2. **Classification primary delta (Line B):** vs **Downsampled ×4 baseline 256**
   - `delta_accuracy_vs_downsampled_baseline = Acc(LineB method) − Acc(Downsampled baseline)`
3. **Classification secondary gap (Line B):** vs **Original baseline 1024**
   - `gap_accuracy_vs_original_baseline = Acc(LineB method) − Acc(Original baseline)`

Absolute geometry metrics use **dense mesh surface / mesh / GT reference** (not called “baseline”).

## Line B geometry deltas vs Original baseline 1024

| Method | ΔCD vs Original | ΔHD vs Original | ΔNUC vs Original | Δ exact P2F vs Original | Geometry status |
|---|---:|---:|---:|---:|---|
| EAR | 0.031809 | 0.089073 | -0.104148 | -0.001315 | available |
| PDANS | 0.013590 | 0.086207 | 0.027981 | -0.002883 | available |
| PU-Net | 0.007567 | 0.017954 | -0.003731 | -0.002403 | available |
| PU-GCN | 0.005279 | 0.045491 | 0.424526 | 0.001309 | available |
| PU-EdgeFormer | 0.013447 | 0.073747 | 0.035923 | 0.001715 | available |

## Line B classification columns (schema reserved; PU-EdgeFormer pending)

| Method | Acc | delta_accuracy_vs_downsampled_baseline (primary) | gap_accuracy_vs_original_baseline (secondary) | Classification status |
|---|---:|---:|---:|---|
| EAR | available | available | available | existing run |
| PDANS | available | available | available | existing run |
| PU-Net | available | available | available | existing run |
| PU-GCN | available | available | available | existing run |
| PU-EdgeFormer | — | pending | pending | **pending** (PointNet++ not started) |

Notes:
- Primary classification ranking for Line B must use `delta_accuracy_vs_downsampled_baseline`.
- `gap_accuracy_vs_original_baseline` is an additional recovery-gap column only.
- Do not invent PU-EdgeFormer accuracy.
- POINTNET_CLASSIFIER_STARTED=NO
- DETECTOR_EVAL_STARTED=NO
- KITTI_AP_EVAL_STARTED=NO

## Update 2026-07-17

- PU-EdgeFormer PointNet++ full training completed (jobs 1749441 / 1749442).
- See `reports/modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.md`.
- POINTNET_FULL_TRAINING_STARTED=YES; DETECTOR_EVAL_STARTED=NO; KITTI_AP_EVAL_STARTED=NO.
