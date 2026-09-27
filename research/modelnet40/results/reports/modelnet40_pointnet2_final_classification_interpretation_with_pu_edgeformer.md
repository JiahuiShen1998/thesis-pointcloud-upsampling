# PointNet++ Final Classification Interpretation (with PU-EdgeFormer)

- Generated: `2026-07-17 04:10:04 UTC`

## Line A

- Strongest method: **Original baseline 1024** at 91.95%.
- PU-EdgeFormer 4096: 90.54% (-1.41 pp vs Original baseline).
- Line A deltas use Original baseline 1024 only.

## Line B

- Best upsampling method on primary Δ: **Downsampled x4 + PU-Net 1024** (91.27%, +0.42 pp vs Downsampled baseline).
- PU-EdgeFormer 1024: 89.32%; primary Δ vs Downsampled = -1.53 pp; secondary gap vs Original = -2.63 pp.
- Primary ranking must use Downsampled ×4 baseline 256.
- Secondary gap answers recovery toward Original baseline 1024.

## Comparison logic reminder

- Geometry Δ: vs Original baseline 1024.
- Classification primary Δ (Line B): vs Downsampled ×4 baseline 256.
- Classification secondary gap (Line B): vs Original baseline 1024.
- Do not mix these references.

## Safety

- DETECTOR_EVAL_STARTED=NO
- KITTI_AP_EVAL_STARTED=NO
