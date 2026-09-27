# Mesh-ref PointNet++ baseline results

- Generated: 2026-08-03 22:49:16 UTC
- Model: `pointnet2_cls_ssg`, 200 epochs, Adam lr=0.001, seed=42, from scratch

| Method | Pts | Best Overall | Final Overall | Best epoch | Compare to | Compare Best | Δ Best | Status |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Mesh-ref 256 | 256 | 90.88% | 90.38% | 129 | Downsampled ×4 (from Original 1024) | 90.85% | +0.03 pp | DONE |
| Mesh-ref 4096 | 4096 | 92.00% | 91.18% | 148 | Original 1024 baseline | 91.95% | +0.05 pp | DONE |

## Interpretation notes

- Mesh-ref 256/4096 are independent area-weighted samples from `.off`, not downsampled from Original 1024.
- Compare Mesh-ref 256 vs existing Downsampled×4 256: same count, different sampling process.
- Compare Mesh-ref 4096 vs Original 1024: different counts; useful as denser mesh-sample baseline for Line A.
