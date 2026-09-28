# ModelNet40 Main Experiment - Current Results Interpretation

- Updated: 2026-07-01

## Master protocol ×4 Results Table

| Line | Method                   | Input points | PointNet++ points | Best acc | Baseline |    Delta |
| ---- | ------------------------ | -----------: | ----------------: | -------: | -------: | -------: |
| A    | Original baseline        |         1024 |              1024 |   91.95% |        - |        - |
| A    | Original + EAR ×4        |         1024 |              4096 |   91.48% |   91.95% | -0.47 pp |
| A    | Original + PDANS ×4      |         1024 |              4096 |   91.62% |   91.95% | -0.33 pp |
| B    | Downsampled50 baseline   |          512 |               512 |   91.26% |        - |        - |
| B    | Downsampled50 + EAR ×4   |          512 |              2048 |   90.61% |   91.26% | -0.65 pp |
| B    | Downsampled50 + PDANS ×4 |          512 |              2048 |   91.47% |   91.26% | +0.21 pp |

## protocol Description

- EAR and PDANS follow the same **×4 main protocol**:
  - Line A: 1024 → ×4 → 4096 → PointNet++ (`allow_resample=false`)
  - Line B: 512 → ×4 → 2048 → PointNet++ (`allow_resample=false`)
- Baseline branch keeps native point count, does not do padding/resample alignment upsampling point count.
- TULIP only supplementary, do not enter main protocol.
- SPU-PMD does not enter ModelNet40 main protocol.

## Completed Method

### EAR ×4 — COMPLETE

- Generation audit: **PASS**
- Method provenance audit: **PASS** (`method_confirmed`)
- PointNet++ training: **COMPLETED** (200/200)

### PDANS ×4 — COMPLETE

- Generation audit: **PASS**
- Method provenance audit: **PASS** (`method_confirmed`)
- PointNet++ training: **COMPLETED** (200/200)

## Next

- **Current Method to Finish: PU-Net ×4**
- PU-GCN after PU-Net (also dependent on TF custom ops validate)
