# ModelNet40 主实验 — 当前结果解读

- Updated: 2026-07-01

## 主协议 ×4 结果表

| Line | Method                   | Input points | PointNet++ points | Best acc | Baseline |    Delta |
| ---- | ------------------------ | -----------: | ----------------: | -------: | -------: | -------: |
| A    | Original baseline        |         1024 |              1024 |   91.95% |        - |        - |
| A    | Original + EAR ×4        |         1024 |              4096 |   91.48% |   91.95% | -0.47 pp |
| A    | Original + PDANS ×4      |         1024 |              4096 |   91.62% |   91.95% | -0.33 pp |
| B    | Downsampled50 baseline   |          512 |               512 |   91.26% |        - |        - |
| B    | Downsampled50 + EAR ×4   |          512 |              2048 |   90.61% |   91.26% | -0.65 pp |
| B    | Downsampled50 + PDANS ×4 |          512 |              2048 |   91.47% |   91.26% | +0.21 pp |

## 协议说明

- EAR 和 PDANS 均遵循同一 **×4 main protocol**：
  - Line A: 1024 → ×4 → 4096 → PointNet++ (`allow_resample=false`)
  - Line B: 512 → ×4 → 2048 → PointNet++ (`allow_resample=false`)
- Baseline 分支保持 native point count，不做 padding/resample 对齐 upsampling 点数。
- TULIP 仅 supplementary，不进入 main protocol。
- SPU-PMD 不进入 ModelNet40 main protocol。

## 已完成方法

### EAR ×4 — COMPLETE

- Generation audit: **PASS**
- Method provenance audit: **PASS** (`method_confirmed`)
- PointNet++ training: **COMPLETED** (200/200)

### PDANS ×4 — COMPLETE

- Generation audit: **PASS**
- Method provenance audit: **PASS** (`method_confirmed`)
- PointNet++ training: **COMPLETED** (200/200)

## 下一步

- **当前待完成方法：PU-Net ×4**
- PU-GCN 排在 PU-Net 之后（同样依赖 TF custom ops validate）
