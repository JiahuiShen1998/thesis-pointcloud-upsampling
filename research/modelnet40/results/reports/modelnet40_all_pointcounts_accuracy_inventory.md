# All PointNet++ SSG retrainings, by point count — complete inventory

- Generated: 2026-08-17
- Scope: every **completed** PointNet++ run in this project (smoke runs excluded)
- Model: `pointnet2_cls_ssg` in **all 22 runs**, trained **from scratch** at its own `num_point`
- No run anywhere in this project loads a checkpoint trained at a different point count

## Master table

| Pts | Run | Best OA | Final OA | Best Class | best ep | Data source | Round |
|---:|---|---:|---:|---:|---:|---|---|
| 256 | `mesh_ref_baseline_256` | 90.88 % | 90.38 % | 86.34 % | 129 | mesh re-sampled from `.off` | final |
| 256 | `lineB_downsampled_x4_baseline` | 90.85 % | 90.46 % | 87.37 % | 124 | Original 1024 decimated ×4 | final |
| 512 | `downsampled50_native512_pointnet2` | 91.26 % | 91.14 % | 88.00 % | 197 | Original decimated ~50 % | early |
| 1024 | `lineA_original_baseline` | 91.95 % | 91.31 % | 88.00 % | 85 | Original mesh sampling | final |
| 1024 | `original_baseline` | 91.95 % | — | 88.00 % | 85 | same data, earlier job | early |
| 1024 | `lineB_.../pu_net` | 91.27 % | 90.65 % | 87.86 % | 148 | 256 → PU-Net ×4 | final |
| 1024 | `downsampled50_baseline` | 91.17 % | — | 87.71 % | 155 | **512 on disk, loader resampled to 1024** | early |
| 1024 | `step8_downsampled50_ear_pointnet2` | 90.82 % | — | 87.06 % | 111 | 512 → EAR ×2 | early |
| 1024 | `lineB_.../pdans` | 90.34 % | 89.46 % | 86.20 % | 145 | 256 → PDANS ×4 | final |
| 1024 | `lineB_.../pu_gcn` | 90.06 % | 89.63 % | 86.41 % | 164 | 256 → PU-GCN ×4 | final |
| 1024 | `lineB_.../pu_edgeformer` | 89.32 % | 88.58 % | 85.21 % | 194 | 256 → PU-EdgeFormer ×4 | final |
| 1024 | `lineB_.../ear` | 88.81 % | 88.02 % | 84.50 % | 116 | 256 → EAR ×4 | final |
| 2048 | `downsampled50_pdans_x4_pointnet2` | 91.47 % | 90.99 % | 88.19 % | 164 | 512 → PDANS ×4 | early |
| 2048 | `downsampled50_ear_x4_pointnet2` | 90.61 % | 89.76 % | 87.45 % | 74 | 512 → EAR ×4 | early |
| 4096 | `mesh_ref_baseline_4096` | 92.00 % | 91.18 % | 88.70 % | 148 | mesh re-sampled from `.off` | final |
| 4096 | `lineA_.../pu_gcn` | 91.63 % | 91.00 % | 87.74 % | 81 | 1024 → PU-GCN ×4 | final |
| 4096 | `lineA_.../pdans` | 91.62 % | 90.87 % | 88.44 % | 84 | 1024 → PDANS ×4 | final |
| 4096 | `original_pdans_x4_pointnet2` | 91.62 % | 90.87 % | 88.44 % | 84 | same data, earlier job | early |
| 4096 | `lineA_.../ear` | 91.48 % | 90.85 % | 88.19 % | 61 | 1024 → EAR ×4 | final |
| 4096 | `original_ear_x4_pointnet2` | 91.48 % | 90.85 % | 88.19 % | 61 | same data, earlier job | early |
| 4096 | `lineA_.../pu_net` | 90.95 % | 90.55 % | 87.46 % | 189 | 1024 → PU-Net ×4 | final |
| 4096 | `lineA_.../pu_edgeformer` | 90.54 % | 90.13 % | 86.35 % | 83 | 1024 → PU-EdgeFormer ×4 | final |

`final` = `pointnet2_results/x4_two_line_final/`, Jul 8 – Aug 4.
`early` = `outputs/` + `reports/<run>/result_summary.csv`, Jun 20 – Jun 30 (superseded protocol).

## Same-seed determinism, verified three times

Three pairs were trained twice, in separate SLURM jobs 8+ days apart, seed 42 both times:

| Pair | Weight tensors identical | Max element-wise diff |
|---|---:|---:|
| Original baseline 1024 | 79 / 79 | 0.000e+00 |
| EAR 4096 | 79 / 79 | 0.000e+00 |
| PDANS 4096 | 79 / 79 | 0.000e+00 |

The only differing checkpoint field is the `variant` string. **The training pipeline is bit-deterministic
given the seed.** This does *not* speak to seed-to-seed variance, which remains unmeasured.

## The 1024-point information ladder

Four ways of arriving at 1024 points, ordered by how much real measurement each retains:

| Source of the 1024 points | Best OA |
|---|---:|
| Genuine sampling (Original) | 91.95 % |
| 512 real points, loader-resampled to 1024 | 91.17 % |
| 256 real points, PU-Net ×4 (best upsampler) | 91.27 % |
| 512 real points, EAR ×2 | 90.82 % |

Naive loader resampling (91.17 %) beats EAR's generated points (90.82 %) by 0.35 pp: duplicating
points is safer than inventing them.

## Protocol-dependent sign flip — PDANS

| Protocol | Branch | Baseline | Δ |
|---|---:|---:|---:|
| old, 512 → ×4 → 2048 | 91.47 % | 91.26 % (native 512) | **+0.21 pp** |
| current, 256 → ×4 → 1024 | 90.34 % | 90.85 % (native 256) | **−0.51 pp** |

Same method, same ratio, same SSG-from-scratch training. Only the decimation depth differs, and the
sign of the conclusion changes with it.
