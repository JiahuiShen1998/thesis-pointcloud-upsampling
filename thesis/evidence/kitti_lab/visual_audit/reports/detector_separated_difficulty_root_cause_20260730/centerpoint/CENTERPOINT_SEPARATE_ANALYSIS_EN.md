# CenterPoint standalone analysis: per-class Easy / Moderate / Hard and the positive exceptions

## 1. Scope of this analysis

This report covers CenterPoint only. Car, Pedestrian, and Cyclist each get their own section and their own figures; different classes are never plotted in the same performance figure. Every class covers BBox, BEV, and 3D, and Easy, Moderate, and Hard.

All formal results come from the 3,769 KITTI val frames. CenterPoint uses 0.05 × 0.05 × 0.1 m voxels, at most 5 points per voxel, at most 40,000 voxels at test time, and compresses the sparse 3D features to BEV before predicting centres and box parameters.

What the user calls `diff/difficult` corresponds to the official KITTI name `Hard` throughout this report.

## 2. Car

### Line A

#### BBox

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 94.94 | 92.38 | 90.38 |
| PDANS | 91.03 (-3.91) | 78.23 (-14.15) | 75.03 (-15.34) |
| PU-GCN | 90.41 (-4.53) | 72.00 (-20.39) | 70.32 (-20.06) |
| PU-EdgeFormer | 80.17 (-14.77) | 58.96 (-33.43) | 57.11 (-33.26) |
| PU-Net | 42.46 (-52.48) | 24.81 (-67.57) | 23.85 (-66.52) |

#### BEV

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 92.18 | 88.43 | 87.03 |
| PDANS | 89.16 (-3.01) | 75.03 (-13.40) | 71.80 (-15.24) |
| PU-GCN | 87.49 (-4.69) | 69.48 (-18.95) | 67.07 (-19.96) |
| PU-EdgeFormer | 76.27 (-15.90) | 55.97 (-32.46) | 53.33 (-33.71) |
| PU-Net | 34.25 (-57.93) | 20.35 (-68.07) | 18.89 (-68.15) |

#### 3D

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 88.39 | 79.28 | 76.74 |
| PDANS | 82.35 (-6.04) | 64.36 (-14.92) | 60.74 (-16.00) |
| PU-GCN | 79.28 (-9.11) | 58.90 (-20.38) | 55.84 (-20.89) |
| PU-EdgeFormer | 66.28 (-22.11) | 45.73 (-33.54) | 42.50 (-34.24) |
| PU-Net | 16.05 (-72.34) | 10.75 (-68.53) | 10.07 (-66.67) |

### Line B

#### BBox

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 93.78 | 81.34 | 77.26 |
| PDANS | 87.87 (-5.91) | 64.31 (-17.03) | 60.57 (-16.69) |
| PU-GCN | 79.73 (-14.05) | 53.97 (-27.37) | 50.14 (-27.12) |
| PU-EdgeFormer | 60.27 (-33.51) | 38.80 (-42.55) | 36.63 (-40.63) |
| PU-Net | 45.04 (-48.74) | 26.48 (-54.86) | 24.86 (-52.40) |

#### BEV

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 89.12 | 77.31 | 73.16 |
| PDANS | 80.38 (-8.74) | 57.59 (-19.71) | 53.75 (-19.41) |
| PU-GCN | 74.37 (-14.76) | 49.82 (-27.49) | 45.72 (-27.43) |
| PU-EdgeFormer | 51.53 (-37.59) | 33.17 (-44.14) | 30.26 (-42.90) |
| PU-Net | 33.50 (-55.62) | 20.64 (-56.67) | 18.62 (-54.54) |

#### 3D

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 81.40 | 64.60 | 59.97 |
| PDANS | 69.63 (-11.77) | 46.75 (-17.85) | 42.59 (-17.38) |
| PU-GCN | 60.42 (-20.98) | 39.07 (-25.53) | 34.97 (-25.00) |
| PU-EdgeFormer | 39.38 (-42.02) | 24.75 (-39.85) | 21.66 (-38.31) |
| PU-Net | 17.53 (-63.87) | 11.72 (-52.88) | 10.44 (-49.53) |

Car degrades on both lines, at all three difficulty levels, for all four methods. The Line B 3D Moderate drop is `-17.85` for PDANS, `-25.53` for PU-GCN, `-39.85` for PU-EdgeFormer, and `-52.88` for PU-Net. Car uses a 0.7 IoU threshold, so even a slight deviation in centre, size, orientation, or extent converts into AP loss; spurious extra voxels are particularly harmful to regressing a complete 3D hull.

The Car figures contain Car only:

- `figures/car_line_a_absolute_ap.png` and `car_line_a_delta_ap.png`
- `figures/car_line_b_absolute_ap.png` and `car_line_b_delta_ap.png`
- `figures/car_line_a_transitions.png` and `car_line_b_transitions.png`
- `figures/car_line_a_loss_by_distance.png` and `car_line_b_loss_by_distance.png`

## 3. Pedestrian

### Line A

#### BBox

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 74.89 | 71.62 | 67.75 |
| PDANS | 67.09 (-7.79) | 62.09 (-9.53) | 58.85 (-8.91) |
| PU-GCN | 65.19 (-9.69) | 60.99 (-10.63) | 57.81 (-9.95) |
| PU-EdgeFormer | 55.77 (-19.12) | 49.91 (-21.70) | 46.95 (-20.80) |
| PU-Net | 33.70 (-41.19) | 30.26 (-41.36) | 28.13 (-39.62) |

#### BEV

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 58.98 | 55.74 | 51.62 |
| PDANS | 54.39 (-4.58) | 49.83 (-5.91) | 46.67 (-4.94) |
| PU-GCN | 51.13 (-7.84) | 48.42 (-7.32) | 44.73 (-6.88) |
| PU-EdgeFormer | 44.66 (-14.31) | 39.33 (-16.41) | 36.32 (-15.30) |
| PU-Net | 26.72 (-32.25) | 23.50 (-32.24) | 20.73 (-30.88) |

#### 3D

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 54.04 | 50.65 | 46.25 |
| PDANS | 49.65 (-4.39) | 45.19 (-5.46) | 41.72 (-4.53) |
| PU-GCN | 46.94 (-7.10) | 44.15 (-6.51) | 40.57 (-5.68) |
| PU-EdgeFormer | 41.96 (-12.08) | 36.82 (-13.83) | 33.65 (-12.60) |
| PU-Net | 22.19 (-31.84) | 19.49 (-31.16) | 17.09 (-29.16) |

Line A degrades at all three difficulty levels. The reference voxel recall of the original scan is already 100%, so added points mainly perturb the voxel means near small-object centres and the BEV heatmap, with no genuinely missing evidence left to recover.

### Line B

#### BBox

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 58.47 | 52.94 | 48.37 |
| PDANS | 58.66 (+0.19) | 53.28 (+0.35) | 48.81 (+0.43) |
| PU-GCN | 52.31 (-6.16) | 46.76 (-6.18) | 42.57 (-5.80) |
| PU-EdgeFormer | 31.36 (-27.11) | 28.20 (-24.74) | 26.09 (-22.28) |
| PU-Net | 24.96 (-33.51) | 22.08 (-30.86) | 20.43 (-27.94) |

#### BEV

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 34.49 | 31.55 | 28.01 |
| PDANS | 37.68 (+3.19) | 33.77 (+2.22) | 30.28 (+2.27) |
| PU-GCN | 33.29 (-1.20) | 30.08 (-1.47) | 26.27 (-1.74) |
| PU-EdgeFormer | 20.41 (-14.08) | 18.40 (-13.15) | 16.38 (-11.63) |
| PU-Net | 16.65 (-17.84) | 14.40 (-17.15) | 12.78 (-15.23) |

#### 3D

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 27.20 | 24.57 | 21.67 |
| PDANS | 31.42 (+4.23) | 28.18 (+3.61) | 24.75 (+3.08) |
| PU-GCN | 27.74 (+0.55) | 24.93 (+0.36) | 21.58 (-0.09) |
| PU-EdgeFormer | 17.40 (-9.79) | 15.59 (-8.98) | 13.63 (-8.04) |
| PU-Net | 14.05 (-13.15) | 11.92 (-12.65) | 10.55 (-11.12) |

PDANS is a positive exception that is consistent across difficulty levels and across spatial metrics:

- BBox Easy/Moderate/Hard: `+0.19 / +0.35 / +0.43`
- BEV: `+3.19 / +2.22 / +2.27`
- 3D: `+4.23 / +3.61 / +3.08`

This is more credible than PointRCNN's single Moderate positive point, because all three difficulty levels and both BEV and 3D point the same way. The behavioural evidence: the total number of Pedestrian predictions falls from `28353` to `27297`, while predictions with score ≥ 0.5 rise from `1277` to `1438`. PDANS therefore removes a large number of low-scoring candidates while strengthening the high-scoring voxel support for a subset of genuine small objects.

PU-GCN on Pedestrian gains only `+0.55 / +0.36` at 3D Easy/Moderate and `-0.09` at Hard, while BBox and BEV both fall, so it cannot be treated as a stable improvement.

The Pedestrian figures contain Pedestrian only; for the positive evidence see:

- `figures/pedestrian_line_b_positive_mechanism.png`

## 4. Cyclist

### Line A

#### BBox

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 87.90 | 75.39 | 71.92 |
| PDANS | 79.43 (-8.47) | 55.14 (-20.26) | 52.74 (-19.18) |
| PU-GCN | 77.40 (-10.50) | 54.85 (-20.54) | 52.40 (-19.52) |
| PU-EdgeFormer | 63.98 (-23.92) | 43.27 (-32.13) | 41.44 (-30.48) |
| PU-Net | 38.46 (-49.44) | 26.28 (-49.11) | 25.91 (-46.01) |

#### BEV

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 82.38 | 68.70 | 65.26 |
| PDANS | 73.90 (-8.48) | 49.03 (-19.67) | 46.54 (-18.72) |
| PU-GCN | 73.04 (-9.34) | 47.57 (-21.13) | 45.35 (-19.91) |
| PU-EdgeFormer | 58.79 (-23.59) | 37.68 (-31.02) | 35.89 (-29.36) |
| PU-Net | 24.76 (-57.62) | 16.42 (-52.28) | 15.90 (-49.36) |

#### 3D

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 79.24 | 64.61 | 60.99 |
| PDANS | 71.08 (-8.17) | 45.90 (-18.71) | 43.19 (-17.80) |
| PU-GCN | 70.72 (-8.52) | 44.31 (-20.30) | 41.86 (-19.13) |
| PU-EdgeFormer | 55.62 (-23.62) | 35.22 (-29.38) | 33.53 (-27.46) |
| PU-Net | 20.84 (-58.40) | 13.19 (-51.41) | 12.89 (-48.10) |

### Line B

#### BBox

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 32.84 | 20.80 | 20.41 |
| PDANS | 39.35 (+6.51) | 24.27 (+3.47) | 23.13 (+2.72) |
| PU-GCN | 39.98 (+7.14) | 23.82 (+3.01) | 23.29 (+2.88) |
| PU-EdgeFormer | 19.10 (-13.74) | 10.78 (-10.02) | 10.61 (-9.80) |
| PU-Net | 19.27 (-13.57) | 11.02 (-9.78) | 11.06 (-9.36) |

#### BEV

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 28.27 | 17.13 | 16.33 |
| PDANS | 32.28 (+4.01) | 18.65 (+1.53) | 17.75 (+1.42) |
| PU-GCN | 33.82 (+5.54) | 19.10 (+1.98) | 18.20 (+1.87) |
| PU-EdgeFormer | 16.32 (-11.95) | 8.59 (-8.54) | 8.16 (-8.17) |
| PU-Net | 10.86 (-17.42) | 6.25 (-10.88) | 6.38 (-9.95) |

#### 3D

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 25.81 | 15.46 | 14.49 |
| PDANS | 28.13 (+2.32) | 15.52 (+0.06) | 14.89 (+0.40) |
| PU-GCN | 30.97 (+5.17) | 16.85 (+1.39) | 16.19 (+1.70) |
| PU-EdgeFormer | 13.49 (-12.32) | 7.04 (-8.42) | 6.93 (-7.57) |
| PU-Net | 8.74 (-17.07) | 5.02 (-10.44) | 4.72 (-9.77) |

Line B contains two explainable positive results:

1. PDANS gives `+2.32 / +0.06 / +0.40` at 3D Easy/Moderate/Hard. In the single-threshold transition over the full validation set, `102` were recovered and `100` lost, a net `+2`, and predictions with score ≥ 0.5 rose from `204` to `322`.
2. PU-GCN gives `+5.17 / +1.39 / +1.70` at 3D Easy/Moderate/Hard, with `94` recovered and `81` lost, a net `+13`, and predictions with score ≥ 0.5 rising to `286`. BBox and BEV also improve at all three difficulty levels, which makes this the most complete small-object recovery evidence currently available.

Cyclist uses a 0.5 IoU threshold, and after downsampling the baseline 3D Moderate is only 15.46. A small number of correctly generated voxels can give a cyclist that previously had very few points a continuous centre response, and the error tolerance is wider than Car's 0.7 IoU.

The Cyclist figures contain Cyclist only; for the positive evidence see:

- `figures/cyclist_line_b_positive_mechanism.png`

## 5. Why small objects can gain while Car still loses

No method reaches the 40k voxel cap in Line B, so neither the positive nor the negative results are explained by cap truncation. The geometric audit shows:

- The downsampled baseline has 45.8% reference recall at 0.2 m voxels.
- PDANS and PU-GCN raise this to 59.0% and 57.3% respectively, but the resulting extra voxels still amount to 45.5% and 57.7%.
- For Pedestrian and Cyclist, a few correctly added voxels can cross the minimum evidence threshold for "forming a detectable centre".
- Car requires a more complete and accurately positioned hull, size, and orientation; the accumulated damage from wrong voxels exceeds the recall benefit.
- PU-EdgeFormer and PU-Net add more extra voxels at lower reference precision, so all three classes degrade.

It therefore cannot be written as "upsampling helps all small objects". The accurate statement is:

> In the downsampled Line B, PDANS on Pedestrian and PU-GCN/PDANS on Cyclist show class-specific recovery. This gain depends on low baseline density, the 0.5 IoU threshold, and a small number of correctly added voxels, and it does not transfer to Car or to Line A.

## 6. Interpretation of Easy / Moderate / Hard

1. Easy small objects are usually closer and less occluded, so correctly added voxels more readily form a stable centre; the Cyclist/PU-GCN Easy gain is the largest (`+5.17`).
2. Moderate contains more partially occluded and mid-range objects. PDANS/Pedestrian still holds `+3.61`, which shows its added voxels are of sufficient quality to improve part of the sparse centre response.
3. Hard objects are further away and more occluded, so effective recovery and spurious-voxel contamination both intensify. PDANS/Pedestrian still reaches `+3.08`, but PU-GCN/Pedestrian is already near zero, showing that method quality determines whether a gain survives across difficulty levels.
4. Car still drops markedly at Hard for every method, which shows current point-generation accuracy is not sufficient for 0.7 IoU 3D box regression.

## 7. Improvement priorities for CenterPoint

1. Do not apply unconditional x4 to the complete Line A scan; prioritise low-occupancy voxels and small-object candidate regions only.
2. Use Pedestrian/PDANS and Cyclist/PU-GCN as positive samples to learn *which generated voxels to keep*, rather than keeping them uniformly per method.
3. Filter out bridging and drifting points using 0.2 m reference consistency, local plane residual, and range-image adjacency.
4. In voxel-level fusion, preserve the real-point mean and treat generated points as weighted residual features, so the MeanVFE centre is not moved directly.
5. Choose the generation budget per class and per distance: stricter for Car, with a higher but controlled local ratio allowed for small objects in Line B.
6. Fix the patch locality and the PU-Net coordinate transform before re-verifying under the same protocol.

## 8. Data available for re-checking

- `tables/formal_ap_all_difficulties.csv`
- `tables/transitions_full_validation.csv`
- `tables/transitions_by_distance.csv`
- `tables/positive_exception_ap.csv`
- `tables/detector_behavior_summary.csv`
- `tables/voxel_mechanism_summary.csv`
