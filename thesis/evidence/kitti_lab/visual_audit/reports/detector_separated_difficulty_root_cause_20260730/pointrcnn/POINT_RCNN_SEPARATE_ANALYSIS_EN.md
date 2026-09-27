# PointRCNN standalone analysis: Easy / Moderate / Hard, positive control, and root cause

## 1. Scope of this analysis

This report covers PointRCNN only. It shares no figure, table, or class-level conclusion with CenterPoint. The frozen PointRCNN configuration is trained and evaluated on `Car` only, so there is no missing-figure problem for Pedestrian/Cyclist here.

- Formal two-line results: all 3,769 KITTI val frames, canonical E2 16,384 points.
- Positive-control ratio experiment: a fixed 256-frame Car subset with nested slot replacement. It explains a mechanism and must not be presented as a full-validation-set conclusion.
- Metrics: BBox, BEV, and 3D AP_R40, reported separately for Easy, Moderate, and Hard.
- What the user calls `diff/difficult` corresponds to the official KITTI name `Hard` throughout this report.

## 2. Formal results: Car

### Line A: original baseline → original + x4 upsampling

#### BBox

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 99.32 | 94.08 | 90.10 |
| PDANS | 93.32 (-6.00) | 81.75 (-12.34) | 76.85 (-13.25) |
| PU-GCN | 92.44 (-6.87) | 71.77 (-22.32) | 67.07 (-23.03) |
| PU-EdgeFormer | 81.95 (-17.36) | 58.80 (-35.29) | 52.30 (-37.80) |
| PU-Net | 38.46 (-60.86) | 23.89 (-70.19) | 20.69 (-69.41) |

#### BEV

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 95.98 | 88.98 | 86.59 |
| PDANS | 90.00 (-5.98) | 78.02 (-10.96) | 73.21 (-13.39) |
| PU-GCN | 86.93 (-9.06) | 66.79 (-22.19) | 62.02 (-24.58) |
| PU-EdgeFormer | 75.97 (-20.01) | 53.83 (-35.15) | 48.94 (-37.65) |
| PU-Net | 28.95 (-67.03) | 18.74 (-70.24) | 16.85 (-69.75) |

#### 3D

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 92.27 | 82.26 | 77.95 |
| PDANS | 84.61 (-7.66) | 67.22 (-15.03) | 62.40 (-15.55) |
| PU-GCN | 80.17 (-12.10) | 58.28 (-23.98) | 53.47 (-24.47) |
| PU-EdgeFormer | 65.68 (-26.60) | 44.96 (-37.30) | 40.19 (-37.75) |
| PU-Net | 11.76 (-80.51) | 8.38 (-73.88) | 7.44 (-70.50) |

Figures:

- `figures/car_line_a_absolute_ap.png`
- `figures/car_line_a_delta_ap.png`

Difficulty-level conclusions:

1. The 3D drop for PDANS is `-7.66` Easy, `-15.03` Moderate, `-15.55` Hard. Moderate and Hard objects lose roughly 7–8 AP more than Easy, which shows that distant, occluded, and truncated objects are more sensitive to incorrectly added neighbourhoods.
2. PU-GCN and PU-EdgeFormer degrade far more at Moderate/Hard than at Easy. What the generated points damage first is not the obvious nearby car, but the difficult vehicles that only had a handful of boundary points to begin with.
3. PU-Net fails at all three difficulty levels. The larger Easy drop does not mean "Easy is more fragile"; it reflects a global geometric misalignment caused by a defect in how coordinate normalisation is wired in, and AP is already near its floor.

### Line B: downsampled-x4 baseline → downsampled + x4 upsampling

#### BBox

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 95.24 | 79.95 | 75.58 |
| PDANS | 83.75 (-11.50) | 60.33 (-19.62) | 53.79 (-21.79) |
| PU-GCN | 60.61 (-34.63) | 38.61 (-41.34) | 32.40 (-43.19) |
| PU-EdgeFormer | 52.75 (-42.49) | 33.49 (-46.46) | 28.71 (-46.88) |
| PU-Net | 33.62 (-61.62) | 20.51 (-59.44) | 18.40 (-57.18) |

#### BEV

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 91.04 | 76.40 | 72.03 |
| PDANS | 76.63 (-14.41) | 54.57 (-21.83) | 48.14 (-23.89) |
| PU-GCN | 57.25 (-33.79) | 36.19 (-40.20) | 31.43 (-40.61) |
| PU-EdgeFormer | 43.65 (-47.39) | 27.74 (-48.66) | 23.38 (-48.65) |
| PU-Net | 23.77 (-67.27) | 15.52 (-60.88) | 13.62 (-58.41) |

#### 3D

| Input | Easy AP (Δ) | Moderate AP (Δ) | Hard AP (Δ) |
|---|---|---|---|
| Baseline | 85.18 | 65.75 | 61.38 |
| PDANS | 65.01 (-20.16) | 44.08 (-21.67) | 39.17 (-22.21) |
| PU-GCN | 45.93 (-39.25) | 28.68 (-37.07) | 24.13 (-37.25) |
| PU-EdgeFormer | 33.76 (-51.41) | 20.57 (-45.18) | 17.62 (-43.77) |
| PU-Net | 12.41 (-72.77) | 8.78 (-56.97) | 7.76 (-53.62) |

Figures:

- `figures/car_line_b_absolute_ap.png`
- `figures/car_line_b_delta_ap.png`

In Line B, PDANS drops by `-20.16` even at Easy, and by `-21.67 / -22.21` at Moderate/Hard; PU-GCN and PU-EdgeFormer drop further still. This shows the networks are not reconstructing the real scan evidence that was removed. Instead they replace part of the reliable observations inside PointRCNN's fixed 16,384-point budget with generated points.

## 3. Why an apparently legitimate improvement appeared earlier

The earlier positive result was:

`Original / PDANS / 2.5% generated slots / Car / 3D Moderate`

It rose from `82.66` to `85.10`, i.e. `+2.43`. But all three difficulty levels and all four metrics have to be laid out in full:

| Metric | Difficulty | Baseline | Observed control (Δ) | PDANS g2.5 (Δ) | PDANS − control |
|---|---|---|---|---|---|
| BBox | Easy | 99.37 | 99.49 (+0.13) | 99.22 (-0.14) | -0.27 |
| BBox | Moderate | 94.12 | 93.93 (-0.19) | 92.31 (-1.81) | -1.62 |
| BBox | Hard | 89.59 | 89.48 (-0.11) | 89.53 (-0.06) | +0.05 |
| BEV | Easy | 96.45 | 96.53 (+0.08) | 96.43 (-0.03) | -0.10 |
| BEV | Moderate | 90.91 | 90.83 (-0.08) | 91.08 (+0.17) | +0.25 |
| BEV | Hard | 88.44 | 86.36 (-2.08) | 86.37 (-2.07) | +0.00 |
| 3D | Easy | 95.09 | 94.80 (-0.29) | 95.44 (+0.35) | +0.64 |
| 3D | Moderate | 82.66 | 84.42 (+1.75) | 85.10 (+2.43) | +0.68 |
| 3D | Hard | 79.93 | 79.82 (-0.10) | 78.63 (-1.30) | -1.20 |
| AOS | Easy | 99.36 | 99.48 (+0.13) | 99.22 (-0.14) | -0.27 |
| AOS | Moderate | 94.01 | 93.80 (-0.21) | 92.19 (-1.82) | -1.61 |
| AOS | Hard | 89.44 | 89.33 (-0.11) | 89.34 (-0.10) | +0.02 |

Key judgements:

1. The only clear rise is in **3D Moderate**. 3D Easy gains just `+0.35`, and 3D Hard actually falls by `-1.30`; BBox Moderate falls by `-1.81` and BEV Moderate gains only `+0.17`. So this is not "overall detection performance improved" — it is a local optimum specific to one difficulty level and one metric.
2. The same-slot real-point control already raises 3D Moderate from `82.66` to `84.42`, contributing `+1.75`. PDANS beats its matched control by only `+0.68`. Most of the gain comes from a change in sampling coverage under a fixed point budget and cannot be attributed to the generative model.
3. 2.5% corresponds to about 410 generated slots; roughly 97.5% of PointRCNN's input is still real observation. A small number of high-quality PDANS points may fill in individual car surfaces that sampling missed, while remaining too few to shift neighbourhood statistics broadly.
4. Raising the ratio to 5%–10% makes the positive gain disappear immediately, and every method decreases monotonically overall across the coarse 10%–50% range. This constitutes dose–response evidence for "low dose supplies local evidence, high dose contaminates the neighbourhood".
5. This result comes from a 256-frame screened subset and cannot yet replace the formal 3,769-frame result. In the thesis it should be called a `positive mechanism screen` or a `positive-control candidate`.

Corresponding figures:

- `figures/car_pdans_g025_positive_attribution.png`
- `figures/car_fine_ratio_easy.png`
- `figures/car_fine_ratio_moderate.png`
- `figures/car_fine_ratio_hard.png`

## 4. The PointRCNN-specific root-cause chain

PointRCNN performs foreground segmentation, local feature aggregation, and proposal generation directly on points. A fixed input point count means newly added generated points are never free: they change which real points are retained, the membership of ball neighbourhoods, local density, and the evidence available for proposal regression.

Box transitions over the full validation set further confirm this is not an AP-parsing artefact:

- Line A, Car lost/baseline-TP: PDANS `23.1%`, PU-GCN `32.7%`, PU-EdgeFormer `47.9%`, PU-Net `80.9%`.
- Line B: `35.4% / 55.4% / 67.0% / 76.5%`.
- The loss rate rises overall across the 0–20 m, 20–40 m, and 40 m+ bands; see `figures/car_line_*_loss_by_distance.png`.
- The sampler-safe fallback affects only 3–12 of 3,769 frames, below 0.4%, which rules it out as the primary cause.

So the causal chain is:

`geometric error or redundancy in generated points → changed real/generated composition within the fixed point budget → changed local neighbourhoods and foreground scores → degraded proposal confidence/localisation → higher box-loss rate at Moderate/Hard → AP drop`

## 5. Improvement priorities for PointRCNN

1. Treat 2.5% as a safe starting point and re-verify PDANS against a matched real-point control on the full 3,769 frames. Do not go straight to a strict x4 high generation ratio.
2. Preserve real points first. Generated points should only fill empty slots or the neighbourhoods of low-coverage objects; randomly replacing reliable real points must be forbidden.
3. Attach a confidence value to generated points and filter by local surface consistency, range-image adjacency, and normal residual.
4. Make the ratio adaptive to object distance and original point count: close to 0% for complete nearby vehicles, a small top-up allowed for sparse distant ones.
5. Fix the shared patch locality and the PU-Net normalisation before comparing model capability.
6. If a higher generation ratio is required, fine-tune PointRCNN on the same mixed distribution rather than only changing the test-time input.

## 6. Data available for re-checking

- `tables/formal_ap_all_difficulties.csv`
- `tables/transitions_full_validation.csv`
- `tables/transitions_by_distance.csv`
- `tables/positive_control_256_frame_all_metrics.csv`
- `tables/positive_control_attribution.csv`
