# KITTI point-cloud upsampling degrades 3D detection: a two-line, two-detector evidence report

Date: 2026-07-30

## 1. Core conclusions

The degradation in these experiments is not a single-detector bug, and it cannot be summarised as "the generated points are noisy". The current results support a three-stage failure chain:

1. **The upstream local-geometry assumption is violated.** The current shared patch extractor sorts points by a coarse spatial ordering and then cuts contiguous 2,048-point blocks. Measured on Line B patches, the XY diagonal has a median of about 30.46 m and a p90 of about 124.10 m. One patch may contain road, several objects, and distant background at once — not the local surface that PU-Net, PU-GCN, PU-EdgeFormer, and PDANS assumed during training.
2. **The *number* of added points far exceeds the added effective sensor evidence.** On Line B, the reference voxel recall rises only from about 45.8% to 54.8%–59.0%, yet 45.5%–62.1% of the 0.2 m voxels in the final cloud are unsupported by the complete original scan; the reference precision of generated voxels is only 23.8%–46.0%.
3. **Wrong or redundant geometry is amplified by the detectors' limited input mechanisms.** PointRCNN's RPN always receives 16,384 points; CenterPoint allows at most 5 points per voxel and at most 40,000 voxels at test time. Generated points change PointRCNN's local neighbourhoods and foreground scores, and they change CenterPoint's voxel occupancy, voxel means, BEV heatmap, and the centre/size/rotation regression of boxes.

The real effect of the current pipeline is therefore not "a sparse cloud becomes dense", but:

`a small amount of correct surface coverage recovered + a large amount of redundant/displaced voxels + detector-specific competition for a limited budget`

This explains why:

- In Line A the original scan is already complete, so upsampling only adds estimates with no real measurement left to recover; all four methods degrade on both detectors.
- In Line B there is some recovery, but the cost of wrong geometry is larger; every Car metric at 0.7 IoU falls.
- CenterPoint's Pedestrian/Cyclist show a few small positive gains on Line B, because the small-object baseline is extremely sparse and the IoU threshold is 0.5. This does not contradict the overall geometric contamination.
- The method ranking is essentially stable across both detectors as `PDANS > PU-GCN > PU-EdgeFormer > the current PU-Net pipeline`, and it matches the ranking of generated-voxel reference precision.

## 2. Frozen protocol and evidence boundary

| Item | PointRCNN | CenterPoint |
|---|---|---|
| Formal input | E1 upsampled cloud through the unified E2 adapter | exact-4N E1 |
| Detector effective input | FOV/range + 0.1 m voxel representative + stratified deterministic sampling, exactly 16,384 points | FOV/range + 0.05 × 0.05 × 0.1 m voxels, at most 5 points per voxel, at most 40,000 voxels |
| Classes | current checkpoint/config is Car only | Car, Pedestrian, Cyclist |
| Full metrics | 3,769 KITTI val frames, BBox/BEV/3D AP_R40 | 3,769 KITTI val frames, BBox/BEV/3D AP_R40 |
| Three-frame audit | same-class oriented 3D IoU, Car 0.70 | Car 0.70, Pedestrian/Cyclist 0.50 |

The per-box matching over three frames explains "what happened to which object". It does not replace the KITTI evaluator's difficulty handling, DontCare handling, or full-threshold integration. Every number in a thesis main table must be taken from `analysis/all_detector_ap_r40_bbox_bev_3d.csv`.

PointRCNN showing Car only is not a missing visualisation class; it is the current frozen config `CLASSES: Car`. CenterPoint's frozen config explicitly includes Car, Pedestrian, and Cyclist.

> Note for the portable bundle: `analysis/` and `scripts/` are **not** shipped here — they live on the experiment host. This bundle contains `case_index.csv`, `selected_frames.txt`, the `frames/` tree, and the per-detector `tables/` under `../detector_separated_difficulty_root_cause_20260730/`.

## 3. Full AP results

### 3.1 PointRCNN: Car Moderate AP_R40

| Line | Input | BBox | BEV | 3D | Δ3D vs this line's baseline |
|---|---|---:|---:|---:|---:|
| A | Original baseline | 94.08 | 88.98 | 82.26 | — |
| A | PDANS | 81.75 | 78.02 | 67.22 | -15.03 |
| A | PU-GCN | 71.77 | 66.79 | 58.28 | -23.98 |
| A | PU-EdgeFormer | 58.80 | 53.83 | 44.96 | -37.30 |
| A | PU-Net* | 23.89 | 18.74 | 8.38 | -73.88 |
| B | Downsampled baseline | 79.95 | 76.40 | 65.75 | — |
| B | PDANS | 60.33 | 54.57 | 44.08 | -21.67 |
| B | PU-GCN | 38.61 | 36.19 | 28.68 | -37.07 |
| B | PU-EdgeFormer | 33.49 | 27.74 | 20.57 | -45.18 |
| B | PU-Net* | 20.51 | 15.52 | 8.78 | -56.97 |

### 3.2 CenterPoint: Moderate 3D AP_R40

| Line | Input | Car | Pedestrian | Cyclist |
|---|---|---:|---:|---:|
| A | Original baseline | 79.28 | 50.65 | 64.61 |
| A | PDANS | 64.36 (-14.92) | 45.19 (-5.46) | 45.90 (-18.71) |
| A | PU-GCN | 58.90 (-20.38) | 44.15 (-6.51) | 44.31 (-20.30) |
| A | PU-EdgeFormer | 45.73 (-33.54) | 36.82 (-13.83) | 35.22 (-29.38) |
| A | PU-Net* | 10.75 (-68.53) | 19.49 (-31.16) | 13.19 (-51.41) |
| B | Downsampled baseline | 64.60 | 24.57 | 15.46 |
| B | PDANS | 46.75 (-17.85) | 28.18 (**+3.61**) | 15.52 (+0.06) |
| B | PU-GCN | 39.07 (-25.53) | 24.93 (+0.36) | 16.85 (**+1.39**) |
| B | PU-EdgeFormer | 24.75 (-39.85) | 15.59 (-8.98) | 7.04 (-8.42) |
| B | PU-Net* | 11.72 (-52.88) | 11.92 (-12.65) | 5.02 (-10.44) |

`*` The current PU-Net numbers are a confirmed wrapper-defect pipeline result and must not be taken as the capability ceiling of the PU-Net method.

The complete 138-row table across Easy/Moderate/Hard and BBox/BEV/3D is at:

`analysis/all_detector_ap_r40_bbox_bev_3d.csv`

## 4. Full per-GT transition statistics

For every evaluable GT across the 3,769 frames, we determine separately whether the baseline and the upsampled input reach the corresponding 3D IoU threshold. Below is the fraction of Car cases where a baseline TP becomes an FN after upsampling:

| Detector | Line | PDANS | PU-GCN | PU-EdgeFormer | PU-Net* |
|---|---|---:|---:|---:|---:|
| PointRCNN | A | 23.1% | 32.7% | 47.9% | 80.9% |
| PointRCNN | B | 35.4% | 55.4% | 67.0% | 76.5% |
| CenterPoint | A | 20.6% | 25.9% | 39.3% | 70.8% |
| CenterPoint | B | 27.6% | 37.4% | 55.3% | 67.0% |

These ratios serve three purposes:

1. They show the AP drop corresponds to a real per-object increase in missed detections, not an AP text-parsing error.
2. Two structurally different detectors produce a consistent method ranking, which rules out "incompatible with one particular detector architecture".
3. In Line B the loss fraction is generally higher than Line A for both PointRCNN and CenterPoint, showing that "restoring the point count from 1/4-sparse back to the original" does not restore the original spatial evidence.

The complete maintained / confidence-degraded / localization-degraded / lost / recovered / missed-by-both statistics by detector, line, method, and class are at:

- `analysis/transition_summary_full_val.csv`
- `analysis/transition_summary_by_distance.csv`
- `figures/full_val_lost_vs_recovered_gt.png`
- `figures/full_val_car_loss_by_distance.png`

## 5. Why these three frames were chosen

The three frames were first ranked by per-box events across methods, lines, and detectors over all 3,769 frames, then constrained to "must contain a recovered box" and "must contain multiple small-object classes":

| Frame | GT composition | Main purpose |
|---|---|---|
| 000590 | 17 Car + 1 Cyclist | Consecutive Cars in a dense street going from high-IoU baseline boxes to missed detections; shows the cascade as methods get worse |
| 005625 | 15 Car + 1 Cyclist | Contains both lost and recovered in the same frame; both PointRCNN and CenterPoint show positive and negative changes |
| 006682 | 3 Car + 7 Pedestrian + 5 Cyclist | CenterPoint multi-class, small-object, 0.5 IoU case; confirms that a little recovery coexists with a lot of loss |

### 5.1 Frame 000590: recovery is not dominant, consecutive misses are

- PointRCNN Line A / PDANS: 4 baseline TPs lost, plus 2 localization-degraded and 6 confidence-degraded.
- Among them, GT 2 (x ≈ 11.35 m) had baseline 3D IoU 0.777, GT 4 (x ≈ 16.25 m) 0.815, GT 10 (x ≈ 19.27 m) 0.738, and GT 15 (x ≈ 39.76 m) 0.807; after upsampling none of them has a Car box reaching 0.7.
- PointRCNN Line A / PU-EdgeFormer and PU-Net each lose 9 baseline TPs.
- CenterPoint Line A / PDANS loses 7 Cars; PU-GCN 10; PU-EdgeFormer and PU-Net 12 each. This frame reproduces the same ranking as the full AP.

### 5.2 Frame 005625: recovered boxes genuinely exist, but far fewer than lost boxes

- PointRCNN Line A / PDANS: GT 4 (x ≈ 5.38 m) goes from baseline FN to upsampled TP with 3D IoU = 0.792; in the same comparison 3 Cars still go from TP to FN.
- PointRCNN Line B / PDANS: GT 7 (x ≈ 13.26 m) recovers with 3D IoU = 0.705, while 5 baseline TPs are lost.
- PointRCNN Line B / PU-EdgeFormer: GT 4 recovers with 3D IoU = 0.795, while 9 baseline TPs are lost.
- CenterPoint Line B / PDANS: the Cyclist at 27.13 m recovers with 3D IoU = 0.549, consistent with the tiny full-set Cyclist gain of +0.06 AP.
- CenterPoint Line B / PU-GCN: the same Cyclist recovers to 3D IoU = 0.655 and a Car at 31.01 m also recovers; yet this frame still loses 7 Cars.

This frame is an important counter-example for the thesis: upsampling does not "never help". Rather, the local help in the current pipeline is not enough to offset the broader loss.

### 5.3 Frame 006682: small objects can benefit occasionally, Car and the overall picture still degrade

- CenterPoint Line A / PDANS: 3 Cyclists lost, 1 Cyclist localization-degraded, 1 Pedestrian lost.
- CenterPoint Line A / PU-Net: 1 Pedestrian recovered, but 3 Pedestrians, 4 Cyclists, and 1 Car lost.
- CenterPoint Line B / PU-GCN: 1 Pedestrian recovered, but 1 Pedestrian and 2 Cyclists lost.
- PointRCNN is Car-only; in Line B all four methods lose the same baseline Car TP in this frame.

## 6. Why PointRCNN degrades

[PointRCNN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.pdf) is a two-stage point-based detector: the RPN generates 3D proposals from point-level local neighbourhoods, and the RCNN stage does RoI pooling and box refinement. The key facts about the current frozen configuration:

- The RPN receives only 16,384 points.
- Set abstraction radii grow from 0.1/0.5 m up to 2/4 m.
- Both the RPN and RCNN score thresholds are 0.3, and the RCNN NMS threshold is 0.1.

In the current E2 input, the upper bound on exact observed rows is typically only about 23%–26% across the four methods and both lines; the remaining ~74%–77% are generated / non-observed rows. Therefore:

1. Generated points directly change the membership and local density of PointNet++ ball-query/kNN neighbourhoods.
2. Displaced points create false foreground continuity and dilute true boundaries, changing the point-wise foreground score.
3. Generated points that occupy the fixed 16,384 slots crowd out independent real measurements; even where Line A's E1 keeps all observed points, E2 cannot possibly keep them all.
4. Once the proposal centre/size/orientation shifts, Car easily falls from above 3D IoU 0.7 to below it. The localization degradation and losses in the per-box statistics are exactly this phenomenon.

The ratio experiment supplies independent dose evidence: as the generated ratio rises from 10% to 50%, PointRCNN Car Moderate 3D AP decreases essentially monotonically across both lines and all four methods, and replacing the same fraction with real points is clearly better than with generated points. The only candidate worth full verification is Line A / PDANS / 2.5%, but the matched real-point control already accounts for most of its +2.43 AP; the generated geometry beats the control by only +0.68.

## 7. Why CenterPoint degrades

[CenterPoint](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html) represents objects as BEV centre points and regresses centre offset, height, size, and rotation. The current OpenPCDet configuration uses MeanVFE, a sparse 3D backbone, HeightCompression, and CenterHead.

The frozen data processing is:

- detection range `[0,70.4] × [-40,40] × [-3,1] m`
- voxel size `0.05 × 0.05 × 0.1 m`
- at most 5 points per voxel
- at most 40,000 voxels at test time

Median pre-cap voxel counts and cap hits over 32 Line A frames:

| Input | Median pre-cap voxels | 40k cap hits |
|---|---:|---:|
| Original | 14,944 | 0/32 |
| PDANS | 36,913 | 6/32 |
| PU-GCN | 45,108 | 29/32 |
| PU-EdgeFormer | 46,182 | 29/32 |
| PU-Net* | 55,384 | 32/32 |

This shows that in Line A some real measurements compete with generated points for the 5-points-per-voxel and 40k-voxel budgets. Merely reordering the identical float32 point set into observed-first order gives a small Line A Moderate 3D AP recovery, most visibly for PU-GCN, PU-EdgeFormer, and PU-Net; this directly demonstrates that input order and the limited voxel budget are a real mechanism.

But it is not the primary cause:

- After observed-first, all Line A methods are still below the original baseline.
- All four Line B methods hit the 40k cap 0 times in 32/32 samples, and Car still drops by 17.85–52.88 AP.
- Changing point order in Line B changes AP by only about -0.0018 to +0.0311.

So CenterPoint's main loss is still the wrong position/distribution of generated voxels; the cap is only an additional amplifier in Line A.

## 8. Method-by-method explanation

### 8.1 PDANS

[PDANS](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Point_Cloud_Upsampling_Using_Conditional_Diffusion_Module_with_Adaptive_Noise_CVPR_2025_paper.pdf) uses a conditional diffusion module with Adaptive Noise Suppression. It has the highest generated-voxel reference precision in the current results:

- Line A: 47.5%
- Line B: 46.0%

PDANS is therefore the strongest of the four methods on both detectors and both lines. It still degrades because ANS can suppress local noise but cannot turn a non-local patch spanning 30–124 m back into a genuine local surface, and it cannot conjure back the sensor-ray evidence that downsampling removed.

### 8.2 PU-GCN

[PU-GCN](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_PU-GCN_Point_Cloud_Upsampling_Using_Graph_Convolutional_Networks_CVPR_2021_paper.pdf) models multi-scale neighbourhoods with Inception DenseGCN and expands points with NodeShuffle. Its current generated-voxel reference precision is:

- Line A: 37.2%
- Line B: 33.0%

When the shared patch is non-local, graph edges no longer connect true neighbours on the same surface, and NodeShuffle propagates the wrong neighbourhood relations into more points. Its AP and per-box loss rate are therefore consistently worse than PDANS.

### 8.3 PU-EdgeFormer

[PU-EdgeFormer](https://arxiv.org/abs/2305.01148) combines EdgeConv with multi-head self-attention, aiming to learn local and global structure simultaneously. Its current generated-voxel reference precision drops to:

- Line A: 29.5%
- Line B: 26.2%

On a non-local patch, "global relations" build attention across road, vehicles, and background, and the effect of a wrong association spreads more easily than in a local GCN. It shows a higher TP loss rate and lower AP on both detectors.

### 8.4 PU-Net

[PU-Net](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.html) is a patch-level method using PointNet++ multi-scale features, feature expansion, and coordinate reconstruction. The current wrapper, at lines 120–124, feeds metric-scale patches straight into a pretrained network with `bradius=1.0` and saves the output directly, with no:

- patch centroid subtraction
- radius/scale normalization
- output inverse scale
- output centroid restoration

The current PU-Net therefore has the lowest reference voxel precision (Line A 17.2%, Line B 23.8%) and the worst AP. This is a confirmed integration defect and must not be written up as "PU-Net is inherently unsuitable for detection".

## 9. Alternative explanations that are ruled out or weakened

| Alternative explanation | Verdict from the evidence |
|---|---|
| It is just the PointRCNN sampler crash/patch | No. The sampler-safe fallback triggers on only 3–12 of 3,769 frames; the full AP and per-box drops are far larger than that range |
| It is just PointRCNN's 16,384 cap | No. PointRCNN is affected, but CenterPoint shows the same method ranking, and CenterPoint Line B degrades without any 40k cap |
| It is just CenterPoint's 40k voxel cap | No. Line B has 0/32 cap hits and still shows a large Car drop |
| It is just point order | No. Observed-first gives only limited recovery in Line A and almost none in Line B |
| It is just a different total point count | No. Line B exact-4N ends close to the original scan's point count, yet effective reference voxel recovery is limited; a same-ratio real-point control beats generated points |
| It is just the AP evaluator/parsing | No. Full recall@0.7, per-frame prediction counts, per-GT lost/recovered, and specific box IoUs all change in step |
| All added points are harmful | No. Frames 005625 and 006682 contain clearly recovered Car/Pedestrian/Cyclist; the problem is that the quantity and stability of the gains are insufficient to offset the losses |

## 10. The routes most likely to improve performance

### P0: fix upstream validity first, do not keep tuning the detector

1. Replace the shared contiguous spatial chunk with a genuinely local patch built from `FPS seed + kNN/ball-query`.
2. Patches must overlap, and be fused with distance-to-centre weights or confidence, to avoid seams and shell artefacts.
3. Reproduce each method's training-time centring, scale normalisation, and output inverse transform exactly; fix PU-Net first.
4. Record, for every generated point, its source patch, local scale, neighbourhood radius, and model confidence / diffusion uncertainty.

### P1: generated points must not carry the same weight as real points

1. Always keep observed points; generated points may only fill the remaining budget.
2. Filter generated points by local plane residual, normal consistency, range-image ray consistency, an upper bound on neighbourhood density, and model uncertainty.
3. For PointRCNN, change E2 to observed-first stratified sampling; verify generated ratios of 2.5%/5%/10% fully before considering anything like 75%.
4. For CenterPoint, voxelise observed-first; prefer real points within a voxel and top up to 5 points with generated ones. This can reduce cap competition in Line A, but do not expect it to fix Line B geometry.

### P2: do detector-aware training, not just a test-time distribution swap

1. Fine-tune the detector on a mix of real KITTI and same-method upsampled input.
2. Randomise the generated ratio and the method during training, so the detector does not adapt to a single artificial distribution.
3. Add feature consistency: the foreground heatmap / proposals / boxes should agree between the original and the upsampled input.
4. Give generated points/voxels a provenance bit or confidence feature so the detector can learn to down-weight them; do not disguise them as measurements fully equivalent to real intensity.

### P3: the smallest falsifiable follow-up experiments

1. Full val: Line A / PDANS / g2.5, matched real-point c2.5, and baseline, with the same seed and the same input slots.
2. After fixing the patch and the PU-Net wrapper, run only a 32-frame geometry gate; do not spend full detector runs until generated-voxel precision is clearly above the current PDANS.
3. Run four CenterPoint ablations: original order, observed-first, observed-only, and observed-first + filtered-generated.
4. Run four PointRCNN ratios: 0/2.5/5/10%, recording per GT box the observed/generated counts, nearest-neighbour distance, and boundary shell fraction.
5. For the thesis main conclusions, use bootstrap frame resampling to give confidence intervals on AP deltas and transition rates, so a single-frame recovery is never written up as an overall improvement.

## 11. Visualisation and data artefacts

Every `frame × line × method × detector` contains:

- `full_frame_3d_bev.png`: the detector-effective 3D and BEV views for baseline and upsampled, showing all GT and all final prediction boxes.
- `case_manifest.json`: point-cloud provenance, all GT, all baseline/upsampled predictions, scores, BEV/3D IoU, transitions, and visualisation paths.
- `object_crops/`: up to 3 representative changed objects pre-rendered per case; any other GT can be cropped on demand with the Open3D viewer's `--gt-index`.

Shared point-cloud page:

- `pointcloud_full_frame.png`: the complete frame, observed as small gray points and generated as small blue points.

Full listings:

- `analysis/all_selected_frame_boxes.csv`: all GT and all final prediction boxes for both detectors across the three frames.
- `analysis/selected_frame_gt_transitions.csv`: per-GT baseline→upsampled transitions.
- `analysis/object_crop_index.csv`: the pre-rendered crop index.
- `case_index.csv`: the unified entry point for the 48 detector cases.

Native 3D interaction without HTML:

```bash
/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python \
  scripts/view_dual_detector_evidence_open3d.py \
  --frame 005625 --line B --method pdans --detector centerpoint \
  --state upsampled --effective --gt-index 4 --point-size 1.0
```

> In the portable bundle, use `viewer/gui_viewer.py` (button-driven GUI) or `viewer/portable_open3d_viewer.py`, which takes the same command-line arguments as above.

## 12. Suggested causal wording for the thesis

Supportable:

> With the detectors, checkpoints, val split, and detector-specific input protocols frozen, four upsampling methods produce a consistent Car performance ranking and a consistent per-GT increase in missed detections across two structurally different detectors. Real-point ratio/order controls show that the limited input budget is an amplifying factor. That Line B degrades without any voxel cap, that a same-ratio real-point control beats generated points, and that generated-voxel reference precision matches the AP ranking, together point to non-local patches and wrong/redundant generated geometry as the primary cause.

Should not be claimed directly:

- That every lost box was caused by one specific generated point.
- That the current PU-Net numbers represent the method's ceiling.
- That three visualisation frames can substitute for the official AP over 3,769 frames.
- That CenterPoint's small-object positive gains represent overall upsampling recovery success.
