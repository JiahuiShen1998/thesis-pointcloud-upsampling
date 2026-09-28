# General record of all implemented, experimental adjustments and results

> 2026-09-17 Update Hint: This is reserved as 09-10 history snapshot.The latest double detector convergence results and a combined time line from February to March.[Full working record of the experiment.](/home/ra87racy/reports/EXPERIMENT_WORK_RECORD_202602_TO_20260917_EN.md); Quick report.[Briefing](/home/ra87racy/reports/EXPERIMENT_BRIEF_20260917_EN.md). Do not misinterpret the status of 09-17 by missing the following operational status, best results or convergence evidence.

**Check deadline: 2026-09-10 (Europe/Berlin)**
**Scope of verification:`/home/ra87racy` Local Workspace**
**Principle: Only code changes, command output, logs, manifest, projection documents, assessment forms, checkpoint, reporting or clean-up lists are recorded for work that can be shown to have actually taken place.**

---

## 1. How does this general record distinguish between "do" and "do"

Status definition:

- **Completed**: Target data and assessments are completed and complete outputs or results tables are available.
- **Partially completed**: Only generation, smoke, subset, training or some stage, no completes the original full assessment.
- **Failed/ Suspended**: actually started, but failed or stopped because of environmental problems, CUDA, data, sampling, running time or resources.
- **Changes completed but no final result**: The code/environment/script has indeed been modified, but no is sufficient evidence that the whole experiment has been completed.
- **Audit/analysis completed**(a) Read the output and calculate the analysis of geometric, voxel, detection transfer, sampling;Not the results of the new model training.
- **Not implemented**: appears only in plan, empty directory or list of feasibility, and does not count as experimental results.

This time scan to PointRCNN Main Project `results/` Down **132 First Level Results Directory**. These include both formal full-scale experiments and smoke, pilot, ablation, trouble diagnosis, visualization and reporting generation.This paper combines the records by “experimental” to avoid miscalculating multiple versions of the same projection into several independent experiments.

### 1.1 Warning for Critical Evaluation

1. Final matrix display for 2026-09-08 to 09-10 **KITTI AP_R40**Every arm must **3,769/3,769** frame only entered main table.
2.  Earlier.  PointRCNN  Local  evaluator  Default  41  individual  precision sample  Medium Every  4  One by one. The essence is old.  **AP_R11**. The old table is kept as historical evidence, but cannot compares directly with the latest AP_R40 table.
3. Early experiments used different kinds of experiments. `RPN.NUM_POINTS`, distance proposal, input sampling, whole frame /patch, point count cap and checkpoint;They cannot is encoded as a "same protocol rankings."
4. CenterPoint entered by voxel and PointRCNN entered by fixed point count sampling;The document level exact 4× does not imply detector actually uses all 4× points.

---

## 2. Overall conclusion of the current work

As of 2026-09-10, the most complete and well-documented conclusion is that:

1. **Strictly expand point count to 4×, no Automatic recovery and even improve 3D detection accuracy.** Under frozen detector, all the official upsampling methods are, in general, below for baseline, Line B (first down sampling and upsampling).
2. **The main problem is not a single “point count is deficient”.**  Reasons for validation include: old  patch  Non-local, normalization error, repeat point, generated points activates a lot of new  voxel, PointRCNN 16,384  Point entrance sampling, CenterPoint  Every  voxel  Most  5  Points / Most  40,000 voxels,  And the upsampling distribution does not match the detector training distribution.
3. **Retain observed points and do detector adaptation, which is clearly recovery, but still no exceeds the pair baseline.** The latest full-val is the closest baseline result is CenterPoint:
   - Line A, observed `N` + predicted `3N`, adapted: Car Moderate 3D AP_R40 **74.7012**Yes, official baseline **79.2773**Oh, shit. **-4.5761**.
   - Line B, observed `M` + predicted `3M`, adapted: Car Moderate **61.6639**Yes, adapted baseline **68.0490**Oh, shit. **-6.3851**.
4. **Training losses have declined, but cannot claims that 3 epochs has been convergence.** Only epoch 3 checkpoint, no epoch 1/2 validation AP, or no plateau or predefined early stopping evidence.
5. **ModelNet40 currently has only PU-GCN small-scale exact-4× smoke;no Full 12,311 Sample local results and no PointNet++ classification accuracy rate.** This part of the cannot is written as a complete experiment.

Current final pipeline state: Line A and Line B **3,769/3,769** point cloud complete, test matrix **20/20 PASS**;  no is running at the time of verification  PU-GCN, PDANS, PointRCNN, CenterPoint, OpenPCDet  or  TULIP  Research process. Disk Balance **31–32 GiB**.

---

## 3. Final application of strict 4× protocol

So the original KITTI point cloud is `N` Point:

- **Line A**: Original `N` Point-to-Pilot upsampling, which is strictly `4N`. The line test "enrichment of existing measurements" is not an information recovery.
- **Line B**: decrease from fixed seed to without replacement `M=floor(N/4)`And upsampling to the strict. `4M`. The line is the protocol that is close to "slough input recovery original density".
- **direct**: detector Direct Reader Strict `4N`/`4M` is the web output.
- **observed-first**: Retain complete observed `N`/`M`, and from the strict prediction point, use the stable seed without replacement `3N`/`3M`Grand total `4N`/`4M`.

Final PU-GCN patch process:

- patch Size `2048`Output `8192`;
- `fps_ball_cover_knn_v3`;
- Line A primary radius 2 m, Line B 4 m; cover radius 6 m; Support point threshold 2,048;
- patch is a single central and unit ball normalization, which is counter-edited to normalization;
- After merging all patch raw candidates, draw exact at fixed seed without replacement `4N`/`4M`; Direct failure when not enough exact target, prohibiting copying points to fill;
- intensity inherits 1-NN from XYZ to observed XYZ;
- PU-GCN fixed by PU1K `model-100`no retrain upsampling network on KITTI.

---

## 4. Updated and complete results: PU-GCN × detector adaptation × full KITTI val

### 4.1 PointRCNN, Car 3D AP_R40, 3,769 frame /arm

| Line | Enter | Weight | Easy / Moderate / Hard | Moderate margin for pairing baselines | Status |
|---|---|---|---:|---:|---|
| A | baseline `N` | official | 92.4325 / **81.9528** / 77.8468 | 0 | Completed |
| A | direct `4N` | official | 84.7443 / **61.9294** / 57.7869 | -20.0235 | Completed |
| A | direct `4N` | adapted | 85.1671 / **68.8800** / 64.6697 | -13.0728¹ | Completed |
| A | observed `N` + predicted `3N` | official | 85.0374 / **64.2748** / 60.0911 | -17.6780 | Completed |
| A | observed `N` + predicted `3N` | adapted | 86.7068 / **71.0075** / 66.7939 | -10.9453¹ | Completed |
| B | baseline `M` | official | 85.2016 / **65.5757** / 61.2831 | 0 | Completed |
| B | baseline `M` | adapted | 84.3920 / **68.3312** / 64.2613 | 0 | Completed |
| B | direct `4M` | official | 46.4657 / **30.0909** / 25.8000 | -35.4848 | Completed |
| B | direct `4M` | adapted | 68.2592 / **46.6150** / 40.2487 | -21.7163 | Completed |
| B | observed `M` + predicted `3M` | official | 55.0177 / **35.5289** / 30.9359 | -30.0469 | Completed |
| B | observed `M` + predicted `3M` | adapted | 74.1970 / **55.2145** / 49.0583 | -13.1167 | Completed |

¹ Line A  no Additional training  adapted original-`N` baseline, so the two adapted margin values are referred to as official Line A baseline, with a change in input and weight, which should be interpreted conservatively.

### 4.2 CenterPoint, Car 3D AP_R40, 3,769 frame /arm

| Line | Enter | Weight | Easy / Moderate / Hard | Moderate margin for pairing baselines | Status |
|---|---|---|---:|---:|---|
| A | baseline `N` | official | 88.3907 / **79.2773** / 76.7371 | 0 | Completed |
| A | direct `4N` | official | 81.8888 / **60.6081** / 57.9061 | -18.6692 | Completed |
| A | observed `N` + predicted `3N` | official | 83.3121 / **63.2064** / 61.0418 | -16.0709 | Completed |
| A | observed `N` + predicted `3N` | adapted | 86.8154 / **74.7012** / 72.6570 | **-4.5761** | Completed |
| B | baseline `M` | official | 81.4003 / **64.5979** / 59.9701 | 0 | Completed |
| B | baseline `M` | adapted | 83.7514 / **68.0490** / 64.6309 | 0 | Completed |
| B | direct `4M` | official | 56.4211 / **35.3530** / 31.2043 | -29.2449 | Completed |
| B | observed `M` + predicted `3M` | official | 66.9217 / **44.0929** / 40.0268 | -20.5050 | Completed |
| B | observed `M` + predicted `3M` | adapted | 80.4109 / **61.6639** / 57.5252 | **-6.3851** | Completed |

### 4.3 CenterPoint Full-category review

Best observed-first adapted and no exceeded the match on Pedestrian/Cyclist baseline:

| Line | Category | baseline Moderate 3D AP_R40 | observed-first adapted | margin |
|---|---|---:|---:|---:|
| A | Pedestrian | 50.6533 | 48.4156 | -2.2377 |
| A | Cyclist | 64.6054 | 57.6989 | -6.9065 |
| B | Pedestrian | 47.4395 (adapted baseline) | 44.0568 | -3.3827 |
| B | Cyclist | 42.0672 (adapted baseline) | 33.5610 | -8.5062 |

### 4.4 Practical Training Adjustment

**PointRCNN adaptation: **Full 3,712 train frame;Each arm independent training;RPN 3 epochs + offline RCNN 3 epochs; batch size 1; workers 0; Adam one-cycle; LR `0.0002`; seed `20260823`; Close GT database augmentation;Save only epoch 3.

**CenterPoint adaptation: **Initialization of the official 80-epoch checkpoint;Full 3,712 train frame;batch size 2; workers 0; 3 epochs; Adam one-cycle; LR `0.0003`; weight decay `0.01`; seed `666`; Close GT sampling;Save only epoch 3.

Training loss E1  /  E3 Change:

- PointRCNN Line A direct: RPN `1.624055→1.347459` (-17.03%), RCNN `1.043027→0.959636` (-8.00%).
- PointRCNN Line B direct: RPN -27.61%, RCNN -7.65%.
- PointRCNN Line B baseline: RPN -23.95%, RCNN -4.16%.
- PointRCNN Line A observed-first: RPN -16.46%, RCNN -4.42%.
- PointRCNN Line B observed-first: RPN -24.82%, RCNN -6.86%.
- CenterPoint Line A: `2.33→2.11` (-9.44%); Line B PU-GCN: `3.69→3.10` (-15.99%); Line B baseline: `3.00→2.65` (-11.67%).

These figures only show that the optimization process is properly completed and loss is down.**No proof that validation has convergence**.

---

## 5.  Unity of the full amount frozen detector  exact-4×  Experiment 2026-07)

### 5.1 PointRCNN All 3,769 frame

The following values were retained in the report at that time;Since evaluator/ entered protocol earlier than 09, the final AP_R40 runner should be compared to history and not directly compared to section 4.

| Enter | Easy / Moderate / Hard 3D AP | Conclusions |
|---|---:|---|
| Original baseline | 92.2731 / **82.2554** / 77.9454 | Upper Border |
| Downsampled baseline | 85.1766 / **65.7460** / 61.3818 | sampling. |
| Line A PDANS | 84.6118 / **67.2235** / 62.3981 | Four ways best, but below original |
| Line A PU-GCN | 80.1727 / **58.2759** / 53.4719 | Decline |
| Line A PU-EdgeFormer | 65.6774 / **44.9596** / 40.1919 | Significant decline. |
| Line A PU-Net old | 11.7639 / **8.3802** / 7.4407 | Severe lapse |
| Line B PDANS | 65.0123 / **44.0752** / 39.1729 | Not recovery downsample baseline |
| Line B PU-GCN | 45.9303 / **28.6803** / 24.1310 | Decline |
| Line B PU-EdgeFormer | 33.7635 / **20.5693** / 17.6153 | Decline |
| Line B PU-Net old | 12.4097 / **8.7807** / 7.7637 | Severe lapse |

### 5.2 CenterPoint All 3,769 frame, Moderate 3D AP_R40

| Enter | Car | Pedestrian | Cyclist | Conclusions |
|---|---:|---:|---:|---|
| Original baseline | 79.2773 | 50.6533 | 64.6054 | original baseline |
| Downsampled baseline | 64.5979 | 24.5690 | 15.4577 | Distinct significant loss |
| Line A PDANS | 64.3574 | 45.1910 | 45.8953 | Four methods are the best, but still below original |
| Line A PU-GCN | 58.8955 | 44.1470 | 44.3082 | Decline |
| Line A PU-EdgeFormer | 45.7342 | 36.8205 | 35.2235 | Decline |
| Line A PU-Net old | 10.7512 | 19.4930 | 13.1928 | Severe lapse |
| Line B PDANS | 46.7512 | 28.1790 | 15.5186 | Car without recovery;Pedestrian Relative sparse baseline Improvement |
| Line B PU-GCN | 39.0714 | 24.9302 | 16.8458 | Total not recovery |
| Line B PU-EdgeFormer | 24.7476 | 15.5896 | 7.0374 | Decline |
| Line B PU-Net old | 11.7178 | 11.9174 | 5.0172 | Severe lapse |

### 5.3 Full Method Implementation Status

- **PDANS**: Initial due to extension source / `pytorch3d` The failure resulted in the repair of environmental and equipment-related codes and the final completion of the Line A, Line B strict exact-4× and double detector assessments.
- **PU-GCN**(b) Rehabilitation of TensorFlow/CUDA op, completion of full generation and assessment of both formal lines;And then I finished detector adaptation, the whole matrix.
- **PU-EdgeFormer**: The direct recurrence of the original warehouse environment failed;The results of protocol were then harmonized by replicating the compatible PU-GCN ops/checkpoint reasoning path.The fact must be spelled out that cannot claims to be an unsuited original warehouse with a single key.
- **PU-Net**(a) The following: Python 2, 3, TF custom op, CPU fallback, compilation and normalization rehabilitation;The results of the old adapter were extremely poor and the small-scale results improved after the rehabilitation, but were not yet the best approach.
- **TULIP**: complete Line A/Line B all 3,769 frame reasoning and PointRCNN/CenterPoint assessment, but output is not strict exact-4× and PointRCNN used `RPN=4096`, close distance proposal, cannot to be incorporated into the Unified Four Method main table.
- **EAR**: Only the early full volume/historical input assessment and subsequent 5 frame strict smoke completed;no completes the new strict Line A+Line B full weight generation.
- **SPU-PMD**: Only 4 frame smoke and PointRCNN small sample tried, no full-val AP.

---

## 6., Key Experimental Adjustments, Why, What happens when they're modified

| Adjustment | Actual modified/run | Result |
|---|---|---|
| Replace "point count approximately" with "strict exact 4×" | Line A `N→4N`; Line B `floor(N/4)→4M`; manifest Verifyed by frame | The method output point count was eliminated, but the test performance was not increased by point count |
| Old sequential/coarse-bin patch Local FPS/ball/kNN | Testing FPS, ball, cover,`fps_ball_cover_knn_v3` | patch Local improvements;The first local version was found with 59.4%, repeat row. |
| Add cover and unique kNN floor | surface-c32 / cover radius 6 m / unique 2048 support | 2048 supports the sole, repeated and zero, geometric coverage improvement;Testing is still not fully in excess of baseline |
| PU-Net unit ball normalization fix | 2×2: wrong/fixed normalization × old/local patch | PointRCNN is best raised from the number of digits in the wrong configuration to 43.5424;CenterPoint to 46.2047, but still poor |
| PointRCNN sampler-safe | far points  /  16,384 was changed from full candidate without replacement sampling;Allow replacement padding when thin | All Line As that have failed due to negative sample size have been completed;fallback % <0.4%; Fix the crash but cannot explains all AP drops |
| PointRCNN E3 observed-first 32,768 | observed Point Priority, keep measured points and then put the prediction point | Line A has improved somewhat, Line B almost no;Point to portal cap/, not the only reason for the sequence. |
| CenterPoint voxel Audit | Statistics occupied voxels, max-5 retention, 40,000 cap | Line A, PU-GCN/PUEF, etc., triggers voxel cap;Line B does not trigger cap, still falling, so cap is not the only reason |
| direct no-resample | Allows the network to read all valid points in an independent PointRCNN copy, no fixed 16,384 | Even original dropped from Moderate 81.95 to 74.91;The upsampling method is still poor, proving simple removal of resample, not repair |
| region split + 3 m halo | 2,067 shared core region, no 16,384 cut-off, three types of OpenPCDet PointRCNN | Some Ped/Cyc indicators have improved, but the whole method exact-4× pilot is still all below baseline main indicators |
| detector-aware PDANS V2 | voxel-only, proposal-only, full; PointRCNN and CenterPoint 256 frame | Failed to cross detector gate;CenterPoint Ped Improved but Car/Cyc Declined |
| V3 true internal proposals | PointRCNN pre-RCNN RPN proposals; CenterPoint pre-NMS heatmap top-K | Close to V2, proving that final-box bias is not the main problem. |
| V4 measured-voxel anchor | Disable generated-only voxel | PointRCNN Car Moderate 3D +0.3093; CenterPoint Car -0.0192, but Ped/Cyc is still slightly degraded |
| V5 predicted-class adaptive gate |  Just for...  CenterPoint  Projected  Car  It's...  proposals  I don't know. PointRCNN  Continue  V4 | pilot256 basically maintains CenterPoint for the entire category baseline;CenterPoint Car in stand-alone holdout64 slightly decreased without creating a stable cross-detector |
| observed-first | Keep observed fully and replace 3N/3M predicted | The frozen model has improved and is still poor;After full-train adaptation, recovery is the most obvious. |
| detector adaptation | 64 frame screening, 3,712 frame training, finally 3,769 frame complete validation | Close the gap significantly, but eventually no arm exceeds the match baseline |

---

## 7. Root Experiment and ablation Analysis

### 7.1 Non-local problem of old patch

- The old extractor, using coarse space bin, cuts the block by line and does not guarantee that the same patch is a local surface.
- Line B patch diagonal median of XY **30.46 m**, p90 **124.10 m**, clearly contradicts the distribution of local object patch during training.
- The first local patch version improves locality, but appears **59.4% duplicate rows**.
- This version: PointRCNN Moderate from **57.1829 to 43.6976**; CenterPoint from **61.1172 to 77.4150**. Also occupied voxels median value from **11,584.5 down to 2,124.5**, indicates that different detector has different preferences for point distribution.

### 7.2 PU-Net  normalization  × patch  It's...  2×2  Correlation ablation 256  frame)

| Configure | PointRCNN Moderate 3D | CenterPoint Moderate 3D |
|---|---:|---:|
| baseline | 81.3033 | 79.8368 |
| wrong norm + old patch | 7.9823 | 15.9788 |
| fixed norm + old patch | 18.2742 | 41.2227 |
| wrong norm + local patch | 6.3008 | 16.4119 |
| fixed norm + local patch | **43.5424** | **46.2047** |

The normalization error and patch are not only real problems, but they can be repaired in such a way as to be significant recovery, but are still not close to baseline.

### 7.3 PointRCNN / CenterPoint input bottleneck

- PointRCNN Default entry fixed 16,384 point.
- CenterPoint voxel size `[0.05,0.05,0.1]`For each voxel up to 5 points, test up to 40,000 voxels.
- 32 frame Line A Audit median voxels / Trigger 40k cap frame Number: original `14,944 / 0`; PDANS `36,913 / 6`; PU-GCN `45,108 / 29`; PU-EdgeFormer `46,182 / 29`; PU-Net `55,384 / 32`.
- Line B no triggers voxel cap, but there is still a marked decline in performance.So cap is an important issue for Line A, but not the only explanation across the board.

### 7.4 direct no-resample Full Volume Experiment

| Enter | PointRCNN Car 3D Easy / Moderate / Hard |
|---|---:|
| standard original, fixed 16,384 | 92.26 / **81.95** / 77.75 |
| original full all-valid | 79.83 / **74.91** / 72.73 |
| Line A PDANS all-valid | 83.59 / **63.49** / 54.34 |
| Line A PU-GCN all-valid | 78.21 / **54.26** / 47.11 |
| Line A PU-EdgeFormer all-valid | 67.11 / **43.85** / 37.10 |
| Line A PU-Net all-valid | 12.23 / **8.82** / 8.17 |

Conclusion: Internet training relies on fixed sampling distribution;Plug all points directly into the model and even damage original, cannot and use "sampling" as a recovery program.

### 7.5 dose / generated points ratio

Test on 256 frame `g2.5/g5/g7.5/g10`:

- Line A PDANS Moderate around `85.10 / 82.37 / 82.25 / 82.42`It's for baseline. `82.66`; Little dose has seen sub-benefits.
- Line B each dose all below corresponds to the thin baseline.
- All 3,769 frame `g10`: Line A PDANS/PU-GCN/PU-EdgeFormer/PU-Net Moderate `79.951/79.478/76.433/72.371`; Line B `59.048/55.954/52.415/45.700`.
- `g25/g50` no finished, cannot report is validated.

### 7.6 detector-aware PDANS V2–V5 (both 256 frame development experiments)

**V2: **

- PointRCNN baseline 81.1768; V1 full 82.3995 (+ 1.2227), but BEV -1.1421;V2 voxel-only 79.3805, proposal-only 75.7533, full 79.2269.
- CenterPoint V2 full: Car 76.7736 (-3.0632), Ped 48.5569 (+5.0476), Cyc 72.2834 (-3.8283).
- Conclusion: improvement of Ped only, not detector-wide;Not extended to 3,769.

**V3: **

- PointRCNN V3 full 79.2933 (relative to baseline -1.8835).
- CenterPoint V3 full: Car 76.7396 (-3.0972), Ped 48.5887 (+5.0794), Cyc 72.2856 (-3.8261).
- The voxel of V2/V3 is almost the same as the detection of the transfer;internal proposal to replace final boxes no to solve the problem.

**V4: **

- PointRCNN Car 81.4861 (+0.3093), BEV +0.2093.
- CenterPoint Car 79.8176 (-0.0192), Ped 43.0727 (-0.4366), Cyc 75.7678 (-0.3439).
- The new active voxel median is down to 0, which proves that generated-only voxel activation is a manageable causal factor;But a strict cross-category gate is not passed.

**V5: **

- PointRCNN continues with V4: Car +0.3093.
- CenterPoint: Car -0.0184, Ped +0.0004, Cyc +0.0000, basically back to baseline.
- V5 corresponds to 256 input files that exist and are evaluated, but it is a methodology development set.
- PDANS surface-c32 of the independent holdout64 **64/64 point cloud Generated**The test for CenterPoint baseline/V5 and AP_R40 are also completed in a separate directory: Car Moderate `74.1881→74.0512` (-0.1369), BEV `86.3893→85.2910` (-1.0983); Pedestrian, Cyclist is identical to baseline.As a result, no is now increasing to detector-wide.

### 7.7 object-preserving and Method Component ablation (PointRCNN, pilot256)

V1 workspace actually ran 8 selector/component variants.Car Moderate 3D AP_R40:

| Variables | Moderate |
|---|---:|
| baseline | 81.1768 |
| density-only | 79.8568 |
| fixed 2.5% PDANS | 81.4430 |
| full | **82.3995** |
| matched-real dynamic | 81.7996 |
| NN-only | 81.5914 |
| no-adaptive | 80.1478 |
| no-confidence | 81.6421 |
| no-sparse | 80.9414 |

A single seed for full which appears to be + 1.2227;The subsequent 16-seed probe proved that the gain was unstable, see 7.10.

Two other sets of patch-pair were actually completed:

- PDANS: old patch 57.9933, surface-c32 68.4744; PU-GCN: old patch 55.8076, surface-c32 63.4482; Same number of baseline 81.5569s.
- PU-EdgeFormer: old patch 41.7379, surface-c32 58.0212; PU-Net fixed: old patch 16.8097, surface-c32 22.9283, cover-kNN 44.0985; Same number of baseline 81.7441s.

Conclusion: surface/local patch is a substantial improvement on all methods, but not enough to reach baseline.

### 7.8 region-split Category III exact-4× pilot256

Co-use 2,067 core regions, 3 m halo, zero core point loss, each detector input does not exceed 16,384.Moderate 3D AP_R40:

| Methodology | Car | Pedestrian | Cyclist |
|---|---:|---:|---:|
| baseline | 77.8161 | 58.2090 | 78.2919 |
| PDANS surface-c32 | 68.8726 | 58.0672 | 71.5795 |
| PU-GCN surface-c32 | 61.5081 | 55.4667 | 57.4759 |
| PU-EdgeFormer surface-c32 | 61.2790 | 56.1431 | 56.1780 |
| PU-Net fixed surface-c32 | 26.1774 | 25.0164 | 28.9221 |

The main 3D indicator for all methods still does not fully exceed baseline;PDANS the closest.

The independent V4 holdout64 region-split has also been evaluated in practice: Car/Pedestrian/Cyclist Moderate 3D `75.7267→77.4777`, `38.1070→45.8565`, `15.0000→17.2222`. These differences apply only to the same region-split protocol;holdout has very few Ped/Cyc GT, cannot instead of frame full-val.

 The same.  V4  In original frame  OpenPCDet PointRCNN  Below displays a clear sampling sensitivity: certainty  `workers=0` holdout64 Car +4.0651, Ped -0.8332, Cyc -3.9286;Native `workers=2` Car -2.2718, Ped +2.5279, Cyc +1.6234.As a result, cannot claims a steady rise across sampling.

### 7.9 Line B c2048 / E1 / consensus / random Select Experiment (pilot256)

PointRCNN Car Moderate 3D:

| Select protocol | baseline | PDANS | PU-GCN | PU-EdgeFormer | PU-Net fixed |
|---|---:|---:|---:|---:|---:|
| c2048-r4 | 64.0416 | 36.6162 | 24.9619 | 21.2409 | 2.3971 |
| E1 | 64.0362 | **44.0487** | 32.8364 | 28.2585 | 10.5793 |
| consensus | 63.4197 | 39.2524 | **33.1648** | 22.4376 | **18.8792** |

CenterPoint  The same.  Line B pilot  It's...  Car Moderate: baseline 61.6558; random PDANS/PU-GCN/PU-EdgeFormer/PU-Net is `47.7441/41.3784/35.4329/15.1487`; consensus is `43.6085/37.3131/27.3251/19.8041`. Selecting the strategy changes the sorting of the method, but no sets recovery baseline.

### 7.10 PointRCNN 16-seed sampling Noise Probe

- fixed input, checkpoint, split and merge rules only change evaluator sampling seed and run seed 0–15.
- The new 24 slot all PASS;6 times CUDA occasional segmentation fault has been internally re-engineered.
- Car Moderate 3D: baseline `80.998±1.198`; V1 full `80.560±1.086`; Pair margin averages **-0.438**, sd **1.769**, Scope `-3.51…+2.23`.
- The original seed + 1.2227 is the maximum value of full arm 16 times;9/16 is negative, median -0.75.
- `MDE95≈3.47 AP`. This proves that the sorting of about 0.4–1.2 AP below split-region path is not readable;cannot sets this noise value directly to native E2 full-val.

### 7.11 exact-4× quality selection little probe

- Line A PDANS actually did confidence selection: no-quota 3 frame, voxel quota=6 is 5 frame.
- no-quota: quality precision median 0.6821, below random 0.8155;unsupported ratio 0.5825, above random 0.4940.
- quota=6: quality precision 0.5965, below random 0.8024;unsupported 0.6329, above random 0.5337.
- This is the 3/5 frame mechanism probe, no detector AP;As a result, the then confidence quality selector was rejected and not extended.

---

## 8. Training Graduation: 64 frame Screening All train pilot  /  full val

### 8.1 PointRCNN finetune64 Six Arm Screening

- fixed 64 train frame, actual 63;seed `20260823`.
- RPN 3 epochs + RCNN 3 epochs, every stage of epoch 93 steps.
- 6 input arm × pretrained/finetuned, for a total of 12 256-frame eval, all produced 256 forecasts and PASS.
- Moderate Change: Line A PU-GCN `60.2662→64.4408` (+4.1746); Line B PU-GCN `32.9201→37.5150` (+4.5949); Line B PDANS `43.0067→47.4327` (+4.4260); Line B baseline `66.6188→62.9139` (-3.7049).
- PDANS Line A values differ in the two versions of the report;A later summary of eval256 should be used, and cannot should select a more favourable old number.
- Conclusion: The very small training set can show that “the area fits the potential of recovery”, but baseline is also degraded and cannot concludes.

### 8.2  Full  3,712 train  It's...  256-val pilot

Training uses a full train split, but only fixed 256 val was evaluated:

- PointRCNN A PU-GCN adapted 67.2345, for baseline 79.1910, bad - 11.9565.
- PointRCNN B PU-GCN adapted 47.0204, for adapted baseline 67.2485, bad - 20.2281.
- CenterPoint A adapted 73.9468, for 79.8368, bad - 5.8900.
- CenterPoint B adapted 60.6978, for 65.5501, bad - 4.8523.
- PointRCNN observed-first: A 67.7909, compared to direct +0.5564;B 56.2443, more than direct +9.2239.

These are 256-frame pilot, which have been replaced by full 3,769-frame of Section 4, but are still progressive evidence of actual run-off.

---

## 9. Early baseline, method access and failure record (2026-05 to 06)

### 9.1 PointRCNN / EAR / PU-Net Early Job

- History PointRCNN baseline Record: 3D AP `89.19/78.85/77.91`.
- Once clean original rerun:`89.2040/78.6795/77.8099`.
- EAR Historical Report:`88.5558/77.9565/76.7821`A little below baseline.
- Early PU-Net x2 Full generation and assessment completed once: 3D AP `0.1976/1.1364/1.1364`This means that old appliances are completely incompatible.
- Follow-up recovered/fullframe report shows the appearance of PU-Net Moderate or about 59.2885;As the input is different from the recovery process, the very low result above is not the same as protocol and is retained as provenance risk.
-  Yes.  PU-Net  Increase  resume/chunk, Python 3, TensorFlow op  Compile, compile, GPU bootstrap, CPU fallback,  Data reading and normalization fix.

### 9.2 May 16 Unharmonized Profiles

Recorded Moderate 3D: original 77.93, downsample50 76.99, EAR 73.89, PU-Net 35.26, PU-GCN 50.65.PU-GCN was used. `RPN=26000`, the configuration is not uniform, so it is reserved for early exploration.

### 9.3 RPN4096 / no-distance-propose Comparison

- original failed with negative dimension sampling at 4,096.
- downsample Completed: 3D `83.6938/65.1662/60.4608`.
- EAR Completed:`81.5120/62.6687/57.8478`.
- PU-Net failed after 16 samples.
- PU-GCN was missing 8 frame, not completed.
- TULIP Completed:`27.5045/16.9675/13.7001`.

### 9.4 TULIP

- Finish full Line A Parsing Approximate `54.3492/35.0242/30.3302`; There are also Line B full and CenterPoint full results.
- CenterPoint Moderate: Line A Car/Ped/Cyc `31.77/15.98/4.59`; Line B `17.68/2.99/0.08`Far from below original `79.28/50.65/64.61`.
-  The primary output is...  range-image  Vertical  4×,  But switch back.  XYZ  I don't know.  exact 4×; Actual output is about 0.41× and 0.73–0.78×.
- 8  A known frame in  4,096/2,048/1,024  Configure Reasons  distance proposal  It's...  far/near bucket  Empty set entry  CUDA NMS  And abort. Add sole-nonempty-bucket guard later and use score-only proposal to bypass.

### 9.5 EAR strict-4× Feasibility

- 5 frame × Line A/B, for a total of 10 entries smoke all PASS.
- CPU whole frame achieves a global kNN/PCA and Python cycle;Line A About 26–29 min/ frame, Line B About 9–10 min/ frame.
- It is estimated that two lines will take approximately 92–103 days;A frame byte-identical duplication was also observed.
- For running costs no completes the new strict full-val regeneration;This is...**Time out/unfeasible**is not the final negative result of the model.

### 9.6 SPU-PMD

- First time because it's not available. `pyvista` Failure;Second cause `knn_cuda` Failure.
- Modify `utils/MeshUtil.py` with `main.py` After lazy import, 4 frame inference succeeded;For each complete frame, go down to 2,048, then export 8,192, finite, i.e. add a point to 3× relative to the internal input.
- PointRCNN `RPN=4096` An assessment of the failure of sampler underflow;was replaced by `RPN=2048` After 4 frame ran, detections was `1/0/0/2`.
- Too few samples, no AP;No full-val.

### 9.7 PDANS Initial failure and follow-up repair

- Initial inference due to lack of extension source and `pytorch3d` (a) Unrun;`pointops` It was successfully compiled and prepared for 4 XYZ input.
-  Modify  pointnet2  It's...  device/tensor  Processing, compilation and  CUDA  After compatibility, the formal follow-up is completed  Line A/Line B  Full.

### 9.8 PU-EdgeFormer failed in the early stages of direct recurrence

- Create a Python 3.6.8 / TensorFlow 1.13.1 environment.
- custom ops due to CUDA 10 path hard-coded, no `nvcc` And checkpoint failed.
- Only 3 KITTI frame was converted to 2,048 and generated Plotly, which was visualized, but at that time no model inference.
- The results of the subsequent formal unification were from the ops/reuse compatible path;The two phases must be presented separately.

### 9.9 ratio audit

The actual audit found that:

- EAR around `1.008×`, not the target x2.
- The PU-Net output ratio changes with frame.
- The PU-GCN/PDANS old process is influenced by 100k cap.
- TULIP is in range image vertical 4×, but may be less than the input point when returned to XYZ.

The audit directly facilitated strict exact-4× Line A/B protocol.

---

## 10. ModelNet40 Job Record

### 10.1 actually completed

- Create `modelnet40_pointnet2_upsampling` protocol and the structure of the report.
- PU-GCN Local smoke: Line A 8/8`1024→4096`; Line B 8/8, `256→1024`; Output finite, point count exact.
- ModelNet40 data preprocessing, upsampling fit, classification training portal, audit and reporting documents have been prepared/modified.
- `thesis_demo` Created `train_cls.py`, `pointnet_cls.py`, dataset/preprocess scaffold.

### 10.2 no Completed, cannot Written Results

- Both lines 12 and 311 complete local PU-GCN generate no submission/ no completion evidence.
- PointNet++ 5 classification branch is `NOT_SUBMITTED`no classification accuracy.
- no complete geometry benchmark, classification training curve or final ModelNet40 comparison table.
- `dgcnn_cls.py`, `train_upsampling.py`, part of metrics/visualization/README was empty;belongs to scaffold, not to run experiments.
- EAR/PDANS/PU-Net 12,311, which is visible on HPC, can only be recorded as “existing product” in the no local log. cannot in this record asserts that it was the task of the current working area itself.

---

## 11. HPC Migration, running and resources Job

### 11.1 KITTI upsampling Migration Package

- - Preparation of 10 variant × 3,769 frame, about 42 GiB migration/submission structure, checksum, job script and instructions.
- dry run Real implementation:`tinyx` DNS parsing failed;`tinyx.nhr.fau.de` SSH authentication failed.
- Thus no actually completes the remote transmission and no official remote operation results.

### 11.2 Conda/ Operating Environment

An independent environment has actually been established:`ear`, `openpcdet_centerpoint`, `pointrcnn_old`, `puedgeformer`, `pugcn`, `punet_tf`, `tulip`, `upsampling_basic`. These environments support the recovery and rehabilitation of the generations that TensorFlow/PyTorch/CUDA relies on.

---

## 12. Disk Cleaning and Data Governance Record

### 12.1 2026-05-18 Safe Cleanup

- Delete empty/ smoke directory, PU-GCN evaluation log, PU-Net preparation log and `__pycache__`.
- The actual disk changes about 420 MiB;There's an executive log.

### 12.2 2026-06-26  Phase I

- Cleaning up pip cache about 13 GiB and conda clean about 4.1 GiB;`df`  Show approximately reduction in space used  16 GiB.
- Move 6 PU-GCN mega-centres to `~/TO_DELETE_REVIEW`About 193 GiB;This phase is subject to review and is not tantamount to immediate and permanent deletion.
- Follow-up de-listed about 1,487,400 `.xyz`29,128 `.png`22,573 `.json`22,349 `.bin`21,966 `.csv`7,201 `.md` Middle file path.

### 12.3 2026-07-31 User-Authorized Major Cleanup

- Clean up old strict line trees, CenterPoint reconstructed inputs, 8 Group raw patch outputs, old Line B cap100k,`thesis_demo` venv, cache/VSCode backup, etc.
- Report the release. **459 GiB** (492,379,832,320 bytes).
- Check retention of 8 group `merged_raw`, 8 `final_bin`, checkpoint, source code and current results.

The current file system is about 1.0 TiB, with a total usage of about 993 GiB and the remaining about 31–32 GiB, still in high occupancy status.

---

## 13. Actual Code Changes Record

### 13.1 PointRCNN Main repository

Current branch:`experiment/centerpoint-unified-line-a-b`. There are still local unsubmitted changes to the tracking document:**4 files, 72 insertions, 8 deletions**.

- `lib/config.py`: `yaml.load` was replaced by `yaml.safe_load`.
- `lib/datasets/kitti_dataset.py`: NFS/ and read to add up to 20 retests, bytes and short-read checks.
- `lib/datasets/kitti_rcnn_dataset.py`: repairing the negative sample size of far points exceeding fixed sampling;Rare input security patches.
- `lib/rpn/proposal_layer.py`: far-only / near-only empty bucket guard, avoid TULIP/region-split to make CUDA NMS.
- A large number of untraceed scripts have been added: strict-x4, patch extraction, methods wrapper, E1/E2/E3, CenterPoint, detector-aware V2–V5, region split, training, full assessment, analysis, visualization, packing and recovery scripts.

### 13.2 PU-Net

Current diff:**35 files, 226 insertions, 122 deletions**It's a new one. `.so`/`.o`.

- Python 2  /  3 compatible;TensorFlow API Adjustment.
- sampling/grouping/interpolation/CD/EMD custom op ABI and CUDA compile script restoration.
- CPU fallback with GPU bootstrap.
- Data reading, provider, model utils, normalization and full-frame adapter rehabilitation.

### 13.3 PDANS

Current tracking diff:**2 files, 5 insertions, 4 deletions**.

- `pointnet2/util.py`, `pointnet2_utils.py` The device-aware tensor/CUDA process.
- Add checkpoint and pointops compilations.

### 13.4 SPU-PMD

Current diff:**9 files, 18 insertions, 42 deletions**.

- lazy imports, remove/replace non-dependent path.
- pointnet2 C++/CUDA extended header file compatible with source.
- operations and utility adjustments;Saved model/extension construction product.

### 13.5 OpenPCDet / CenterPoint

Current diff:**3 files, 21 insertions, 7 deletions**, add 1 tool scripts.

- dataset import is optionally dependent on treatment.
- data processor point input/ voxel compatible.
- checkpoint from detector template `weights_only=False` Wait for recovery compatible.
- Add `create_kitti_val_infos_only.py`.

### 13.6 PU-GCN

- `tf_ops/compile.sh`: **14 insertions, 3 deletions**, fix the computer TensorFlow/CUDA op compile path.
- Add PU1K pretrained checkpoint, local backup and strict/full-frame wrapper.

### 13.7 TULIP / PU-EdgeFormer

- TULIP Source Repository Files remain largely unchanged, but add local `scripts/`, `results/` And the cache.
- The official PU-EdgeFormer experiment is accessed mainly through ops-reuse compatible copies;Evidence of the failure of the original warehouse ' s initial custom-op was retained.

### 13.8 version control state risk

PointRCNN Most of the experimental scripts, reports, results, weights, backups and tools in the main warehouse remain **untracked**; Git History is mainly upstream and old submission, cannot only dependent `git log` Revert this job.Current replicability relies on workspace documents, results manifest and reports.09. **151 Source/ CUDA/ Configuration/ split Files**and generate SHA-256 manifest, which reduces this risk.

---

## 14. Analysis, visualization, reporting and dissertation material

The actual generated documents are as follows:

- 2026-08-04: Full Experiment master inventory (Chinese).
- 2026-08-08: upsampling Failed Study Report, in Chinese, English and English in detail.
- 2026-08-09/10 : systematic report English deck.
- 2026-08-13: English progress report PDF/PPT.
- 2026-08-17: 42 page advisor complete PPT/PDF, defense report and talk track.
- 2026-08-18: lost-car detected  /  missed cross-graph evidence.
- 2026-08-28: PU-GCN full retraining presentation.
- 2026-08-26: Chapter 2 expanded, MD/HTML/PDF.
- 2026-09-02: Chapter 3/4 KITTI, MD/HTML/PDF.
- 2026-09-05: Chapter 3/4 ModelNet40+KITTI integrated, MD/PDF.
- 2026-09-08: Chapter 5/6 detailed, MD/PDF, approximately 35 and figure manifest.
- Multiple visualizations have been created for object crop, Open3D, Plotly, BEV, detection frames, frame, lost-car, dual-detector, geometric / voxel distribution, etc.

 Note: 09-08  Previous  thesis chapter  I used it earlier.  256-frame adaptation  Results; They...**The 3,769-frame 20/20 final matrix that has not been automatically updated to 09-10**.

---

## 15. Methodological feasibility survey: investigated, but no model experiment

detection-oriented triage of 2026-08-12 actually checked PUDet, GFAS, PDANet, DAPU, TULIP, etc.:

- PUDet/GFAS: no finds an enforceable public code.
- PDANet: Coded but no available weights.
- DAPU: no publicly runs the code directly and can only be achieved on its own.
- TULIP: has actually run away and is no longer a new candidate.

At the same time, 200 repeater of sampling estimated the uncertainty of the margin:

| frame Number | Standard margin margin | About 95% Minimum detectable difference |
|---:|---:|---:|
| 20 | 6.02 | 11.80 |
| 50 | 3.71 | 7.27 |
| 100 | 2.87 | 5.63 |
| 256 | 1.75 | 3.44 |
| 512 | 1.26 | 2.46 |

The conclusion: 20-frame can only rule out major failures, and cannot reliably proves a small increase.PUDet/GFAS/PDANet/DAPU in this working area**no Official inference/AP**, must not is included in the run-by-run list.

---

## 16. Explicitly failed, not completed or does not establish

1. ModelNet40 full 12,311 × two lines: not completed.
2. PointNet++  branch classification Training  accuracy:  Not submitted, not results.
3. EAR New strict Line A/B full-val: not completed is not acceptable due to CPU running time.
4. SPU-PMD full-val: not completed, only 4-frame smoke.
5. PU-EdgeFormer Recapturing the original environment: custom ops/checkpoint blocking;The follow-up is the result of compatible pathways.
6. HPC KITTI Migration: DNS/SSH authentication Block, no completes remote transmission.
7. dose `g25/g50`: not completed.
8. V5 disjoint holdout64: 64/64 point cloud and CenterPoint baseline/V5 AP have all been completed, but no is now increasing;OpenPCDet Three categories holdout sensitizes sampling and fails to adopt the “Stable detector-wide” judgement.
9. The only empty level result directory currently identified is `pugcn_metrics_batch_filter_range_dedup_voxel_003`; Empty directories are not counted as experiments.The name with the wrong date suffix, the path that does not actually exist, is also not recorded in the record.
10. PUDet, GFAS, PDANet, DAPU: Only a feasibility study has been conducted, no results of a formal model.
11. 3-epoch detector adaptation convergence: no proved.
12. Recent `protocol_and_code_en.md` At the end, keep the old "Line A running" paragraph;should be updated with the same directory 09-10 `current_progress.md` and `full_val_detector_matrix.md` Yes, the latter has been completed by 20/20.

---

## 17. Total line of work by time

| Time | Jobs that have occurred | Status |
|---|---|---|
| 2026-02 to 03 | PointNet/ModelNet/KITTI scaffold, Data Script, Initial Test/ upsampling Project Structure | Changes completed;Most of them don't have a final experiment. |
| 2026-05-03 to 05-08 | PU-Net, EAR, PointRCNN baseline; PU-GCN Environment, smoke, full-frame recovery | A mix of completed/failed, resulting in the first AP and compatibility issues |
| 2026-05-12 to 05-19 | fair comparison, RPN4096, TULIP, PU-EdgeFormer Feasibility, environmental audit | Partially completed;Multiple configurations are not uniform or failed |
| 2026-06-06 to 06-15 | TULIP full/audit, PDANS/SPU-PMD access, HPC migration attempt | TULIP finished;SPU-PMD smoke; Migration failed |
| 2026-06-20 to 06-30 | ratio audit, strict x4 design, visualization, first round of major cleanup | Audit/modification completed |
| 2026-07-03 to 07-18 | Four methods strict Line A/B, PointRCNN full eval, sampler-safe, E1/E2/E3 | Main experiment complete. |
| 2026-07-19 to 07-31 | CenterPoint full, dose, direct no-resample, patch gyn, PU-Net 2×2, surface-c32 | Completed/partially completed;Access to evidence of major causes and consequences |
| 2026-08-04 to 08-11 | surface all-method pilot, region split, detector-aware PDANS V2–V5, Line B c2048 series | 256-frame Development/diagnosis completed;V5/region-split holdout64 also completed but not stabilized across protocol gain |
| 2026-08-12 to 08-18 | New methodology triage, sampling variance, system reporting, lost-car evidence | Analysis/reporting completed |
| 2026-08-24 to 08-28 | finetune64, full 3,712-train PU-GCN detector adaptation, observed-first pilot | Training completed, 256-val pilot completed |
| 2026-09-02 to 09-08 | Chapter of the paper 3–6, charts and list of sources | Document completed, but partially citing old pilot |
| 2026-09-08 to 09-10 | PU-GCN Two Lines full 3,769 Generation, 20-arm Double detector full-val, Total Category AP, convergence Audit, Source Packing | **20/20 Completed** |

---

## 18. Final conclusions to be written in the paper

You can write:

>  Under strict point count control  KITTI Line A/Line B  Under protocol, the generic pre-training point-cloud upsampling model can be generated  exact-4×  Enter, but frozen  PointRCNN  and  CenterPoint  This has not resulted in stabilization testing gains. Local patch, normalization, input sampling, voxel activation and detector training distributions significantly influence the results.The retention of observed points and the introduction of detector adaptation for the distribution of new inputs will significantly reduce the performance gap;However, on 3,769-frame full validation, the current optimal configuration is still below with its pair baseline.Therefore, this experiment supports “point count Add does not imply mission information recovery” instead of “upsampling has increased final detection accuracy”.

cannot writes:

- “All upsampling methods run directly and fairly under the same primary code, the same checkpoint, and the same evaluator” - PU-EdgeFormer, PU-Net, TULIP have compatible paths or protocol.
- “3 epochs already convergence” - no evidence.
- “V5 Stabilizes on stand-alone holdout” - AP has run, but CenterPoint Car has slightly declined, and three types of OpenPCDet result are again sensitive to sampling protocol.
- "ModelNet40 classification Experiment Completed" - no accuracy.
- "PU-GCN trained 3 epochs on KITTI" - 3 epochs is detector adaptation;PU-GCN fixed by PU1K model-100.
- “3N/3M is the best new point for model confidence or geometric” - it is currently only the sample uniformly without replacement that stabilizes seed.

---

## 19. Main Evidence Portal

- 08-04 General list:`/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_EXPERIMENT_MASTER_INVENTORY_20260804_EN.md`
- Update:`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/current_progress.md`
- Latest 20-arm main table:`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/full_val_detector_matrix.md`
- Latest full category AP:`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/all_classes_ap_r40.csv`
- convergence Audit:`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/convergence_audit.md`
- Full protocol:`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/protocol_and_code_en.md`
- 151-file Source Snapshot:`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/experiment_source_full.md`
- 256-frame full-train pilot: `/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_full_retrain_20260824/reports/full_retraining_report.md`
- finetune64: `/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pointrcnn_finetune64_six_arms_20260824/reports/finetune64_screen_report.md`
- direct no-resample: `/home/ra87racy/projects/baseline_detectors/PointRCNN_direct4n_noresample/experiments/direct4n_noresample/runs/full_val_frozen_direct4n_v1/direct4n_comparison.md`
- region-split all methods: `/home/ra87racy/projects/baseline_detectors/PointRCNN/results/openpcdet_pointrcnn_three_class_region_split_exact4n_all_methods_pilot256_20260808/EXACT4N_REGION_SPLIT_ALL_METHODS_RESULT_ZH.md`
- detector-aware V2/V3/V4/V5: `/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detector_aware_pdans_v2_20260805/reports/v2_decision_report.md`, `.../detector_aware_pdans_v3_20260805/reports/v3_decision_report.md`, `.../detector_aware_pdans_v4_20260805/reports/v4_decision_report.md`, `.../detector_aware_pdans_v5_20260805/reports/v5_pilot256_decision_report.md`
- V5 holdout64: `/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detector_aware_pdans_v5_holdout64_20260805/`
- PointRCNN Three categories holdout and sampling Sensitivity:`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/openpcdet_pointrcnn_three_class_holdout64_20260808/POINT_RCNN_THREE_CLASS_RUNTIME_AND_RESULT_ZH.md`
- 16-seed sampling Noise:`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pointrcnn_sampling_variance_probe_20260810/SIGMA_16SEED_VERDICT_ZH.md`
- PU-Net Cascade ablation:`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731`
- ModelNet40 protocol State:`/home/ra87racy/projects/modelnet40_pointnet2_upsampling/reports/modelnet40_x4_final_protocol_report.md`
- HPC migration log:`/home/ra87racy/projects/kitti_upsampling_lab_to_hpc_migration/reports/`
- Task log 05-04:`/home/ra87racy/projects/baseline_detectors/PointRCNN/WORKLOG_2026-05-04.md`
- Disk cleanup:`/home/ra87racy/projects/disk_cleanup_execution_log.md`, `/home/ra87racy/disk_cleanup_stage1_report.md`, `/home/ra87racy/disk_cleanup_20260731_456g_manifest.md`
- Delete the list of documents:`/home/ra87racy/deleted_pugcn_intermediate_manifest.txt`, `/home/ra87racy/deleted_pugcn_filetype_counts.txt`

---

## 20. Current finish state

- The latest full-val core experiment:**Completed**.
- Two PU-GCN validation inputs:**3,769/3,769 + 3,769/3,769 completed**.
- Test assessment:**20/20 PASS**.
- Current training/drill process:**None**.
- Final conclusions:**adaptation Significant recovery, but no configuration exceeding pair baseline**.
- Material that still needs to be manually synchronized: 256-frame adaptation tables in previous sections of 09-08 and in the report should be replaced by 09-10 full-val tables;This HotSync is currently no complete, so it is not written as updated.
