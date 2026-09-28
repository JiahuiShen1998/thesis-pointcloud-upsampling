# point-cloud upsampling — downstream Survey of Testing Experiments (paper writing version)

Original master record time: 2026-08-04
Latest update: 2026-09-13 (see section 18—25)
Main item: KITTI point-cloud upsampling × PointRCNN / CenterPoint downstream 3D Testing
Use: Harmonization of test calibres, results, work history, code changes, basis of papers and conclusions

> **Version description:**  I don't think so.  0—17  Retain the section  2026-08-04  A complete historical snapshot of the hour to track not completed  surface-c32, PDANS  and  detector-adaptation  Status; Section 18—25 was added to the follow-up experiment of 2026-08 to 2026-09 as “the state of the art at present”.If the old state is not consistent with the updated state, section 18—25 and its direct outcome document will prevail.

---

## 0. First read this page: It can now be written in the core conclusions of the paper

### 0.1 Research issues

The real study of this project is not “Can cannot transform point count mechanically into 4”, but:

> Under strict control of the output point count, the retention of sites and detector unchanged, can point-cloud upsampling be used for recovery valid surface evidence of thin LiDAR and improve downstream 3D detection?

The answer given by the current evidence is:

1. Strict 4× point count does not imply effective information increased.
   point count meets 4N only as a counting constraint;The test is whether the new point is close to the real surface, whether the useful space is covered, whether it avoids duplication and false voxel.

2. Under the current unified full volume protocol, all four methods are cannot recovery original point cloud for PointRCNN detection.
   PDANS of Line A is the most robust, but Car Moderate 3D AP_R40 is still down from 82.2554 of the original point cloud to 67.2235.All methods in Line B are also below downsampling baseline.

3. The overall sequencing of the method is consistent on two detectors:
   PDANS > PU-GCN > PU-EdgeFormer > Old PU-Net pipe.
   But the old PU-Net results contain a clear normalization / counter-normalization conformity defects and cannot is used as a fair ceiling for PU-Net methodology itself.

4. downstream detector has a limited input budget and magnifies the impact of the error point.
   PointRCNN fixed extract 16384 points;CenterPoint/ voxel maximum 5 points and test maximum 40000 voxel.Repetition points, pseudo voxel or abnormal spatial distributions will compete with the budget for real observations.

5. The old patch extractor is one of the main engineering root causes.
   It divides the 2048 point on a continuous basis after using the coarse space compartment, and does not guarantee a real local surface patch.XY medium of the contours of Line B patch is about 30.46 m and p90 is about 124.10 m, which is a serious violation of the training assumptions of the generic patch upsampling network.

6. Nor is “localization” sufficient in itself.
   The first version of the local cover-kNN patch, while significantly improving the locality of geometric, fills 2048 points by refilling them;This has led to a sharp drop in voxel coverage of PointRCNN, while CenterPoint has significantly improved because voxel will absorb duplication.Therefore, local, sole support and detector budgets must be controlled at the same time.

7. When 2026-08-04, surface-c32 was still a geometric candidate;Follow-up PU-GCN has advanced to the full detector adaptation matrix.
   The full results of the multi-mode surface-c32 unified double detector have not yet been reached;However, PU-GCN ' s subsequent use of the improved local patch protocol has completed the full certification of 3712 frame detector adaptation and 3769 frame and 20/20 arms PointRCNN/CenterPoint.For the most recent results see section 18—20.

### 0.2 The most suitable main experiment

Suggested paper main experiment reads:

1. Establishment of a fair protocol strict exact-4N, double input line, frozen detector;
2. Prove "point count recovery" and "does not imply" "Detect recovery”;
3. Failed to explain using point/ voxel budget, patch locality, duplicate filling and generated points surface consistency;
4. Identification of causal factors and consequences through E3, prioritization of sites, dosage experiments, matching of real point controls and patch/ normalization 2×2 ablation;
5. The conditions for upsampling for downstream testing are: local, sole, superficial, budgetary and mission-specific.

---

## 1. Scope and Experiment branch

| branch | Objective | Current Status | Could it be the main result of the paper? |
|---|---|---:|---:|
| KITTI × PointRCNN | Check the effects of exact-4N upsampling on Car 3D tests | Full master test complete. | Yes. |
| KITTI × CenterPoint | Verification of whether conclusions can be generalized using different testing mechanisms | Full master test complete. | Yes. |
| Scale/dosage experiments | Separate a small number of generated points Gains to High Ratio generated points Harm | 256 frame fine grains completed;Full g10 completed | Yes, scale to be indicated |
| patch/PU-Net Causal ablation | Position patch non-local, duplicate fill and normalization defects | 256 frame 2×2 Completed | Yes, as a mechanism experiment |
| surface-c32 New candidate | Construct a local, single-support, zero-duplicate patch | geometric and partially 256 have been produced;Test Not Run | It can only be written as ongoing. |
| ModelNet40 × PointNet++ | Check 4× upsampling for classification | only smoke;HPC Formal Experiment Not Submitted | cannot Write Results |
| TULIP / SPU-PMD / EAR | Methodological feasibility, early engineering exploration | Non-uniform protocol or not entered into the final method set | Only background / |

---

## 2. Official Experiment protocol

### 2.1 Data Sets and Divisions

- Data set: KITTI 3D Object Detection.
- validation set: 3769 frame.
- The official comparison is based on the offline KITTI AP_R40.
- PointRCNN is officially the main mission of Car;CenterPoint also reports Car, Pedestrian, Cyclist.
- AP type: BBox, BEV, 3D;Difficulties: Easy, Moderate, Hard.
- IoU threshold: Car is 0.7;Pedestrian/Cyclist is 0.5.

KITTI  For the original paper and reference references, see  11  Section.

### 2.2 Two Input Lines

#### Line A: Original point cloud Decryption

- Input: Original scan N point.
- Output: Keep N sites and generate 3N points.
- Finally point count: Strict 4N.
- Question: Whether the addition of a large number of generated points will assist in detection when the original sensor evidence is complete.

#### Line B: Rare point cloud recovery

- Input: Determination 4× downsampling, M = floor(N/4).
- Output: Keep M sites and generate 3M points.
- Finally point count: Strict 4M, roughly equivalent to the original scan point count.
- Research question: Whether the learning upsampling can input low resolution into recovery to the original sensor level.

### 2.3 Formal Method

1. PDANS;
2. PU-GCN;
3. PU-EdgeFormer;
4. PU-Net.

Significant qualifications:

- The old PU-Net of the official main table uses the adaptor that was subsequently identified as defective: no, normalization, a training-time unit, and no, a correct reversal.It should therefore be marked "PU-Net" (old adapter/deficient pipe).
- TULIP, SPU-PMD and EAR no entered the final exact-4N method to harmonize main table.

### 2.4 Third Level Test Input

#### E1: Original exact-4N input

Strictly inspects the composition of each frame point count, limited values and observations/generations, not as a special detector sub-coded sampling.

#### E2: PointRCNN Normalize 16384 input

This is PointRCNN official result:

1. Strict FOV/ distance filter;
2. 0.1 m voxel points;
3. Depth range is as [0,20), [20,40), [40,60), [60,70.4] m;
4. (b) The allocation of a quota of largest-remainder according to the number of candidates;
5. No label, no AP selection;
6. Output fixed is 16384 point.

#### E3: 32768 Sensitivity Experiment for Point Priority

- (a) Retain the real observations and fill them in with generated points to 32768;
- (a) The 16384 point ceiling used to determine whether E2 is the main source of loss;
- E3 is a sensitivity experiment and should not be written into a new master configuration.

protocol Documentation:

- [E1/E2/E3 Root Report](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_e3_final_root_cause_report.md)
- [exact-4N protocol.json](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/protocol.json)
- [E1/E2 AP Summary](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_live_ap_summary.csv)
- [E3 AP Summary](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e3_real_first_32768_live_ap_summary.csv)

### 2.5 PointRCNN Settings

- Configure:[tools/cfgs/default.yaml](/home/ra87racy/projects/baseline_detectors/PointRCNN/tools/cfgs/default.yaml)
- Weight:[tools/PointRCNN.pth](/home/ra87racy/projects/baseline_detectors/PointRCNN/tools/PointRCNN.pth)
- Weight frozen, replaces input only and does not retrain.
- RPN Enter point count: 16384.
- PointNet++ MSG radius:
  - [0.1, 0.5];
  - [0.5, 1.0];
  - [1.0, 2.0];
  - [2.0, 4.0].
- Per scale neighbourhood sampling: 16/32.
- RPN and RCNN score threshold: 0.3.
- RCNN NMS threshold: 0.1.
- rect camera Coordinate Range:
  - x: [-40, 40];
  - y: [-1, 3];
  - z: [0, 70.4].
- No reflection intensity is used as a network input feature.

### 2.6 CenterPoint Settings

- Configure:[centerpoint.yaml](/home/ra87racy/projects/baseline_detectors/PointRCNN/external/OpenPCDet/tools/cfgs/kitti_models/centerpoint.yaml)
- Data configuration:[kitti_dataset.yaml](/home/ra87racy/projects/baseline_detectors/PointRCNN/external/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml)
- Weight:[checkpoint_epoch_80.pth](/home/ra87racy/projects/baseline_detectors/PointRCNN/external/centerpoint_hpc_bundle_20260717/outputs/checkpoints/CenterPoint_KITTI_original_80epoch/checkpoint_epoch_80.pth)
- Trained 80 epoch on original KITTI;All inputs reuse the same frozen weights.
- Structure: MeanVFE — VoxelResBackBone8x — HeightCompression — BaseBEVBackbone — CenterHead.
- Category: Car, Pedestrian, Cyclist.
- point cloud Range:[0,-40,-3,70.4,40,1].
- voxel Size: 0.05 × 0.05 × 0.1 m.
- Every voxel tops 5 points.
- Test maximum 40000 voxel.
- Test not shuffle.

---

## 3. Official Main Results

### 3.1 PointRCNN: Full 3769 frame E2 AP_R40

The following table presents the most important main findings of the paper.Value order is Easy / Moderate / Hard.

| Enter | BBox AP_R40 | BEV AP_R40 | 3D AP_R40 | Moderate 3D relative baseline |
|---|---:|---:|---:|---:|
| original baseline | 99.3170 / 94.0842 / 90.1018 | 95.9809 / 88.9807 / 86.5942 | 92.2731 / 82.2554 / 77.9454 | — |
| downsampling baseline | 95.2406 / 79.9487 / 75.5846 | 91.0369 / 76.3979 / 72.0313 | 85.1766 / 65.7460 / 61.3818 | — |
| Line A PDANS | 93.3196 / 81.7462 / 76.8537 | 90.0035 / 78.0160 / 73.2078 | 84.6118 / 67.2235 / 62.3981 | -15.0319 |
| Line A PU-GCN | 92.4442 / 71.7663 / 67.0732 | 86.9259 / 66.7872 / 62.0159 | 80.1727 / 58.2759 / 53.4719 | -23.9796 |
| Line A PU-EdgeFormer | 81.9530 / 58.7956 / 52.2988 | 75.9674 / 53.8311 / 48.9408 | 65.6774 / 44.9596 / 40.1919 | -37.2958 |
| Line A PU-Net (old defective pipe) | 38.4606 / 23.8945 / 20.6926 | 28.9543 / 18.7383 / 16.8468 | 11.7639 / 8.3802 / 7.4407 | -73.8753 |
| Line B PDANS | 83.7454 / 60.3295 / 53.7948 | 76.6291 / 54.5678 / 48.1389 | 65.0123 / 44.0752 / 39.1729 | -21.6708 |
| Line B PU-GCN | 60.6102 / 38.6101 / 32.3995 | 57.2512 / 36.1932 / 31.4262 | 45.9303 / 28.6803 / 24.1310 | -37.0657 |
| Line B PU-EdgeFormer | 52.7497 / 33.4898 / 28.7087 | 43.6454 / 27.7381 / 23.3813 | 33.7635 / 20.5693 / 17.6153 | -45.1767 |
| Line B PU-Net (old defective pipe) | 33.6163 / 20.5057 / 18.4031 | 23.7717 / 15.5219 / 13.6207 | 12.4097 / 8.7807 / 7.7637 | -56.9653 |

It can be concluded that:

- All Line A methods are significant below original point cloud;Add 3N generated points will hurt frozen detector.
- All Line B methodologies are below downsampling baselines;This indicates that generated points does not replace only no with lost evidence and interferes with the already rare observations.
- PDANS is the best on both lines, but "relatively best" does not imply "recovery Success".
- The rankings are broadly consistent between the two lines, supporting a stable relationship between geometric quality and the detection of performance.

### 3.2 PointRCNN: E3 32768 Observatory Priority Sensitivity Experiment

| Enter | 3D AP_R40 Easy / Moderate / Hard | Moderate: E3 - E2 |
|---|---:|---:|
| original baseline | 92.03 / 80.38 / 77.52 | -1.88 |
| downsampling baseline | 84.59 / 64.14 / 57.50 | -1.61 |
| Line A PDANS | 86.27 / 70.06 / 65.27 | +2.84 |
| Line A PU-GCN | 85.19 / 64.95 / 58.66 | +6.68 |
| Line A PU-EdgeFormer | 71.54 / 51.63 / 45.65 | +6.67 |
| Line A PU-Net (old defective pipe) | 27.92 / 18.93 / 17.88 | +10.55 |
| Line B PDANS | 64.92 / 44.04 / 39.16 | -0.04 |
| Line B PU-GCN | 45.09 / 28.28 / 23.83 | -0.40 |
| Line B PU-EdgeFormer | 32.50 / 20.03 / 16.20 | -0.54 |
| Line B PU-Net (old defective pipe) | 10.99 / 7.48 / 7.19 | -1.30 |

Explanation:

- Line A, after expanding the budget and giving priority to the real point, has a recovery showing that the 16384-point competition did cause losses.
- But even if E3, all methods are still far from below original baseline, so the input ceiling is not the only or dominant root cause.
- Line B has hardly improved, suggesting that its failure is mainly due to the generation of geometric, not 16384 cap.

### 3.3 CenterPoint: Full master result (Moderate 3D AP_R40)

| Enter | Car | Pedestrian | Cyclist |
|---|---:|---:|---:|
| original baseline | 79.2773 | 50.6533 | 64.6054 |
| Line A PDANS | 64.3574 | 45.1910 | 45.8953 |
| Line A PU-GCN | 58.8955 | 44.1470 | 44.3082 |
| Line A PU-EdgeFormer | 45.7342 | 36.8205 | 35.2235 |
| Line A PU-Net (old defective pipe) | 10.7512 | 19.4930 | 13.1928 |
| downsampling baseline | 64.5979 | 24.5690 | 15.4577 |
| Line B PDANS | 46.7512 | 28.1790 | 15.5186 |
| Line B PU-GCN | 39.0714 | 24.9302 | 16.8458 |
| Line B PU-EdgeFormer | 24.7476 | 15.5896 | 7.0374 |
| Line B PU-Net (old defective pipe) | 11.7178 | 11.9174 | 5.0172 |

Explanation:

- Car reproduces PointRCNN ' s general ranking and failure trends, suggesting that the conclusion is not a single detector.
- Line B  It's...  PDANS/PU-GCN  Yeah.  Pedestrian/Cyclist  There's very small recovery, but  Car  Visible decline; cannot thus claims the overall recovery success.
- Cross-category differences suggest that small targets may benefit from a small number of local fillings, but a large number of unreliable points still undermine the main mission.

Full CenterPoint report:

- [Observed-first Full Report](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/centerpoint_exact4n_reconstructed_order_safe_20260729/CENTERPOINT_OBSERVED_FIRST_FULL_REPORT.md)
- [CenterPoint All AP Table](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/centerpoint_exact4n_reconstructed_order_safe_20260729/full_ap_summary.csv)

---

## 4. Roots and institutional evidence

### 4.1 Old patch Ripper is not a local surface patch

The core of the old process is:

1. (a) Coarse space breaker for scenes;
2. Split 2048 points sequentially in the results of the boxes;
3.  Use these points as inputs to the generic object level upsampling network  patch.

The problem is that 2048 lines does not imply are compact in space, on the same surface or with the same object.Measured:

- Line B patch XY median value of approximately 30.46 m;
- p90 About 124.10 m.

As a result, a patch may be mixed into ground, vehicles, buildings, vegetation and long-range structures.The pre-trained network on object surface patch generates this mixed input as a local shape, creating cross-surface plugs and no support points.

Main causes report:

- [It's a dissertation-cause report.](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/dual_detector_three_frame_root_cause_20260730/THESIS_ROOT_CAUSE_REPORT_EN.md)
- [Double detector Full AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/dual_detector_three_frame_root_cause_20260730/analysis/all_detector_ap_r40_bbox_bev_3d.csv)
- [Portable analysis package description](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/current_task_visualization_analysis_portable_20260731/README_ZH.md)

### 4.2 is really effective by adding "Purpose evidence" instead of "point count"

The geometric analysis shows that while the current methodology generates a large number of points, the new real support voxel is limited, and the additional voxel that does not exist in point cloud is produced with a large number of references.The mechanism could be summarized as follows:

> A small and right amount of surface recovery + has a large amount of redundancy/misplacement voxel + detector budget competition – AP has fallen.

The following categories of indicators are highlighted in the proposed paper:

- generated-to-real's nearest neighboring distance;
- reference-to-output recall;
- generated voxel precision;
- extra voxel fraction;
- occupied voxel;
- Local PCA flat distance;
- patch XY diameter and object purity;
- Repeat density per point/ per voxel.

### 4.3 PointRCNN and CenterPoint, why are they responding differently?

#### PointRCNN

- PointNet++ local aggregation directly on the dot set;
- fixed 16384 input slot;
- (a) High-level repetition or clustering points would take over the original point level sampling budget;
- The close repeater does not necessarily add neighbourhood information, but instead reduces the independent space position covered.

#### CenterPoint

- Quantify point cloud as voxel;
- Repeated points in the same voxel will be aggregated by MeanVFE;
- The hazard of repeat points can be partially absorbed by voxel;
- But a lot of new voxel will trigger 40000 voxel caps and compete with the real voxel.

This explains why the same version of the local patch may improve CenterPoint, but significantly harm PointRCNN.

### 4.4 voxel Budget Evidence

 Yes.  Line A  It's...  32  frame Statistics, enter  CenterPoint  Test  cap  Previous voxel medians  40000 cap  The result was as follows:

| Enter | voxel median | Over/hit 40000 cap |
|---|---:|---:|
| Original point cloud | 14944 | 0 / 32 |
| PDANS | 36913 | 6 / 32 |
| PU-GCN | 45108 | 29 / 32 |
| PU-EdgeFormer | 46182 | 29 / 32 |
| PU-Net (old defective pipe) | 55384 | 32 / 32 |

Line B failed 40000 cap, so:

- voxel of Line A is a clearly defined amplifier;
- Line B failed cannot, attributed to CenterPoint cap, still pointing to the generation of geometric quality.

### 4.5 Observatory Prioritization Experiment

The experiment maintains the same multiple assembly for each frame point, with only N sites placed before 3N generated points.All 8 methods/input combinations x 3769 frame have been audited together.

Average change of Moderate 3D AP_R40 relative to original reordering order:

- Line A PDANS: +0.0869;
- Line A PU-GCN: +0.8939;
- Line A PU-EdgeFormer: +0.7919;
- Line A PU-Net (old defective pipe line): + 5.8776;
- Line B: only -0.0018 to + 0.0311.

Conclusions:

- Order and budget competition are real;
- But with the exception of the extreme failure of PU-Net, the benefits are small;
- It's a secondary mechanism, not enough to explain the main AP reduction.

### 4.6 GT Transfer Analysis

 Baseline relative to corresponding baseline  TP  ♪ Turn into  FN  It's...  Car  Proportion:

| detector / input line | PDANS | PU-GCN | PU-EdgeFormer | PU-Net Old pipe |
|---|---:|---:|---:|---:|
| PointRCNN Line A | 23.1% | 32.7% | 47.9% | 80.9% |
| PointRCNN Line B | 35.4% | 55.4% | 67.0% | 76.5% |
| CenterPoint Line A | 20.6% | 25.9% | 39.3% | 70.8% |
| CenterPoint Line B | 27.6% | 37.4% | 55.3% | 67.0% |

On behalf of frame:

- 000590;
- 005625;
- 006682.

The 3 frame × 2 lines x 4 method x 2 detector = 48 cases have been completed.There is a a small number of recovery case, but the loss is significantly greater.GT  Transfer and visualization should be treated as  AP  cannot instead of formal  AP.

---

## 5. dosage, control experiments and ablation

### 5.1 generated points Scale Experiment

Reports:

- [Joint dosage analysis](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_x4_detector_recovery_combined_analysis_20260725/combined_analysis_report.md)
- [Full BEV/3D control table](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_x4_detector_recovery_combined_analysis_20260725/complete_bev_3d_comparison.md)

#### 256 frame fine particle replacement ratio

Ratio: g2.5, g5, g7.5, g10;All 32/32 combinations are audited.

Line A Car Moderate 3D AP_R40:

| Methodology | g2.5 | g5 | g7.5 | g10 | Optimal relative original baseline 82.66 |
|---|---:|---:|---:|---:|---:|
| PDANS | 85.10 | 82.37 | 82.25 | 82.42 | +2.43 |
| PU-GCN | 82.69 | 82.17 | 81.88 | 79.65 | +0.03 |
| PU-EdgeFormer | 81.97 | 82.08 | 79.39 | 77.04 | -0.58 |
| PU-Net Old pipe | 79.10 | 77.82 | 75.64 | 71.67 | -3.56 |

Line B Car Moderate 3D AP_R40:

| Methodology | g2.5 | g5 | g7.5 | g10 | downsampling baseline |
|---|---:|---:|---:|---:|---:|
| PDANS | 66.40 | 63.26 | 61.34 | 60.86 | 68.29 |
| PU-GCN | 65.54 | 61.82 | 60.87 | 56.16 | 68.29 |
| PU-EdgeFormer | 63.23 | 60.32 | 53.96 | 53.20 | 68.29 |
| PU-Net Old pipe | 62.60 | 57.35 | 51.43 | 47.59 | 68.29 |

Matches the real point control:

- Original point cloud real-control c2.5 = 84.42;
- PDANS g2.5 = 85.10;
- Thus PDANS is relatively the same point count true filling control is only about 0.68 AP;
- And the result is from 256 frame, which must be fully confirmed, and cannot is written directly as a stable increase.

#### 256 frame Crude ratio

g10, g15, g25, g35, g40, g50 were adopted by a combination of 48/48.The overall trend is that the higher the generated points rate, the worse the detection performance.

#### Full 3769 frame g10

Car Moderate 3D AP_R40:

| Input Line | PDANS | PU-GCN | PU-EdgeFormer | PU-Net Old pipe |
|---|---:|---:|---:|---:|
| Line A | 79.951 | 79.478 | 76.433 | 72.371 |
| Line B | 59.048 | 55.954 | 52.415 | 45.700 |

The full amount of g25/g50 is not completed.

It can be concluded that:

- generated points has significant dose effects for detection;
- a small number of High-quality generated points may help or maintain performance at Line A;
- When generated points increases, redundancy, misalignment and budget competition accumulate, and performance decreases rapidly;
- Line B is more difficult because thin input itself lacks local evidence for reliable network reconstruction.

### 5.2 sampler-safe fix

 Old  PointRCNN  sampling Logic  far points ≥ 16384  When negative sampling is generated, and  000378  Wait till frame crashes.

Restoration:

-  ♪ When the distance has reached ♪  16384  , from all candidates without replacement  16384  Points;
- Allows with replacement sampling when it is necessary to fill points but the target is filled with point count more than the available point count;
- Record fallback triggers.

Results:

- All previously failed Line A combinations complete 3769/3769;
- fallback per method only about 3–12/3769 frame less than 0.4%;
- So the bug explains "why is running down," not "why is AP falling."

Reports:

- [sampler-safe Final report](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_pointrcnn_eval_line_a_sampler_safe_v2_20260717_235037/reports/final_line_a_sampler_safe_v2_report.md)

### 5.3 First Edition Local patch: geometric Improved, but repeated filling to harm PointRCNN

Candidate: fps_ball_cover_knn_v3.

Settings:

- Line A Main radius 2 m;
- cover radius 6 m;
- cover_min = 32;
- The only kNN floor = 256;
- patch overlap ratio = 3;
- Uncertain duplicate fills less than 2048 point.

geometric 20 frame:

- patch XY p50/p90: the old 9.439/84.097 m new 5.334/16.443 m;
- Source coverage is close to 1.

But frame 000001:

- 277 patch;
- Coverage 99.9867%;
- Average of 4.72 patch per point of origin;
- The median of patch for the sole source point only 464/2048;
- p10 is 256/2048;
- Repeat the protocol 59.4%.

256 frame Test changes:

| detector | Indicators | Old patch | Local Repeat patch | Change |
|---|---|---:|---:|---:|
| PointRCNN | Car Moderate BEV | 70.3912 | 51.6618 | -18.7293 |
| PointRCNN | Car Moderate 3D | 57.1829 | 43.6976 | -13.4853 |
| CenterPoint | Car Moderate BEV | 70.6567 | 86.4220 | +15.7653 |
| CenterPoint | Car Moderate 3D | 61.1172 | 77.4150 | +16.2978 |
| CenterPoint | Pedestrian Moderate 3D | — | — | +1.3431 |
| CenterPoint | Cyclist Moderate 3D | — | — | +23.4861 |

PointRCNN Institutional Indicators:

- (a) The median number of occupied voxels: 11584.5  /  2124.5, down 81.7%;
- Point share of the most secret 10% voxel: 27.23%  /  87.85%.

Conclusions:

- patch does not automatically bring downstream benefits;
- Repeating the number will allow the limited slot of point detector to be occupied by the nearest repeat point.
- The voxelization of CenterPoint absorbs part of the duplication and thus shows the opposite direction;
- That candidature should not be extended to the full.

Reports:

- [PDANS Double detector Causes of disagreement](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/analysis/pdans_old_vs_cover_knn_v3/PDANS_DETECTOR_DISAGREEMENT_ROOT_CAUSE.md)
- [patch/PU-Net Causal protocol](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/PROTOCOL.md)

### 2×2 ablation from 5.4 PU-Net normalization × patch

Four conditions:

- A: Error normalization + old patch;
- B: unit ball normalization / reverse transformation + old patch;
- C: error normalization + local repetition patch;
- D: unit ball normalization / reverse transformation + local repetition patch.

256 frame Car Moderate 3D AP_R40:

| detector | Baseline | A | B | C | D | D - A | D - baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| PointRCNN | 81.3033 | 7.9823 | 18.2742 | 6.3008 | 43.5424 | +35.5602 | -37.7609 |
| CenterPoint | 79.8368 | 15.9788 | 41.2227 | 16.4119 | 46.2047 | +30.2259 | -33.6321 |

Conclusions:

- The correct normalization / reverse transformation is the primary repair of PU-Net;
- Correct normalization and local patch are active interactive;
- Even if D is large outperforms A, it is still far from the below test baseline;
- The main result of the old PU-Net should be interpreted as evidence of a pipe failure rather than the performance of the method itself;
- D is also not worth extending directly to 3769 frame, and the problem of duplication and source support needs to be addressed first.

Reports:

- [PU-Net 2×2 Final decision-making](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/punet256_2x2/reports/FINAL_DECISION_CN.md)
- [PU-Net 2×2 Double detector Effects](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/punet256_2x2/reports/PUNET_2X2_DETECTOR_EFFECTS.md)

### 5.5 Newest surface-c32: only support, zero repetition

Selected candidate: surface_cover_exact_pr1_c32.

Settings:

- patch Size: 2048;
- Each patch must have 2048 as its sole source;
- Zero duplicate fill;
- primary patch_num_ratio = 1;
- Main radius 2 m;
- cover radius 6 m;
- cover_min = 32.

20 frame patch Indicators:

| Indicators | Result |
|---|---:|
| patch Total | 2025 |
| Coverage p50 | 0.999975 |
| Minimal coverage | 0.999638 |
| membership mean p50 | 1.739 |
| unique source p10 | 2048 |
| Repetitive ratio | 0 |
| XY diameter p50 | 5.332 m |
| XY diameter p90 | 24.180 m |
| Medium value of multiple objects patch | 0.00476 |

20 frame Method Output generated-to-real Recent median of p90:

| Methodology | Old patch | surface-c32 |
|---|---:|---:|
| PU-GCN | 0.1557 | 0.1127 |
| PU-EdgeFormer | 0.2236 | 0.0991 |
| PDANS | 0.1399 | 0.0974 |
| PU-Net (rehabilitation normalization) | 0.7920 | 0.2759 |

The 1 mm of all methods approximates the single rate of 99.97%–100%, and there is no longer a repeat collapse of the first version of the local patch.

Current status (2026-08-04):

| Methodology | 256 frame Generation | exact-4N Check | PointRCNN AP | CenterPoint AP |
|---|---:|---:|---:|---:|
| PU-GCN | Completed | Pass. | Not Run | Not Run |
| PU-EdgeFormer | Completed | Pass. | Not Run | Not Run |
| PU-Net Restoration | Completed | Pass. | Not Run | Not Run |
| PDANS | Interrupted: 231/256 merged raw;232 logs | No final bin | Not Run | Not Run |

The PDANS status file still shows RUNNING, but the current no matching process/ tmux;The final frame 006812 log stops in DDIM.So it's old, not still running.

geometric report:

- [surface Search summary](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/geometry20_surface_search_v1/reports/summary.json)
- [Method Output summary](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/geometry20_surface_search_v1/reports/method_outputs/summary.json)
- [256 Generation State](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/surface_candidate256_v1/sequence_state.json)

---

## 6. History Experiment: Keep a working record but do not mix into the official main table

### 6.1 Early PointRCNN / EAR / PU-Net

Historical report:

- [WORKLOG_2026-05-04](/home/ra87racy/projects/baseline_detectors/PointRCNN/WORKLOG_2026-05-04.md)
- [PointRCNN baseline report](/home/ra87racy/projects/baseline_detectors/PointRCNN/POINTRCNN_BASELINE_REPORT.md)
- [PointRCNN EAR report](/home/ra87racy/projects/baseline_detectors/PointRCNN/POINTRCNN_EAR_REPORT.md)
- [PU-Net integration report](/home/ra87racy/projects/baseline_detectors/PointRCNN/PU_NET_INTEGRATION_REPORT.md)
- [KITTI upsampling comparison](/home/ra87racy/projects/baseline_detectors/PointRCNN/KITTI_UPSAMPLING_COMPARISON.md)

Early representation results:

- baseline 3D AP about 89.19 / 78.85 / 77.91;
- EAR about 88.56 / 77.96 / 76.78;
- Early PU-Net about 0.20 / 1.14 / 1.14.

These experiments exist in x2, whole-field input, old adapter, old sampling or other non-uniform settings, which can only be used as a record of project exploration.

### 6.2 2026-05-16 Weekly Report

[weekly_progress_report](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/weekly_report_20260516_full_comparison/weekly_progress_report.md)

Of which Moderate 3D AP:

- original baseline: 77.93;
- 50% downsample: 76.99;
- EAR: 73.89;
- PU-Net: 35.26;
- PU-GCN: 50.65.

However, PU-GCN Line B uses RPN 26000 and safe sampling, which are incompatible with the subsequent unified E2 16384.cannot is directly compared with 3 main table.

### 6.3 Actual Multiplication Audit of Old Output

[upsampling_ratio_audit](/home/ra87racy/reports/upsampling_ratio_audit.md)

Found:

- EAR Practically about 1.008×;
- The old PU-Net ratio is not consistent;
- Some of the PU-GCN, PDANS outputs were cut off by 100k cap;
- TULIP is range-image vertical 4×, but point count may be less than input after projection xyz.

This directly contributes to the follow-up strict exact-4N protocol.It can be written as “protocol Design Motivation”, but not as a fair way to compare these old results.

### 6.4 TULIP, SPU-PMD, PU-EdgeFormer, PDANS

- [TULIP/PDANS/SPU-PMD feasibility](/home/ra87racy/projects/experiments/method_feasibility_check/TULIP_PDANS_SPUPMD_FEASIBILITY_REPORT.md)
- [TULIP/PDANS/SPU-PMD reproduction](/home/ra87racy/projects/experiments/method_repro_check/TULIP_PDANS_SPUPMD_REPRO_REPORT.md)
- [PU-EdgeFormer point-cloud-only report](/home/ra87racy/projects/experiments/puef_pointcloud_only/PU_EDGEFORMER_POINTCLOUD_ONLY_REPORT.md)

Historical process:

- TULIP has been given priority as a candidate for the original LiDAR range-image method;
- PUGAN/PU1K checkpoint of PDANS was later confirmed to be available, approximately 1.7 GB per person;
- SPU-PMD has completed feasibility/smoke but has not entered the formal form of the final four methods;
- PU-EdgeFormer was initially blocked by checkpoint and CUDA10, which became fully operational by reusing PU-GCN ops/ pre-training environment.

EAR is a self-defined EAR-style engineering method, a local no retroactivity source.It is not to be disguised in the paper as an existing formal literature method.

---

## 7. ModelNet40 / PointNet++ branch

Main report:

- [ModelNet40 x4 final protocol](/home/ra87racy/projects/modelnet40_pointnet2_upsampling/reports/modelnet40_x4_final_protocol_report.md)

_Other Organiser

- PU-GCN local capacity is ready;
- Line A: 1024 → 4096;
- Line B: 256 → 1024;
-  Each line is split  8  individual  smoke  Samples passed;
- point count and NaN/Inf checked through.

not completed:

- The full amount of 12311 samples/lines generated no for submission to HPC;
- PU-EdgeFormer checkpoint is a not completed branch connection;
- PointNet++ Line B  It's...  5  individual  smoke  Submitted by branch no;
- no classification Accuracy Rate, geometric quality Indicators or complete training results.

Reasons for obstruction:

- Current agent host no HPC mounted;
- no for SLURM environment;
- The required job file is not available for the context of the current host.

Thesis boundary:

> ModelNet40 can only be written as “protocol and smoke” and cannot as already available classification.

---

## 8. Code and pipe modifications performed

This section distinguishes between " current work tree verifiable changes " and " historical report record changes " .

### 8.1 PointRCNN Current working tree verifiable changes

The current tracked diff is about 50 insertions / 9 deletions.

1. [lib/config.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/lib/config.py)
   yaml.load is changed to yaml.safe_load, which corresponds to a new PyYAML and reduces the risk of unsafe back-serialization.

2. [lib/datasets/kitti_dataset.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/lib/datasets/kitti_dataset.py)
   Add NFS
   - Up to 20 retries;
   - Check bytes and short read;
   - Every time waiting for 0.1 s.

3. [lib/datasets/kitti_rcnn_dataset.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/lib/datasets/kitti_rcnn_dataset.py)
   Add sampler-safe:
   - far points  /  16384 Security without replacement sampling;
   - Allow replacement when the patch is insufficient;
   - Avoid negative sampling and crash.

4. [tools/eval_rcnn.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/tools/eval_rcnn.py)
   args.test skips the built-in AP in even split=val and allows the unified offline evaluation of recovery AP.

### 8.2 Add Key Script

Core wrapper:

- [kitti_patch_extractor.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/kitti_patch_extractor.py)
- [tf_pugcn_family_patch_infer.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/tf_pugcn_family_patch_infer.py)
- [tf_punet_patch_infer.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/tf_punet_patch_infer.py)
- [pdans_patch_infer.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/pdans_patch_infer.py)
- [strict_x4_from_merged_raw.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/strict_x4_from_merged_raw.py)

Experimental organization:

- [run_patch_causal_upsampling.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_patch_causal_upsampling.py)
- [run_surface_candidate_256_sequence.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_surface_candidate_256_sequence.py)
- [prepare_pointrcnn_e1_e2_inputs.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/prepare_pointrcnn_e1_e2_inputs.py)
- [run_pointrcnn_e1_e2_evals.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pointrcnn_e1_e2_evals.py)
- [run_centerpoint_exact4n_observed_first.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_centerpoint_exact4n_observed_first.py)
- [run_centerpoint_exact4n_local_staged.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_centerpoint_exact4n_local_staged.py)

Key functions:

- exact-4N point count Validation;
- (a) Preservation of the observation site and generated points provenance;
- Recent Neighborhood Values;
- Various patch models;
- Local cover supplementary;
- Unit ball normalization / reverse transformation;
- (b) Methodological-specific reasoning environments;
- E1/E2/E3 and double detector automatic assessment;
- geometric, voxel, GT Transfer and visualization of cases.

### 8.3 PU-Net Critical Repair

Historical integration report:

- [PU_NET_INTEGRATION_REPORT](/home/ra87racy/projects/baseline_detectors/PointRCNN/PU_NET_INTEGRATION_REPORT.md)

Historical changes include:

- Python 2 → Python 3;
- Add data_folder parameters;
- (a) Fixing the integer division;
- Add CPU fallback;
- Updates the custom op compilation script.

The most critical cause and effect repairs are:

- [tf_punet_patch_infer.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/wrappers/tf_punet_patch_infer.py)

unit_sphere_v1 Process:

1. patch less heart;
2. (a) Zoom to a unit ball by the maximum radius;
3. Delineating in scale, which corresponds to the distribution of training;
4. Output times scale and adds back the hysteria.

### 8.4 PDANS Changes

- pointnet2/util.py: Create torch.normal/tensor for device perception, remove hard encoding. cuda();
- pointnet2_utils.py: Set TORCH_CUDA_ARCH_LIST=7.5 with setdefault;
- checkpoint and compilers are local untraceed assets.

### 8.5 PU-GCN Changes

tf_ops/compile.sh:

- Remove hard-coded CUDA 10;
- Support CUDA_HOME/CUDA_PATH;
- Support conda libcudart;
- Using TensorFlow actual compile/link flags;
- Adjust PATH/LD_LIBRARY_PATH.

### 8.6 OpenPCDet / CenterPoint Changes

- KITTI-only replace Argo2 data set registration with an optional one, avoiding forced reliance on av2;
- Use weights_only=False for a trusted checkpoint metadata iconic format, suitable for PyTorch 2.6+;
- Add create_kitti_val_infos_only.py.

### 8.7 patch Ripper Integrator

Achieved/tried:

- deterministic_spatial_chunk;
- FPS + kNN;
- FPS + ball query;
- cover supplement;
- fps_ball_cover_knn_v3;
- surface-c32 (through the only 2048 support and zero-duplication constraint).

Design evolution:

> The box is a series of slices that are partially covered but are seriously repeated, local, fully covered, only supported, and zero duplication.

---

## 9. Timeline for Job Records

### 2026-05: Baseline recurrence and early method access

- Reproducing PointRCNN baseline;
- Create early input of EAR-style, PU-Net and PU-GCN;
- Fix Python/CUDA/TensorFlow Custom op compatibility;
- (a) Completion of early full or weekly grade comparisons;
- The PU-Net output anomaly and sampling crash problem were found.

### 2026-06: Methodological feasibility and HPC migration

- Check TULIP, PDANS, SPU-PMD, PU-EdgeFormer;
- Combination of checkpoint, algorithms and environmental barriers;
- Statistics 10 old final variant, each 3769 frame, total approximately 42 GB;
- Development of a laboratory to HPC for migration plan;
- 2026-06-15 SSH dry-run did not really transmit because of the failure of authentication.

Migration report:

- [lab_to_hpc_transfer_plan](/home/ra87racy/projects/kitti_upsampling_lab_to_hpc_migration/reports/lab_to_hpc_transfer_plan.md)

### 2026-07: Multiplication Audit and Harmonization protocol

- It was found that the old output was not strictly 4×;
- Definition of Line A / Line B;
- (a) Convergence the output into exact-4N for observed + generated;
- Introduction of deterministic, label-free, AP-free test input options;
- Finish sampler-safe fix.

### 2026-07 Middle: PointRCNN E1/E2/E3 Full

- Completed four methods x two lines x 3769 frame;
- Generating E1/E2 official main table;
- Operation of the E3 32768 Observatory Priority Sensitivity Experiment;
- Confirm that PointRCNN cap is a magnifying factor and not the only root cause.

### 2026-07 Later: CenterPoint and cross detector

- Access to OpenPCDet CenterPoint;
- fixed The same 80 epoch checkpoint;
- Complete exact-4N double-line full;
- Conducting observed-first in group sorting experiments;
- Statistics voxel cap, GT Transfer and Representation Cases.

### 2026-07-25: dosage and matching control

- Run 256 frame g2.5–g50;
- Run matched observed-fill control;
- Completed full g10;
- a small number of PDANS was found to have a partial gain at Line A, but the high percentage has steadily deteriorated.

### 2026-07-30 to 2026-08-04: Gene, patch and PU-Net

- Positioning old patch non-local;
- Construct fps_ball_cover_knn_v3;
- The discovery of duplicate fillings resulted in a PointRCNN/CenterPoint disagreement;
- Completed PU-Net normalization × patch 2×2;
- Designing surface-c32 is the only candidate to support;
- Completion of 20 frame geometric and method output validation;
- Completion of three methods 256 frame generation;
- PDANS 256 frame was interrupted after 231 merged raw;
- The double detector AP has not been run by surface-c32.

---

## 10. Index to Reports, Results and Logs

### 10.1 Prioritizes the writing of papers

1. [It's a dissertation-cause report.](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/dual_detector_three_frame_root_cause_20260730/THESIS_ROOT_CAUSE_REPORT_EN.md)
2. [E1/E2/E3 Root Report](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_e3_final_root_cause_report.md)
3. [Joint dosage analysis](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_x4_detector_recovery_combined_analysis_20260725/combined_analysis_report.md)
4. [CenterPoint observed-first Full Report](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/centerpoint_exact4n_reconstructed_order_safe_20260729/CENTERPOINT_OBSERVED_FIRST_FULL_REPORT.md)
5. [patch/PU-Net Causal protocol](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/PROTOCOL.md)
6. [PDANS detector Reason for Difference](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/analysis/pdans_old_vs_cover_knn_v3/PDANS_DETECTOR_DISAGREEMENT_ROOT_CAUSE.md)
7. [PU-Net 2×2 Final decision-making](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731/punet256_2x2/reports/FINAL_DECISION_CN.md)

### 10.2 Original Number

- [PointRCNN E1/E2 AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_live_ap_summary.csv)
- [PointRCNN E3 AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e3_real_first_32768_live_ap_summary.csv)
- [Double detector All AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/dual_detector_three_frame_root_cause_20260730/analysis/all_detector_ap_r40_bbox_bev_3d.csv)
- [CenterPoint full AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/centerpoint_exact4n_reconstructed_order_safe_20260729/full_ap_summary.csv)
- [Full dosage BEV/3D Table](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_x4_detector_recovery_combined_analysis_20260725/complete_bev_3d_comparison.md)

### 10.3 History and Project Records

- [2026-05-04 Job Log](/home/ra87racy/projects/baseline_detectors/PointRCNN/WORKLOG_2026-05-04.md)
- [2026-05-16 Weekly Report](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/weekly_report_20260516_full_comparison/weekly_progress_report.md)
- [PU-Net Integration Report](/home/ra87racy/projects/baseline_detectors/PointRCNN/PU_NET_INTEGRATION_REPORT.md)
- [Multiplication audit](/home/ra87racy/reports/upsampling_ratio_audit.md)
- [HPC Migration plan](/home/ra87racy/projects/kitti_upsampling_lab_to_hpc_migration/reports/lab_to_hpc_transfer_plan.md)
- [ModelNet40 protocol State](/home/ra87racy/projects/modelnet40_pointnet2_upsampling/reports/modelnet40_x4_final_protocol_report.md)

---

## The paper used by 11., the methodological principles and the role of the project

### 11.1 data set and detector

1. Andreas Geiger, Philip Lenz, Raquel Urtasun.
   “Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite.” CVPR 2012.
   Effects: KITTI data sets and 3D testing baseline source.
   Original language:[KITTI/CVPR PDF](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Geiger2012CVPR.pdf)

2. Shaoshuai Shi, Xiaogang Wang, Hongsheng Li.
   “PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud.” CVPR 2019.
    Principle: Phase one is to do foreground on the original point.  bottom-up proposal; The second stage details 3D box in canonical coordinates.
   The role of the project: downstream detector, the main point;Its 16384 point budget and PointNet++ local aggregation is used to analyse generated points competition.
   Original language:[CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.html)

3. Tianwei Yin, Xingyi Zhou, Philipp Krähenbühl.
   “Center-Based 3D Object Detection and Tracking.” CVPR 2021.
   Rationale: The 3D target is presented as the focal point, using BEV keypoint head testing centre and returning to dimensions, directional properties.
   The role of this project: voxel /BEV detector, which is used to verify whether root causes are in place across the detection architecture.
   Original language:[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html)

4. Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas.
   “PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.” NeurIPS 2017.
   The principle is to apply PointNet on the embedded neighbourhood to learn the local structure of points in multiple scale.
   This project function: PointRCNN backbone is the basis for local aggregation and downstream network of ModelNet40 classification branch.
   Original language:[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html)

### 11.2 Official upsampling Method

5. Lequan Yu, Xianzhi Li, Chi-Wing Fu, Daniel Cohen-Or, Pheng-Ann Heng.
   “PU-Net: Point Cloud Upsampling Network.” CVPR 2018.
   Practising: Learning multi-layered, point-by-point characteristics, amplified in multiple branch volumes, expanding point count in characteristic space, training patch-level network in combined loss combinations with surface matching and even distribution.
   The role of this project: the earliest learning model upsampling baseline;Its patch-level and normalization scenarios directly expose the current whole scene adaptation problem.
   Original language:[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.html)

6. Guocheng Qian, Abdulellah Abualshour, Guohao Li, Ali Thabet, Bernard Ghanem.
   “PU-GCN: Point Cloud Upsampling Using Graph Convolutional Networks.” CVPR 2021.
   Principle: Inception DenseGCN does more than scale feature extraction, NodeShuffle uses a graphic volume neighbourhood information extension point.
   The role of this project: the formal method of volume-forming, the performance of which is usually only less than PDANS.
   Original language:[CVF Open Access PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_PU-GCN_Point_Cloud_Upsampling_Using_Graph_Convolutional_Networks_CVPR_2021_paper.pdf)

7. Dohoon Kim, Minwoo Shin, Joonki Paik.
   “PU-EdgeFormer: Edge Transformer for Dense Prediction in Point Cloud Upsampling.” ICASSP 2023 / arXiv:2305.01148.
   Rationale: Combining EdgeConv volume and multi-headed attention with modelling of local geometric and global structures.
   The role of this project: Transformer/graph Mixture.
   Original language:[arXiv](https://arxiv.org/abs/2305.01148)

8. Boqian Zhang, Shen Yang, Hao Chen, Chao Yang, Jing Jia, Guang Jiang.
   “Point Cloud Upsampling Using Conditional Diffusion Module with Adaptive Noise Suppression.” CVPR 2025.
   (b) Rationale: the proliferation of conditions to generate density points;ANS Self-adaptation noise inhibition based on point-to-neighbourhood;TreeTrans integrates cross-layer features.
   The role of this project: the most robust of the formal methods, especially under noise and distributional deviations, is generally the best.
   Original language:[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Point_Cloud_Upsampling_Using_Conditional_Diffusion_Module_with_Adaptive_Noise_CVPR_2025_paper.html)

### 11.3 Feasibility/background Method

9. Bin Yang, Patrick Pfreundschuh, Roland Siegwart, Marco Hutter, Peyman Moghadam, Vaishakh Patil.
   “TULIP: Transformer for Upsampling of LiDAR Point Clouds.” CVPR 2024.
   Rationale: The LiDAR projection is range image and the patch/window geometric of Swin Transformer is modified to match LiDAR range map characteristics.
   Role of the project: Original LiDAR method candidate;The old output has a xyz point count does not imply problem with protocol, which is strictly 4×, not entering the final four methods main table.
   Original language:[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_TULIP_Transformer_for_Upsampling_of_LiDAR_Point_Clouds_CVPR_2024_paper.html)

10. Yanzhe Liu, Rong Chen, Yushi Li, Yixi Li, Xuehou Tan.
    “SPU-PMD: Self-Supervised Point Cloud Upsampling via Progressive Mesh Deformation.” CVPR 2024.
    Principle: To consider upsampling as a condensable condensation, gradually optimising the structure through a thick grid plug-in and a multistage mesh deformation.
    Impact of the project: Completion of feasibility/ smoke, not entering the official main table.
    Original language:[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_SPU-PMD_Self-Supervised_Point_Cloud_Upsampling_via_Progressive_Mesh_Deformation_CVPR_2024_paper.html)

---

## 12. Core principles used by this project

### 12.1 Principle of Fair Comparison

1. The output point count must be strictly the same;
2. Observatory must keep and record provenance;
3. Only enter point cloud, frozen detector and weight;
4. All sampling certainty;
5. Select input without GT label or AP;
6. Small-scale mechanism screening before full validation;
7. Distinction between engineering bug, protocol impact and methodological capabilities.

### 12.2 patch Local geometric

The generic point-cloud upsampling network usually assumes that the input is a local patch surface that regulates scale.It is therefore necessary to:

- The FPS guarantee centre coverage;
- kNN Guaranteed fixed neighbourhood;
- ball query guarantees the locality of space;
- cover supplement ensures that uncovered sources are included;
- Single source constraints to avoid duplication of effort;
- overlap controls patch boundary stability;
- normalization / Inverted the KITTI scene map back to train scale.

### 12.3 Test budget competition

- PointRCNN: Limited point slot;Repeated points directly occupy point budgets.
- CenterPoint: Limited voxel slot;Repeat with voxel may be absorbed, but a new pseudo voxel will take over voxel budget.
- upsampling should therefore optimize “independent, supported space evidence”, not total point count.

### 12.4 Surface Consistency

Additional points should:

- Close to existing/reference surfaces;
- (a) Expand along local cut-off plane;
- Avoiding Depth Breaking and Object boundary;
- Not forming unsupported voxels in the air;
- Maintain scale adaptation for long and thin targets.

Local PCA can decompose neighbourhood into the direction of the plane and the method;Ideally, the new points are being scaled up, but the law is being reduced.

### 12.5 Dose - Response and Match Control

The higher the generated points rate and the lower the AP, the dose effect of the injury is indicated.
matched real-fill for answer:

> Change of performance from "More Point" or "generated points Mass"?

The gap between PDANS g2.5 and real-control, only or about + 0.68, explains the need for a careful explanation of a small number of benefits.

### 12.6 AP Relationship to Institutional Indicators

- AP_R40 is the final mission indicator;
- geometric, voxel and GT are intermediate evidence of the mechanism of interpretation;
- Okay, the geometric average does not guarantee an increase in the detection;
- detector is probably the most sensitive point of error near the a small number of key target, empty voxel and budget competition.

---

## 13. can write with cannot

### 13.1.

- (a) The establishment of a rigorous exact-4N, double line, frozen double detector assessment protocol;
-  Proofs current common  patch  upsampling Method  KITTI  On the scene, cannot automatic recovery downstream tests;
- PDANS is the most robust of the four formal methods;
- PointRCNN is in the same general order as CenterPoint;
- Input point/ voxel budget is the error amplifier;
- (a) Old non-local patch and duplicate fillings are measurable key tube defects;
- The unit ball PU-Net normalization / reverse transformation is a necessary match;
- a small number of generated points may have partial benefits at Line A, but a high rate of steady degradation;
- surface-c32 has significantly improved on 20 frame geometric.

### 13.2 cannot Write

- "4× upsampling recovery with original detection performance";
- "PU-Net Method per se only 8 AP";
- "surface-c32 has been upgraded to test AP";
- “ModelNet40 classification Results Completed”;
- "PDANS g2.5 stabilizes the full amount of data above original baseline;
- "TULIP, SPU-PMD and the four official methods are already the same exact-4N protocol fair comparison";
- Combining the results of earlier x2,100k cap, RPN 26000 or old adapters into main table;
- The 3 frame or 256 frame mechanism is formally concluded by 3769 frame.

---

## 14. Recommended Paper Structure

### Chapter 1: Introduction

- LiDAR Sludge and downstream tests;
- point-cloud upsampling usually optimize geometric metrics, but it is not clear whether or not it will help with the testing;
- This paper is concerned with the effectiveness and failure mechanisms of the task under strict control.

### Chapter 2: Related work

- point-cloud upsampling: PU-Net, PU-GCN, PU-EdgeFormer, PDANS;
- Original LiDAR upsampling: TULIP;
-  upsampling: SPU-PMD;
- Point test: PointRCNN/PointNet++;
- voxel Central Test: CenterPoint.

### Chapter 3: Methods and Experiments protocol

- exact-4N;
- Line A / Line B;
- observed/generated provenance;
- E1/E2/E3;
- frozen Double detector;
- geometric, voxel and mission indicators.

### Chapter 4: Main experimental results

- PointRCNN Full main table;
- CenterPoint cross detector results;
- The method is sorted differently from Line A/Line B.

### Chapter 5: Root causes and ablation

- patch is not local;
- Point/ voxel budget;
- observed-first;
- E3;
- Dose with matched real control;
- PU-Net normalization × patch;
- Locally repeating patch's detector disagreement.

### Chapter 6: Improving the way forward

- surface-c32;
- Local, sole, prima facie, budgetary awareness;
- You need mission awareness training or KITTI fine-tuning.

### Chapter 7: Limitations and Conclusions

- (a) Domain differences;
- The fitncies of the old PU-Net main table;
- surface-c32/ModelNet40 San not completed;
- The conclusions are limited to the current frozen detector and the input protocol.

---

## 15. Next Trial Priority

### P0: Finish the current surface-c32 Cascades

1. Continue or safely rerun from 231/256 PDANS;
2. Generate PDANS strict final bins;
3. Re-audit exact-4N for the four methodologies;
4. First 256 frame PointRCNN and CenterPoint;
5. Compared to the old patch, the first version of the local repeat patch, the original/downsampling baseline;
6. Only double detector is considered for 3769 frame at least if it does not deteriorate.

### P1: supplementary surface-c32 Table of key mechanisms

- patch unique support;
- patch diameter;
- multi-object fraction;
- generated-to-real NN;
- occupied voxels;
- top10% voxel concentration;
- CenterPoint cap hits;
- The true point retention rate after PointRCNN 16384;
- GT TP→FN / FN→TP.

### P2: Reorder the main result of PU-Net

- The results of the old adapter are retained as a breakdown comparison;
- Correct normalization + surface-c32 as a fixation;
- Do not replace the old table with a fixated version without indicating protocol changes.

### P3: Full confirmation of a small number of dose

- Priority full PDANS g2.5;
- Accompanying full-val real-control c2.5;
- Pre-registration of the same random seed and the same 16384 rules;
- Check if promotions still exist.

### P4: mission perception improvement

- fine-tuning on KITTI local surface patch;
- Different patch rules for objects boundary, deep fractures and ground;
- (a) Consistency of the accession method;
- Set up detector-aware confidence for generated points;
- Control the number of new voxel instead of only control point count;
- The study uses generated points as an optional candidate instead of forcing the full 3N.

### P5: ModelNet40 branch

- Only after HPC Environment recovery;
- Finish PU-GCN and PointNet++ baseline first;
- To decide whether it is worth accessing PU-EdgeFormer;
- The classification results should be made stand-alone and not commingled with the KITTI results.

---

## 16. Recapturing the experimental inspection list

- [ ] (a) Data split is 3769 frame val;
- [ ] Consistency of Line A/Line B definitions;
- [ ] Is every frame strict 4N/4M;
- [ ] (b) Whether all of the sites are retained;
- [ ] Can generated points provenance be traced;
- [ ] Whether intensity has been given value in accordance with the Harmonized Neighbourhood Rules;
- [ ] Existence of NaN/Inf;
- [ ] PointRCNN Is fixed the same checkpoint/config;
- [ ] CenterPoint Is fixed the same checkpoint/config;
- [ ] Whether E2 16384 rules are fully consistent;
- [ ] Whether to use seed for certainty;
- [ ] Whether or not to select input using label/AP;
- [ ] PU-Net Is the old/new normalization clearly marked;
- [ ] patch is a sectoral, sole support and repetition rate for statistical offices;
- [ ] Check CenterPoint 40000 voxel cap;
- [ ] Whether 256 frame and 3769 frame results are clearly differentiated;
- [ ] (b) Whether the historical unharmonized results are isolated from the official table;
- [ ] Whether the not completed experiment is clearly marked as pending.

---

## 17. Final sentence

The most valuable thesis contribution of the current project is not to prove that a common upsampling network can “create more points” on KITTI, but to prove it by means of strict counting, double entry lines, double detector and causal ablation:

> For the downstream 3D test, it is decisive whether the new points can form local, sole, real surface support and supplementary within the limited point/ voxel budget, rather than crowding in sensor evidence.

---

## 18. 2026-09-13 Current Status Overview (update)

### 18.1 results must be divided into three layers.

As of 2026-09-13, KITTI has developed three types of evidence that cannot has replaced each other:

| Level | Main purpose | Data size | detector State | Current Use |
|---|---|---:|---|---|
| Four methods to harmonize the main experiment. | Fair comparison PDANS, PU-GCN, PU-EdgeFormer, PU-Net* | 3769 frame × Line A/B | frozen Official/established weight | Main horizontal comparison of methods |
| Mechanism and ablation Experiment | Position patch, normalization, dosage, dos/voxel budget, migration of objects | 20/32/64/256 frame | frozen or small-scale adaptation | Explain why, cannot replaces full AP |
| PU-GCN detector-adaptation Special purpose | Determine whether detector input field adaptation recovers loss | 3712 Training frame;3769 Validation frame;20 arms | official and 3-epoch adapted weights | The latest and strongest input field adaptation evidence |

Two sentences should therefore be retained in the paper or report:

1. The fair ranking of the four methods frozen detector is still given by 3 main table, PDANS is the most robust, and four methods overall do not have recovery baselines.
2. Follow-up PU-GCN special certificates detector adaptation and observed-first can significantly recover some AP;The 20/20 arm matrix is complete, but all PU-GCN input conditions are still no above the corresponding baseline.

### 18.2 Current Completion

- Four methods strict-4N Full master matrix: completed.
- PointRCNN/CenterPoint Cross detector Full master matrix: completed.
- E1/E2/E3, observed-first, dosage, voxel, object transfer, distance layer: completed.
- 256 frame 2×2 for PU-Net normalization × patch: completed;A fixated version of not completed 3769 frame double detector full main table.
- PU-GCN detector adaptation: 3712 frame training, 3769 frame validation, 20/20 arms completed.
- Multi-mode surface-c32 Full Double detector Rerun: The unified final matrix has not yet been formed.
- ModelNet40 branch: Still only smoke/ protocol is ready, cannot is written in full classification.

Recent direct evidence:

- [PU-GCN Full 3769 frame Test Matrix](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/full_val_detector_matrix.md)
- [PU-GCN Full protocol with code description](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/protocol_and_code_en.md)
- [3-epoch convergence Audit](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/convergence_audit.md)

---

## 19. PU-GCN detector adaptation: The latest 3769 frame full result

### 19.1 protocol

- upsampler: fixed PU-GCN PU1K `model-100`, no retrained on KITTI by 3 epochs.
- detector adaptation training set: KITTI train 3712 frame.
- Final evaluation set: KITTI val 3769 frame.
- PointRCNN: RPN 3 epochs, then offline RCNN 3 epochs;batch size 1, workers 0, Adam one-cycle, LR 0.0002, seed 20260823.
- CenterPoint: 3 epochs; batch size 2, workers 0, Adam one-cycle, LR 0.0003, weight decay 0.01, seed 666.
- Save only epoch 3;no epoch 1/2 is certified AP, so "loss is down" does not imply "convergence has been proven".
- The most recent full-val evaluator feature uses AP_R40; must not Put the old  PointRCNN legacy evaluator  It's...  AP_R11  Numbers are commingled into this table.

Type of input:

- `baseline`: Real observations of input lines;Line A is the original N point and Line B is M=floor(N/4) point.
- `direct`: PU-GCN strict 4N/4M Direct output.
- `observed-first`(a) The complete retention of N/M sites and the extraction of 3N/3M from the strict prediction pool for certainty;The total number is still 4N/4M.
- `official`: Established detector weight.
- `adapted`: carry out the weight after 3-epoch detector adaptation for the corresponding input field.

### 19.2 20/20 arms Full Matrix

The following are Car 3D AP_R40 Easy / Moderate / Hard;Each arm is 3769/3769, status PASS.

| Detector | Line | Input | Weights | AP_R40 E/M/H | Moderate margin |
|---|---|---|---|---:|---:|
| PointRCNN | A | baseline N | official | 92.4325 / 81.9528 / 77.8468 | 0.0000 |
| PointRCNN | A | direct 4N | official | 84.7443 / 61.9294 / 57.7869 | -20.0235 |
| PointRCNN | A | direct 4N | adapted | 85.1671 / 68.8800 / 64.6697 | -13.0728* |
| PointRCNN | A | observed N + predicted 3N | official | 85.0374 / 64.2748 / 60.0911 | -17.6780 |
| PointRCNN | A | observed N + predicted 3N | adapted | 86.7068 / 71.0075 / 66.7939 | -10.9453* |
| PointRCNN | B | baseline M | official | 85.2016 / 65.5757 / 61.2831 | 0.0000 |
| PointRCNN | B | baseline M | adapted | 84.3920 / 68.3312 / 64.2613 | 0.0000 |
| PointRCNN | B | direct 4M | official | 46.4657 / 30.0909 / 25.8000 | -35.4848 |
| PointRCNN | B | direct 4M | adapted | 68.2592 / 46.6150 / 40.2487 | -21.7163 |
| PointRCNN | B | observed M + predicted 3M | official | 55.0177 / 35.5289 / 30.9359 | -30.0469 |
| PointRCNN | B | observed M + predicted 3M | adapted | 74.1970 / 55.2145 / 49.0583 | -13.1167 |
| CenterPoint | A | baseline N | official | 88.3907 / 79.2773 / 76.7371 | 0.0000 |
| CenterPoint | A | direct 4N | official | 81.8888 / 60.6081 / 57.9061 | -18.6692 |
| CenterPoint | A | observed N + predicted 3N | official | 83.3121 / 63.2064 / 61.0418 | -16.0709 |
| CenterPoint | A | observed N + predicted 3N | adapted | 86.8154 / 74.7012 / 72.6570 | -4.5761* |
| CenterPoint | B | baseline M | official | 81.4003 / 64.5979 / 59.9701 | 0.0000 |
| CenterPoint | B | baseline M | adapted | 83.7514 / 68.0490 / 64.6309 | 0.0000 |
| CenterPoint | B | direct 4M | official | 56.4211 / 35.3530 / 31.2043 | -29.2449 |
| CenterPoint | B | observed M + predicted 3M | official | 66.9217 / 44.0929 / 40.0268 | -20.5050 |
| CenterPoint | B | observed M + predicted 3M | adapted | 80.4109 / 61.6639 / 57.5252 | -6.3851 |

`*` Line A no Individual training "adapted original-N baseline";These values refer to official Line A baseline and therefore change both input and weight, cannot being treated as a pure input margin.Line B has a pair of adapted baseline, which is cleaner with a causal caliber.

### 19.3 Activates directly from the matrix

#### detector adaptation recovery

| Detector/Line/Input | Adapted − Official Moderate AP |
|---|---:|
| PointRCNN A direct | +6.9506 |
| PointRCNN A observed-first | +6.7327 |
| PointRCNN B baseline | +2.7555 |
| PointRCNN B direct | +16.5241 |
| PointRCNN B observed-first | +19.6856 |
| CenterPoint A observed-first | +11.4948 |
| CenterPoint B baseline | +3.4511 |
| CenterPoint B observed-first | +17.5710 |

#### observed-first recovery

| Detector/Line/Weights | Observed-first − Direct Moderate AP |
|---|---:|
| PointRCNN A official | +2.3454 |
| PointRCNN A adapted | +2.1275 |
| PointRCNN B official | +5.4380 |
| PointRCNN B adapted | +8.5995 |
| CenterPoint A official | +2.5983 |
| CenterPoint B official | +8.7399 |

CenterPoint no `direct adapted` arm, so cannot calculates the pure observed-first−direct margin under adapted weights from the existing matrix.

### 19.4, Latest Conclusion

1. detector adaptation is an effective means of mitigation, particularly for Line B;For example, PointRCNN observed-first upgrade 19.6856 AP and CenterPoint observed-first upgrade 17.5710 AP.
2. observed-first is also an effective input protection tool, and Line B benefits more than Line A, which means that real observations under thin input are more easily diluted in generated points.
3. The two devices are still superseding no over pair baseline.The closest to the baseline is CenterPoint Line A observed-first adapted, still low 4.5761 AP;CenterPoint Line B with a pair of adapted baseline is still low 6.3851 AP and PointRCNN Line B is still low 13.1167 AP.
4. Thus, "frozen detector caused all the failures" is not valid;The local deviation explains the larger part, but the generation of geometric, patch and input budgets still leave an inescapable gap.
5. The three rounds of training have not yet proved to be the best.Only epoch-3 checkpoint and declining training loss, no by epoch full-val curve;cannot writes “3 epochs already convergence”.

### 19.5 Training loss Audit

All arm of PointRCNN has fallen from epoch 1 to 3 median loss:

- Line A direct: RPN -17.03%, RCNN -8.00%;
- Line B direct: RPN -27.61%, RCNN -7.65%;
- Line B baseline: RPN -23.95%, RCNN -4.16%;
- Line A observed-first: RPN -16.46%, RCNN -4.42%;
- Line B observed-first: RPN -24.82%, RCNN -6.86%.

CenterPoint mean loss:

- Line A PU-GCN: 2.33  /  2.11, down 9.44%;
- Line B PU-GCN: 3.69  /  3.10, down 15.99%;
- Line B baseline: 3.00  /  2.65, down 11.67%.

These figures confirm that the process is being optimized and that validation AP has reached the platform period.

---

## 20. 2026-08 to 2026-09 Additional working hours

### 2026-08-05: detector-aware Low dose PDANS route

- Construct the V1—V5 low dose, voxel perception and proposal-gated candidate.
- V4 anchored generated points on pilot256 to the already measured voxel;CenterPoint is about 423.5 for each frame generated points medium range and voxel for new activation is 0.
- PointRCNN Car Moderate 3D/BEV of V4 relative baseline to + 0.3093/+0.2093;CenterPoint Car 3D is - 0.0192, Pedestrian/Cyclist 3D, respectively - 0.4366/-0.3439.
- V5 uses CenterPoint Car, which has no GT, and the NMS candidate for class door control has basically returned Pedestrian/Cyclist to baseline, but pilot256 is used only for development, cannot when it is eventually upgraded.
- The subsequent independent holdout and multiple seed audits did not support the small positives being expanded into stable returns.

Evidence:

- [V4 Decision-making Report](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detector_aware_pdans_v4_20260805/reports/v4_decision_report.md)
- [V5 pilot Decision-making Report](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detector_aware_pdans_v5_20260805/reports/v5_pilot256_decision_report.md)
- [Cross-detector diagnosis](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detector_aware_cross_detector_diagnosis256_v1_20260805/summary/diagnosis_report.md)

### 2026-08-08 : OpenPCDet Category III PointRCNN pilot256

- Using the same three categories OpenPCDet PointRCNN checkpoint, a parallel evaluation of Car, Pedestrian and Cyclist was conducted.
- The four traditional exact-4N methods of Line A/B do not exceed the corresponding baseline on Moderate 3D and BEV.
- PDANS is the most robust in general, PU-GCN is usually the second;Failure is not limited to Car.
- detector-aware V4 is the only candidate close to baseline, but it is about 2.4%, inserted at low doses, which does not belong to exact-4N horizontal main table.

[Full comparison of the three categories](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/openpcdet_pointrcnn_three_class_all_methods_pilot256_20260808/ALL_METHODS_THREE_CLASS_COMPARISON_ZH.md)

### 2026-08-10 to 2026-08-12: sampling Noise and Experimental Scale Audit

- split-region  Path in  16 seeds  I'm not sure if I'm going anywhere. V1−baseline  It's...  Car Moderate 3D AP  Mean value as  -0.438,  Standard deviation  1.769; The original single seed + 1.2227 was convicted of being a lucky sample.
- 9 out of a logarithm of 16 is negative;MDE95=3.47 AP.
- (a) Matching of 20 frame resampling with delta standard deviations of 6.02 AP, MDE95 approximately 11.8 AP;So 20 frame is only suitable for finding a catastrophic retreat, cannot judging 1—3 AP's methodological gain.
- The noise conclusion applies only to the split-region path, which requires random refilling;cannot goes directly to fixed 16384 point E2 path.

Evidence:

- [16-seed Judgement](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pointrcnn_sampling_variance_probe_20260810/SIGMA_16SEED_VERDICT_ZH.md)
- [Candidate methodology and 20 frame interpretation skills](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detection_oriented_upsampling_triage_20260812/DETECTION_ORIENTED_UPSAMPLING_TRIAGE_EN.md)

### 2026-08-24 to 2026-08-27: detector fine-tune to full training

-  First, fixed  64  Training frame completed  6 arms  It's...  PointRCNN RPN+RCNN 3-epoch screening,  And the same.  256  frame on evaluation.
- PU-GCN Line A/B for fine-tuning recovery +4.1746/+4.5949 AP;PDANS Line B recovery +4.4260 AP;But all appropriate inputs are still below corresponding to baseline.
- The small sample fine-tuning has also reduced baseline itself and can therefore only be used as filter evidence.
- The PointRCNN/CenterPoint input field adaptation of PU-GCN was subsequently completed using all 3712 training frame and complemented by observed-first branch.

[64-frame fine-tune screening](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pointrcnn_finetune64_six_arms_20260824/reports/finetune64_screen_report.md)

### 2026-09-08 to 2026-09-10: full 20-arm

- Regenerated/audited Line A, Line B PU-GCN input.
- Completed direct, observed-first, official, adapted PointRCNN 11 arms and CenterPoint 9 arms.
- Intensifier only `frame_count=3769` and `status=PASS` (a) Results;Finally 20/20 through.
- Visible correction AP cal. R40 to avoid a mix of repository legacy R11.

### 2026-09-11 to 2026-09-13: Thesis and delivery

- Chapter 3—4 KITTI experimental configuration/methodology body, chapter 5—6 outcome/discussion body.
- (b) The formation of 37 KITTI academic drawings, contact sheet, Markdown/LaTeX/PDF multiformats.
- Collapse reporting and code evidence in English for PU-GCN detector-adaptation.

Main delivery:

- [Chapter 3—4 KITTI Detailed body](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_CHAPTERS_3_4_KITTI_DETAILED_ZH.md)
- [Chapter 5—6 KITTI Detailed body](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH.md)
- [Chapter 5—6 PDF](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH.pdf)
- [Harmonization of academic kits](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/CHAPTERS_5_6_UNIFIED_ACADEMIC_FIGURES_20260911)

---

## 21. Code Usage and Recurrence Access (current version)

### 21.1 Run PU-GCN Full detector-adaptation Matrix

```bash
cd /home/ra87racy/projects/baseline_detectors/PointRCNN
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh all
```

However, the recovery phase:

```bash
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh generate_a
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh generate_b
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh prepare
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh pointrcnn
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh centerpoint
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh aggregate
```

`all` The real data flow is:

1. `run_patch_causal_upsampling.py`: extract patch, call PU-GCN, merge raw candidate and generate strict 4×.
2. `verify_patch_causal_strict_x4.py`Check frame overlay, point count, limited values and strict rules.
3. `prepare_centerpoint_observed_first_train.py`: Construct observed N/M + predicted 3N/3M.
4. `run_pointrcnn_checkpoint_split_eval.py`: Evaluation of PointRCNN by visible split, visible LiDAR catalogue, visible checkpoint.
5. `run_patch_causal_centerpoint_eval.py`: Evaluate CenterPoint.
6. `summarize_pugcn_detector_adaptation_full_val_20260908.py`: Reject any 3769 frame or non-PASS arm and generate the final matrix.

### 21.2 Line B downsampling

Core rules:

```text
M = floor(N/4)
seed(frame) = 20260702 + int(frame_id)
choice(N, M, replace=False)
sort(indices)
```

Entry:

```bash
/home/ra87racy/projects/baseline_detectors/PointRCNN/venv_pointrcnn/bin/python \
  scripts/prepare_kitti_downsampled_x4_val.py
```

### 21.3 patch and PU-GCN Key parameters

| Parameters | Line A | Line B |
|---|---:|---:|
| patch Size | 2048 | 2048 |
| primary patch budget | ceil(N/2048) × 1 | ceil(M/2048) × 1 |
| primary ball radius | 2 m | 4 m |
| primary center Minimum Support | 2048 | 2048 |
| cover radius | 6 m | 6 m |
| cover eligibility | At least 32 points in radius | At least 32 points in radius |
| supplemental kNN floor | 2048 unique | 2048 unique |
| seed | 20260702 + frame_id | Same Left |

PU-GCN Parameter: PU1K `model-100`, PUGCN, Inception DenseGCN, `n_blocks=2`, `channels=32`, `k=20`, `d=2`, NodeShuffle, 4×. Each 2048 patch first makes a unit ball of normalization with a mass heart and the remotest distance, and output 8192 points and then reverses the LiDAR coordinates.

### 21.4 strict 4× and observed-first selection rules

- patch raw If the number of candidates is greater than 4N/4M, fixed seed sample uniformly without replacement to the target number;If the candidate is short, the candidate fails and does not copy the list.
- intensity is not predicted by PU-GCN;Use 1-NN intensity for generating XYZ with the same frame observed XYZ.
- 3N/3M of observed-first is not the confidence output of PU-GCN, nor is it the best subset of geometric;It uses SHA-256 to be born frame seed after sample uniformly without replacement.
- The current no uses confidence, warp, FPS, the distance to observed or voxel to re-elect 3N/3M.

### Evidence that 21.5 must save before and after running

- `protocol.json`: protocol and the path;
- per-frame manifest: N/M, 4N/4M, seed, status;
- provenance: entrypoint, checkpoint, real call model, fallback;
- Input and output limit values, shape, point count, observation prefix/multi-pool audit;
- detector checkpoint/config;
- 3769 projection documents and empty documentation statistics;
- AP Original Output, Parsing CSV, run_complete Status;
- Source snapshot, file size and SHA-256.

---

## 22. Code Modification Summary (Current working area verifiable)

### 22.1 PointRCNN Main Repository tracked Changes

As of this review, tracked diff is 4 documents, 72 insertions / 8 deletions:

1. `lib/config.py`
   - `yaml.load` was replaced by `yaml.safe_load`; Fits new PyYAML and avoids unsafe back-sequencing.
2. `lib/datasets/kitti_dataset.py`
   - NFS robustly read;Up to 20 times at intervals of 0.1 s;Validate bytes, 16-byte/point alignment and short read.
3. `lib/datasets/kitti_rcnn_dataset.py`
   - far points has reached 16384 from the total safe without replacement sample;
   - point count is insufficient and point count is more than point count allows replacement;
   - Record sampler-safe fallback frame and how many times.
4. `lib/rpn/proposal_layer.py`
   - near proposal bucket entered by spatial-region/far-only is empty and no longer triggers the old assertion;
   - The complete proposal budget is allocated to the only non-empty far bucket and follows the established NMS.

### 22.2 OpenPCDet tracked Changes

1. `pcdet/datasets/__init__.py`
   - Argoverse 2 registration to optional import;KITTI-only Environment is no longer mandatory `av2/pyarrow`.
2. `pcdet/datasets/processor/data_processor.py`
   - When point count is insufficient, only is enabled when filling point count with more than already point count `replace=True`, fix Line B boundary error.
3. `pcdet/models/detectors/detector3d_template.py`
   - For trusted projects checkpoint Visible `weights_only=False`, compatible with PyTorch 2.6+ default behavior.
4. `tools/create_kitti_val_infos_only.py`
   - Adds a new KITTI val-only info generator entry.

### 22.3 PU-GCN Changes

`tf_ops/compile.sh`:

- Delete `/usr/local/cuda-10.0` Hard code;
- Support `CUDA_HOME`/`CUDA_PATH`;
- Prefer to conda `libcudart.so`;
- Read TensorFlow Actual compile/link flags;
- Complete CUDA PATH with LD_LIBRARY_PATH.

These are compilation compatible changes, no changes Inception DenseGCN, NodeShuffle or network losses.

### 22.4 PDANS Changes

- `pointnet2/util.py`: DDIM Initial Noise, label, diffusion step tensor Follow condition/net device, remove hard code `.cuda()`.
- `pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py`: used `setdefault` Settings `TORCH_CUDA_ARCH_LIST=7.5`, allows coverage in the external environment.
- The official sampling uses float32;The semi-precision test shows NaN and is not used for the result.

### 22.5 PU-Net Changes and Version boundary

- Python 2  /  Python 3 Compatible, Integer Division, Data Path Parameters, CPU fallback, Custom op ABI/ Compiled Script Restoration.
- `tf_punet_patch_infer.py` Add `unit_sphere_v1`: Centralization, maximum radius normalization, web reasoning, counter scale and inverse parsing.
- The full main table is still `legacy_none` Old adapters;The restored version appears only in 256 frame 2×2.must not Silently replaces or mixes.

### 22.6 Modified Academic boundary

- "Compatibility fixation" allows the code to run, and does not imply proposes a new network.
- sampler-safe and far-only fixes the crash and does not automatically explain AP changes.
- observed-first, patch rules and detector adaptation are part of the protocol / input adaptation changes and must be named arm separately.
- Both the current main warehouse and the external warehouse have not submitted changes to the work area;Recording commit hash is not sufficient for full recurrence, and diff/source snapshot must be saved at the same time.

---

## 23. Environment, configuration, weights and Hash

### 23.1 Actual Importable Environment (2026-09-02 Review)

| Modules | Python | Main framework | Critical dependence |
|---|---:|---|---|
| PointRCNN | 3.9.25 | PyTorch 2.8.0+cu128 | NumPy 2.0.2, SciPy 1.13.1 |
| CenterPoint/OpenPCDet | 3.10.20 | PyTorch 2.7.1+cu128 | NumPy 1.26.4, spconv 2.3.6 |
| PU-Net/PU-GCN | 3.6.8 | TensorFlow 1.13.1 | NumPy 1.19.5 |
| PU-EdgeFormer | 3.6.8 | TensorFlow 1.13.1 | NumPy 1.16.6 |
| PDANS full run | 3.11.15 | PyTorch 2.11.0+cu130 | NumPy 2.4.4, ninja 1.13.0 |
| TULIP | 3.8.20 | PyTorch 1.12.0+cu113 | NumPy 1.24.3, timm 1.0.27, einops 0.8.1 |

The running log record hardware is CentOS Stream 9, Intel Core i5-10505, about 46 GiB RAM, NVIDIA Quadro RTX 4000 8 GiB.This is a history run log, which does not mean that GPU drives a certain amount of online while writing.

### 23.2 detector fixed Configuration

PointRCNN: Car-only Main mission, RPN 16384 point, PointNet++ SA centers `[4096,1024,256,64]`Radius `[0.1,0.5] / [0.5,1.0] / [1.0,2.0] / [2.0,4.0]` m, every scale 16/32 neighbourhood, RPN/RCNN score 0.3, RCNN NMS 0.1, RPN test pre/post NMS 9000/100, RPN NMS 0.8, not using intensity.

CenterPoint: Car/Pedestrian/Cyclist, MeanVFE, VoxelResBackBone8x, HeightCompression, BaseBEVBackbone, CenterHead; Scope `[0,-40,-3,70.4,40,1]`; voxel `0.05×0.05×0.1 m`; 5 points/voxel; 40000 test voxels; test No shuffle;Original weight training 80 epochs.

### 23.3 Key Hash

| Documentation | SHA-256 |
|---|---|
| `tools/cfgs/default.yaml` | `ce6473a5b3451106c9866701701cbce95e71c8daf35b2c0b31e4876ad55a653e` |
| `tools/PointRCNN.pth` | `4631beaa311d7b17b0a934e96e3eacbb4944b4bec2d217801541324a83f8ecff` |
| `centerpoint.yaml` | `445d12f0355ac01232b8837ab001cc08e0742061ff0618200a909af9383c501f` |
| `kitti_dataset.yaml` | `a6c44a7b0f15b94e135946cecbd8b5541906a69d37811705e9884a411e7b1797` |
| `checkpoint_epoch_80.pth` | `c3e68c693ad98606f6c790be9e9aaa8bc723df2b67282f21a02350ec50fa69bd` |

Repository submission: main project `1d0dee91262b970f460135252049112d80259ca0`, OpenPCDet `233f849829b6ac19afb8af8837a0246890908755`, PU-GCN `0d29ee7c819f3cf80c4535c581d701f4e9bbb8b8`, PU-EdgeFormer Recurring Directory `03b8118fa86e32b57a5fb56aea285e17b70bda99`, PDANS `15355e200e8e86658c602951a2751cd9125b2e51`.

---

## 24., which explains the conflict between discipline and discovered documents

### 24.1 AP caliber

- The official main result is KITTI AP_R40.
- The latest full-val runner takes 40 recall samples.
- Part of the old PointRCNN repository evaluator default produces legacy AP_R11; Old  pilot  If no  R40  Evidence, must be preserved.  evaluator  label, cannot  R40  nudity values are directly mixed.

### 24.2 Data Size

- 3769: Official val in full.
- 3712: detector adaptation train.
- 256: fixed pilot/ audit;Can do strong supplementary evidence instead of full-val.
- 64/32/20: Mechanism, holdout, voxel or geometric analysis.
- 20 frame MDE95 is about 11.8 AP, so cannot uses it to confirm a small gain.

### 24.3 TULIP status correction

 The old text used to say, "Not yet discovered  TULIP CenterPoint  Full evaluation” but the current directory contains two completes  PASS  Results:

- Line A CenterPoint Car/Pedestrian/Cyclist Moderate 3D AP_R40: 31.7697 / 15.9778 / 4.5920;
- Line B: 17.6847 / 2.9893 / 0.0798.

The direct evidence is that [TULIP CenterPoint full AP CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/centerpoint_tulip_native_extended_20260729/full_ap_summary.csv). Therefore, the current status should be written as “CenterPoint Line A/B full quantity completed, but TULIP output is range-image native point count, which does not belong to the four methods exact-4N main table”.PointRCNN TULIP uses a 4096 point and a special compatible configuration to close distance-based proposal, and cannot sorts directly with the default 16384 point main table.

### 24.4 The latest available and unwritten boundary

You can write:

- detector adaptation and observed-first, respectively, can perform the test in the recovery section PU-GCN;
- Both 20/20 full-val arm have been completed;All PU-GCN input arm still below corresponding baseline;
- frozen detector is an important factor, but not the only cause;
- paired adapted baseline of Line B provides the cleanest evidence of disability;
- Three rounds of training completed and loss dropped.

cannot writes:

- "3 epochs has been convergence" or "epoch 3 is the best checkpoint";
- "PU-GCN after adaptation has recovery or more than baseline";
- "Line A adapted margin is a pure input effect";
- Extension of PU-GCN-specific adaptation results to a ceiling for all four methods;
- Consider PU-Net * the very low AP of the old adapter as the PU-Net structural capacity;
- Write down a small positive of detector-aware V4/V5 single seed/pilot as a stable full return.

---

## 25. Final Delivery Index and Recommended Reading Order

If only one-time knowledge of the entire KITTI job is required, read in the following order:

1. This general ledger: configuration, history, code changes, main table, ablation, timeline and latest updates.
2. [Chapter 3—4 Detailed body](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_CHAPTERS_3_4_KITTI_DETAILED_ZH.md): research design, environment, data, methodological access, format conversion, detector and indicator formulae.
3. [Chapter 5—6 Detailed body](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH/THESIS_CHAPTERS_5_6_KITTI_DETAILED_ZH.md): Full result, object/distance/geometric / voxel evidence, discussion, limitation and outlook.
4. [The latest PU-GCN 3769 frame Matrix](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/full_val_detector_matrix.md): 2026-09 final detector-adaptation numbers.
5. [The latest protocol with the code](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/protocol_and_code_en.md): 3N/3M How to choose, patch How to extract, 3 epochs What to train, where to enter the code.

Overall result of one sentence:

> Under strict exact-4N, Fair Double Line and Double detector, the traditional general upsampling does not automatically test recovery KITTI 3D; Improvements  patch,  Protection of observations and input-level adaptation to detector can recover significant performances, but up to  20/20  Full validation. PU-GCN  Still below Corresponds to a baseline that indicates that the effective enrichment of the mission must address both real surface support, observation protection, and point / voxel budget and detector distribution appropriate.
