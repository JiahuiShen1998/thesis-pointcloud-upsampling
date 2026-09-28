# Historical Draft of Chapters 5 and 6: KITTI Results, Discussion, and Outlook

> **Scope and citation rules.** This historical draft continues Chapters 1–4 and covers the KITTI material for Chapters 5 and 6. Literature references retain the numbering [1]–[35] from `thesis.pdf`; no unverifiable papers are added. Experimental configurations, CSV files, audit reports, and visualizations [L14]–[L20] are traceable local evidence, not published literature. Results on the full validation set of 3769 frames support the main conclusions. The fixed 256-frame set supports adaptation and object-level audits; the 64-frame and 32-frame subsets support mechanism analysis only. Unless specified otherwise, detection AP is KITTI `AP_R40`, with a 3D IoU threshold of 0.70 for Car. The current KITTI wrapper for PU-Net has a confirmed scale-normalization error; starred results diagnose that pipeline and do not evaluate the PU-Net architecture itself.

## Notation

| Symbol | Meaning |
|---|---|
| \(s\) | KITTI frame identifier |
| \(l\in\{A,B\}\) | Input line: A densifies a complete scan; B recovers a scan after fourfold sparsification |
| \(m\) | Upsampling method: primarily PDANS, PU-GCN, PU-EdgeFormer, or PU-Net* |
| \(d\) | Downstream detector, \(d\in\{\mathrm{PointRCNN},\mathrm{CenterPoint}\}\) |
| \(\mathcal X_s^{(l)}\) | Observed point set for frame \(s\) under input line \(l\) |
| \(\mathcal G_{s,m}^{(l)}\) | Generated points produced by method \(m\) and retained by the common filter |
| \(\mathcal Y_{s,m}^{(l)}\) | Final exact-\(4N\) set: \(N\) observations plus \(3N\) generated points in the main protocol |
| \(T_{d,m}^{(l)}\) | Task metric for detector \(d\), method \(m\), and input line \(l\) |
| \(\Delta T_{d,m}^{(l)}\) | Percentage-point change relative to the real baseline on the same input line |
| \(\tau\) | Geometric proximity threshold: 0.20 m for voxel analysis and 0.25 m for near-surface vehicle analysis |

# 5. Results and Evaluation

This chapter asks whether fourfold output density provides correct geometric evidence for the detector. It proceeds from evaluation definitions to full task results, controlled ablations, point-cloud cases, and mechanism analysis. A single case can explain a failure but cannot replace full-dataset AP; an AP value describes performance but cannot by itself locate the stage at which performance changed.

## 5.1 Overview of Experiments

### 5.1.1 Evaluation object, two input lines and level of evidence

Line A and Line B test different propositions. Given an original KITTI scan \(\mathcal X_s^{\mathrm{orig}}\), Line A directly produces four times as many points:

\[
\mathcal X_s^{(A)}=\mathcal X_s^{\mathrm{orig}},\qquad
\mathcal Y_{s,m}^{(A)}
=\mathcal X_s^{(A)}\cup\mathcal G_{s,m}^{(A)},\qquad
|\mathcal G_{s,m}^{(A)}|=3|\mathcal X_s^{(A)}|.
\tag{5.1}
\]

Line A does not recover deliberately removed observations. An AP gain would indicate that generated points provide useful evidence beyond the original scan. An AP decline suggests that altered sampling, local neighborhoods, or voxel occupancy can dilute the useful signal.

Line B applies fixed downsampling \(D_4\) to retain one quarter of the input, then recovers the original point-count scale:

\[
\mathcal X_s^{(B)}=D_4(\mathcal X_s^{\mathrm{orig}}),\qquad
|\mathcal X_s^{(B)}|\simeq\frac14|\mathcal X_s^{\mathrm{orig}}|,
\tag{5.2}
\]

\[
\mathcal Y_{s,m}^{(B)}
=\mathcal X_s^{(B)}\cup\mathcal G_{s,m}^{(B)},\qquad
|\mathcal Y_{s,m}^{(B)}|=4|\mathcal X_s^{(B)}|.
\tag{5.3}
\]

Line B must be compared with the detector baseline on the same sparse input. It tests whether upsampling recovers task performance lost through sparsification. Averaging Line A and Line B AP into a single score is not appropriate.

Five layers of evidence are used in this chapter.R1 and R2 are full validation set evaluations for the main findings;R3 Observes whether detector adaptation is capable of mitigating area deviations;R4 and R5 Explanatory Mechanism, cannot instead of the official AP.

| Level | Data size | Compare Objects | Main uses | Conclusion Strength |
|---|---:|---|---|---|
| R1 | 3769 frame | PointRCNN; 4 method;Line A/B; E1/E2 | Full Car AP and input budget sensitivity | Main result |
| R2 | 3769  frame  | CenterPoint; 4  Methodology; Line A/B; exact-\(4N\) |  Full 3 categories  AP  Contrast with voxel  |  Main result  |
| R3 | 256 fixed Evaluation frame;3712 Training frame | PU-GCN; PointRCNN/CenterPoint; Before and after. | Check if detector retrains to resolve distribution deviations | Supplementary evidence |
| R4 | 64 Cars frame | patch Range, Generate voxel Precision, Distance Layer | Position input geometric problem | Mechanistic evidence |
| R5 | 256 frame, 527 Moderate Car GT;There are also 32 frame voxel audits | Target IoU, TP/FN conversion, voxel cap | Explain recovery and one example of failure | Diagnostic evidence |

### 5.1.2 AP, IoU and full definition of the margin

 given projection box  \(B_p\)  With Real Box  \(B_g\),  Three-dimensional.

\[
\operatorname{IoU}_{3D}(B_p,B_g)
=\frac{\operatorname{Vol}(B_p\cap B_g)}
{\operatorname{Vol}(B_p\cup B_g)}
=\frac{\operatorname{Vol}(B_p\cap B_g)}
{\operatorname{Vol}(B_p)+\operatorname{Vol}(B_g)-\operatorname{Vol}(B_p\cap B_g)}.
\tag{5.4}
\]

 Trustability threshold  \(q\),  The accuracy rate is recall.

\[
P(q)=\frac{TP(q)}{TP(q)+FP(q)},\qquad
R(q)=\frac{TP(q)}{TP(q)+FN(q)}.
\tag{5.5}
\]

When using KITTI R40 evaluation, calculate the interpolated precision

\[
P_{\mathrm{interp}}(r)
=\max_{\tilde r\ge r}P(\tilde r),
\tag{5.6}
\]

And take an average of 40 recall locations for fixed:

\[
AP_{R40}=\frac1{40}\sum_{k=1}^{40}
P_{\mathrm{interp}}\!\left(\frac{k}{40}\right).
\tag{5.7}
\]

 That's a definition.  KITTI/VOC  Accuracy — recall Evaluation tradition  [1,2,34,35].  This chapter is reported simultaneously  3D AP  with  BEV AP: 3D AP  Requires horizontal location, height, size and common direction; BEV AP  Only the rotation box is evaluated on bird view.  BEV  and  3D  At the same time, the problem is generally not just a high level of return; if the gap widens significantly, additional checks are required  \(z\)  Axes and height estimates.

All “improvements” are based on a single line baseline of zero:

\[
\Delta AP_{d,m,c,h}^{(l)}
=AP_{d,m,c,h}^{(l)}-AP_{d,\mathrm{base},c,h}^{(l)},
\tag{5.8}
\]

 of which  \(c\)  is a category, \(h\in\{\mathrm{Easy},\mathrm{Moderate},\mathrm{Hard}\}\).  Positive values indicate that the baseline is exceeded and negative values indicate degradation.  Line B,  Defines relative original — recovery rate for rare performance gaps:

\[
R_{d,m,c,h}^{(B)}
=\frac{AP_{d,m,c,h}^{(B)}-AP_{d,\mathrm{sparse},c,h}^{(B)}}
{AP_{d,\mathrm{orig},c,h}^{(A)}-AP_{d,\mathrm{sparse},c,h}^{(B)}}.
\tag{5.9}
\]

\(R=1\)  This means that there is a complete recovery of the sizable losses. \(R=0\)  It's the same as sparse baseline. \(R<0\)  This indicates that upsampling was later interpreted as below  sparse baseline. The recovery rate is only explained when the denominator is positive and both uses the same detector and the evaluation is achieved.

### 5.1.3 Integrity check and conclusion discipline

PointRCNN  full validation set  E1/E2  And every experiment that comes out of it.  3769  A projection document; CenterPoint exact-\(4N\)  It's...  10  All group main input through frame, limited coordinates, point count and  evaluator  Integrity check  [L14,L15].  This excludes "the missing frame" as  AP  decline, but does not automatically establish that the adapter is correct. To avoid overgeneralizing from implementation defects, this chapter follows these rules:

1. The margin is calculated only in the same data disaggregation, IoU threshold, evaluator and baseline calibre.
2. 3769 frame results are used for the overall conclusions;The results of 256 frame are used to adapt to the diagnosis of the subject and are not written in full validation set's conclusions.
3. 64 frame patch/ geometric results indicate whether the mechanism exists and is not used to estimate the full data set effect.
4. 32 frame voxel The audit responded only “if the cap is triggered and if the order of entry is likely to have an impact”.
5. The PU-Net* value is retained to demonstrate that the current packaging chain may be seriously failing, but does not draw structured conclusions that are appropriate for other methods.

## 5.2 ModelNet40 Classification Results

This section is entitled " Maintenance of continuity of the original paper catalogue " , but does not repeat the ModelNet40 numeric results.The reason is that the mission requires the KITTI section to be organized and that the current complete and verifiable chain of evidence is focused on LiDAR tests. Chapter 5 Cross-domain discussion only quotes  3, 4  Chapter defined  ModelNet40  By logic, do not make up the missing classification accuracy rate. When the paper is eventually merged, the existing ModelNet40 5.2 text should be placed here and maintained [1]–[35] The reference number unchanged.

 This boundary per se influences the wording of the conclusion: this chapter can prove that the current object class upsampler —KITTI  scene adapter — detector's behavior is based on cannot only  KITTI  Negative results judge that a network is regulating  CAD  On the surface, also cannot reverses  ModelNet40  classification Improve replacement reality  LiDAR  Test the evidence.

## 5.3 KITTI PointRCNN Detection Results

PointRCNN  3D directly from point cloud  proposals [11]As a result, the point-based sampling budget, local grouping and pseudo-point ratios are more sensitive.This section first reports on the results of full validation set and then gives detector adaptation, observational reservations and object-level diagnosis.The absolute AP of the different experiments is authorized to weigh and input the difference in the effect of the difference, which should be read across the tables by comparing the respective line differentials and not by directly comparing the nudity values of the different rows.

### 5.3.1  full validation set  E1:  Keeps the observations.  exact-\(4N\)  Result

E1  The output structure is:  \(N\)  Observatories and  \(3N\)  A generated points. All input points are kept point by point, therefore  E1  Specially to test whether “retention of the true point is sufficient to avoid degradation of performance”.  5.1  Give  3769  frame  Car 3D AP\(_{R40}\).

** Table  5.1　PointRCNN  full validation set  E1 Car 3D AP\(_{R40}\) (%)**

| Input Line | Methodology | Easy | Moderate | Hard | Moderate margin relative to the same line baseline |
|---|---|---:|---:|---:|---:|
| A | Original baseline (E2 reference) | 92.273 | 82.255 | 77.945 | 0.000 |
| A | PDANS | 83.014 | **64.515** | 59.668 | -17.741 |
| A | PU-GCN | 78.766 | 56.961 | 52.193 | -25.294 |
| A | PU-EdgeFormer | 64.258 | 42.981 | 38.259 | -39.274 |
| A | PU-Net* | 14.414 | 9.717 | 9.010 | -72.538 |
| B | 1/4 sparse baseline (E2 reference) | 85.177 | 65.746 | 61.382 | 0.000 |
| B | PDANS | 65.062 | **45.126** | 39.240 | -20.620 |
| B | PU-GCN | 45.378 | 29.687 | 25.290 | -36.059 |
| B | PU-EdgeFormer | 34.010 | 20.721 | 17.747 | -45.025 |
| B | PU-Net* | 12.522 | 8.878 | 8.067 | -56.868 |

Four immediate and visible conclusions are set out below.

First, all Line A methods are below original scan baseline.The original scan already contains real observations available at the time of the evaluation, generated points no "recovery " ;The best PDANS still drops 17.741 Moderate AP.This negates the assumption that "just a little bit more, and a little detector will be done in one way or another."

 Second of all, Line B  The best of them.  PDANS  It's a little thinner than that.  baseline  Low  20.620 AP;  Type(s) 5.9) Calculates that its recovery rate is negative. This means that the output returns to approximately the original point count, which is not equal to the original scan information for recovery. point count is only bound by constant  \(|\mathcal Y|\),  no binding  \(\mathcal G\)  Whether it falls on the missing real surface.

Thirdly, the two lines are in exactly the same order: PDANS > PU-GCN > PU-EdgeFormer > PU-Net*. This sorting will later correspond to the actual precision of generating voxel.It is more institutional than “how many APs have been lost by a certain method”, because both starting points and the same detector are in the same order.

Fourth, both Hard and Moderate have significantly declined, not only in a small number of Easy cases.At the same time, the very low PU-Net* AP was affected by the identified adapter error: rice patch `bradius=1.0` Directly sent to a network using normalization object training, no for the same centralization/ scale normalization and reverse transformation [L17]. So it proves that the package protocol can destroy the test results, not PU-Net. [4] . The structure performance cap.

### 5.3.2 E2 fixed 16384 Point: Excludes "PointRCNN just not accept more points"

E2 executes a unified field of view filter for all conditions, 0.1 m voxel points and layers of distance sampling, and ultimately saves 16384 points.If E1's main problem is only PointRCNN fixed point budget, E2 should be systematic recovery AP.Table 5.2 shows that this expectation did not occur.

** Table  5.2　PointRCNN E1  with  E2  It's...  Moderate 3D AP\(_{R40}\)  Contrast %)**

|  Input Line  |  Methodology  | E1: \(N+3N\) | E2:  Harmonization  16384 | E2−E1 |
|---|---|---:|---:|---:|
| A | PDANS | 64.515 | 67.223 | +2.709 |
| A | PU-GCN | 56.961 | 58.276 | +1.314 |
| A | PU-EdgeFormer | 42.981 | 44.960 | +1.979 |
| A | PU-Net* | 9.717 | 8.380 | -1.337 |
| B | PDANS | 45.126 | 44.075 | -1.051 |
| B | PU-GCN | 29.687 | 28.680 | -1.006 |
| B | PU-EdgeFormer | 20.721 | 20.569 | -0.151 |
| B | PU-Net* | 8.878 | 8.781 | -0.097 |

(a) Line A, which has three suitable methods, only recovery 1.3–2.7 AP, which is much smaller than their 17.7–39.3 AP gap relative to baseline;Line B is all no improvements. Directer counterevidence from stand-alone voxel: E2  Original  baseline  It's...  0.1 m  Independent voxel median  11116, Moderate AP  Yes.  82.255; Line B PU-GCN has 13660 independent voxel, AP, but only 28.680;PU-EdgeFormer has 14321 independent voxel, AP is 20.569.More standalone locations no to higher AP, indicating whether the decisive variable is the correct location, not whether the dot or voxel is sufficient.

### 5.3.3 PU-GCN: Field offset can be mitigated, but cannot eliminated

 To test whether frozen detector is expanding the training field differences and using them further  3712  Trained frame to fully fit detector and in fixed  256  frame on evaluation. Table 5.3 gives both generated-only and observed-first.The "generated-only" here is the direct tectonic detection input using upsampling output;observed-first gives priority to the retention of real observations in the output order, so that the follow-up limited budget operation consumes first observations.

** Table  5.3　PointRCNN  After full training, fixed  256  frame  Car  Outcome (%) %)**

| Line | Enter | Easy 3D | Moderate 3D | Hard 3D | Moderate BEV | Moderate 3D margin relative to baseline |
|---|---|---:|---:|---:|---:|---:|
| A | baseline | 89.701 | 79.191 | 77.882 | 87.513 | 0.000 |
| A | PU-GCN generated-only | 83.774 | 67.235 | 59.934 | 76.924 | -11.957 |
| A | PU-GCN observed-first | 86.372 | **67.791** | 65.830 | 78.576 | -11.400 |
| B | baseline | 86.899 | 67.249 | 59.831 | 77.679 | 0.000 |
| B | PU-GCN generated-only | 67.484 | 47.020 | 41.123 | 57.755 | -20.228 |
| B | PU-GCN observed-first | 76.757 | **56.244** | 50.067 | 66.614 | -11.004 |

![Figure 5.1 PointRCNN PU-GCN baseline, generated-only and observed-first AP.](figures/fig5_01_pointrcnn_pugcn_ap.pdf)

observed-first of Line B vs. generated-only recovery 9.224 Moderate 3D AP, indicating that real points do compete with generated points under a limited input budget;Line A is only recovery 0.556 AP, indicating that the order of observations is not the main source of Line A losses. More importantly, two.  observed-first  The results are still comparable.  baseline  Low  11.400  and  11.004 AP. Therefore, “reserve and give priority to the point of truth” is a necessary engineering constraint, but not a sufficient condition for geometric's reliability.

The result also avoids another misreading: fit training does teach some input distribution changes, but no turns the negative margin to zero.As a result, the remaining gap cannot is attributable entirely to detector, which has never seen any additional input;At least some of the losses resulted from statistical deviations that generated geometric, patch or could not be absorbed by limited training data.

### 5.3.4 64 frame re-training screening: benefits coexist with baseline degradation

The earlier 64 frame fine-tuning experiment provided a directional contrast. [L20].  In the same  256  frame on, PU-GCN  It's...  Line A Moderate AP  From  60.266  Increase to  64.441 (+4.175), Line B  From  32.920  Increase to  37.515 (+4.595); PDANS  It's...  Line B  From  43.007  Increase to  47.433 (+4.426). However, the same fine-tuning has reduced Line A baseline by 1.148 AP and Line B baseline by 3.705 AP;All upsampling conditions are still below fine-tuned baseline.

So this set of results, cannot, is written as "Retraining solves the problem."It can only support two narrower judgments: first, detector has a learning space for generated points distribution;Second, the fine-tuning of the small sample itself would introduce differentials and baseline degradation, which would have to be controlled through the full training set fit, independent validation set and the same condition baseline.Because of the 64 frame results in both directions, the final conclusion was based on 3712 frame.

### 5.3.5 Object Level IoU and Distance Layer

 At fixed  256  frame, sifted  527  Matches  Moderate  Conditional  Car GT; The distance is 0–20 m: 170, 20–40 m: 272, 40 m and more: 85.The object's diagnosis matches the greed of the same category, rotation 3D IoU, threshold being 0.70.It is used to explain the TP/FN migration and does not replace the official AP with credit.

PointRCNN Line B baseline, generated-only and observed-first match 386,258 and 320, respectively, with the corresponding diagnosis calling back 73.24%, 48.96% and 60.72%.generated-only relative baseline Add 134 `TP→FN`; observed-first, of which 80 `FN→TP`However, there are still 78 baseline TP which becomes FN under observed-first.This trend suggests that the overall improvement of observed-first is not a small increase in all targets, but that some recovery targets still undermine others.

Distance further explains the source of the gap.Line B baseline was summoned to 0–20, 20–40, 40+ m, 97.65%, 71.32% and 30.59%, respectively;observed-first PU-GCN is 96.47%, 53.31% and 12.94% respectively.The close target was almost maintained, while the median distance was reduced by 18.01 percentage points and the distance by 17.65 percentage points.The generated points risk is therefore not evenly distributed: the less the observation is, the stronger the cover, the harder it is for local patch to maintain a single surface, and the more likely the new point is to change the frame that was close to threshold.

### 5.3.6 point cloud scene and local recovery case

 Figure  5.9  Show the same.  KITTI  frame  000104  Four point clouds: Original scan  121994  Point, four times downsampling  30498  Dot. Line B PDANS exact-\(4N\) 121992  Point and  Line B PU-GCN observed-first 121992  Point. Only superimpose in the chart  GT  Box, without superstitioning the method projection, intended to provide a distribution of observation points rather than replacing the overall evaluation with a selected projection box.

 The whole picture shows two questions. First of all, exact-\(4N\)  The line check does visualize the whole density  recovery to near the original input; and secondly, remote and object  boundary space support and no automatically recovery. Large-area road background can absorb a large number of points, making “the whole frame look more dense”, while the vehicle may still lack the correct surface inside or near its target.

Figure 5.10 gives a recovery case.frame 000440's Car GT 1 from LiDAR 31.2 m.Line B baseline has 26 points in GT, receiving TP, IoU 0.867;generated-only has increased to 73 points within GT, but no produces an effective match (FN, best IoU 0);observed-first has 81 frame points and recovery TP, IoU 0.806.

 This case supports “observation reservations can save some of the targets”, but gives a strict boundary: 81  It's a point-to-point.  IoU  Still below  26  It's a real thinner one.  baseline IoU. Add point count to the 0.70 threshold and no recovery to the baseline positioning accuracy.It is consistent with the overall results of AP recovery in table 5.3, but still below baseline.

Figure 5.11 is a complementary case of failure. frame  004846  It's...  Car GT 2  Distance  LiDAR 38.2 m, baseline  only  20  I got a little bit of it in the box.  IoU 0.880; generated-only and observed-first have 28 and 43, respectively, but both are FN.

 So, count in the box.  \(n_{\mathrm{in}}\)  cannot is a single upsampling quality indicator. detector relies on the organization of visible surfaces, edges and local neighbourhood. A more appropriate explanation variable should include at least point to reference surfaces, additional voxel, target boundary shell ratio and observations / Generate the source, not the " GT  There are more points in the box, the direct equivalent is "more information about the target".

## 5.4 KITTI CenterPoint Detection Results

CenterPoint [12] Different from PointRCNN's input response.It starts by placing the dot on fixed grid voxel and then predicts the target centre on the profile;The repeat points in the same voxel may be aggregated, while the a small number of pseudo-point crossing voxel boundary will create a new BEV activation.The purpose of the second detector is not to find an evaluator that is more “contributive” to the outcome, but to test whether the same upsampling geometric is in the same direction as the voxel expression.

### 5.4.1 Original full validation set with sparse baseline

The CenterPoint main experiment uses the full 3769 frame validation set, frozen official detector, the original voxel chemical and the same score/NMS configuration.All 10 input groups checked for completeness [L15]. The baseline of the original scan and the four-fold thin scan is as table 5.4.

** Table  5.4　CenterPoint  full validation set baseline  3D AP\(_{R40}\) (%)**

| Enter | Category | Easy | Moderate | Hard | Moderate Changes due to thinning |
|---|---|---:|---:|---:|---:|
| Original | Car | 88.391 | 79.277 | 76.737 | — |
| 1/4 sparse | Car | 81.400 | 64.598 | 59.970 | -14.679 |
| Original | Pedestrian | 54.037 | 50.653 | 46.252 | — |
| 1/4 sparse | Pedestrian | 27.196 | 24.569 | 21.673 | -26.084 |
| Original | Cyclist | 79.244 | 64.605 | 60.990 | — |
| 1/4 sparse | Cyclist | 25.805 | 15.458 | 14.492 | -49.148 |

The impact of thinning on the three categories of objectives is significantly different.Car Moderate dropped 14.679 AP while Pedestrian and Cyclist decreased 26.084 and 49.148 AP, respectively.Small targets have less effective echoes, and the removal of points at the same scale would destroy local shapes and central heat map evidence more quickly.Thus, the average of the categories hides the most important phenomena;Follow-up must be reported by category.

### 5.4.2 exact-\(4N\) observed-first  Main results for the three categories

Table 5.5 gives Moderate 3D AP for observed-first, master protocol.In brackets is the difference between the real baseline of the input line, Line A for original baseline and Line B for 1/4 sparse baseline.

** Table  5.5　CenterPoint exact-\(4N\) observed-first: Moderate 3D AP\(_{R40}\) (%)**

| Line | Methodology | Car | Pedestrian | Cyclist |
|---|---|---:|---:|---:|
| A | PDANS | **64.466** (-14.812) | **45.251** (-5.403) | **46.103** (-18.503) |
| A | PU-GCN | 59.932 (-19.346) | 44.504 (-6.149) | 45.597 (-19.009) |
| A | PU-EdgeFormer | 47.248 (-32.030) | 37.085 (-13.569) | 35.822 (-28.784) |
| A | PU-Net* | 14.988 (-64.290) | 21.687 (-28.967) | 24.396 (-40.210) |
| B | PDANS | **46.751** (-17.847) | **28.201** (+3.632) | 15.519 (+0.061) |
| B | PU-GCN | 39.070 (-25.528) | 24.961 (+0.392) | **16.846** (+1.388) |
| B | PU-EdgeFormer | 24.748 (-39.850) | 15.596 (-8.973) | 7.037 (-8.420) |
| B | PU-Net* | 11.718 (-52.880) | 11.916 (-12.653) | 5.017 (-10.440) |

![Figure 5.2 CenterPoint exact-4N observed-first Moderate 3D AP margin relative to the line baseline;The brackets are absolute AP.](figures/fig5_02_centerpoint_delta_heatmap.pdf)

Line A's 12 units are all negative, indicating that the current four scenarios do not provide net mission income when complete scanning already exists.PDANS has the smallest loss for Pedestrian (-5.403), but it is not an improvement.This result is consistent with the direction of PointRCNN Line A: the new point does not allow only to become redundant, but also changes the characteristics of the detection through the new voxel, polymerization and training distribution bias.

Line B shows three small positives: PDANS against Pedestrian +3.632 and Cyclist +0.061;PU-GCN for Pedestrian +0.392 and Cyclist +1.388.They prove that “sparse-input recovery is not entirely impossible in some small categories”.These improvements, however, need to be explained in the diluted gap.By type (5.9), PDANS only recovery Pedestrian gap

\[
R_{\mathrm{PDANS,Ped}}^{(B)}
=\frac{28.201-24.569}{50.653-24.569}
=0.1393,
\tag{5.10}
\]

About 13.9%;The recovery rate for Cyclist is approximately 0.12%.PU-GCN  Yeah.  Pedestrian  with  Cyclist  The recovery rate is about the same  1.5%  with  2.8%. At the same time, Car has dropped 17.847 and 25.528 AP, respectively.This result therefore supports, at best, "the local recovery, which is relevant to the category, and cannot, which is written as an overall test for recovery.

### 5.4.3 reconstructed E1 and observed-first: Controlled ablation

CenterPoint reserves a maximum of 5 points per voxel and a maximum of 40000 non-empty voxel per frame.Sets the range to be valid

\[
\Omega=[0,70.4)\times[-40,40)\times[-3,1),
\tag{5.11}
\]

 voxel Size  \((v_x,v_y,v_z)=(0.05,0.05,0.10)\) m.  A little bit.  \(\mathbf p_i=(x_i,y_i,z_i)\in\Omega\)  Discrepancies index as

\[
\mathbf q_i=\left(
\left\lfloor\frac{x_i-x_{\min}}{v_x}\right\rfloor,
\left\lfloor\frac{y_i-y_{\min}}{v_y}\right\rfloor,
\left\lfloor\frac{z_i-z_{\min}}{v_z}\right\rfloor
\right).
\tag{5.12}
\]

 When input generated non-empty voxel collection  \(\mathcal V_s\)  Satisfied  \(|\mathcal V_s|>K_{\max}=40000\)  , the limited voxel budget means that part of voxel will not enter the network; when a voxel  point count is greater  \(T_{\max}=5\)  , the order of the points may also change the subset that has been retained. observed-first  Prior to generated points, the role of the site can be written as a priority:

\[
\mathcal S_v^{\mathrm{obs}}
=\operatorname{First}_{T_{\max}}
\bigl([\mathcal X_v;\mathcal G_v]\bigr),
\qquad
\mathcal S_v^{\mathrm{ctrl}}
=\operatorname{First}_{T_{\max}}
\bigl([\mathcal G_v;\mathcal X_v]\bigr).
\tag{5.13}
\]

 Table  5.6  Summary  observed-first  Relative Matching  control  Category 3  Moderate AP  Average change. This compares exactly the same set of points, only, and thus is closer to the cause and effect of ablation than "upsampling vs baseline".

**Table 5.6 CenterPoint observed-first Average Moderate AP change relative to control**

| Line | PDANS | PU-GCN | PU-EdgeFormer | PU-Net* |
|---|---:|---:|---:|---:|
| A | +0.087 | +0.894 | +0.792 | +5.878 |
| B | +0.007 | +0.010 | +0.002 | -0.001 |

In the 32 frame voxel audit, the original input 0/32 frame reached the 40000 voxel limit;Line A  It's...  PDANS, PU-GCN, PU-EdgeFormer, PU-Net*  Individually.  6/32, 29/32, 29/32, 32/32; Line B's four methods are 0/32.The sequenced gain clearly corresponds to the cap trigger: Line A has a visible improvement, Line B almost unchanged.This supports the restrictive conclusion that observed-first has limited observation coverage under voxel;no triggers voxel upper limit Line B, which cannot corrects generated points coordinate error.

PU-Net * In Line A + 5.878 AP cannot explains the superiority of the method.On the contrary, its error geometric triggers voxel cap for almost every frame, and the sequence of intervention only reduces the worst input damage;Its absolute AP is still far from below baseline.

### 5.4.4 Matched PU-GCN and detector dependent

 fixed  256  frame, all training set  CenterPoint  The results are shown in the table.  5.7. Like PointRCNN, Line A baseline uses the official weight, Line B baseline and upsampling conditions use the full ratio weight [L14].

** Table  5.7　CenterPoint  After full training, fixed  256  frame  Car  Outcome (%) %)**

| Line | Enter | Easy 3D | Moderate 3D | Hard 3D | Moderate BEV | Moderate margin relative to baseline |
|---|---|---:|---:|---:|---:|---:|
| A | baseline | 93.635 | 79.837 | 77.357 | 88.962 | 0.000 |
| A | PU-GCN observed-first | 89.439 | 73.947 | 72.663 | 83.628 | -5.890 |
| B | baseline | 83.595 | 65.550 | 61.865 | 78.525 | 0.000 |
| B | PU-GCN observed-first | 80.731 | 60.698 | 56.695 | 73.508 | -4.852 |

 Same  256  frame on, PointRCNN  It's...  Line A/B  The margin is  -11.400/-11.004 AP, CenterPoint  Yes.  -5.890/-4.852 AP.

![Figure 5.3 The observed-first PU-GCN is the Car Moderate AP margin on two detector.](figures/fig5_03_detector_dependency.pdf)

Define detector Sensitivity difference

\[
\Gamma_m^{(l)}
=\Delta AP_{\mathrm{PointRCNN},m}^{(l)}
-\Delta AP_{\mathrm{CenterPoint},m}^{(l)}.
\tag{5.14}
\]

PU-GCN  Yes.  Line A  with  Line B  It's...  \(\Gamma\)  Individually.  -5.510  and  -6.152 AP,  Organisation  PointRCNN  The degradation of the world's oceans and seas  CenterPoint  More  5.5–6.2  Percentage points. The direction is the same as the two lines, but supports the interpretation of "voxel Convergence for Part Point Disturbing."  CenterPoint  It's still visible to itself, below.  baseline,  So the detector structure is just a magnification or buffer factor, not the root cause of the co-negative result.

Target-level audits give the same direction.Line B  It's...  527  individual  Car GT  I don't know. CenterPoint baseline  with  PU-GCN observed-first  The diagnosis is called back to...  77.61%  and  74.38%,  Bad.  3.23  (b) Percentage points; PointRCNN should be 73.24% and 60.72%, or 12.52 percentage points. Especially.  20–40 m, CenterPoint  From  79.78%  Down to  74.26%, PointRCNN  From  71.32%  Down to  53.31%. This indicates that geometric is more destructive than proposal, but does not mean voxel detector is immune to false points.

## 5.5 Discussion of Results

### 5.5.1 Effect of Upsampling

 Crossing two detector and two input lines, the most robust conclusion is: ** Current  strict-\(4N\)  The point count increase is not in itself a sufficient condition for the increase in mission performance. ** Line A  Yes.  PointRCNN  and  CenterPoint  Systemic negative margins are observed in all cases; Line B  Just...  CenterPoint  Some small categories appear to have limited positive values, and no  recovery is mostly raw — Scary gap.

![Diagram 5.4 Generates voxel Real Precision in relation to PointRCNN AP method sorting.](figures/fig5_04_geometry_vs_ap.pdf)

Distinguishing the distinction between “density” and “effective evidence” for formalisation so that the dosage is

\[
\delta_N=\frac{|\mathcal Y|-|\mathcal X|}{|\mathcal X|}=3,
\tag{5.15}
\]

 And the return on the mission is in the form ( 5.8) It's...  \(\Delta AP\).  All  strict-\(4N\)  The method is the same.  \(\delta_N\),  But...  AP  From  PDANS  Present.  PU-Net*  Over a dozen percentage points. This is a direct indication.  \(\delta_N\)  cannot. Reasonable intermediate variable is a valid geometric dose:

\[
\delta_{\mathrm{eff}}(\tau)
=\frac{1}{|\mathcal G|}
\sum_{\mathbf g\in\mathcal G}
\mathbb I\!\left[min_{\mathbf r\in\mathcal R}
\|\mathbf g-\mathbf r\|_2\le\tau\right],
\tag{5.16}
\]

 of which  \(\mathcal R\)  is the complete-scan reference for the same frame. This quantity measures how many generated points lie near the reference surface. It is calculated only for offline diagnosis and is not used for generation or detection.

For Line B, recovery capabilities also require generated points to cover the diluted reference area.Reference coverage defined as

\[
C_{\tau}(\mathcal Y,\mathcal R)
=\frac{1}{|\mathcal R|}
\sum_{\mathbf r\in\mathcal R}
\mathbb I\!\left[min_{\mathbf y\in\mathcal Y}
\|\mathbf r-\mathbf y\|_2\le\tau\right].
\tag{5.17}
\]

 High  \(\delta_{\mathrm{eff}}\)  It doesn't have to be high.  \(C_\tau\):  The method may repeat points over and over again in the vicinity of observed surfaces without covering the missing areas. Conversely, increasing coverage alone can also create a large number of false points. The effective upsampling requires a balance between proximity to surfaces, coverage of missing areas and control error support.

### 5.5.2 Transfer from Object-Level Data to LiDAR Scenes

PU-Net [4], PU-GCN [5], PU-EdgeFormer [6] and PDANS [7] The core design revolves around object level or local surface point set.KITTI [1,2] It's a LiDAR scan with rice, whole scene, band strength, background and distance decay.The difference can be written as a joint distribution deviation:

\[
p_{\mathrm{train}}(\mathbf x,\mathcal N,\rho,I,\kappa)
\ne
p_{\mathrm{KITTI}}(\mathbf x,\mathcal N,\rho,I,\kappa),
\tag{5.18}
\]

 of which  \(\mathbf x\)  It's the coordinates. \(\mathcal N\)  It's a local neighbourhood pounce. \(\rho\)  It's sampling  density. \(I\)  It's a reflection strength. \(\kappa\)  Appearance curvature / boundary. The local deviation is not a problem of a target that can be completely eliminated by “Universal normalization”, but a common change between scale, neighbourhood and sampling mechanisms and attributes.

The clearest transport failure of this experiment comes from the common patch extractor.It quantified the full frame coordinates to 32 thick bin, press `(x-bin, y-bin, z, index)` Sorted and slashed to 2048 points.The continuous block is not kNN or ball query. Yeah.  64  A car with frame. Line A/Line B  It's...  patch  Internal  p90  Median Radius  3.16/10.02 m,  Maximum radius median is  4.79/19.45 m, XY  Middle value of diagonal  8.26/30.46 m, XY  Diagonal  p90  Achieved  77.12/124.10 m.

![Fig. 5.5 Common 2048 Point patch Ripper Space Range;Line B's typical patch is no longer local.](figures/fig5_05_patch_locality.pdf)

A wide 124 m patch may contain both roads, vehicles, buildings and multiple unaccompanied targets.If the whole block is centralized and scaled to a unit ball, the network will compress the non-continuous scene to a "object" and then press object to preface generated points.Line B is more likely to fail than Line A because of the thinning of the dots, which are crossing a larger space to fill 2048 points.This mechanism explains why object-level models cannot be transferred unconditionally to real scenes, even when they achieve good geometric metrics on normalized data.

The strength attribute forms the second layer of deviation.The current unified policy is to copy every generated points intensity from the most recent input point.Set Recent Input Index As

\[
j^*(\mathbf g)=\arg\min_j\|\mathbf g-\mathbf x_j\|_2,
\qquad I(\mathbf g)=I(\mathbf x_{j^*(\mathbf g)}).
\tag{5.19}
\]

The strategy ensures that the methods are consistent and that label is not used, but that the same measured strength is copied to multiple points that have been moved to create a platform of less intense strength in the training distribution.It is not a direct cause of the method ' s ranking, as all methods use the same strategy;However, it may result in a common detector distribution deviation, which should in the future shield ablation with uniform local plug-in values or strength.

### 5.5.3 Relationship Between Geometry and Detection

64 frame geometric audit uses original full scan as a read-only reference.Generating voxel

\[
P_{\mathrm{vox},\tau}
=\frac{|\mathcal V_{\tau}(\mathcal G)\cap\mathcal V_{\tau}(\mathcal R)|}
{|\mathcal V_{\tau}(\mathcal G)|},
\tag{5.20}
\]

E1 Additional voxel ratio defined as

\[
E_{\mathrm{vox},\tau}
=\frac{|\mathcal V_{\tau}(\mathcal Y)\setminus\mathcal V_{\tau}(\mathcal R)|}
{|\mathcal V_{\tau}(\mathcal Y)|},
\tag{5.21}
\]

 of which  \(\mathcal V_{\tau}\)  It means it's long.  \(\tau=0.20\) m  Separated from the voxel collection. Line A  It's...  PDANS, PU-GCN, PU-EdgeFormer, PU-Net*  Generate voxel Precision in order  48.83%, 45.48%, 36.32%, 23.11%,  Corresponding  PointRCNN E1 Moderate AP  Yes.  64.51, 56.96, 42.98, 9.72; Line B  Precision in order  50.68%, 47.53%, 38.10%, 34.17%, AP  Yes.  45.13, 29.69, 20.72, 8.88.

 Four methods in two lines.  Spearman  It's all connected.  \(\rho_s=1.00\).  And when there are no parallels,

\[
\rho_s
=1-\frac{6\sum_{m=1}^{M}d_m^2}{M(M^2-1)},
\tag{5.22}
\]

 of which  \(d_m\)  It's a way.  \(m\)  At geometric Precision and  AP  It's the difference in the ranking. \(M=4\).  All in this experiment  \(d_m=0\).  It's not a big sample of outstandingness, it's does not establish per year.  1%  geometric will result in fixed  AP  increment; it indicates that the current inter-method sequence of tasks is fully consistent with geometric 's authenticity and with  E1/E2  point count controls co-support "location quality is more critical than the nominal density".

Line A  Original  baseline  It's true, voxel.  100%.  There's no way to raise it when it's over, but there is.  45.3%–62.5%  It's...  E1  Occupancy voxel is not supported by original scans. Line B baseline  voxel  48.5%;  Add  \(3N\)  generated points was raised only to  57.7%–60.9%,  At the same time.  41.7%–54.4%  Additional voxel. In other words, only part of the new capacity covers the deleted reference position, and a large amount of it is used for duplicate or pseudo-structures.

The distance stratum further shows that the "overlay increase " can be separated from " position reliability " .The Line B baseline vehicle reference surfaces in 0–20, 20–40 and 40–70.4 m are covered by 92.04%, 65.51% and 41.89%, respectively.PDANS E1  Increase to  96.15%, 76.10%, 56.62%,  Its generated points near reference surface ratio remains  98.24%, 88.52%, 80.80%; PU-GCN covers 96.91%, 77.54%, 50.00%, with a near-surface ratio down to 95.19%, 77.11%, 50.00%;PU-Net * Remotely over 50.93%, but only 22.22% on the near surface.

![Figure: Distance layer of 5.6 Line B: ratio of reference surface cover to the near surface of generated points.](figures/fig5_06_distance_geometry.pdf)

This gives a more precise explanation than a “distant point”: the distance upsampling may increase some of the overlay counts, while placing most generated points in a location that is not supported by a real scan;To the detector, these points are high-confidence geometric noise rather than recovered missing observations.Future evaluations should not only report Chamfer Distance or coverage, but also precision-like and recall-like geometric.

### 5.5.4 Detector Dependency

PointRCNN and CenterPoint are down, indicating the root causes before detector;CenterPoint has a smaller reduction, indicating that the input indicates that the loss will be adjusted.You can decompose the final mission.

\[
\Delta T_{d,m}^{(l)}
=\underbrace{\alpha_d\,\Delta G_m^{(l)}}_{\text{generated geometry}}
+\underbrace{\beta_d\,\Delta A_{d,m}^{(l)}}_{\text{input adaptation/budget}}
+\underbrace{\gamma_d\,\Delta Q_{d,m}^{(l)}}_{\text{detector distribution response}}
+\varepsilon_{d,m}^{(l)},
\tag{5.23}
\]

 of which  \(\Delta G\)  This is the first time that the government has been able to provide an overview of real surface support, additional voxel and the target boundary pollution. \(\Delta A\)  General point count / voxel Upper limit and sequence, \(\Delta Q\)  Summarizes the extent to which weights are adapted to the distribution of new inputs. The formula is an explanatory framework and is not a linear cause-and-effect model developed from existing samples.

 The existing ablation provides each of them with evidence of direction: E1/E2  Change only  PointRCNN  Enter budget, recovery limited, description  \(\Delta A\)  Not all; CenterPoint observed-first  I'm on it.  40000  voxel cap improvement, almost unchanged when not reached, confirmed  \(\Delta A\)  no is eliminated, described  \(\Delta Q\)  mitigateable but cannot overall; geometric precision and  AP  It's the same ranking. Support.  \(\Delta G\)  Common ownership.

Thus, cannot summarizes the results as “PointRCNN is not suitable for upsampling” or “CenterPoint can solve upsampling”.More precisely, the conclusion is that both are damaged by the current generation of geometric;PointRCNN  It's more sensitive to point errors. CenterPoint  The voxel condensed part of the disturbance, but the new voxel and voxel budgets still spread errors.

![Figure 5.7 Enter order gain and 40000 voxel upper limit hit rate.](figures/fig5_07_observed_first_voxel_budget.pdf)

### 5.5.5 Object-Level and Qualitative Evidence

The mission margin, geometric statistics and detector mechanisms ultimately need to be checked back on the same actual target.Chart 5.8 gives the total 527 Moderate Car GT IoU distributions and distance layers;Figure 5.9 gives the entire scene under the same protocol point cloud;The figures 5.10 and 5.11 show a target that was added by observed-first recovery and a point count that still failed.Four maps in turn provide evidence of the overall distribution, the scenario density, the positive and negative transformations, and avoid only displaying success stories.

![Figure 5.8 fixed 256 frame object IoU cumulative distribution and range recall;The figure is a diagnosis and does not replace the official AP.](figures/fig5_08_object_iou_distance_audit.pdf)

![ Figure  5.9　 frame  000104  The whole scene is looking at point cloud: raw, thin, PDANS  and  PU-GCN observed-first; The dotted line frame is GT.](figures/fig5_09_pointcloud_scene_bev.pdf)

![ Figure  5.10　observed-first  recovery Case: A combination of the real point and generated points is the recovery box. ](figures/fig5_10_pointcloud_recovery_case.pdf)

![Figure 5.11 Residual Failed: point count has increased in the box, but the spatial distribution of points has not produced available test evidence.](figures/fig5_11_pointcloud_failure_case.pdf)

The contrast between 5.10 and 5.11 is the visual expression of the findings of this study boundary: observed-first is capable of enabling recovery to target certain targets that have been damaged by generated-only, but the "More Boxes" do not guarantee predictions or the return of positioning accuracy to baseline.The overall conclusion is still determined by the conversion of AP and 527 targets.

### 5.5.6 Extension Parallel Contrast, Parameters and Parallel Evidence

The foregoing analysis has led to the conclusion that, if only a few AP tables and two cases were retained, it would not be sufficient to answer the question “what exactly was done, whether the parameters were consistent and whether there were multiple layers of evidence for the decline in performance”.This section thus re-introduces the same chain of experiments to a verifiable parallel.The extension does not introduce a new test calibre and does not alter the above-mentioned conclusions;It converts the completed operational results into five types of evidence: complete difficulty versus evaluation space; operational integrity versus calculation; three categories versus gap recovery; geometric — task against target versus distance stratum.

#### 5.5.6.1 fixed experimental parameters and implemented protocol

Table 5.8 summarizes the key parameters shared by all the charts in this chapter.In particular, a distinction is drawn between " generator parameters " " detector " and " read-only analysis threshold " .For example, 0.20 m voxel is used only for geometric audits and does not write back point cloud or help with candidate selection;0.25 m Near Surface threshold is used only to explain whether a vehicle generated points is close to a reference scan and no is involved in AP optimization.This avoids the misuse of label or test set for ex post-diagnosis using reference scans.

**Table 5.8 KITTI Main Experiment, Fitness Experiment and Diagnosis fixed Parameters**

| Modules | Parameters | fixed value | Role and boundary |
|---|---|---|---|
| Data disaggregation | KITTI training/certification | 3712 / 3769 frame | 3769 frame is used for full master results;3712 frame only for detector adaptation |
| Input Line | Line A | Full Original Scan | Check that there's a full view of the marginal increase. |
|  Input Line  | Line B |  fixed One quarter downsampling  \(D_4\) |  Check if the thinness loss can be caused by recovery  |
| Current patch adapter | Every patch point count | 2048 | Four approaches are shared;Completing the current rule when insufficient |
| Current patch adapter | Crude Space bin Number | 32 | Scratch blocks after they are sorted according to bin;Audited as non-real local |
|  upsampling Output  |  Number protocol  | \(N\) observed \(+3N\) generated \(=4N\) |  Original observations kept point-by-point, generating candidate certainty to complement  |
| Point Properties | generated points intensity | Recent Entry Point Copying | (a) Harmonization of approaches;Instead of choosing a more advantageous strategy. |
| PointRCNN E1 |  Enter Budget  | detector-native  It's...  exact-\(4N\)  Documentation  |  Four-fold complete input after checking the true point  |
| PointRCNN E2 | Enter Budget | Unified 16384 Point | FOV, 0.1 m voxel representative point, distance layer sampling;Only for sensitivity analysis. |
| CenterPoint |  Effective space  | \(x\in[0,70.4),y\in[-40,40),z\in[-3,1)\) m |  With frozen  KITTI  The configuration is consistent  |
| CenterPoint |  voxel Size  | \((0.05,0.05,0.10)\) m |  Native  voxelizer  Parameters  |
| CenterPoint | voxel Capacity | Every voxel maximum 5 points;Test maximum 40000 voxel | observed-first Sequence Budget Mechanism ablation |
|  Official mandate indicators  | AP | KITTI \(AP_{R40}\) |  min  BBox, BEV, 3D  with  Easy/Moderate/Hard  Report  |
| geometric Diagnostic | Reference voxel / Near Surface threshold | 0.20 m / 0.25 m | Read diagnostics only, do not enter generation or detector reasoning |
| Object diagnosis | fixed Subset | 256 frame, 527 Moderate Car GT | Greed is a kind of 3D IoU match;threshold 0.70 |
| Distance Layer | Near/medium/ Far | 0–20 / 20–40 / 40+ m | Including 170 / 272 / 85 and Car GT |

 The table of parameters corresponds to the actual implementation chain and not to the later recommendations.  5.28  Give the input line at the end of this section, patch, strict-\(4N\),  Full flowchart of detector to the third-level audit; all intermediate files can be accessed by  [L14]–[L20]  Retroactive.

#### 5.5.6.2 PointRCNN: A parallel comparison between complete AP and operational integrity

Figure 5.12 expands the full result of Table 5.1 to Easy, Moderate and Hard, which is a combination of three difficulties.The most important visual evidence is not a single column height, but a consistent sequence of two input lines and three difficult approaches.Easy/Moderate/Hard of baseline in Line A is 92.273/82.255/77.945 and PDANS is 83.014/64.515/59.668;baseline of Line B is 85.177/65.746/61.382 and PDANS is 65.062/45.126/39.240.Thus, the negative margin runs through the difficulty range and is not caused by a difficult definition or a a small number of boundary sample.

![ Figure  5.12　PointRCNN  full validation set  exact-\(4N\) E1:  Two input lines, four methods, and three difficult parallel columns. ](figures/fig5_12_pointrcnn_fullval_e1_all_methods.pdf)

 Table  5.9  Go further.  Moderate  The indicator is broken down to the 2D image frame, BEV  rotation and 3D boxes. BBox  with  BEV  It should be relatively stable.  3D AP  Declines alone; the actual result is a simultaneous decline in three spaces and from  BBox  Present.  BEV  Again.  3D  . This indicates that the error has affected the three-dimensional return of the candidate, the horizontal positioning and the integrity of the candidate, rather than the single  \(z\)  Axes error.

** Table  5.9　PointRCNN  full validation set  Moderate AP\(_{R40}\)  It's...  BBox/BEV/3D  Decompose %)**

| Line | Enter | BBox | BEV | 3D | 3D Relative Consistency baseline |
|---|---|---:|---:|---:|---:|
| A | Original baseline (E2) | 94.084 | 88.981 | 82.255 | 0.000 |
| A | PDANS E1 | 78.983 | 75.092 | 64.515 | -17.741 |
| A | PU-GCN E1 | 71.034 | 66.238 | 56.961 | -25.294 |
| A | PU-EdgeFormer E1 | 54.437 | 50.947 | 42.981 | -39.274 |
| A | PU-Net* E1 | 23.489 | 18.067 | 9.717 | -72.538 |
| B | 1/4 sparse baseline (E2) | 79.949 | 76.398 | 65.746 | 0.000 |
| B | PDANS E1 | 60.236 | 54.801 | 45.126 | -20.620 |
| B | PU-GCN E1 | 39.193 | 36.430 | 29.687 | -36.059 |
| B | PU-EdgeFormer E1 | 33.102 | 27.840 | 20.721 | -45.025 |
| B | PU-Net* E1 | 20.774 | 15.904 | 8.878 | -56.868 |

![Chart 5.13 PointRCNN E1 contrasts with Moderate 3D AP parallel coordinates for the unification of 16384 point E2;The right number is E2−E1.](figures/fig5_13_pointrcnn_e1_e2_parallel.pdf)

Figure 5.13 connects the same “Line× Method” to the E1 line under E2.Line A  It's...  PDANS, PU-GCN, PU-EdgeFormer only Increase  2.709, 1.314, 1.979 AP; Line B ' s four methodological changes are not greater than zero.This pairing is more direct than the two stand-alone columns: if 16384 point ceiling is the common cause, the connection line should move generally to the right in both lines;In fact, only a small right shift was made in Line A, while Line B slightly moved left.

![Figures: 5.14 PointRCNN Moderate AP from BBox, BEV to 3D](figures/fig5_14_pointrcnn_metric_profiles.pdf)

In order to prove that the low AP was not caused by the “mission no running out”, table 5.10 also shows the total number of projected documents per group, the number of empty files, the time of detection run and the throughput rate.All conditions have 3769 projection documents;The so-called empty projection file is the frame evaluator input that exists but the detector no output box is different from the missing file.Defines empty output rate and detection throughput

\[
r_{\mathrm{empty}}
=\frac{n_{\mathrm{empty}}}{3769},\qquad
v_{\mathrm{eval}}
=\frac{3769}{t_{\mathrm{eval}}}.
\tag{5.24}
\]

**Table 5.10 PointRCNN full validation set Operational Integrity, Space Forecasting and Testing throughput**

| Line | Conditions | Projection documents | Empty File | Empty File Rate | Run time (s) | throughput (frame /s) |
|---|---|---:|---:|---:|---:|---:|
| A | Baseline E2 | 3769 | 94 | 2.49% | 639.2 | 5.90 |
| A | PDANS E1 | 3769 | 80 | 2.12% | 1489.0 | 2.53 |
| A | PDANS E2 | 3769 | 81 | 2.15% | 647.2 | 5.82 |
| A | PU-GCN E1 | 3769 | 196 | 5.20% | 1254.9 | 3.00 |
| A | PU-GCN E2 | 3769 | 244 | 6.47% | 657.7 | 5.73 |
| A | PU-EdgeFormer E1 | 3769 | 215 | 5.70% | 1240.5 | 3.04 |
| A | PU-EdgeFormer E2 | 3769 | 276 | 7.32% | 654.2 | 5.76 |
| A | PU-Net* E1 | 3769 | 441 | 11.70% | 1236.9 | 3.05 |
| A | PU-Net* E2 | 3769 | 621 | 16.48% | 643.1 | 5.86 |
| B | Baseline E2 | 3769 | 143 | 3.79% | 644.4 | 5.85 |
| B | PDANS E1 | 3769 | 194 | 5.15% | 798.2 | 4.72 |
| B | PDANS E2 | 3769 | 214 | 5.68% | 645.7 | 5.84 |
| B | PU-GCN E1 | 3769 | 621 | 16.48% | 803.8 | 4.69 |
| B | PU-GCN E2 | 3769 | 625 | 16.58% | 652.6 | 5.78 |
| B | PU-EdgeFormer E1 | 3769 | 705 | 18.71% | 806.3 | 4.67 |
| B | PU-EdgeFormer E2 | 3769 | 704 | 18.68% | 648.0 | 5.82 |
| B | PU-Net* E1 | 3769 | 767 | 20.35% | 791.1 | 4.76 |
| B | PU-Net* E2 | 3769 | 800 | 21.23% | 641.5 | 5.88 |

 Figure  5.15  Compare the number of empty projected files with the number of files  Moderate 3D AP  Put it at the same coordinates. 18  A completed condition.  Spearman  Relevant as  \(\rho_s=-0.89\):  The more empty the output, the more often the condition.  AP  Lower. This is not about "empty files lead to all."  AP  The cause and effect of the margin, because the location error in the non-empty frame and false positive will also decrease  AP;  But it's an independent operational level evidence that the process has deteriorated to the point where more complete frame  no is a valid frame, not just a single one.  IoU  A little move.

![Fig. 5.15 PointRCNN Fragmentation point for the number of empty documentation and Moderate 3D AP;Circle/ E1/E2.](figures/fig5_15_empty_predictions_vs_ap.pdf)

 Figure  5.16  Here's the report.  detector evaluation  run time, does not include  upsampling network generated points cloud time, so cannot is miswritten as an end-to-end speed. E2  The throughput focuses on  5.73–5.88  frame /s,  Harmonization of statements  16384  After the dot.  detector  (a) The calculated amount is close; E1  Yes.  Line A  It's only a date.  2.53–3.05  frame /s,  And...  Line B  Yes.  4.67–4.76  frame /s,  This corresponds to the difference in the size of the actual original points.  E1  It does make it bigger.  exact-\(4N\)  Enter Send  detector,  Instead of reading the same point count when the file layer is labelled four times.

![Figure 5.16 PointRCNN recorded detector evaluation throughput;Does not contain upsampling generation time.](figures/fig5_16_pointrcnn_runtime_throughput.pdf)

Table 5.11 places E1 and E2 in the same pair of records for AP changes and for space projected changes.Line A  The three effective adapters are here.  E2  Next only  (1.314\simThe limited improvement of 2.709) AP and the addition of PU-GCN and PU-EdgeFormer by 48 and 61 frame, respectively;PU-Net* point count Unified reduces AP and adds 180 empty outputs.AP of the four methods Line B did not increase, and the empty output of no declined.Thus, fixed input point count can significantly change detector throughput, but cannot separately repairs the distribution mismatch between upsampling input and detector.

**Table 5.11 PointRCNN E1/E2 Matching Change: point count Harmonized to improve AP and empty output**

| Line | Methodology | E1 AP | E2 AP | E2−E1 AP | E1 Empty File | E2 Empty File | Blank File Change |
|---|---|---:|---:|---:|---:|---:|---:|
| A | PDANS | 64.515 | 67.223 | +2.709 | 80 | 81 | +1 |
| A | PU-GCN | 56.961 | 58.276 | +1.314 | 196 | 244 | +48 |
| A | PU-EdgeFormer | 42.981 | 44.960 | +1.979 | 215 | 276 | +61 |
| A | PU-Net* | 9.717 | 8.380 | −1.337 | 441 | 621 | +180 |
| B | PDANS | 45.126 | 44.075 | −1.051 | 194 | 214 | +20 |
| B | PU-GCN | 29.687 | 28.680 | −1.006 | 621 | 625 | +4 |
| B | PU-EdgeFormer | 20.721 | 20.569 | −0.151 | 705 | 704 | −1 |
| B | PU-Net* | 8.878 | 8.781 | −0.097 | 767 | 800 | +33 |

#### 5.5.6.3 CenterPoint: Three categories, three difficulties and gaps recovery

Figure 5.17 first compares raw with four times thin baseline and avoids mixing upsampling methods into sensor thinning effects.The three difficulties show that Cyclist has the greatest relative loss, followed by Pedestrian and Car, the smallest.This model is consistent with the expectation that small target echoes will be low and that local central evidence will be more easily deleted, and suggests that any “class average AP” may mask the category most in need of upsampling.

![Fig. 5.17 CenterPoint full validation set Original/Simple baseline: Three categories, three difficulty parallel columns.](figures/fig5_17_centerpoint_baseline_sparsity.pdf)

 Table  5.12–5.14  Give  CenterPoint  ;" A-” and B-” Separately  observed-first  It's...  Line A, Line B exact-\(4N\)  Enter. They complement the table  5.5  Columns Only  Moderate  Inadequate.

** Table  5.12　CenterPoint Car  full validation set  3D AP\(_{R40}\) (%)**

| Enter | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Original baseline | 88.391 | 79.277 | 76.737 |
| 1/4 sparse baseline | 81.400 | 64.598 | 59.970 |
| A-PDANS | 82.444 | 64.466 | 60.810 |
| A-PU-GCN | 80.013 | 59.932 | 56.575 |
| A-PU-EdgeFormer | 68.036 | 47.248 | 44.166 |
| A-PU-Net* | 22.805 | 14.988 | 14.189 |
| B-PDANS | 69.625 | 46.751 | 42.590 |
| B-PU-GCN | 60.422 | 39.070 | 34.967 |
| B-PU-EdgeFormer | 39.379 | 24.748 | 21.666 |
| B-PU-Net* | 17.534 | 11.718 | 10.436 |

** Table  5.13　CenterPoint Pedestrian  full validation set  3D AP\(_{R40}\) (%)**

| Enter | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Original baseline | 54.037 | 50.653 | 46.252 |
| 1/4 sparse baseline | 27.196 | 24.569 | 21.673 |
| A-PDANS | 49.592 | 45.251 | 41.683 |
| A-PU-GCN | 47.518 | 44.504 | 40.834 |
| A-PU-EdgeFormer | 42.150 | 37.085 | 33.855 |
| A-PU-Net* | 24.479 | 21.687 | 19.389 |
| B-PDANS | 31.467 | 28.201 | 24.756 |
| B-PU-GCN | 27.742 | 24.961 | 21.588 |
| B-PU-EdgeFormer | 17.395 | 15.596 | 13.646 |
| B-PU-Net* | 14.046 | 11.916 | 10.533 |

** Table  5.14　CenterPoint Cyclist  full validation set  3D AP\(_{R40}\) (%)**

| Enter | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Original baseline | 79.244 | 64.605 | 60.990 |
| 1/4 sparse baseline | 25.805 | 15.458 | 14.492 |
| A-PDANS | 71.063 | 46.103 | 43.476 |
| A-PU-GCN | 70.712 | 45.597 | 42.965 |
| A-PU-EdgeFormer | 54.831 | 35.822 | 34.106 |
| A-PU-Net* | 37.058 | 24.396 | 23.453 |
| B-PDANS | 28.118 | 15.519 | 14.895 |
| B-PU-GCN | 30.978 | 16.846 | 16.193 |
| B-PU-EdgeFormer | 13.487 | 7.037 | 6.925 |
| B-PU-Net* | 8.735 | 5.017 | 4.749 |

Chart 5.18 displays Line A and Line B in rows with a pair of pixmaps. The dots and hollow blocks represent two experimental lines, respectively, and the horizontal line directly gives the absolute AP interval of the same method.It must be read in conjunction with the three categories of the margin bar figure for 5.2: the margin chart answers “how much has been changed relative to baseline”, and the absolute figure answers “what level has finally been reached”.For example, Cyclist Moderate AP of Line B PU-GCN is 16.846, although 1.388 has been added to sparse baseline, it is still far from below, the original scanning 64.605;If only the positive margin is shown, it is easy to exaggerate recovery.

![ Figure  5.18　CenterPoint exact-\(4N\) observed-first  2 input lines, 4 methods and 3 categories absolute  Moderate AP. ](figures/fig5_18_centerpoint_absolute_parallel.pdf)

 Table  5.15  and figure  5.19  Usage format(s) 5.9) Put it.  Line B  The margin divided by the original — The negative percentage indicates that the method does not only  no recover the missing information and further below  sparse baseline;  More than  \(-100\%\)  Means that the additional loss is greater than the original thin loss itself. PDANS  Yeah.  Pedestrian  recovery  13.925%,  It is the clearest positive outcome at this time; PDANS/PU-GCN  Yeah.  Cyclist  recovery only  0.124%/2.824%,  And all of it.  Car  And two poor adaptation methods are negative.

**Table 5.15 CenterPoint Line B Moderate AP Gap recovery Rate (%)**

| Methodology | Car | Pedestrian | Cyclist |
|---|---:|---:|---:|
| PDANS | -121.575 | **+13.925** | +0.124 |
| PU-GCN | -173.900 | +1.504 | **+2.824** |
| PU-EdgeFormer | -271.470 | -34.400 | -17.133 |
| PU-Net* | -360.233 | -48.510 | -21.243 |

![(a) The original-sorted performance gap recovery for 5.19 CenterPoint Line B;100% indicates full recovery.](figures/fig5_19_centerpoint_gap_recovery.pdf)

#### 5.5.6.4 geometric Trueness, Segregation of Contamination and Mission Indicators Closed

Table 5.16 summarizes the same set of six geometric — mission variables on the same number of 64 with car frame.The higher the accuracy and near surface ratio of voxel, the lower the extra voxel and the lower the frame;Reference recall reflects coverage, but must be interpreted in conjunction with precision.Line A ' s reference is always 100% because the original scan has been fully preserved;The addition of a point cannot to the observed reference voxel will only change precision and additional occupation.Line B  * The present document is being issued without formal editing.  sparse baseline  It's...  48.49%  Raised to Appearance  57.74%–60.95%,  But in the meantime,  41.74%–54.39%  It's...  E1  Occupation voxel is not supported by reference scanning.

**Table 5.16 geometric Trueness, allocation of vehicle surfaces and PointRCNN AP**

| Line | Methodology | Generate voxel precision | E1 Reference Callback | Extra voxel | Close to the surface in the car. | Shell/Box | Moderate AP |
|---|---|---:|---:|---:|---:|---:|---:|
| A | PDANS | 48.83% | 100.00% | 45.29% | 97.07% | 0.355 | 64.515 |
| A | PU-GCN | 45.48% | 100.00% | 48.93% | 94.01% | 0.422 | 56.961 |
| A | PU-EdgeFormer | 36.32% | 100.00% | 55.49% | 87.06% | 0.621 | 42.981 |
| A | PU-Net* | 23.11% | 100.00% | 62.45% | 57.34% | 1.049 | 9.717 |
| B | PDANS | 50.68% | 60.78% | 41.74% | 93.13% | 0.451 | 45.126 |
| B | PU-GCN | 47.53% | 60.95% | 45.70% | 82.40% | 0.528 | 29.687 |
| B | PU-EdgeFormer | 38.10% | 60.78% | 53.01% | 69.09% | 0.812 | 20.721 |
| B | PU-Net* | 34.17% | 57.74% | 54.39% | 58.06% | 1.125 | 8.878 |

![Figure 5.20 geometric —The dots/spaced squares are Line A/Line B, the numbers are original and the arrows give the preferred direction.](figures/fig5_20_geometry_evidence_matrix.pdf)

Figure 5.20 disassembly the six variables of the schematics into stand-alone micrograms, avoiding the use of normalization to make visual distance that cannot be directly compared.Each small chart retains the original tic and point count values, and the arrow indicates only the preferred direction of the indicator.(b) PDANS/PU-GCN is more advanced in terms of authenticity and mission indicators, PU-EdgeFormer/PU-Net* additional voxel and more serious shell contamination;Both input lines follow the same direction.The judgement is supported by the original measure rather than by a subjective visual rating.

5.21 uses the additional voxel ratio as a cross-axis, PointRCNN AP as a vertical axis and a point area code for the near surface ratio of the vehicle.Both lines are arranged from the top left to the bottom right: even fewer extras and a higher rate of AP for vehicles.This figure and 5.4's “Generating voxel precision - AP” constitute parallel evidence in both directions of precision/error.

![Figure 5.21 Additional voxel ratio to PointRCNN AP;The area of point represents the proportion of the near reference surface of the vehicle generated points.](figures/fig5_21_geometry_tradeoff_scatter.pdf)

#### 5.5.6.5 detector adaptation, object migration and evidence of distance

Figure 5.22 shows the results of 64 frame fine-tuned screening.PU-GCN and Line B PDANS have a positive change in 4.175–4.595 AP, but Line A PDANS has slightly decreased, as have two baseline.Table 5.17 also lists the remaining gaps of the corresponding fine-tuning baseline, avoiding the miswording of “relative pretrained weights” as “more than baseline”.

**Table 5.17 PointRCNN 64 frame Micro-screening: fixed 256 frame Moderate 3D AP (%)**

| Line | Enter | Pre-training | After fine-tuned | fine-tune changes | baseline |
|---|---|---:|---:|---:|---:|
| A | baseline | 79.191 | 78.043 | -1.148 | 0.000 |
| A | PU-GCN | 60.266 | 64.441 | +4.175 | -13.603 |
| A | PDANS | 68.598 | 68.165 | -0.433 | -9.879 |
| B | baseline | 66.619 | 62.914 | -3.705 | 0.000 |
| B | PU-GCN | 32.920 | 37.515 | +4.595 | -25.399 |
| B | PDANS | 43.007 | 47.433 | +4.426 | -15.481 |

![Figure 5.22 64 frame detectorThe right number is fine-tuned - pre-training.](figures/fig5_22_finetune_screening_parallel.pdf)

 Object-level evidence goes on to answer." AP  The margin consists of which objectives”. Defines the net change of target from paired baseline to observed-first as

\[
\Delta N_{\mathrm{TP}}
=N_{\mathrm{obs\ TP,base\ FN}}
-N_{\mathrm{base\ TP,obs\ FN}}.
\tag{5.25}
\]

 If  \(\Delta N_{\mathrm{TP}}<0\),  This means that the new recovery target is less than the destroyed target.  5.18  Four of them.  detector/line  The net change in conditions is negative: PointRCNN A/B  Individually.  \(13-72=-59\), \(12-78=-66\), CenterPoint A/B  Yes.  \(25-41=-16\), \(28-45=-17\).  It's with two detectors.  AP  The direction is the same.

**Table 5.18 fixed 256 frame, 527 Moderate Car GT Migration of objects with IoU quartiles**

| Detector | Line | Enter | TP | Call back. | IoU P25 | IoU P50 | IoU P75 |
|---|---|---|---:|---:|---:|---:|---:|
| PointRCNN | A | baseline | 458 | 86.91% | 0.755 | 0.813 | 0.858 |
| PointRCNN | A | generated-only | 390 | 74.00% | 0.694 | 0.784 | 0.844 |
| PointRCNN | A | observed-first | 399 | 75.71% | 0.703 | 0.797 | 0.849 |
| PointRCNN | B | baseline | 386 | 73.24% | 0.695 | 0.784 | 0.843 |
| PointRCNN | B | generated-only | 258 | 48.96% | 0.000 | 0.698 | 0.802 |
| PointRCNN | B | observed-first | 320 | 60.72% | 0.567 | 0.749 | 0.813 |
| CenterPoint | A | baseline | 454 | 86.15% | 0.742 | 0.811 | 0.862 |
| CenterPoint | A | observed-first | 438 | 83.11% | 0.734 | 0.804 | 0.859 |
| CenterPoint | B | baseline | 409 | 77.61% | 0.713 | 0.790 | 0.842 |
| CenterPoint | B | observed-first | 392 | 74.38% | 0.698 | 0.771 | 0.833 |

![Figure: The 5.23 four detector/line conditions for the object TP/FN migration count;CenterPoint does not run generated-only object audit.](figures/fig5_23_object_transition_counts.pdf)

![Figure: 5.24 between P25–P75 and median for the best-like 3D IoU object;The dotted line is 0.70.](figures/fig5_24_object_iou_quartiles.pdf)

The 5.24 indicates that only is not degraded as a change in the number of TP/FN on both sides of threshold, and IoU distribution itself is moved to the left as a whole.The P25 of PointRCNN Line B generated-only is 0, which means that at least one quarter of eligible GT no produces the same candidate as IoU;observed-first raised P25 to 0.567, median to 0.749, but still below baseline to 0.695/0.784.CenterPoint is less mobile and corresponds to its AP loss.

In parallel, 5.25 compares PointRCNN with CenterPoint's immediate, medium and far-reaching recall.Each point is marked by a difference in the percentage point of observed-first relative to paired baseline.Line A and Line B show that the error is magnified with the increase in distance;CenterPoint is relatively robust, but the distance is still falling.As a result, detector relies on changing the loss margin, no reverses the common direction of “high-risk distance”.

![ Figure  5.25　 Two types of detector in  Line A/B  Near, Middle, Far  Car  (a) The subject ' s recall; Mark as PU-GCN−baseline.](figures/fig5_25_detector_distance_parallel.pdf)

#### 5.5.6.6 Enter point count, cross-distance point cloud and final evidence chain

 Table  5.19  Gives every frame  point count range using full input audit records. Line A exact-\(4N\)  Minimum / The maximum value is exactly four times the original input; Line B exact-\(4N\)  has almost the same point-count range as the original scans. This directly confirms that the count constraint is satisfied and makes the negative result more informative: Line B  Already recovery to about  7.86–12.68  A thousand points. / frame, but no  recovery  baseline AP.

**Table 5.19 Enter point count and Complete Audit**

| Enter | Audit of frame | Every frame minimum point count | Maximum point count per frame | Limited coordinates check | Status |
|---|---:|---:|---:|---|---|
| Original reference | 3769 (source directory with training frame) | 78596 | 126797 | PASS | PASS |
| 1/4 sparse baseline | 3769 | 19649 | 31699 | PASS | PASS |
| Line A exact-\(4N\) | 3769 | 314384 | 507188 | PASS | PASS |
| Line B exact-\(4N\) | 3769 | 78596 | 126796 | PASS | PASS |

![Figure 5.26 Minimum - Maximum point count range per frame entered into the audit;The size of Line B is already recovery to original scan.](figures/fig5_26_input_point_count_ranges.pdf)

Figure 5.27 Adds a comparison of the nine-gauge point cloud for the next, medium and far three CarThe close stabilization cases are frame 004686, GT 0, and distance 15.0 m: baseline/generated-only/observed-first are TP and IoU are 0.871/0.818/0.834, indicating that even if the intensive target is detected, the new points will not necessarily improve its positioning.The mid-range recovery cases are frame 004902, GT 2, 21.8 m: generated-only; IoU 0.646 does not exceed threshold, while observed-first recovery to 0.866. The remote failure case is frame  006039, GT 3,  Distance  41.6 m: baseline  Only Encumbers  9  The frame points are still reached  IoU 0.868,  And...  generated-only/observed-first  There's a difference.  11/18  It's all in the box.  FN.

![Figs. 5.27 Line B Real point cloud Palaces with stable and medium distance recovery and remote residue failure;The blue dotted line is GT.](figures/fig5_27_pointcloud_range_gallery.pdf)

Three cases were not used to estimate overall probabilities, but to validate the geometric form behind the distance curve: close-range generated points is more than, but not necessarily more precisely located, medium-range observed-first can be recovered from a target near threshold by retaining a real structure, while the distance may be “more points but no effective box” due to insufficient geometric support.These three behaviours correspond to the overall tiered results of 527 GT.

![5.28 The KITTI input - upsampling — Test - Audit of the chain of evidence and fixed Parameters that have been implemented in this chapter.](figures/fig5_28_experiment_evidence_pipeline.pdf)

#### 5.5.6.7 Compares two test lines to the whole method scale point cloud

To avoid displaying only PU-GCN or selecting only successful cases, figure 5.29–5.33 compares Baseline, PDANS, PU-GCN, PU-EdgeFormer and PU-Net* to implement a fully parallel point cloud.(a) The overall scenario fixed is the same frame 000104 and the same KITTI effective range and coordinate ratio;To ensure print readability, only shows a sample of each input at the same 8% scale.Sampling is used only for dispersing, with the title and point count used in statistics coming from unsampled `.bin` Documentation.Black means the observation point retained in the input, light gray means generated points, and blue dotted lines means KITTI real box.Therefore, colours do not carry the method of sorting, and the differences in methods are mainly expressed in spatial distribution, local patterns and subsequent statistics.

baseline of Line A contains 121,994 points, and the four methods are 487 and 976 points;baseline of Line B contains 30,498 points, all four methods being 121 and 992 points.Figure 5.29 and chart 5.30 give a direct control: within each line, the total of point count is strictly the same, and the visual difference cannot is attributed to a method that “only only produces more points”.At the same time, Line B exports point count almost recovery to the original frame, but the full AP still does not have recovery, which is consistent with the 3 audit of Table 5.19,769 frame point count.

![The 5.29 Line A frame 000104 parallels the Baseline and the four upsampling methods, BEV;All panels display samples using the same range as 8%.](figures/fig5_29_line_a_all_methods_scene_bev.pdf)

![ Figure  5.30　Line B  frame  000104  It's...  Baseline  Full scene with four upsampling methods  BEV  parallel contrast; all methods are  exact-\(4N\). ](figures/fig5_30_line_b_all_methods_scene_bev.pdf)

The whole scene will test the range of the road and the distance zone, but will compress the local shape of the bicycle.Thus the diagram 5.31–5.32 fixed frame 004902, Moderate Car GT 2 displays five types of input in parallel with the same target coordinate system and within the same display range.Local counts are defined as

\[
n_{s,l,m}^{\mathrm{obs}}
=\sum_{\mathbf p\in\mathcal X_s^{(l)}}
\mathbb I[\mathbf p\in\Omega(B_g)],\qquad
n_{s,l,m}^{\mathrm{gen}}
=\sum_{\mathbf p\in\mathcal G_{s,m}^{(l)}}
\mathbb I[\mathbf p\in\Omega(B_g)],
\tag{5.26}
\]

 of which  \(\Omega(B_g)\)  Yes.  GT  Heading towards alignment and extending in a long, wide direction.  3 m  table.  5.20  Gives a point count that is not sampled from each panel. Line A  The local sum of point count for the medium method is  2,748–3,192, Line B  Yes.  773–863;  That means again.  exact-\(4N\)  It's a total frame constraint, not "each target is just four times more. Different ways to allocate the budget to the target neighbourhood ratio, so the local point count should be geometric for reality, IoU  and  AP  cannot is ranked as a quality.

**Table 5.20 frame 004902, GT 2 Local point count (observation/generation/total)**

| Enter | Line A | Line B |
|---|---:|---:|
| Baseline | 810/0/810 | 222/0/222 |
| PDANS | 655/2093/2748 | 186/587/773 |
| PU-GCN | 777/2409/3186 | 207/646/853 |
| PU-EdgeFormer | 820/2372/3192 | 213/650/863 |
| PU-Net* | 774/2374/3148 | 221/613/834 |

![Fig. 5.31 Line A frame 004902 and GT 2 local BEV;Each panel shows the number of unsampled sites/ generated points.](figures/fig5_31_line_a_all_methods_local_bev.pdf)

![Fig. 5.32 Line B frame 004902 and GT 2 local BEV;The range of coordinates is exactly the same as the chart 5.31.](figures/fig5_32_line_b_all_methods_local_bev.pdf)

The single frame still cannot represents the distribution of data sets. For this, from fixed  256  frame Object Audit Set selected at intervals, for example, by sorting location  32  frame, calculated for each input  10 m  The point count in the directional circle is summarized by the median number that runs across frame:

\[
h_{s,l,m,k}=\sum_{\mathbf p_i\in\mathcal Y_{s,m}^{(l)}}
\mathbb I\!\left[r_k\leq\sqrt{x_i^2+y_i^2}<r_{k+1}\right],
\qquad
\widetilde h_{l,m,k}=\operatorname{median}_{s\in\mathcal S_{32}}h_{s,l,m,k}.
\tag{5.27}
\]

Figure 5.33 uses a logarithmic vertical axis because the distance from point count crosses three orders of magnitude.Of the two lines, the curves of the four methods are four times closer to their respective baseline in most distance bands, but different methods are visible in 20–50 m and beyond. For example:  Line B  It's...  40–50 m  Ring median from  baseline  It's...  214  Increase to  PDANS  It's...  956.5, PU-GCN  It's...  1389.5, PU-EdgeFormer  It's...  1384.5  and  PU-Net*  It's...  1302.0. The result proved that the budget generation did reach the medium- and remote-range area;Combined with AP without recovery, the problem is closer to "the support position of the new point and the local geometric is incorrect" rather than "model no produces a remote point".

![Figure: 5.33 fixed 32 frame two input lines and five input paths to point count median;The axis is a logarithm scale and the ring width is 10 m.](figures/fig5_33_radial_point_density_all_methods.pdf)

#### 5.5.6.8 pair IoU distribution and the distance from confidence interval

In order not to hide the heterogeneity of the target level in the average or a small number of case, 3D IoU is calculated for each detector, input line and GT as the best equivalent of observed-first relative to paired baseline, and the cumulative distribution of experience is drawn:

\[
\delta_i^{(d,l)}=I_{i,\mathrm{obs}}^{(d,l)}-I_{i,\mathrm{base}}^{(d,l)},
\qquad
\widehat F_{d,l}(t)=\frac{1}{M}\sum_{i=1}^{M}\mathbb I[\delta_i^{(d,l)}\leq t],
\quad M=527.
\tag{5.28}
\]

The median of the four distributions in 5.34 is on the left side of zero.The median difference for PointRCNN Line A/B is 0.0144/-0.0254, with the target for improvement being 38.33%/26.94% and the target for degradation being 60.53%/63.95%;The median difference for CenterPoint Line A/B is - 0.0058/-0.0084, improving 45.16%/40.04% and degradation 53.89%/57.50%.This is a clearer indication than a single AP of detector dependency: CenterPoint ' s target level disturbance is smaller, but neither detector is "a slight decline in all the targets " , but it is better to improve the presence of both degradation and degradation.

![Figure: 5.34 527 Moderate Car GT paired Best 3D IoU cumulative distribution;The right side of the zero line is improved.](figures/fig5_34_paired_iou_delta_ecdf.pdf)

 Further 2-scaled distance recall  95% Wilson  Area. If there's a distance, there's a box.  \(n\)  individual  GT,  of which  \(x\)  Fulfilled  IoU 0.70,  then  \(\hat p=x/n\),  Centers and semi-wides

\[
c=\frac{\hat p+z^2/(2n)}{1+z^2/n},\qquad
w=\frac{z}{1+z^2/n}
\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}},
\qquad \mathrm{CI}_{95\%}=[c-w,c+w],\ z=1.96.
\tag{5.29}
\]

The sample quantities of the five distance boxes were 32,138,153,119 and 85.Table 5.21 has some estimates in 5.35, the unit format being baseline/observed-first.PointRCNN  The gap is mainly from  30 m  After that, expand: Line A  Yes.  30–40 m  From  84.9%  Down to  58.8%, 40 m  Later from  61.2%  Down to  32.9%; Line B pairs should be reduced from 52.9% to 33.6% and from 30.6% to 12.9%.CenterPoint has declined in the same direction, but is usually smaller.(a) The inter-temporal no smooth or cross-box shared samples in the figure; For example:  PointRCNN Line B  Yes.  40–70.4 m  It's...  observed-first  Recall as  12.9%, Wilson  The area is  7.4%–21.7%,  Still with  paired baseline  It's...  30.6%  Make a clear separation.

**Table 5.21 Recall by Car (%, baseline/observed-first)**

| Detector / Line | 0–10 m | 10–20 m | 20–30 m | 30–40 m | 40–70.4 m |
|---|---:|---:|---:|---:|---:|
|  Sample Volume  \(n\) | 32 | 138 | 153 | 119 | 85 |
| PointRCNN / A | 100.0/100.0 | 96.4/97.1 | 91.5/88.2 | 84.9/58.8 | 61.2/32.9 |
| PointRCNN / B | 100.0/100.0 | 97.1/95.7 | 85.6/68.6 | 52.9/33.6 | 30.6/12.9 |
| CenterPoint / A | 93.8/90.6 | 98.6/97.8 | 90.2/91.5 | 83.2/74.8 | 60.0/52.9 |
| CenterPoint / B | 93.8/93.8 | 97.1/97.8 | 86.9/85.6 | 70.6/59.7 | 32.9/29.4 |

![(b) 5.35 PointRCNN and CenterPoint were recalled at a five-way box at Line A/B;The bar is box-by-box 95% Wilson.](figures/fig5_35_recall_distance_wilson.pdf)

Finally, table 5.22 maps the main findings with the location of the evidence.The purpose of the table is to ensure that each conclusion has at least one full-scale mission result and a complementary evidence, rather than relying on a single visual point cloud.

**Table 5.22 Core Conclusions, Main Evidence, supplementary Index of Evidence and Data Sources**

| Questions to answer | Main evidence | Complementary evidence | Corresponding Chart | Data sources |
|---|---|---|---|---|
| strict-\(4N\)  Real implementation  | 3769  frame  point count / Limited coordination audit  | E1 detector  throughput Change  |  Table  5.10, 5.19;  Figure  5.16, 5.26 | [L15,L16] |
| point count recovery is equal to performance recovery | PointRCNN/CenterPoint Full AP | Line B point count close to original | Tables 5.1, 5.5, 5.19;Figures 5.12, 5.18, 5.26 | [L15,L16] |
| Whether 16384 points budget is the primary cause | E1/E2 Match AP | E2 Independent voxel and Air Forecast | Table 5.2, 5.10;Figure 5.13, 5.15 | [L16,L17] |
| geometric | Two-line voxel Aligns the precision of AP | Additional voxel, near surface, shell ratio | Table 5.16;Figures 5.4, 5.20, 5.21 | [L17] |
| observed-first Valid | PointRCNN generated-only/observed-first AP | Object recovery / Loss Migration | Table 5.3, 5.18;Figures 5.1, 5.23, 5.24 | [L14,L18] |
| detector adaptation solves field deviations | 3712 frame Fit, fixed 256 frame AP | 64 frame Positive/negative changes in screening | Tables 5.3, 5.7, 5.17;Figure 5.3, 5.22 | [L14,L20] |
| Whether each method completes a comparable point cloud check on two lines | Same frame, same range, same five input grid as the sample shown | 32 frame Radius to point count median | Table 5.20;Figure 5.29–5.33 | [L15,L17] |
| Whether impacts change with distance | 527 GT Recall and Wilson | Three Distance point cloud Case | Table 5.21;Figures 5.8, 5.25, 5.27, 5.35 | [L18] |
| AP Whether the drop was caused by a small number of off-group target | 527 paired IoU margin ECDF | TP/FN Migration and IoU quartile | Figures 5.23, 5.24, 5.34 | [L18] |
| Existence of detector dependency | PointRCNN/CenterPoint input margin | voxel Upper limit and sequence ablation | Table 5.6, 5.7;Figure 5.3, 5.7 | [L14,L15] |

### 5.5.7 Alternative explanation, restriction and evidence boundary

**Evaluate script error.** All main groups have complete projection files and use fixed evaluator for detector;The baseline can reproduce AP, which is reasonable, and therefore the “evaluationr's total failure” does not match the evidence.This does not, however, preclude differences in weights or pre-treatment over different experimental years, so the body text calculates the margin only within its own experiment.

**point count cap.** E2 controls 16384 after Line B no recovery;CenterPoint also failed to trigger 40000 voxel in Line B, but still fell significantly.The limited budget is therefore a magnifying factor rather than a common root cause.

** no Keeps observation. ** E1  Keep whole  \(N\)  Just be real. Still below.  baseline; observed-first  But recovery  AP,  There is still a double-digit gap. The explanation covers only part of the loss.

**detector no fit.** 3712 frame two detector still below baseline.Domain appliance is helpful, but not enough to make the error geometric a correct observation.

**Reference scanning is not a real continuous surface.** The original KITTI scan itself is limited and noise-intensive, so "Refer to voxel" is not necessarily a physical error;It may also be that the network reasonably completes the surface not captured by the sensor.Therefore, this chapter does not directly label a single additional voxel as false point, but instead uses “original scans are not supported”.However, when the percentage of non-support is 40%–60%, when there is an increase in pollution from the vehicle's shell, when AP is ranked in the same order of precision and when two detector drops at the same time, it is not consistent with the joint evidence to interpret all the additional points as beneficial completion.

**Selected case deviations.** Figures 5.10 and 5.11 are representative cases of interpretation mechanisms and are not used to estimate recovery probabilities.The overall effect was determined by the conversion count of 527 GT and the official AP.

## 5.6 Summary of Main Findings

The main findings of this chapter can be condensed to the following nine points, each of which is tied to a clear level of evidence.

1. **Strictly four times point count does not imply four times the information.** The main results of 3769 frame PointRCNN and CenterPoint show that most upsampling conditions below co-line baseline;Line A no Any CenterPoint category improved.
2. **Line A and Line B must be interpreted separately.** Line A measure marginal increase, Line B detector loss recovery.Line B  a small number of  AP  Just show up.  CenterPoint  It's...  Pedestrian/Cyclist,  And only recovery Original — A small fraction of the thin gap.
3. **The retention of sites is necessary but insufficient.** PointRCNN Line B observed-first  That's right.  generated-only  Increase  9.224 AP,  But it still compares.  baseline  Low  11.004 AP.
4. **fixed input budget is not the main cause.** PointRCNN unified to point 16384 without recovery Line B;CenterPoint Line B no Triggered voxel cap is still down.
5. **The non-local patch is a located upstream failure.** XY medians 30.46 m and p90 124.10 m of Line B patch are incompatible with the object level local surface assumptions.
6. ** geometric Trueness corresponds to the sequence of tasks. **  In two lines, 0.2 m  Generate voxel Real Precision and  PointRCNN AP  The four methods are relevant.  \(\rho_s=1.00\).
7. **detector determines the magnitude of the loss and does not change the common direction.** After full training, PU-GCN dropped about 4.9–5.9 AP on CenterPoint and about 11.0–11.4 AP on PointRCNN.
8. **More box dots do not guarantee detection.** recovery and the failed point cloud case, together with the audit of 527 GT, show that space location, surface support and origin order are more interpretative than nudity point count.
9. **The whole approach parallels point cloud and distribution statistics exclude display deviations.** The five input grids of the two input lines, 32 frame diameter point count and 527 paired IoU/ recalls indicate that the budget generation did reach medium distance, but that the target margin was degraded higher and that 30 m loss after PointRCNN was clearly supported by Wilson.

 These results do not support "the current upsampling current line generally raised"  KITTI  A three-dimensional test's strong proposition. They support the more nuanced conclusion that localism, geometric authenticity, observation protection and detector, with no label fit in, indicate the value of a common decision as to whether enrichment is of mission value, the most important of which now occurs before the generator and the generation of geometric itself.

# 6. Conclusion and Outlook

## 6.1 Answers to the Research Questions

### 6.1.1 RQ1:  point-cloud upsampling  KITTI  Three-dimensional target test?

Under protocol, which has been completed and has been checked for completeness, the answer is:**cannot sees upsampling as a generally effective test enhancement;Only limited recovery was observed under specific detector, categories and thin input conditions.**

The evidence comes from three complementary levels. First of all, 3769  frame  PointRCNN E1  of which four methods are in  Line A  and  Line B  It's...  Car Moderate AP  Both below Convergence  baseline. Second, in 3769 frame CenterPoint, all Line A “method x category” combinations are negative;PDANS/PU-GCN of Line B only gets + 0.061 to + 3.632 AP on Pedestrian or Cyclist, while Car is significantly down.Third, PU-GCN is still on two detectors and two input lines below baselines after 3712 frame input fields are adapted.

Therefore, the conclusions of the paper must be written as a conditional proposition:

\[
\exists(d,c,l,m):\Delta AP_{d,m,c}^{(l)}>0
\quad\not\Rightarrow\quad
\forall(d,c,l,m):\Delta AP_{d,m,c}^{(l)}>0.
\tag{6.1}
\]

This experiment only proves that the left side exists in a small number of CenterPoint Line B small grouping, which does not satisfy the general improvement on the right.Correspondingly, upsampling recovery KITTI Testing is not a summary supported by data;“upsampling shows limited recovery under some small target conditions, but is generally bound by geometric”.

### 6.1.2 RQ2: What does geometric quality have to do with testing performance?

 In the current comparison of the four methods, geometric is highly consistent with the sequence of the tests. Line A  and  Line B  It's...  0.2 m  Generate voxel with real precision sorting  PDANS > PU-GCN > PU-EdgeFormer > PU-Net*,  with  PointRCNN Moderate AP  It's the same sort of thing. Spearman \(\rho_s=1.00\).  The near-referenced surface ratio in the vehicle frame is in the same direction, while additional voxel and shell contamination are in reverse order.

However, that conclusion should be interpreted as “necessary candidacy” rather than as an adequate causal rule.geometric The evaluation uses original scans as a limited reference and does not allow for the observation of all real physical surfaces;Four methods are not sufficient to stabilize the estimated continuous effect curve;The same geometric dots enter different detector and then go through sampling, voxel and characterization.The more rigorous conclusion is that:

\[
\Delta AP
=F\!\left(
P_{\mathrm{vox}},
C_{\tau},
E_{\mathrm{vox}},
\eta_{\mathrm{obs}},
A_d,
\phi_d
\right),
\tag{6.2}
\]

 of which  \(P_{\mathrm{vox}}\)  Is generating voxel precision, \(C_\tau\)  It's a reference cover. \(E_{\mathrm{vox}}\)  It's an extra voxel rate. \(\eta_{\mathrm{obs}}\)  It's actually keeping the rate of observation. \(A_d\)  with  \(\phi_d\)  detector adaptation and weights, respectively. The current results confirm that these variables are important in common, but cannot counterplay functions from four methods  \(F\)  It's a common form.

### 6.1.3 RQ3: Is the results consistent among detector?

The direction is consistent and the range is inconsistent.PU-GCN  It's a training and adaptation experiment. PointRCNN  Yes.  Line A/B  Falling separately.  11.400/11.004 AP, CenterPoint  Falling separately.  5.890/4.852 AP. In both cases no exceeds baseline, so the common root cause of cannot is attributed to one detector;CenterPoint’s lesser loss is an indication that voxel aggregates are more stable to some point disturbances.

Enter sequence ablation further reveals the detector-specific mechanism.PointRCNN Line B  It's...  observed-first  Relative  generated-only  Increase  9.224 AP; CenterPoint ' s sequenced gains occurred mainly when Line A reached the 40000 voxel ceiling, and Line B was almost zero. In other words, PointRCNN  The key budget is point sampling and local features. CenterPoint  The key budget is point count in voxel and the non-empty voxel. Any conclusions of the paper on “upsampling Effectiveness” should indicate downstream detector, while cannot extended the response of a model to the entire test mission.

### 6.1.4 Quantitative evidence summary of research issues

Table 6.1 does not repeat chapter V methodologically, but rather establishes a “main result-mechanical evidence-conclusive strength” correspondence for the three research issues.The positive and negative numbers before the values are always referenced by paired line baseline;Experiments of different sizes are not mixed for averages.

**Table 6.1 Quantified answers to three research questions and evidence intensity**

| Research issues | full validation set Main result | supplementary / Mechanism Evidence | Supported conclusions | Strength |
|---|---|---|---|---|
| RQ1: upsampling Increase detection | PointRCNN Best PDANS: Line A/B is - 17.741/-20.620 AP;CenterPoint Line B Only a small number of | point count range recovery to original size, still not recovery Car AP | (b) Not to support general upgrading;only Supports conditionality, small categories recovery | Strong. |
| RQ2:  How geometric relates to the mission  |  voxel for the generation of two lines  PointRCNN AP  It's all in the same order. \(\rho_s=1.00\) |  Extra voxel  41.74%–62.45%;  Shell ratio to  AP  Inverse Sorting  |  The location is more authentic than the nominal point count, which explains the differences in current methods.  |  medium strength;methods only  4 |
| RQ3: dependent on detector | After full training, PU-GCN: PointRCNN -11.400/-11.004 AP, CenterPoint -5.890/-4.852 AP | Net object TP Change PointRCNN -59/-66, CenterPoint -16/-17 | Two detector in the same direction, voxel model detector with smaller losses | John supplementary |

Figure 6.1 compresses the conclusions with four separate quantitative panels.Top left is full validation set PointRCNN and the best method is still below baseline;The top right is CenterPoint PDANS, only Line B Pedestrian/Cyclist is partially positive;Bottom left is 3712 frame after adaptation detector gap; Down right.  527  individual  GT  Let's go.  observed-first  Relative  baseline  Net object  TP  Change. The four panels are derived from the full AP, the class margin, the locale fit and the match of the object, and are drawn over and over again from the same number.

![Figure 6.1 Quantification panel for the main conclusions of chapter V: full AP, category margin, appropriate disability differential and net object TP.](figures/fig6_01_quantitative_conclusion_dashboard.pdf)

 This aggregation allows for a distinction between “conclusion determination” and “pending validation interpretation”. It can be ascertained that:  strict-\(4N\)  Chain no universal  AP  improvements;limited / voxel Budget  detector  The field deviations explain only part of the loss; geometric is true to the order of the task. Still pending validation is: repaired  patch  with  PU-Net*  After standardization, the absolutes of each method  AP  How much recovery and whether the same range is generated by the use of real low-line sensors.

## 6.2 Main Contributions of the KITTI Study

The contribution of the study is not to claim that one method has acquired a new KITTI, the best AP, but to establish and empirically establish a task-oriented evaluation framework that can explain negative results.

**First, there are two assessment lines that are not confused.** Line A Checks the marginal increase on the full scan, Line B Checks the performance of recovery after the thinning.The design avoids mixing the words “a slight recovery after a low baseline” with the words “above the original scan”.

** Second, implementation  strict-\(4N\)  It is traceable with observations. **  Main output defined  \(N\)  A real observation.  \(3N\)  generated points, the original observations are not impersonated by the retrogression point of the network. The number of inputs, the source of observations and the source of generation can follow  evaluator  Full-chain tracking.

**Third, cross-check with voxel type detector.** PointRCNN [11] and CenterPoint [12] Shares the same upsampling input with different front-ends.The two dropped together before detector;The margin range also reveals the reconciliation of downstream input budgets.

**Fourth, the task indicators are linked to the geometric mechanism.** The total AP and patch ranges, the generation of voxel precision, reference cover, additional voxel, distance layers and the object TP/FN converted from a "AP" to a positionable causal candidate chain.

**Fifthly, the evidence of failure should be retained without removing the unusual method.** PU-Net* * The result was marked as a pipeline diagnostic due to a confirmed scale packaging error and is not used for structural ranking.This treatment preserves the facts of the project and avoids unfair academic conclusions.

## 6.3 Limitations

### 6.3.1 Data and Sorbing Model

A quarter of Line B downsampling is a definitive pressure test, which is not equivalent to a real lowline beam LiDAR.Real sensor changes also involve vertical agular resolution, scanned phase, motion malformation, reflecting material and shielding order.The current results provide an indication of whether recovery can be obtained under this downsampling algorithm, and cannot directly predicts the actual benefits of hardware replacement from 64 to 16.

The original KITTI scan has been used as a geometric reference, but it is not a continuous, noiseless surface of real ground value.The network may also fall into voxel, which is not occupied by original scans, when a reasonable value is inserted between the two sites. Therefore, the extra voxel ratio should  AP,  The object boundary and many threshold are explained together, cannot as a single error rate.

### 6.3.2 Method Weight & Fitr

 The public weights, training data and input preprocessing of upsampling methods are not identical.  exact-\(4N\)  Controls the number of outputs, cannot makes the network a priori consistent.  patch  The non-locality of the extractor is currently the most explicit project limitation; the value may change after the repair.

PU-Net * Packagings lack centralization consistent with training, scale normalization and reverse transformation.So its current AP can only be used for failure diagnosis.If this method is retained in the final defence, it must be reruned in its entirety by 3769 frame, two detector master protocol;Otherwise, it should be removed from the MCP table, only into the limit of implementation.

### 6.3.3 detector adaptation and Statistical Uncertainty

Full data adaptation is currently concentrated on PU-GCN;The complete detector-adaptation matrix of the four methods has not yet been fully established.Thus, cannot asserted that the other methods remained exactly the same AP margin after full adaptation.64 frame fine-tuning can only provide evidence of screening.

Official KITTI evaluator provides data set level AP, but the current report no implements more random seed training or frame level bootstrap confidence interval for all conditions.The fixed weighting reasoning itself is established, and training suitability may still be affected by initialization and mini-batch.For future adaptation experiments, pairs shall be calculated in frame in resampling units bootstrap:

\[
\Delta AP^{*(b)}
=AP\!\left(\{s_i^{*(b)}\}_{i=1}^{n};m\right)
-AP\!\left(\{s_i^{*(b)}\}_{i=1}^{n};\mathrm{base}\right),
\tag{6.3}
\]

 & Use  \(B\)  The next resampling  2.5%  with  97.5%  bits formed  95%  . This must be the same resampling  frame calculation method and  baseline,  to preserve a pair structure.

### 6.3.4 Object Diagnosis and Attribution boundary

The audit of 527 Moderate Car GT uses the same matching of greed and is used only to explain the IoU conversion to TP/FN;It no is a predictive confidence score, and no fully replicates the official ignore area and difficulty processing, cannot instead of AP.The selected recovery / failure is visual evidence, not an overall frequency estimate.

 geometric Precision and  AP  It's...  \(\rho_s=1.00\)  Based on four methods. The rankings are very consistent, but the number of samples is too small to claim a general statistical relationship.  patch  scale, different candidate filters and several training seeds to create sufficient conditionality points.

## 6.4 Recommended Corrected Experimental Pipeline

The current results have indicated the order of the most valuable corrections for the next round of experiments.This sequence is not intended to change protocol in the search for higher numbers, but rather to remove the identified mix.

### 6.4.1 Real local no label patch extraction

 Use it.  FPS seed  Add  kNN  or  ball query  Replace rough  bin  Slice in consecutive blocks. To torrents  \(\mathbf c_j\), kNN patch  Defined

\[
\mathcal P_j^{\mathrm{kNN}}
=\operatorname*{arg\,min}_{\substack{\mathcal P\subset\mathcal X\\|\mathcal P|=K}}
\sum_{\mathbf x\in\mathcal P}\|\mathbf x-\mathbf c_j\|_2^2,
\tag{6.4}
\]

ball-query patch is

\[
\mathcal P_j^{\mathrm{ball}}
=\{\mathbf x\in\mathcal X:\|\mathbf x-\mathbf c_j\|_2\le r_j\}.
\tag{6.5}
\]

 Radius  \(r_j\)  You can adjust to local point spacing, but the rules must only depend on the input point, must not access  GT  Box, category or  evaluator.  Use certainty repetition if point count is insufficient / Close Fill & Record  padding  ratio; cannot extended to a few dozen metres to fill  2048  Point.

### 6.4.2 Normalization and reverse transformation consistent with training

 Every one  patch  Preservation Centre  \(\boldsymbol\mu_j\)  With scale  \(a_j>0\):

\[
\widetilde{\mathbf x}_i
=\frac{\mathbf x_i-\boldsymbol\mu_j}{a_j},
\qquad
\widehat{\mathbf g}_k
=a_j\widetilde{\mathbf g}_k+\boldsymbol\mu_j.
\tag{6.6}
\]

 Network input and output must execute the change and reverse change, respectively, for each  patch  Records  \(\boldsymbol\mu_j,a_j\). PU-Net*  Only when this link is consistent with the original training code, can the main method sheet be re-entered.

### 6.4.3  Observe retention, candidate consolidation and  exact-\(4N\)

 Overlap  patch  There'll be more than that.  \(3N\)  Candidates. Unanimous, no label Chooser should solve

\[
\mathcal G^*
=\operatorname*{arg\,max}_{\mathcal G\subseteq\mathcal C}
\left[
\lambda_1\operatorname{Cov}(\mathcal G,\mathcal X)
-\lambda_2\operatorname{Dup}(\mathcal G)
-\lambda_3\operatorname{Out}(\mathcal G,\mathcal X)
\right],
\quad |\mathcal G|=3N,
\tag{6.7}
\]

Of these, coverage, duplicate and outlier are calculated from input geometric instead of testing GT.Finally press

\[
\mathcal Y=[\mathcal X;\mathcal G^*],\qquad|\mathcal Y|=4N
\tag{6.8}
\]

To assemble and maintain the observation/generation source mask.(a) For PointRCNN, further recording of the two categories of actual access to 16384 point sampling;For CenterPoint, the maximum number of voxel after the filtration of the records, the voxel ratio to the maximum point of 5 and 40000 voxel hit.

### 6.4.4 Phased acceptance conditions

It is recommended that the following pre-registration threshold be adopted before re-exercising the full 3769 frame of two detector:

| Phase | Acceptance and acceptance | Recommendation requirements |
|---|---|---|
| patch | p90 radius, XY diagonal, cross-component ratio | Line B Medium Range no longer reaches multiple targets/wide scene scale |
| Coordinates | finite, Range, Reverse Error | No NaN/Inf;The rice scale is reasonable;Return error close to value accuracy |
| strict  Output  |  point count, observation Hash, source mask  |  Every frame happens  \(4N\); \(N\)  One observation-by-point retention  |
|  geometric  | \(P_{\mathrm{vox}}\), \(C_\tau\), \(E_{\mathrm{vox}}\) |  Accuracy increases and additional voxel drops, cannot increases the coverage only  |
| Objective | Near/medium/far vehicle near surface ratio | Unaccepted systemic collapse without distance |
| detector | Forecast file, air forecast, AP and logarithm | All 3769 frame pass;Consisting baseline running simultaneously |

The amount threshold in the “Recommended Requirements” should be determined on the basis of the development set before viewing the final AP, avoiding the test results being used to reverse the selection of the most advantageous threshold.

### 6.4.5 Rerun sequence and parameter registration for evidence door control

 Figure  6.2  Split the next round into five cannot.  gate. Gate 0  Current  10  Group  CenterPoint  Enter audit and  PointRCNN  (a) Full-volume document check passed; Gate 1  Not adopted at this time because of the common  patch  Yes.  Line B  It's...  XY  Diagonal median reached  30.46 m, p90  Ta-da.  124.10 m,  And...  PU-Net*  Lack of a scale transformation consistent with training; Gate 2  It's not approved at this time because  exact-\(4N\)  Enter content  41.74%–62.45%  The reference scan does not support voxel; Gate 3  Show  observed-first  There's a partial return, but...  baseline gap  Not yet closed. Only  Gate 1–3  After the amendment and pre-registration, Gate 4  It's...  3769  frame is the full rerun that makes sense for a relatively new architecture.

![Figure 6.2 Amended KITTI Experiment Evidentiary Timeline;The dot colour indicates the pass, failure, partial completion and pending, and the later stage cannot fails to cover up the pre-stage.](figures/fig6_02_evidence_gated_roadmap.pdf)

**Table 6.2 Current state of evidence and products required to move to the next stage**

| Gate | Current Status | There's evidence. | The products that have to be produced before the next stage. |
|---|---|---|---|
| 0  protocol Lock  | PASS |  Every frame  exact-\(4N\),  Limited coordinates, 3769  Full file.  |  frozen  frame List, \(D_4\), intensity  and  evaluator  Hash.  |
| 1 unit test for adapters | FAIL | Non-local patch;PU-Net * scale Error | patch Local Report, Centre/scale round trip error, source mask audit |
| 2 geometric Development Collection | FAIL | Generating voxel precision 23.11%–50.68%;Extra voxel 41.74%–62.45% | fixed development report precision, recall, extra, near-surface, shell |
| 3 detector pilot | PARTIAL | observed-first recovery Part AP/TP but still below baseline |  The same.  256  frame, Convergence  baseline,  Two.  detector,  Object / Clear distance.  |
| 4 full validation set | Current reference completed | 3769 frame PointRCNN and CenterPoint main table | Gate 1–3  After rerun  3  Category ×3  Difficulty × Two-line complete matrix.  |

Table 6.3 gives parameters for the next round of experiments that need to be registered before running.The “locked” in the table means that the current definition is maintained in order to be comparable;"Stick after development selection" means you can choose on an independent development set, but once you look at the official validation of AP, must not is modified;"Reading training configurations by method" means that the parameters must be faithful to the original method, and cannot consolidates the error scale in order to output better numbers.

**Table 6.3 Registration Form for Parameters for Amendment Experiments**

| Parameter Group | Parameters | Registration rules | Whether to allow changes after checking AP |
|---|---|---|---|
| Data | train/val/256 pilot frame List | Use Saved List with Hash | Yes |
|  Shrink  | \(D_4\)  Count, random seed  |  Keep Current  Line B  definitions;in addition to a separate column for real low-line beam experiments  |  Yes  |
| patch | seed algorithm | FPS or certainty over sampling;Lock after development set selection | Yes |
| patch | neighbourhood | 2048 point kNN or pre-registered ball query;Record padding, Radius & Overlay Rate | Yes |
|  Coordinates  | center/scale |  Method-by-method definition of training; saving  \(\boldsymbol\mu_j,a_j\)  and execute reverse transformations  |  Yes  |
|  Generate  |  Multiplication & Merge  |  Every  patch  methods;total frame  \(3N\) generated |  Yes  |
|  Output  |  Observation protection  | \(N\) observed  Keep point-by-point and ahead;save  source mask |  Yes  |
| Strength | Main Policy | nearest-input as a Recoverable Master Policy | Yes |
| Strength | Sensibility | Local plug-in or intensity mask, all methods using the same rule | Yes |
| PointRCNN | E1/E2 | E1  Yes.  detector-native exact-\(4N\); E2  fixed  16384,  Sensitivity.  |  Yes  |
| CenterPoint | range/voxel/cap | \([0,70.4)\times[-40,40)\times[-3,1)\) m; 0.05/0.05/0.10 m; 5  Points / voxel; 40000  voxel  |  Yes  |
|  Evaluation  | AP |  fixed  KITTI \(AP_{R40}\),  Reservations  BBox/BEV/3D,  Three difficulties, three categories.  |  Yes  |
| Statistics | Training seeds/areas | Registration of seeds before running;Use a pair of bootstrap to report 95% | Yes |

Door-control logic can be put into form.

\[
\operatorname{RunFull}
=\mathbb I(G_0=1)\mathbb I(G_1=1)\mathbb I(G_2=1)\mathbb I(G_3=1),
\tag{6.9}
\]

 of which  \(G_k\)  No. No.  \(k\)  individual  gate  Whether or not. The purpose of the formula is not to simplify complex experiments into a fraction, but to prevent "full"  AP  It's already running. It's wrong to think that the upstream adaptor is right.  strict  Any output not passed, new full volume  AP  I can only continue to diagnose the current line, cannot as a fair framework conclusion.

## 6.5 Outlook and Future Work

### 6.5.1 from "equivalent increase" to "mission uncertainty driven increase"

 The current protocol sets the point count factor per input point at four times the fixed rate, but  LiDAR  The lack of information on the scene is not uniform. The road plane usually does not need the same generation budget as the remote edge of the vehicle. The future can be defined as no label uncertainty  \(u_i\),  Combining local density, curve rate, blocking boundary and distance,  patch  Allocation budget:

\[
n_j
=3N\cdot
\frac{\exp(u_j/\tau_u)}{\sum_k\exp(u_k/\tau_u)},
\qquad \sum_j n_j=3N.
\tag{6.10}
\]

 The strategy remains fully frame.  exact-\(4N\),  But to prevent label from being detected, the new point is being moved from a large flat background to a truly missing local area that allows recovery. \(u_j\)  The main protocol shall be generated only by input geometric; use  detector feature  The version should be included as a mission awareness extension.

### 6.5.2 Visible Model LiDAR Scan Structure & Distance

 object Level Euro neighbourhood cannot be fully represented  LiDAR  The angle sampling. The future is available  range image  Medium Modeling Level / Vertical adjoining, similar  TULIP  Yeah.  LiDAR  Sensor structure considerations for upsampling  [32],  Map back to the 3-D coordinates.  \((x,y,z)\),  The coordinates of the ball are

\[
r=\sqrt{x^2+y^2+z^2},\qquad
\theta=\operatorname{atan2}(y,x),\qquad
\varphi=\arcsin(z/r).
\tag{6.11}
\]

 Yes.  \((\theta,\varphi)\)  Grid interpolation can keep the scan line purged and allow for the Quantified Quantification Error Visible Model. The route needs to be forecast simultaneously  range  and effectiveness / Cover the probabilities and avoid refilling the rear surface on the rays that are shielded by foreground.

### 6.5.3 geometric — Co-generation of strength

Recent Neighborhood Replications, while re-emerging, do not satisfy the physical consistency of the coordinates after they have been moved.Future models can combine output position and intensity distribution:

\[
p(\mathbf g,I_g\mid\mathcal P)
=p(\mathbf g\mid\mathcal P)\,
p(I_g\mid\mathbf g,\mathcal P),
\tag{6.12}
\]

And predict uncertainty in the form of an alien difference:

\[
\mathcal L_I
=\sum_g
\left(
\frac{|I_g-\widehat I_g|}{\sigma_g}
+\log\sigma_g
\right).
\tag{6.13}
\]

This design allows detector to reduce the weight of generated points to a high degree of uncertainty, rather than viewing each generated points as as as as reliable as the measured echo.

### 6.5.4 Source Perception and Confidence Awareness detector

 The current output saves the source mask, detector is usually consumed only  \((x,y,z,I)\).  The future can be marked with observations.  \(o_i\in\{0,1\}\)  And generate confidence  \(w_i\in[0,1]\):

\[
\mathbf f_i^{(0)}=[x_i,y_i,z_i,I_i,o_i,w_i].
\tag{6.14}
\]

This way detector can learn "real observed-first, generated points as soft evidence" instead of allowing the two to compete without distinction.In order to maintain equity, three sets should be compared: frozen detector, which does not use the source, only, which trains detector and the generator and detector.As a result of the joint training, frozen is the best in the system, and cannot is placed in the same column without any indication.

### 6.5.5 More frame Information compared to single frame

If the time context is applied, the actual neighboring frame may provide a more reliable missing surface than the single frame hallucination.Future should be added to multi-sweep baseline:

\[
\mathcal X_t^{\mathrm{multi}}
=\bigcup_{\delta=-K}^{K}
\mathbf T_{t\leftarrow t+\delta}
\mathcal X_{t+\delta},
\tag{6.15}
\]

 of which  \(\mathbf T\)  The dynamic target needs to be compensated by movement, otherwise more frame super-heavy will result in another geometric tow. Compare the study upsampling with the real number of frame observations, you can answer the generated points calculation. / It is only with a delay that it is of real value.

### 6.5.6 More complete statistical design

 Future experiments should contain at least three types of repetition: upsampling web-based training seeds, detector  Matchable and downsampling seeds. For condition  \(c\)  The hierarchy model can be written as:

\[
T_{c,r,s}
=\mu+\alpha_c+u_r+v_s+\epsilon_{c,r,s},
\tag{6.16}
\]

 of which  \(u_r\)  It's an expression of the repetitive effect of training. \(v_s\)  This means that the data is diluted and duplicated.  AP  It is also important to report averages, standard deviations, pairs of confidence interval and effects to avoid a training session.  0.1–0.5 AP  Volatility is written as a steady improvement.

## 6.6 Final Conclusion

 This study is based on  KITTI  The whole scene set out to establish a traceable evaluation chain for four types of upsampling method, two input lines and two three-dimensional detector. The results show that current current pipelines can be stabilized and generated.  exact-\(4N\)  File, but cannot stabilizes the generation of real geometric information at the same price. PointRCNN  and  CenterPoint  of the Parties to the Kyoto Protocol E1/E2  Enter budget ablation, non-local  patch  Measuring, generating voxel precisions and  AP  Consistency of ranking, distance stratification and object level  TP/FN  Switch to position the main issues to the scene.  patch  Construct and generate geometric instead of a single  evaluator  Or single.  detector.

The conclusion is not that "point-cloud upsampling is never useful for testing."CenterPoint Line B ' s limited positive value on Pedestrian/Cyclist, observed-first ' s part recovery and detector ' s adaptations indicate that upsampling has available space.What is actually denied is a simpler assumption: by packaging the object-grade network directly to the whole scene, making point count four times the size of the scene, it automatically increases the detection.

 The priority for follow-up is clearly given by the evidence: first, to repair the real locals.  patch  with  PU-Net  scale transformation; second  strict-\(4N\)  Add reference coverage and precision to reduce additional voxel; re-use source / Trust senses.  detector  Enter; lastly, multi-rate, multi-sensor and joint training. Only when these amendments are made  PointRCNN  with  CenterPoint,  full validation set and distance / In the category hierarchy, the "geometric" can be upgraded to "more useful on mission".

# Quote links, add new literature and the basis of local experiments

## Follow the previous academic literature.

The following number is complete succession: `thesis.pdf`, do not renumber the final merge.The entries actually cited in this chapter are listed here and are not cited but exist [3], [8]–[10], [13]–[31], [33] It is retained in the general reference table for the entire paper.

[1] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? The KITTI vision benchmark suite,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2012, pp. 3354–3361.

[2] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The KITTI dataset,” *Int. J. Robot. Res.*, vol. 32, no. 11, pp. 1231–1237, 2013.

[4] L. Yu, X. Li, C.-W. Fu, D. Cohen-Or, and P.-A. Heng, “PU-Net: Point cloud upsampling network,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 2790–2799.

[5] G. Qian, A. Abualshour, G. Li, A. Thabet, and B. Ghanem, “PU-GCN: Point cloud upsampling using graph convolutional networks,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 11683–11692.

[6] D. Kim, M. Shin, and J. Paik, “PU-EdgeFormer: Edge transformer for dense prediction in point cloud upsampling,” in *Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP)*, 2023, pp. 1–5.

[7] B. Zhang, S. Yang, H. Chen, C. Yang, J. Jia, and G. Jiang, “Point cloud upsampling using conditional diffusion module with adaptive noise suppression,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2025.

[11] S. Shi, X. Wang, and H. Li, “PointRCNN: 3D object proposal generation and detection from point cloud,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 770–779.

[12] T. Yin, X. Zhou, and P. Krähenbühl, “Center-based 3D object detection and tracking,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 11784–11793.

[15] Y. Zhou and O. Tuzel, “VoxelNet: End-to-end learning for point cloud based 3D object detection,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 4490–4499.

[16] Y. Yan, Y. Mao, and B. Li, “SECOND: Sparsely embedded convolutional detection,” *Sensors*, vol. 18, no. 10, p. 3337, 2018.

[32] B. Yang, P. Pfreundschuh, R. Siegwart, M. Hutter, P. Moghadam, and V. Patil, “TULIP: Transformer for upsampling of LiDAR point clouds,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024, pp. 15354–15364.

[34] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL visual object classes (VOC) challenge,” *Int. J. Comput. Vis.*, vol. 88, no. 2, pp. 303–338, 2010.

[35] A. Simonelli, S. R. Bulò, L. Porzi, M. López-Antequera, and P. Kontschieder, “Disentangling monocular 3D object detection,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 1991–1999.

## New academic literature

**None.** All academic numbers of the draft are consistent with the previous text, no [36] after the entry.If chapter 3 and chapter 4 of the final paper have been added [36]–[37], it is sufficient to retain its original number;This chapter no quotes them and does not register again.

## Basis for local experiments (excluding academic references)

[L14] `results/pugcn_full_retrain_20260824/reports/full_retraining_report.md`, `full_retraining_summary.csv` with `pointrcnn_observed_first_report.md`: 3712 frame Fit Training, fixed 256 frame PointRCNN/CenterPoint PU-GCN Evaluation.

[L15] `results/centerpoint_exact4n_e1_20260729/CENTERPOINT_LINE_A_B_REPORT.md`  with  `results/centerpoint_exact4n_reconstructed_order_safe_20260729/full_ap_summary.csv`: CenterPoint 3769  frame, 10  Group  exact-\(4N\)  The main evaluation and input order ablation.

[L16] `results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_live_ap_summary.csv`: PointRCNN 3769 frame E1/E2 evaluation.

[L17] `results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_concrete_root_cause_report.md` and `geometry_root_cause_v1/`: 64 frame patch Locality, generation voxel, vehicle surface and distance layer audit.

[L18] `results/pugcn_full_retrain_presentation_20260828/evidence/current_object_evidence.json` with `current_object_records.csv`: fixed 256 frame, 527 Moderate Car GT object IoU and case evidence.

[L19] `results/current_original_downsampled_upsampling_comparison/current_comparison_report.md`: an earlier cross-entry comparison;Because protocol is different from the current main experiment, only is used to check history and does not enter main table's conclusions.

[L20] `results/pointrcnn_finetune64_six_arms_20260824/reports/eval256_ap_report.md`: 64 frame fine-tuning the results of the screening, not as a final adaptation conclusion.
