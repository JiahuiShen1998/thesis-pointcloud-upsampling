# Detector-Aware Controlled Upsampling: thesis improvement experiment proposal

## 0. Positioning of this proposal

Working name:

**Detector-Aware Sparse Gated Point Injection (DAS-Gate)**

This proposal no longer tries to prove "more points means better detection". It answers:

> How can we keep only the small number of generated points that are useful for detection, while avoiding the contamination that strict x4 upsampling inflicts on the real-point budget, the local neighbourhoods, and the voxel space?

It builds directly on three facts the current experiments have already established:

1. In PointRCNN, PDANS at 2.5% produced a candidate gain of `+2.43` on 256-frame Car 3D Moderate, but performance fell once the generation ratio was raised further.
2. In CenterPoint Line B, PDANS on Pedestrian and PDANS/PU-GCN on Cyclist show positive results over the full validation set.
3. Strict x4 input produces a large number of neighbourhoods or extra voxels that are unsupported by the real scan; benefit and damage occur simultaneously.

The core design is therefore not a new point-generation network, but an auditable **generated-point selection and detector-input adaptation layer**.

---

## 1. Research questions

### RQ1

Can a low ratio of generated points improve PointRCNN stably on the complete KITTI validation set, rather than only on a 256-frame screened subset?

### RQ2

Can geometric consistency and sparsity gating preserve CenterPoint's positive gains on Pedestrian/Cyclist while reducing the Car performance loss?

### RQ3

Do PointRCNN and CenterPoint require different generated-point budgets and retention mechanisms?

### RQ4

Does detector-aware selection outperform:

- random generated-point selection;
- selection by generation ratio alone;
- selection by geometric quality alone;
- a same-ratio real-point replacement control?

---

## 2. Hypotheses

### H1: the low-dose hypothesis

The effect of generated points on detection has a dose range. A small number of high-quality generated points may supply object evidence that sampling missed or downsampling removed; a high ratio of generated points changes neighbourhood and voxel distributions and causes degradation.

### H2: the sparse-object hypothesis

Positive gains occur mainly:

- in Line B;
- where the original object point count is low;
- for small objects such as Pedestrian and Cyclist;
- at mid-to-long range or in partially occluded regions.

The complete Line A scan and high-density nearby Cars do not need unconditional point addition.

### H3: the detector-mechanism hypothesis

- PointRCNN requires strict control over the proportion of generated points within the fixed 16,384-point budget.
- CenterPoint requires control over the newly created voxels, the point count within each voxel, and the shift in voxel means.

The same raw-point selection strategy must not be assumed equivalent for the two detectors.

### H4: the matched-control hypothesis

If a generated-point scheme only beats the baseline but does not beat the same-slot real-point control, the gain comes mainly from a change in sampling coverage. Only the margin above the matched real-point control counts as the net contribution of the generated geometry.

---

## 3. Method overview

Inputs:

- the original or downsampled observed point set `P_obs`;
- the candidate point set `P_gen` produced by an upsampling method;
- a frozen detector;
- no GT boxes, class labels, or evaluation results used for test-time selection.

Outputs:

- a PointRCNN-specific input `P_point`;
- a CenterPoint-specific input `P_voxel`;
- a per-frame generated-point selection manifest;
- every gating score and rejection reason for each candidate point.

Overall pipeline:

`candidate generation → exact observed/generated split → geometric quality filter → sparsity gate → optional detector ROI gate → detector-specific budget → frozen-detector evaluation`

---

## 4. Candidate point quality score

For each generated candidate point `p`, define:

`S(p) = w_surface S_surface + w_range S_range + w_sparse S_sparse + w_roi S_roi`

The first version trains no additional network and uses deterministic rules, to avoid introducing new training variables.

### 4.1 Local surface consistency `S_surface`

1. Find kNN among the observed points.
2. Fit a local PCA plane to the neighbourhood.
3. Compute the candidate point's distance to that local plane.
4. Reject candidates with an excessive plane residual, an unstable normal, or an over-wide neighbourhood span.

Suggested initial parameters:

- `k ∈ {8, 16, 32}`;
- point-to-plane distance threshold `{0.05, 0.10, 0.20} m`;
- maximum neighbourhood XY diameter adapted to range.

This term removes bridging points that span road, vehicle, and background surfaces.

### 4.2 Scan geometry consistency `S_range`

Project the candidate onto the LiDAR azimuth/elevation grid:

1. Check whether adjacent angular bins have observed support.
2. Compare the candidate range against the median range of nearby observations.
3. Reject floating points that sit between a front and a back surface.
4. Keep candidates that extend continuously along an existing surface.

This term does not need the complete original point cloud as an oracle; it uses only the observed points of the current input.

### 4.3 Sparsity benefit `S_sparse`

Add points only where the scene is genuinely sparse:

- the local observed kNN distance is large;
- the current voxel or neighbourhood occupancy is low;
- the candidate would not be a near-duplicate of an existing observed point;
- no injection once a voxel already holds enough real points.

In Line A, most high-density regions should be rejected by this gate; in Line B, sparse small-object regions receive higher priority.

### 4.4 Detector region score `S_roi`

This part is an ablation item for later and is not mandatory in the first version.

At test time, run one low-threshold detection pass on the baseline input first:

- PointRCNN: keep low-threshold RPN proposals or foreground objectness regions;
- CenterPoint: keep low-threshold centre-heatmap peak regions.

Inject points only inside the dilated candidate ROI, then run a second, formal detection pass. No GT is read at any point in this process.

A geometric gate floor must be set, so that low-confidence false proposals cannot pull spurious points into the background.

---

## 5. Separate adaptation for the two detectors

## 5.1 PointRCNN adapter

Goal:

Keep the 16,384-point input unchanged and preserve as much real observation as possible.

### Input strategy

1. Start from the canonical observed-only E2 baseline.
2. Fixed retention ratios:

   - `97.5% observed + 2.5% generated`;
   - `95% observed + 5% generated` as a secondary ablation;
   - do not treat anything above 10% as a main experiment.

3. Generated slots are filled only from candidates that pass the gates.
4. If there are not enough qualified generated points, fill the remaining slots with real observed points rather than forcing low-quality generated points in.
5. The final input is still exactly 16,384 points.

### PointRCNN main controls

- P0: original baseline;
- P1: observed-fill 2.5%;
- P2: PDANS random/generated 2.5%;
- P3: PDANS + geometry gate 2.5%;
- P4: PDANS + geometry + sparsity gate 2.5%;
- P5: PDANS + geometry + sparsity + RPN ROI gate 2.5%.

### PointRCNN main research lines

- Primary: Line A, to verify whether the earlier 256-frame positive point generalises to 3,769 frames.
- Secondary: Line B, to test whether gating at least substantially reduces the loss caused by strict x4.

---

## 5.2 CenterPoint adapter

Goal:

Control the newly created voxels, not the total raw point count.

### Input strategy

1. Keep all real observed points as they are.
2. Filter candidates first by CenterPoint's actual FOV, range, and voxel size.
3. Select at most 1 generated point per empty voxel in the first stage.
4. For voxels that already contain real observations:

   - do not inject by default;
   - or allow 1 high-quality generated point only when the real point count is below 2.

5. Set frame-level and ROI-level budgets on generated voxels.
6. If the pre-cap voxel count approaches 40,000, prioritise real voxels and high-quality generated voxels.

### CenterPoint main controls

- C0: line baseline;
- C1: strict exact-4N input;
- C2: voxel-unique generated candidates;
- C3: C2 + geometry gate;
- C4: C3 + sparsity gate;
- C5: C4 + CenterHead low-threshold ROI gate;
- C6: C5 + class/distance-adaptive budget.

### CenterPoint main research lines

Line B is primary:

- PDANS/Pedestrian;
- PDANS/Cyclist;
- PU-GCN/Cyclist;
- while monitoring Car, so small-object gains are not bought at an unacceptable Car loss.

Line A serves as a safety test: an ideal strategy should be close to a no-op there, rather than continuing to apply unconditional x4.

---

## 6. Experiment phases

## Phase 0: complete the outstanding positive control

This phase is not a new-method experiment; it closes the evidence loop for the current thesis.

### PointRCNN full validation set

Run the Original line only:

1. original baseline;
2. original observed-fill control 2.5%;
3. original PDANS generated 2.5%.

Requirements:

- 3,769/3,769 frames;
- reuse the nested slot order from the 256-frame experiment exactly;
- BBox, BEV, 3D, AOS;
- Easy, Moderate, Hard;
- save all predicted boxes and the input manifest.

Decision rule:

- if PDANS does not beat the full baseline, the 256-frame result was screened-subset noise or selection bias;
- if it beats the baseline but not the control, the gain comes mainly from sampling coverage;
- only if it beats both the baseline and the control can it count as evidence of a net gain from the generated geometry.

## Phase 1: 256-frame development and ablation

Tune DAS-Gate on the existing screen256 subset.

Do not use AP directly to select a large number of thresholds. Recommended:

1. first narrow to 2–3 candidate parameter sets using geometric metrics;
2. then use 256-frame AP for the development choice;
3. once locked, do not look at the full results to tune further.

Development-phase outputs:

- generated-point retention rate;
- generated 0.2 m voxel precision;
- extra voxel fraction;
- reference voxel recall;
- observed/generated point count per object;
- PointRCNN generated-slot utilisation;
- CenterPoint newly created voxel count.

## Phase 2: full 3,769-frame confirmation

Keep at most 3–4 locked configurations per detector, to avoid an unconstrained experimental search.

### PointRCNN full

- P0, P1, P2, P4, P5;
- Car;
- both lines;
- three difficulty levels and BBox/BEV/3D.

### CenterPoint full

- C0, C1, C4, C5;
- Car, Pedestrian, Cyclist reported separately;
- both lines;
- three difficulty levels and BBox/BEV/3D.

## Phase 3: box-level mechanism re-check

Continue with the current three frames:

- `000590`;
- `005625`;
- `006682`.

For baseline, strict x4, and DAS-Gate, produce:

- full 3D/BEV comparison;
- lost/recovered/localization/confidence transitions;
- single-object crops;
- three-colour visualisation of observed points, rejected generated points, and accepted generated points.

---

## 7. Ablation matrix

| ID | Observed-first | Fixed low ratio | Geometry | Sparsity | Detector ROI | Purpose |
|---|---:|---:|---:|---:|---:|---|
| A0 | — | — | — | — | — | line baseline |
| A1 | ✓ | — | — | — | — | strict/current adapter reference |
| A2 | ✓ | ✓ | — | — | — | the low ratio by itself |
| A3 | ✓ | ✓ | ✓ | — | — | contribution of geometric quality |
| A4 | ✓ | ✓ | ✓ | ✓ | — | contribution of sparse-region selection |
| A5 | ✓ | ✓ | ✓ | ✓ | ✓ | contribution of detector awareness |
| A6 | ✓ | ✓ | ✓ | ✓ | ✓ | class/distance budget |

A2 must be kept, otherwise it is impossible to tell whether an improvement comes from "the ratio was lowered" or from the gating mechanism.

---

## 8. Evaluation metrics

## 8.1 Detection metrics

The two detectors must be reported separately.

### PointRCNN

- Car BBox AP_R40 Easy/Moderate/Hard;
- Car BEV AP_R40 Easy/Moderate/Hard;
- Car 3D AP_R40 Easy/Moderate/Hard;
- AOS;
- recall@0.3/0.5/0.7;
- lost/recovered/localization/confidence transitions.

### CenterPoint

Reported separately for Car, Pedestrian, and Cyclist:

- BBox AP_R40 Easy/Moderate/Hard;
- BEV AP_R40 Easy/Moderate/Hard;
- 3D AP_R40 Easy/Moderate/Hard;
- recall@0.3/0.5/0.7;
- prediction count and score ≥ 0.5 prediction count per class;
- lost/recovered/localization/confidence transitions.

## 8.2 Geometry and input-mechanism metrics

- candidate acceptance ratio;
- accepted generated point count;
- accepted generated voxel count;
- generated 0.2 m voxel precision;
- reference 0.2 m voxel recall;
- extra voxel fraction;
- mixed observed/generated voxel fraction;
- PointRCNN actual generated-slot ratio;
- CenterPoint pre-cap voxel count and cap hits;
- change in per-object point count by distance band and by class.

## 8.3 Computational cost

- gating time per frame;
- time for the first and second detection passes;
- peak memory;
- final input point/voxel count.

---

## 9. Statistics and reproducibility

1. All selection uses a fixed seed and a stable hash by default.
2. Where random sampling is involved, run at least 3 seeds for the final candidate configuration.
3. Recompute 95% confidence intervals on AP differences using a frame-level paired bootstrap.
4. Full-run parameters must be locked during the 256-frame phase.
5. Save with every output:

   - the source point-cloud path;
   - the exact observed/generated relationship;
   - the score of each candidate point;
   - the accept/reject reason;
   - the final point count;
   - the detector config hash;
   - the code commit or working-tree state.

---

## 10. Success criteria

## 10.1 Minimum scientific success

Even without beating the baseline, the following still constitutes a valid thesis result:

- a significant reduction in AP loss relative to strict x4;
- geometric metrics and box-level transitions improving in step;
- ablations that demonstrate the individual roles of the geometry, sparsity, and detector gates;
- the two detectors exhibiting explainably different optimal strategies.

## 10.2 Strong success for PointRCNN

Priority criteria:

- full 3,769-frame Original/PDANS g2.5 beats the baseline;
- and, where possible, beats the matched observed-fill control;
- at least two of Easy/Moderate/Hard agree in direction;
- no large BBox/BEV degradation that contradicts the 3D gain.

## 10.3 Strong success for CenterPoint

- preserve or extend the Line B Pedestrian/Cyclist positive gains;
- while reducing the Car loss relative to strict x4 by at least 30%;
- accepted generated voxel precision clearly higher than for unfiltered input;
- extra voxel fraction clearly reduced.

## 10.4 Safety criteria

- Line A should not keep adding large numbers of generated voxels unconditionally;
- the recommended limit is to keep the Line A 3D Moderate degradation within 1 AP of the baseline;
- if candidate quality is insufficient, the algorithm is allowed to return the observed-only input.

---

## 11. Implementation suggestions

Add new code rather than modifying old experiment results:

### `scripts/build_detector_aware_point_inputs.py`

Responsibilities:

- split observed/generated exactly;
- compute the surface/range/sparsity/ROI scores;
- build the PointRCNN and CenterPoint inputs separately;
- emit a per-frame manifest.

Main interface:

```text
--detector pointrcnn|centerpoint
--line A|B
--method pdans|pu_gcn
--policy quota|geometry|geometry_sparse|geometry_sparse_roi
--generated-ratio 0.025
--split-file ...
--output-dir ...
```

### `scripts/run_detector_aware_upsampling_ablation.py`

Responsibilities:

- run the 256-frame development matrix;
- lock configurations;
- run the full 3,769 frames;
- skip existing PASS results;
- keep the two detectors' output directories isolated.

### `scripts/analyze_detector_aware_upsampling.py`

Responsibilities:

- parse BBox/BEV/3D/AOS and the three difficulty levels;
- compute box status transitions;
- summarise geometry, voxel, and point budgets;
- produce separate PointRCNN and CenterPoint reports.

### Suggested results directory

```text
results/detector_aware_controlled_upsampling_YYYYMMDD/
├── pointrcnn/
│   ├── inputs/
│   ├── manifests/
│   ├── eval_outputs/
│   ├── tables/
│   ├── figures/
│   └── report.md
└── centerpoint/
    ├── inputs/
    ├── manifests/
    ├── eval_outputs/
    ├── tables/
    ├── figures/
    └── report.md
```

---

## 12. Relationship to the PU-Net and shared-patch fixes

The PU-Net fix does not block the first stage of DAS-Gate, because that stage can use PDANS, currently the highest-quality method, and PU-GCN, which has a positive Cyclist result.

But if the thesis is to claim "a fair comparison of four upsampling methods", one of the following is required:

1. fix the PU-Net normalisation/inverse transform and the shared-patch locality, then re-evaluate;
2. or explicitly label the current PU-Net as an `adapter-defective diagnostic result` and remove it from the fair model-ranking main table.

The FPS+kNN fix for the shared patch should be ablated on at least PU-GCN or PU-EdgeFormer, to answer:

> Does the current degradation come from the upsampling model itself, or from a non-local KITTI patch adaptation?

So it is necessary for any claim about *method fairness*, but it is not a prerequisite for first verifying whether DAS-Gate can improve PDANS/PU-GCN inputs.

---

## 13. Final contributions the thesis can make

If the experiments succeed, the thesis can make contributions at three levels:

1. **Systematic finding**: strict point-count upsampling is not equivalent to an increase in effective detection evidence.
2. **Mechanistic evidence**: PointRCNN is affected by the fixed point budget and neighbourhood contamination; CenterPoint is affected by extra voxels and voxel-feature shift.
3. **Methodological contribution**: DAS-Gate, which injects a small number of generated points only at trustworthy, sparse, and detection-relevant locations.

Even without beating the baseline across the board, as long as DAS-Gate significantly reduces the strict-x4 loss and the ablations demonstrate why, it moves the thesis from "a report of a failure phenomenon" to "a verifiable mechanism plus a corrective method".
