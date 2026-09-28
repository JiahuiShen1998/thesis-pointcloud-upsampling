# The five candidate methods for "upsampling" - a feasibility consultation + 20 frame scale interpretation test

Date: 2026-08-12
Candidates: PUDet, GFAS, DAPU, PDANet, TULIP
Request: A small-scale validation experiment of 20 frame for these methods

## Summary of conclusions

1. **In five ways, only TULIP can run, and it's over.** - Full 3769 frame × two detector, 2026-05/07 complete.It's the worst of all the methods detected by this project (CenterPoint Car Moderate 3D: 31.77 vs original 79.28)**−47.5 AP**).
2. **PUDet / GFAS / DAPU no public code;PDANet with code but no weight**(README) checkpoint is a word volume `***.pth`, 0 release)So the four methods of "run 20 frame" can't be implemented today.
3. **PUDet / GFAS / PDANet is not the same question as this project.**. All three are detector internal modules + joint training, not pre-plugged upsampler.Even if the re-emergence is successful, cannot is used to counter the failure of the line "frozen detector + for input only".
4. **20 frame, this size cannot be interpreted by itself.**(This exercise, see below).20 frame with matching delta standard deviation: **6.02 AP**, 95% Minimum detectable effect **≈11.8 AP**. And the added value of these methods is + 1.2 / +1.84 / +6.89 AP —— 20 frame, not even the biggest one.
5. **The only thing worth investing in is DAPU.**: It is a non-learning geometric rule method (no checkpoint, can be repeated by dissertation) and is the only one who really meets the "one-way + frozen detector + label-free" candidate.But it claims + 1.2 AP **below Evaluator of this project sampling Noise 1.09 AP**, so it needs to be interpreted in full + more seed.

## I. METHODOLOGICAL APPLICATION

| Methodology | Type | Code | Weight | Do you need to retrain detector? | Is it pre-processed upsampling? | We can run. |
|---|---|---|---|---|---|---|
| **TULIP** | Range-image LiDAR (CVPR 2024) | Local `~/projects/external/TULIP` | ✅ `trained/tulip_kitti.pth` (311 MB) | Yes | ✅ | **Full run.** |
| **PDANet** | IA-SSD Source detector (TGRS 2024) | ✅ `Geo3DSmart/PDANet` | README Written `***.pth`, 0 releases | **Yes.**(from zero training) | It's detector. | Yes |
| **PUDet** | Generating upsampling Embedded CT3D | Unfinished | ❌ | **Yes.**(joint training) |  /  detector internal module | Yes |
| **GFAS** | Chart feature enhancement sampling Layer | Unfinished | ❌ | **Yes.**(supervised with foreground / background label) | detector inside sampling Layer | Yes |
| **DAPU** | Ruler geometric upsampling (Neurocomputing 2026-01) | Unfinished | **I don't need it.**(non-learning) | Thesis is not explicit. | ✅ | **Reachievable** |

Retrieval method: GitHub repositories/code search (multiple keywords), GitHub releases API, Web search, journal page.PUDet is published in North Technology (Chinese periodical) and GFAS in the Journal of Engineering Sciences, neither of which provides a code link.

### Three structural problems of detector internal methods

PUDet (LDEM/DDAM embedded in CT3D), GFAS (replacement of detector sampling floor, supervised by foreground / background label), PDANet (PDA-SSD, OpenPCDet 0.5 + IA-SSD) - all three benefits are derived from the ""**Retrain a detector with an extra module**♪ The gain, not the "**A better input for frozen detector**I'm not sure what I'm talking about.

protocol 5 of this project is "Replace only input without retrainment" (see `upsampling-experiment-protocol`). So:

- Recreate them.**cannot**"upsampling Can cannot for frozen detector"
- And their success.**Does not constitute**Responding to the negative results of the project - the two measures are different;
- In turn, they happen to be a useful argument:**Currently all published positive gains on KITTI require training signals to enter upsampling / sampling module**. This is in itself a dissertationable conclusion.

### The GFAS numeric credibility hint

Abstract Car Upgrade + 6.2 / +6.89 / +8.58 (Easy/Mod/Hard), but**No baseline absolute value given**Nor does it indicate whether 3D is BEV, IoU threshold, val or test.If the baseline is the standard PointRCNN (Car Moderate 3D  /  78–82), + 6.89 will fall to 85–89, a significant above KITTI val.The more likely explanation is that the baseline is a weakened sampling variant (e.g. random sampling contrast).**The absolute baseline value should be obtained before the input is recovered.**

## TULIP: Full results of completed

TULIP already runs and is fully certified (3769/3769 frame, 0 missing, 0 redundant):

- Generation:`~/projects/external/TULIP/results/tulip_upsampling_only/` (Line A `tulip_original_up`, Line B `tulip_downsampled_up`)
- detector input:`data/KITTI/object/training/tulip_{original,downsampled}_up_bin`(Both 3769)
- Every frame point count: Line A 29,795–51,404;Line B 26,366–47,528 (= 64×1024 Inverse projection of distance map, not exact-4N)
- Strength retained (mean  /  0.20, non-Zero 34.5k/46.8k), not minus zero

### AP Results

**CenterPoint (Standard config, in direct comparison with main table and protocol)** —— 3D AP_R40 Moderate:

| Enter | Car | Ped | Cyc |
|---|---:|---:|---:|
| Original point cloud (baseline) | **79.28** | 50.65 | 64.61 |
| TULIP Line A | 31.77 (**−47.5**) | 15.98 (−34.7) | 4.59 (−60.0) |
| TULIP Line B | 17.68 | 2.99 | 0.08 |

Source:`results/centerpoint_tulip_native_extended_20260729/full_ap_summary.csv`

**PointRCNN** —— Car 3D AP_R40 Moderate: Line A 35.02, Line B 16.97.

⚠️ **PointRCNN  Two.  TULIP  It's not standard.  config**: `RPN.NUM_POINTS=4096` + `TEST.RPN_DISTANCE_BASED_PROPOSE=False`(Or else the default 16384 will crash, the long distance proposal branch empty-spaced NMS will signal 6).The no and config pair baselines in this directory, so **35.02 cannot Direct minus 82.26**. See `tulip_line_b_..._full_validation/compatibility_note.txt`. CenterPoint that group no this question is a clean contrast.

**Conclusions**: TULIP is catastrophic on two detectors, far above any noise interpretation ( - 47.5).It ranks after PDANS/PU-GCN/PU-EdgeFormer, on the same level as PU-Net.**No more 20 frame validation required.**

## iii. 20 frame-scale interpretation capability (newly measured)

### Methodology

Do a full 3769 frame PointRCNN projection**resampling re-evaluate**: randomly smoke n frame subset, recalculate Car 3D AP_R40 on the subset, repeat 200 times, and see the distribution of the estimates themselves.Read only available txt and label, without changing any detector input, which is "analytical" rather than "input construction" and does not touch the red line.

Script `scripts/subset_size_power_probe.py`, original result `analysis/subset_size_power_probe.json`.
Full reference value (achieved by this evaluator R40): original 81.99, downsampling 65.94, downsampling +PDANS 39.03.

### Result

| frame Number | Car Moderate GT in subset | One-armed AP standard deviation | One-armed 5–95% (real 81.99) | **Match delta standard deviation** | **MDE₉₅** |
|---:|---:|---:|---|---:|---:|
| **20** | 41 | 11.91 | 52.0 – 89.9 | **6.02** | **11.80** |
| 50 | 105 | 3.97 | 75.9 – 88.8 | 3.71 | 7.27 |
| 100 | 209 | 3.02 | 76.5 – 86.2 | 2.87 | 5.63 |
| 256 | 533 | 2.07 | 78.1 – 85.2 | 1.75 | 3.44 |
| 512 | 1068 | 1.39 | 79.7 – 83.3 | 1.26 | 2.46 |

MDE₉₅ = 1.96 ×  Match  delta  Standard deviation  =  A single experiment can be arranged.  95%  Determination 「 Non-zero 」 The minimum real effect.

 Press  sd ∝ 1/√n  Expulsion.  256  frame: 768  frame  → MDE₉₅ ≈ 1.98; 1500 frame  /  1.42;3769 frame  /  0.90 AP.

### Reading

- **Under 20 frame, an input of a real value 81.99 found that AP had a 90% probability between 52 and 90.** It's not "More noise," it's the basic no information.The reasons for this are straightforward: only about 41 Car Moderate GT are included in 20 frame, while AP_R40 is to be valued at 40, which is equal to one recall grid for each target.
- **20 frame can only see the effects of  /  11.8 AP.** The gains claimed against these four methods are:

| Methodology | Declaring gains | Can you see 20 frame? | How much do you need frame? |
|---|---|---|---|
| DAPU | +1.2 AP 3D | ❌ | Total 3769 is still on edge (MDE  /  0.9 + seed Noise Consider 1.09) |
| PUDet | +1.84 mAP | ❌ |  /  1500 frame with additional seed |
| GFAS | +6.89 Car Mod | ❌ | 100 frame |
| TULIP (actual) | −47.5 AP | That's more than enough. | 20 frame is fine. |

- **Attention to the second source of noise.** The above table is just a sample of frame noise.`pointrcnn-sampling-noise-floor` I've detected sampling noise from the evaluation machine itself. **σ ≈ 1.09 AP**(4 seed, native path 3D Moderate)**Does not decline with frame**.  When the two are combined, the full list.  seed  Actual  MDE₉₅ ≈ 2.1–2.4 AP.
- **Inference: DAPU + 1.2 and PUDet + 1.84 cannot be read at the current measurement accuracy of the project, even if the method is fully effective.** To interpret this scale, you have to do more than seed on average to press down.

## Recommendations

**Don't do the 20 frame validation.** It's redundant for the one that can run (TULIP) (full completion);It is not enforceable for the remaining four;And even if it is enforceable, this scale does not see the effects they claim.

Sort by value for money:

1. **DAPU is the only candidate worthy of input.** Reason: (c) a) non-learning, without checkpoint, can be duplicated by the three modules of the paper - Ground Points Recognizer (Identified on the ground by the high/non-cosmetic differential of patch), Distribution-Aware Patch KNN (sampling radius by distributed dynamic), Neighbors Upsampling (linear inter-neighbour interply);(b)) It's pure front processing, natural satisfying one-way + frozen detector;(c)) Its three modules happen to be in place. `forward-only-upsampling-pipeline` It's already listed.**Legal**geometric (local PCA co-exist, local kNN density and neighbourhood scale, highly filtered/non-densification ground).
   - Cost: Re-achievement (without code) is required and interpretation requires full + more seed.
   - By-product value: Even if DAPU is not effective, "no densification ground + local density self-adaptation" is a lever that has not been separately measured for this project.

2. **First seed average, then anything. <3 AP Conclusions.** It's all a little conclusion of prerequisite's condition, and the cost is lower than any replicating method.

3. **PUDet / GFAS / PDANet is classified as related work and is not reproduced.** No code/no weight and belonging to the category "detector" is different from the issue of this project.This observation can be used in the paper:**The positive gains published on KITTI basically require a training signal to enter upsampling / sampling module**— This is just in support of your existing negative results, not in weakening them.

4. **GFAS if you still want to pursue**, obtain the absolute baseline value from the author for the reasons § 1.

## Annex: Products

- `scripts/subset_size_power_probe.py` - resampling probe.`openpcdet_centerpoint` Environmental running, numba + GPU)
- `analysis/subset_size_power_probe.json` – Complete results of 200 times x 5 scales
