# ModelNet40 Point-Cloud Upsampling and PointNet++ Classification: Complete Experimental Dossier

> **Purpose**: Organize the research record and prepare the thesis and defence.
> **Original project root**: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`
> **Original compilation date**: 2026-08-04
> **Primary numerical sources**: `pointnet2_results/x4_two_line_final/**/metrics.json`, `reports/modelnet40_geometry_equal_n_*`, `reports/modelnet40_pointnet2_final_two_line_classification_summary_with_pu_edgeformer.*`
> **Companion PDF**: [ModelNet40_Thesis_Complete_Dossier.pdf](ModelNet40_Thesis_Complete_Dossier.pdf). Paths below refer to the original project unless stated otherwise; see the archive README for the current directory layout.

---

## Main finding

Under a common **×4 two-line** protocol on ModelNet40, five upsampling methods were compared with **PointNet++ trained from scratch for each input variant**. Better geometry did not automatically improve classification: **every Line A (densification) result was below Original 1024**, and **only PU-Net in Line B (sparse-input recovery) slightly exceeded the downsampled baseline (+0.42 pp)**, while remaining below Original.

---

## Contents

1. [1. Background and research questions](#1-background-and-research-questions)
2. [2. Evaluation principles](#2-evaluation-principles)
3. [3. Canonical experimental protocol](#3-canonical-experimental-protocol)
4. [4. Methods and technical background](#4-methods-and-technical-background)
5. [5. Classification results](#5-classification-results)
6. [6. Equal-N geometry results](#6-equal-n-geometry-results)
7. [7. Geometry and classification](#7-geometry-and-classification)
8. [8. Mesh-ref controls](#8-mesh-ref-controls)
9. [9. Protocol history](#9-protocol-history)
10. [10. Implementation and work record](#10-implementation-and-work-record)
11. [11. Ablations and supplementary results](#11-ablations-and-supplementary-results)
12. [12. Thesis structure and usable conclusions](#12-thesis-structure-and-usable-conclusions)
13. [13. File index](#13-file-index)
14. [Appendix A. Training settings and data paths](#appendix-a-training-settings-and-data-paths)
15. [Appendix B. Code and checkpoint sources](#appendix-b-code-and-checkpoint-sources)

---

## 1. Background and research questions

Point-cloud upsampling aims to produce a denser, geometrically more complete point set from a sparse point cloud. Much of the literature evaluates Chamfer Distance (CD), Hausdorff Distance (HD), and point-to-surface distance (P2F), assuming that increased density and better geometry benefit downstream three-dimensional tasks.

This project tests that assumption under the **official ModelNet40 classification setting**: does Overall Accuracy (OA) improve when the upsampled output is supplied to **PointNet++**?

### 1.1 Research questions

- **RQ1**: Does ×4 densification of an already well-sampled Original 1024 input to 4096 points improve PointNet++ classification?
- **RQ2**: After ×4 downsampling to 256 points, can recovery to 1024 points recover lost classification performance?
- **RQ3**: Is the method with the best geometric metrics also the best for downstream classification?
- **RQ4**: Can increased point count alone, tested by directly sampling 4096 mesh points, explain the classification changes?

### 1.2 Scope

| Item | Setting |
|---|---|
| Task | ModelNet40 shape classification; **not** KITTI detection or AP |
| Downstream model | PointNet++ SSG (`pointnet2_cls_ssg`), XYZ input only |
| Upsamplers | Inference with existing pretrained weights; **the upsampling networks are not retrained in these experiments** |
| Classifier | Trained **from scratch** for each point-count setting, without fine-tuning the 1024-point weights |
| Main ratio | **R=×4** |
| TULIP | Range-image / LiDAR method; **supplementary only**, excluded from the main xyz comparison |

**Contribution**: Comparing five methods under the same ×4 two-line protocol shows why geometric recovery and classification utility must be reported separately. Densification did not help in this setting; only PU-Net produced a marginal recovery benefit, still below Original.

---

## 2. Evaluation principles

### 2.1 Two-line design

The two experimental lines distinguish densification of a well-sampled input from recovery of a sparse input. **Cross-line comparisons do not establish the main conclusions.**

| Experimental line | Purpose | Input path | PointNet++ input points |
|---|---|---|---|
| **Line A Densification** | Densify well-sampled data | Original 1024 → Up ×4 | baseline 1024 / up 4096 |
| **Line B Recovery** | Recover after sparsification | Original → Down×4(256) → Up×4 | baseline 256 / up 1024 |

### 2.2 Mandatory point-count protocol

1. **Keep each baseline at its native point count**: do not pad or resample it to the upsampled output size.
2. **Keep each upsampled output at its output point count**: do not crop it back to the baseline size.
3. **Do not artificially match baseline and upsampled point counts**; doing so confounds the upsampling effect with the point-count effect.
4. **Within a line and branch**, use the same upsampling ratio and output point count.
5. **The Original and Downsampled lines are independent**; there is no requirement to match point counts across lines.

### 2.3 Classification metrics

- Dataset: the official ModelNet40 split, **9843 train / 2468 test**, with **no independent validation set** for classification, following the recorded PointNet++ setup.
- **Best Overall Accuracy** is the highest overall (instance) accuracy observed during training. This implementation saves the best checkpoint when test accuracy improves.
- **Final Overall** is the overall accuracy at epoch 200, reported as supplementary information.
- Best OA is the primary reported metric; differences are in **percentage points (pp)**.
- For Line B, the primary Δ is relative to the Downsampled ×4 baseline; the secondary gap is relative to Original.

### 2.4 Geometric metrics: distinct reference protocols

| Version | Reference | Role | Interpretation |
|---|---|---|---|
| **Equal-N (current)** | A: mesh-ref 4096; B down: mesh-ref 256; B up: Original 1024 | Main geometry table | Use equal-cardinality CD/HD for the intended comparison |
| vs Original (unequal cardinality) | 4096 vs 1024, etc. | Historical / supplementary | Measures agreement between discrete sets, not continuous-surface accuracy |
| Earlier mesh / exact P2F | dense mesh surface | Computed in earlier work | Removed from the main narrative in the 2026-08 PPT revision |

**Metric definitions:**

- **CD (Chamfer)**: the sum of mean nearest-neighbor distances in both directions, measuring overall agreement.
- **HD (Hausdorff)**: the worst nearest-neighbor discrepancy, sensitive to outliers.
- **NUC**: point-distribution non-uniformity; use relative comparisons within the protocol, with lower values generally indicating greater uniformity.
- **Mesh-ref sampling**: triangle **area-weighted** sampling from `.off` meshes followed by **unit-sphere** normalization; `seed=stable_seed(42, ...)`.
- Mesh-ref is **sampled independently** from the same mesh family, rather than downsampled from Original 1024.

---

## 3. Canonical experimental protocol

The main protocol was finalized from 2026-07 and supplemented with equal-N / mesh-ref baselines in 2026-08. Historical ×2 / 512-point experiments are ablations only; see Section 11.

| Line | Branch | Points on disk | `num_point` | `allow_resample` | Role |
|---|---|---:|---:|---|---|
| A | Original baseline | 1024 | 1024 | false | Main baseline |
| A | Original + EAR / PDANS / PU-Net / PU-GCN / EdgeFormer | 4096 | 4096 | false | Main upsampling |
| B | Downsampled ×4 baseline | 256 | 256 | false | Main baseline |
| B | Down ×4 + the same five methods | 1024 | 1024 | false | Main upsampling |
| — | Mesh-ref 256 | 256 | 256 | false | Sampling-process control |
| — | Mesh-ref 4096 | 4096 | 4096 | false | Dense mesh-sampling control |

### Shared training settings

| Item | Value |
|---|---|
| Model | `pointnet2_cls_ssg` |
| Number of classes | 40 |
| Input features | XYZ only |
| epoch | 200 |
| optimizer | Adam |
| learning_rate | 0.001 |
| decay_rate | 0.0001 |
| batch_size | 24 |
| seed | 42 |
| num_workers | 4 |
| Configuration directory | `configs/pointnet2_x4_two_line/*.yaml` |
| Results directory | `pointnet2_results/x4_two_line_final/**/metrics.json` |

---

## 4. Methods and technical background

| Method | Type | Principle | Paper / source |
|---|---|---|---|
| **EAR** | Classical geometry, non-learning | Samples away from sharp features to estimate reliable normals, then progressively upsamples near edges; preserves sharp features and controls density | Huang et al., ACM TOG 2013 — *Edge-aware point set resampling* |
| **PU-Net** | Learned (TF1) | Multi-level point features and multi-branch feature-space expansion; a pioneering learned point-cloud upsampler | Yu et al., CVPR 2018 — [arXiv:1801.06761](https://arxiv.org/abs/1801.06761) |
| **PU-GCN** | Learned (TF1) | Graph convolutions model local geometric relationships; often performs well on CD | Qian et al., CVPR 2021 — [arXiv:1912.03264](https://arxiv.org/abs/1912.03264) |
| **PDANS** | Learned (PyTorch) | Conditional diffusion upsampling with Adaptive Noise Suppression / Tree-Trans, emphasizing noise robustness and detail | CVPR 2025 — [github.com/Baty2023/PDANS](https://github.com/Baty2023/PDANS) |
| **PU-EdgeFormer** | Learned (TF) | EdgeFormer combines local EdgeConv with global multi-head self-attention | Kim et al., ICASSP/arXiv 2023 — [arXiv:2305.01148](https://arxiv.org/abs/2305.01148) |
| **PointNet++** | Downstream classification | Hierarchical sampling and local PointNet aggregation; SSG is used here | Qi et al., NeurIPS 2017; yanx27 PyTorch implementation |
| **ModelNet40** | Dataset | Point clouds sampled from CAD meshes, with the official train/test split | Wu et al., CVPR 2015 |
| **TULIP** | Supplementary only | LiDAR range-image upsampling; incompatible with the main xyz point-count comparison | ETHZ ASL TULIP; **excluded from the main table** |

### Implementation and inference notes

- **EAR**: deterministic script without a checkpoint; migrated from the lab to HPC.
- **PU-Net / PU-GCN / PU-EdgeFormer**: TensorFlow with custom CUDA/tf_ops; matching build environments are required.
- **PDANS**: PyTorch with PyTorch3D / Chamfer / pointops and large `.pkl` checkpoints (PU1K / PUGAN).
- **PU-EdgeFormer**: integrated on 2026-07-16, with an authenticity audit covering the model path, checkpoint, differences from PU-GCN output, and absence of leakage.
- All main methods generated the complete **12311** samples (9843+2468) on both lines before point-count / NaN checks and PointNet++ evaluation.

---

## 5. Classification results

### 5.1 Line A — Densification relative to Original 1024

| Method | Points | Best OA | Final OA | Best Class | Δ Best vs Orig (pp) |
|---|---:|---:|---:|---:|---:|
| Original baseline | 1024 | **91.95%** | 91.31% | 88.00% | 0.00 |
| PU-GCN | 4096 | 91.63% | 91.00% | 87.74% | −0.32 |
| PDANS | 4096 | 91.62% | 90.87% | 88.44% | −0.33 |
| EAR | 4096 | 91.48% | 90.85% | 88.19% | −0.47 |
| PU-Net | 4096 | 90.95% | 90.55% | 87.46% | −1.00 |
| PU-EdgeFormer | 4096 | 90.54% | 90.13% | 86.35% | −1.41 |

**Interpretation**: No 4096-point upsampled branch exceeded Original 1024. PU-GCN (−0.32 pp) and PDANS (−0.33 pp) were closest, while PU-EdgeFormer had the largest decline (−1.41 pp). In this setting, densifying already well-sampled CAD point clouds did not reliably improve PointNet++ classification.

### 5.2 Line B — Recovery relative to Down 256, with the gap to Original

| Method | Points | Best OA | Final OA | Δ vs Down (pp) | gap vs Orig (pp) |
|---|---:|---:|---:|---:|---:|
| Downsampled ×4 | 256 | 90.85% | 90.46% | 0.00 | −1.10 |
| **PU-Net** | 1024 | **91.27%** | 90.65% | **+0.42** | −0.68 |
| PDANS | 1024 | 90.34% | 89.46% | −0.51 | −1.61 |
| PU-GCN | 1024 | 90.06% | 89.63% | −0.79 | −1.89 |
| PU-EdgeFormer | 1024 | 89.32% | 88.58% | −1.53 | −2.63 |
| EAR | 1024 | 88.81% | 88.02% | −2.04 | −3.14 |

**Interpretation**: Restoring point count did not automatically restore classification performance. Only PU-Net improved over the sparse baseline (+0.42 pp), while remaining 0.68 pp below Original. EAR had the weakest classification result (−2.04 pp). Returning to 1024 points therefore did not necessarily restore semantic separability.

---

## 6. Equal-N geometry results

- Split: test, N=2468
- Script: `scripts/compute_geometry_equal_n.py`
- Outputs: `reports/modelnet40_geometry_equal_n_*.{csv,md,json}`

### 6.1 Line A: 4096 vs Mesh-ref 4096; lower is better

| Method | CD | HD | NUC | ΔNUC |
|---|---:|---:|---:|---:|
| Mesh-ref 4096 (self) | 0.000000 | 0.000000 | 0.641852 | 0.000 |
| PU-GCN | 0.042306 | 0.093219 | 0.520398 | −0.121 |
| PU-EdgeFormer | 0.046666 | 0.110932 | 0.675435 | +0.034 |
| PDANS | 0.046821 | 0.113510 | 0.661176 | +0.019 |
| PU-Net | 0.049719 | 0.111177 | 0.605211 | −0.037 |
| EAR | 0.054722 | 0.115396 | 0.815631 | +0.174 |

### 6.2 Line B: Down 256 vs Mesh-ref 256; Ups 1024 vs Original 1024

| Method | Points | Reference | CD | HD | NUC |
|---|---:|---|---:|---:|---:|
| Mesh-ref 256 (self) | 256 | Mesh-ref 256 | 0.000 | 0.000 | 2.126 |
| Downsampled ×4 | 256 | Mesh-ref 256 | 0.133 | 0.205 | 2.093 |
| Original (self) | 1024 | Original 1024 | 0.000 | 0.000 | 1.084 |
| PU-GCN | 1024 | Original 1024 | 0.0558 | 0.1514 | 1.524 |
| PU-EdgeFormer | 1024 | Original 1024 | 0.0619 | 0.1776 | 1.123 |
| PDANS | 1024 | Original 1024 | 0.0627 | 0.1905 | 1.053 |
| PU-Net | 1024 | Original 1024 | 0.0646 | 0.1083 | 1.080 |
| EAR | 1024 | Original 1024 | 0.0767 | 0.1932 | 0.965 |

**Key observations**: PU-GCN generally had the best CD on both lines, but its Line B NUC deteriorated to 1.524. PU-Net had the best Line B HD (0.108) and NUC close to Original, consistent with its unique classification gain. EAR could have a lower NUC while still performing poorly on CD/HD and classification.

> **Note**: Unequal-cardinality CD/HD comparisons (4096 vs 1024 and 256 vs 1024) are deliberately excluded from the main ranking.

---

## 7. Geometry and classification

| Observation | Evidence | Implication |
|---|---|---|
| Better geometry does not imply better classification | In Line B, PU-GCN has the lowest CD, but lower Best OA than PU-Net | Report geometry and task metrics separately |
| Densification did not help | All Line A 4096-point methods are below Original 1024 | The densification hypothesis is unsupported in this setting |
| Recovery is limited | Only PU-Net gains +0.42 pp and remains below Original | Upsampling does not fully offset the ×4 information loss |
| Distribution matters | PU-Net has more balanced HD/NUC; PU-GCN NUC deteriorates | PointNet++ is sensitive to local structure and distribution |
| Point count alone is a weak explanation | Mesh-ref 4096 is only +0.05 pp above Original | Increased density alone does not explain the effect |

---

## 8. Mesh-ref controls

To examine whether classification changes arise merely from point count or sampling procedure, mesh-ref 256 and mesh-ref 4096 were independently sampled from `.off` meshes with area weighting. PointNet++ was trained from scratch with the same hyperparameters.

| Method | Points | Best OA | Final OA | best epoch | Control | Control Best | Δ pp |
|---|---:|---:|---:|---:|---|---:|---:|
| Mesh-ref 256 | 256 | 90.88% | 90.38% | 129 | Down ×4 256 | 90.85% | +0.03 |
| Mesh-ref 4096 | 4096 | 92.00% | 91.18% | 148 | Original 1024 | 91.95% | +0.05 |

**Interpretation:**

1. At the same point count, direct mesh sampling and downsampling Original gave nearly identical classification (+0.03 pp).
2. Denser mesh-ref 4096 was only +0.05 pp above Original 1024, with no substantial advantage.
3. The consistent deficit of Line A upsampled branches is therefore more plausibly associated with their geometric / distributional properties than with the number 4096 itself.

Report: `reports/modelnet40_pointnet2_mesh_ref_baseline_results.md`

---

## 9. Protocol history

| Stage | Change | Current status |
|---|---|---|
| Early work | KITTI + CenterPoint/PointRCNN detection; migration of methods and data from lab to HPC | Parallel engineering background, outside the main results here |
| 2026-06-15 | Established `modelnet40_experiments` and `method_registry` | Asset index retained |
| 2026-06-20+ | Built the main `modelnet40_pointnet2_upsampling` pipeline | Current project root |
| 2026-06-25 | Main ratio changed from ×2 to ×4; earlier 512→1024 EAR experiment became an ablation | Implemented |
| Subsequent finalization | Line B changed to **256→1024** recovery; 512→2048 ceased to be the main line | Canonical |
| Implementation | Fixed CUDA/TF ops, the PDANS environment, PU-Net timeout/resume, and PU-GCN stalls | See `reports/` |
| 2026-07-16 | Integrated PU-EdgeFormer after a successful authenticity audit | Fifth method |
| 2026-07-12~17 | Finalized thesis tables, figures and interpretations; clarified geometry vs classification | Basis of the results chapter |
| 2026-08-03 | Recomputed equal-N mesh-ref geometry and completed mesh-ref classifier training | Main geometry protocol |
| 2026-08-04 | Removed the main mesh/P2F narrative from the PPT; prepared Full Report and emphasized training from scratch | Presentation version |

### Corrections relevant to limitations and appendices

1. R=×2 was briefly recommended, then changed to ×4 to match the default of the PU methods and the experimental objective.
2. Early Line B used `downsampled50` (about 512 points), later replaced by `downsampled_x4` (256 points) for strict ×4 recovery.
3. The baseline loader was previously resampled to 1024 in Step 6. This is now classified as a **naive ablation**, not a main baseline.
4. The geometric narrative changed from dense mesh + P2F to **equal-N**, avoiding misleading rankings from unequal-cardinality CD/HD.
5. Best OA is explicitly defined as the highest recorded checkpoint overall accuracy, without a separate validation set.

---

## 10. Implementation and work record

### 10.1 Data and generation

- Original 1024 point clouds were sampled from ModelNet40 meshes and normalized to the unit sphere.
- Downsampled ×4: `datasets/modelnet40_downsampled_x4/` (256 points, 12311 shapes).
- Line A/B outputs passed strict point-count checks (4096 or 1024 expected) with zero NaN/Inf values.
- PointNet++ inputs were organized under `pointnet2_inputs/` using symbolic links and preparation scripts.

### 10.2 Failures and repairs

- **PDANS**: an early job failed with `torch.cuda.is_available()=False`; aligning modules and `LD_LIBRARY_PATH` resolved the issue.
- **PU-Net / PU-GCN**: TF ops depend on compatible CUDA build and runtime environments.
- **PU-Net Line B full**: multiple chunks reached the 24h timeout; resuming the missing 3161 samples eventually produced a 12311/12311 pass.
- **PU-GCN**: an RTX 3080 stall and a SyntaxError were resolved through repairs and resume runs.
- **PU-EdgeFormer**: separate audits covered repeat-copy behavior, call paths, checkpoints, differences from PU-GCN, leakage, and output freshness.

### 10.3 Figures and presentation materials

- Static figures: `figures/modelnet40/pointnet2_final*_with_pu_edgeformer/`, `geometry_delta_*`, `final_combined/`.
- Interactive HTML: `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/` (10 dropdown views).
- PPT: `ModelNet40_PointNet2_Full_Report.pptx` (28 slides) and `Original_Reference_Revised.pptx` (21 slides).
- Matched elevation, azimuth, axis limits, and point-size rules support fair qualitative comparisons between methods.

---

## 11. Ablations and supplementary results

| Item | Content | Role |
|---|---|---|
| ×2 EAR Line B | 512→1024, earlier Step 7b/8 | Ratio ablation, outside the main ×4 protocol |
| Naive resampling baseline | 512 points on disk, resampled by the loader to 1024 in Step 6 | Illustrates contamination from artificial point-count matching |
| Earlier downsampled50 main line | 512-point protocol | Replaced by ×4/256 |
| TULIP | Range-image upsampling | Supplementary only |
| Unequal-cardinality CD/HD ranking | 4096 vs 1024, etc. | Excluded from the main geometric ranking |
| Earlier P2F narrative | Dense mesh exact P2F | Appendix material, outside the current main table |
| KITTI / CenterPoint | Detection AP experiments | Separate chapter, not expanded in this dossier |

---

## 12. Thesis structure and usable conclusions

### Suggested structure

1. **Introduction**: motivate systematic downstream classification evaluation beyond geometric metrics and introduce the two-line ×4 question.
2. **Related Work**: EAR → PU-Net → PU-GCN → EdgeFormer / diffusion (PDANS), PointNet++, and ModelNet40.
3. **Method**: protocol principles, two-line processing, classification from scratch, and equal-N geometry.
4. **Experimental Setup**: data, hyperparameters, metric definitions, and implementation details.
5. **Results**: classification, equal-N geometry, and mesh-ref tables; report absolute OA and Δ pp together.
6. **Discussion**: the gap between geometry and task utility, densification vs recovery, and the roles of distribution, HD, and NUC.
7. **Ablation / Appendix**: ×2, naive resampling, mesh-ref controls, and engineering audits.
8. **Conclusion**: the main findings and implications listed below.

### Conclusions for adaptation into the thesis

1. Point-cloud upsampling does not generally improve PointNet++ classification accuracy on ModelNet40 in these experiments.
2. Under densification (1024→4096), all five methods have lower Best OA than native Original 1024 (91.95%).
3. Under recovery (256→1024), only PU-Net improves over the sparse baseline (+0.42 pp), while remaining 0.68 pp below Original.
4. Geometric metrics and classification utility differ: PU-GCN often has better CD but weaker classification than PU-Net.
5. Mesh-ref 4096 is only +0.05 pp above Original, showing that increased point count alone cannot explain or guarantee classification gains.
6. Upsampling evaluations should report geometric quality and downstream utility together, explicitly distinguishing densification from recovery.

---

## 13. File index

| Purpose | Path relative to the original project root |
|---|---|
| This Markdown dossier | `presentations/ModelNet40_Thesis_Complete_Dossier.md` |
| Companion PDF | `presentations/ModelNet40_Thesis_Complete_Dossier.pdf` |
| Full PPT/PDF | `presentations/ModelNet40_PointNet2_Full_Report.*` |
| Revised PPT | `presentations/ModelNet40_PointNet2_Original_Reference_Revised.*` |
| Classification summary | `reports/modelnet40_pointnet2_final_two_line_classification_summary_with_pu_edgeformer.*` |
| Geometry and classification summary | `reports/modelnet40_final_thesis_summary_table_with_pu_edgeformer.*` |
| Interpretation | `reports/modelnet40_thesis_result_analysis.md` |
| Equal-N geometry | `reports/modelnet40_geometry_equal_n_*.{csv,md,json}` |
| Mesh-ref classification | `reports/modelnet40_pointnet2_mesh_ref_baseline_results.*` |
| ×4 protocol report | `reports/modelnet40_x4_final_protocol_report.md` |
| Protocol correction log | `reports/modelnet40_pointnet2_final_protocol_corrected.md` |
| EdgeFormer audit | `reports/pu_edgeformer_modelnet40_authenticity_verification_20260716.md` |
| Figure captions | `reports/modelnet40_thesis_figure_captions.md` |
| Figure index | `figures/modelnet40/figure_index.md` |
| Method registry | `../modelnet40_experiments/configs/method_registry.yaml` |
| Training configurations | `configs/pointnet2_x4_two_line/*.yaml` |
| Metrics JSON | `pointnet2_results/x4_two_line_final/**/metrics.json` |

---

## Appendix A. Training settings and data paths

| Item | Value |
|---|---|
| Data split | 9843 train / 2468 test, official split |
| Model | `pointnet2_cls_ssg` |
| epoch / batch / lr / seed | 200 / 24 / 0.001 / 42 |
| optimizer / decay | Adam / 0.0001 |
| Features | XYZ only |
| Line A data | `pointnet2_inputs/lineA_*` |
| Line B data | `pointnet2_inputs/lineB_*` |
| Downsampled root | `datasets/modelnet40_downsampled_x4/` |
| Mesh-ref roots | `datasets/modelnet40_mesh_ref_{256,4096}/` |
| Results root | `pointnet2_results/x4_two_line_final/` |

---

## Appendix B. Code and checkpoint sources

| Method | Code source | Weights / notes |
|---|---|---|
| EAR | `code/upsampling_methods/EAR`, migrated from lab | No checkpoint; non-learning method |
| PU-Net | yulequan/PU-Net, migrated | TF1 model-120, etc. |
| PU-GCN | guochengqian/PU-GCN, migrated | TF1 model-120, etc. |
| PDANS | Baty2023/PDANS, migrated | `PU1K_PDANS.pkl` / `PUGAN_PDANS.pkl` |
| PU-EdgeFormer | `external/pu_edgeformer_ops_reuse` | `checkpoint_model100`; audited 2026-07-16 |
| TULIP | ethz-asl/TULIP, migrated | `tulip_kitti.pth`; supplementary |
| PointNet++ | `external/Pointnet_Pointnet2_pytorch` | All classifier training in these experiments was from scratch |

---

## Document notes

- Where earlier Markdown conflicts with this record, the **2026-08 Full Report / equal-N / `classification_summary_with_pu_edgeformer`** takes precedence for this dossier.
- KITTI / CenterPoint detection belongs to earlier or parallel work. A detection chapter requires the separate `centerpoint_variant_eval` and KITTI `upsampled_variants` records.
- Regenerate the companion PDF from the archive root with `python research/modelnet40/code/scripts/generate_thesis_complete_dossier_pdf.py`.
