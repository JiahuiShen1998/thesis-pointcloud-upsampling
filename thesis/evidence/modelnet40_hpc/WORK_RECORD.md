# ModelNet40 All Experiments, Adjustments, Failures and Results

- Summary date: 2026-09-10 (Europe/Berlin)
- Practical laboratory warehouse:`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`
- Paper and final delivery catalogue:`/home/hpc/iwnt/iwnt189h`
- The principle of verification: records only work that can be proved by existing scripts, configurations, Slurm bills, logs, checkpoint, CSV/JSON, graphs or compilers as " actually running or actually modified " .only has plan and the unexecuted content is not completed.

## 1. Current General Status

1. ModelNet40 point-cloud upsampling —PointNet++ The main experiment is closed.The main protocol consists of two lines and five upsampling methods: EAR, PDANS, PU-Net, PU-GCN, PU-EdgeFormer.
2. All five of the final two lines have 12,311 accessable point cloud results (9,843 train + 2,468 test);Line A is 4,096, Line B is 1,024.PU-EdgeFormer files are connected by link and used `find -L` Verify as 12 per line, 311.
3. The project includes a total of 22 completed PointNet++ SSG formal training (excluding smoke);This includes the final master experiment, mesh-reference control, repeated training and the early protocol that was subsequently replaced.
4. The final main experiment consists of 12 classification branch: Line A 1 baseline + 5 methodology, Line B 1 baseline + 5 methodology;There are also 2 mesh-reference controls branch.
5. The Slurm team is currently listed as empty, and no is still running or waiting for jobs.
6. The experimental warehouse is not a clean work tree: branch `share/modelnet40-pointnet2-presentation`, there are 20 tracked files modified, 59 untracked entries.Only one in the warehouse has been submitted. `daacfaf`(2026-07-27) As a result, a large number of follow-up results have not yet been submitted.
7.  The current experimental warehouse is visible in size: 146  A script. 69  individual  Slurm/job  Documentation, 373  Report documents, 1,005  Log files, 24  Configure, 182  individual  PointNet++  the outcome document, 273  Graphical files, 25  A presentation document.

## 2. End-use Experiment protocol

| Experiment Lines | Baseline | upsampling branch | PointNet++ method of training |
|---|---|---|---|
| Line A: Original point cloud Decryption | Original 1,024 | 1,024 → ×4 → 4,096 | The baseline is based on 1,024 point training;Each method uses 4,096 points for training from scratch. |
| Line B: recovery after four times the fall of sampling | Original 1,024 → decimate ×4 → 256 | 256 → ×4 → 1,024 | Training from scratch on 256 basis;Each method uses 1,024 points for training from scratch. |

Uniform settings:`pointnet2_cls_ssg`, 200 epochs, Adam, initial learning rate 0.001, seed 42, no normals.Finally branch does not allow DataLoader to make unauthorized patches or tailors to match another branch.

### protocol Actual adjust trajectory

| Phase | The actual protocol. | Follow-up |
|---|---|---|
| Early baseline | Original 1,024; Downsampled50 512 | Keep as Early Result |
| Early EAR | 512 → EAR ×2 → 1,024;  I've run all the time.  512  I'm sorry.  loader  resampling to  1,024 | Decline to preliminary/ablation |
| First edition x 4 | Line A 1,024→4,096; Line B 512→2,048 | EAR, PDANS and some PU-Net were actually generated/trained, but no longer the final Line B master protocol |
| Final x 4 | Line A 1,024→4,096; Line B 256→1,024 | To become the theorist of the paper protocol |
| Extension of methodology | Finish EAR/PDANS/PU-Net/PU-GCN and access PU-EdgeFormer | Finally, main table contains five methods. |
| geometric Assessment Amendment | Early presence of 4,096 vs 1,024 et al. unequal-cardinality CD/HD | Subsequently, it became equal-N: Line A 4,096 vs mesh-ref 4,096;Line B 256 vs mesh-ref 256, 1,024 vs Original 1,024 |

TULIP is always only marked supplementary, SPU-PMD does not enter ModelNet40 main protocol;Both no final master test results, cannot are listed as completed methods.

## 3. Actual working record by time

### 2026-06-20: Data and classifier Foundation

- Run ModelNet40 pre-treatment: 40, 9,843 train, 2,468 test, for a total of 12,311 samples;Every sample from `.off` Weighted by triangle area sampling 1,024 points, then centralized to unit balls;153.7 seconds complete, 0 failed.
- Run complete data check: Count, shape `(1024,3)`, NaN, Inf, unit balls and random samples are PASS.
- Access `yanx27/Pointnet_Pointnet2_pytorch` It's... `pointnet2_cls_ssg`, prepare custom NPY DataLoader, training/assessment packaging, configuration and sbatch.
- GPU smoke job 1709664 Completed: Load, Forward, mini-train, eval, checkpoint All PASS.
- Submit Original baseline job 1709666, finally complete 200 epochs.
- Generating Downsampled50: 1,024  /  512, 12,311 samples and 0 failed;subset, Recoverability and Limited Value Check All PASS.
- Run loader-resampled 512  /  1,024 baseline job 1710180;The result was subsequently redefined as ablation, no longer the main baseline.

### 2026-06-21 to 2026-06-22: EAR ×2 and early PointNet++

- EAR CPU smoke Actual running: Original branch in `target_n=1024` Time is identity copy;Downsampled50  branch Completed  512→1,024,160  individual  smoke  All samples  PASS.
- EAR full Line B 512→1,024  Completed  12,311/12,311, shape/NaN/Inf/DataLoader  Inspection  PASS.
- EAR ×2 + PointNet++ job 1711437 finished, Best OA 90.82%;Later marked as `ablation_preliminary_x2`.
- The original 512 point PointNet++ job 1716131 was completed and became the early 512 baseline.

### 2026-06-25: Primary Multiplication and Fairness Rule Correction

- Correct the wrong rule that “baseline and upsampling must be in line with point count”: native point count should be retained in the main baseline, resulting in upsampling after point count.
- The main multiplier is changed from x 2 to x 4.
- EAR 512 = 1,024 and loader 512 = 1,024 was retained but downgraded to preliminary/ablation, without deleting data and logs.
-  The plan was...  Line A 1,024→4,096, Line B 512→2,048; The Line B design was later replaced by the final 256  /  1,024.

### 2026-06-26 to 2026-07-01: Methodological environment, upsampling and early x 4 training

- Two missions of EAR Line B array 1716175 failed because of the metadata symlink competition.Modify `ensure_metadata_link()` Capture `FileNotFoundError`, only reruns missing blocks, and eventually EAR ×4 creates PASS.
- The first simultaneous scripts misnamed the method. `pdans}`, `punet}`, `pugcn}`And form a band. `}` (a) Catalogues/reports;Amend the submission logic and resubmit it.
-  Revised  6  individual  smoke (1717035/37/39/41/43/45) All of them still failed: PDANS  It's...  CUDA/PyTorch  Environmental incoherence; PU-Net/PU-GCN's TF1 custom ops couldn't be correct. `nvcc`, `libcudart`, `libtensorflow_framework`.
- Modifying CUDA activation, TF framework link and custom-op compilation scripts;Recompile on GPU node.Finally, PU-Net `tf_sampling` and PU-GCN `tf_grouping` compile/import PASS.
- PDANS  I met her first.  PyTorch3D ABI  Not Match() job 1717964),  And then...  job 1722422  Shortfalls encountered  `open3d`; job 1723850 complete GPU validate PASS after installation/ adjustment compatibility.
- PDANS smoke and two lines full generation PASS;The early training of EAR/PDANS x 4 was completed.
- PU-Net smoke 1725581/1725582 PASS, actually calling the official generator checkpoint instead of a general plugin.

### 2026-07-02 to 2026-07-07: final Line B, recovery mission and final smoke

- Finally, fixed Line B is 1,024, decimate ×4,256, upsample ×4, 1,024;The old 512  /  2,048 data are reserved but withdrawn from the main protocol.
- EAR, PDANS and finally Line B each completed 12,311/12,311 and conducted an audit with shape.
- PU-Net smoke 1727073: Successfully reasoned but not created before the strict output, 40/40 failed to write;Patch `strict_out.parent.mkdir(parents=True, exist_ok=True)` After 1729225 PASS.
- PU-Net full array 1729235: 4  individual  chunk  It's done. 11  individual  chunk 24  Hours  TIMEOUT, 1  individual  chunk  Yes.  `bookshelf_0424` Up TF subprocess `rc=-6` Failure;Retain 9,150 valid audited outputs.
- PU-Net missing-only resume array 1731611 finished, finally raw/strict 12,311/12,311, all `(1024,3)`No NaN/Inf.
- PU-GCN smoke 1731790/1731791 is being repaired. `open3d`/`plyfile`, the parallel output catalogue competition and the smoke sample alignment, followed by the two lines PASS.
- PU-GCN Line A full 1731793 finished 12,311/12,311.
- PU-GCN Line B full 1731794 is stuck at 10,010/12 and 311: tasks 13–15 is on RTX 3080.`knn_point_2` 100% failed to reason, 120 minutes without new documents;Cancel the three tasks manually and keep the existing output.
- PU-GCN First recovery array 1732373 8 missions due to Python `global` The statement of location led to SyntaxError, which failed before the reasoning;Change it to a local variable with a visible cross-section.`py_compile` PASS.
- (a) smoke 1732385 for 2 missing samples PASS;The second recovery array 1732386 is all finished on RTX 2080 Ti limit point.Current actual verification of raw=12,311, strict=12,311, missing list=0.
- PointNet++ Line B First round smoke: Baseline PASS, four methods job 1733160–1733163 failed all due to lack of label-map metadata.
- metadata v1 links the entire metadata directory, and job 1733167–1733170 failed because manifest pointed back 256 point baseline.
- metadata v2 only links `class_to_idx.json` and `idx_to_class.json`, not linked to manifests;job 1733171–1733174 All PASS.
- PointNet++ Line A First round smoke: Baseline PASS, job 1733476–1733479 also failed due to the absence of metadata;Applying the v2 rule to 1733487–1733490 all PASS.

### 2026-07-07 to 2026-07-08: Last two lines PointNet++

- Line B Official training jobs 1733191–1733195: 5/5 COMPLETED, all 200 epochs.
- Line A Official training jobs 1733491–1733495: 5/5 COMPLETED, all 200 epochs.
- Each branch independently from random initialization training, no uses 1,024 point checkpoint for 4,096 point branch.

### 2026-07-12 to 2026-07-17: Analysis of results and PU-EdgeFormer

- Generating classification, geometric, results by category, method ranking, geometric — classification relationships and paper tables.
- Production of static/interactive point cloud comparisons;v2  Total  10  individual  Line A grid, 10  individual  Line B grid, 10  individual  dropdown, 22  individual  single-method HTML.
- Access to PU-EdgeFormer by completing the audits of authenticity, call path, freshness, repeat/copy and discrepancies with PU-GCN.
- PointNet++ smoke jobs 1749430/1749431 double line PASS.
- Official training jobs 1749441/1749442 double line COMPLETED: Line B Best OA 89.32%, Line A Best OA 90.54%.
- Final visual audit PASS: 33 final package PNG, 52 interactive HTML;no Retrain or modify data.

### 2026-07-23 : 4,096

-  Review of five  Line A  upsampling branch  YAML, sbatch,  Training code, head  batch,  First  loss, epoch 200, metrics  and independence  checkpoint.
- Conclusions: Five branch were indeed entered in 4,096 points and trained from scratch;No more clean rerun.

### 2026-08-03 to 2026-08-18: equal-N, control experiments, mechanism probes and general report

- From `.off` Area-weighted resampling to build mesh-ref 256 and mesh-ref 4,096, 12 and 311;Build audit PASS.
- Run equal-cardinality CD/HD/NUC: test 2,468 objects, workers time 128.1 seconds, 0 failed.
- Trained mesh-ref PointNet++ to control jobs 1768767/1768768, all completed 200 epochs.
- Run cardinality-bias probe:`repeat_x4` 4,096 vs Original 1,024 were given CD=HD=0, but the worst under equal-N, proving that the old reference design incentive “does not do upsampling”;This led to the abandonment of unequal-cardinality's main ranking.
- Run PointNet++ SA1 reception probe: 256 point 59.2% slot by duplicate filling;1,024, discard 31.9%;4,096, abandoned 78.7%, explaining why the benefits of continued enrichment are small.
-  Yeah.  Original 1,024, EAR 4,096, PDANS 4,096  Three pairs of repeated training.  checkpoint  Compared by size: 79/79  Weights are the same, the worst.  0; The seed 42 process proved to be bit-deterministic but not more than seed statistically stable.
- Generates complete dossier, Full Report, Progress Report, PPTX/PDF, charts and all point count training inventory.

### 2026-09-04: Articles 3, 4

- Extracted and expanded Idea and Concept, Experimental Setup, covering research assumptions, two lines, equity, upsampling protocol, downstream assessment, indicators and reproduction checks.
- Finally, PDF 27 page;Chinese characters 18,055;13 copies of citation key entries;No undefined reference, missing glyph, overfull box, repeat the main paragraph.
- Three rounds of build, page PNG preview, contact sheet and final page check were actually performed.Successfully compiled;The log has a non-stop underfull box hint.

### 2026-09-07 to 2026-09-08: Articles 5, 6 and Additional Audit

- Change from v1/expanded/pre-readable to final `chapters_5_6_body.tex`.
- Redo all the main maps for "Tooflower, Data Visible, point cloud Insufficient Contrast, Evidence and Parameters": Finally PDF 54 page, the text uses 28 unique diagrams, each with PDF and 300 dpi PNG.
-  Transfer  14  The end. / Control of branch  2,800  Article by article  epoch  Records, 600  Accuracy rate, by category, 34,552  Article  equal-N  Object geometric Records, 29,616  Article  dense-reference/P2F  Sub-records, 150  Article  pairwise  Winning, winning. 112  A visualized point cloud array.
- Delete the crowded old hot map from the main text and replace it with the grey steps, the fine grid, the hollow mark, the 0–100% by class and four pages of point cloud evidence.
- Final audit: 54 page, 28/28 citation, no missing chart, no overfull, no undefined reference, no LaTeX error;SHA256 is `0c94b8a8f825a4c33a109f9c5adfa58f3de094812be8a547df41737546bc9488`.

## 4. All 22 completed PointNet++ formal training

The results are set out below;smoke doesn't count.`early` The protocol, which indicates that it was replaced or repeated later, is still the true result of the operation.

| point count | Run | Best OA | Final OA | Best Class | Best epoch | Data sources | Round |
|---:|---|---:|---:|---:|---:|---|---|
| 256 | mesh_ref_baseline_256 | 90.88% | 90.38% | 86.34% | 129 | `.off` Area weighted resampling | final control |
| 256 | lineB_downsampled_x4_baseline | 90.85% | 90.46% | 87.37% | 124 | Original 1,024 decimate ×4 | final |
| 512 | downsampled50_native512_pointnet2 | 91.26% | 91.14% | 88.00% | 197 | Original by sampling or about 50% | early |
| 1,024 | lineA_original_baseline | 91.95% | 91.31% | 88.00% | 85 | Original mesh sampling | final |
| 1,024 | original_baseline | 91.95% | Unrecorded | 88.00% | 85 | Early duplication of data with previous item | early |
| 1,024 | lineB PU-Net | 91.27% | 90.65% | 87.86% | 148 | 256→PU-Net×4 | final |
| 1,024 | downsampled50_baseline | 91.17% | Unrecorded | 87.71% | 155 | Disk 512, loader resampling to 1,024 | early ablation |
| 1,024 | step8_downsampled50_ear | 90.82% | Unrecorded | 87.06% | 111 | 512→EAR×2 | early ablation |
| 1,024 | lineB PDANS | 90.34% | 89.46% | 86.20% | 145 | 256→PDANS×4 | final |
| 1,024 | lineB PU-GCN | 90.06% | 89.63% | 86.41% | 164 | 256→PU-GCN×4 | final |
| 1,024 | lineB PU-EdgeFormer | 89.32% | 88.58% | 85.21% | 194 | 256→PU-EdgeFormer×4 | final |
| 1,024 | lineB EAR | 88.81% | 88.02% | 84.50% | 116 | 256→EAR×4 | final |
| 2,048 | downsampled50_pdans_x4_pointnet2 | 91.47% | 90.99% | 88.19% | 164 | 512→PDANS×4 | early |
| 2,048 | downsampled50_ear_x4_pointnet2 | 90.61% | 89.76% | 87.45% | 74 | 512→EAR×4 | early |
| 4,096 | mesh_ref_baseline_4096 | 92.00% | 91.18% | 88.70% | 148 | `.off` Area weighted resampling | final control |
| 4,096 | lineA PU-GCN | 91.63% | 91.00% | 87.74% | 81 | 1,024→PU-GCN×4 | final |
| 4,096 | lineA PDANS | 91.62% | 90.87% | 88.44% | 84 | 1,024→PDANS×4 | final |
| 4,096 | original_pdans_x4_pointnet2 | 91.62% | 90.87% | 88.44% | 84 | Early duplication with data | early |
| 4,096 | lineA EAR | 91.48% | 90.85% | 88.19% | 61 | 1,024→EAR×4 | final |
| 4,096 | original_ear_x4_pointnet2 | 91.48% | 90.85% | 88.19% | 61 | Early duplication with data | early |
| 4,096 | lineA PU-Net | 90.95% | 90.55% | 87.46% | 189 | 1,024→PU-Net×4 | final |
| 4,096 | lineA PU-EdgeFormer | 90.54% | 90.13% | 86.35% | 83 | 1,024→PU-EdgeFormer×4 | final |

### Finally, classification Conclusions

- Line A: no exceeds Original 1,024 91.95%;The closest are PU-GCN 91.63% ( - 0.32 pp) and PDANS 91.62% ( - 0.33 pp).
- Line B: only PU-Net exceeds 256 point baseline, 91.27% against 90.85%, up + 0.42 pp;The remaining EAR −2.04 pp, PDANS −0.51 pp, PU-GCN −0.79 pp, PU-EdgeFormer −1.53 pp.
- Mesh-reference Control: mesh-ref 256 is 90.88%, only higher than downsampled 256 0.03 pp;mesh-ref 4,096 is 92.00%, only higher than Original 1,024 higher than 0.05 pp.
-  protocol Sensitivity: PDANS  In the old one.  512→2,048  protocol relative  native 512  Yes.  +0.21 pp,  In the end.  256→1,024  protocol  −0.51 pp,  The conclusions will be reversed at the depth of sampling.

## 5. Final equal-N geometric Results

CD/HD  For the same point count reference  test-set mean, N=2,468; The lower the better.

| Line | Methodology | References | CD | HD | NUC |
|---|---|---|---:|---:|---:|
| A | EAR | mesh-ref 4,096 | 0.054722 | 0.115396 | 0.815631 |
| A | PDANS | mesh-ref 4,096 | 0.046821 | 0.113510 | 0.661176 |
| A | PU-Net | mesh-ref 4,096 | 0.049719 | 0.111177 | 0.605211 |
| A | PU-GCN | mesh-ref 4,096 | **0.042306** | **0.093219** | 0.520398 |
| A | PU-EdgeFormer | mesh-ref 4,096 | 0.046666 | 0.110932 | 0.675435 |
| B | EAR | Original 1,024 | 0.076743 | 0.193182 | 0.965292 |
| B | PDANS | Original 1,024 | 0.062696 | 0.190541 | 1.052826 |
| B | PU-Net | Original 1,024 | 0.064573 | **0.108319** | 1.080102 |
| B | PU-GCN | Original 1,024 | **0.055802** | 0.151403 | 1.523996 |
| B | PU-EdgeFormer | Original 1,024 | 0.061900 | 0.177572 | 1.122505 |

Conclusion: CD, the lowest average of both lines, is PU-GCN;Line B Minimum HD is PU-Net.The geometric CD leader and classification OA leader are not a stable match.

## 6. failed, cancelled, timeout, return to work and final disposal

| Date/operation | Failure or problem | Actual modifications/treatments | Final result |
|---|---|---|---|
| 1716175 tasks 4/6 | metadata symlink and competing,`FileNotFoundError` | `unlink()` There's only a rerun of missing pieces. | EAR full Final PASS |
| 1716440–1716451 Submitted | shell `}` | Method of amendment for roll-out and resubmission process | Error Catalogue Retain as Evidence |
| 1717035/37 | PDANS Not Available CUDA | Harmonized modules, conda, CUDA path | Follow-up GPU validate PASS |
| 1717039/41, 1717043/45 | PU-Net/PU-GCN TF custom ops compiled/import failed | Amendments `CUDA_HOME`, nvcc, cudart, TF framework Link, compiled at GPU node | compile/import PASS |
| 1717964 | PyTorch3D ABI mismatch | Adjust torch/PyTorch3D Compatible Environment | Follow-up over that error |
| 1722422 | PDANS is missing `open3d` | Install open3d 0.17.0 and revalidate | 1723850 PASS |
| 1727073 | PU-Net strict  Save directory does not exist  | Add parent `mkdir` | 1729225 smoke PASS |
| 1729235 array | 11 TIMEOUT, 1 `rc=-6`, only 9,150 | missing-only, skip-existing recovery | 1731611 Completed 12,311 |
| 1731794 tasks 13–15 | RTX 3080 `knn_point_2` 100% failed | Cancel the stuck task and keep 10,010;recovery Task Limit RTX 2080 Ti | Data final 12,311 |
| 1732373 array | recovery Script SyntaxError | Get rid of the problem global. `--max-samples`, py_compile+smoke | 1732386 Completed |
| 1733160–1733163 | Line B upsampling Directory No label map | metadata v1 Entire Directory Link | v1 still fails |
| 1733167–1733170 | v1 manifest Return 256 dot file | v2 only Link two class-map without manifest | 1733171–1733174 PASS |
| 1733476–1733479 | Line A Similar metadata Missing | Apply v2 | 1733487–1733490 PASS |
| Old geometric Table | unequal-cardinality CD/HD will reward duplicate input | Add mesh-ref, equal-N and repeat×4 probe and redo charts/papers | The old ranking exits the main conclusion |
| Early PPT | Combining dense mesh/P2F and vs-Original with links to `file://` | rewrite geometry loader/ charts, add mesh-ref controls, recheck cameras, recreate 21 pages PPT/PDF and 10 dropdown_v2 | The document has been modified;Open HTTPS base is still not set |

## 7. Evidence still in existence boundary

These are not “failures of mission”, but are the results of a cross-border interpretation of cannot:

- Each final classification branch has only one independent training of seed 42;no more than seed mean, variance or significantness test.
- Best OA select checkpoint using test set;no Independent validation set.The official presentation should give priority to validation set or to this point.
- It has been verified that seed can be bit-deterministic, does not represent cross seed robust.
- The current Chamfer is the sum of unsquare Euros' recent proximity;A uniform definition is required if the the thesis is written at square distance.
- NUC Query Centre is not fully shared among all methods according to shape ID;P2F uses a stand-alone unit ball normalization and is therefore only subject to secondary audit.
- no saves the object-by-object classification logits/ probability, cannot strictly generates PR curves, complete confusion matrix, McNemar or calibration.
- PU-GCN Final 12,311 Count, 0 missing, PointNet++ Completion has both actual documents and Slurm/metrics evidence, but the warehouse lacks the name `modelnet40_pugcn_final_count_audit.md` and `modelnet40_pugcn_provenance_audit.md` The final Markdown closed-ring file.
- Chapters 3–4 and 5–6 of the dissertation are located in `/home/hpc/...`, which is not part of the current experiment Git repository, there is no historical protection of Git.

## 8. No changes currently submitted

### tracked Changes (20)

- 6 `figures/modelnet40/original_reference_revised/` geometric / geometric — classification.
- 10 `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/` Interactive HTML.
- `presentations/ModelNet40_PointNet2_Original_Reference_Revised.pdf`.
- `presentations/ModelNet40_PointNet2_Original_Reference_Revised.pptx`.
- `presentations/ModelNet40_Presentation_Revision_Notes.md`.
- `scripts/generate_original_reference_presentation.py`: The script is the main text change in the entire diff that was added to approximately 507 rows and deleted from line 321;The core is equal-N data priority, mesh-ref control, geometric label and page description corrections.

### untracked entries (59 Git top level entries)

The main components are: CUDA compatible link, external dependency, mesh-ref 256/4,096 configuration and jobs, equal-N and SA1/cardinality probe scripts and reports, two sets of mesh-ref data audit, complete PPT/PDF/dossier/progress report, results matrix picture, progress charts and two tar.gz delivery packages.They all exist in the work tree, but no goes into Git.

## 9. Key Verifyable Entry

- All completed training matrix:`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_all_pointcounts_accuracy_inventory.md`
- Finally classification:`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.md`
- equal-N geometric:`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_geometry_equal_n_summary.csv`
- Failed diagnosis:`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/` Down with everything. `failure_diagnosis`, `stall_diagnosis`, `cancelled_for_resume` and `correction` Documentation.
- Table of final results:`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_final/`
- Chapters 3, 4:`/home/hpc/iwnt/iwnt189h/modelnet40_revision/ModelNet40_Chapters_3_4_Expanded.pdf`
- Chapters 5, 6:`/home/hpc/iwnt/iwnt189h/modelnet40_chapters_5_6/ModelNet40_Chapters_5_6.pdf`
- Final audit of Chapters 5, 6:`/home/hpc/iwnt/iwnt189h/modelnet40_chapters_5_6/FINAL_AUDIT.txt`

## 10. One-word conclusion

This was not “one best accuracy rate”, but the complete chain from data construction, five upsampler adaptations, failed recovery, two fair protocol PointNet++ training, equal-N geometric correction, control experiments and mechanism probes, to presentation and dissertation chapter 3–6; In the end, classification, there's only one on top.  Line B  It's...  PU-Net  Relative  256  Point baseline has  +0.42 pp  lift, and  Line A  There's nothing more than that.  Original 1,024,  And geometric's best method is not consistent with classification's best.
