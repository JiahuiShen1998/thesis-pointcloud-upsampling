# From February to March to the present, the full work of the experiment.

Update date: 2026-09-17 (Europe/Berlin).Scope: evidence available in the local work area;This is not an addition to the day-by-day event for which the log is not kept.

## Reading notes and evidence boundary

The record is divided into two parts:

- **Part I: Unified time line, latest results, problems and treatment, code and evidence portal up to 09-17.** The current conclusion is based on this part.
- **Part II: Full text of 09-10 General Historical Record.** (a) Keep all records of early experiments, failures, ablation, subset results, code changes, documents and clean-up;Keep the text as it is, only lowers the title level."Current" "convergence" in it refers to 09-10, which was not 09-17 at the time.

The term “complete” refers to the activity of cannot to restore no as much as possible to cover all job categories in existing evidence.II. In March, preparatory work could be confirmed from historical retrospective records and from ongoing work documents, but no simultaneous training logs that could be validated on a day-by-day basis have been found;In April, too, a sufficiently independent daily record was found.cannot used the time when the document existed or was modified as proof that the experiment had been completed during the month.

Status distinction: Completed/partially completed/failed or discontinued/code completed without formal pilot/audit analysis/not implemented.pilot, smoke, plan and the empty directory none impersonate full-val.

The historical indicators must be read in the original protocol: the early PointRCNN had an old AP_R11 style evaluator, different input sampling and proposal configurations;cannot and 09 lateral fusion of AP_R40 main table.There is no direct comparison between different subsets, random seed or AP, which has different weights.

## 1. Current sentence

Multiple methods are rigorous and upsampling no has been steadily tested and upgraded;observed-first plus detector adaptation significant recovery performance. Now.  PointRCNN  and  CenterPoint  It's...  Line A/B  We've got a full take-out.  epoch  Validation and satisfaction of the established time frame test, but the best result remains below ' s respective reference baseline. The PU-GCN network itself has always been fixed, no conducting a upsampling training on KITTI.

## 2. Timeline from beginning to end

| Phase | Purpose and practical work | Problems, outcomes and status | Evidence positioning |
|---|---|---|---|
| 2026-02 to 03, retroactive | Create a PointNet/ModelNet/KITTI project, preprocessing of data, classification training and upsampling interface | (b) Engineering preparation;no can confirm the full ModelNet40 classification accuracy rate.Date only can be traced to the general record of history, cannot by day | Part II, sections 10, 17;thesis_demo |
| 2026-04 | The transition from early engineering to May. | Sufficient independent simultaneous records were not found, and no remedial experiments and achievements were made | Evidence gap |
| 05-03 to 05-08 | KITTI baseline, EAR, PU-Net x2; PU-GCN Environment and reasoning recovery | The development of early and complete assessments also exposes serious points of loss and environmental compatibility problems.05-04 PU-Net Old protocol Moderate 3D AP 1.1364 | 05-04 work log;Part II, section 9 |
| 05-12 to 05-19 | fair comparison, RPN4096, TULIP, PU-EdgeFormer Initial access | The configuration is not uniform;PointRCNN negative sampling, missing part of the document, old TF/CUDA compilation and weighting problems.cannot, mix these protocol results and make it official. | Part II, Section 6, 9 |
| 06-06 to 06-15 | TULIP Full and Output Audit;PDANS/SPU-PMD access;HPC Prepare for migration | TULIP has run but not unified exact-4×;SPU-PMD only 4-frame smoke;HPC is blocked by DNS/ | Part II, sections 5, 9, 11 |
| 06-20 to 06-30 | Audit of real point count multiplier, design strict x4 Line A/B, supplementary visualization and cleanup | It was found that the old output was not always x4 and that some methods had point count caps;Large intermediate file migration and cache cleanup list | Part II, sections 3, 9, 12 |
| 07-03 to 07-18 | PDANS, PU-GCN, PU-EdgeFormer, PU-Net Strict Line A/B;PointRCNN in full;E1/E2/E3 input control | The main experiment was completed;upsampling does not exceed baseline;Add sampler-safe, separate the file layer point count and detector to read point count | Part II, section 5–7;master inventory |
| 07-19 to 07-31 | CenterPoint in full;dose, direct no-resample, patch root, PU-Net normalization 2×2 ablation, surface patch | Finds repeat points, non-local patch, normalization and voxel activation problems.(a) Repair of recovery part AP, but does not imply exceeds baseline;Part full dose not completed | Part II, Section 5–7, 12 |
| 08-04 to 08-11 | Multiple methods local patch pilot, region split, detector-aware PDANS V2–V5, Line B selection policy | 256-frame Development and Part holdout64 completed;Local positive result no becomes a stable cross protocol / cross detector gain | Part II, section 7 |
| 08-12 to 08-18 | Methodology feasibility survey, sampling variance, systematic reporting, target missed detection case | 16-seed Checks to reduce the one-time positive gain interpretation;The new methodology survey is not counted as a run-off experiment;Generate report and lost-car | Part II, sections 7, 14, 15 |
| 08-24 to 08-28 | finetune64 Six branch screening;(a) detector adaptation for the complete train source data;observed-first pilot | (a) Presumption test and complete training, 256-val pilot has been improved;3 epochs is fixed length, not selected by convergence | Part II, section 8;full_retraining_report |
| 09-02 to 09-08 | Thesis chapter 3–6, ModelNet/KITTI Integration, chart and source list | Completion of document outputs;Part of the table is still the old pilot, cannot automatically as the later full-val result | Part II, section 14 |
| 09-08 to 09-10 | Strict PU-GCN A/B each 3,769 frame;Double detector 20-arm full matrix;Total category AP and source storage | 20/20 PASS; Clarification of 3 epochs no convergence evidence;The meaning of fixed upsampling and detector adaptation was clarified | full_val_detector_matrix; convergence_audit |
| 09-14, convergence supplementary | CenterPoint A/B Each 12 epochs, saved and fully validated per round | Two lines to satisfy 0.2 AP / patience 3;Best A 75.9661, B 62.9007, all e12 | CenterPoint convergence summary / CSV |
| 09-14 to 09-15 | PointRCNN Initial 12-epoch RPN convergence Experiment | A RPN e12 has been significantly raised to 69.5780, end of platform not yet due, cannot announcing convergence | 12→18 schedule_extension_protocol |
| 09-15 to 09-16 | New 18-epoch PointRCNN schedule, A/B RPN and RCNN Phase full verification | A Two phases, B RPN;B RCNN e16 to 56.8876, e17/e18 only 2 rounds not significantly improved, not reaching patience 3 | 18-epoch Phase by Stage CSV/JSON |
| 09-16 to 09-17 | Keep B RPN e14 and open 24-epoch RCNN full schedule | Best e19 is 57.0485;e20–24 Continuous 5 Round without significant improvement, status CONVERGED;24 has satisfied the sentence and does not require execution of 30/36 schedule | final_comparison.json; status.json |
| 09-17, this time. | Combining the latest convergence evidence and all previous work records | New bulletin and this document to keep the original 09-10 record;no Starts new training, no Modifys experimental weights | This document, brief, updated results CSV |

Early engineering evidence:[thesis_demo classification training](/home/ra87racy/projects/thesis_demo/train_cls.py), [PointNet Model](/home/ra87racy/projects/thesis_demo/models/pointnet_cls.py), [ModelNet Data Preprocessing](/home/ra87racy/projects/thesis_demo/scripts/preprocess_modelnet40_off.py). Of these, train_upsampling.py, dgcnn_cls.py is still empty;The interface/file name exists for does not represent training in matching models completed.

May Direct Log:[WORKLOG_2026-05-04](/home/ra87racy/projects/baseline_detectors/PointRCNN/WORKLOG_2026-05-04.md). Earlier general list:[master inventory](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_EXPERIMENT_MASTER_INVENTORY_20260804_EN.md). Historical dates are mainly based on the stage attributed to the reporting period, and the date of the document name does not necessarily equal the date of completion.

## 3. Current official results: not suitable, former 3-epoch, platform period results and baseline

Harmonized indicator: Car 3D Moderate AP_R40;Each evaluation uses 3,769 frame.The 'unsuitable' here refers to the same observed-first point cloud input with the official detector weight, not the original point cloud baseline.

| detector | Line | Not fit for input | Former 3-epoch | It's the best after training. | Reference baseline | Difference with baseline |
|---|---|---:|---:|---:|---:|---:|
| PointRCNN | A | 64.2748 | 71.0075 | **71.5920** | 81.9528 | -10.3608 |
| PointRCNN | B | 35.5289 | 55.2145 | **57.0485** | 68.3312 | -11.2827 |
| CenterPoint | A | 63.2064 | 74.7012 | **75.9661** | 79.2773 | -3.3112 |
| CenterPoint | B | 44.0929 | 61.6639 | **62.9007** | 68.0490 | -5.1483 |

| detector | Line | Relatively unsuited upgrades | Relatively old 3-epoch Upgrade | Best checkpoint |
|---|---|---:|---:|---|
| PointRCNN | A | +7.3172 | +0.5845 | RPN e14 + RCNN e2, 18-epoch schedule |
| PointRCNN | B | +21.5197 | +1.8340 | RPN e14 + RCNN e19, 24-epoch schedule |
| CenterPoint | A | +12.7597 | +1.2649 | detector e12, 12-epoch schedule |
| CenterPoint | B | +18.8078 | +1.2368 | detector e12, 12-epoch schedule |

Evidence:[PointRCNN Final summary](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/FINAL_RESULTS.md), [PointRCNN Proof by stage](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/final_comparison.json), [CenterPoint convergence Summary](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_20260914/reports/centerpoint_convergence_summary.md).

Comparative limitations:

1. Line A baseline is the original N point, the official detector weight; no has been retrained by convergence.  original-N baseline.
2. Line B baseline is a thin M point with existing 3-epoch adapted detector;Also no was extended this time to convergence baseline.
3. These differences are the difference between the results of the already available references and not the causal effects of all branch training budgets that are fully consistent.
4. checkpoint selects the same Car Moderate AP as validation set and reports the same validation set results;The validation-selected result is not an independent test generalization conclusion, and there is a risk of selection bias.
5. The platform-time rule Car, cannot, is automatically extended to all categories and all indicators are convergence. For the results of the old whole category see Part Two, paragraph 1  4.3  Section, cannot Put the old  checkpoint  It's...  Pedestrian/Cyclist  Number to the new best  checkpoint.
6. The 20-arm matrix of 09-10 remains the original 3-epoch experiment in which cannot continued to be known as the original matrix of the time after replacing adapted lines silently.

## 4. convergence, what steps have you made?

### Code for 4.1

Notable improvement condition is current AP **Stricter than**Distinguished best recorded AP + 0.2;Otherwise, there is no significant consecutive improvement in the count plus 1.At least 3 is counted at the end and marked as reaching the platform period.Full report schedule;It was triggered by patience, and if it rebounded significantly, it would be recounted.

The best global values for checkpoint and the reference checkpoint can be different.For example, A RCNN e2 is much higher than e1, but not higher than 0.2, so no significant improvements are reset;CenterPoint e12 is the same.

This rule provides operational convergence evidence under this training setting, does not prove that the parameter gradient is zero, does not prove the best in the world, and does not prove that the change in learning rates or training programmes will not continue to improve.

### 4.2 PointRCNN Two-stage evidence

AP of the RPN phase is the corresponding combination test assessment and should not be treated as a training phase with AP of the final RCNN model.

| Line | Phase | Completed epochs | Best Value epoch | Best AP |  No significant improvement at the end of the period  epochs | Status |
|---|---|---:|---:|---:|---:|---|
| A | RPN | 18 | 14 | 70.9712 | 4 | CONVERGED |
| A | RCNN | 18 | 2 | 71.5920 | 17 | CONVERGED |
| B | RPN | 18 | 14 | 52.7646 | 4 | CONVERGED |
| B | RCNN | 24 | 19 | 57.0485 | 5 | CONVERGED |

The four curves eventually selected are 78 epoch assessments (18+18+18+24), not 78 stand-alone random seed experiments.The old 18-epoch B RCNN and earlier 12-epoch A RPN is another schedule, which is preserved as historical evidence and does not collide into a continuous curve.

B RCNN 24-epoch End:

| epoch | Car Moderate AP_R40 |
|---:|---:|
| 19 | 57.0485 |
| 20 | 55.6280 |
| 21 | 55.3521 |
| 22 | 55.3774 |
| 23 | 55.6153 |
| 24 | 55.5902 |

 So end-use.  e19,  Not only, which is used for final saving.  e24.

### 4.3 CenterPoint, double-line evidence.

| Line | Completed epochs | Best epoch | Best AP | Significant improvement of reference epoch / AP | No significant improvement in the number of rounds at the end |
|---|---:|---:|---:|---|---:|
| A | 12 | 12 | 75.9661 | e8 / 75.7998 | 4 |
| B | 12 | 12 | 62.9007 | e9 / 62.8977 | 3 |

A Last value increment 0.1663, B is 0.0030, all does not exceed 0.2.Thus, the term “e12 maximum” does not conflict with “threshold to meet this platform period”;Nor should it be said that the final AP will not rise at all.This verification of CSV for all lines of 24, all of which are identified as 3,769 frame, 24, all of the checkpoint paths exist.

### 4.4 Why is cannot going straight behind 18 to 24?

The OneCycle learning rate trajectory depends on the total epochs. This time.  24-epoch B RCNN  From the same one.  RCNN  Initialize, fixed  RPN e14  Start Full New  schedule,  Not old.  e18  Additional after  6  Equivalent.  epoch. An interruption within the same total schedule can be obtained from checkpoint recovery;The change of the total schedule needs to be explicitly recorded as a new experiment.

Evidence:[12 — 18 protocol](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_e18_20260915/schedule_extension_protocol.json), [B RCNN 24 schedule and Initializing Hashi](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/final_comparison.json).

(b) There has been a process interruption and a phase link to recovery in the course of training, and one interruption should not be described as a final failure;The current outcome document clearly gives CONVERGED.This sandbox restricted cannot connects the user systemd bus and therefore does not prove that the whole machine has no training tasks in this list of isolation environments.

## 5. Two core issues and details of previous inquiries

### Question one: Why is upsampling always below baseline, can you just add a little more to recovery?

Existing results do not support “more point, more detection”.The four main experiments, double detector, input control, patch/ normalization ablation and sampling /voxel audits all suggest that geometric quality is as important as input distribution.generated points is not a new sensor observation;Add point count and do not guarantee recovery drop sampling lost target evidence.

However, the contribution of each mechanism has not been completely causally broken down.One patch repair, for example, led to a AP rise, which can only support the help of this configuration, and cannot quantifys the only reason for all the remaining gaps.The full details are given in Part II, section 5–7.

### Question two: Is it because only 3 epochs has run away and cannot's judgment?

It is true that cannot claims that 3 epochs has become convergence;It has now been completed on the basis of epoch checkpoint, 3,769 frame Validation and clarification of the rules of the platform period.The four best sets of AP increased after the extension, but still do not meet the reference baseline.Thus, only, due to a lack of three rounds of training, cannot fully explains the current gap;This still does not imply excludes other optimisation options.

**Why did you choose 3 at first?** The code proves that it is adaptation in advance of fixed;Existing log no records the basis for the selection of 3 instead of other values.cannot adds to the fact that speculations such as “the limited budget” “three rounds are sufficient”.

### 3N/3M How do you choose, what is the principle?

Sets the original point count N;Line A directly from N upsampling to 4N.Line B first fixed seed without replacement reserves M=floor(N/4) and then creates 4M.

- direct: Only strict 4N/4M rows generated by the network.
- observed-first: Keep all the real observed N/M rows and stabilize seed sample uniformly without replacement 3N/3M from the creation of rows, which together remains 4N/4M.
- Thus, 3N is intended to keep the total 4N while retaining N real points, which are not the three new sets of high-confidence points identified by the network.
- no scores 3N by confidence, curve, FPS or geometric;Also no assures that the coordinates are completely different from observed.without replacement assures row index does not repeat, does not imply geometric coordinates are necessarily the only one.
- PU-GCN patch enter 2048, output 8192, single patch normalization, and reasoning reverse normalization;generated points intensity with observed 1-NN succession.These are actual processes and no additional filtering mechanisms have been added that are not addressed.

The following extracts the key logic of the actual construction code, not the new algorithm:

```python
BASE_SEED = 20260718

def stable_seed(*parts):
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF

expected = 4 * observed.shape[0]  # predicted  Number of lines checked first equals  expected
seed = stable_seed(BASE_SEED, "e1", args.reference_token, frame)
rng = np.random.default_rng(seed)
selected = rng.choice(expected, size=3 * observed.shape[0], replace=False)
final = np.concatenate((observed, predicted[selected]), axis=0).astype(
    np.float32, copy=False
)
```

Full code:[prepare_centerpoint_observed_first_train.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/prepare_centerpoint_observed_first_train.py). Scripts also audit observed prefixes and postfixes and record SHA-256 for index selection and output.Although the document is known as CenterPoint, the resulting observed-first input is also used for the corresponding PointRCNN experiment;Specific input path can be found in final_comparison.json.

Detailed protocol:[protocol_and_code_en.md](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/protocol_and_code_en.md). Its end-state of operation only reflects the current progress, cannot overwrites completed main table or the latest results.

### Who the hell did you retrain?Trained frame Why do you have different numbers?

PU-GCN fixed by PU1K model-100;It's RPN/RCNN and CenterPoint from PointRCNN.KITTI train split Source 3,712 frame, PointRCNN Physical retention of 3,265 frame after training screening, related to target/scope filtering;val is complete 3,769 frame.These numbers are in different stages, and cannot writes 3,265 as validation set missing and cannot as detector epochs as PU-GCN epochs.

## 6. Key Issues - Processing - Results Table

Historical values keep their respective subsets and protocol and do not mix with the new AP_R40 table in 3.

| Problem | What did you do? | Results and supportable conclusions |
|---|---|---|
| Old Environment/ Custom CUDA ops Not Compatible | Rehabilitation of TF path, extended compilation/dependency, equipment tensor, optional import;Use of compatible environments for different projects | Multi-methods are operational;PU-EdgeFormer official result from ops-reuse compatible path, cannot says the original environment is reproduced |
| upsampling Results no Strict 4× | ratio audit; Redefinition of A=N  /  4N, B=M  /  4M;PV frame Audit | 09 month A/B 3,769 frame strict input completed;TULIP Retain Independence protocol |
| Old patch Space is not local | Audit of spatial boundaries to change local/override patch | The old Line B patch XY p90 can reach 124.10 m;Partially repair recovery partial performance, but not all drop points |
| First edition local patch copy/assemble | Statistical repetition rate, voxel occupied and modified construction | A batch of outputs repeat rows 59.4%;PointRCNN does not react with CenterPoint, which means that geometric is not enough. |
| PU-Net normalization Error | Old/correct normalization × Old/ Local patch 2×2 ablation | 256-frame PointRCNN, from the old combination 7.9823 to correct + local 43.5424, still below, also approved baseline 81.3033 |
| PointRCNN fixed sampling and negative sample size | sampler-safe, direct no-resample, E1/E2/E3 | (a) Fixing running errors;Cancel sampling does not guarantee improvement, even baseline changes, so cannot mixes two protocol explanations |
| CenterPoint voxel Interrupted/ Activated Changes | Number of voxel, cap hits and density distribution | The A method of generation more often triggers cap;B without trigger or drop point, cap is not the only explanation. |
| Whether to destroy detection by generating over-representation | dose pilot, full g10, small-scale real points | Kid dolls have a positive, full g10 unstable victory baseline;Full g25/g50 not completed, not confused with multi-scale pilot |
| Partial positive gain for detector-aware PDANS V1–V5 | Component ablation, Double detector, holdout64, 16-seed | V1 single time + 1.2227 does not constitute a steady advantage;16-seed average difference - 0.438, 9/16 negative;V5 holdout CP Car -0.1369 |
| region split Do we have more targets? | Three types of PointRCNN divisional assessment, holdout and sampling Sensitivity check | Individual holdout has added value, but the whole method pilot and worker/ sampling are unstable;cannot claims stability detector-wide improvement |
| From direct Generation Input to observed-first | Keep N/M real point and smoke 3N/3M generated points | observed-first outperforms of full-val corresponds to direct; Cooperation  adaptation  recovery, Evidently, hasn't won.  baseline |
| 3 epochs Enough | Former convergence audit;Later checkpoint and Full Validation | There was no evidence;The four groups are now set to threshold to reach the platform period. The latest values are shown in section 3–4 |
| Is a short medium platform equal to the end? | Check full curve and end patience | 12-epoch A RPN and 18-epoch B RCNN all show later improvements, which eventually extend to the right complete schedule |
| baseline Same as training budget | Clear list of original/ rare, official/adapted references |  _Other Organiser  baseline  (a) convergence training; The report retains this relative limitation and does not create fairness. |
| Children's collection and random sampling error | 16-seed, Duplicate sampling Uncertainty estimate | 20-frame smoke/pilot cannot supports small gains;Select the basis to be separated from the conclusion of an independent generalization |
| EAR strict is too slow. | a small number of frame measured and estimated | At the time, full A/B was estimated to be about 92–103 days, not completed;Do not impersonate the old EAR assessment as the new strict full |
| SPU-PMD Access Difficulties | Fix Import/Environment and Run 4-frame smoke | finite output and local detection available, but not complete AP |
| ModelNet40 classification Chain not completed | Preparation project, protocol;PU-GCN/line 8 Sample smoke | No local complete 12,311 ×, no PointNet++ classification accuracy;There are HPC files cannot and no log is classified as a new experiment |
| HPC migration failed | bundle, Checklist, dry-run, Connection Check | DNS/ Authentication Block, not completed Remote Transmission/Opt, cannot Written as HPC Full run |
| Disk Pressure | Multiplely rotated authorization to clean up, retain end product and list | For historical clean-up capacity see Part II, section 12;Those capacity and free space are not today's real-time values. |
| There's a lot of untraceable lab codes. | Source Snapshot and SHA-256 manifest | 09 month 151-file snapshot has a record;The old snapshot does not include, later added convergence script, has to be quoted separately |
| Delay in the publication of papers and reports | Generate this unified record and keep the original version | This no rewrites all existing PPT/ dissertations sections, and old materials still need to be manually synchronized with the latest tables. |

## 7. Completed, Partially Completed with no boundary

Completed: major historical multi-mode experiments and multiple diagnostics (based on their protocol);PU-GCN 09-10 double detector 20-arm full matrix;CenterPoint 12-epoch A/B certification for observed-first;PointRCNN A/B Phased Platform Period Certification;This is a Chinese briefing and a consolidated record.

 Not completed or does not establish: PU-GCN  Network.  KITTI  and retracing, ModelNet40  Official classification, EAR  New  strict full-val, SPU-PMD full-val, HPC  Successful migration, complete approach.  detector adaptation,  All  baseline  convergence, each category has convergence, independent  test  Improved, all of it.  PPT/ _Other Organiser

does not imply "All experiments completed" and does not imply "The algorithm has exceeded baseline".

## 8. Results, Charts, Codes and Evidence Navigator

### Recent results and off-the-shelf curve

- [PointRCNN Final result](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/FINAL_RESULTS.md)
- [Final complete JSON (includes each wheel tracks, old results and source code/weight Hash)](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/final_comparison.json)
- [Final Task Status](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/status.json)
- [A RPN 18-epoch Curve](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_e18_20260915/reports/pointrcnn_line_a_pugcn_observed_first_rpn_convergence.png)
- [A RCNN 18-epoch Curve](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_e18_20260915/reports/pointrcnn_line_a_pugcn_observed_first_rcnn_convergence.png)
- [B RPN 18-epoch Curve](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_e18_20260915/reports/pointrcnn_line_b_pugcn_observed_first_rpn_convergence.png)
- [B RCNN 24-epoch Curve](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/schedule_24/reports/pointrcnn_line_b_pugcn_observed_first_rcnn_convergence.png)
- [B RCNN 24-epoch Round by Round CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/schedule_24/reports/pointrcnn_line_b_pugcn_observed_first_rcnn_convergence.csv)
- [CenterPoint A/B Curve by Curve](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_20260914/reports/centerpoint_epoch_validation_curve.png)
- [CenterPoint A/B Round by Round CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_20260914/reports/centerpoint_epoch_validation_curve.csv)
- [09-10 Original 20-arm full-val Table](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/full_val_detector_matrix.md)
- [09-10 Full Category AP](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/all_classes_ap_r40.csv)
- [This is the latest comparison of four sets of CSV.](/home/ra87racy/reports/LATEST_DETECTOR_COMPARISON_20260917.csv)

### Key codes used in practice

- [observed-first input construction and frame audit](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/prepare_centerpoint_observed_first_train.py)
- [Original full-val Main Process](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pugcn_detector_adaptation_full_val_20260908.sh)
- [PointRCNN Phased training](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pointrcnn_full_train_stage.py)
- [PointRCNN checkpoint Authentication](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pointrcnn_checkpoint_split_eval.py)
- [PointRCNN convergence Main Process](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pugcn_pointrcnn_convergence_20260914.sh)
- [PointRCNN Platform period determination and summary](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/summarize_pugcn_pointrcnn_convergence_20260914.py)
- [B RCNN Extension of dispatcher to satisfy sentence](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_line_b_rcnn_until_converged_20260916.py)
- [CenterPoint convergence Main Process](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pugcn_centerpoint_convergence_20260914.sh)
- [CenterPoint Platform period determination and summary](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/summarize_pugcn_centerpoint_convergence_20260914.py)
- [Old 3-epoch convergence audit](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/convergence_audit.md)
- [09-10 Source Snapshot Description](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/experiment_source_full.md)
- [ModelNet40 State](/home/ra87racy/projects/modelnet40_pointnet2_upsampling/reports/modelnet40_x4_final_protocol_report.md)

## 9. Conclusions for dissertation/responsiveness

> In KITTI Line A/B protocol, which is strictly controlled by point count, the generic pre-training upsampling method does not result in a stable 3D detection gain.The retention of real sites and the introduction of detector adaptation have significantly reduced the gap with reference baseline.Further epoch verification shows that the current training settings of PointRCNN and CenterPoint meet the pre-defined AP period, but the best results are still below and the existing baseline.The results show that the addition of only to point count and the extension of the current fit schedule are not sufficient to close the gap;At the same time, baseline training budget is not fully matched and validation set is used for checkpoint, which limits the causal interpretation and generalization of conclusions.

---

## Part II: Full historical history as of 2026-09-10

**The following is a historical snapshot, not the latest.Particular attention is paid to the fact that “3-epoch” did not prove that convergence was retained as a fact at that time;09-14  to  09-17  Extensions  schedule  It was a new experiment which did not retroactively turn the original three-wheel experiment into a convergence. The old operational status, available space, and current best is valid only for 09-10.**

Retain the original results, failures and non-implementation and do not delete negative results.Original file:[09-10 General History](/home/ra87racy/reports/ALL_EXECUTED_WORK_EXPERIMENT_CHANGE_RESULT_SUMMARY_20260910_EN.md).

### General record of all implemented, experimental adjustments and results

**Check deadline: 2026-09-10 (Europe/Berlin)**
**Scope of verification:`/home/ra87racy` Local Workspace**
**Principle: Only code changes, command output, logs, manifest, projection documents, assessment forms, checkpoint, reporting or clean-up lists are recorded for work that can be shown to have actually taken place.**

---

#### 1. How does this general record distinguish between "do" and "do"

Status definition:

- **Completed**: Target data and assessments are completed and complete outputs or results tables are available.
- **Partially completed**: Only generation, smoke, subset, training or some stage, no completes the original full assessment.
- **Failed/ Suspended**: actually started, but failed or stopped because of environmental problems, CUDA, data, sampling, running time or resources.
- **Changes completed but no final result**: The code/environment/script has indeed been modified, but no is sufficient evidence that the whole experiment has been completed.
- **Audit/analysis completed**(a) Read the output and calculate the analysis of geometric, voxel, detection transfer, sampling;Not the results of the new model training.
- **Not implemented**: appears only in plan, empty directory or list of feasibility, and does not count as experimental results.

This time scan to PointRCNN Main Project `results/` Down **132 First Level Results Directory**. These include both formal full-scale experiments and smoke, pilot, ablation, trouble diagnosis, visualization and reporting generation.This paper combines the records by “experimental” to avoid miscalculating multiple versions of the same projection into several independent experiments.

##### 1.1 Warning for Critical Evaluation

1. Final matrix display for 2026-09-08 to 09-10 **KITTI AP_R40**Every arm must **3,769/3,769** frame only entered main table.
2.  Earlier.  PointRCNN  Local  evaluator  Default  41  individual  precision sample  Medium Every  4  One by one. The essence is old.  **AP_R11**. The old table is kept as historical evidence, but cannot compares directly with the latest AP_R40 table.
3. Early experiments used different kinds of experiments. `RPN.NUM_POINTS`, distance proposal, input sampling, whole frame /patch, point count cap and checkpoint;They cannot is encoded as a "same protocol rankings."
4. CenterPoint entered by voxel and PointRCNN entered by fixed point count sampling;The document level exact 4× does not imply detector actually uses all 4× points.

---

#### 2. Overall conclusion of the current work

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

#### 3. Final application of strict 4× protocol

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

#### 4. Updated and complete results: PU-GCN × detector adaptation × full KITTI val

##### 4.1 PointRCNN, Car 3D AP_R40, 3,769 frame /arm

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

##### 4.2 CenterPoint, Car 3D AP_R40, 3,769 frame /arm

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

##### 4.3 CenterPoint Full-category review

Best observed-first adapted and no exceeded the match on Pedestrian/Cyclist baseline:

| Line | Category | baseline Moderate 3D AP_R40 | observed-first adapted | margin |
|---|---|---:|---:|---:|
| A | Pedestrian | 50.6533 | 48.4156 | -2.2377 |
| A | Cyclist | 64.6054 | 57.6989 | -6.9065 |
| B | Pedestrian | 47.4395 (adapted baseline) | 44.0568 | -3.3827 |
| B | Cyclist | 42.0672 (adapted baseline) | 33.5610 | -8.5062 |

##### 4.4 Practical Training Adjustment

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

#### 5.  Unity of the full amount frozen detector  exact-4×  Experiment 2026-07)

##### 5.1 PointRCNN All 3,769 frame

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

##### 5.2 CenterPoint All 3,769 frame, Moderate 3D AP_R40

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

##### 5.3 Full Method Implementation Status

- **PDANS**: Initial due to extension source / `pytorch3d` The failure resulted in the repair of environmental and equipment-related codes and the final completion of the Line A, Line B strict exact-4× and double detector assessments.
- **PU-GCN**(b) Rehabilitation of TensorFlow/CUDA op, completion of full generation and assessment of both formal lines;And then I finished detector adaptation, the whole matrix.
- **PU-EdgeFormer**: The direct recurrence of the original warehouse environment failed;The results of protocol were then harmonized by replicating the compatible PU-GCN ops/checkpoint reasoning path.The fact must be spelled out that cannot claims to be an unsuited original warehouse with a single key.
- **PU-Net**(a) The following: Python 2, 3, TF custom op, CPU fallback, compilation and normalization rehabilitation;The results of the old adapter were extremely poor and the small-scale results improved after the rehabilitation, but were not yet the best approach.
- **TULIP**: complete Line A/Line B all 3,769 frame reasoning and PointRCNN/CenterPoint assessment, but output is not strict exact-4× and PointRCNN used `RPN=4096`, close distance proposal, cannot to be incorporated into the Unified Four Method main table.
- **EAR**: Only the early full volume/historical input assessment and subsequent 5 frame strict smoke completed;no completes the new strict Line A+Line B full weight generation.
- **SPU-PMD**: Only 4 frame smoke and PointRCNN small sample tried, no full-val AP.

---

#### 6., Key Experimental Adjustments, Why, What happens when they're modified

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

#### 7. Root Experiment and ablation Analysis

##### 7.1 Non-local problem of old patch

- The old extractor, using coarse space bin, cuts the block by line and does not guarantee that the same patch is a local surface.
- Line B patch diagonal median of XY **30.46 m**, p90 **124.10 m**, clearly contradicts the distribution of local object patch during training.
- The first local patch version improves locality, but appears **59.4% duplicate rows**.
- This version: PointRCNN Moderate from **57.1829 to 43.6976**; CenterPoint from **61.1172 to 77.4150**. Also occupied voxels median value from **11,584.5 down to 2,124.5**, indicates that different detector has different preferences for point distribution.

##### 7.2 PU-Net  normalization  × patch  It's...  2×2  Correlation ablation 256  frame)

| Configure | PointRCNN Moderate 3D | CenterPoint Moderate 3D |
|---|---:|---:|
| baseline | 81.3033 | 79.8368 |
| wrong norm + old patch | 7.9823 | 15.9788 |
| fixed norm + old patch | 18.2742 | 41.2227 |
| wrong norm + local patch | 6.3008 | 16.4119 |
| fixed norm + local patch | **43.5424** | **46.2047** |

The normalization error and patch are not only real problems, but they can be repaired in such a way as to be significant recovery, but are still not close to baseline.

##### 7.3 PointRCNN / CenterPoint input bottleneck

- PointRCNN Default entry fixed 16,384 point.
- CenterPoint voxel size `[0.05,0.05,0.1]`For each voxel up to 5 points, test up to 40,000 voxels.
- 32 frame Line A Audit median voxels / Trigger 40k cap frame Number: original `14,944 / 0`; PDANS `36,913 / 6`; PU-GCN `45,108 / 29`; PU-EdgeFormer `46,182 / 29`; PU-Net `55,384 / 32`.
- Line B no triggers voxel cap, but there is still a marked decline in performance.So cap is an important issue for Line A, but not the only explanation across the board.

##### 7.4 direct no-resample Full Volume Experiment

| Enter | PointRCNN Car 3D Easy / Moderate / Hard |
|---|---:|
| standard original, fixed 16,384 | 92.26 / **81.95** / 77.75 |
| original full all-valid | 79.83 / **74.91** / 72.73 |
| Line A PDANS all-valid | 83.59 / **63.49** / 54.34 |
| Line A PU-GCN all-valid | 78.21 / **54.26** / 47.11 |
| Line A PU-EdgeFormer all-valid | 67.11 / **43.85** / 37.10 |
| Line A PU-Net all-valid | 12.23 / **8.82** / 8.17 |

Conclusion: Internet training relies on fixed sampling distribution;Plug all points directly into the model and even damage original, cannot and use "sampling" as a recovery program.

##### 7.5 dose / generated points ratio

Test on 256 frame `g2.5/g5/g7.5/g10`:

- Line A PDANS Moderate around `85.10 / 82.37 / 82.25 / 82.42`It's for baseline. `82.66`; Little dose has seen sub-benefits.
- Line B each dose all below corresponds to the thin baseline.
- All 3,769 frame `g10`: Line A PDANS/PU-GCN/PU-EdgeFormer/PU-Net Moderate `79.951/79.478/76.433/72.371`; Line B `59.048/55.954/52.415/45.700`.
- `g25/g50` no finished, cannot report is validated.

##### 7.6 detector-aware PDANS V2–V5 (both 256 frame development experiments)

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

##### 7.7 object-preserving and Method Component ablation (PointRCNN, pilot256)

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

##### 7.8 region-split Category III exact-4× pilot256

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

##### 7.9 Line B c2048 / E1 / consensus / random Select Experiment (pilot256)

PointRCNN Car Moderate 3D:

| Select protocol | baseline | PDANS | PU-GCN | PU-EdgeFormer | PU-Net fixed |
|---|---:|---:|---:|---:|---:|
| c2048-r4 | 64.0416 | 36.6162 | 24.9619 | 21.2409 | 2.3971 |
| E1 | 64.0362 | **44.0487** | 32.8364 | 28.2585 | 10.5793 |
| consensus | 63.4197 | 39.2524 | **33.1648** | 22.4376 | **18.8792** |

CenterPoint  The same.  Line B pilot  It's...  Car Moderate: baseline 61.6558; random PDANS/PU-GCN/PU-EdgeFormer/PU-Net is `47.7441/41.3784/35.4329/15.1487`; consensus is `43.6085/37.3131/27.3251/19.8041`. Selecting the strategy changes the sorting of the method, but no sets recovery baseline.

##### 7.10 PointRCNN 16-seed sampling Noise Probe

- fixed input, checkpoint, split and merge rules only change evaluator sampling seed and run seed 0–15.
- The new 24 slot all PASS;6 times CUDA occasional segmentation fault has been internally re-engineered.
- Car Moderate 3D: baseline `80.998±1.198`; V1 full `80.560±1.086`; Pair margin averages **-0.438**, sd **1.769**, Scope `-3.51…+2.23`.
- The original seed + 1.2227 is the maximum value of full arm 16 times;9/16 is negative, median -0.75.
- `MDE95≈3.47 AP`. This proves that the sorting of about 0.4–1.2 AP below split-region path is not readable;cannot sets this noise value directly to native E2 full-val.

##### 7.11 exact-4× quality selection little probe

- Line A PDANS actually did confidence selection: no-quota 3 frame, voxel quota=6 is 5 frame.
- no-quota: quality precision median 0.6821, below random 0.8155;unsupported ratio 0.5825, above random 0.4940.
- quota=6: quality precision 0.5965, below random 0.8024;unsupported 0.6329, above random 0.5337.
- This is the 3/5 frame mechanism probe, no detector AP;As a result, the then confidence quality selector was rejected and not extended.

---

#### 8. Training Graduation: 64 frame Screening All train pilot  /  full val

##### 8.1 PointRCNN finetune64 Six Arm Screening

- fixed 64 train frame, actual 63;seed `20260823`.
- RPN 3 epochs + RCNN 3 epochs, every stage of epoch 93 steps.
- 6 input arm × pretrained/finetuned, for a total of 12 256-frame eval, all produced 256 forecasts and PASS.
- Moderate Change: Line A PU-GCN `60.2662→64.4408` (+4.1746); Line B PU-GCN `32.9201→37.5150` (+4.5949); Line B PDANS `43.0067→47.4327` (+4.4260); Line B baseline `66.6188→62.9139` (-3.7049).
- PDANS Line A values differ in the two versions of the report;A later summary of eval256 should be used, and cannot should select a more favourable old number.
- Conclusion: The very small training set can show that “the area fits the potential of recovery”, but baseline is also degraded and cannot concludes.

##### 8.2  Full  3,712 train  It's...  256-val pilot

Training uses a full train split, but only fixed 256 val was evaluated:

- PointRCNN A PU-GCN adapted 67.2345, for baseline 79.1910, bad - 11.9565.
- PointRCNN B PU-GCN adapted 47.0204, for adapted baseline 67.2485, bad - 20.2281.
- CenterPoint A adapted 73.9468, for 79.8368, bad - 5.8900.
- CenterPoint B adapted 60.6978, for 65.5501, bad - 4.8523.
- PointRCNN observed-first: A 67.7909, compared to direct +0.5564;B 56.2443, more than direct +9.2239.

These are 256-frame pilot, which have been replaced by full 3,769-frame of Section 4, but are still progressive evidence of actual run-off.

---

#### 9. Early baseline, method access and failure record (2026-05 to 06)

##### 9.1 PointRCNN / EAR / PU-Net Early Job

- History PointRCNN baseline Record: 3D AP `89.19/78.85/77.91`.
- Once clean original rerun:`89.2040/78.6795/77.8099`.
- EAR Historical Report:`88.5558/77.9565/76.7821`A little below baseline.
- Early PU-Net x2 Full generation and assessment completed once: 3D AP `0.1976/1.1364/1.1364`This means that old appliances are completely incompatible.
- Follow-up recovered/fullframe report shows the appearance of PU-Net Moderate or about 59.2885;As the input is different from the recovery process, the very low result above is not the same as protocol and is retained as provenance risk.
-  Yes.  PU-Net  Increase  resume/chunk, Python 3, TensorFlow op  Compile, compile, GPU bootstrap, CPU fallback,  Data reading and normalization fix.

##### 9.2 May 16 Unharmonized Profiles

Recorded Moderate 3D: original 77.93, downsample50 76.99, EAR 73.89, PU-Net 35.26, PU-GCN 50.65.PU-GCN was used. `RPN=26000`, the configuration is not uniform, so it is reserved for early exploration.

##### 9.3 RPN4096 / no-distance-propose Comparison

- original failed with negative dimension sampling at 4,096.
- downsample Completed: 3D `83.6938/65.1662/60.4608`.
- EAR Completed:`81.5120/62.6687/57.8478`.
- PU-Net failed after 16 samples.
- PU-GCN was missing 8 frame, not completed.
- TULIP Completed:`27.5045/16.9675/13.7001`.

##### 9.4 TULIP

- Finish full Line A Parsing Approximate `54.3492/35.0242/30.3302`; There are also Line B full and CenterPoint full results.
- CenterPoint Moderate: Line A Car/Ped/Cyc `31.77/15.98/4.59`; Line B `17.68/2.99/0.08`Far from below original `79.28/50.65/64.61`.
-  The primary output is...  range-image  Vertical  4×,  But switch back.  XYZ  I don't know.  exact 4×; Actual output is about 0.41× and 0.73–0.78×.
- 8  A known frame in  4,096/2,048/1,024  Configure Reasons  distance proposal  It's...  far/near bucket  Empty set entry  CUDA NMS  And abort. Add sole-nonempty-bucket guard later and use score-only proposal to bypass.

##### 9.5 EAR strict-4× Feasibility

- 5 frame × Line A/B, for a total of 10 entries smoke all PASS.
- CPU whole frame achieves a global kNN/PCA and Python cycle;Line A About 26–29 min/ frame, Line B About 9–10 min/ frame.
- It is estimated that two lines will take approximately 92–103 days;A frame byte-identical duplication was also observed.
- For running costs no completes the new strict full-val regeneration;This is...**Time out/unfeasible**is not the final negative result of the model.

##### 9.6 SPU-PMD

- First time because it's not available. `pyvista` Failure;Second cause `knn_cuda` Failure.
- Modify `utils/MeshUtil.py` with `main.py` After lazy import, 4 frame inference succeeded;For each complete frame, go down to 2,048, then export 8,192, finite, i.e. add a point to 3× relative to the internal input.
- PointRCNN `RPN=4096` An assessment of the failure of sampler underflow;was replaced by `RPN=2048` After 4 frame ran, detections was `1/0/0/2`.
- Too few samples, no AP;No full-val.

##### 9.7 PDANS Initial failure and follow-up repair

- Initial inference due to lack of extension source and `pytorch3d` (a) Unrun;`pointops` It was successfully compiled and prepared for 4 XYZ input.
-  Modify  pointnet2  It's...  device/tensor  Processing, compilation and  CUDA  After compatibility, the formal follow-up is completed  Line A/Line B  Full.

##### 9.8 PU-EdgeFormer failed in the early stages of direct recurrence

- Create a Python 3.6.8 / TensorFlow 1.13.1 environment.
- custom ops due to CUDA 10 path hard-coded, no `nvcc` And checkpoint failed.
- Only 3 KITTI frame was converted to 2,048 and generated Plotly, which was visualized, but at that time no model inference.
- The results of the subsequent formal unification were from the ops/reuse compatible path;The two phases must be presented separately.

##### 9.9 ratio audit

The actual audit found that:

- EAR around `1.008×`, not the target x2.
- The PU-Net output ratio changes with frame.
- The PU-GCN/PDANS old process is influenced by 100k cap.
- TULIP is in range image vertical 4×, but may be less than the input point when returned to XYZ.

The audit directly facilitated strict exact-4× Line A/B protocol.

---

#### 10. ModelNet40 Job Record

##### 10.1 actually completed

- Create `modelnet40_pointnet2_upsampling` protocol and the structure of the report.
- PU-GCN Local smoke: Line A 8/8`1024→4096`; Line B 8/8, `256→1024`; Output finite, point count exact.
- ModelNet40 data preprocessing, upsampling fit, classification training portal, audit and reporting documents have been prepared/modified.
- `thesis_demo` Created `train_cls.py`, `pointnet_cls.py`, dataset/preprocess scaffold.

##### 10.2 no Completed, cannot Written Results

- Both lines 12 and 311 complete local PU-GCN generate no submission/ no completion evidence.
- PointNet++ 5 classification branch is `NOT_SUBMITTED`no classification accuracy.
- no complete geometry benchmark, classification training curve or final ModelNet40 comparison table.
- `dgcnn_cls.py`, `train_upsampling.py`, part of metrics/visualization/README was empty;belongs to scaffold, not to run experiments.
- EAR/PDANS/PU-Net 12,311, which is visible on HPC, can only be recorded as “existing product” in the no local log. cannot in this record asserts that it was the task of the current working area itself.

---

#### 11. HPC Migration, running and resources Job

##### 11.1 KITTI upsampling Migration Package

- - Preparation of 10 variant × 3,769 frame, about 42 GiB migration/submission structure, checksum, job script and instructions.
- dry run Real implementation:`tinyx` DNS parsing failed;`tinyx.nhr.fau.de` SSH authentication failed.
- Thus no actually completes the remote transmission and no official remote operation results.

##### 11.2 Conda/ Operating Environment

An independent environment has actually been established:`ear`, `openpcdet_centerpoint`, `pointrcnn_old`, `puedgeformer`, `pugcn`, `punet_tf`, `tulip`, `upsampling_basic`. These environments support the recovery and rehabilitation of the generations that TensorFlow/PyTorch/CUDA relies on.

---

#### 12. Disk Cleaning and Data Governance Record

##### 12.1 2026-05-18 Safe Cleanup

- Delete empty/ smoke directory, PU-GCN evaluation log, PU-Net preparation log and `__pycache__`.
- The actual disk changes about 420 MiB;There's an executive log.

##### 12.2 2026-06-26  Phase I

- Cleaning up pip cache about 13 GiB and conda clean about 4.1 GiB;`df`  Show approximately reduction in space used  16 GiB.
- Move 6 PU-GCN mega-centres to `~/TO_DELETE_REVIEW`About 193 GiB;This phase is subject to review and is not tantamount to immediate and permanent deletion.
- Follow-up de-listed about 1,487,400 `.xyz`29,128 `.png`22,573 `.json`22,349 `.bin`21,966 `.csv`7,201 `.md` Middle file path.

##### 12.3 2026-07-31 User-Authorized Major Cleanup

- Clean up old strict line trees, CenterPoint reconstructed inputs, 8 Group raw patch outputs, old Line B cap100k,`thesis_demo` venv, cache/VSCode backup, etc.
- Report the release. **459 GiB** (492,379,832,320 bytes).
- Check retention of 8 group `merged_raw`, 8 `final_bin`, checkpoint, source code and current results.

The current file system is about 1.0 TiB, with a total usage of about 993 GiB and the remaining about 31–32 GiB, still in high occupancy status.

---

#### 13. Actual Code Changes Record

##### 13.1 PointRCNN Main repository

Current branch:`experiment/centerpoint-unified-line-a-b`. There are still local unsubmitted changes to the tracking document:**4 files, 72 insertions, 8 deletions**.

- `lib/config.py`: `yaml.load` was replaced by `yaml.safe_load`.
- `lib/datasets/kitti_dataset.py`: NFS/ and read to add up to 20 retests, bytes and short-read checks.
- `lib/datasets/kitti_rcnn_dataset.py`: repairing the negative sample size of far points exceeding fixed sampling;Rare input security patches.
- `lib/rpn/proposal_layer.py`: far-only / near-only empty bucket guard, avoid TULIP/region-split to make CUDA NMS.
- A large number of untraceed scripts have been added: strict-x4, patch extraction, methods wrapper, E1/E2/E3, CenterPoint, detector-aware V2–V5, region split, training, full assessment, analysis, visualization, packing and recovery scripts.

##### 13.2 PU-Net

Current diff:**35 files, 226 insertions, 122 deletions**It's a new one. `.so`/`.o`.

- Python 2  /  3 compatible;TensorFlow API Adjustment.
- sampling/grouping/interpolation/CD/EMD custom op ABI and CUDA compile script restoration.
- CPU fallback with GPU bootstrap.
- Data reading, provider, model utils, normalization and full-frame adapter rehabilitation.

##### 13.3 PDANS

Current tracking diff:**2 files, 5 insertions, 4 deletions**.

- `pointnet2/util.py`, `pointnet2_utils.py` The device-aware tensor/CUDA process.
- Add checkpoint and pointops compilations.

##### 13.4 SPU-PMD

Current diff:**9 files, 18 insertions, 42 deletions**.

- lazy imports, remove/replace non-dependent path.
- pointnet2 C++/CUDA extended header file compatible with source.
- operations and utility adjustments;Saved model/extension construction product.

##### 13.5 OpenPCDet / CenterPoint

Current diff:**3 files, 21 insertions, 7 deletions**, add 1 tool scripts.

- dataset import is optionally dependent on treatment.
- data processor point input/ voxel compatible.
- checkpoint from detector template `weights_only=False` Wait for recovery compatible.
- Add `create_kitti_val_infos_only.py`.

##### 13.6 PU-GCN

- `tf_ops/compile.sh`: **14 insertions, 3 deletions**, fix the computer TensorFlow/CUDA op compile path.
- Add PU1K pretrained checkpoint, local backup and strict/full-frame wrapper.

##### 13.7 TULIP / PU-EdgeFormer

- TULIP Source Repository Files remain largely unchanged, but add local `scripts/`, `results/` And the cache.
- The official PU-EdgeFormer experiment is accessed mainly through ops-reuse compatible copies;Evidence of the failure of the original warehouse ' s initial custom-op was retained.

##### 13.8 version control state risk

PointRCNN Most of the experimental scripts, reports, results, weights, backups and tools in the main warehouse remain **untracked**; Git History is mainly upstream and old submission, cannot only dependent `git log` Revert this job.Current replicability relies on workspace documents, results manifest and reports.09. **151 Source/ CUDA/ Configuration/ split Files**and generate SHA-256 manifest, which reduces this risk.

---

#### 14. Analysis, visualization, reporting and dissertation material

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

#### 15. Methodological feasibility survey: investigated, but no model experiment

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

#### 16. Explicitly failed, not completed or does not establish

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

#### 17. Total line of work by time

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

#### 18. Final conclusions to be written in the paper

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

#### 19. Main Evidence Portal

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

#### 20. Current finish state

- The latest full-val core experiment:**Completed**.
- Two PU-GCN validation inputs:**3,769/3,769 + 3,769/3,769 completed**.
- Test assessment:**20/20 PASS**.
- Current training/drill process:**None**.
- Final conclusions:**adaptation Significant recovery, but no configuration exceeding pair baseline**.
- Material that still needs to be manually synchronized: 256-frame adaptation tables in previous sections of 09-08 and in the report should be replaced by 09-10 full-val tables;This HotSync is currently no complete, so it is not written as updated.
