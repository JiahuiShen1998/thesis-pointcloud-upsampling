# Code walkthrough and contribution map

This page provides three ways into the same project. The primary contribution is a source-backed investigation of how point-cloud upsampling interacts with downstream classification and LiDAR detection, supported by experiment automation and diagnostics.

See [the technology stack](TECH_STACK.md) for the framework/library inventory, model families, operator dependencies and recorded environment versions.

## Training and evaluation

For ML/CV engineering, start with the data contract and follow a sample through training, evaluation and result reporting.

| Component | Implementation to inspect | Engineering point |
| --- | --- | --- |
| Point-cloud dataset | [modelnet_npy_dataloader.py](../research/modelnet40/code/scripts/modelnet_npy_dataloader.py) | Manifest/path fallback, stable sampling seeds, finite XYZ checks, strict counts and centroid/radius normalization |
| Training driver | [train_pointnet2.py](../research/modelnet40/code/scripts/train_pointnet2.py) | Training augmentation, optimizer/scheduler, evaluation, checkpoint selection and per-class result files |
| Protocol records | [14 YAML configurations](../research/modelnet40/code/configs/pointnet2_x4_two_line/) | Explicit inputs, cardinalities and budgets; translate into CLI arguments |
| Runnable model example | [demo_inference.py](../tools/demo_inference.py) | Real archived weights, strict input validation, label provenance and CPU execution |
| Evidence reconstruction | [reproduce_results.py](../tools/reproduce_results.py) | Recompute aggregates and deltas, validate against source records and record hashes |
| Integrity checks | [tests](../tests/) and [check_archive.py](../tools/check_archive.py) | Fail on corrupt evidence; verify archive bytes after complete LFS download |

A useful review exercise is to compare `--no-allow-resample` with the loader's default behavior, then trace how the reported OA is computed. The strict setting prevents point-count mismatches from silently becoming a different experiment. The OA implementation explains why the stored value should not be relabelled as a globally sample-weighted metric.

## Experimental evidence

For research roles, begin with the competing explanations and controls:

1. **Densification versus recovery.** Line A asks whether increasing the original input density helps; Line B asks whether a method can recover useful information after fourfold sparsification. These questions use different baselines.
2. **Geometry versus downstream utility.** [Equal-cardinality geometry evaluation](../research/modelnet40/code/scripts/compute_geometry_equal_n.py) and its [protocol](../research/modelnet40/results/reports/modelnet40_geometry_equal_n_protocol.md) separate geometric agreement from classification. PU-GCN's strong CD does not establish superior Line B recognition.
3. **Input distribution versus model adaptation.** The [KITTI experiment brief](../research/kitti/results/EXPERIMENT_BRIEF_20260917_EN.md) and [convergence records](../research/kitti/results/convergence/) distinguish the same input before/after detector adaptation from comparison against a different baseline input and training budget.
4. **Evidence traceability.** Follow a README number into a rebuilt CSV, then through `provenance.json` to the original training metrics, curve or per-shape rows. Fourteen classifier run folders contain both checkpoint/model snapshots and full training logs.

The negative and mixed results are central findings. Extra density is not evidence of recovered object structure, a better detector input, or statistically significant improvement. The manuscript and reproduction guide state the one-seed protocol, test-based checkpoint selection, incomplete upstream environment locks and the scope of selected examples. A future study should use separate validation selection, multiple seeds and matched adaptation budgets; those extensions are not presented as completed work.

## LiDAR perception

For perception roles, follow the boundary between generated points and the detector's actual input.

| Component | Implementation to inspect | Question it addresses |
| --- | --- | --- |
| Exact x4 assembly | [assemble_exact4n_quality_selected.py](../research/kitti/source/pointrcnn_workspace/scripts/assemble_exact4n_quality_selected.py) | Which candidate points enter the assembled cloud under a fixed cardinality? |
| Observed-point preservation | [assemble_lineb_c2048_observed_preserved.py](../research/kitti/source/pointrcnn_workspace/scripts/assemble_lineb_c2048_observed_preserved.py) | Are measured observations preserved when filling a sparse cloud? |
| Count/protocol checks | [verify_patch_causal_strict_x4.py](../research/kitti/source/pointrcnn_workspace/scripts/verify_patch_causal_strict_x4.py) | Does the prepared input satisfy the intended x4 contract? |
| Retention analysis | [analyze_e2_observed_retention.py](../research/kitti/source/pointrcnn_workspace/scripts/analyze_e2_observed_retention.py) | How many original observations survive the input pipeline? |
| CenterPoint integration | [run_centerpoint_exact4n_observed_first.py](../research/kitti/source/pointrcnn_workspace/scripts/run_centerpoint_exact4n_observed_first.py) | How is the assembled cloud routed into the voxel-based detector? |
| Adaptation preparation | [prepare_centerpoint_observed_first_train.py](../research/kitti/source/pointrcnn_workspace/scripts/prepare_centerpoint_observed_first_train.py) | How is the detector training input aligned with the changed distribution? |
| Object-level comparison | [compare_patch_causal_detector_boxes.py](../research/kitti/source/pointrcnn_workspace/scripts/compare_patch_causal_detector_boxes.py) | Which detections change, and how do box matches differ? |

PointRCNN consumes a fixed 16,384-point input, while CenterPoint's voxelization and point/voxel limits constrain what survives preprocessing. A larger cloud on disk does not imply that all new points reach the detector or contribute useful signal. The pipeline therefore examines observed-point retention, exact input cardinality and adaptation alongside AP.

The [selected lost-car figure](../thesis/figure/k_case_cp_lost.png) makes that issue concrete: the baseline has a qualifying match and the denser input does not. This is a diagnostic example. The [full-validation matrix](../research/kitti/results/latest_full_validation/full_val_detector_matrix.csv) and final convergence files support dataset-wide conclusions.

This is an offline perception research project. Real-time latency guarantees, vehicle deployment, tracking, sensor fusion and production safety validation are outside its demonstrated scope.

## Authorship and reused components

The project-specific work is the experimental framing, method integration, data preparation and validation, training/evaluation orchestration, reporting and failure analysis. The runnable repository tools additionally package the archived evidence for inspection.

PointNet++ model implementations, pretrained upsampling architectures, PointRCNN, OpenPCDet/CenterPoint and third-party operations originate in the cited upstream projects. The per-run `pointnet2_cls_ssg.py` and `pointnet2_utils.py` copies are preserved model snapshots and marked as vendored for language statistics. They are not claimed as newly invented architectures. Other integration files may combine upstream code with project modifications; consult their headers and retained notices before reuse.

EAR is the project's EAR-style geometric baseline. Its name must not imply verified equivalence to every setting in the original EAR paper. Likewise, training the downstream classifier from scratch does not mean that the upsamplers were trained from scratch.

References and research context are in [the manuscript](../thesis.pdf) and [bibliography source](../thesis/bibfiles/). Third-party notices remain with the supplied code. No new blanket licence is applied to the archive.

## Suggested review sequence

1. Read the two result tables and protocol caveats on the [homepage](../README.md).
2. Run evidence reconstruction and open its report.
3. Run the CPU checkpoint demo, then inspect the training/data code above.
4. Choose the geometry/control analysis or LiDAR input/adaptation path according to the role.
5. Consult [REPRODUCING.md](REPRODUCING.md) before attempting a full rerun.
