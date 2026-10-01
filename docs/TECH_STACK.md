# Technology stack and implementation details

This is a map of the technologies used in the thesis code, experiment records and runnable repository tools. Each entry explains its purpose and points to source evidence. The work spans ML engineering, experimental research and LiDAR perception; dependencies belong to different stages and environments.

[Project overview](../README.md) · [Reproduction commands](REPRODUCING.md) · [Contribution map](CODE_WALKTHROUGH.md)

**Reading the scope:** the CPU demo and evidence reconstruction are verified runnable entry points. Research scripts and job files document the historical experiments and often require external data, weights or operator builds. Model architectures and detector frameworks retain their upstream attribution.

## 1. Languages, runtimes and software structure

| Technology | Concrete use in this project | Source evidence |
| --- | --- | --- |
| Python | Dataset preparation, method adapters, training/evaluation, geometry, diagnostics and result packaging | [Research scripts](../research/modelnet40/code/scripts/), [portable tools](../tools/) |
| Bash | Environment activation, CUDA diagnostics, experiment launch sequences and recovery jobs | [Environment activation](../research/modelnet40/code/scripts/activate_upsampling_env.sh), [adaptation launcher](../research/kitti/source/pointrcnn_workspace/scripts/run_pugcn_detector_adaptation_full_val_20260908.sh) |
| NumPy | Float32 point arrays, XYZ/XYZI transformations, deterministic index selection, `.npy` and binary file I/O | [Dataset loader](../research/modelnet40/code/scripts/modelnet_npy_dataloader.py), [strict KITTI conversion](../research/kitti/source/pointrcnn_workspace/scripts/wrappers/strict_x4_from_raw.py) |
| Python standard library | `argparse` CLIs, `pathlib` paths, dataclasses, CSV/JSON records, subprocess-based environment boundaries and SHA-256 provenance | [Result reconstruction](../tools/reproduce_results.py), [PU-EdgeFormer runner](../research/modelnet40/code/scripts/run_pu_edgeformer_modelnet40_labparams_full.py) |
| Process-based parallelism | `ProcessPoolExecutor` and `as_completed` for independent object preparation and geometry evaluation | [Mesh preparation](../research/modelnet40/code/scripts/prepare_modelnet40.py), [geometry evaluation](../research/modelnet40/code/scripts/compute_geometry_equal_n.py) |
| YAML / PyYAML | Experiment configuration records and research configuration loading | [Classifier configurations](../research/modelnet40/code/configs/pointnet2_x4_two_line/), [detector configuration loader](../research/kitti/source/pointrcnn_workspace/lib/config.py) |
| LaTeX / BibTeX | Manuscript source, mathematical notation, citations, cross-references and tables | [Main manuscript](../thesis/thesis.tex), [bibliography](../thesis/bibfiles/) |

The classifier YAML files record settings; `train_pointnet2.py` takes explicit CLI arguments. The reproduction guide maps the recorded values to those arguments.

## 2. Deep-learning frameworks and model families

| Component | Framework / method family | How it is used here | Entry point |
| --- | --- | --- | --- |
| PointNet++ SSG | PyTorch; hierarchical point-set feature learning | A separately trained classifier for each ModelNet40 input representation; archived weights also power the CPU demo | [Training driver](../research/modelnet40/code/scripts/train_pointnet2.py), [saved model](../research/modelnet40/results/pointnet2_final_runs/lineA_original_baseline/pointnet2_cls_ssg.py) |
| EAR-style baseline | Geometric point-cloud upsampling | Project baseline exposed through a wrapper and strict output checks; exact equivalence to the published EAR method is not established | [EAR utilities](../research/modelnet40/code/scripts/ear_modelnet40_utils.py) |
| PU-Net | Legacy TensorFlow integration; point-feature expansion | Pretrained x4 point generation through a separate method environment | [ModelNet40 wrapper](../research/modelnet40/code/scripts/punet_modelnet40_utils.py), [KITTI adapter](../research/kitti/source/pointrcnn_workspace/scripts/punet_kitti_adapter.py) |
| PU-GCN | Legacy TensorFlow integration; graph convolution and NodeShuffle | Pretrained object/patch upsampling; the final KITTI study uses frozen PU1K `model-100` weights | [ModelNet40 wrapper](../research/modelnet40/code/scripts/pugcn_modelnet40_utils.py), [patch inference](../research/kitti/source/pointrcnn_workspace/scripts/wrappers/tf_pugcn_family_patch_infer.py) |
| PU-EdgeFormer | Legacy TensorFlow implementation in the recorded lab environment; EdgeConv and multi-head self-attention | Pretrained upsampling with method-specific patch preparation, output reconciliation and full-dataset generation records | [Full generation runner](../research/modelnet40/code/scripts/run_pu_edgeformer_modelnet40_labparams_full.py) |
| PDANS | PyTorch; conditional diffusion with adaptive noise scheduling | Pretrained upsampling, explicit tensor/device handling and GPU nearest-neighbour dependencies | [ModelNet40 wrapper](../research/modelnet40/code/scripts/pdans_modelnet40_utils.py), [KITTI adapter](../research/kitti/source/pointrcnn_workspace/scripts/pdans_kitti_adapter.py) |
| PointRCNN | PyTorch; proposal generation and second-stage refinement | Frozen detector evaluation and staged RPN/RCNN adaptation to changed point-cloud inputs | [Training entry](../research/kitti/source/pointrcnn_workspace/tools/train_rcnn.py), [adaptation sequence](../research/kitti/source/pointrcnn_workspace/scripts/run_pugcn_pointrcnn_convergence_20260914.sh) |
| CenterPoint / OpenPCDet | PyTorch and sparse-convolution/voxel processing | A second detection pipeline for cross-detector comparisons, input-mechanism analysis and adaptation | [Observed-first runner](../research/kitti/source/pointrcnn_workspace/scripts/run_centerpoint_exact4n_observed_first.py), [input analysis](../research/kitti/source/pointrcnn_workspace/scripts/analyze_centerpoint_input_mechanism.py) |

The model-family descriptions are explained and cited in [the thesis fundamentals](../thesis/texfiles/02_fundamentals.tex). The thesis implementation contribution is the integration, experimental protocol and analysis. The learned upsamplers use existing pretrained weights; downstream classifier training and detector adaptation are separate stages.

## 3. PyTorch training and inference implementation

The [classifier driver](../research/modelnet40/code/scripts/train_pointnet2.py) provides a concrete path from a dataset to checkpoints and numerical reports:

- **Dataset / DataLoader:** manifest or directory-based sample discovery, class mapping, batches, worker processes and per-sample normalization.
- **Tensor layout:** finite `(N, 3)` XYZ samples become `(B, 3, N)` tensors for PointNet++. Class targets remain integer indices over 40 categories.
- **Point-set network operations:** the retained [PointNet++ utilities](../research/modelnet40/results/pointnet2_final_runs/lineA_original_baseline/pointnet2_utils.py) implement farthest-point sampling, radius-based grouping and set abstraction. These are upstream model components.
- **Network layers:** the saved SSG classifier uses learned point features, fully connected classification layers, batch normalization, dropout and log-softmax/NLL loss.
- **Training augmentation:** point dropout, random scaling and coordinate shifts are called through the upstream `provider` module.
- **Optimization:** final classifier runs use Adam, learning rate 0.001, weight decay 0.0001 and a StepLR schedule with a 20-epoch interval and factor 0.7. The driver also exposes SGD as an option; the final configuration records Adam.
- **Budget and logging:** batch size 24, 200 epochs and seed 42 are recorded for the final runs. `tqdm`, Python logging, per-class CSV output and `metrics.json` support inspection.
- **Checkpoint handling:** training records the model and optimizer state. The portable demo uses `weights_only=True`, strict model-state loading, `eval()` and `torch.inference_mode()`.
- **Device control:** the demo exposes CPU/CUDA selection, thread count and a seed, and rejects unavailable CUDA or invalid point clouds. Its default CPU path needs no custom CUDA operators.

The retained [PointRCNN trainer](../research/kitti/source/pointrcnn_workspace/tools/train_rcnn.py) includes TensorBoardX logging, optimizer/scheduler construction and an optional `DataParallel` path. That source capability is distinct from evidence that a particular archived experiment used multiple GPUs.

## 4. Geometry, spatial search and data preparation

| Technique / library | Implementation detail | Why it matters | Source |
| --- | --- | --- | --- |
| OFF mesh parsing | Parse vertices/faces and handle mesh input before sampling | CAD surfaces and LiDAR observations have different data contracts | [prepare_modelnet40.py](../research/modelnet40/code/scripts/prepare_modelnet40.py) |
| Area-weighted surface sampling | Choose triangles in proportion to area and sample points on the selected surface | Produces object clouds and independent mesh-reference controls | [Mesh preparation](../research/modelnet40/code/scripts/prepare_modelnet40.py), [reference generation](../research/modelnet40/code/scripts/build_mesh_reference_points.py) |
| Centroid / unit-sphere normalization | Subtract the sample centroid and divide by the maximum radius | Aligns the classifier's preprocessing with the stored experiment | [Dataset loader](../research/modelnet40/code/scripts/modelnet_npy_dataloader.py) |
| Seeded sampling without replacement | Stable object/frame identities determine sparse input selection | Reuses the same sparse observations across compared methods | [ModelNet40 protocol](../research/modelnet40/code/scripts/modelnet40_x4_protocol.py), [KITTI downsampling](../research/kitti/source/pointrcnn_workspace/scripts/prepare_kitti_downsampled_x4_val.py) |
| SciPy `cKDTree` | Nearest-neighbour distances and radius queries | Efficient CD/HD calculation, local density statistics and spatial diagnostics | [Equal-cardinality geometry](../research/modelnet40/code/scripts/compute_geometry_equal_n.py) |
| scikit-learn `NearestNeighbors` | Map generated XYZ to the nearest input observation and inspect method-output similarity | Recovers an intensity channel for XYZ-only method outputs and supports audits | [XYZI conversion](../research/kitti/source/pointrcnn_workspace/scripts/wrappers/strict_x4_from_raw.py), [similarity audit](../research/modelnet40/code/scripts/audit_method_output_similarity.py) |
| PyTorch3D GPU operators | Environment validation explicitly exercises `pytorch3d.ops.knn_points` on CUDA | Detects missing or incompatible PDANS operator dependencies before generation | [GPU validation gate](../research/modelnet40/code/scripts/validate_pdans_x4_gpu.sh) |
| Open3D | Point-cloud quality analysis and interactive inspection of detector evidence | Connects numerical diagnostics with spatial structure | [Quality-metric script](../research/modelnet40/code/scripts/compute_modelnet40_x4_quality_metrics.py), [evidence viewer](../research/kitti/source/pointrcnn_workspace/scripts/view_dual_detector_evidence_open3d.py) |

**Seed provenance:** the later sparse-input helpers use stable hashing. The initial mesh sampler uses Python's built-in `hash()`, so a fresh regeneration also needs control of Python hash randomization. The already stored original clouds are the inputs to the completed runs; this limitation is documented in [the experimental setup](../thesis/texfiles/04_experimental_setup.tex).

## 5. LiDAR perception and detector integration

The KITTI branch connects point generation to the inputs actually consumed by a detector:

1. **XYZI binary data:** frames contain float32 `(x, y, z, intensity)` records. [Input checks](../research/kitti/source/pointrcnn_workspace/scripts/check_detector_input_bins.py) and the [PointRCNN dataset loader](../research/kitti/source/pointrcnn_workspace/lib/datasets/kitti_rcnn_dataset.py) handle record shape, reading and sampling.
2. **Spatial patch construction:** the audited PU-GCN pipeline combines farthest-point centre selection, ball coverage and kNN support. The recorded final patch size is 2,048 input points and 8,192 output points. See [full-frame reconstruction](../research/kitti/source/pointrcnn_workspace/scripts/pugcn_kitti_full_frame_reconstruct.py) and the [experimental protocol](../thesis/texfiles/04_experimental_setup.tex).
3. **Coordinate restoration:** each patch retains its centroid and scale, enabling predictions to return to metric scene coordinates before merging. Ground-truth labels and calibration belong to downstream detection/evaluation, rather than candidate generation.
4. **Intensity reconstruction and cardinality:** [strict conversion](../research/kitti/source/pointrcnn_workspace/scripts/wrappers/strict_x4_from_raw.py) assigns nearest-input intensity and writes valid XYZI; [protocol checks](../research/kitti/source/pointrcnn_workspace/scripts/verify_patch_causal_strict_x4.py) validate the intended x4 relationship.
5. **Observed-first assembly:** [observed-point preservation](../research/kitti/source/pointrcnn_workspace/scripts/assemble_lineb_c2048_observed_preserved.py) and [retention analysis](../research/kitti/source/pointrcnn_workspace/scripts/analyze_e2_observed_retention.py) examine how measured and generated points survive input preparation.
6. **Detector input budgets:** PointRCNN consumes a fixed 16,384-point input in the featured protocol. [CenterPoint input analysis](../research/kitti/source/pointrcnn_workspace/scripts/analyze_centerpoint_input_mechanism.py) traces field-of-view/range filtering and voxel occupancy, including point and voxel limits.
7. **Boxes, proposals and NMS:** the [PointRCNN proposal layer](../research/kitti/source/pointrcnn_workspace/lib/rpn/proposal_layer.py) decodes 3D proposals and calls GPU rotated/normal NMS; [CenterPoint pre-NMS export](../research/kitti/source/pointrcnn_workspace/scripts/export_centerpoint_prenms_proposals.py) supports proposal-level inspection.
8. **Calibration and evaluation interfaces:** the [OpenPCDet-based evaluator](../research/kitti/source/pointrcnn_workspace/scripts/evaluate_kitti_r40_openpcdet_fast.py) connects predictions to KITTI evaluation; [box comparison](../research/kitti/source/pointrcnn_workspace/scripts/compare_patch_causal_detector_boxes.py) supports object-level diagnostics.
9. **Adaptation and convergence:** separate RPN/RCNN stages and the CenterPoint adaptation run record full-validation curves, selected epochs and recovery relative to the same unadapted input. See [convergence analysis](../research/kitti/source/pointrcnn_workspace/scripts/analyze_pugcn_detector_adaptation_convergence.py).

These interfaces are the central perception-engineering component: a denser file, retained sensor observations and a detector's effective input are different quantities that must be measured separately.

## 6. Evaluation and research methodology

| Metric / technique | Definition or role in this repository | Evidence |
| --- | --- | --- |
| Overall accuracy and mean class accuracy | Archived classifier evaluation averages batch-level quantities; best and final-epoch results are separate fields | [Classifier evaluation](../research/modelnet40/code/scripts/train_pointnet2.py), [final report](../research/modelnet40/results/reports/modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.csv) |
| Chamfer distance (CD) | Sum of the two mean nearest-neighbour distances, using unsquared Euclidean distance | [Geometry implementation](../research/modelnet40/code/scripts/compute_geometry_equal_n.py) |
| Hausdorff distance (HD) | Maximum of the two directed maximum nearest-neighbour distances | [Geometry implementation](../research/modelnet40/code/scripts/compute_geometry_equal_n.py) |
| Non-uniformity coefficient (NUC) | Radius-query neighbour-count variation; final implementation averages three radii, 0.02 / 0.05 / 0.10, using up to 128 sampled query centres | [Geometry implementation](../research/modelnet40/code/scripts/compute_geometry_equal_n.py) |
| Equal-cardinality controls | Match reference point counts to the compared representation before geometric ranking | [Protocol](../research/modelnet40/results/reports/modelnet40_geometry_equal_n_protocol.md) |
| KITTI AP_R40 | Featured results use Car 3D Moderate AP_R40 on all 3,769 validation frames | [Full-validation matrix](../research/kitti/results/latest_full_validation/full_val_detector_matrix.csv) |
| 3D / BEV box comparison | Per-object matching and detection transitions help explain selected failures | [Comparison tool](../research/kitti/source/pointrcnn_workspace/scripts/compare_patch_causal_detector_boxes.py) |
| Convergence and checkpoint selection | Compare curves, stage budgets and selected epochs; preserve the reference baseline | [Convergence archive](../research/kitti/results/convergence/), [reconstruction tool](../tools/reproduce_results.py) |

The classification runs use one seed and test-based best-checkpoint selection. Adaptation baselines have different training budgets. Selected-frame failures are diagnostics, while full-validation AP supports dataset-level conclusions. The [metric contracts](REPRODUCING.md#metric-and-checkpoint-contracts) retain these distinctions.

## 7. GPU integration, environment debugging and HPC

| Technology / practice | Recorded use | Evidence |
| --- | --- | --- |
| NVIDIA CUDA runtime and toolkit | Method-specific GPU execution, device checks and custom-op builds | [CUDA setup](../research/modelnet40/code/scripts/setup_cuda_toolkit_env.sh), [GPU diagnostics](../research/modelnet40/code/scripts/print_gpu_cuda_diagnostics.sh) |
| GCC / `nvcc` / shared-library paths | Export `CUDA_HOME`, configure compiler/runtime lookup, compile upstream operators | [Toolkit setup](../research/modelnet40/code/scripts/setup_cuda_toolkit_env.sh), [operator build](../research/modelnet40/code/scripts/recompile_tf_ops.sh) |
| TensorFlow custom operations | Compile/load the PU-Net and PU-GCN research extensions in their legacy environments | [Build script](../research/modelnet40/code/scripts/recompile_tf_ops.sh), [operator smoke validation](../research/modelnet40/code/scripts/validate_tf_custom_ops_gpu.sh) |
| TensorFlow framework linking | Resolve framework-library names and environment-specific linker paths | [Framework-link setup](../research/modelnet40/code/scripts/setup_tf_framework_link.sh) |
| GPU tensor/device consistency | PDANS integration records device-following tensors and float32 sampling after a non-finite half-precision trial | [Implementation-change record](../thesis/texfiles/04_implementation_changes.tex) |
| Slurm | GPU/CPU resource requests, runtime limits, log paths and array/chunk jobs | [Final training job](../research/modelnet40/code/jobs/pointnet2_full/run_lineA_original_baseline_1024.sbatch), [generation jobs](../research/modelnet40/code/jobs/) |
| Conda and environment modules | Isolate legacy method stacks and load the appropriate cluster toolchain | [Activation script](../research/modelnet40/code/scripts/activate_upsampling_env.sh), [environment notes](../research/modelnet40/ENVIRONMENT_NOTES.md) |
| Resume and completion audits | Recover missing objects/frames and check shape, finite values and coverage before downstream evaluation | [Full generation runner](../research/modelnet40/code/scripts/run_pu_edgeformer_modelnet40_labparams_full.py), [PU-GCN recovery job](../research/modelnet40/code/jobs/pugcn_x4/run_lineB_pugcn_resume_missing_2080ti.sbatch) |

The integration work includes short-read handling, sampling edge cases, empty proposal buckets, sparse fill sampling and framework/operator compatibility. [The manuscript implementation table](../thesis/texfiles/04_implementation_changes.tex) records these changes and their effect on the experiments. CUDA/operator integration does not imply that the underlying research kernels were newly authored for this thesis.

## 8. Visualization, reporting and scientific communication

| Technology | Output / purpose | Source |
| --- | --- | --- |
| Matplotlib | Accuracy plots, convergence curves, BEV diagnostics and comparative figures | [Result figure generator](../research/modelnet40/code/scripts/generate_pointnet2_final_visualization.py), [detection analysis](../research/kitti/source/pointrcnn_workspace/scripts/build_pointrcnn_detector_recovery_combined_analysis.py) |
| Plotly | Interactive 3D scatter views with rotation, zoom and method selection; exported HTML reports | [Interactive generator](../research/modelnet40/code/scripts/generate_interactive_pointcloud_v2.py) |
| Open3D | Interactive point-cloud and detection-evidence inspection | [Viewer](../research/kitti/source/pointrcnn_workspace/scripts/view_dual_detector_evidence_open3d.py) |
| Pillow | Image composition, sizing and presentation assets | [Presentation builder](../research/kitti/source/pointrcnn_workspace/scripts/build_english_upsampling_presentation.py) |
| OpenCV | Image reading in the retained KITTI dataset interface | [Dataset source](../research/kitti/source/pointrcnn_workspace/lib/datasets/kitti_dataset.py) |
| pandas | Tabular input and result handling in reporting/presentation code | [Presentation builder](../research/kitti/source/pointrcnn_workspace/scripts/build_english_upsampling_presentation.py) |
| TensorBoardX | Scalar logging in the retained PointRCNN training/evaluation tooling | [Training entry](../research/kitti/source/pointrcnn_workspace/tools/train_rcnn.py) |
| python-pptx | Editable progress/result presentations | [ModelNet40 presentation generator](../research/modelnet40/code/scripts/generate_results_presentation.py) |
| ReportLab | Programmatic PDF research dossiers | [Dossier generator](../research/modelnet40/code/scripts/generate_thesis_complete_dossier_pdf.py) |
| pdfLaTeX / BibTeX / latexmk | Build the thesis from supplied source, bibliography and figure assets | [Verified build notes](../thesis/README.md) |

HTML files are generated visualization artifacts. The project remains a Python-based research and perception pipeline; the exports are marked as generated in [.gitattributes](../.gitattributes). Existing presentation decks are supporting materials; the final defence deck remains pending.

## 9. Data formats and reproducibility tooling

| Interface / tool | What is recorded or checked | Entry point |
| --- | --- | --- |
| `.off` mesh files | Input geometry for object-level surface sampling | [Mesh parser](../research/modelnet40/code/scripts/prepare_modelnet40.py) |
| `.npy` XYZ arrays | Prepared ModelNet40 inputs, method outputs and selected demo samples | [Dataset loader](../research/modelnet40/code/scripts/modelnet_npy_dataloader.py) |
| KITTI `.bin` XYZI | Scene-level float32 point records and detector inputs | [Input checks](../research/kitti/source/pointrcnn_workspace/scripts/check_detector_input_bins.py) |
| YAML, CSV and JSON | Configuration records, manifests, per-shape metrics, training summaries and convergence traces | [Configurations](../research/modelnet40/code/configs/pointnet2_x4_two_line/), [reports](../research/modelnet40/results/reports/) |
| PyTorch `.pth` checkpoints | Saved classifier state and training metadata; 14 archived best checkpoints | [Final/control runs](../research/modelnet40/results/pointnet2_final_runs/) |
| Git / Git LFS | Version-controlled source and reports, with large checkpoints and selected binary assets in LFS | [.gitattributes](../.gitattributes), [download guide](REPRODUCING.md#clone-and-access) |
| SHA-256 | Whole-archive file integrity and source hashes for regenerated tables/inference | [Archive verifier](../tools/check_archive.py), [result tool](../tools/reproduce_results.py), [demo](../tools/demo_inference.py) |
| Python `unittest` | 11 tests cover numerical comparisons, incomplete records, incorrect splits and asset diagnostics | [Tests](../tests/) |
| GitHub Actions | Independent evidence-reconstruction and actual CPU-inference jobs on hosted Linux runners | [Workflow](../.github/workflows/reproducibility.yml) |
| `venv` / pip | Isolated, pinned top-level dependencies for the portable CPU demo | [Requirements](../requirements.txt), [commands](REPRODUCING.md#cpu-checkpoint-inference) |

## 10. Version and environment evidence

These rows describe separate environments. They must not be combined into one universal requirements file.

| Environment / stage | Recorded versions | Evidence and verification scope |
| --- | --- | --- |
| Portable CPU demo | Python 3.12, PyTorch 2.6.0+cpu, NumPy 2.2.6 | [Pinned dependencies](../requirements.txt); actual checkpoint inference verified locally on Windows and in [GitHub Actions](../.github/workflows/reproducibility.yml) on Linux |
| ModelNet40 HPC classifier jobs | Cluster module `python/pytorch2.6py3.12` | [Final job file](../research/modelnet40/code/jobs/pointnet2_full/run_lineA_original_baseline_1024.sbatch); historical job configuration, with the complete GPU lock unavailable |
| HPC operator toolchain setup | GCC 11.5.0, CUDA toolkit 11.8.0, `python/3.12-conda` module | [Toolkit script](../research/modelnet40/code/scripts/setup_cuda_toolkit_env.sh); a recorded setup path, separate from each method's actual Python environment |
| Lab PointRCNN | Python 3.9.25, PyTorch 2.8.0+cu128, NumPy 2.0.2, SciPy 1.13.1 | [Archived lab environment inventory](../thesis/evidence/kitti_lab/MASTER_INVENTORY_UPDATED_20260913.md) |
| Lab CenterPoint / OpenPCDet | Python 3.10.20, PyTorch 2.7.1+cu128, NumPy 1.26.4, spconv 2.3.6 | [Archived lab environment inventory](../thesis/evidence/kitti_lab/MASTER_INVENTORY_UPDATED_20260913.md) |
| Lab PU-Net / PU-GCN | Python 3.6.8, TensorFlow 1.13.1, NumPy 1.19.5 | [Archived lab environment inventory](../thesis/evidence/kitti_lab/MASTER_INVENTORY_UPDATED_20260913.md) |
| Lab PU-EdgeFormer | Python 3.6.8, TensorFlow 1.13.1, NumPy 1.16.6 | [Archived lab environment inventory](../thesis/evidence/kitti_lab/MASTER_INVENTORY_UPDATED_20260913.md) |
| Manuscript build | TeX Live 2025 with pdfLaTeX/latexmk | [Build verification](../thesis/README.md) |

Hardware records distinguish FAU Slurm resources from the lab workstation. The manuscript records an RTX 2080 Ti recovery path for PU-GCN after RTX 3080 failures, and a lab Quadro RTX 4000 with 8 GiB memory. These are historical execution observations, not recommended minimum specifications or currently allocated hardware. See [the environment section](../thesis/texfiles/04_experimental_setup.tex).

## 11. Read the stack by role

| Role | Most relevant demonstrated work | Suggested code path |
| --- | --- | --- |
| ML / CV engineering | PyTorch data contracts, training/evaluation orchestration, checkpoint inference, dependency boundaries, validation and CI | Dataset loader → training driver → CPU demo → tests/workflow |
| Research | Controlled input lines, multiple method families, equal-cardinality geometry, per-sample evidence, baseline and selection analysis | Preparation → geometry evaluator → final logs/curves → result reconstruction |
| Perception | XYZI handling, local patch reconstruction, preserved observations, detector point/voxel budgets, NMS/proposal diagnostics and adaptation | KITTI adapters → strict assembly → detector input analysis → adaptation/box comparison |

For exact runnable commands and external-asset requirements, use [REPRODUCING.md](REPRODUCING.md). For authorship and upstream attribution, use [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md#authorship-and-reused-components).
