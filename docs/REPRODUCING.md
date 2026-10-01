# Reproducing the thesis project

Run commands from the repository root unless a different directory is stated. The tested portable environment is Python 3.12 on Windows; the CPU dependency pins also target Linux. The original full experiments used separate GPU environments on research servers.

## What each level reproduces

| Level | Input | Output | Included and verified locally? |
| --- | --- | --- | --- |
| Evidence reconstruction | Archived metrics, logs, per-shape geometry and detector curves | Recomputed tables, deltas and source hashes | Yes; standard library only |
| Classifier inference | Archived PointNet++ checkpoint and one included cloud | Top-five predictions and provenance | Yes; CPU, PyTorch 2.6.0+cpu / NumPy 2.2.6 |
| Manuscript build | LaTeX, bibliography and supplied figures | Thesis PDF | Verified in the handover audit with TeX Live 2025; see [build notes](../thesis/README.md) |
| Full experiments | Complete datasets, preprocessing, method code/weights and GPU stacks | New upsampled data, trained models and dataset-wide metrics | Not rerun during packaging; several required assets are external |

The first level reaggregates saved evidence. The second executes a real saved model. Neither constitutes full numerical reproduction of the thesis experiments.

## Clone and access

The GitHub repository is public; viewing and cloning it do not require signing in. Install Git LFS, then skip large assets for the initial lightweight checks:

```sh
git -c filter.lfs.smudge= -c filter.lfs.required=false -c filter.lfs.process= clone https://github.com/JiahuiShen1998/thesis-pointcloud-upsampling.git
cd thesis-pointcloud-upsampling
```

The same portable commands apply to the school archive, after cloning its `thesis-archive` branch:

```sh
git -c filter.lfs.smudge= -c filter.lfs.required=false -c filter.lfs.process= clone --single-branch --branch thesis-archive git@gitlab.lms.tf.fau.de:marina.ritthaler/preprocessingpc.git thesis-archive
cd thesis-archive
```

The two clone blocks are alternatives. SSH requires a registered school GitLab key. A ZIP may contain LFS pointers, so it is not the verified route for obtaining checkpoint assets.

## Reconstruct the reported tables

```sh
python tools/reproduce_results.py
python -m unittest discover -s tests -v
```

Expected reconstruction status:

```text
PASS: 14 classifier runs, 14 geometry groups, 4 detector comparisons
```

Outputs under `outputs/reproduced/`:

- `modelnet40_classification.csv`: all 12 main runs plus two mesh-reference controls, with best/final accuracy and baseline deltas.
- `modelnet40_geometry.csv`: mean CD, HD and NUC rebuilt from 2,468 per-shape test records for each of 14 groups.
- `kitti_adaptation.csv`: four same-input adaptation comparisons, recomputed from archived validation curves and baseline records.
- `report.md`: readable tables with metric and selection caveats.
- `provenance.json`: SHA-256 hashes of the evidence actually used.

Use `--output-dir PATH` for another destination. The script checks all 200 logged classifier epochs, strict point-count settings, recorded seed 42, full KITTI validation coverage, curve maxima and agreement with published tables. It exits with a nonzero status for inconsistent evidence. Tests deliberately corrupt metrics, split labels and curve coverage to verify these failure paths.

Geometry reconstruction aggregates existing distances; recomputing those distances requires the full point-cloud variants and reference sets. KITTI reconstruction does not regenerate boxes or evaluate raw predictions.

## CPU checkpoint inference

Fetch the two assets required by the default demo:

```sh
git lfs install --local
git lfs pull --include="research/modelnet40/results/pointnet2_final_runs/lineA_original_baseline/checkpoints/best_model.pth,thesis/evidence/modelnet40_hpc/data/pointcloud_examples/lineA_airplane_0627_original.npy"
python -m venv .venv
```

Linux:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/demo_inference.py
```

Windows PowerShell, without changing the script-execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/demo_inference.py
```

The root `requirements.txt` pins the CPU demo's top-level dependencies. The optional `environment.yml` describes the same scope for Conda (`conda env create -f environment.yml`); the venv route was tested locally, while the Conda route was not separately exercised. These files are not a lock file for the historical training servers.

Expected default result: `lineA_airplane_0627_original.npy`, 1,024 points, top class `airplane`, checkpoint epoch 85. The locally verified probability was approximately 0.999998; small floating-point differences across platforms are possible. No benchmark accuracy is inferred from this selected example.

`outputs/demo/prediction.json` records the top-five probabilities, CPU/GPU device, seed, versions, forward time and hashes of the sample, checkpoint, label mapping and saved model code. Forward time is a local observation, not a hardware benchmark.

```sh
python tools/demo_inference.py --sample /path/to/cloud.npy --output outputs/demo/custom.json
```

A custom sample must be finite XYZ data with shape `(1024, 3)`. The script rejects wrong counts, zero-radius clouds and LFS pointer files. It subtracts the centroid and normalizes by the maximum radius, matching the archived dataset loader. There is no silent resampling. CUDA is optional via `--device cuda`, after installing a compatible GPU PyTorch build in a separate environment.

## Full ModelNet40 experiments

### Assets and environments

| Component | Included | Required separately |
| --- | --- | --- |
| Dataset | Selected clouds, result records and preparation scripts | Full [ModelNet40](https://modelnet.cs.princeton.edu/) mesh data; all prepared train/test variants and references |
| PointNet++ | Training driver, dataset loader, per-run model snapshots and 14 classifier checkpoints | Upstream `provider.py` and supporting repository; GPU environment for full training |
| Five upsampling methods | Project integration scripts/configurations and experimental records | Applicable method repositories, pretrained weights and compiled operators |
| Geometry evaluation | Protocol, evaluation code, per-shape CSV and summaries | Complete input/reference point clouds to recompute distances |
| Scheduler | Historical Slurm job files | A matching cluster or locally adapted commands and resource limits |

Start with [environment notes](../research/modelnet40/ENVIRONMENT_NOTES.md), [the source scripts](../research/modelnet40/code/scripts/) and [all 14 protocol configurations](../research/modelnet40/code/configs/pointnet2_x4_two_line/). The YAML files record experiment settings; the training driver **does not accept `--config`**. Transfer their values to the CLI shown below.

The original complete upstream revisions and CUDA/operator environment locks are unavailable. Installing current upstream code alone cannot establish a bit-for-bit rerun. Record the upstream commit, weight checksum, Python/framework/CUDA/compiler versions and data hashes for any new run; compare its model source with the archived snapshots.

### Required dataset layout

```text
/path/to/variant/
  metadata/
    class_to_idx.json          {"airplane": 0, ...} for all 40 classes
    idx_to_class.json          {"0": "airplane", ...}
    train_manifest.csv        optional; see fields below
    test_manifest.csv         optional
  train/<class>/<shape_id>.npy
  test/<class>/<shape_id>.npy
```

Manifest fields are `output_npy` (or `output_path`), `shape_id`, `class_name`, `label`. Keep the original label mapping and the 9,843/2,468 train/test shape split. The loader can fall back to the directory layout when manifests are absent or old paths no longer exist. Verify each variant contains the same intended shapes before scheduling training.

Expected point counts: A Original 1,024; A upsampled 4,096; B sparse 256; B upsampled 1,024. Mesh-reference controls use 256 and 4,096. Use `--no-allow-resample` so mismatches fail instead of silently changing the input protocol.

### Run the archived classifier driver

Obtain [the upstream PointNet/PointNet++ implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) into the path expected by the driver. Run this only when preparing full training, not for the standalone demo:

```sh
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git research/modelnet40/code/external/Pointnet_Pointnet2_pytorch
git -C research/modelnet40/code/external/Pointnet_Pointnet2_pytorch rev-parse HEAD
```

Use a separate training environment with NumPy, PyTorch compatible with your GPU and `tqdm`. The historical cluster recorded PyTorch 2.6 / Python 3.12, but the complete GPU dependency lock is missing. Check compiled-method environments separately; the CPU demo requirements do not install CUDA upsampling operators.

The following command encodes the archived A-baseline training configuration. Replace `/path/to/variant` with prepared data. It is a documented rerun recipe, not a claim that the full dataset and GPU run were validated during packaging:

```sh
python research/modelnet40/code/scripts/train_pointnet2.py \
  --variant lineA_original_baseline \
  --data-root /path/to/variant \
  --model pointnet2_cls_ssg --num-category 40 --num-point 1024 \
  --no-allow-resample --batch-size 24 --epoch 200 \
  --learning-rate 0.001 --decay-rate 0.0001 --optimizer Adam \
  --seed 42 --gpu 0 --num-workers 4 \
  --output-dir outputs/modelnet40/lineA_original_baseline \
  --log-file outputs/modelnet40/lineA_original_baseline/train.log \
  --reports-dir outputs/modelnet40/reports
```

This multiline example uses Bash continuation. In PowerShell, place it on one line or use PowerShell's continuation syntax. For an initial wiring check, set `--epoch 1 --max-train-batches 2 --max-eval-batches 2`; this is not a scientific result. `--use-cpu --num-workers 0` supports CPU debugging but is not recommended for the full training budget. Keep all new outputs separate from archived results.

For each other variant, change `--variant`, `--data-root`, `--num-point` and output paths according to its configuration. The original training logic, including scheduler ordering and test-based checkpoint selection, is preserved. Method inference/preprocessing and geometry evaluation have their own scripts and data contracts; inspect `--help` and the recorded protocol before running them. Do not substitute random resampling for the omitted method outputs and label it a thesis rerun.

## Full KITTI experiments

Obtain [KITTI 3D object detection data](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) with the applicable terms, preserving the thesis train/validation split of 3,712 / 3,769 frames. Full input trees, detector checkpoints and raw per-frame predictions are not in this archive. The frozen PU-GCN upsampler uses the recorded PU1K `model-100` asset; it must also be obtained separately.

Prepare separate compatible environments for [PointRCNN](https://github.com/sshaoshuai/PointRCNN), [OpenPCDet/CenterPoint](https://github.com/open-mmlab/OpenPCDet), and the upsamplers. [Dependency boundaries](../research/kitti/environment/REPRODUCIBILITY_NOTES.md) and [omitted assets](../research/kitti/DATA_AND_LARGE_FILES.md) explain the remaining gaps. The archived source contains local integration changes; cloning upstream is not a replacement for applying those changes.

The executable experiment sequence is recorded in these entry points under `research/kitti/source/pointrcnn_workspace/scripts/`:

1. [Observed-first input assembly](../research/kitti/source/pointrcnn_workspace/scripts/assemble_lineb_c2048_observed_preserved.py) and [strict x4 checks](../research/kitti/source/pointrcnn_workspace/scripts/verify_patch_causal_strict_x4.py).
2. [Full-validation adaptation run](../research/kitti/source/pointrcnn_workspace/scripts/run_pugcn_detector_adaptation_full_val_20260908.sh).
3. [PointRCNN convergence run](../research/kitti/source/pointrcnn_workspace/scripts/run_pugcn_pointrcnn_convergence_20260914.sh) and [Line B RCNN extension](../research/kitti/source/pointrcnn_workspace/scripts/resume_pugcn_pointrcnn_line_b_rcnn_convergence_20260916.sh).
4. [CenterPoint convergence run](../research/kitti/source/pointrcnn_workspace/scripts/run_pugcn_centerpoint_convergence_20260914.sh).
5. [Convergence analysis](../research/kitti/source/pointrcnn_workspace/scripts/analyze_pugcn_detector_adaptation_convergence.py) and [object-level comparison](../research/kitti/source/pointrcnn_workspace/scripts/compare_patch_causal_detector_boxes.py).

These are historical server entry points, with absolute paths and environment assumptions. Review and replace data, model, output and interpreter paths before execution; check their command-line arguments and logging destinations. A validated portable full-KITTI launcher is not included. Never overwrite the archived result folders while testing a new setup.

## Metric and checkpoint contracts

- **Classification:** best test OA, final-epoch test OA and per-class accuracy are different fields. The archived implementation averages batch-level OA and class-specific batch accuracies. Seed 42 and test-based best-checkpoint selection limit statistical and generalization claims.
- **Geometry:** CD uses unsquared Euclidean nearest-neighbour distances, not squared Chamfer. Equal-cardinality reference selection differs between lines; retain the reference labels in the generated CSV. HD and NUC measure different aspects of geometry and do not replace downstream evaluation.
- **KITTI:** the featured result is Car 3D Moderate AP_R40 on 3,769 validation frames. It is not a KITTI test-server leaderboard rank. Selected-frame and object-level figures cannot stand in for full-set AP.
- **PointRCNN:** use RCNN records in [final_comparison.json](../research/kitti/results/convergence/final_comparison.json). The final selected RPN epoch is 14 for both lines; RCNN selection is epoch 2 for A and 19 for B. Recorded RCNN schedules are 18 and 24 epochs. Earlier partial convergence files describe intermediate runs.
- **CenterPoint:** both selected checkpoints are epoch 12 of 12 in [the convergence summary](../research/kitti/results/convergence/centerpoint_convergence_summary.json). Curves are in [centerpoint_epoch_validation_curve.csv](../research/kitti/results/convergence/centerpoint_epoch_validation_curve.csv).
- **Reference budgets:** A compares against official original-input checkpoints; B against an archived three-epoch adapted sparse baseline. The before/after upsampled-input gain must not be described as a gain over that baseline.
- **Inference demo:** one selected sample, one archived classifier, no training. It says nothing about a new dataset-wide score.

## Complete-download integrity and troubleshooting

For the complete handover, fetch every LFS asset and verify all archived bytes:

```sh
git lfs pull
python tools/check_archive.py
git lfs fsck
```

A lightweight or selective-LFS clone is expected to fail the full-archive checksum test for assets not downloaded. `tools/check_archive.py --refresh` is for intentional maintainer updates; do not use it to hide download errors. Generated root `outputs/`, local virtual environments and Python caches are excluded from checksum refresh.

| Symptom | Action |
| --- | --- |
| Checkpoint/sample is an LFS pointer | Repeat the selective `git lfs pull` above with an authenticated account |
| `No module named torch` | Run pip with the same `.venv` interpreter that runs the demo |
| CUDA unavailable | Use the verified CPU demo or configure a separate GPU environment |
| Wrong point count | Check the variant and data preparation; do not silently crop/pad |
| Evidence check fails | Inspect the named source and compare with a clean download; retain the failure |
| Full training cannot import `provider` | Obtain the PointNet++ repository at the documented `external/` path |
| Reviewer cannot open the school GitLab archive | Use the public GitHub portfolio link, or request school project access |

See [the complete archive guide](ARCHIVE_GUIDE.md) for the manuscript build, provenance and outstanding final-defence materials.
