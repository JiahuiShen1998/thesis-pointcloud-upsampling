MODELNET40 RESEARCH ARCHIVE
===========================

Purpose
-------
This folder contains the source code and result files for the ModelNet40
point-cloud upsampling experiments described in the thesis.

Folder contents
---------------
code/scripts/
    Python programs for data preparation, upsampling integration, PointNet++
    training/evaluation, geometric evaluation, audits, figures, and reports.

code/configs/
    YAML configuration files. The final two-line PointNet++ configurations are
    in configs/pointnet2_x4_two_line/.

code/jobs/
    Slurm job files for smoke tests, full training, and method-specific runs.

code/LEGACY_PROJECT_README.md
    Historical share-package README. Its unequal-cardinality geometry wording
    is obsolete and must not override the final equal-N reports.

code/POINTNET2_UPSTREAM_LICENSE
    License copied from the upstream PointNet++ implementation associated with
    model source snapshots stored in the result folders.

results/reports/
    CSV, JSON, Markdown, and text reports, including per-sample geometry,
    classification summaries, protocol audits, and failure/recovery evidence.

results/pointnet2_final_runs/
    Fourteen final/control PointNet++ run directories. Each contains metrics,
    a 200-epoch training log, result summaries, a best-model checkpoint, and
    the model source snapshot saved by the training code.

results/figures/
    Result plots, point-cloud comparisons, and interactive HTML visualizations.

Authoritative result files
--------------------------
1. reports/modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.csv
2. reports/modelnet40_geometry_equal_n_protocol.md
3. reports/modelnet40_geometry_equal_n_summary.csv
4. reports/modelnet40_geometry_equal_n_per_sample.csv
5. reports/modelnet40_sa1_reception_probe.md
6. pointnet2_final_runs/*/metrics.json and train.log

Data not duplicated in this Git archive
---------------------------------------
The raw ModelNet40 dataset, generated per-object NPY point clouds, third-party
method repositories, external pretrained upsampling weights, temporary output
folders, and full scheduler logs are not duplicated. They are large, may have
separate licenses, and are not necessary to inspect the supplied final tables.
The scripts/configurations document their expected locations. Re-running them
requires obtaining those dependencies separately and updating cluster-specific
paths.

Important interpretation limits
-------------------------------
- Final classification branches use one seed (42), not a multi-seed study.
- Best checkpoints were selected on the test set because no validation split
  was used.
- The reported OA/mAcc implementation averages batch-level quantities.
- Implemented Chamfer distance uses unsquared Euclidean nearest-neighbour
  distances.
- “Training from scratch” refers to downstream PointNet++ classifiers; the
  upsampling methods use existing pretrained weights.

Snapshot provenance
-------------------
Copied on 2026-09-27 from a working tree based on Git commit
daacfaf1a76be4c42d01431c225a69a715818db2. The source working tree contained
later modified and untracked files; SHA256SUMS fixes the exact archive content.

