RESEARCH ARCHIVE CONTENTS
=========================

This directory contains the locally authored integration, evaluation, auditing,
training, and reporting code used for the thesis experiments, together with
compact result evidence.

source/pointrcnn_workspace/
  - scripts/: KITTI preparation, strict x4 assembly, method adapters, detector
    evaluation, adaptation training, convergence evaluation, diagnostics, and
    figure/report generation.
  - tools/: locally added evaluation and analysis utilities.
  - lib/: tracked PointRCNN files modified in the local experiment workspace.
  - README.md and LICENSE: upstream PointRCNN context and licence.

source/modelnet40/
  - scripts/: ModelNet40 sampling, PU-GCN execution, count checks, and provenance
    audits.
  - reports/: compact ModelNet40 protocol and smoke-result records.

results/
  - overall experiment brief, work record, and experiment inventory;
  - final full-validation detector matrices and strict-x4 audits;
  - CenterPoint and PointRCNN convergence summaries, curves, and final comparison.

environment/
  - documentation about dependencies and omitted external assets.

NOT INCLUDED
------------

The package does not duplicate KITTI or ModelNet40 datasets, third-party method
repositories, virtual environments, pretrained weights, trained detector
checkpoints, complete generated point-cloud trees, or raw per-frame detector
predictions. These assets are large, may be licensed separately, and are not
appropriate for direct publication in GitHub/GitLab. See DATA_AND_LARGE_FILES.md.

REPRODUCIBILITY BOUNDARY
------------------------

The compact files preserve the experimental protocol, code, aggregate results,
and provenance evidence. A full numerical rerun additionally requires the
licensed datasets, upstream repositories at compatible revisions, pretrained
weights, CUDA/TensorFlow/PyTorch custom operations, and substantial GPU compute.

