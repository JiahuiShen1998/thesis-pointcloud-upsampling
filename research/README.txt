THESIS RESEARCH ARCHIVE
=======================

modelnet40/
  HPC-origin ModelNet40 point-cloud upsampling research. Contains scripts,
  configurations, Slurm jobs, aggregate and per-sample result reports,
  result figures, and 14 final/control PointNet++ training runs.
  All 14 runs were checked for valid metrics.json, train.log containing
  Epoch 200/200, and best_model.pth.
  WINDOWS_FILENAME_MAP.json maps 17 recovered historical submission-note
  filenames: colon characters in timestamps were replaced with hyphens.
  Original bytes were recovered from the supplied Git index and matched
  the original SHA256SUMS exactly. Only the new filenames differ.

kitti/
  Lab-origin KITTI detection source code, configuration/protocol records,
  compact full-validation results and extended-training convergence records.
  The original source/modelnet40 subfolder provides lab protocol/smoke work;
  completed HPC classification runs are in ../modelnet40/.

../thesis/scripts/ and ../thesis/evidence/
  Figure/table generators and local evidence used by the current complete
  manuscript. These supplement the two original research archives.

Scope and limitations
---------------------
This preserves available authored source and result evidence, not a complete
copy of all original training machines. Complete KITTI/ModelNet datasets,
most upstream repositories and pretrained upsampler weights, KITTI detector
checkpoints, full generated point-cloud collections and raw per-frame
predictions are not included. Some selected per-object evidence is present
under thesis/evidence. ModelNet final classifier checkpoints ARE included.
See each research subfolder's README and dependency/data notes.
The teacher's email explicitly requests source code, result files and README;
it does not explicitly require all raw data or every checkpoint.

Historical code retains machine-specific paths and external dependencies.
No full GPU experiment rerun was attempted in this audit. The archive must
not be described as one-command numerical reproduction.

Retain metric/split labels. ModelNet final training uses seed 42, selected
best checkpoint using the test set, and batch-averaged OA/mAcc. KITTI full
validation and selected-frame diagnostics are different evidence scopes.
