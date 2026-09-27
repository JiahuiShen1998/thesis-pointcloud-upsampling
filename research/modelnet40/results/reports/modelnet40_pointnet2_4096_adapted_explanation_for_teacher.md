# Clarification: PointNet++ Adaptation to 4096-Point Input (Line A)

- Generated: `2026-07-23 15:53:35 UTC`
- Based on audit: `reports/modelnet40_pointnet2_lineA_4096_retraining_audit_for_teacher.md`

## Short answer

The Line A upsampled branches were **not** evaluated with a fixed 1024-point PointNet++ model.
Each 4096-point branch used a PointNet++ classifier that was **retrained from scratch with `num_point=4096`**.

## Details for the teacher

1. **Line A upsampled branches were not evaluated using a fixed 1024-point PointNet++ model.**
   There is no eval-only transfer of the Original-baseline 1024 checkpoint onto 4096 inputs.

2. **Each 4096-point branch was trained and evaluated with `num_point=4096`.**
   Confirmed in configs, Slurm launch flags, `metrics.json`, and logged dataloader batch shape `(B, 3, 4096)` = `(24, 3, 4096)`.

3. **Therefore the classifier was retrained for the 4096-point input setting.**
   Training script initializes a new `pointnet2_cls_ssg` model and does not load any 1024-point pretrained weights.
   Covered methods: EAR, PDANS, PU-Net, PU-GCN, PU-EdgeFormer (each independent full run, 200 epochs).

4. **The observed accuracy drop vs Original baseline 1024 is not caused by directly applying a 1024-trained classifier to 4096-point inputs.**
   Any drop should be interpreted under a matched protocol where the classifier itself is adapted to 4096 points.

5. **If required for reproducibility packaging, an additional clean rerun can still be performed** under a new experiment name
   `pointnet2_lineA_4096_retrain_clean`, without overwriting existing results.
   Based on this audit, such a rerun is **not necessary** to answer the fairness concern.

## Recommended table wording

| Line | Branch | Classifier setting |
| --- | --- | --- |
| A | Original baseline 1024 | PointNet++ trained with 1024-point input |
| A | Original + upsampling 4096 (EAR / PDANS / PU-Net / PU-GCN / PU-EdgeFormer) | **PointNet++ retrained with 4096-point input** |

Avoid wording such as: “PointNet++ trained only for 1024 points”.

