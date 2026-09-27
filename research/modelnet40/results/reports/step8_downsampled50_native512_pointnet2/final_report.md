# Downsampled50 Native512 PointNet++ — Final Report

- Generated at: 2026-06-26
- Experiment: **downsampled50_native512_pointnet2**
- Job ID: **1716131** (`p2_ds512`)
- Slurm state: **COMPLETED** (exit 0)
- Role in main protocol: **Line B baseline** (512 pts in → 512 pts to PointNet++)

---

## Training results

| Metric | Value |
| --- | --- |
| **Best test accuracy (overall)** | **91.26%** (epoch 197) |
| Best test accuracy (class-mean) | 88.00% (epoch 197) |
| **Final test accuracy (overall)** | **91.14%** (epoch 200) |
| Final test accuracy (class-mean) | 87.07% (epoch 200) |
| Best epoch | **197** |
| Final epoch | **200** / 200 |
| Total training time | **04:50:01** (wall clock) |
| GPU node | tg080 (RTX 3080) |
| Job window | 2026-06-26 11:16:57 → 16:06:58 CEST |

Training curve behavior: **normal**. Train accuracy rises from ~41% (epoch 1) to ~96% (epoch 200); test overall accuracy improves steadily to ~91% with no collapse or NaN. Best checkpoint saved at epoch 197; epochs 198–200 remain within 0.1% of best.

---

## Protocol verification

| Setting | Configured | Verified |
| --- | --- | --- |
| Dataset branch | `downsampled50_baseline` | `data_root=datasets/modelnet40_downsampled50` |
| Disk / input point count | 512 | native 512 in `.npy` files |
| `num_point` | 512 | `num_point=512` in args |
| `allow_resample` | false | `allow_resample=False` |
| Loader resample 512→1024 | **no** | no padding / upsampling in loader |
| First batch sample shape | — | **`(512, 3)`** (log line) |
| PointNet++ input | B×512×3 → B×3×512 | standard transpose in training script |
| Train / test count | 9843 / 2468 | matches dataset manifests |

**Protocol status: PASS**

### Ablation distinction

- **This experiment** = main protocol **Line B baseline**: true native 512-point input.
- **`downsampled50_baseline` (loader 512→1024, `num_point=1024`)** = legacy **naive-resampled ablation only**; not the main Line B baseline under the ×4 corrected protocol.

---

## Artifacts

| Type | Path |
| --- | --- |
| Config | `configs/downsampled50_native512_pointnet2.yaml` |
| Slurm job | `jobs/train_downsampled50_native512_pointnet2_tinygpu.sbatch` |
| Primary log | `logs/downsampled50_native512_pointnet2/train_1716131.log` |
| Symlink log | `logs/downsampled50_native512_pointnet2/train.log` |
| Stderr | `logs/downsampled50_native512_pointnet2/train_1716131.err` |
| Checkpoint (best) | `outputs/downsampled50_native512_pointnet2/checkpoints/best_model.pth` (~17.0 MB) |
| Metrics JSON | `outputs/downsampled50_native512_pointnet2/metrics.json` |
| Per-class CSV | `reports/downsampled50_native512_pointnet2/downsampled50_native512_pointnet2_result.csv` |
| Summary CSV (this step) | `reports/step8_downsampled50_native512_pointnet2/result_summary.csv` |

---

## Per-class notes (weakest classes at best epoch)

| Class | Accuracy |
| --- | ---: |
| flower_pot | 11.1% |
| cup | 55.0% |
| radio | 70.0% |
| wardrobe | 70.0% |
| night_stand | 72.6% |

Strong classes (airplane, curtain, keyboard, laptop, person): 100%.

---

## Comparison context (Line B)

| Branch | Input pts | PointNet++ pts | Status |
| --- | ---: | ---: | --- |
| **native512 baseline** (this) | 512 | 512 | **DONE — 91.26%** |
| Upsampling ×4 (EAR/PU-Net/PU-GCN/PDANS) | 512 | 2048 | generation pending |
| Legacy `downsampled50_baseline` loader 512→1024 | 512 disk | 1024 loader | ablation only |

PointNet++ training on upsampled Line B variants remains **blocked** until respective full-generation audit PASS and explicit user confirmation.
