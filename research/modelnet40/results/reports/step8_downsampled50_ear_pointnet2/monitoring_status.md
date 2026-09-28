# Step 8 — Training Monitor Status Report

- Check time: 2026-06-22
- Experiment name: `downsampled50_ear_pointnet2`
- Job ID: **1711437**
- Job name: `p2_ear_b`
- Cluster: tinygpu

## 1. Job State

```
squeue.tinygpu -u $USER
```

| Fields | Value |
| --- | --- |
| JOBID | 1711437 |
| PARTITION | work |
| NAME | p2_ear_b |
| ST | **PD** (Pending) |
| TIME | 0:00 |
| REASON | Priority |

`sacct` Confirm: State = **PENDING**, not yet started (Start = Unknown).

**Processing:** No job cancelled and continues to await scheduling.

## 2. Log Check

Slurm Log Directory `logs/step8_downsampled50_ear_pointnet2/` Current is empty:

- `train_1711437.log` — **Not Generated**(job not started)
- `train_1711437.err` — **Not Generated**
- `train.log`(Python Training Log) **Not Generated**

Output Directory `outputs/step8_downsampled50_ear_pointnet2/` Current is empty.

## 3.  Log entries to be confirmed after training starts

Job Enter **R (Running)** After which, implement:

```bash
tail -n 120 logs/step8_downsampled50_ear_pointnet2/train_1711437.log
```

It is expected that:

| Checkpoint | Expected value |
| --- | --- |
| Data set path | `datasets/modelnet40_downsampled50_ear`(soft link) |
| train Samples | 9843 |
| test Samples | 2468 |
| batch shape | `(B, 1024, 3)` or model input `(B, 3, 1024)` |
| num_classes | 40 |
| Epoch 1 | Start training. |
| loss / accuracy | Normal Value Output |

## 4. Experimental Note (to be written in final_report)

This experiment is **Downsampled50 + EAR** The downstream PointNet++ classification experiment.

- **Do Not Force**Same point count as Downsampled50 baseline.
- Downsampled50 baseline: **512 points**(Previous)
- Downsampled50 + EAR: **1024 points**(EAR upsampling )
- Evaluation target: The actual enhancement of EAR upsampling process to PointNet++ classification end

## 5. Comparative Reference (baseline completed)

| Experiment | point count | best test acc | best epoch |
| --- | ---: | ---: | ---: |
| downsampled50_baseline | 512→1024 (loader resample) | 91.17% | 155 |
| downsampled50_ear_pointnet2 | 1024 (native) | **Pending completion of training** | — |

## 6. Follow-up

Upon completion of training, automatically generate:

- `reports/step8_downsampled50_ear_pointnet2/final_report.md`
- `reports/step8_downsampled50_ear_pointnet2/result_summary.csv`

If training fails (not restart job):

- `reports/step8_downsampled50_ear_pointnet2/error_report.md`

## 7. Surveillance Command

```bash
#  Queue Status
squeue.tinygpu -u $USER

# job  Real-time post-start log
tail -f logs/step8_downsampled50_ear_pointnet2/train_1711437.log

# Python  Training Log
tail -f logs/step8_downsampled50_ear_pointnet2/train.log
```
