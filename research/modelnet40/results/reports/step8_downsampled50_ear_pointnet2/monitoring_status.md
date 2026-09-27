# Step 8 — 训练监控状态报告

- 检查时间: 2026-06-22
- 实验名: `downsampled50_ear_pointnet2`
- Job ID: **1711437**
- Job name: `p2_ear_b`
- 集群: tinygpu

## 1. Job 状态

```
squeue.tinygpu -u $USER
```

| 字段 | 值 |
| --- | --- |
| JOBID | 1711437 |
| PARTITION | work |
| NAME | p2_ear_b |
| ST | **PD** (Pending) |
| TIME | 0:00 |
| REASON | Priority |

`sacct` 确认: State = **PENDING**，尚未开始（Start = Unknown）。

**处理:** 未取消 job，继续等待调度。

## 2. 日志检查

Slurm 日志目录 `logs/step8_downsampled50_ear_pointnet2/` 当前为空：

- `train_1711437.log` — **尚未生成**（job 未启动）
- `train_1711437.err` — **尚未生成**
- `train.log`（Python 训练日志）— **尚未生成**

输出目录 `outputs/step8_downsampled50_ear_pointnet2/` 当前为空。

## 3. 训练启动后需确认的日志项

Job 进入 **R (Running)** 后，执行：

```bash
tail -n 120 logs/step8_downsampled50_ear_pointnet2/train_1711437.log
```

预期应出现：

| 检查项 | 预期值 |
| --- | --- |
| 数据集路径 | `datasets/modelnet40_downsampled50_ear`（软链接） |
| train 样本数 | 9843 |
| test 样本数 | 2468 |
| batch shape | `(B, 1024, 3)` 或模型输入 `(B, 3, 1024)` |
| num_classes | 40 |
| Epoch 1 | 开始训练 |
| loss / accuracy | 正常数值输出 |

## 4. 实验说明（待写入 final_report）

本实验为 **Downsampled50 + EAR** 的 downstream PointNet++ 分类实验。

- **不强制**与 Downsampled50 baseline 保持相同点数。
- Downsampled50 baseline: **512 points**（原生）
- Downsampled50 + EAR: **1024 points**（EAR 上采样后）
- 评估目标: EAR 上采样流程对 PointNet++ 分类端的实际提升效果

## 5. 对比参考（已完成 baseline）

| 实验 | 点数 | best test acc | best epoch |
| --- | ---: | ---: | ---: |
| downsampled50_baseline | 512→1024 (loader resample) | 91.17% | 155 |
| downsampled50_ear_pointnet2 | 1024 (native) | **待训练完成** | — |

## 6. 后续操作

训练完成后自动生成：

- `reports/step8_downsampled50_ear_pointnet2/final_report.md`
- `reports/step8_downsampled50_ear_pointnet2/result_summary.csv`

若训练报错（不重启 job）：

- `reports/step8_downsampled50_ear_pointnet2/error_report.md`

## 7. 监控命令

```bash
# 队列状态
squeue.tinygpu -u $USER

# job 开始后实时日志
tail -f logs/step8_downsampled50_ear_pointnet2/train_1711437.log

# Python 训练日志
tail -f logs/step8_downsampled50_ear_pointnet2/train.log
```
