# PU-GCN Line B Stall Diagnosis — Job 1731794

**诊断时间:** 2026-07-06 18:55 CEST (16:55 UTC)

---

## 1. 数据计数

| 指标 | 数量 |
|------|------|
| Line A strict (`strict_4N/pu_gcn`) | **12311 / 12311** (completed) |
| Line B raw (`raw/pu_gcn`) | **10010** |
| Line B strict (`strict_N/pu_gcn`) | **10010** |
| 期望输入 (`modelnet40_downsampled_x4`) | **12311** |
| Line B missing | **2301** |

**判断:** `raw == strict == 10010`，推理与 strict 写入同步停滞，**不是** raw>strict 的后处理问题。

---

## 2. Slurm 状态 (1731794)

| Task | State | Elapsed | ExitCode | Node | GPU |
|------|-------|---------|----------|------|-----|
| 0–12 | COMPLETED | ~2h31m–2h41m | 0:0 | tg066/tg068/tg069 等 | RTX **2080 Ti** |
| **1731794_13** | **RUNNING** | **21:39:01** | 0:0 | tg084 | RTX **3080** |
| **1731794_14** | **RUNNING** | **21:39:01** | 0:0 | tg084 | RTX **3080** |
| **1731794_15** | **RUNNING** | **21:39:01** | 0:0 | tg084 | RTX **3080** |

- 无 TIMEOUT / FAILED / CANCELLED / OUT_OF_MEMORY（sacct 记录）
- `#SBATCH --time=24:00:00` → 距 TIMEOUT 约 **2h21m**
- 三个剩余 task 均在同一节点 tg084

---

## 3. 最近文件写入

| 窗口 | strict 新文件 | raw 新文件 |
|------|--------------|-----------|
| 30 min | 0 | 0 |
| 60 min | 0 | 0 |
| 120 min | 0 | 0 |

**结论:** 虽然 Slurm 显示 RUNNING，但 **120 分钟内无任何有效产出**。

---

## 4. 日志路径

主要日志:
- `logs/pugcn_x4/lineB_pugcn_1731794_13.out` / `.err`
- `logs/pugcn_x4/lineB_pugcn_1731794_14.out` / `.err`
- `logs/pugcn_x4/lineB_pugcn_1731794_15.out` / `.err`

失败样本记录:
- `logs/pugcn_x4/lineB_chunk_13_failed.csv` (~442 failures)
- `logs/pugcn_x4/lineB_chunk_14_failed.csv` (~454 failures)
- `logs/pugcn_x4/lineB_chunk_15_failed.csv` (~458 failures)

完整日志列表: `reports/modelnet40_pugcn_lineB_log_files.txt`

---

## 5. 错误分析

### 现象
Tasks 13–15 日志持续打印:
```
[lineB] N/770 success=0 skipped=0 failed=N elapsed=...
```
即 **success 始终为 0，failed 线性增长**，约 170s/样本，但无 `.npy` 写出。

### 错误类型
全部失败为 `PU-GCN inference failed (rc=1)`，子进程 TensorFlow traceback 指向:
```
knn_point_2() → dil_knn() → densegcn() → batch_mat_mul_v2
```
首个失败样本示例:
- chunk 13: `train/sofa/sofa_0425`
- chunk 14: `train/table/table_0301`
- chunk 15: `train/tv_stand/tv_stand_0172`

### 关键关联
- Tasks **0–12**（RTX 2080 Ti）: 全部 COMPLETED，`success=770` 或 `success=765 skipped=5`
- Tasks **13–15**（RTX 3080, tg084）: **100% 推理失败**

强烈怀疑 **RTX 3080 上 PU-GCN tf_ops / CUDA 兼容性问题**，而非样本本身问题。

---

## 6. Missing Samples

- 列表: `reports/modelnet40_lineB_pugcn_missing_samples.txt`
- 总数: **2301**（≈ 770×3，对应 chunks 13/14/15）
- 分布: **全部在 train split**
  - train/vase: 475
  - train/table: 392
  - train/toilet: 344
  - train/tv_stand: 267
  - train/sofa: 256
  - …（详见 `reports/modelnet40_lineB_pugcn_missing_distribution.txt`）

---

## 7. 已完成输出 Quick Audit

- 检查: **10010** 个 strict `.npy`
- 失败: **0**
- 全部 PASS: shape `(1024, 3)`, 无 NaN/Inf
- 报告: `reports/modelnet40_lineB_pugcn_existing_quick_audit.csv`

**已完成的 10010 个输出可安全保留。**

---

## 8. 综合判断

| 维度 | 结论 |
|------|------|
| still progressing | ❌ 无新文件产出 |
| likely stalled but jobs still running | ✅ **是** — CPU 在跑但 100% 推理失败 |
| timeout expected soon | ✅ 已运行 21h39m / 24h limit，约 2h21m 后 TIMEOUT |
| failed | ⚠️ sacct 仍 RUNNING，但 functionally failed（零产出） |
| safe to resume missing samples | ✅ 已有 10010 个 PASS，可 resume 2301 missing |

**诊断结论:** **不应继续等待。** 当前三个 task 在空转重试失败样本，不会产出任何新 `.npy`。继续等待只会消耗配额直至 24h TIMEOUT。

---

## 9. 建议下一步

### 推荐方案（B + C 组合）

1. **不必主动 scancel**（本轮不执行）— 可等 ~2h 自然 TIMEOUT，或若急于释放资源可手动 `scancel 1731794_13 1731794_14 1731794_15`。
2. **保留** 已有 10010 个 raw/strict 输出（audit PASS）。
3. **创建 resume missing job**（下一轮，本轮不提交）:
   - 输入列表: `reports/modelnet40_lineB_pugcn_missing_samples.txt`（2301 样本）
   - 使用 `--resume` 跳过已有输出
   - **关键:** 限制到 RTX 2080 Ti 节点（或与成功 chunks 相同的 GPU 类型），避免再次调度到 tg084/RTX 3080
   - 考虑在 resume 前于 3080 上重编译 tf_ops 或添加 GPU 约束
4. **不要** 重跑 Line A 或已完成的 10010 个 Line B 输出。
5. **不要** 跑 post-process（raw == strict，无后处理 backlog）。

### 不适用
- **方案 D（raw > strict）:** 不适用，raw == strict。

---

## 10. 附属报告

| 文件 | 内容 |
|------|------|
| `reports/modelnet40_pugcn_lineB_1731794_slurm_status.txt` | Slurm 状态 |
| `reports/modelnet40_pugcn_lineB_count_status.txt` | 计数 |
| `reports/modelnet40_pugcn_lineB_recent_write_status.txt` | 最近写入 |
| `reports/modelnet40_pugcn_lineB_error_scan.txt` | 错误扫描 |
| `reports/modelnet40_lineB_pugcn_missing_samples.txt` | 缺失列表 |
| `reports/modelnet40_lineB_pugcn_missing_distribution.txt` | 缺失分布 |
| `reports/modelnet40_lineB_pugcn_existing_quick_audit.csv` | 已有输出审计 |
