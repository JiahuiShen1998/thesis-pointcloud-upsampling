# ModelNet40 实验配置、代码使用、修改、参数指标、比较与结果完整工作记录

- 整理日期：2026-09-13（Europe/Berlin）。
- 实验仓库（下文简称 **E**）：`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`。
- 论文与交付目录（下文简称 **D**）：`/home/hpc/iwnt/iwnt189h`。
- 证据范围：实验仓库的脚本、YAML、Slurm 脚本和日志、结果 `metrics.json`、CSV/Markdown 审计、现有论文数据表，以及截至本次检查的 Git 状态。不把计划、未运行的方案或其他数据集的检测结果写成已完成的 ModelNet40 结果。
- 数值口径：分类准确率统一写为百分数，差值为**百分点（pp）**；“Best OA”是 200 个 epoch 中以 **test 集**指标选出的最高值，“Final OA”是第 200 个 epoch 的值。几何 CD/HD 越小越好，NUC 为本项目的局部密度变异系数，越小表示按该定义更均匀。

## 1. 总体完成状态与核心结论

项目完成了 ModelNet40 点云数据构建、EAR/PDANS/PU-Net/PU-GCN/PU-EdgeFormer 五种 ×4 上采样方法的接入和审计、两条主实验线的 PointNet++ SSG 从头训练、同点数几何比较、mesh-reference 控制、失败恢复、机制探针、逐类分析及论文图表。共找到 **22 次完成的正式 PointNet++ 训练**（不计 smoke），其中最终协议主实验为 **12 个分支**（Line A/B 各 1 个基线和 5 个方法），另有 **2 个 mesh-ref 控制分支**。本次直接逐一检查了最终 14 份 YAML 所指向的 `metrics.json`、`train.log` 和 `best_model.pth`：14/14 均存在，日志均有 `Epoch 200/200`，配置均为 `allow_resample: false`。

最重要的结果：Line A 从原始 1,024 点增密到 4,096 点，五种方法的分类 Best OA **均低于** 1,024 点原始基线 91.95%；最接近的 PU-GCN 为 91.63%（−0.32 pp）。Line B 从 1,024 点降到 256 点再恢复到 1,024 点，**只有 PU-Net** 超过 256 点基线：91.27% 对 90.85%，差 +0.42 pp，但仍低于原始 1,024 点的 91.95%。几何 CD 的最佳者在两线均为 PU-GCN；分类最佳者并不总是几何最佳者。

截至 2026-09-13，本账号的 `squeue` 无待运行/运行中的任务；实验仓库最近可见提交为 `daacfaf`（2026-07-27），工作树仍有 **20 个已跟踪文件修改、59 个未跟踪顶层条目**。这些未提交内容包括 mesh-ref 控制、equal-N 评估及文稿/图表，交付时应保留当前文件状态或另行版本化。

## 2. 研究问题与最终实验设计

研究的问题是：上采样生成的额外点能否改善 ModelNet40 物体分类，以及几何质量与分类效果是否同向变化。输入均为 XYZ，不使用 normals。40 类；训练集 9,843、测试集 2,468，共 12,311 个形状。每条线的上采样器输入、输出与分类器输入如下：

| 线别 | 分类基线 | 上采样方法的输入 → 输出 | 方法分支分类器 | 主比较 |
|---|---|---|---|---|
| A：原始点云增密 | Original 1,024 | Original 1,024 → ×4 → 4,096 | 每种方法独立以 4,096 点训练 | 各方法 vs Original 1,024 |
| B：稀疏输入恢复 | Original 1,024 经随机无放回降采样到 256 | 256 → ×4 → 1,024 | 每种方法独立以 1,024 点训练 | 各方法 vs 原生 256 点基线；Original 1,024 为次级上限参照 |

公平性规则是保留每条分支自己的实际点数，训练 DataLoader 不偷偷裁剪或复制点数。因此 Line A 的 4,096 点分类器确实按 4,096 点**重新从随机初始化训练**，Line B 的 256 点基线也确实按 256 点训练；没有拿 1,024 点 checkpoint 直接评价不同点数。Line A 的训练真实性另有逐分支审计：配置、sbatch、首 batch `(24,3,4096)`、首 loss、200 epoch、独立 checkpoint 全通过。

早期协议曾有 512→1,024 的 EAR ×2 和 512→2,048 的 ×4 Line B；最终主协议改为 **256→1,024**。旧数据与日志保留供消融/协议敏感性分析，不能混进最终 Line B 主表。TULIP 只作 supplementary 标记、SPU-PMD 未进入 ModelNet40 主协议；两者没有上述主实验的完成结果。

## 3. 数据构建与输入处理

1. **原始数据**：读取 ModelNet40 的 `.off` 网格；存在面片时按三角形面积加权采样表面点，非三角面会扇形三角化；每形状采 1,024 点，中心化并按最大半径缩放到单位球，保存 `(1024,3)` float32 `.npy`，同时写 class map 与 train/test manifest。2026-06-20 预处理报告记录用 8 workers、153.7 秒、12,311/12,311 成功、0 失败；完整检查计数、形状、NaN/Inf、单位球均 PASS。注意 YAML 模板写 `num_workers: 16`，但**实际执行报告**记载 8，复现历史运行应采用实际报告口径。
2. **256 点 Line B 基线**：从已生成的 Original 1,024 点按固定样本种子随机无放回选 256 点，而不是从 `.off` 重新采样。`build_modelnet40_x4_downsampled.py` 默认 `--method random`，也支持 FPS；最终审计中的数据为随机降采样。基线与五个恢复方法共享这一稀疏输入来源。
3. **上采样输出**：按 `train/test/class/shape_id.npy` 保存；EAR/PDANS/PU-Net/PU-GCN 的生成流程将原始输出存 `raw`，整理后的精确点数输出存 `strict_4N`（A）或 `strict_N`（B）；PU-EdgeFormer 使用其独立的输出目录，再通过链接接入分类入口。每方法两线均有完整 9,843 train + 2,468 test、共 12,311 个可访问结果；PU-EdgeFormer 核数时须沿链接遍历。严格输出经形状与有限值审计后才成为分类器输入。此处“strict”保证数值与点数符合协议，不意味着不同上采样器具有同一生成机制。
4. **mesh-ref 控制**：2026-08-03 从每个 `.off` 独立面积加权重采样 256 和 4,096 点，各 12,311 个形状、构建审计 0 失败。它们并非 Original 1,024 的简单重复/降采样，用于同点数几何参考及额外分类控制。
5. **metadata 修正**：方法目录必须有 `class_to_idx.json` 与 `idx_to_class.json`。曾把整个基线 metadata 目录链接进去，导致 manifest 指回 256 点文件；最终 v2 仅链接两个 class map，让 DataLoader 按方法目录扫描各自 `.npy`。该修复避免了“名义上方法分支、实际读取基线输入”的错误。

现有 `prepare_modelnet40.py` 在按文件构造采样种子时使用 Python 内建 `hash((class_name, split, shape_id))`。它受进程的 hash 随机化影响；因此“相同 seed=42 的训练 checkpoint bit-identical”只证明**固定磁盘数据上的训练流程**可重复，不能直接推断重新从 `.off` 生成的 1,024 点文件在不同 Python 进程中逐点相同。若需要端到端重建，应固定 `PYTHONHASHSEED` 或把此处改成稳定哈希后重新审计；本报告没有改动历史数据。

## 4. 模型、训练参数与资源配置

| 项目 | 最终实际配置 |
|---|---|
| 分类网络 | `pointnet2_cls_ssg`，40 类，`normal_channel=False`，仅 XYZ |
| 网络 SA1 | FPS 512 中心、半径 0.2、每邻域至多 32 点、MLP `[64,64,128]` |
| 网络 SA2 | 128 中心、半径 0.4、至多 64 点、MLP `[128,128,256]`；SA3 全局聚合 |
| 分类头 | 1024→512→256→40，两个 dropout 均 0.4，负对数似然损失 |
| 训练轮数/批量 | 200 epochs、batch size 24；训练 loader `shuffle=True`、`drop_last=True` |
| 优化器 | Adam，初始 LR 0.001，weight decay 0.0001，betas `(0.9,0.999)`，eps `1e-8` |
| 学习率调度 | `StepLR(step_size=20, gamma=0.7)`；代码在每个 epoch 开头调用 `scheduler.step()` |
| 随机性 | `seed=42`，设置 Python/NumPy/PyTorch/CUDA 种子；训练增强为随机点 dropout、随机缩放、平移 |
| 数据读取 | `num_workers=4`，`--no-allow-resample`；每样本再次中心化并缩放到单位球 |
| 最终分支点数 | A 基线 1024、A 方法 4096、B 基线 256、B 方法 1024、mesh-ref 256/4096 |
| Slurm 例子 | 主训练脚本 `partition=work`、1 GPU、8 CPU、24 小时；mesh-ref 控制在 `v100` 分区 1 张 V100、8 CPU、24 小时 |

训练脚本保存各分支的 `train.log`、`metrics.json`、`checkpoints/best_model.pth` 以及模型源文件副本。每 epoch 在 test 集计算整体与类别准确率；当 test 整体准确率 `>=` 历史最佳时覆盖 best checkpoint，平手选较晚 epoch。`Best Class` 表示 **Best OA 所在 epoch 的类别指标**，不是 200 epoch 中类别指标的单独最大值。

**指标实现口径需特别说明**：`train_pointnet2.py::evaluate()` 的 OA 是各 test batch 正确率的**简单平均**，不是把所有 2,468 个样本的正确数相加再除以 2,468；最后不足 24 的 batch 与完整 batch 权重相同。类别指标也先按 batch 内各类正确率累计再平均。因此报告保留代码输出的“OA/mAcc”标签，但若与论文或其他实现对比，应按此实现解释，不能无核验地当作精确样本加权 OA/mAcc。

### 五个上采样器的实际接入参数

| 方法 | 调用与权重 | 关键参数/约束 |
|---|---|---|
| EAR | 外部 `EAR/ear_upsampling.py`，包装器 `ear_modelnet40_utils.py` | ×4；`k_neighbors=20`、`edge_sensitivity=4.0`、`up_threshold=0.93`、`sigma_p=0`、`max_iter=5`；XYZ 补零 intensity 调用后只取 XYZ |
| PDANS | `PDANS/checkpoints/PU1K_PDANS.pkl`、`pointnet2/exp_configs/PU1K.json` | CUDA 推理；`R=4`、DDIM `step=30`、`gamma=0.5`；每形状独立调用并校验 finite |
| PU-Net | 官方/迁移生成器 `PU-Net/model/generator2_new6` | TF1 环境，`--phase test --up_ratio 4 --num_point <实际输入点数>`；调用真实 checkpoint 生成器 |
| PU-GCN | `PU-GCN/pretrained/pu1k-pugcn` | TF1 环境，`--model pugcn --upsampler nodeshuffle --up_ratio 4 --num_point <实际输入点数> --k 20 --seed 42` |
| PU-EdgeFormer | `external/pu_edgeformer_ops_reuse/checkpoint_model100`，模型 `edgetransformer` | Line B 直接 256→1024；Line A 使用 A1 直接 1024→4096；推理 seed `20260702`、倍率 4；输出不足目标点数报错，不通过填充/插值伪造点 |

PU-EdgeFormer 另做了 checkpoint 变量、模型调用路径、输出新鲜度、重复/复制、与 PU-GCN 输出差异等真实性审计；其两线各 12,311 个输出均为 float32、精确点数、无 NaN/Inf、无 raw shortage。各方法适配的是现有预训练上采样器，**没有证据表明在 ModelNet40 上重新训练了这些上采样器**；从头训练指的是下游 PointNet++ 分类器。

## 5. 代码入口与使用方式

关键源码与作用（路径均相对 E）：

| 阶段 | 文件 | 实际用途 |
|---|---|---|
| 原始预处理 | `scripts/prepare_modelnet40.py`、`scripts/check_modelnet40_original.py` | OFF→NPY、manifest、计数/形状/单位球检查 |
| ×4 降采样 | `scripts/build_modelnet40_x4_downsampled.py` | Original 1024→256，构建 Line B 稀疏输入 |
| 协议常量 | `scripts/modelnet40_x4_protocol.py` | 两线点数、路径、方法别名与稳定种子 |
| 上采样生成 | `scripts/run_lineB_downsampled_x4_upsampling_chunk.py`、`scripts/run_pugcn_modelnet40_x4.py`、`scripts/run_pu_edgeformer_modelnet40_labparams_full.py` | 分块/独立方法推理，保存 raw 与 strict 输出 |
| 方法包装 | `scripts/{ear,pdans,punet,pugcn}_modelnet40_utils.py` | 外部模型与 checkpoint 的实际调用路径 |
| 数据接入 | `scripts/prepare_pointnet2_inputs_x4_two_lines.py`、`scripts/prepare_pu_edgeformer_pointnet2_smoke.py` | 分支输入目录与 class metadata 接入 |
| 分类器 | `scripts/modelnet_npy_dataloader.py`、`scripts/train_pointnet2.py`、`scripts/eval_pointnet2.py` | NPY 读取、训练、评价与 checkpoint 保存 |
| 正式作业 | `jobs/pointnet2_full/run_*.sbatch`；配置在 `configs/pointnet2_x4_two_line/*.yaml` | 分支专用 Slurm 命令与参数；YAML 是对照配置，sbatch 显式传给 Python |
| 几何与控制 | `scripts/build_mesh_reference_points.py`、`scripts/compute_geometry_equal_n.py`、`scripts/compute_cardinality_bias_probe.py` | 同点数参考、CD/HD/NUC 与重复点偏差验证 |
| 分析和展示 | `scripts/generate_thesis_analysis.py`、`scripts/generate_final_visualization_package.py` 等 | 总表、逐类、几何—分类关系、点云展示与报告 |

**复现入口示例（列出历史脚本用法，本次没有重新提交训练或覆盖数据）：**

```bash
cd /home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling
python scripts/prepare_modelnet40.py --seed 42 --num-points 1024 --num-workers 8
python scripts/check_modelnet40_original.py
python scripts/build_modelnet40_x4_downsampled.py --method random --workers 8
# 已准备好各分支输入、环境和依赖后，可分别提交正式训练：
sbatch jobs/pointnet2_full/run_lineA_pugcn_4096.sbatch
sbatch jobs/pointnet2_full/run_lineB_punet_1024.sbatch
# 几何控制及评估：
python scripts/build_mesh_reference_points.py --num-points 256 4096 --workers 16
python scripts/compute_geometry_equal_n.py --splits test --workers 16
```

上面单独的 `sbatch` 只是两个**示例**，不能替代其余 12 个分支的作业；最终 14 个分支应以对应 YAML、sbatch、`metrics.json` 和审计表逐一核对。需要查看已有结果而不重跑时，首选 E 下的 `reports/modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.csv`、`reports/modelnet40_geometry_equal_n_summary.csv` 和 `pointnet2_results/x4_two_line_final/`。

## 6. 按时间的实验与修改记录

| 时间 | 实际工作与结果 | 关键返工或解释 |
|---|---|---|
| 6 月 20 日 | OFF→Original 1024 数据建好；40 类、9,843/2,468、0 失败；PointNet++ NPY DataLoader 和训练包装器接入；smoke 1709664 PASS，Original 正式训练 1709666 开始；另建 512 点早期数据及 loader 重采样基线 1710180。 | 后来发现“磁盘 512、loader 补到 1024”不是原生 512 点基线，改列早期消融。 |
| 6 月 21–22 日 | EAR ×2 512→1024 全量 12,311 完成；分类 job 1711437 得 90.82%；原生 512 点分类 job 1716131 完成。 | EAR ×2 随主倍率调整退出最终主表。 |
| 6 月 25 日 | 纠正“分类基线与上采样输出必须同点数”的错误公平性假设，主倍率改为 ×4。 | 初版 Line B 仍是 512→2048，随后再次调整。 |
| 6 月 26 日–7 月 1 日 | EAR/PDANS 早期 ×4 两线生成与分类；PU-Net/PU-GCN TF custom ops 修复，PDANS GPU 环境修复，PU-Net 官方 generator smoke PASS。 | metadata symlink 竞态、方法名多一个 `}`、CUDA/PyTorch3D ABI、`open3d` 缺失、TF 编译链接错误均有失败日志和修复记录。 |
| 7 月 2–6 日 | 最终固定 Line B 为 256→1024；EAR/PDANS 全量重建；PU-Net 经 missing-only 恢复到 12,311；PU-GCN A 全量完成，B 因 GPU 兼容问题先停在 10,010，再恢复到 12,311。 | 旧 512→2048 结果保留但退出主协议；PU-Net strict 目录创建修复；PU-GCN RTX 3080 推理错误，恢复转到 RTX 2080 Ti。 |
| 7 月 7–8 日 | metadata v2 修复后 A/B 方法 smoke 均通过；Line B jobs 1733191–1733195、Line A jobs 1733491–1733495 全部完成 200 epochs。 | v1 metadata 整目录链接使 manifest 指向 256 点基线，v2 只接 class map。 |
| 7 月 12–17 日 | 分类、几何、逐类、图表与互动点云汇总；接入 PU-EdgeFormer，真实性审计和两线输出通过，正式分类 jobs 1749441/1749442 完成。 | 最终五方法齐备；后续几何评价因参考集偏差还要更正。 |
| 7 月 23 日 | 复核五个 A 线 4,096 点分支：首 batch、首 loss、epoch 200、metrics 和独立 checkpoint 全部吻合。 | 审计结论：无需额外 clean rerun。 |
| 8 月 3–18 日 | mesh-ref 256/4096 建立；equal-N 几何评价 2,468 test 形状、34,552 条对象-条件记录、0 失败；控制训练 jobs 1768767/1768768 完成；重复点偏差、SA1 接收探针、同 seed 重复训练审计；制作 dossier、PPT/PDF。 | 旧 4096 vs 1024 等非等点数 CD/HD 排名退出主结论。 |
| 9 月 4–8 日 | 论文第 3–4 章 PDF 27 页；第 5–6 章最终 PDF 54 页，引用 28 张唯一 ModelNet40 图；汇入 14 分支×200=2,800 条训练曲线、600 条逐类记录及几何对象记录。 | 第 5–6 章最终审计无缺图、未定义引用、overfull 和 LaTeX error。 |
| 9 月 10–11 日 | 先形成全实验/修改汇总记录；随后制作统一学术图件包，ModelNet40 28 张、KITTI 11 张，共 39 张，每张 PDF/SVG/300 dpi PNG，最终 QA issues=0。 | 统一图件属于展示交付，不代表重新训练或新增 ModelNet40 分类结果。 |

### 主要失败与具体修复链

- **EAR metadata 并发竞态**：array 1716175 的两个 task 因链接文件并发 `FileNotFoundError` 失败；`ensure_metadata_link()` 对缺失异常容错并只重跑缺块，最终两线 full PASS。
- **提交脚本拼写错误**：1716440–1716451 一批方法名出现 `pdans}`/`punet}`/`pugcn}`，形成错误目录；修正方法展开后重提交流程，保留错误目录作证据。
- **方法依赖**：1717035/37 PDANS 无可用 CUDA；1717039/41 与 1717043/45 的 PU-Net/PU-GCN TF ops 因 `nvcc`、`libcudart`、`libtensorflow_framework` 等链接问题失败。调整模块/环境与 GPU 节点编译后通过；PDANS 另经历 1717964 PyTorch3D ABI 不匹配、1722422 缺 `open3d`，1723850 GPU validate PASS。
- **PU-Net 输出目录与长作业**：1727073 推理成功但 strict 保存父目录未建，40/40 写入失败；加 `strict_out.parent.mkdir(parents=True, exist_ok=True)` 后 1729225 smoke PASS。full array 1729235 有 11 个 24h TIMEOUT、1 个 TF `rc=-6`，已审计的 9,150 文件保留；1731611 missing-only 恢复至 raw/strict 各 12,311，均 `(1024,3)`、无非有限值。
- **PU-GCN Line B GPU 兼容**：1731794 在 10,010/12,311 停滞；RTX 3080 tasks 13–15 的 `knn_point_2` 推理 100% 失败且 120 分钟无新文件。取消空转任务后第一次恢复 1732373 因 Python `global` 位置引起 SyntaxError，改局部变量传参并 `py_compile`/smoke 1732385 PASS；1732386 限制 RTX 2080 Ti 完成余量，最终 raw/strict 各 12,311、missing 0。
- **分类 metadata**：Line B 首批 1733160–1733163、v1 1733167–1733170、Line A 首批 1733476–1733479 因 label map/manifest 问题失败；v2 后 B 1733171–1733174、A 1733487–1733490 全通过。
- **几何参考偏差**：旧表将 4,096 点输出直接对 1,024 点输入算 CD/HD，重复输入的假上采样 `repeat_x4` 可得 CD=HD=0。加入 mesh-ref 4,096 与 equal-N 后，其 CD=0.055734、HD=0.117819，在六种对象（五方法+重复控制）中最差；因此重做几何排序、图表与论文解释。

## 7. 最终分类结果：主协议 12 分支 + 2 控制

Best OA 为 test 选择的 checkpoint；表内 `mAcc@Best` 是该 checkpoint 的类别指标。每行都是独立的 200 epoch 训练。百分数按报告四舍五入至两位。

| 线 | 方法/对照 | 输入点数 | Best OA | Final OA | mAcc@Best | 最佳 epoch | 相对本线主基线 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Original 基线 | 1024 | **91.95%** | 91.31% | 88.00% | 85 | 0.00 pp |
| A | EAR | 4096 | 91.48% | 90.85% | 88.19% | 61 | −0.47 pp |
| A | PDANS | 4096 | 91.62% | 90.87% | 88.44% | 84 | −0.33 pp |
| A | PU-Net | 4096 | 90.95% | 90.55% | 87.46% | 189 | −1.00 pp |
| A | PU-GCN | 4096 | **91.63%** | 91.00% | 87.74% | 81 | −0.32 pp |
| A | PU-EdgeFormer | 4096 | 90.54% | 90.13% | 86.35% | 83 | −1.41 pp |
| B | Downsampled ×4 基线 | 256 | 90.85% | 90.46% | 87.37% | 124 | 0.00 pp |
| B | EAR | 1024 | 88.81% | 88.02% | 84.50% | 116 | −2.04 pp |
| B | PDANS | 1024 | 90.34% | 89.46% | 86.20% | 145 | −0.51 pp |
| B | PU-Net | 1024 | **91.27%** | 90.65% | 87.86% | 148 | **+0.42 pp** |
| B | PU-GCN | 1024 | 90.06% | 89.63% | 86.41% | 164 | −0.79 pp |
| B | PU-EdgeFormer | 1024 | 89.32% | 88.58% | 85.21% | 194 | −1.53 pp |
| 控制 | mesh-ref 256 | 256 | 90.88% | 90.38% | 86.34% | 129 | vs B 稀疏基线 +0.03 pp |
| 控制 | mesh-ref 4096 | 4096 | 92.00% | 91.18% | 88.70% | 148 | vs Original 1024 +0.05 pp |

B 线 PU-Net 虽提高 0.42 pp，离 Original 1,024 仍差 **0.68 pp**。其余 B 线方法分别比 Original 低 EAR 3.14、PDANS 1.61、PU-GCN 1.89、PU-EdgeFormer 2.63 pp。A 线“点数增至四倍”本身没有带来可见的 Best OA 优势；mesh-ref 4,096 只比 Original 1,024 高 0.05 pp，也说明在此 SSG 设定与数据中，纯粹更密的真实表面采样收益很小。

逐类结果提供补充而非替代主指标。以 B 线为例，PU-Net 相对 256 基线在 40 类中 15 类提高、15 类不变、10 类下降，40 类差值均值约 +0.49 pp；EAR 为 8 类提高、8 类不变、24 类下降，均值约 −2.86 pp。单类别测试样本可只有 20 个，这些差值波动较大，不宜由个别类别推断总体稳健性。

## 8. 早期和重复训练：全部 22 次正式训练清单

下表包含上节 14 个最终/控制训练及 8 个早期/重复训练。`early` 不是虚构或失败结果，但协议已被替代；重复结果不得当作独立 seed 证据。

| 点数 | 运行/数据来源 | Best OA | Final OA | 最佳 epoch | 地位 |
|---:|---|---:|---:|---:|---|
| 256 | mesh-ref 256 | 90.88% | 90.38% | 129 | final control |
| 256 | B：Original 降采样 ×4 | 90.85% | 90.46% | 124 | final |
| 512 | 原生 Downsampled50 | 91.26% | 91.14% | 197 | early |
| 1024 | A：Original 基线 | 91.95% | 91.31% | 85 | final |
| 1024 | 早期 Original 同数据重复 | 91.95% | 未记录 | 85 | early repeat |
| 1024 | B：PU-Net | 91.27% | 90.65% | 148 | final |
| 1024 | 磁盘 512、loader 重采样为 1024 | 91.17% | 未记录 | 155 | early ablation |
| 1024 | 512→EAR ×2 | 90.82% | 未记录 | 111 | early ablation |
| 1024 | B：PDANS | 90.34% | 89.46% | 145 | final |
| 1024 | B：PU-GCN | 90.06% | 89.63% | 164 | final |
| 1024 | B：PU-EdgeFormer | 89.32% | 88.58% | 194 | final |
| 1024 | B：EAR | 88.81% | 88.02% | 116 | final |
| 2048 | 512→PDANS ×4 | 91.47% | 90.99% | 164 | early |
| 2048 | 512→EAR ×4 | 90.61% | 89.76% | 74 | early |
| 4096 | mesh-ref 4096 | 92.00% | 91.18% | 148 | final control |
| 4096 | A：PU-GCN | 91.63% | 91.00% | 81 | final |
| 4096 | A：PDANS | 91.62% | 90.87% | 84 | final |
| 4096 | 早期 Original→PDANS ×4 同数据重复 | 91.62% | 90.87% | 84 | early repeat |
| 4096 | A：EAR | 91.48% | 90.85% | 61 | final |
| 4096 | 早期 Original→EAR ×4 同数据重复 | 91.48% | 90.85% | 61 | early repeat |
| 4096 | A：PU-Net | 90.95% | 90.55% | 189 | final |
| 4096 | A：PU-EdgeFormer | 90.54% | 90.13% | 83 | final |

**协议敏感性例子**：PDANS 旧 512→2048 的 91.47% 相对原生 512 基线 91.26% 为 **+0.21 pp**，最终 256→1024 的 90.34% 相对原生 256 基线 90.85% 却为 **−0.51 pp**。这说明结论依赖稀疏化程度，旧表不能替代最终主表。Original 1024、A-EAR 4096、A-PDANS 4096 各自与相同 seed 的早期独立 Slurm job 重复训练比较，79/79 个权重张量逐值相同、最大差 0；这是固定数据与 seed 下的 bit-determinism，**不是多 seed 的稳定性**。

## 9. 几何指标、同点数结果与排序纠偏

最终几何协议只在相同点数的集合间比较：A 线各 4,096 点方法 vs mesh-ref 4,096；B 线稀疏 256 点 vs mesh-ref 256；B 线各恢复 1,024 点方法 vs Original 1,024。评估只取 test 2,468 个形状，共 14 条条件×2,468=34,552 条对象级记录，0 失败。所有点云按单位球尺度处理。

定义：`CD = mean_{a∈A} min_{b∈B} ||a-b||₂ + mean_{b∈B} min_{a∈A} ||b-a||₂`；这里是**未平方 L2 距离**。`HD = max(max_{a∈A} min_{b∈B} ||a-b||₂, max_{b∈B} min_{a∈A} ||b-a||₂)`。NUC 在每对象抽 128 个中心，半径 0.02/0.05/0.10 统计邻居数，对每尺度计算邻居数标准差/均值，再取三尺度均值；它是项目实现的局部均匀性指标，查询中心在方法间并未完全逐对象配对。

| 线 | 对象 | 参考 | CD（均值） | HD（均值） | NUC（均值） |
|---|---|---|---:|---:|---:|
| A | EAR 4096 | mesh-ref 4096 | 0.054722 | 0.115396 | 0.815631 |
| A | PDANS 4096 | mesh-ref 4096 | 0.046821 | 0.113510 | 0.661176 |
| A | PU-Net 4096 | mesh-ref 4096 | 0.049719 | 0.111177 | 0.605211 |
| A | **PU-GCN 4096** | mesh-ref 4096 | **0.042306** | **0.093219** | 0.520398 |
| A | PU-EdgeFormer 4096 | mesh-ref 4096 | 0.046666 | 0.110932 | 0.675435 |
| B | Downsampled 256 | mesh-ref 256 | 0.133319 | 0.204914 | 2.093341 |
| B | EAR 1024 | Original 1024 | 0.076743 | 0.193182 | 0.965292 |
| B | PDANS 1024 | Original 1024 | 0.062696 | 0.190541 | 1.052826 |
| B | PU-Net 1024 | Original 1024 | 0.064573 | **0.108319** | 1.080102 |
| B | **PU-GCN 1024** | Original 1024 | **0.055802** | 0.151403 | 1.523996 |
| B | PU-EdgeFormer 1024 | Original 1024 | 0.061900 | 0.177572 | 1.122505 |

A 线 PU-GCN 同时具有最低 CD/HD，但分类 Best OA 仍比 Original 低 0.32 pp。B 线 PU-GCN 的 CD 最低，PU-Net 的 HD 最低且分类最佳；PU-GCN B 的 NUC 1.5240 则高于 Original 1.0836，表示按本指标局部密度更不均匀。几何指标描述输出结构，不能直接替代下游任务验证。

旧的非等点数几何表尤其不能作为“表面质量排序”：以 A 线 `repeat_x4` 为反例，把 1,024 个输入点重复四遍后，对旧的 Original 1,024 参考有 CD=HD=0，却没有新增几何信息；对 mesh-ref 4,096 的 equal-N CD=0.055734、HD=0.117819，六种比较里最差。旧参考下 A 线 CD 最好的是 PDANS、HD 最好的是 PU-EdgeFormer；equal-N 下两者都变为 PU-GCN，排序确实改变。仓库 `README.md` 目前仍用旧“geometry vs Original 1024”叙述，和最终 equal-N 报告并存；引用几何结论应以 `reports/modelnet40_geometry_equal_n_protocol.md` 与 `reports/modelnet40_geometry_equal_n_summary.csv` 为准。

## 10. 机制探针与训练稳定性补充

PointNet++ 首层 SA1 固定最多读取半径 0.2 球内 32 点。对 2,468 个 test 对象的输入探针显示：256 点稀疏云邻域平均 14.5 点，约 **59.2% 读取槽由重复点填充**、仅 3.1% 邻域内点因上限丢弃；Original 1,024 点平均 54.8，重复槽 12.1%、丢弃 31.9%；mesh-ref 4,096 点平均 213.4，重复槽 0.2%、丢弃 78.7%。这提供了“从 256 增至 1,024 可能有效、从 1,024 再增至 4,096 收益有限”的**机制线索**，但不能单凭探针证明分类差异的因果关系。256 点输入时 SA1 仍要求 512 个 FPS 中心，存在重复中心，这是架构层面的边界，两种 256 点对照同样受影响。

14 个最终/控制分支各有 200 条逐 epoch 曲线，总计 2,800 行。最后 20 epoch 的 test OA 标准差例如 A Original 0.231 pp、A PU-GCN 0.319 pp、B 稀疏基线 0.198 pp、B PU-Net 0.189 pp；这些是**同一次训练随 epoch 的波动**，不是跨独立 seed 的方差。A Original 最终训练准确率 97.82%、test Final OA 91.31%（差 6.51 pp）；A PU-Net 为 99.51% 对 90.55%（差 8.96 pp），用于描述训练/测试差距，不能直接解释为纯粹上采样误差。

## 11. 证据边界与下一步可审计问题

1. **单 seed**：最终每分支 seed 42 一次，未形成不同 seed 的均值、标准差、置信区间或显著性检验。+0.42 pp 等小差值是当前设置下的观察值，不宜表述为已证实的稳定提升。
2. **test 选 checkpoint**：没有独立 validation split；200 epoch 用 test 选择 Best OA 会引入选择偏差。若论文要估计泛化性能，须新增 validation 选模、固定一次 test 评价，或在正文明确当前口径。
3. **分类指标实现**：OA/类别指标是 batch 平均口径，见第 4 节；与其他论文比较前应重新核对样本加权定义。
4. **几何定义**：CD 使用未平方 L2；NUC 为自定义三尺度局部 CV，查询中心没有在方法间完全按 shape 对齐。P2F 使用另一套独立单位球归一化，宜只作次级审计，勿混进 equal-N 主排序。
5. **逐对象分类证据**：现有训练结果没有保存完整逐对象 logits/概率；不能严谨补出完整 confusion matrix、PR 曲线、calibration 或 McNemar 检验。
6. **端到端随机性**：初始 OFF 采样种子的 Python `hash()` 用法需要在重建数据时固定或替换。已存磁盘数据与 checkpoint 的同 seed 重复核验仍成立。
7. **版本与文档**：实验仓库有未提交文件；根 README 的几何叙述是旧版。论文第 3–6 章与统一图件位于 D，不在实验仓库 Git 历史内。建议在最终提交/归档时固定 SHA256 与当前配置、结果表版本。

## 12. 关键核验证据与交付物

| 内容 | 路径 |
|---|---|
| 本次以前的完整时间线 | `D/ModelNet40_全部实验与修改工作记录_2026-09-10.md` |
| 22 次正式训练 inventory | `E/reports/modelnet40_all_pointcounts_accuracy_inventory.md`（D 根目录也有副本） |
| 最终 12 分支分类 | `E/reports/modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.{md,csv}` |
| 14 个最终 `metrics.json`/checkpoint | `E/pointnet2_results/x4_two_line_final/` |
| 最终 YAML 与 sbatch | `E/configs/pointnet2_x4_two_line/`、`E/jobs/pointnet2_full/` |
| equal-N 几何与对象级证据 | `E/reports/modelnet40_geometry_equal_n_{protocol.md,summary.csv,per_sample.csv,audit.md}` |
| 偏差与机制探针 | `E/reports/modelnet40_geometry_cardinality_bias_probe.md`、`E/reports/modelnet40_sa1_reception_probe.md` |
| Line A 4096 从头训练审计 | `E/reports/modelnet40_pointnet2_lineA_4096_retraining_audit_for_teacher.md` |
| mesh-ref 控制结果 | `E/reports/modelnet40_pointnet2_mesh_ref_baseline_results.{md,csv}` |
| 失败诊断与恢复 | `E/reports/` 下 `failure_diagnosis`、`stall_diagnosis`、`resume`、`metadata_fix` 文件，以及相应 Slurm 日志 |
| 逐 epoch、逐类、几何衍生表 | `D/modelnet40_chapters_5_6/data/` |
| 论文第 3–4 章 PDF | `D/modelnet40_revision/ModelNet40_Chapters_3_4_Expanded.pdf` |
| 论文第 5–6 章 PDF 与审计 | `D/modelnet40_chapters_5_6/ModelNet40_Chapters_5_6.pdf`、`FINAL_AUDIT.txt` |
| 统一图件包 | `D/chapter5_6_unified_figures/`；39 图中 28 张为 ModelNet40，`audit/FINAL_QA.txt` issues=0 |

**本次核验动作**：读取并对照旧总记录、22 次训练 inventory、分类/几何/SA1 原始审计、最终 14 份配置与对应 `metrics.json`；逐一确认 14 条 `train.log` 均到 epoch 200 且 14 个 best checkpoint 存在；查看代码中的训练、DataLoader、几何公式及主要方法调用参数；检查 Git 工作树与 `squeue`。本次仅整理与核验，没有启动训练、上采样、几何重算，也没有修改实验仓库的代码或结果。
