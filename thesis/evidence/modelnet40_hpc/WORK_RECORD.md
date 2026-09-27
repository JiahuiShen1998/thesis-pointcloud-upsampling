# ModelNet40 全部实验、调整、失败与结果工作记录

- 汇总日期：2026-09-10（Europe/Berlin）
- 实际实验仓库：`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling`
- 论文与最终交付目录：`/home/hpc/iwnt/iwnt189h`
- 核验原则：只记录能由现存脚本、配置、Slurm 账单、日志、检查点、CSV/JSON、图表或编译产物证明“实际运行过或实际修改过”的工作。仅有计划而未执行的内容不作为完成工作。

## 1. 当前总状态

1. ModelNet40 点云上采样—PointNet++ 主实验已闭环。最终主协议包含两条线、五种上采样方法：EAR、PDANS、PU-Net、PU-GCN、PU-EdgeFormer。
2. 最终两条线的五种方法均有 12,311 个可访问点云结果（9,843 train + 2,468 test）；Line A 为 4,096 点，Line B 为 1,024 点。PU-EdgeFormer 文件通过链接接入，使用 `find -L` 核验为每条线 12,311 个。
3. 项目内共有 22 次完成的 PointNet++ SSG 正式训练（不含 smoke）；其中包含最终主实验、mesh-reference 控制、重复训练和后来被替代的早期协议。
4. 最终主实验有 12 个分类分支：Line A 1 个基线 + 5 个方法，Line B 1 个基线 + 5 个方法；另有 2 个 mesh-reference 控制分支。
5. 当前 Slurm 队列为空，没有仍在运行或等待的作业。
6. 实验仓库当前不是干净工作树：分支 `share/modelnet40-pointnet2-presentation`，有 20 个 tracked 文件被修改、59 个 untracked 条目。仓库只有一个已见提交 `daacfaf`（2026-07-27），因此大量后续结果尚未提交。
7. 当前实验仓库可见规模：146 个脚本、69 个 Slurm/job 文件、373 个报告文件、1,005 个日志文件、24 个配置、182 个 PointNet++ 结果文件、273 个图形文件、25 个演示文稿文件。

## 2. 最终采用的实验协议

| 实验线 | 基线 | 上采样分支 | PointNet++ 训练方式 |
|---|---|---|---|
| Line A：原始点云增密 | Original 1,024 | 1,024 → ×4 → 4,096 | 基线按 1,024 点从头训练；每个方法按 4,096 点分别从头训练 |
| Line B：四倍降采样后的恢复 | Original 1,024 → decimate ×4 → 256 | 256 → ×4 → 1,024 | 基线按 256 点从头训练；每个方法按 1,024 点分别从头训练 |

统一设置：`pointnet2_cls_ssg`、200 epochs、Adam、初始学习率 0.001、seed 42、无 normals。最终分支不允许 DataLoader 擅自补点或裁剪来匹配另一分支。

### 协议实际调整轨迹

| 阶段 | 实际做过的协议 | 后续处理 |
|---|---|---|
| 早期基线 | Original 1,024；Downsampled50 512 | 保留为早期结果 |
| 早期 EAR | 512 → EAR ×2 → 1,024；同时跑过 512 经 loader 重采样到 1,024 | 均降级为 preliminary/ablation |
| 第一版 ×4 | Line A 1,024→4,096；Line B 512→2,048 | EAR、PDANS 和部分 PU-Net 实际生成/训练过，但不再是最终 Line B 主协议 |
| 最终 ×4 | Line A 1,024→4,096；Line B 256→1,024 | 成为论文主协议 |
| 方法扩展 | 先完成 EAR/PDANS/PU-Net/PU-GCN，后接入 PU-EdgeFormer | 最终主表包含五种方法 |
| 几何评估修正 | 早期存在 4,096 vs 1,024 等 unequal-cardinality CD/HD | 后来改为 equal-N：Line A 4,096 vs mesh-ref 4,096；Line B 256 vs mesh-ref 256、1,024 vs Original 1,024 |

TULIP 始终只被标记为 supplementary，SPU-PMD 未进入 ModelNet40 主协议；二者没有最终主实验结果，不能列为已完成方法。

## 3. 按时间的实际工作记录

### 2026-06-20：数据与分类器基础

- 运行 ModelNet40 预处理：40 类、9,843 train、2,468 test，共 12,311 个样本；每个样本从 `.off` 按三角形面积加权采样 1,024 点，再中心化到单位球；153.7 秒完成，0 失败。
- 运行完整数据检查：计数、shape `(1024,3)`、NaN、Inf、单位球和随机样本均 PASS。
- 接入 `yanx27/Pointnet_Pointnet2_pytorch` 的 `pointnet2_cls_ssg`，编写自定义 NPY DataLoader、训练/评估包装器、配置和 sbatch。
- GPU smoke job 1709664 完成：加载、前向、mini-train、eval、checkpoint 全部 PASS。
- 提交 Original baseline job 1709666，最终完成 200 epochs。
- 生成 Downsampled50：1,024→512，12,311 个样本、0 失败；subset、可复现性和有限值检查全部 PASS。
- 运行 loader-resampled 512→1,024 基线 job 1710180；该结果后来被重新定义为消融，不再是主基线。

### 2026-06-21 至 2026-06-22：EAR ×2 与早期 PointNet++

- EAR CPU smoke 实际运行：Original 分支在 `target_n=1024` 时是 identity copy；Downsampled50 分支完成 512→1,024，160 个 smoke 样本全部 PASS。
- EAR full Line B 512→1,024 完成 12,311/12,311，shape/NaN/Inf/DataLoader 检查 PASS。
- EAR ×2 + PointNet++ job 1711437 完成，Best OA 90.82%；后来标记为 `ablation_preliminary_x2`。
- 原生 512 点 PointNet++ job 1716131 完成，成为早期 512 基线。

### 2026-06-25：主倍率与公平性规则更正

- 先纠正“基线和上采样必须同点数”的错误规则：主基线应保留原生点数，生成结果保留上采样后点数。
- 主倍率从 ×2 改为 ×4。
- 将 EAR 512→1,024 和 loader 512→1,024 结果保留但降级为 preliminary/ablation，不删除数据和日志。
- 当时规划为 Line A 1,024→4,096、Line B 512→2,048；该 Line B 设计后来又被最终 256→1,024 取代。

### 2026-06-26 至 2026-07-01：多方法环境、上采样与早期 ×4 训练

- EAR Line B array 1716175 中两个任务因 metadata symlink 竞态失败；修改 `ensure_metadata_link()` 捕获 `FileNotFoundError`，只重跑缺失块，最终 EAR ×4 两线生成 PASS。
- 第一批并行提交脚本把方法名错误写成 `pdans}`、`punet}`、`pugcn}`，形成带 `}` 的目录/报告；修正提交逻辑后重新提交。
- 修正后的 6 个 smoke（1717035/37/39/41/43/45）仍全部失败：PDANS 的 CUDA/PyTorch 环境不一致；PU-Net/PU-GCN 的 TF1 custom ops 找不到正确 `nvcc`、`libcudart`、`libtensorflow_framework`。
- 修改 CUDA 激活、TF framework 链接和 custom-op 编译脚本；在 GPU 节点上重新编译。最终 PU-Net `tf_sampling` 和 PU-GCN `tf_grouping` compile/import PASS。
- PDANS 先遇到 PyTorch3D ABI 不匹配（job 1717964），后又在 job 1722422 中遇到缺少 `open3d`；安装/调整兼容依赖后 job 1723850 完整 GPU validate PASS。
- PDANS smoke 及两线 full generation PASS；EAR/PDANS 的早期 ×4 正式训练完成。
- PU-Net smoke 1725581/1725582 PASS，实际调用官方 generator checkpoint，而非通用插值。

### 2026-07-02 至 2026-07-07：最终 Line B、恢复任务与最终 smoke

- 最终固定 Line B 为 1,024→decimate ×4→256→upsample ×4→1,024；旧 512→2,048 数据保留但退出主协议。
- EAR、PDANS 最终 Line B 各完成 12,311/12,311，并做来源与 shape 审计。
- PU-Net smoke 1727073：推理成功但保存 strict 输出前未创建父目录，40/40 写入失败；补 `strict_out.parent.mkdir(parents=True, exist_ok=True)` 后 1729225 PASS。
- PU-Net full array 1729235：4 个 chunk 完成，11 个 chunk 24 小时 TIMEOUT，1 个 chunk 在 `bookshelf_0424` 上 TF 子进程 `rc=-6` 失败；保留 9,150 个已审计有效输出。
- PU-Net missing-only resume array 1731611 完成，最终 raw/strict 12,311/12,311，全部 `(1024,3)`、无 NaN/Inf。
- PU-GCN smoke 1731790/1731791 在修复 `open3d`/`plyfile`、并行输出目录竞争和 smoke 样本对齐后两线 PASS。
- PU-GCN Line A full 1731793 完成 12,311/12,311。
- PU-GCN Line B full 1731794 在 10,010/12,311 时卡住：tasks 13–15 都在 RTX 3080 节点上，`knn_point_2` 100% 推理失败、120 分钟无新文件；手动取消这三个任务并保留已有输出。
- PU-GCN 第一次恢复 array 1732373 的 8 个任务均因 Python `global` 声明位置导致 SyntaxError，在推理前失败；修改为局部变量并显式传参，`py_compile` PASS。
- 修复后 smoke 1732385 对 2 个缺失样本 PASS；第二次恢复 array 1732386 在 RTX 2080 Ti 限定节点上全部完成。当前实际核验 raw=12,311、strict=12,311、missing list=0。
- PointNet++ Line B 第一轮 smoke：基线 PASS，四个方法 job 1733160–1733163 因缺少 label-map metadata 全部失败。
- metadata v1 把整个 metadata 目录链接进来，job 1733167–1733170 又因 manifest 指回 256 点基线而全部失败。
- metadata v2 只链接 `class_to_idx.json` 和 `idx_to_class.json`，不链接 manifests；job 1733171–1733174 全部 PASS。
- PointNet++ Line A 第一轮 smoke：基线 PASS，job 1733476–1733479 同样因 metadata 缺失失败；应用 v2 规则后 1733487–1733490 全部 PASS。

### 2026-07-07 至 2026-07-08：最终两线 PointNet++

- Line B 正式训练 jobs 1733191–1733195：5/5 COMPLETED，均 200 epochs。
- Line A 正式训练 jobs 1733491–1733495：5/5 COMPLETED，均 200 epochs。
- 每个分支独立从随机初始化训练，没有把 1,024 点 checkpoint 用于 4,096 点分支。

### 2026-07-12 至 2026-07-17：结果分析与 PU-EdgeFormer

- 生成分类、几何、逐类结果、方法排名、几何—分类关系和论文表格。
- 制作静态/交互式点云对比；v2 共 10 个 Line A grid、10 个 Line B grid、10 个 dropdown、22 个 single-method HTML。
- 接入 PU-EdgeFormer，先完成真实性、调用路径、freshness、repeat/copy、与 PU-GCN 差异等审计。
- PointNet++ smoke jobs 1749430/1749431 两线 PASS。
- 正式训练 jobs 1749441/1749442 两线 COMPLETED：Line B Best OA 89.32%，Line A Best OA 90.54%。
- 最终可视化审计 PASS：33 个最终 package PNG、52 个 interactive HTML；没有重新训练或修改数据。

### 2026-07-23：4,096 点训练真实性复核

- 复核五个 Line A 上采样分支的 YAML、sbatch、训练代码、首 batch、首 loss、epoch 200、metrics 和独立 checkpoint。
- 结论：五个分支均确实以 4,096 点输入、从头训练；无需再次 clean rerun。

### 2026-08-03 至 2026-08-18：equal-N、控制实验、机制探针与总报告

- 从 `.off` 面积加权重采样构建 mesh-ref 256 和 mesh-ref 4,096，各 12,311 个；构建审计 PASS。
- 运行 equal-cardinality CD/HD/NUC：test 2,468 个对象，workers 用时 128.1 秒，0 失败。
- 训练 mesh-ref PointNet++ 控制 jobs 1768767/1768768，均完成 200 epochs。
- 运行 cardinality-bias probe：`repeat_x4` 在旧的 4,096 vs Original 1,024 指标下得到 CD=HD=0，却在 equal-N 下最差，证明旧参考设计奖励“不做上采样”；由此废弃 unequal-cardinality 主排名。
- 运行 PointNet++ SA1 reception probe：256 点时 59.2% 槽位靠重复填充；1,024 点时丢弃 31.9%；4,096 点时丢弃 78.7%，解释继续增密为何收益很小。
- 对 Original 1,024、EAR 4,096、PDANS 4,096 三对重复训练 checkpoint 做逐张量比较：79/79 权重完全一致，最大差 0；证明 seed 42 下流程是 bit-deterministic，但不等价于多 seed 统计稳定。
- 生成完整 dossier、Full Report、Progress Report、PPTX/PDF、图表和全部点数训练 inventory。

### 2026-09-04：论文第 3、4 章

- 提取并扩写 Idea and Concept、Experimental Setup，覆盖研究假设、两条线、公平性、上采样协议、下游评估、指标和复现检查。
- 最终 PDF 27 页；正文中文字符 18,055；13 个 citation key 均有条目；无 undefined reference、missing glyph、overfull box、重复主段落。
- 实际做过三轮 build、页面 PNG 预览、contact sheet 和最终页面检查。编译成功；日志存在非阻断的 underfull box 提示。

### 2026-09-07 至 2026-09-08：论文第 5、6 章及附加审计

- 从 v1/expanded/pre-readable 多版正文修改到最终 `chapters_5_6_body.tex`。
- 针对“图太花、数据看不清、点云对比不足、证据与参数不足”重做全部主图：最终 PDF 54 页、正文使用 28 张唯一图，每张有 PDF 和 300 dpi PNG。
- 汇入 14 个最终/控制分支的 2,800 条逐 epoch 记录、600 条逐类准确率、34,552 条 equal-N 对象几何记录、29,616 条 dense-reference/P2F 次级记录、150 条 pairwise 胜率、112 个可视化点云数组。
- 删除主文中拥挤的旧热图，改为灰阶、细网格、空心标记、0–100% 逐类图和四页并列点云证据。
- 最终审计：54 页、28/28 图引用、无缺图、无 overfull、无 undefined reference、无 LaTeX error；SHA256 为 `0c94b8a8f825a4c33a109f9c5adfa58f3de094812be8a547df41737546bc9488`。

## 4. 全部 22 次已完成 PointNet++ 正式训练

以下均有完成结果；smoke 不计入。`early` 表示后来被替代或重复的协议，仍是真实运行结果。

| 点数 | 运行 | Best OA | Final OA | Best Class | 最佳 epoch | 数据来源 | 轮次 |
|---:|---|---:|---:|---:|---:|---|---|
| 256 | mesh_ref_baseline_256 | 90.88% | 90.38% | 86.34% | 129 | `.off` 面积加权重采样 | final control |
| 256 | lineB_downsampled_x4_baseline | 90.85% | 90.46% | 87.37% | 124 | Original 1,024 decimate ×4 | final |
| 512 | downsampled50_native512_pointnet2 | 91.26% | 91.14% | 88.00% | 197 | Original 降采样约 50% | early |
| 1,024 | lineA_original_baseline | 91.95% | 91.31% | 88.00% | 85 | Original mesh sampling | final |
| 1,024 | original_baseline | 91.95% | 未记录 | 88.00% | 85 | 与上项同数据的早期重复 | early |
| 1,024 | lineB PU-Net | 91.27% | 90.65% | 87.86% | 148 | 256→PU-Net×4 | final |
| 1,024 | downsampled50_baseline | 91.17% | 未记录 | 87.71% | 155 | 磁盘 512，loader 重采样到 1,024 | early ablation |
| 1,024 | step8_downsampled50_ear | 90.82% | 未记录 | 87.06% | 111 | 512→EAR×2 | early ablation |
| 1,024 | lineB PDANS | 90.34% | 89.46% | 86.20% | 145 | 256→PDANS×4 | final |
| 1,024 | lineB PU-GCN | 90.06% | 89.63% | 86.41% | 164 | 256→PU-GCN×4 | final |
| 1,024 | lineB PU-EdgeFormer | 89.32% | 88.58% | 85.21% | 194 | 256→PU-EdgeFormer×4 | final |
| 1,024 | lineB EAR | 88.81% | 88.02% | 84.50% | 116 | 256→EAR×4 | final |
| 2,048 | downsampled50_pdans_x4_pointnet2 | 91.47% | 90.99% | 88.19% | 164 | 512→PDANS×4 | early |
| 2,048 | downsampled50_ear_x4_pointnet2 | 90.61% | 89.76% | 87.45% | 74 | 512→EAR×4 | early |
| 4,096 | mesh_ref_baseline_4096 | 92.00% | 91.18% | 88.70% | 148 | `.off` 面积加权重采样 | final control |
| 4,096 | lineA PU-GCN | 91.63% | 91.00% | 87.74% | 81 | 1,024→PU-GCN×4 | final |
| 4,096 | lineA PDANS | 91.62% | 90.87% | 88.44% | 84 | 1,024→PDANS×4 | final |
| 4,096 | original_pdans_x4_pointnet2 | 91.62% | 90.87% | 88.44% | 84 | 同数据的早期重复 | early |
| 4,096 | lineA EAR | 91.48% | 90.85% | 88.19% | 61 | 1,024→EAR×4 | final |
| 4,096 | original_ear_x4_pointnet2 | 91.48% | 90.85% | 88.19% | 61 | 同数据的早期重复 | early |
| 4,096 | lineA PU-Net | 90.95% | 90.55% | 87.46% | 189 | 1,024→PU-Net×4 | final |
| 4,096 | lineA PU-EdgeFormer | 90.54% | 90.13% | 86.35% | 83 | 1,024→PU-EdgeFormer×4 | final |

### 最终分类结论

- Line A：五种方法都没有超过 Original 1,024 的 91.95%；最接近的是 PU-GCN 91.63%（−0.32 pp）和 PDANS 91.62%（−0.33 pp）。
- Line B：只有 PU-Net 超过 256 点基线，91.27% 对 90.85%，提升 +0.42 pp；其余 EAR −2.04 pp、PDANS −0.51 pp、PU-GCN −0.79 pp、PU-EdgeFormer −1.53 pp。
- Mesh-reference 控制：mesh-ref 256 为 90.88%，仅比 downsampled 256 高 0.03 pp；mesh-ref 4,096 为 92.00%，仅比 Original 1,024 高 0.05 pp。
- 协议敏感性：PDANS 在旧 512→2,048 协议相对 native 512 为 +0.21 pp，在最终 256→1,024 协议却为 −0.51 pp，结论会随降采样深度翻转。

## 5. 最终 equal-N 几何结果

CD/HD 为同点数参考下的 test-set mean，N=2,468；越低越好。

| Line | 方法 | 参考 | CD | HD | NUC |
|---|---|---|---:|---:|---:|
| A | EAR | mesh-ref 4,096 | 0.054722 | 0.115396 | 0.815631 |
| A | PDANS | mesh-ref 4,096 | 0.046821 | 0.113510 | 0.661176 |
| A | PU-Net | mesh-ref 4,096 | 0.049719 | 0.111177 | 0.605211 |
| A | PU-GCN | mesh-ref 4,096 | **0.042306** | **0.093219** | 0.520398 |
| A | PU-EdgeFormer | mesh-ref 4,096 | 0.046666 | 0.110932 | 0.675435 |
| B | EAR | Original 1,024 | 0.076743 | 0.193182 | 0.965292 |
| B | PDANS | Original 1,024 | 0.062696 | 0.190541 | 1.052826 |
| B | PU-Net | Original 1,024 | 0.064573 | **0.108319** | 1.080102 |
| B | PU-GCN | Original 1,024 | **0.055802** | 0.151403 | 1.523996 |
| B | PU-EdgeFormer | Original 1,024 | 0.061900 | 0.177572 | 1.122505 |

结论：两条线的最低平均 CD 都是 PU-GCN；Line B 最低 HD 是 PU-Net。几何 CD 领先者与分类 OA 领先者不是稳定的一一对应关系。

## 6. 失败、取消、超时、返工与最终处理

| 日期/作业 | 失败或问题 | 实际修改/处理 | 最终结果 |
|---|---|---|---|
| 1716175 tasks 4/6 | metadata symlink 并发竞态，`FileNotFoundError` | `unlink()` 捕获缺失异常，只重跑缺块 | EAR full 最终 PASS |
| 1716440–1716451 提交 | shell 方法名多出 `}` | 修正方法展开和重提交流程 | 错误目录保留作证据 |
| 1717035/37 | PDANS 无可用 CUDA | 统一模块、conda、CUDA 路径 | 后续 GPU validate PASS |
| 1717039/41、1717043/45 | PU-Net/PU-GCN TF custom ops 编译/导入失败 | 修正 `CUDA_HOME`、nvcc、cudart、TF framework 链接，在 GPU 节点编译 | compile/import PASS |
| 1717964 | PyTorch3D ABI mismatch | 调整 torch/PyTorch3D 兼容环境 | 后续越过该错误 |
| 1722422 | PDANS 缺 `open3d` | 安装 open3d 0.17.0，并重新验证 | 1723850 PASS |
| 1727073 | PU-Net strict 保存目录不存在 | 增加 parent `mkdir` | 1729225 smoke PASS |
| 1729235 array | 11 个 TIMEOUT，1 个 `rc=-6`，只完成 9,150 | missing-only、skip-existing 恢复 | 1731611 完成 12,311 |
| 1731794 tasks 13–15 | RTX 3080 上 `knn_point_2` 100% 失败 | 取消卡住任务，保留 10,010；恢复任务限定 RTX 2080 Ti | 数据最终 12,311 |
| 1732373 array | 恢复脚本 SyntaxError | 去掉有问题的 global，显式传参，加 `--max-samples`，py_compile+smoke | 1732386 完成 |
| 1733160–1733163 | Line B 上采样目录无 label map | metadata v1 整目录链接 | v1 仍失败 |
| 1733167–1733170 | v1 manifest 指回 256 点文件 | v2 仅链接两个 class-map，不带 manifest | 1733171–1733174 PASS |
| 1733476–1733479 | Line A 同类 metadata 缺失 | 套用 v2 | 1733487–1733490 PASS |
| 旧几何表 | unequal-cardinality CD/HD 会奖励重复输入 | 增加 mesh-ref、equal-N 和 repeat×4 probe，重做图表/论文 | 旧排名退出主结论 |
| 早期 PPT | 混用 dense mesh/P2F 和 vs-Original 叙述，且链接为 `file://` | 重写 geometry loader/图表、加入 mesh-ref 控制、重选相机、重生成 21 页 PPT/PDF 和 10 个 dropdown_v2 | 文件已修改；公开 HTTPS base 仍未设置 |

## 7. 当前仍存在的证据边界

这些不是“任务失败”，而是现有结果不能越界解释的限制：

- 每个最终分类分支只有 seed 42 一次独立训练；没有多 seed 均值、方差或显著性检验。
- Best OA 使用 test set 选择 checkpoint；没有独立 validation set。论文正式表述应优先使用补验证集或明确这一点。
- 已验证同 seed 可 bit-deterministic，不代表跨 seed 稳健。
- 当前 Chamfer 实现是未平方欧氏最近邻距离之和；论文式若写成平方距离，需要统一定义。
- NUC 查询中心并非在所有方法间按 shape ID 完全共享；P2F 使用独立单位球归一化，因此只作次级审计。
- 没有保存逐对象分类 logits/概率，不能严谨生成 PR 曲线、完整 confusion matrix、McNemar 或 calibration 曲线。
- PU-GCN 最终 12,311 计数、0 missing、PointNet++ 完成均有实际文件和 Slurm/metrics 证据，但仓库缺少命名为 `modelnet40_pugcn_final_count_audit.md` 和 `modelnet40_pugcn_provenance_audit.md` 的最终 Markdown 闭环文件。
- 论文第 3–4 章和第 5–6 章目录位于 `/home/hpc/...`，不属于当前实验 Git 仓库，尚无 Git 版本历史保护。

## 8. 当前未提交修改

### tracked 修改（20 个）

- 6 张 `figures/modelnet40/original_reference_revised/` 几何/几何—分类图。
- 10 个 `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/` 交互 HTML。
- `presentations/ModelNet40_PointNet2_Original_Reference_Revised.pdf`。
- `presentations/ModelNet40_PointNet2_Original_Reference_Revised.pptx`。
- `presentations/ModelNet40_Presentation_Revision_Notes.md`。
- `scripts/generate_original_reference_presentation.py`：约 507 行新增、321 行删除的整体 diff 中，该脚本是主要文本改动；核心为 equal-N 数据优先、mesh-ref 控制、几何标签和页面叙述更正。

### untracked 条目（59 个 Git 顶层条目）

主要包含：CUDA 兼容链接、external 依赖、mesh-ref 256/4,096 配置与 jobs、equal-N 与 SA1/cardinality 探针脚本和报告、两套 mesh-ref 数据审计、完整 PPT/PDF/dossier/progress report、结果总表图片、progress charts、两个 tar.gz 交付包。它们都存在于工作树，但还没有进入 Git。

## 9. 关键可核验入口

- 全部完成训练总表：`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_all_pointcounts_accuracy_inventory.md`
- 最终分类：`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.md`
- equal-N 几何：`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/modelnet40_geometry_equal_n_summary.csv`
- 失败诊断：`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/reports/` 下所有 `failure_diagnosis`、`stall_diagnosis`、`cancelled_for_resume` 和 `correction` 文件。
- 最终结果目录：`/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/pointnet2_results/x4_two_line_final/`
- 第 3、4 章：`/home/hpc/iwnt/iwnt189h/modelnet40_revision/ModelNet40_Chapters_3_4_Expanded.pdf`
- 第 5、6 章：`/home/hpc/iwnt/iwnt189h/modelnet40_chapters_5_6/ModelNet40_Chapters_5_6.pdf`
- 第 5、6 章最终审计：`/home/hpc/iwnt/iwnt189h/modelnet40_chapters_5_6/FINAL_AUDIT.txt`

## 10. 一句话结论

这批工作不是“只得到一个最好准确率”，而是完成了从数据构建、五种上采样器适配、失败恢复、两条公平协议的 PointNet++ 从头训练、equal-N 几何纠偏、控制实验与机制探针，到演示文稿和论文第 3–6 章的完整链条；最终分类上只有 Line B 的 PU-Net 相对 256 点基线有 +0.42 pp 提升，而 Line A 所有方法均未超过 Original 1,024，且几何最优方法与分类最优方法并不一致。
