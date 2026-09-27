# PU-GCN → detector adaptation：完整协议、点选择与代码说明

## 0. 两个问题的直接答案

**“3 epochs”指 detector adaptation，不是 PU-GCN 训练。**PointRCNN 是 RPN 3 epochs，再做 offline RCNN 3 epochs；CenterPoint 是 3 epochs。PU-GCN 没有在 KITTI 上按 3 epochs 重训，本实验固定使用作者 PU1K 的 `model-100`。代码只证明 3 是预先写入脚本的固定 adaptation 长度，找不到“通过 validation 收敛曲线选择 3”的记录。只保存了 epoch 3，且没有 epoch 1/2 的 validation AP，所以不能声称已经 converge；只能说三轮完成且 training loss 下降。

**“3N/3M”不是 PU-GCN 自己标出的三组新增点。**PU-GCN 对每个 anchor 输出 4 个 residual 坐标；重叠 patch 合并后，先用固定 seed 的均匀无放回 sampling 得到严格 4N/4M。构造 observed-first 输入时，完整保留 N/M 个 observed rows，再从 strict 4N/4M 预测 rows 中用另一固定 seed 均匀无放回抽 3N/3M。当前没有按 confidence、到 observed 的距离、FPS、曲率或 voxel 去重来选 3N/3M。

三次选点必须区分：

| 阶段 | 候选 → 结果 | 实际准则 |
|---|---|---|
| patch 输入选择 | 全帧 N/M → 多个 2048-point patch | 密度门槛 + seeded first center + XYZ FPS + radius/kNN coverage |
| strict 4× | 所有 patch raw 输出 `J×8192` → 4N/4M | `np.random.default_rng(20260702+frame_id).choice(..., replace=False)` |
| observed-first 新增点 | strict predicted 4N/4M → 3N/3M | SHA-256 派生逐帧 seed 后 `choice(..., replace=False)`；与几何和 score 无关 |

## 1. 先把容易混淆的三件事分开

1. **PU-GCN 没有只训练 3 epochs。**本实验没有在 KITTI 上重新训练 PU-GCN，而是固定使用作者发布的 PU1K checkpoint `model-100`。原 checkpoint 的训练参数写的是 `max_epochs: 200`，checkpoint 指针实际指向 `model-100`。
2. **3 epochs 是 detector adaptation。**PointRCNN 的 RPN 训练 3 epochs，之后离线 RCNN 再训练 3 epochs；CenterPoint adaptation 训练 3 epochs。初始化权重分别是官方 PointRCNN 和官方 80-epoch CenterPoint KITTI checkpoint。
3. **旧 detector 数字只来自 pilot256。**旧目录中虽然 adaptation 使用完整 3712 个 KITTI train frame，但其 AP evaluation 只有 256 个 validation frame。它不能回答“完整 validation 是否跑过”。本目录重新生成并评估全部 **3769** 个 validation frame；`3796` 是误写。

数据边界如下：

| 用途 | split | 实际非空唯一 frame 数 | 是否使用 label |
|---|---|---:|---|
| detector adaptation | KITTI train | 3712 | 是，仅训练 |
| 最终 validation | KITTI val | 3769 | 仅在最终 KITTI AP 计算时使用 |

## 2. Line A、Line B 和两种 detector 输入

令一帧原始 KITTI 点云为

\[
O_i\in\mathbb{R}^{N_i\times4},\quad (x,y,z,r)
\]

其中 `r` 是 intensity。

### Line A

- observed 输入：原始点云 `O_i`，点数 `N_i`；
- PU-GCN direct 输出：严格 `4N_i` 个点；
- observed-first detector 输入：`N_i` 个原始 observed 点，加上从 direct `4N_i` 中无放回抽取的 `3N_i` 个网络输出点，最终仍是 `4N_i`。

公式为：

\[
D^A_i=\operatorname{StrictSample}(\operatorname{Merge}(\operatorname{PUGCN}(P^A_{ij})),4N_i)
\]

\[
X^A_i=O_i\;\Vert\;D^A_i[S_i],\qquad |S_i|=3N_i
\]

### Line B

先从原始 `N_i` 个点中确定性地下采样：

\[
M_i=\lfloor N_i/4\rfloor
\]

- `M_i` 个 row index 由 NumPy `Generator.choice(N_i, M_i, replace=False)` 取得；
- seed 为 `20260702 + int(frame_id)`；
- index 排序后再索引，因此保留原文件中的相对行序；
- validation manifest 已审计 3769/3769 帧，全部满足 `M=floor(N/4)`。

随后：

\[
D^B_i=\operatorname{StrictSample}(\operatorname{Merge}(\operatorname{PUGCN}(P^B_{ij})),4M_i)
\]

\[
X^B_i=B_i\;\Vert\;D^B_i[T_i],\qquad |T_i|=3M_i
\]

所以 `4M_i=4 floor(N_i/4)=N_i-(N_i mod 4)`，它非常接近、但不一定等于 `N_i`。

## 3. 3N / 3M 到底怎样获得

它不是 PU-GCN 原网络直接返回的“额外三组点”标签，而是 detector 输入适配层执行的第二次确定性抽样：

```python
expected = 4 * observed.shape[0]
assert predicted.shape[0] == expected
seed = stable_seed(20260718, "e1", reference_token, frame_id)
rng = np.random.default_rng(seed)
selected = rng.choice(expected, size=3 * observed.shape[0], replace=False)
detector_input = np.concatenate((observed, predicted[selected]), axis=0)
```

`stable_seed` 是字符串拼接后取 SHA-256 digest 的前 8 bytes，再截成 32-bit integer。Line A 的 `reference_token` 是 `original_x4_pu_gcn`；Line B 是 `downsampled_x4_pu_gcn`。

这里“predicted/generated”是**来源定义**：该 row 来自 PU-GCN 输出。它不是集合差定义。实现中没有坐标去重，也没有删除恰好落在 observed 附近的输出。因此不能声称 3N/3M 在几何上都与 observed 不同。

## 4. 一整帧怎样选择 2048 点 patch 输入 PU-GCN

本实验使用 `fps_ball_cover_knn_v3`，每个 patch 固定 `K=2048`，Line A 与 Line B 的参数为：

| 参数 | Line A | Line B |
|---|---:|---:|
| primary patch budget | `ceil(N/2048) × 1` | `ceil(M/2048) × 1` |
| primary ball radius | 2 m | 4 m |
| primary center 最低支持点 | 2048 | 2048 |
| cover radius | 6 m | 6 m |
| cover eligibility | radius 6 m 内至少 32 点 | 同左 |
| supplemental kNN floor | 2048 unique points | 同左 |
| center/FPS 距离 | XYZ 欧氏距离，单位 m | 同左 |
| seed | `20260702 + frame_id` | 同左 |

具体顺序：

1. 用 `cKDTree` 统计每个输入点在 primary radius 内的支持点数，只允许支持点数 `>=2048` 的点成为 primary center。
2. 在 eligible center 中用 seeded uniform 选择第一个 center。
3. 后续 center 用标准 FPS：选择到已有 center 集合的最小平方距离最大的 eligible 点。
4. 对每个 primary center 做 radius query，按 `(distance, original_index)` 稳定排序，取前 2048 个。因为 center 已要求 radius 内至少 2048 点，所以 primary patch 不需要复制填充。
5. 统计输入点是否已出现在任一 patch 中。对于 radius 6 m 内至少有 32 个点、但仍未覆盖的点，选择“距离已有 centers 最远”的未覆盖点作为 supplemental center。
6. supplemental patch 中先放未覆盖的 6 m ball 点，再放已覆盖的 6 m ball 点；仍不足 2048 时，用该 center 的全帧 kNN 补到 2048 unique rows。
7. 对 cover-eligible 输入点要求 100% coverage。radius 6 m 内少于 32 点的孤立点不在该 100% 保证内；协议不会伪造这些点来宣称全覆盖。

所有 patch 只读取 XYZ，选点时不 jitter、不插值、不改坐标。每帧 metadata 保存每个 patch 的 center、原始 row index、支持点数、覆盖率和几何诊断。

## 5. PU-GCN 在每个 patch 上做什么

预训练参数：

- dataset：PU1K；
- model：PUGCN；
- training patch：256 → 1024；
- inference patch：本协议利用全卷积/图网络形式改成 2048 → 8192；
- block：Inception DenseGCN；
- `n_blocks=2`, `channels=32`, `k=20`, `d=2`；
- upsampler：NodeShuffle；
- ratio：4；
- fixed checkpoint：`model-100`。

每个 2048-point patch 会先独立做中心化和单位球归一化：

\[
\hat p=(p-c)/r_{max}
\]

网络在归一化坐标上预测 residual。原始实现将每个输入 anchor tile 4 次，并把预测 offset 加到相应 anchor：

\[
q_{j,t}=p_j+\Delta_{j,t},\quad t=1,2,3,4
\]

最后再执行 `q = centroid + q_hat * furthest_distance` 还原到 LiDAR 坐标。因此每个 2048 patch 返回 8192 个网络输出 row。Line A 的 2 m 与 Line B 的 4 m 是 patch 构造的绝对空间范围；送入网络后每个 patch 仍会各自归一化。

## 6. 从重叠 patch 输出得到严格 4N / 4M

假设一帧最终有 `J_i` 个 patch，则 raw candidate 数是：

\[
R_i=J_i\times8192
\]

runner 按 patch ID 顺序 concatenate 全部 raw output。随后 strict adapter 使用 seed `20260702 + frame_id`：

- 若 `R_i == 4N_i`（或 `4M_i`），直接复制；
- 若 `R_i > target`，从 `R_i` 行中均匀、无放回抽取 exact target；
- 若 `R_i < target`，立即失败；禁止复制输入点、重复 raw row 或合成假点补足；
- intensity 不由 PU-GCN 预测，而是从 strict XYZ 的最近 observed XYZ 做 1-NN 赋值；
- 最终保存标准 KITTI float32 XYZI `.bin`。

重叠 patch 可能对同一个输入 anchor 产生多套输出；当前协议不做 voxel/grid/坐标去重。这是协议的一部分，不能在结果出来后更换。

## 7. detector adaptation 协议

### PointRCNN

- official init：`tools/PointRCNN.pth`；
- train split：完整 3712 帧；
- 每种输入 arm 独立 adaptation；
- RPN：3 epochs，batch size 1，workers 0；
- 从 epoch-3 RPN 导出 train split proposals 与 features；
- 将 adapted RPN 和 official RCNN 合并，作为 offline RCNN 初始化；
- offline RCNN：3 epochs，batch size 1，workers 0；
- optimizer：Adam one-cycle；
- LR：0.0002；LR clip：1e-6；decay step `[2]`；无 LR warmup；
- seed：20260823；cuDNN deterministic；
- GT database augmentation：关闭；
- 只在 epoch 3 保存 checkpoint。

PointRCNN 分别训练了 direct 与 observed-first 的 matched checkpoint；Line B baseline 也有独立 adapted checkpoint。

### CenterPoint

- official init：KITTI 80-epoch checkpoint；
- train split：完整 3712 帧；
- batch size 2，workers 0；
- epochs：3；
- optimizer：Adam one-cycle；
- LR：0.0003；weight decay 0.01；
- seed：666；
- GT sampling：关闭；其他 world flip/rotation/scaling 保留；
- checkpoint interval：3，因此只保存 epoch 3；
- PU-GCN adaptation 训练输入是 observed-first，而不是 generated-only/direct。

## 8. 为什么是 3 epochs，是否 converge

现有记录能支持的结论是：**3 epochs 是脚本预先固定的 adaptation 长度，没有根据 validation convergence 选择 epoch 的记录。**历史脚本没有记录为什么恰好选 3 而不是 6 或 10，所以不能把计算预算方面的推测写成已证实的实验依据。

PointRCNN 每个 arm 的 epoch 内 loss 中位数都下降；例如 Line A direct RPN 从 1.6241 降到 1.3475，RCNN 从 1.0430 降到 0.9596。CenterPoint Line A mean loss 从 2.33 降到 2.11，Line B PU-GCN 从 3.69 降到 3.10。完整逐 arm 数字见 `convergence_audit.md`。

但不能据此声称 converge：

- 只保存 epoch 3，没有 epoch 1、2 checkpoint；
- 训练时没有逐 epoch full-val AP；
- loss 仍在下降，未展示 plateau；
- training loss 下降不等于 detector validation AP 已到最优。

严谨说法应是：“3-epoch adaptation 正常完成，optimization loss 下降；是否收敛没有被当前 checkpoint schedule 证明。”若论文必须写 convergence，需要重新跑至少 6–10 epochs、每 epoch 保存，并用预先声明的 held-out selection split 或逐 epoch full-val 曲线判断；不能用最终 test/val 结果事后挑 epoch 却仍声称无 selection bias。

## 9. detector 真正读取哪些点

### PointRCNN

frame-level `.bin` 先按相机 FOV 和配置范围过滤。RPN 再固定输入 16384 点：

- 若候选多于 16384，通常保留全部深度 `>=40 m` 的 far points，再从 near points 无放回补足；
- 若 far points 本身已 `>=16384`，审计后的 sampler-safe 路径改为从全候选均匀无放回抽 16384，避免原代码出现负 sample size；
- 若候选少于 16384，保留全部并抽额外 row 补足；
- full-val runner 在每次取 frame 时重置 NumPy seed 为 `(20260908 + int(frame_id)) & 0xffffffff`，使同一输入在 official/adapted 两组权重下选到同一批点。原始 `eval_rcnn.py` 会在函数内部重置 seed，所以本次在 dataset 取点函数内落实逐 frame seed；每帧保存实际 16384-point tensor 的 SHA-256 与 shape。

### CenterPoint

test 配置不做 fixed-count point sampling。它保留 FOV 内且位于 `[0,-40,-3,70.4,40,1]` 的点，然后以 voxel size `[0.05,0.05,0.1]` voxelize；每 voxel 最多 5 点，test 最多 40000 voxels。test 不 shuffle。

因此“文件中严格 4N/4M”与“网络内部实际张量中的有效点/voxel 数”不是同一个概念，报告时应同时说明。

## 10. 新的完整 validation 比较矩阵

每一个 arm 都必须满足 `status=PASS` 且 `frame_count=3769` 才能进入最终表。主指标为 KITTI Car AP_R40 的 3D/BEV easy、moderate、hard；CenterPoint 同时保留 Pedestrian 与 Cyclist 结果。

特别说明：本地 PointRCNN 历史 evaluator 的 `get_mAP` 默认对 41 个 precision samples 每隔 4 个取一次，得到 **AP_R11**。新的 runner 显式改为取 `precision[..., 1:41]` 的均值，即 **AP_R40**，与本地 OpenPCDet 的 `get_mAP_R40` 定义一致。旧 PointRCNN pilot 数字需按其原始 evaluator 口径标注，不能直接称为 AP_R40 或与 CenterPoint R40 混为同一指标。

计划矩阵：

- PointRCNN：11 arms，包括 A baseline，A direct official/adapted，A observed-first official/adapted，B baseline official/adapted，B direct official/adapted，B observed-first official/adapted；
- CenterPoint：9 arms，包括 A/B baseline official（以及 B baseline adapted）、A/B direct official、A/B observed-first official/adapted。

执行中的实时汇总写入 `full_val_detector_matrix.md`。旧 pilot256 的 AP 不会写进这个 full-val 表。

## 11. 原始代码与复现入口

### 原作者 PU-GCN 代码/参数

- `external/PU-GCN/Upsampling/generator.py`：PUGCN graph、NodeShuffle、coordinate residual；
- `external/PU-GCN/Upsampling/model.py`：patch normalization/inverse normalization；
- `external/PU-GCN/Common/ops.py`：feature extractor 和 up-unit；
- `external/PU-GCN/tf_lib/gcn_lib/vertex.py`：Inception DenseGCN / NodeShuffle 实现；
- `external/PU-GCN/pretrained/pu1k-pugcn/args.txt`：作者 checkpoint 的完整训练参数；
- `external/PU-GCN/pretrained/pu1k-pugcn/checkpoint`：实际 checkpoint 指针。

### 本实验协议代码

- `scripts/prepare_kitti_downsampled_x4_val.py`：Line B 的 `M=floor(N/4)`；
- `scripts/wrappers/kitti_patch_extractor.py`：2048 点 patch 选择；
- `scripts/wrappers/tf_pugcn_family_patch_infer_many.py`：一次 restore 后批量逐 patch inference；
- `scripts/wrappers/strict_x4_from_merged_raw.py`：raw candidate → exact 4× 与 intensity 1-NN；
- `scripts/prepare_centerpoint_observed_first_train.py`：N+3N / M+3M；
- `scripts/run_pointrcnn_full_train_stage.py`：PointRCNN adaptation 参数；
- `scripts/run_centerpoint_full_train.py`：CenterPoint adaptation 参数；
- `scripts/run_pointrcnn_checkpoint_split_eval.py`：显式 split、显式 LiDAR tree 的 full-val PointRCNN evaluator；
- `scripts/run_patch_causal_centerpoint_eval.py`：CenterPoint evaluator；
- `scripts/run_pugcn_detector_adaptation_full_val_20260908.sh`：从生成到 20-arm 汇总的完整 resumable 入口；
- `scripts/summarize_pugcn_detector_adaptation_full_val_20260908.py`：只接受 3769-frame PASS arm 的最终汇总器。

完整入口：

```bash
cd /home/ra87racy/projects/baseline_detectors/PointRCNN
bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh all
```

各阶段也可独立恢复：`generate_a`、`generate_b`、`prepare`、`pointrcnn`、`centerpoint`、`aggregate`。

源码交付文件：

- `pugcn_full_val_sources.tar.gz`：151 个本地源码、CUDA/C++、配置与 split 文件的逐字节快照；
- `source_manifest.json`：各文件相对路径、字节数和 SHA-256；
- `experiment_source_full.md`：实验关键脚本的完整正文；
- 快照明确代表工作区实际使用版本，不表示所有文件均与作者上游仓库未修改版本一致。

## 12. 当前运行状态

- 3769-frame Line A patch extraction：全部完成；
- Line A PU-GCN inference / strict-4N：运行中，每 64 frame 推理、校验和清理可重建的临时点云；
- Line B 生成：等待 Line A；
- full-val detector matrix：等待两条线的输入通过完整性审计；
- PointRCNN 单 frame 端到端 inference + CPU KITTI AP_R40 + 实际输入审计烟测：已 PASS。

最终结果只有在 20/20 arms 均为 3769-frame PASS 后才标记完成。

运行时进度见 `current_progress.md` 与 `current_progress.json`，由独立进度进程每 30 秒刷新。生成流程按 64-frame batch 运行，只有完成严格 4× 检查后的临时 patch/raw/merged 文件才会清理；metadata、provenance、最终 `.bin` 和代码快照保留，可重建临时结果。
