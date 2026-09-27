# Detector-Aware Controlled Upsampling：论文改进实验 Proposal

## 0. Proposal定位

工作名称：

**Detector-Aware Sparse Gated Point Injection（DAS-Gate）**

中文名称：

**面向检测器的稀疏门控点注入**

本proposal不再尝试证明“点越多检测越好”，而是回答：

> 如何只保留对检测有用的少量生成点，同时避免严格x4上采样对真实点预算、局部邻域和体素空间的污染？

它直接建立在当前实验已经发现的三个事实上：

1. PointRCNN中，PDANS 2.5%在256帧Car 3D Moderate上出现`+2.43`候选增益，但生成比例继续提高后性能下降。
2. CenterPoint Line B中，PDANS对Pedestrian、PDANS/PU-GCN对Cyclist存在全验证集正向结果。
3. 严格x4输入产生大量不受真实扫描支持的邻域或额外体素，收益与损害同时存在。

因此核心设计不是新的点生成网络，而是一个可审计的**生成点选择与检测器输入适配层**。

---

## 1. 研究问题

### RQ1

低比例生成点是否能够在完整KITTI验证集上稳定改善PointRCNN，而不是只在256帧筛选集上提高？

### RQ2

几何一致性和稀疏度门控能否保留CenterPoint对Pedestrian/Cyclist的正向收益，同时降低Car性能损失？

### RQ3

PointRCNN与CenterPoint是否需要不同的生成点预算和保留机制？

### RQ4

检测器感知选择是否优于：

- 随机生成点选择；
- 只按生成比例选择；
- 只按几何质量选择；
- 相同比例真实点替换对照？

---

## 2. 假设

### H1：低剂量假设

生成点对检测的影响存在剂量区间。少量高质量生成点可能补充被采样遗漏或下采样删除的目标证据；高比例生成点会改变邻域和体素分布并导致退化。

### H2：稀疏目标假设

正向收益主要发生在：

- Line B；
- 原始目标点数较少；
- Pedestrian/Cyclist等小目标；
- 中远距离或部分遮挡区域。

Line A完整扫描和近距离高密度Car不需要无条件增加点。

### H3：检测器机制假设

- PointRCNN需要严格控制生成点在固定16,384点预算中的比例。
- CenterPoint需要控制生成点形成的新增体素、体素内点数和体素均值偏移。

同一套raw-point筛选策略不应被假定对两个检测器等价。

### H4：匹配对照假设

若生成点方案只超过baseline、但不超过同槽位真实点对照，则增益主要来自采样覆盖变化；只有超过匹配真实点对照的部分才能作为生成几何的净贡献。

---

## 3. 方法概述

输入：

- 原始或downsampled观测点集合 `P_obs`；
- 上采样方法生成的候选点集合 `P_gen`；
- 冻结检测器；
- 不使用GT框、类别标签或评测结果进行测试时筛选。

输出：

- PointRCNN专用输入 `P_point`；
- CenterPoint专用输入 `P_voxel`；
- 每帧生成点选择manifest；
- 每个候选点的各项门控分数和拒绝原因。

总体流程：

`候选生成 → 观测/生成点精确拆分 → 几何质量过滤 → 稀疏度门控 → 可选检测器ROI门控 → 检测器专用预算 → 冻结检测器评测`

---

## 4. 候选点质量分数

为每个生成候选点 `p` 定义：

`S(p) = w_surface S_surface + w_range S_range + w_sparse S_sparse + w_roi S_roi`

第一版不训练额外网络，使用确定性规则，避免引入新的训练变量。

### 4.1 局部表面一致性 `S_surface`

1. 在观测点中查找kNN。
2. 对邻域做局部PCA平面拟合。
3. 计算候选点到局部平面的距离。
4. 拒绝平面残差过大、法向不稳定或邻域跨度过大的候选点。

建议初始参数：

- `k ∈ {8, 16, 32}`；
- 点到平面距离阈值 `{0.05, 0.10, 0.20} m`；
- 邻域最大XY直径按距离自适应。

这一项用于去除跨道路、车辆和背景表面的桥接点。

### 4.2 扫描几何一致性 `S_range`

将候选点投影到LiDAR方位角/俯仰角网格：

1. 检查相邻角度bin是否有观测支持。
2. 比较候选range与邻近观测range中位数。
3. 拒绝位于前后表面之间的悬空点。
4. 保留沿已有表面连续延伸的候选点。

这一项不需要完整原始点云作为oracle，只使用当前输入的观测点。

### 4.3 稀疏度收益 `S_sparse`

只向真正稀疏的位置增加点：

- 局部观测kNN距离较大；
- 当前体素或邻域占用低；
- 候选点不会成为已有观测点的近重复；
- 同一体素已有足够真实点时不再注入。

Line A中，大部分高密度区域应被该门控拒绝；Line B中，小目标稀疏区域得到更高优先级。

### 4.4 检测器区域分数 `S_roi`

该部分作为后续消融项，不在第一版强制启用。

测试时先使用baseline输入运行一次低阈值检测：

- PointRCNN：保留低阈值RPN proposal或foreground objectness区域；
- CenterPoint：保留低阈值center heatmap峰值区域。

只在扩张后的候选ROI内注入点，再运行第二次正式检测。整个过程不读取GT。

必须设置一个几何门控下限，避免低置信度假proposal把伪点吸入背景。

---

## 5. 两个检测器分开适配

## 5.1 PointRCNN适配器

目标：

保持16,384点输入不变，并尽量保留真实观测。

### 输入策略

1. 从canonical observed-only E2 baseline开始。
2. 固定保留比例：

   - `97.5% observed + 2.5% generated`；
   - `95% observed + 5% generated`作为次级消融；
   - 不把10%以上作为主实验。

3. generated slots只从通过门控的候选点中选择。
4. 如果合格生成点不足，剩余槽位使用真实观测点填充，不用低质量生成点强行补满。
5. 最终仍为精确16,384点。

### PointRCNN主对照

- P0：Original baseline；
- P1：observed-fill 2.5%；
- P2：PDANS random/generated 2.5%；
- P3：PDANS + geometry gate 2.5%；
- P4：PDANS + geometry + sparsity gate 2.5%；
- P5：PDANS + geometry + sparsity + RPN ROI gate 2.5%。

### PointRCNN主要研究线

- 首要：Line A，验证此前256帧正点能否推广到3769帧。
- 次要：Line B，检验门控是否至少显著缩小严格x4导致的损失。

---

## 5.2 CenterPoint适配器

目标：

控制新增体素，而不是控制raw点总数。

### 输入策略

1. 所有真实观测点原样保留。
2. 候选点先按CenterPoint真实FOV、range和voxel size过滤。
3. 每个空体素第一阶段最多选择1个生成点。
4. 已有真实观测的体素：

   - 默认不注入；
   - 或仅在真实点数小于2时允许1个高质量生成点。

5. 对生成体素设置frame-level和ROI-level预算。
6. 若预cap体素接近40,000，优先保留真实体素和高质量生成体素。

### CenterPoint主对照

- C0：Line baseline；
- C1：strict exact-4N输入；
- C2：voxel-unique generated candidates；
- C3：C2 + geometry gate；
- C4：C3 + sparsity gate；
- C5：C4 + CenterHead低阈值ROI gate；
- C6：C5 + 类别/距离自适应预算。

### CenterPoint主要研究线

以Line B为主：

- PDANS/Pedestrian；
- PDANS/Cyclist；
- PU-GCN/Cyclist；
- 同时监测Car，避免用小目标增益换取不可接受的Car损失。

Line A作为安全性测试：理想策略应接近no-op，而不是继续无条件x4。

---

## 6. 实验阶段

## Phase 0：补齐尚未完成的正对照

这一阶段不是新方法实验，而是完成当前论文证据闭环。

### PointRCNN全验证集

只运行Original Line：

1. Original baseline；
2. Original observed-fill control 2.5%；
3. Original PDANS generated 2.5%。

要求：

- 3769/3769帧；
- 完全复用256帧实验的嵌套slot order；
- BBox、BEV、3D、AOS；
- Easy、Moderate、Hard；
- 保存全部预测框和输入manifest。

决策：

- 若PDANS不超过full baseline：256帧结果属于筛选集波动或选择偏差。
- 若超过baseline但不超过control：提升主要来自采样覆盖。
- 若同时超过baseline和control：才能作为生成几何带来净增益的证据。

## Phase 1：256帧开发与消融

在已有screen256上调试DAS-Gate。

不直接使用AP选择大量阈值。推荐：

1. 先用几何指标确定2–3组候选参数；
2. 再用256帧AP进行开发选择；
3. 锁定参数后不再查看full结果进行调参。

开发阶段输出：

- 生成点保留率；
- generated 0.2m voxel precision；
- extra voxel fraction；
- reference voxel recall；
- 每目标观测/生成点数；
- PointRCNN generated-slot利用率；
- CenterPoint新增体素数。

## Phase 2：完整3769帧确认

每个检测器最多保留3–4个锁定配置，避免形成无约束实验搜索。

### PointRCNN full

- P0、P1、P2、P4、P5；
- Car；
- 两条线；
- 三种难度和BBox/BEV/3D。

### CenterPoint full

- C0、C1、C4、C5；
- Car、Pedestrian、Cyclist分别报告；
- 两条线；
- 三种难度和BBox/BEV/3D。

## Phase 3：框级机制复核

继续使用当前三帧：

- `000590`；
- `005625`；
- `006682`。

对baseline、strict x4、DAS-Gate三者进行：

- 完整3D/BEV对比；
- lost/recovered/localization/confidence迁移；
- 单目标crop；
- 观测点、被拒绝生成点、被接受生成点三色可视化。

---

## 7. 消融矩阵

| ID | Observed-first | Fixed low ratio | Geometry | Sparsity | Detector ROI | Purpose |
|---|---:|---:|---:|---:|---:|---|
| A0 | — | — | — | — | — | Line baseline |
| A1 | ✓ | — | — | — | — | strict/current adapter reference |
| A2 | ✓ | ✓ | — | — | — | 低比例本身 |
| A3 | ✓ | ✓ | ✓ | — | — | 几何质量贡献 |
| A4 | ✓ | ✓ | ✓ | ✓ | — | 稀疏区域选择贡献 |
| A5 | ✓ | ✓ | ✓ | ✓ | ✓ | 检测器感知贡献 |
| A6 | ✓ | ✓ | ✓ | ✓ | ✓ | 类别/距离预算 |

必须保留A2，否则无法判断改进来自“比例降低”还是来自门控机制。

---

## 8. 评测指标

## 8.1 检测指标

两个检测器必须分开报告。

### PointRCNN

- Car BBox AP_R40 Easy/Moderate/Hard；
- Car BEV AP_R40 Easy/Moderate/Hard；
- Car 3D AP_R40 Easy/Moderate/Hard；
- AOS；
- recall@0.3/0.5/0.7；
- lost/recovered/localization/confidence迁移。

### CenterPoint

Car、Pedestrian、Cyclist分别报告：

- BBox AP_R40 Easy/Moderate/Hard；
- BEV AP_R40 Easy/Moderate/Hard；
- 3D AP_R40 Easy/Moderate/Hard；
- recall@0.3/0.5/0.7；
- 每类预测数和score≥0.5预测数；
- lost/recovered/localization/confidence迁移。

## 8.2 几何与输入机制指标

- candidate acceptance ratio；
- accepted generated point count；
- accepted generated voxel count；
- generated 0.2m voxel precision；
- reference 0.2m voxel recall；
- extra voxel fraction；
- mixed observed/generated voxel fraction；
- PointRCNN实际generated slot ratio；
- CenterPoint pre-cap voxel count和cap hit；
- 每距离段和每类别的目标内点数变化。

## 8.3 计算成本

- 门控耗时/帧；
- 第一次与第二次检测耗时；
- 峰值内存；
- 最终输入点/体素数。

---

## 9. 统计与可重复性

1. 所有选择默认使用固定seed和稳定hash。
2. 如果存在随机采样，对最终候选配置至少运行3个seed。
3. 使用frame-level paired bootstrap重新计算AP差值95%置信区间。
4. full参数必须在256帧阶段锁定。
5. 每个输出保存：

   - 源点云路径；
   - observed/generated精确关系；
   - 每个候选点分数；
   - 接受/拒绝原因；
   - 最终点数；
   - detector config hash；
   - code commit或工作树状态。

---

## 10. 成功判据

## 10.1 最低科学成功

即使没有超过baseline，只要满足以下条件，仍然是有效论文结果：

- 相比strict x4显著减少AP损失；
- 几何指标和框级迁移同步改善；
- 消融能够证明geometry/sparsity/detector gate各自的作用；
- 两种检测器表现出可解释的不同最优策略。

## 10.2 PointRCNN强成功

优先判据：

- full 3769帧Original/PDANS g2.5超过baseline；
- 并尽可能超过matched observed-fill control；
- Easy/Moderate/Hard中至少两个难度方向一致；
- BBox/BEV没有出现与3D增益矛盾的大幅退化。

## 10.3 CenterPoint强成功

- 保持或扩大Line B Pedestrian/Cyclist正增益；
- 同时将Car相对strict x4的损失减少至少30%；
- accepted generated voxel precision明显高于未过滤输入；
- extra voxel fraction明显下降。

## 10.4 安全性判据

- Line A不应继续无条件增加大量生成体素；
- 建议将Line A 3D Moderate相对baseline退化控制在1 AP以内；
- 若候选质量不足，算法允许返回observed-only输入。

---

## 11. 代码实施建议

建议新增，不修改旧实验结果：

### `scripts/build_detector_aware_point_inputs.py`

职责：

- 精确拆分observed/generated；
- 计算surface/range/sparsity/ROI分数；
- 分别构建PointRCNN与CenterPoint输入；
- 输出per-frame manifest。

主要接口：

```text
--detector pointrcnn|centerpoint
--line A|B
--method pdans|pu_gcn
--policy quota|geometry|geometry_sparse|geometry_sparse_roi
--generated-ratio 0.025
--split-file ...
--output-dir ...
```

### `scripts/run_detector_aware_upsampling_ablation.py`

职责：

- 运行256帧开发矩阵；
- 锁定配置；
- 运行full 3769帧；
- 跳过已有PASS结果；
- 保持两个检测器输出目录隔离。

### `scripts/analyze_detector_aware_upsampling.py`

职责：

- 解析BBox/BEV/3D/AOS和三种难度；
- 计算框状态迁移；
- 汇总几何、体素和点预算；
- 生成PointRCNN与CenterPoint独立报告。

### 建议结果目录

```text
results/detector_aware_controlled_upsampling_YYYYMMDD/
├── pointrcnn/
│   ├── inputs/
│   ├── manifests/
│   ├── eval_outputs/
│   ├── tables/
│   ├── figures/
│   └── report.md
└── centerpoint/
    ├── inputs/
    ├── manifests/
    ├── eval_outputs/
    ├── tables/
    ├── figures/
    └── report.md
```

---

## 12. PU-Net和共同patch修复与本proposal的关系

PU-Net修复不是DAS-Gate第一阶段的阻塞项，因为第一阶段可以使用当前质量最好的PDANS和具有Cyclist正向结果的PU-GCN。

但如果论文要声称“四种上采样方法的公平比较”，则必须二选一：

1. 修复PU-Net归一化/逆变换和共同patch局部性后重新评测；
2. 将当前PU-Net明确标为`adapter-defective diagnostic result`，从公平模型排名主表中移出。

共同patch的FPS+kNN修复至少应在PU-GCN或PU-EdgeFormer上做一组消融，用来回答：

> 当前退化究竟来自上采样模型本身，还是来自非局部KITTI patch适配？

因此它对“方法公平性结论”是必要的，对“先验证DAS-Gate能否改善PDANS/PU-GCN输入”不是前置阻塞。

---

## 13. 论文可形成的最终贡献

如果实验成功，论文可以形成三个层次的贡献：

1. **系统发现**：严格点数上采样不等价于有效检测证据增加。
2. **机制证据**：PointRCNN受固定点预算和邻域污染影响，CenterPoint受额外体素和体素特征偏移影响。
3. **方法贡献**：提出DAS-Gate，只在可信、稀疏且检测相关的位置注入少量生成点。

即使最终没有全面超过baseline，DAS-Gate只要显著减少strict x4损失，并通过消融证明原因，仍然能够把论文从“失败现象报告”推进为“可验证的机制与修正方案”。

