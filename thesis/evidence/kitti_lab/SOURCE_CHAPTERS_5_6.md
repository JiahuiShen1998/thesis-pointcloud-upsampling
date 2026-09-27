# 第五章与第六章：KITTI 结果、讨论、结论与展望

> **文稿边界与引用规则。** 本稿续接现有论文第 1–4 章，专门整理 KITTI 部分的第 5、6 章。学术文献编号沿用 `thesis.pdf` 的 [1]–[35]，不重排；本稿没有为了增加篇幅而新增无法核验的论文。实验配置、CSV、审计报告与可视化产物以 [L14]–[L20] 单列为“本地实验依据”，它们是结果可追踪证据，不冒充公开文献。不同规模的实验严格区分：3769 帧为全验证集主结果，256 帧为固定适配/目标级审计集，64 帧和 32 帧仅用于机制分析。除特别注明外，检测 AP 均为 KITTI `AP_R40`，Car 的 3D IoU 阈值为 0.70。PU-Net 当前 KITTI 包装器存在已确认的尺度归一化错误，因此带星号的结果只能诊断当前流水线，不能用于评价 PU-Net 架构本身。

## 结果阅读所需符号

| 符号 | 含义 |
|---|---|
| \(s\) | KITTI 帧标识 |
| \(l\in\{A,B\}\) | 输入线；A 为完整扫描直接增密，B 为四分之一稀疏扫描再恢复 |
| \(m\) | 上采样方法，主要包括 PDANS、PU-GCN、PU-EdgeFormer 和 PU-Net* |
| \(d\) | 下游检测器，\(d\in\{\mathrm{PointRCNN},\mathrm{CenterPoint}\}\) |
| \(\mathcal X_s^{(l)}\) | 帧 \(s\) 在输入线 \(l\) 下可见的观测点集 |
| \(\mathcal G_{s,m}^{(l)}\) | 方法 \(m\) 产生并通过统一筛选的生成点 |
| \(\mathcal Y_{s,m}^{(l)}\) | 最终 exact-\(4N\) 点集，主协议为 \(N\) 个观测点加 \(3N\) 个生成点 |
| \(T_{d,m}^{(l)}\) | 检测器 \(d\) 在方法 \(m\)、输入线 \(l\) 下的任务指标 |
| \(\Delta T_{d,m}^{(l)}\) | 相对同一输入线真实基线的百分点变化 |
| \(\tau\) | 几何邻近阈值；体素分析使用 0.20 m，车辆近表面分析使用 0.25 m |

# 5. Results and Evaluation

本章回答的核心问题不是“输出文件是否包含四倍点”，而是四倍点是否形成了检测器可以利用的正确几何证据。结果按照“评价定义—全量任务结果—受控消融—点云案例—机制解释”的顺序组织。该顺序尤其重要：单幅点云图能够解释一个案例，却不能替代全数据集 AP；单个 AP 数字能够描述最终性能，却不能独立定位性能变化发生在哪个环节。

## 5.1 Overview of Experiments

### 5.1.1 评价对象、两条输入线与证据层级

Line A 与 Line B 检验两个不同命题。对原始 KITTI 扫描 \(\mathcal X_s^{\mathrm{orig}}\)，Line A 直接生成四倍点：

\[
\mathcal X_s^{(A)}=\mathcal X_s^{\mathrm{orig}},\qquad
\mathcal Y_{s,m}^{(A)}
=\mathcal X_s^{(A)}\cup\mathcal G_{s,m}^{(A)},\qquad
|\mathcal G_{s,m}^{(A)}|=3|\mathcal X_s^{(A)}|.
\tag{5.1}
\]

Line A 不存在“恢复被删除观测”的空间；如果其 AP 上升，只能解释为生成点提供了超过原始扫描的新任务证据。若 AP 下降，则说明新增点改变了点采样、局部邻域或体素占据，使有效信号被稀释。

Line B 先通过固定下采样算子 \(D_4\) 构造四分之一稀疏输入，再恢复到原始点数规模：

\[
\mathcal X_s^{(B)}=D_4(\mathcal X_s^{\mathrm{orig}}),\qquad
|\mathcal X_s^{(B)}|\simeq\frac14|\mathcal X_s^{\mathrm{orig}}|,
\tag{5.2}
\]

\[
\mathcal Y_{s,m}^{(B)}
=\mathcal X_s^{(B)}\cup\mathcal G_{s,m}^{(B)},\qquad
|\mathcal Y_{s,m}^{(B)}|=4|\mathcal X_s^{(B)}|.
\tag{5.3}
\]

因此，Line B 的正确比较对象是同一稀疏输入的 detector baseline，而不是 Line A 原始扫描。它回答的是“上采样能否追回稀疏化造成的任务损失”。Line A 与 Line B 的 AP 不能取平均后当作单一总分。

本章使用五层证据。R1 与 R2 是用于主结论的全验证集评价；R3 观察检测器适配能否缓解域偏移；R4 与 R5 解释机制，不能代替正式 AP。

| 层级 | 数据规模 | 比较对象 | 主要用途 | 结论强度 |
|---|---:|---|---|---|
| R1 | 3769 帧 | PointRCNN；4 方法；Line A/B；E1/E2 | 全量 Car AP 与输入预算敏感性 | 主结果 |
| R2 | 3769 帧 | CenterPoint；4 方法；Line A/B；exact-\(4N\) | 全量三类别 AP 与体素检测对照 | 主结果 |
| R3 | 256 固定评价帧；3712 训练帧 | PU-GCN；PointRCNN/CenterPoint；适配前后 | 检查检测器再训练是否解决分布偏移 | 强补充证据 |
| R4 | 64 个含车帧 | patch 跨度、生成体素精度、距离分层 | 定位输入几何问题 | 机制证据 |
| R5 | 256 帧、527 个 Moderate Car GT；另有 32 帧体素审计 | 目标 IoU、TP/FN 转换、体素上限 | 解释恢复和失败个例 | 诊断证据 |

### 5.1.2 AP、IoU 与差值的完整定义

给定预测框 \(B_p\) 与真实框 \(B_g\)，三维交并比为

\[
\operatorname{IoU}_{3D}(B_p,B_g)
=\frac{\operatorname{Vol}(B_p\cap B_g)}
{\operatorname{Vol}(B_p\cup B_g)}
=\frac{\operatorname{Vol}(B_p\cap B_g)}
{\operatorname{Vol}(B_p)+\operatorname{Vol}(B_g)-\operatorname{Vol}(B_p\cap B_g)}.
\tag{5.4}
\]

对置信度阈值 \(q\)，精确率与召回率分别为

\[
P(q)=\frac{TP(q)}{TP(q)+FP(q)},\qquad
R(q)=\frac{TP(q)}{TP(q)+FN(q)}.
\tag{5.5}
\]

采用 KITTI R40 评价时，先计算插值精确率

\[
P_{\mathrm{interp}}(r)
=\max_{\tilde r\ge r}P(\tilde r),
\tag{5.6}
\]

再在固定的 40 个召回位置上取平均：

\[
AP_{R40}=\frac1{40}\sum_{k=1}^{40}
P_{\mathrm{interp}}\!\left(\frac{k}{40}\right).
\tag{5.7}
\]

这种定义承接 KITTI/VOC 的精确率—召回率评价传统 [1,2,34,35]。本章同时报告 3D AP 与 BEV AP：3D AP 要求水平位置、高度、尺寸与朝向共同正确；BEV AP 只在鸟瞰平面评价旋转框。若 BEV 和 3D 同时下降，问题一般不只是高度回归；若二者差距明显扩大，则需额外检查 \(z\) 轴与高度估计。

所有“改善”均以同线基线为零点：

\[
\Delta AP_{d,m,c,h}^{(l)}
=AP_{d,m,c,h}^{(l)}-AP_{d,\mathrm{base},c,h}^{(l)},
\tag{5.8}
\]

其中 \(c\) 为类别，\(h\in\{\mathrm{Easy},\mathrm{Moderate},\mathrm{Hard}\}\)。正值表示超过基线，负值表示退化。对于 Line B，还定义相对原始—稀疏性能缺口的恢复率：

\[
R_{d,m,c,h}^{(B)}
=\frac{AP_{d,m,c,h}^{(B)}-AP_{d,\mathrm{sparse},c,h}^{(B)}}
{AP_{d,\mathrm{orig},c,h}^{(A)}-AP_{d,\mathrm{sparse},c,h}^{(B)}}.
\tag{5.9}
\]

\(R=1\) 表示完全追回稀疏化损失，\(R=0\) 表示与稀疏基线相同，\(R<0\) 表示上采样后反而低于稀疏基线。恢复率只在分母为正且两项使用同一检测器与评价实现时解释。

### 5.1.3 完整性检查与结论纪律

PointRCNN 全验证集 E1/E2 的每个实验均产生 3769 个预测文件；CenterPoint exact-\(4N\) 的 10 组主输入全部通过帧数、有限坐标、点数和 evaluator 完整性检查 [L14,L15]。这排除了“少跑了部分帧”作为 AP 下降的解释，但没有自动证明适配器正确。为避免由工程缺陷过度外推，本章遵守以下规则：

1. 只在相同数据划分、IoU 阈值、评价器和基线口径内计算差值。
2. 3769 帧结果用于总体结论；256 帧结果用于适配与对象诊断，不写成全验证集结论。
3. 64 帧 patch/几何结果说明机制是否存在，不用于估计全数据集效应量。
4. 32 帧体素审计只回答“是否触发上限以及输入顺序是否可能有影响”。
5. PU-Net* 的数值保留用于证明当前包装链可能严重失败，但不与其他正确适配的方法作架构优劣结论。

## 5.2 ModelNet40 Classification Results

本节标题为保持原论文目录连续性而保留，但本稿不重复撰写 ModelNet40 数值结果。原因是本次任务要求整理 KITTI 部分，而且当前可核验的完整证据链集中于 LiDAR 检测。第五章跨域讨论只引用第 3、4 章已经定义的 ModelNet40 对照逻辑，不补造缺失的分类准确率。最终合并论文时，应将已有的 ModelNet40 5.2 正文置于此处，并保持 [1]–[35] 的参考文献编号不变。

这个边界本身影响结论措辞：本章能够证明当前“物体级上采样器—KITTI 场景适配器—检测器”组合的行为，却不能仅凭 KITTI 负结果判断某一网络在规范化 CAD 表面上的能力，也不能反向用 ModelNet40 分类改善替代真实 LiDAR 检测证据。

## 5.3 KITTI PointRCNN Detection Results

PointRCNN 直接从点云产生三维 proposals [11]，因此对点的采样预算、局部分组和伪点比例较敏感。本节先报告全验证集四方法结果，再给出检测器适配、观测保留和对象级诊断。不同实验的绝对 AP 受权重和输入适配差异影响，跨表读取时应比较各自的同线差值，不直接比较不同行的裸数值。

### 5.3.1 全验证集 E1：保留观测的 exact-\(4N\) 结果

E1 的输出结构为 \(N\) 个观测点与 \(3N\) 个生成点。所有输入真实点逐点保留，因而 E1 专门检验“保留真实点是否足以避免性能退化”。表 5.1 给出 3769 帧 Car 3D AP\(_{R40}\)。

**表 5.1　PointRCNN 全验证集 E1 Car 3D AP\(_{R40}\)（%）**

| 输入线 | 方法 | Easy | Moderate | Hard | 相对同线 baseline 的 Moderate 差值 |
|---|---|---:|---:|---:|---:|
| A | Original baseline（E2 参考） | 92.273 | 82.255 | 77.945 | 0.000 |
| A | PDANS | 83.014 | **64.515** | 59.668 | -17.741 |
| A | PU-GCN | 78.766 | 56.961 | 52.193 | -25.294 |
| A | PU-EdgeFormer | 64.258 | 42.981 | 38.259 | -39.274 |
| A | PU-Net* | 14.414 | 9.717 | 9.010 | -72.538 |
| B | 1/4 sparse baseline（E2 参考） | 85.177 | 65.746 | 61.382 | 0.000 |
| B | PDANS | 65.062 | **45.126** | 39.240 | -20.620 |
| B | PU-GCN | 45.378 | 29.687 | 25.290 | -36.059 |
| B | PU-EdgeFormer | 34.010 | 20.721 | 17.747 | -45.025 |
| B | PU-Net* | 12.522 | 8.878 | 8.067 | -56.868 |

四个直接可见的结论如下。

第一，Line A 的所有方法都低于原始扫描 baseline。原始扫描已经包含评价时可用的真实观测，生成点没有“恢复”对象；最好的 PDANS 仍下降 17.741 个 Moderate AP。这否定了“只要点更密，点式检测器就会单调变好”的假设。

第二，Line B 中最好的 PDANS 也比稀疏 baseline 低 20.620 AP；按式（5.9）计算，其恢复率为负值。这说明输出回到约原始点数，不等价于恢复原始扫描的信息。点数守恒只约束 \(|\mathcal Y|\)，没有约束 \(\mathcal G\) 是否落在缺失的真实表面。

第三，两条线的方法排序完全一致：PDANS > PU-GCN > PU-EdgeFormer > PU-Net*。该排序稍后会与生成体素真实精度逐一对应。它比“某一方法掉了多少 AP”更有机制意义，因为两种起点和同一检测器下都出现相同秩序。

第四，Hard 目标与 Moderate 目标都显著下降，并非只在少量 Easy 案例出现波动。与此同时，PU-Net* 的极低 AP 受到确定的适配器错误影响：米制 patch 使用 `bradius=1.0` 直接送入按归一化物体训练的网络，没有执行相同中心化/尺度归一化及逆变换 [L17]。因此该行证明的是包装协议可以摧毁检测结果，不是 PU-Net [4] 的架构性能上限。

### 5.3.2 E2 固定 16384 点：排除“PointRCNN 只是不接受更多点”

E2 对所有条件执行统一视场过滤、0.1 m 体素代表点和分层距离采样，最终保存恰好 16384 点。若 E1 的主要问题只是 PointRCNN 固定点预算，那么 E2 应系统性恢复 AP。表 5.2 显示这一预期并未出现。

**表 5.2　PointRCNN E1 与 E2 的 Moderate 3D AP\(_{R40}\) 对照（%）**

| 输入线 | 方法 | E1：\(N+3N\) | E2：统一 16384 | E2−E1 |
|---|---|---:|---:|---:|
| A | PDANS | 64.515 | 67.223 | +2.709 |
| A | PU-GCN | 56.961 | 58.276 | +1.314 |
| A | PU-EdgeFormer | 42.981 | 44.960 | +1.979 |
| A | PU-Net* | 9.717 | 8.380 | -1.337 |
| B | PDANS | 45.126 | 44.075 | -1.051 |
| B | PU-GCN | 29.687 | 28.680 | -1.006 |
| B | PU-EdgeFormer | 20.721 | 20.569 | -0.151 |
| B | PU-Net* | 8.878 | 8.781 | -0.097 |

Line A 的三个可用适配方法仅恢复 1.3–2.7 AP，远小于它们相对 baseline 的 17.7–39.3 AP 缺口；Line B 则全部没有改善。更直接的反证来自独立体素数：E2 原始 baseline 的 0.1 m 独立体素中位数为 11116，Moderate AP 为 82.255；Line B PU-GCN 有 13660 个独立体素，AP 却只有 28.680；PU-EdgeFormer 有 14321 个独立体素，AP 为 20.569。更多的独立位置没有转化为更高 AP，表明决定性变量是位置是否正确，而非点或体素是否足够多。

### 5.3.3 全数据适配的 PU-GCN：域偏移可以缓解，但不能消除

为检验冻结检测器是否把训练域差异放大，进一步使用 3712 个训练帧对检测器进行完整输入域适配，并在固定的 256 帧上评价。表 5.3 同时给出 generated-only 与 observed-first。这里的“generated-only”表示用上采样输出直接构造检测输入；“observed-first”在输出顺序中优先保留真实观测，使后续有限预算操作先消费观测点。

**表 5.3　PointRCNN 全训练适配后固定 256 帧 Car 结果（%）**

| Line | 输入 | Easy 3D | Moderate 3D | Hard 3D | Moderate BEV | 相对 baseline 的 Moderate 3D 差值 |
|---|---|---:|---:|---:|---:|---:|
| A | baseline | 89.701 | 79.191 | 77.882 | 87.513 | 0.000 |
| A | PU-GCN generated-only | 83.774 | 67.235 | 59.934 | 76.924 | -11.957 |
| A | PU-GCN observed-first | 86.372 | **67.791** | 65.830 | 78.576 | -11.400 |
| B | baseline | 86.899 | 67.249 | 59.831 | 77.679 | 0.000 |
| B | PU-GCN generated-only | 67.484 | 47.020 | 41.123 | 57.755 | -20.228 |
| B | PU-GCN observed-first | 76.757 | **56.244** | 50.067 | 66.614 | -11.004 |

![图 5.1　PointRCNN 上 PU-GCN 的 baseline、generated-only 与 observed-first AP。](figures/fig5_01_pointrcnn_pugcn_ap.pdf)

Line B 中 observed-first 相对 generated-only 恢复 9.224 个 Moderate 3D AP，说明真实点确实会在有限输入预算下与生成点竞争；Line A 只恢复 0.556 AP，说明观测顺序不是 Line A 损失的主要来源。更重要的是，两个 observed-first 结果仍分别比 baseline 低 11.400 和 11.004 AP。因此，“保留并优先使用真实点”是必要的工程约束，但不是几何可靠性的充分条件。

该结果也避免了另一个误读：适配训练确实可以学会一部分输入分布变化，却没有把负差值变为零。因而剩余差距不能全部归咎于检测器从未见过增密输入；至少还有一部分损失来自生成几何、patch 构造或无法由有限训练数据吸收的统计偏移。

### 5.3.4 64 帧再训练筛查：收益与基线退化同时存在

较早的 64 帧微调实验提供了方向性对照 [L20]。在相同 256 帧上，PU-GCN 的 Line A Moderate AP 从 60.266 增至 64.441（+4.175），Line B 从 32.920 增至 37.515（+4.595）；PDANS 的 Line B 从 43.007 增至 47.433（+4.426）。然而，同样的微调使 Line A baseline 下降 1.148 AP、Line B baseline 下降 3.705 AP；所有上采样条件仍低于各自微调后的 baseline。

因此这组结果不能写成“再训练解决了问题”。它只能支持两个较窄的判断：其一，检测器对生成点分布具有可学习的适应空间；其二，小样本微调本身会引入方差和基线退化，必须通过全训练集适配、独立验证集和同条件 baseline 控制。正因为 64 帧结果存在这两个方向，最终结论以 3712 帧适配实验为准。

### 5.3.5 对象级 IoU 与距离分层

在固定 256 帧中，共筛得 527 个符合 Moderate 条件的 Car GT；距离分层为 0–20 m：170 个，20–40 m：272 个，40 m 以上：85 个。对象诊断使用同类别、旋转 3D IoU 的贪心匹配，阈值为 0.70。它用于解释 TP/FN 迁移，不替代按置信度积分的官方 AP。

PointRCNN Line B baseline、generated-only 与 observed-first 分别匹配 386、258 和 320 个目标，对应诊断召回 73.24%、48.96% 和 60.72%。generated-only 相对 baseline 新增 134 个 `TP→FN`；observed-first 又将其中 80 个 `FN→TP`，但仍有 78 个 baseline TP 在 observed-first 下变成 FN。这个流向说明 observed-first 的总体改善并不是所有目标的小幅提升，而是恢复一部分目标的同时仍破坏另一部分目标。

距离进一步解释差距来源。Line B baseline 在 0–20、20–40、40+ m 的对象召回分别为 97.65%、71.32% 和 30.59%；observed-first PU-GCN 分别为 96.47%、53.31% 和 12.94%。近距目标几乎保持，而中距下降 18.01 个百分点，远距下降 17.65 个百分点。生成点的风险因此不是均匀分布的：观测越稀疏、遮挡越强，局部 patch 越难保持单一表面，新增点越可能改变原本接近阈值的框。

### 5.3.6 点云场景与局部恢复案例

图 5.9 展示同一 KITTI 帧 000104 的四种点云：原始扫描 121994 点、四倍下采样 30498 点、Line B PDANS exact-\(4N\) 121992 点和 Line B PU-GCN observed-first 121992 点。图中只叠加 GT 框，不叠加方法预测，目的是观察点分布而不是用挑选的预测框代替总体评价。

全景图说明两个问题。首先，exact-\(4N\) 的行数检查在视觉上确实把整体密度恢复到接近原始输入；其次，远距离和物体边界处的空间支持并没有因此自动恢复。大面积道路背景能够吸收大量点，使“全帧看起来更密”，而车辆目标内部或轮廓附近仍可能缺失正确表面。

图 5.10 给出一个恢复案例。帧 000440 的 Car GT 1 距 LiDAR 31.2 m。Line B baseline 在 GT 内有 26 个点，得到 TP、IoU 0.867；generated-only 虽在 GT 内增加到 73 个点，却没有产生有效匹配（FN，最佳 IoU 0）；observed-first 有 81 个框内点并恢复 TP，IoU 0.806。

这个案例支持“观测保留能够救回部分目标”，但同时给出严格边界：81 个点对应的 IoU 仍低于 26 个真实稀疏点的 baseline IoU。增加点数帮助框重新跨过 0.70 阈值，却没有恢复到基线定位精度。它与表 5.3 中“AP 恢复但仍低于 baseline”的总体结果一致。

图 5.11 是互补的失败案例。帧 004846 的 Car GT 2 距 LiDAR 38.2 m，baseline 仅 20 个框内点却得到 IoU 0.880；generated-only 和 observed-first 分别有 28 与 43 个框内点，但都为 FN。

因此，框内点计数 \(n_{\mathrm{in}}\) 不能单独作为上采样质量指标。检测器依赖点在可见表面、边缘与局部邻域中的组织方式。更恰当的解释变量应至少包含点到参考表面的距离、额外体素、目标边界外壳比例和观测/生成来源，而不是把“GT 框里有更多点”直接等价为“目标信息更多”。

## 5.4 KITTI CenterPoint Detection Results

CenterPoint [12] 与 PointRCNN 的输入响应不同。它先把点按固定网格体素化，再在鸟瞰特征图上预测目标中心；同一体素中的重复点可能被聚合，而少量跨越体素边界的伪点会创建新的 BEV 激活。使用第二个检测器的目的不是寻找一个对结果更“有利”的评价器，而是检验同一上采样几何是否在点式和体素式表示下呈现一致方向。

### 5.4.1 原始与稀疏基线的全验证集复现

CenterPoint 主实验使用完整 3769 帧验证集、冻结的官方检测器、原生体素化器和相同 score/NMS 配置。10 个输入组全部通过完整性检查 [L15]。原始扫描与四倍稀疏扫描的基线如表 5.4。

**表 5.4　CenterPoint 全验证集基线 3D AP\(_{R40}\)（%）**

| 输入 | 类别 | Easy | Moderate | Hard | 稀疏化造成的 Moderate 变化 |
|---|---|---:|---:|---:|---:|
| Original | Car | 88.391 | 79.277 | 76.737 | — |
| 1/4 sparse | Car | 81.400 | 64.598 | 59.970 | -14.679 |
| Original | Pedestrian | 54.037 | 50.653 | 46.252 | — |
| 1/4 sparse | Pedestrian | 27.196 | 24.569 | 21.673 | -26.084 |
| Original | Cyclist | 79.244 | 64.605 | 60.990 | — |
| 1/4 sparse | Cyclist | 25.805 | 15.458 | 14.492 | -49.148 |

稀疏化对三类目标的影响显著不同。Car Moderate 下降 14.679 AP，而 Pedestrian 和 Cyclist 分别下降 26.084 与 49.148 AP。小目标拥有更少有效回波，删除同样比例的点会更快地破坏局部形状与中心热图证据。因此，类别平均值会掩盖最重要的现象；后续必须按类别报告。

### 5.4.2 exact-\(4N\) observed-first 的三类别主结果

表 5.5 给出 observed-first 主协议的 Moderate 3D AP。括号内为相对该输入线真实 baseline 的差值，Line A 对应 original baseline，Line B 对应 1/4 sparse baseline。

**表 5.5　CenterPoint exact-\(4N\) observed-first：Moderate 3D AP\(_{R40}\)（%）**

| Line | 方法 | Car | Pedestrian | Cyclist |
|---|---|---:|---:|---:|
| A | PDANS | **64.466** (-14.812) | **45.251** (-5.403) | **46.103** (-18.503) |
| A | PU-GCN | 59.932 (-19.346) | 44.504 (-6.149) | 45.597 (-19.009) |
| A | PU-EdgeFormer | 47.248 (-32.030) | 37.085 (-13.569) | 35.822 (-28.784) |
| A | PU-Net* | 14.988 (-64.290) | 21.687 (-28.967) | 24.396 (-40.210) |
| B | PDANS | **46.751** (-17.847) | **28.201** (+3.632) | 15.519 (+0.061) |
| B | PU-GCN | 39.070 (-25.528) | 24.961 (+0.392) | **16.846** (+1.388) |
| B | PU-EdgeFormer | 24.748 (-39.850) | 15.596 (-8.973) | 7.037 (-8.420) |
| B | PU-Net* | 11.718 (-52.880) | 11.916 (-12.653) | 5.017 (-10.440) |

![图 5.2　CenterPoint exact-4N observed-first 相对同线 baseline 的 Moderate 3D AP 差值；括号为绝对 AP。](figures/fig5_02_centerpoint_delta_heatmap.pdf)

Line A 的 12 个“方法×类别”单元全部为负，表明当完整扫描已经存在时，当前四种场景适配均未提供净任务收益。PDANS 对 Pedestrian 的损失最小（-5.403），但仍不是改善。这个结果与 PointRCNN Line A 的方向一致：新增点不仅可能冗余，还会通过新体素、聚合统计和训练分布偏移改变检测特征。

Line B 出现三个小幅正值：PDANS 对 Pedestrian +3.632、对 Cyclist +0.061；PU-GCN 对 Pedestrian +0.392、对 Cyclist +1.388。它们证明“稀疏恢复在部分小类别上并非完全不可能”。然而，这些改善要放回稀疏化缺口中解释。按式（5.9），PDANS 仅恢复 Pedestrian 缺口的

\[
R_{\mathrm{PDANS,Ped}}^{(B)}
=\frac{28.201-24.569}{50.653-24.569}
=0.1393,
\tag{5.10}
\]

即约 13.9%；对 Cyclist 的恢复率约为 0.12%。PU-GCN 对 Pedestrian 与 Cyclist 的恢复率分别约 1.5% 与 2.8%。与此同时，两种方法的 Car 分别下降 17.847 和 25.528 AP。因而本结果最多支持“类别相关的局部恢复”，不能写成总体检测性能得到恢复。

### 5.4.3 reconstructed E1 与 observed-first：输入顺序的受控消融

CenterPoint 对每个体素最多保留 5 个点，每帧最多保留 40000 个非空体素。设有效范围为

\[
\Omega=[0,70.4)\times[-40,40)\times[-3,1),
\tag{5.11}
\]

体素大小为 \((v_x,v_y,v_z)=(0.05,0.05,0.10)\) m。任一点 \(\mathbf p_i=(x_i,y_i,z_i)\in\Omega\) 的离散索引为

\[
\mathbf q_i=\left(
\left\lfloor\frac{x_i-x_{\min}}{v_x}\right\rfloor,
\left\lfloor\frac{y_i-y_{\min}}{v_y}\right\rfloor,
\left\lfloor\frac{z_i-z_{\min}}{v_z}\right\rfloor
\right).
\tag{5.12}
\]

当输入产生的非空体素集合 \(\mathcal V_s\) 满足 \(|\mathcal V_s|>K_{\max}=40000\) 时，有限体素预算意味着部分体素不会进入网络；当某一体素点数大于 \(T_{\max}=5\) 时，点的排列也可能改变被保留的子集。observed-first 把观测点置于生成点之前，其作用可以写成优先选择：

\[
\mathcal S_v^{\mathrm{obs}}
=\operatorname{First}_{T_{\max}}
\bigl([\mathcal X_v;\mathcal G_v]\bigr),
\qquad
\mathcal S_v^{\mathrm{ctrl}}
=\operatorname{First}_{T_{\max}}
\bigl([\mathcal G_v;\mathcal X_v]\bigr).
\tag{5.13}
\]

表 5.6 汇总 observed-first 相对配对 control 的三类别 Moderate AP 平均变化。这里比较的是完全相同点集、仅顺序不同的输入，因此比“上采样 vs baseline”更接近顺序机制的因果消融。

**表 5.6　CenterPoint observed-first 相对配对 control 的平均 Moderate AP 变化**

| Line | PDANS | PU-GCN | PU-EdgeFormer | PU-Net* |
|---|---:|---:|---:|---:|
| A | +0.087 | +0.894 | +0.792 | +5.878 |
| B | +0.007 | +0.010 | +0.002 | -0.001 |

32 帧体素审计中，原始输入 0/32 帧达到 40000 体素上限；Line A 的 PDANS、PU-GCN、PU-EdgeFormer、PU-Net* 分别为 6/32、29/32、29/32、32/32；Line B 四个方法均为 0/32。顺序收益与上限触发具有清晰对应：Line A 有可见改善，Line B 几乎不变。这支持以下限定结论：observed-first 主要修复有限体素预算下的观测覆盖问题；在没有触发体素上限的 Line B，它不能纠正生成点的坐标错误。

PU-Net* 在 Line A 的 +5.878 AP 不能解释成方法优越。恰恰相反，它的错误几何几乎每帧触发体素上限，顺序干预只是减少最坏输入破坏；其绝对 AP 仍远低于 baseline。

### 5.4.4 适配后的 PU-GCN 与检测器依赖

固定 256 帧、全训练集适配的 CenterPoint 结果见表 5.7。与 PointRCNN 一样，Line A baseline 使用官方权重，Line B baseline 与上采样条件使用完整适配权重 [L14]。

**表 5.7　CenterPoint 全训练适配后固定 256 帧 Car 结果（%）**

| Line | 输入 | Easy 3D | Moderate 3D | Hard 3D | Moderate BEV | 相对 baseline 的 Moderate 差值 |
|---|---|---:|---:|---:|---:|---:|
| A | baseline | 93.635 | 79.837 | 77.357 | 88.962 | 0.000 |
| A | PU-GCN observed-first | 89.439 | 73.947 | 72.663 | 83.628 | -5.890 |
| B | baseline | 83.595 | 65.550 | 61.865 | 78.525 | 0.000 |
| B | PU-GCN observed-first | 80.731 | 60.698 | 56.695 | 73.508 | -4.852 |

相同 256 帧上，PointRCNN 的 Line A/B 差值为 -11.400/-11.004 AP，CenterPoint 为 -5.890/-4.852 AP。

![图 5.3　全训练适配后，observed-first PU-GCN 在两种检测器上的 Car Moderate AP 差值。](figures/fig5_03_detector_dependency.pdf)

定义检测器敏感度差为

\[
\Gamma_m^{(l)}
=\Delta AP_{\mathrm{PointRCNN},m}^{(l)}
-\Delta AP_{\mathrm{CenterPoint},m}^{(l)}.
\tag{5.14}
\]

PU-GCN 在 Line A 与 Line B 的 \(\Gamma\) 分别为 -5.510 和 -6.152 AP，表示 PointRCNN 的退化约比 CenterPoint 多 5.5–6.2 个百分点。方向在两条线一致，支持“体素聚合对部分点级扰动更鲁棒”的解释。但是 CenterPoint 自身仍明显低于 baseline，所以检测器结构只是放大或缓冲因子，不是共同负结果的根因。

目标级审计给出相同方向。Line B 的 527 个 Car GT 中，CenterPoint baseline 与 PU-GCN observed-first 的诊断召回为 77.61% 和 74.38%，差 3.23 个百分点；PointRCNN 对应为 73.24% 和 60.72%，差 12.52 个百分点。尤其在 20–40 m，CenterPoint 从 79.78% 降至 74.26%，PointRCNN 从 71.32% 降至 53.31%。这说明中距离生成几何对点式 proposal 的破坏更大，但并不意味着体素检测器对伪点免疫。

## 5.5 Discussion of Results

### 5.5.1 Effect of Upsampling

跨越两种检测器与两条输入线，最稳健的结论是：**当前 strict-\(4N\) 场景适配下，点数增加本身不是任务性能增加的充分条件。** Line A 在 PointRCNN 和 CenterPoint 中均出现系统性负差值；Line B 只有 CenterPoint 的部分小类别出现有限正值，且没有恢复大部分原始—稀疏缺口。

![图 5.4　生成体素真实精度与 PointRCNN AP 的方法排序关系。](figures/fig5_04_geometry_vs_ap.pdf)

为形式化区分“密度”和“有效证据”，令生成剂量为

\[
\delta_N=\frac{|\mathcal Y|-|\mathcal X|}{|\mathcal X|}=3,
\tag{5.15}
\]

而任务收益为式（5.8）的 \(\Delta AP\)。所有 strict-\(4N\) 方法具有相同 \(\delta_N\)，但 AP 从 PDANS 到 PU-Net* 跨越数十个百分点。这直接表明 \(\delta_N\) 不能解释方法差异。合理的中介变量是有效几何剂量：

\[
\delta_{\mathrm{eff}}(\tau)
=\frac{1}{|\mathcal G|}
\sum_{\mathbf g\in\mathcal G}
\mathbb I\!\left[min_{\mathbf r\in\mathcal R}
\|\mathbf g-\mathbf r\|_2\le\tau\right],
\tag{5.16}
\]

其中 \(\mathcal R\) 是同帧完整扫描参考。该量描述生成点有多少落在参考表面邻域内；它不使用于生成或检测，只在离线诊断中计算。

对 Line B 而言，恢复能力还要求生成点覆盖被稀疏化删除的参考区域。参考覆盖率定义为

\[
C_{\tau}(\mathcal Y,\mathcal R)
=\frac{1}{|\mathcal R|}
\sum_{\mathbf r\in\mathcal R}
\mathbb I\!\left[min_{\mathbf y\in\mathcal Y}
\|\mathbf r-\mathbf y\|_2\le\tau\right].
\tag{5.17}
\]

高 \(\delta_{\mathrm{eff}}\) 不必然产生高 \(C_\tau\)：方法可能反复在已观测表面附近复制点，却不覆盖缺失区域。反之，只提高覆盖也可能同时产生大量伪点。任务有效的上采样需要在贴近表面、覆盖缺失区域和控制错误支持之间取得平衡。

### 5.5.2 Transfer from Object-Level Data to LiDAR Scenes

PU-Net [4]、PU-GCN [5]、PU-EdgeFormer [6] 和 PDANS [7] 的核心设计主要围绕物体级或局部表面点集。KITTI [1,2] 则是米制、全场景、带强度、具有背景与距离衰减的 LiDAR 扫描。二者差异可以写成联合分布偏移：

\[
p_{\mathrm{train}}(\mathbf x,\mathcal N,\rho,I,\kappa)
\ne
p_{\mathrm{KITTI}}(\mathbf x,\mathcal N,\rho,I,\kappa),
\tag{5.18}
\]

其中 \(\mathbf x\) 是坐标，\(\mathcal N\) 是局部邻域拓扑，\(\rho\) 是采样密度，\(I\) 是反射强度，\(\kappa\) 表示表面曲率/边界。域偏移不是一个可由“统一归一化”完全消除的标量问题，而是尺度、邻域组成、采样机制与属性的共同变化。

本实验最明确的迁移故障来自共同 patch 提取器。它把全帧坐标量化到 32 个粗 bin，按 `(x-bin, y-bin, z, index)` 排序后连续切成 2048 点。连续块不是 kNN，也不是 ball query。对 64 个含车帧，Line A/Line B 的 patch 内 p90 半径中位数分别为 3.16/10.02 m，最大半径中位数为 4.79/19.45 m，XY 对角线中位数为 8.26/30.46 m，XY 对角线 p90 达到 77.12/124.10 m。

![图 5.5　共同 2048 点 patch 提取器的空间跨度；Line B 的典型 patch 已不再局部。](figures/fig5_05_patch_locality.pdf)

一个宽 124 m 的 patch 可能同时包含道路、车辆、建筑和多个不相邻目标。若对整块中心化并缩放到单位球，网络会把本不连续的场景压缩为一个“物体”，再按物体表面先验生成点。Line B 因点更稀，为凑满 2048 点而跨越更大空间，所以它比 Line A 更容易失败。这一机制解释了为什么物体级模型即使在规范化数据上具有良好几何指标，也不能无条件迁移到真实场景。

强度属性形成第二层偏移。当前统一策略把每个生成点的 intensity 从最近输入点复制。设最近输入索引为

\[
j^*(\mathbf g)=\arg\min_j\|\mathbf g-\mathbf x_j\|_2,
\qquad I(\mathbf g)=I(\mathbf x_{j^*(\mathbf g)}).
\tag{5.19}
\]

该策略保证各方法一致且不使用标签，却会把同一实测强度复制到多个已经位移的坐标，产生训练分布中较少见的强度平台。由于所有方法采用相同策略，它不是方法排名的直接原因；但它可能造成共同的检测器分布偏移，应在未来以统一局部插值或强度遮蔽消融检验。

### 5.5.3 Relationship Between Geometry and Detection

64 帧几何审计以原始完整扫描作为只读参考。生成体素真实精度定义为

\[
P_{\mathrm{vox},\tau}
=\frac{|\mathcal V_{\tau}(\mathcal G)\cap\mathcal V_{\tau}(\mathcal R)|}
{|\mathcal V_{\tau}(\mathcal G)|},
\tag{5.20}
\]

E1 额外体素比例定义为

\[
E_{\mathrm{vox},\tau}
=\frac{|\mathcal V_{\tau}(\mathcal Y)\setminus\mathcal V_{\tau}(\mathcal R)|}
{|\mathcal V_{\tau}(\mathcal Y)|},
\tag{5.21}
\]

其中 \(\mathcal V_{\tau}\) 表示以边长 \(\tau=0.20\) m 离散后的占据体素集合。Line A 的 PDANS、PU-GCN、PU-EdgeFormer、PU-Net* 生成体素精度依次为 48.83%、45.48%、36.32%、23.11%，对应 PointRCNN E1 Moderate AP 为 64.51、56.96、42.98、9.72；Line B 精度依次为 50.68%、47.53%、38.10%、34.17%，AP 为 45.13、29.69、20.72、8.88。

四方法在两条线中的 Spearman 秩相关均为 \(\rho_s=1.00\)。无并列秩时，

\[
\rho_s
=1-\frac{6\sum_{m=1}^{M}d_m^2}{M(M^2-1)},
\tag{5.22}
\]

其中 \(d_m\) 是方法 \(m\) 在几何精度与 AP 排名中的秩差，\(M=4\)。本实验中所有 \(d_m=0\)。这不是大样本显著性检验，也不能证明每 1% 几何精度会导致固定 AP 增量；它说明当前方法间任务排序与几何真实性完全一致，且与 E1/E2 点数控制共同支持“位置质量比名义密度更关键”。

Line A 原始 baseline 的真实体素召回本来是 100%。增密后真实召回不可能再提高，却有 45.3%–62.5% 的 E1 占据体素不受原始扫描支持。Line B baseline 的参考体素召回为 48.5%；加入 \(3N\) 生成点后只提高到 57.7%–60.9%，同时出现 41.7%–54.4% 的额外体素。也就是说，新增容量中只有一部分覆盖了被删除的参考位置，大量容量用于重复或伪结构。

距离分层进一步显示“覆盖增加”与“位置可靠”可以分离。Line B baseline 在 0–20、20–40、40–70.4 m 的车辆参考表面覆盖分别为 92.04%、65.51%、41.89%。PDANS E1 提高到 96.15%、76.10%、56.62%，其生成点近参考表面比例仍为 98.24%、88.52%、80.80%；PU-GCN 的覆盖为 96.91%、77.54%、50.00%，近表面比例降至 95.19%、77.11%、50.00%；PU-Net* 远距覆盖为 50.93%，但近表面比例只有 22.22%。

![图 5.6　Line B 的距离分层：参考表面覆盖与生成点近表面比例。](figures/fig5_06_distance_geometry.pdf)

由此可以得到一个比“远距点少”更精确的解释：远距上采样可能提高某种覆盖计数，同时把大部分生成点放在不受真实扫描支持的位置；这些点对检测器是高置信度几何噪声，而不是缺失观测。未来评价不应只报告 Chamfer Distance 或覆盖率，而应同时报告 precision-like 与 recall-like 几何量。

### 5.5.4 Detector Dependency

PointRCNN 与 CenterPoint 都下降，说明根因位于检测器之前；CenterPoint 降幅更小，说明输入表示会调节损失大小。可以把最终任务变化分解为

\[
\Delta T_{d,m}^{(l)}
=\underbrace{\alpha_d\,\Delta G_m^{(l)}}_{\text{生成几何}}
+\underbrace{\beta_d\,\Delta A_{d,m}^{(l)}}_{\text{输入适配/预算}}
+\underbrace{\gamma_d\,\Delta Q_{d,m}^{(l)}}_{\text{检测器分布响应}}
+\varepsilon_{d,m}^{(l)},
\tag{5.23}
\]

其中 \(\Delta G\) 概括真实表面支持、额外体素与目标边界污染，\(\Delta A\) 概括点数/体素上限和顺序，\(\Delta Q\) 概括权重对新输入分布的适应程度。该式是解释框架，不是从现有样本拟合出的线性因果模型。

现有消融给每一项提供了方向证据：E1/E2 只改变 PointRCNN 输入预算，恢复有限，说明 \(\Delta A\) 不是全部；CenterPoint observed-first 在达到 40000 体素上限时改善，在未达到时几乎不变，确认 \(\Delta A\) 的局部作用；全训练适配缩小差距但没有消除，说明 \(\Delta Q\) 可缓解但不能覆盖全部；几何精度与 AP 排名一致，支持 \(\Delta G\) 是共同主导因素。

因此，不能把结果概括为“PointRCNN 不适合上采样”或“CenterPoint 可以解决上采样”。更准确的结论是：在当前生成几何下，两者都受损；PointRCNN 对点级错误更敏感，CenterPoint 的体素聚合缓冲一部分扰动，但新占据体素和体素预算仍能传播错误。

![图 5.7　输入顺序收益与 40000 体素上限命中率。](figures/fig5_07_observed_first_voxel_budget.pdf)

### 5.5.5 Object-Level and Qualitative Evidence

任务差值、几何统计和 detector 机制最终需要回到同一批实际目标上核对。图 5.8 给出 527 个 Moderate Car GT 的整体 IoU 分布与距离分层；图 5.9 给出相同协议下的全场景点云；图 5.10 与图 5.11 分别展示一个被 observed-first 恢复的目标和一个点数增加后仍失败的目标。四幅图依次从总体分布、场景密度、正向转换和负向转换提供证据，避免仅展示成功案例。

![图 5.8　固定 256 帧对象级 IoU 累积分布与距离分层召回；该图是诊断，不替代官方 AP。](figures/fig5_08_object_iou_distance_audit.pdf)

![图 5.9　帧 000104 的全场景鸟瞰点云：原始、稀疏、PDANS 和 PU-GCN observed-first；虚线框为 GT。](figures/fig5_09_pointcloud_scene_bev.pdf)

![图 5.10　observed-first 恢复案例：真实点与生成点的组合恢复有效框。](figures/fig5_10_pointcloud_recovery_case.pdf)

![图 5.11　残余失败案例：框内点数增加，但点的空间分布未形成可用检测证据。](figures/fig5_11_pointcloud_failure_case.pdf)

图 5.10 与图 5.11 的对照是本研究结论边界的直观表达：观测优先能够恢复某些被 generated-only 破坏的目标，但“更多框内点”既不保证产生预测，也不保证定位精度回到 baseline。总体结论仍由全量 AP 与 527 个目标的转换计数决定。

### 5.5.6 扩展的平行对照、参数与并列证据

前述分析已经给出主结论，但若只保留若干 AP 表和两个案例，仍不足以回答“具体做了什么、参数是否一致、性能下降是否有多层证据”的问题。本节因此把同一实验链重新按可核验的平行对照展开。扩展部分不引入新的实验口径，也不改变前述结论；它将已经完成的运行结果转换为以下五类证据：完整难度与评价空间对照、运行完整性与计算量对照、三类别与缺口恢复对照、几何—任务联合对照，以及对象与距离分层对照。

#### 5.5.6.1 固定实验参数与已执行协议

表 5.8 汇总本章所有图表共享的关键参数。这里特别区分“生成器参数”“检测器原生预处理”和“只读分析阈值”。例如 0.20 m 体素只用于几何审计，不会回写点云或帮助候选选择；0.25 m 近表面阈值只用于解释车辆生成点是否贴近参考扫描，也没有参与 AP 优化。这样可以避免把使用参考扫描的事后诊断误写成生成时使用了标签或测试集信息。

**表 5.8　KITTI 主实验、适配实验与诊断实验的固定参数**

| 模块 | 参数 | 固定值 | 作用与边界 |
|---|---|---|---|
| 数据划分 | KITTI 训练/验证 | 3712 / 3769 帧 | 3769 帧用于全量主结果；3712 帧只用于 detector adaptation |
| 输入线 | Line A | 完整原始扫描 | 检验已有完整观测上的边际增密 |
| 输入线 | Line B | 固定四分之一下采样 \(D_4\) | 检验稀疏化损失能否被恢复 |
| 当前 patch 适配器 | 每 patch 点数 | 2048 | 四种方法共享；不足时由当前规则补足 |
| 当前 patch 适配器 | 粗空间 bin 数 | 32 | 按 bin 排序后连续切块；已被审计为非真正局部 |
| 上采样输出 | 数量协议 | \(N\) observed \(+3N\) generated \(=4N\) | 原始观测逐点保留，生成候选确定性补足 |
| 点属性 | 生成点 intensity | 最近输入点复制 | 四方法统一；不按方法选择更有利策略 |
| PointRCNN E1 | 输入预算 | detector-native 的 exact-\(4N\) 文件 | 检验真实点保留后完整四倍输入 |
| PointRCNN E2 | 输入预算 | 统一 16384 点 | FOV、0.1 m 体素代表点、距离分层采样；只作敏感性分析 |
| CenterPoint | 有效空间 | \(x\in[0,70.4),y\in[-40,40),z\in[-3,1)\) m | 与冻结 KITTI 配置一致 |
| CenterPoint | 体素大小 | \((0.05,0.05,0.10)\) m | 原生 voxelizer 参数 |
| CenterPoint | 体素容量 | 每体素最多 5 点；测试最多 40000 体素 | observed-first 顺序消融的预算机制 |
| 正式任务指标 | AP | KITTI \(AP_{R40}\) | 分 BBox、BEV、3D 与 Easy/Moderate/Hard 报告 |
| 几何诊断 | 参考体素/近表面阈值 | 0.20 m / 0.25 m | 只读诊断，不进入生成或 detector 推理 |
| 对象诊断 | 固定子集 | 256 帧、527 个 Moderate Car GT | 贪心同类 3D IoU 匹配；阈值 0.70 |
| 距离分层 | 近/中/远 | 0–20 / 20–40 / 40+ m | 分别包含 170 / 272 / 85 个 Car GT |

该参数表对应的是实际执行链，而不是后验建议。图 5.28 在本节末给出从输入线、patch、strict-\(4N\)、检测器到三层审计的完整流程图；所有中间文件均可由 [L14]–[L20] 追溯。

#### 5.5.6.2 PointRCNN：从完整 AP 到运行完整性的平行对照

图 5.12 把表 5.1 的全量结果扩展为 Easy、Moderate、Hard 三难度并排柱状图。最重要的视觉证据不是单个柱高，而是两条输入线、三个难度下的方法顺序保持一致。Line A 中 baseline 的 Easy/Moderate/Hard 分别为 92.273/82.255/77.945，最优 PDANS 为 83.014/64.515/59.668；Line B 中 baseline 为 85.177/65.746/61.382，PDANS 为 65.062/45.126/39.240。由此可见，负差值贯穿难度区间，不是由某一个难度定义或少量边界样本造成。

![图 5.12　PointRCNN 全验证集 exact-\(4N\) E1：两条输入线、四种方法与三个难度的并行柱状对照。](figures/fig5_12_pointrcnn_fullval_e1_all_methods.pdf)

表 5.9 进一步把 Moderate 指标分解到二维图像框、BEV 旋转框和三维框。若问题只发生在高度估计，BBox 与 BEV 应相对稳定而 3D AP 单独下降；实际结果却表现为三种空间同时下降，并且从 BBox 到 BEV 再到 3D 的绝对值逐级降低。这说明错误已经影响候选检出、水平定位与完整三维回归，而非单一 \(z\) 轴误差。

**表 5.9　PointRCNN 全验证集 Moderate AP\(_{R40}\) 的 BBox/BEV/3D 分解（%）**

| Line | 输入 | BBox | BEV | 3D | 3D 相对同线 baseline |
|---|---|---:|---:|---:|---:|
| A | Original baseline（E2） | 94.084 | 88.981 | 82.255 | 0.000 |
| A | PDANS E1 | 78.983 | 75.092 | 64.515 | -17.741 |
| A | PU-GCN E1 | 71.034 | 66.238 | 56.961 | -25.294 |
| A | PU-EdgeFormer E1 | 54.437 | 50.947 | 42.981 | -39.274 |
| A | PU-Net* E1 | 23.489 | 18.067 | 9.717 | -72.538 |
| B | 1/4 sparse baseline（E2） | 79.949 | 76.398 | 65.746 | 0.000 |
| B | PDANS E1 | 60.236 | 54.801 | 45.126 | -20.620 |
| B | PU-GCN E1 | 39.193 | 36.430 | 29.687 | -36.059 |
| B | PU-EdgeFormer E1 | 33.102 | 27.840 | 20.721 | -45.025 |
| B | PU-Net* E1 | 20.774 | 15.904 | 8.878 | -56.868 |

![图 5.13　PointRCNN E1 与统一 16384 点 E2 的 Moderate 3D AP 平行坐标对照；右侧数字为 E2−E1。](figures/fig5_13_pointrcnn_e1_e2_parallel.pdf)

图 5.13 将同一个“Line×方法”在 E1 与 E2 下用线段连接。Line A 的 PDANS、PU-GCN、PU-EdgeFormer仅增加 2.709、1.314、1.979 AP；Line B 的四种方法变化全部不大于零。这种配对呈现比两个独立柱状图更直接：如果 16384 点上限是共同主因，连接线应在两条线中普遍向右移动；实际只在 Line A 有小幅右移，Line B 则轻微左移。

![图 5.14　PointRCNN Moderate AP 从 BBox、BEV 到 3D 的评价空间剖面。](figures/fig5_14_pointrcnn_metric_profiles.pdf)

为了证明低 AP 不是由“任务没有跑完”造成，表 5.10 同时列出每组预测文件总数、空预测文件数、检测运行时间和吞吐率。所有条件都有 3769 个预测文件；所谓空预测文件是该帧 evaluator 输入存在但 detector 没有输出有效框，和缺失文件不同。定义空输出率与检测吞吐率为

\[
r_{\mathrm{empty}}
=\frac{n_{\mathrm{empty}}}{3769},\qquad
v_{\mathrm{eval}}
=\frac{3769}{t_{\mathrm{eval}}}.
\tag{5.24}
\]

**表 5.10　PointRCNN 全验证集运行完整性、空预测与检测吞吐**

| Line | 条件 | 预测文件 | 空文件 | 空文件率 | 运行时间（s） | 吞吐（帧/s） |
|---|---|---:|---:|---:|---:|---:|
| A | Baseline E2 | 3769 | 94 | 2.49% | 639.2 | 5.90 |
| A | PDANS E1 | 3769 | 80 | 2.12% | 1489.0 | 2.53 |
| A | PDANS E2 | 3769 | 81 | 2.15% | 647.2 | 5.82 |
| A | PU-GCN E1 | 3769 | 196 | 5.20% | 1254.9 | 3.00 |
| A | PU-GCN E2 | 3769 | 244 | 6.47% | 657.7 | 5.73 |
| A | PU-EdgeFormer E1 | 3769 | 215 | 5.70% | 1240.5 | 3.04 |
| A | PU-EdgeFormer E2 | 3769 | 276 | 7.32% | 654.2 | 5.76 |
| A | PU-Net* E1 | 3769 | 441 | 11.70% | 1236.9 | 3.05 |
| A | PU-Net* E2 | 3769 | 621 | 16.48% | 643.1 | 5.86 |
| B | Baseline E2 | 3769 | 143 | 3.79% | 644.4 | 5.85 |
| B | PDANS E1 | 3769 | 194 | 5.15% | 798.2 | 4.72 |
| B | PDANS E2 | 3769 | 214 | 5.68% | 645.7 | 5.84 |
| B | PU-GCN E1 | 3769 | 621 | 16.48% | 803.8 | 4.69 |
| B | PU-GCN E2 | 3769 | 625 | 16.58% | 652.6 | 5.78 |
| B | PU-EdgeFormer E1 | 3769 | 705 | 18.71% | 806.3 | 4.67 |
| B | PU-EdgeFormer E2 | 3769 | 704 | 18.68% | 648.0 | 5.82 |
| B | PU-Net* E1 | 3769 | 767 | 20.35% | 791.1 | 4.76 |
| B | PU-Net* E2 | 3769 | 800 | 21.23% | 641.5 | 5.88 |

图 5.15 将空预测文件数与 Moderate 3D AP 放在同一坐标系。18 个已完成条件的 Spearman 相关为 \(\rho_s=-0.89\)：空输出越多的条件通常 AP 越低。这个相关不是“空文件导致全部 AP 差值”的因果证明，因为非空帧中的定位误差和误检同样会降低 AP；但它是独立的运行级证据，说明方法退化已经严重到使更多完整帧没有有效框，而不只是每个框的 IoU 略微移动。

![图 5.15　PointRCNN 空预测文件数与 Moderate 3D AP 的配对散点；圆/方分别为 E1/E2。](figures/fig5_15_empty_predictions_vs_ap.pdf)

图 5.16 报告的是 detector evaluation 的运行时间，不包含上采样网络生成点云的时间，因此不能被误写成端到端速度。E2 的吞吐集中在 5.73–5.88 帧/s，说明统一 16384 点后 detector 计算量接近；E1 在 Line A 只有约 2.53–3.05 帧/s，而 Line B 为 4.67–4.76 帧/s，与二者实际原始点规模差异一致。速度证据再次确认 E1 确实把更大的 exact-\(4N\) 输入送入 detector，而不是在文件层标称四倍、推理时仍读取相同点数。

![图 5.16　PointRCNN 已记录的 detector evaluation 吞吐；不含上采样生成时间。](figures/fig5_16_pointrcnn_runtime_throughput.pdf)

表 5.11 把 E1 与 E2 的 AP 变化和空预测变化放在同一组配对记录中。Line A 的三个有效适配器在 E2 下仅获得 (1.314\sim2.709) AP 的有限改善，同时 PU-GCN 与 PU-EdgeFormer 的空预测反而分别增加 48 和 61 帧；PU-Net* 的点数统一既降低 AP 又增加 180 个空输出。Line B 四种方法的 AP 均未提高，空输出也没有一致下降。因此，固定输入点数能够显著改变 detector 吞吐，却不能单独修复上采样输入与检测器之间的分布失配。

**表 5.11　PointRCNN E1/E2 配对变化：点数统一是否同时改善 AP 与空输出**

| Line | 方法 | E1 AP | E2 AP | E2−E1 AP | E1 空文件 | E2 空文件 | 空文件变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | PDANS | 64.515 | 67.223 | +2.709 | 80 | 81 | +1 |
| A | PU-GCN | 56.961 | 58.276 | +1.314 | 196 | 244 | +48 |
| A | PU-EdgeFormer | 42.981 | 44.960 | +1.979 | 215 | 276 | +61 |
| A | PU-Net* | 9.717 | 8.380 | −1.337 | 441 | 621 | +180 |
| B | PDANS | 45.126 | 44.075 | −1.051 | 194 | 214 | +20 |
| B | PU-GCN | 29.687 | 28.680 | −1.006 | 621 | 625 | +4 |
| B | PU-EdgeFormer | 20.721 | 20.569 | −0.151 | 705 | 704 | −1 |
| B | PU-Net* | 8.878 | 8.781 | −0.097 | 767 | 800 | +33 |

#### 5.5.6.3 CenterPoint：三类别、三难度与缺口恢复

图 5.17 首先只比较原始与四倍稀疏 baseline，避免把上采样方法混入传感器稀疏化效应。三种难度均显示 Cyclist 的相对损失最大，其次为 Pedestrian，Car 最小。该模式与小目标回波数量少、局部中心证据更易被删除的预期一致，也说明任何“类别平均 AP”都可能掩盖最需要上采样的类别。

![图 5.17　CenterPoint 全验证集原始/稀疏 baseline：三类别、三难度并行柱状图。](figures/fig5_17_centerpoint_baseline_sparsity.pdf)

表 5.12–5.14 给出 CenterPoint 的完整三难度数值；“A-”和“B-”分别表示 observed-first 的 Line A、Line B exact-\(4N\) 输入。它们补足表 5.5 只列 Moderate 的不足。

**表 5.12　CenterPoint Car 全验证集 3D AP\(_{R40}\)（%）**

| 输入 | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Original baseline | 88.391 | 79.277 | 76.737 |
| 1/4 sparse baseline | 81.400 | 64.598 | 59.970 |
| A-PDANS | 82.444 | 64.466 | 60.810 |
| A-PU-GCN | 80.013 | 59.932 | 56.575 |
| A-PU-EdgeFormer | 68.036 | 47.248 | 44.166 |
| A-PU-Net* | 22.805 | 14.988 | 14.189 |
| B-PDANS | 69.625 | 46.751 | 42.590 |
| B-PU-GCN | 60.422 | 39.070 | 34.967 |
| B-PU-EdgeFormer | 39.379 | 24.748 | 21.666 |
| B-PU-Net* | 17.534 | 11.718 | 10.436 |

**表 5.13　CenterPoint Pedestrian 全验证集 3D AP\(_{R40}\)（%）**

| 输入 | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Original baseline | 54.037 | 50.653 | 46.252 |
| 1/4 sparse baseline | 27.196 | 24.569 | 21.673 |
| A-PDANS | 49.592 | 45.251 | 41.683 |
| A-PU-GCN | 47.518 | 44.504 | 40.834 |
| A-PU-EdgeFormer | 42.150 | 37.085 | 33.855 |
| A-PU-Net* | 24.479 | 21.687 | 19.389 |
| B-PDANS | 31.467 | 28.201 | 24.756 |
| B-PU-GCN | 27.742 | 24.961 | 21.588 |
| B-PU-EdgeFormer | 17.395 | 15.596 | 13.646 |
| B-PU-Net* | 14.046 | 11.916 | 10.533 |

**表 5.14　CenterPoint Cyclist 全验证集 3D AP\(_{R40}\)（%）**

| 输入 | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Original baseline | 79.244 | 64.605 | 60.990 |
| 1/4 sparse baseline | 25.805 | 15.458 | 14.492 |
| A-PDANS | 71.063 | 46.103 | 43.476 |
| A-PU-GCN | 70.712 | 45.597 | 42.965 |
| A-PU-EdgeFormer | 54.831 | 35.822 | 34.106 |
| A-PU-Net* | 37.058 | 24.396 | 23.453 |
| B-PDANS | 28.118 | 15.519 | 14.895 |
| B-PU-GCN | 30.978 | 16.846 | 16.193 |
| B-PU-EdgeFormer | 13.487 | 7.037 | 6.925 |
| B-PU-Net* | 8.735 | 5.017 | 4.749 |

图 5.18 以配对点图并排展示 Line A 与 Line B，圆点和空心方块分别表示两条实验线，横向连线直接给出同一方法的绝对 AP 间隔。它和图 5.2 的三类别差值条形图必须同时阅读：差值图回答“相对配对 baseline 改变多少”，绝对值图回答“最终达到什么水平”。例如 Line B PU-GCN 的 Cyclist Moderate AP 为 16.846，虽然相对 sparse baseline 增加 1.388，但仍远低于原始扫描的 64.605；若只显示正差值，容易夸大恢复程度。

![图 5.18　CenterPoint exact-\(4N\) observed-first 的两条输入线、四种方法与三类别绝对 Moderate AP。](figures/fig5_18_centerpoint_absolute_parallel.pdf)

表 5.15 与图 5.19 使用式（5.9）把 Line B 的差值除以原始—稀疏缺口。负百分比表示方法不仅没有追回缺失信息，还进一步低于 sparse baseline；超过 \(-100\%\) 表示新增损失大于原始稀疏化损失本身。PDANS 对 Pedestrian 恢复 13.925%，是当前最明确的正向结果；PDANS/PU-GCN 对 Cyclist 只恢复 0.124%/2.824%，而所有 Car 和两种较差适配方法为负。

**表 5.15　CenterPoint Line B 的 Moderate AP 缺口恢复率（%）**

| 方法 | Car | Pedestrian | Cyclist |
|---|---:|---:|---:|
| PDANS | -121.575 | **+13.925** | +0.124 |
| PU-GCN | -173.900 | +1.504 | **+2.824** |
| PU-EdgeFormer | -271.470 | -34.400 | -17.133 |
| PU-Net* | -360.233 | -48.510 | -21.243 |

![图 5.19　CenterPoint Line B 的原始—稀疏性能缺口恢复率；100% 表示完全恢复。](figures/fig5_19_centerpoint_gap_recovery.pdf)

#### 5.5.6.4 几何真实性、占据污染与任务指标闭环

表 5.16 汇总同一批 64 个含车帧上的六个几何—任务变量。生成体素精度和近表面比例越高越好，额外体素和外壳/框内比越低越好；参考召回反映覆盖，但必须和精度共同解释。Line A 的参考召回恒为 100%，因为原始扫描已完整保留；这时新增点不能增加已观测参考体素，只会改变精度和额外占据。Line B 的参考召回由 sparse baseline 的 48.49% 提高至约 57.74%–60.95%，但同时 41.74%–54.39% 的 E1 占据体素不受参考扫描支持。

**表 5.16　几何真实性、车辆表面分配与 PointRCNN AP 联合证据**

| Line | 方法 | 生成体素精度 | E1 参考召回 | 额外体素 | 车内近表面点 | 外壳/框内 | Moderate AP |
|---|---|---:|---:|---:|---:|---:|---:|
| A | PDANS | 48.83% | 100.00% | 45.29% | 97.07% | 0.355 | 64.515 |
| A | PU-GCN | 45.48% | 100.00% | 48.93% | 94.01% | 0.422 | 56.961 |
| A | PU-EdgeFormer | 36.32% | 100.00% | 55.49% | 87.06% | 0.621 | 42.981 |
| A | PU-Net* | 23.11% | 100.00% | 62.45% | 57.34% | 1.049 | 9.717 |
| B | PDANS | 50.68% | 60.78% | 41.74% | 93.13% | 0.451 | 45.126 |
| B | PU-GCN | 47.53% | 60.95% | 45.70% | 82.40% | 0.528 | 29.687 |
| B | PU-EdgeFormer | 38.10% | 60.78% | 53.01% | 69.09% | 0.812 | 20.721 |
| B | PU-Net* | 34.17% | 57.74% | 54.39% | 58.06% | 1.125 | 8.878 |

![图 5.20　几何—任务六指标并列点图；圆点/空心方块分别表示 Line A/Line B，数字为原始实测值，箭头给出优选方向。](figures/fig5_20_geometry_evidence_matrix.pdf)

图 5.20 把六个量纲不同的变量拆成独立小图，避免用归一化色块制造不可直接比较的视觉距离。每个小图保留原始刻度和逐点数值，箭头只说明该指标的优选方向。跨方法重复出现的模式是：PDANS/PU-GCN 在真实性和任务指标上更靠前，PU-EdgeFormer/PU-Net* 的额外体素和外壳污染更严重；两条输入线均呈相同方向。该判断由原始测量值支撑，而不是主观视觉评分。

图 5.21 把额外体素比例作为横轴、PointRCNN AP 作为纵轴，并用点面积编码车辆近表面比例。两条线都从左上向右下排列：额外占据更少、车辆近表面比例更高的方法具有更高 AP。该图与图 5.4 的“生成体素精度—AP”散点构成 precision/error 两个方向的平行证据。

![图 5.21　额外体素比例与 PointRCNN AP；点面积表示车辆生成点近参考表面的比例。](figures/fig5_21_geometry_tradeoff_scatter.pdf)

#### 5.5.6.5 检测器适配、对象迁移与距离证据

图 5.22 显示 64 帧微调筛查的配对结果。PU-GCN 与 Line B PDANS 有 4.175–4.595 AP 的正向变化，但 Line A PDANS 轻微下降，两个 baseline 也下降。表 5.17 同时列出微调后相对对应微调 baseline 的剩余缺口，避免把“相对预训练权重改善”误写成“超过 baseline”。

**表 5.17　PointRCNN 64 帧微调筛查：固定 256 帧 Moderate 3D AP（%）**

| Line | 输入 | 预训练 | 微调后 | 微调变化 | 微调后相对同线 baseline |
|---|---|---:|---:|---:|---:|
| A | baseline | 79.191 | 78.043 | -1.148 | 0.000 |
| A | PU-GCN | 60.266 | 64.441 | +4.175 | -13.603 |
| A | PDANS | 68.598 | 68.165 | -0.433 | -9.879 |
| B | baseline | 66.619 | 62.914 | -3.705 | 0.000 |
| B | PU-GCN | 32.920 | 37.515 | +4.595 | -25.399 |
| B | PDANS | 43.007 | 47.433 | +4.426 | -15.481 |

![图 5.22　64 帧 detector 微调前后配对结果；右侧数字为微调后−预训练。](figures/fig5_22_finetune_screening_parallel.pdf)

对象级证据进一步回答“AP 差值由哪些目标组成”。定义 paired baseline 到 observed-first 的目标净变化为

\[
\Delta N_{\mathrm{TP}}
=N_{\mathrm{obs\ TP,base\ FN}}
-N_{\mathrm{base\ TP,obs\ FN}}.
\tag{5.25}
\]

若 \(\Delta N_{\mathrm{TP}}<0\)，表示新恢复目标少于被破坏目标。表 5.18 中四个 detector/line 条件的净变化均为负：PointRCNN A/B 分别为 \(13-72=-59\)、\(12-78=-66\)，CenterPoint A/B 为 \(25-41=-16\)、\(28-45=-17\)。这与两个检测器的 AP 方向一致。

**表 5.18　固定 256 帧、527 个 Moderate Car GT 的对象迁移与 IoU 四分位数**

| Detector | Line | 输入 | TP | 召回 | IoU P25 | IoU P50 | IoU P75 |
|---|---|---|---:|---:|---:|---:|---:|
| PointRCNN | A | baseline | 458 | 86.91% | 0.755 | 0.813 | 0.858 |
| PointRCNN | A | generated-only | 390 | 74.00% | 0.694 | 0.784 | 0.844 |
| PointRCNN | A | observed-first | 399 | 75.71% | 0.703 | 0.797 | 0.849 |
| PointRCNN | B | baseline | 386 | 73.24% | 0.695 | 0.784 | 0.843 |
| PointRCNN | B | generated-only | 258 | 48.96% | 0.000 | 0.698 | 0.802 |
| PointRCNN | B | observed-first | 320 | 60.72% | 0.567 | 0.749 | 0.813 |
| CenterPoint | A | baseline | 454 | 86.15% | 0.742 | 0.811 | 0.862 |
| CenterPoint | A | observed-first | 438 | 83.11% | 0.734 | 0.804 | 0.859 |
| CenterPoint | B | baseline | 409 | 77.61% | 0.713 | 0.790 | 0.842 |
| CenterPoint | B | observed-first | 392 | 74.38% | 0.698 | 0.771 | 0.833 |

![图 5.23　四个 detector/line 条件的对象级 TP/FN 迁移计数；CenterPoint 未运行 generated-only 对象审计。](figures/fig5_23_object_transition_counts.pdf)

![图 5.24　对象最佳同类 3D IoU 的 P25–P75 区间与中位数；虚线为 0.70。](figures/fig5_24_object_iou_quartiles.pdf)

图 5.24 说明退化不仅表现为阈值两侧 TP/FN 数量改变，IoU 分布本身也整体左移。PointRCNN Line B generated-only 的 P25 为 0，意味着至少四分之一 eligible GT 没有产生正 IoU 的同类候选；observed-first 将 P25 提升到 0.567、median 提升到 0.749，但仍低于 baseline 的 0.695/0.784。CenterPoint 的移动较小，和其 AP 损失较小相符。

图 5.25 在同一张图上平行比较 PointRCNN 与 CenterPoint 的近、中、远对象召回。每个点旁标记 observed-first 相对 paired baseline 的百分点差。Line A 与 Line B 都显示误差随距离增加而放大；CenterPoint 相对更稳健，但远距仍下降。因而 detector 依赖改变了损失幅度，没有反转“远距高风险”的共同方向。

![图 5.25　两种检测器在 Line A/B 的近、中、远 Car 对象召回；标注为 PU-GCN−baseline。](figures/fig5_25_detector_distance_parallel.pdf)

#### 5.5.6.6 输入点数、跨距离点云与最终证据链

表 5.19 使用全量输入审计记录给出每帧点数范围。Line A exact-\(4N\) 的最小/最大值恰为原始输入的四倍；Line B exact-\(4N\) 与原始扫描处于几乎相同点数范围。这是“数量约束已满足”的直接证据，也使负结果更有解释力：Line B 已恢复到约 7.86–12.68 万点/帧，却没有恢复 baseline AP。

**表 5.19　输入点数与完整性审计**

| 输入 | 审计帧数 | 每帧最少点数 | 每帧最多点数 | 有限坐标检查 | 状态 |
|---|---:|---:|---:|---|---|
| Original reference | 3769（源目录另含训练帧） | 78596 | 126797 | PASS | PASS |
| 1/4 sparse baseline | 3769 | 19649 | 31699 | PASS | PASS |
| Line A exact-\(4N\) | 3769 | 314384 | 507188 | PASS | PASS |
| Line B exact-\(4N\) | 3769 | 78596 | 126796 | PASS | PASS |

![图 5.26　输入审计的每帧最小—最大点数范围；Line B 的数量规模已恢复至原始扫描。](figures/fig5_26_input_point_count_ranges.pdf)

图 5.27 新增近、中、远三个 Car 的九宫格点云对照。近距稳定案例为帧 004686、GT 0、距离 15.0 m：baseline/generated-only/observed-first 均为 TP，IoU 为 0.871/0.818/0.834，说明密集目标即使保持检出，新增点也不必然提高定位。中距恢复案例为帧 004902、GT 2、距离 21.8 m：generated-only 的 IoU 0.646 未过阈值，而 observed-first 恢复至 0.866。远距失败案例为帧 006039、GT 3、距离 41.6 m：baseline 只含 9 个框内点仍达到 IoU 0.868，而 generated-only/observed-first 分别有 11/18 个框内点却均为 FN。

![图 5.27　Line B 近距稳定、中距恢复与远距残余失败的真实点云九宫格；蓝色虚线为 GT。](figures/fig5_27_pointcloud_range_gallery.pdf)

三个案例并非用于估计总体概率，而是验证距离曲线背后的几何形态：近距生成点多但定位不一定更准，中距 observed-first 可以通过保留真实结构救回阈值附近目标，远距则可能因几何支持不足而出现“点更多但没有有效框”。这三种行为与 527 个 GT 的总体分层结果相互对应。

![图 5.28　本章已执行的 KITTI 输入—上采样—检测—审计证据链与固定参数。](figures/fig5_28_experiment_evidence_pipeline.pdf)

#### 5.5.6.7 两条实验线与全部方法的同尺度点云对照

为避免只展示 PU-GCN 或只挑选成功案例，图 5.29–5.33 对 Baseline、PDANS、PU-GCN、PU-EdgeFormer 和 PU-Net* 执行完全平行的点云比较。全场景图固定为帧 000104、相同 KITTI 有效区域与坐标比例；仅为保证印刷可读性，对每个输入按相同的 8% 比例抽样显示。抽样只作用于散点渲染，标题与统计所用点数均来自未抽样的 `.bin` 文件。黑色表示输入中保留的观测点，浅灰色表示生成点，蓝色虚线表示 KITTI 真实框。因而颜色不承担方法排序，方法差异主要由空间分布、局部形态和后续统计量表达。

Line A 的 baseline 含 121,994 点，四种方法均为 487,976 点；Line B 的 baseline 含 30,498 点，四种方法均为 121,992 点。图 5.29 与图 5.30 由此给出一个直接控制：每条线内部，各方法总点数严格相同，视觉差异不能归因于某个方法“仅仅输出了更多点”。同时，Line B 输出点数几乎恢复至原始帧，但全量 AP 仍未恢复，这与表 5.19 的 3,769 帧点数审计一致。

![图 5.29　Line A 帧 000104 的 Baseline 与四种上采样方法全场景 BEV 平行对照；所有面板使用相同范围与 8% 显示抽样。](figures/fig5_29_line_a_all_methods_scene_bev.pdf)

![图 5.30　Line B 帧 000104 的 Baseline 与四种上采样方法全场景 BEV 平行对照；所有方法均为 exact-\(4N\)。](figures/fig5_30_line_b_all_methods_scene_bev.pdf)

全场景图能检验道路范围与远距点带，却会压缩单车局部形状。因此图 5.31–5.32 固定帧 004902、Moderate Car GT 2，并在同一个目标坐标系和同一显示范围内并排展示五种输入。局部计数定义为

\[
n_{s,l,m}^{\mathrm{obs}}
=\sum_{\mathbf p\in\mathcal X_s^{(l)}}
\mathbb I[\mathbf p\in\Omega(B_g)],\qquad
n_{s,l,m}^{\mathrm{gen}}
=\sum_{\mathbf p\in\mathcal G_{s,m}^{(l)}}
\mathbb I[\mathbf p\in\Omega(B_g)],
\tag{5.26}
\]

其中 \(\Omega(B_g)\) 是以 GT 朝向对齐并在长宽方向扩展 3 m 的局部窗口。表 5.20 给出各面板未抽样的点数。Line A 中各方法的局部总点数为 2,748–3,192，Line B 为 773–863；这再次说明 exact-\(4N\) 是全帧约束，而不是“每个目标局部恰好四倍”。不同方法把生成预算分配到目标邻域的比例不同，所以局部点数应与几何真实性、IoU 和 AP 联合解读，不能独立作为质量排名。

**表 5.20　帧 004902、GT 2 的同窗口局部点数（观测/生成/合计）**

| 输入 | Line A | Line B |
|---|---:|---:|
| Baseline | 810/0/810 | 222/0/222 |
| PDANS | 655/2093/2748 | 186/587/773 |
| PU-GCN | 777/2409/3186 | 207/646/853 |
| PU-EdgeFormer | 820/2372/3192 | 213/650/863 |
| PU-Net* | 774/2374/3148 | 221/613/834 |

![图 5.31　Line A 帧 004902、GT 2 的五输入局部 BEV；每个面板标出未抽样的观测点/生成点数。](figures/fig5_31_line_a_all_methods_local_bev.pdf)

![图 5.32　Line B 帧 004902、GT 2 的五输入局部 BEV；坐标范围与图 5.31 完全一致。](figures/fig5_32_line_b_all_methods_local_bev.pdf)

单帧仍不能代表数据集分布。为此，从固定 256 帧对象审计集按排序位置等间隔选择 32 帧，对每个输入计算 10 m 径向环带的点数，并以跨帧中位数汇总：

\[
h_{s,l,m,k}=\sum_{\mathbf p_i\in\mathcal Y_{s,m}^{(l)}}
\mathbb I\!\left[r_k\leq\sqrt{x_i^2+y_i^2}<r_{k+1}\right],
\qquad
\widetilde h_{l,m,k}=\operatorname{median}_{s\in\mathcal S_{32}}h_{s,l,m,k}.
\tag{5.27}
\]

图 5.33 使用对数纵轴，是因为近距与远距点数跨越三个数量级。两条线中，四种方法的曲线都在多数距离环带接近各自 baseline 的四倍，但不同方法在 20–50 m 及更远区域发生可见分离。例如 Line B 的 40–50 m 环带中位数从 baseline 的 214 增至 PDANS 的 956.5、PU-GCN 的 1389.5、PU-EdgeFormer 的 1384.5 和 PU-Net* 的 1302.0。该结果证明生成预算确实到达中远距区域；结合 AP 未恢复，问题更接近“新增点的支持位置和局部几何不正确”，而不是“模型没有生成远处点”。

![图 5.33　固定 32 帧上两条输入线、五种输入的径向点数中位数；纵轴为对数尺度，环带宽度为 10 m。](figures/fig5_33_radial_point_density_all_methods.pdf)

#### 5.5.6.8 配对 IoU 分布与带置信区间的距离召回

为了不让均值或少量案例掩盖目标级异质性，对每个 detector、输入线和 GT 计算 observed-first 相对 paired baseline 的最佳同类 3D IoU 差值，并绘制经验累积分布：

\[
\delta_i^{(d,l)}=I_{i,\mathrm{obs}}^{(d,l)}-I_{i,\mathrm{base}}^{(d,l)},
\qquad
\widehat F_{d,l}(t)=\frac{1}{M}\sum_{i=1}^{M}\mathbb I[\delta_i^{(d,l)}\leq t],
\quad M=527.
\tag{5.28}
\]

图 5.34 中四条分布的中位数均位于零左侧。PointRCNN Line A/B 的中位差分别为 -0.0144/-0.0254，改善目标占 38.33%/26.94%，退化目标占 60.53%/63.95%；CenterPoint Line A/B 的中位差为 -0.0058/-0.0084，改善占 45.16%/40.04%，退化占 53.89%/57.50%。这比单一 AP 更清楚地说明 detector dependency：CenterPoint 的目标级扰动较小，但两个检测器都不是“所有目标一起轻微下降”，而是改善与退化同时存在、退化数量占优。

![图 5.34　527 个 Moderate Car GT 的 paired 最佳 3D IoU 差值经验累积分布；零线右侧为改善。](figures/fig5_34_paired_iou_delta_ecdf.pdf)

距离召回进一步采用二项比例的 95% Wilson 区间。若某距离箱共有 \(n\) 个 GT、其中 \(x\) 个满足 IoU 0.70，则 \(\hat p=x/n\)，区间中心和半宽为

\[
c=\frac{\hat p+z^2/(2n)}{1+z^2/n},\qquad
w=\frac{z}{1+z^2/n}
\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}},
\qquad \mathrm{CI}_{95\%}=[c-w,c+w],\ z=1.96.
\tag{5.29}
\]

五个距离箱的样本量依次为 32、138、153、119 和 85。表 5.21 报告图 5.35 中所有点估计，单元格式为 baseline/observed-first。PointRCNN 的差距主要从 30 m 后扩大：Line A 在 30–40 m 从 84.9% 降至 58.8%，40 m 后从 61.2% 降至 32.9%；Line B 对应从 52.9% 降至 33.6%、从 30.6% 降至 12.9%。CenterPoint 同方向下降，但幅度通常更小。图中的区间没有平滑或跨箱共享样本；例如 PointRCNN Line B 在 40–70.4 m 的 observed-first 召回为 12.9%，Wilson 区间为 7.4%–21.7%，仍与 paired baseline 的 30.6% 形成清楚分离。

**表 5.21　按距离分箱的 Car 召回（%，baseline/observed-first）**

| Detector / Line | 0–10 m | 10–20 m | 20–30 m | 30–40 m | 40–70.4 m |
|---|---:|---:|---:|---:|---:|
| 样本量 \(n\) | 32 | 138 | 153 | 119 | 85 |
| PointRCNN / A | 100.0/100.0 | 96.4/97.1 | 91.5/88.2 | 84.9/58.8 | 61.2/32.9 |
| PointRCNN / B | 100.0/100.0 | 97.1/95.7 | 85.6/68.6 | 52.9/33.6 | 30.6/12.9 |
| CenterPoint / A | 93.8/90.6 | 98.6/97.8 | 90.2/91.5 | 83.2/74.8 | 60.0/52.9 |
| CenterPoint / B | 93.8/93.8 | 97.1/97.8 | 86.9/85.6 | 70.6/59.7 | 32.9/29.4 |

![图 5.35　PointRCNN 与 CenterPoint 在 Line A/B 的五距离箱召回；误差棒为逐箱 95% Wilson 区间。](figures/fig5_35_recall_distance_wilson.pdf)

最后，表 5.22 把主要结论与证据位置一一映射。该表的用途是确保每个结论至少有一个全量任务结果和一个互补证据，而不是依赖单张视觉上“看起来合理”的点云图。

**表 5.22　核心结论、主证据、补充证据与数据来源索引**

| 要回答的问题 | 主证据 | 互补证据 | 对应图表 | 数据来源 |
|---|---|---|---|---|
| strict-\(4N\) 是否真正执行 | 3769 帧点数/有限坐标审计 | E1 detector 吞吐变化 | 表 5.10、5.19；图 5.16、5.26 | [L15,L16] |
| 点数恢复是否等于性能恢复 | PointRCNN/CenterPoint 全量 AP | Line B 点数范围接近 original | 表 5.1、5.5、5.19；图 5.12、5.18、5.26 | [L15,L16] |
| 16384 点预算是否为主因 | E1/E2 配对 AP | E2 独立体素与空预测 | 表 5.2、5.10；图 5.13、5.15 | [L16,L17] |
| 几何真实性是否解释方法排序 | 两线体素精度与 AP 秩一致 | 额外体素、近表面、外壳比例 | 表 5.16；图 5.4、5.20、5.21 | [L17] |
| observed-first 是否有效 | PointRCNN generated-only/observed-first AP | 对象恢复/损失迁移 | 表 5.3、5.18；图 5.1、5.23、5.24 | [L14,L18] |
| detector adaptation 是否解决域偏移 | 3712 帧适配、固定 256 帧 AP | 64 帧筛查的正/负变化 | 表 5.3、5.7、5.17；图 5.3、5.22 | [L14,L20] |
| 每个方法是否都在两条线上完成可比点云检查 | 同帧、同范围、同显示抽样的五输入网格 | 32 帧径向点数中位数 | 表 5.20；图 5.29–5.33 | [L15,L17] |
| 影响是否随距离变化 | 527 个 GT 五距离箱召回及 Wilson 区间 | 三距离点云案例 | 表 5.21；图 5.8、5.25、5.27、5.35 | [L18] |
| AP 下降是否由少量离群目标造成 | 527 个 paired IoU 差值 ECDF | TP/FN 迁移和 IoU 四分位 | 图 5.23、5.24、5.34 | [L18] |
| 是否存在 detector dependency | PointRCNN/CenterPoint 同输入差值 | 体素上限与顺序消融 | 表 5.6、5.7；图 5.3、5.7 | [L14,L15] |

### 5.5.7 替代解释、限制与证据边界

**评价脚本错误。** 所有主组具有完整预测文件且使用对应 detector 的固定 evaluator；基线能复现合理 AP，因而“评价器整体失效”不符合证据。但这不排除不同实验年代的权重或预处理差异，所以正文只在各自实验内部计算差值。

**点数上限。** E2 控制 16384 点后 Line B 没有恢复；CenterPoint 在 Line B 也未触发 40000 体素上限，却仍明显下降。因此有限预算是放大因素，而不是共同根因。

**没有保留观测。** E1 完整保留 \(N\) 个真实点，仍低于 baseline；observed-first 可恢复部分 AP，仍有双位数缺口。该解释只能覆盖部分损失。

**检测器没有适配。** 3712 帧训练适配后两检测器仍低于 baseline。域适配有帮助，但不足以使错误几何变为正确观测。

**参考扫描并非真实连续表面。** 原始 KITTI 扫描本身有限且带噪，所以“参考体素外”不一定都是物理错误；它也可能是网络合理补全了传感器未采到的表面。因此本章不把单个额外体素直接标为 false point，而是使用“原始扫描不支持”。然而，当不支持比例达到 40%–60%、车辆外壳污染增加、AP 排名与精度排序一致且两检测器同时下降时，把全部额外点解释为有益补全也不符合联合证据。

**选定案例偏差。** 图 5.10 和图 5.11 是用于解释机制的代表性案例，不用于估计恢复概率。总体效应由 527 个 GT 的转换计数和正式 AP 决定。

## 5.6 Summary of Main Findings

本章的主要发现可以压缩为以下九点，但每一点都绑定到明确证据层级。

1. **严格四倍点数不等于四倍信息。** 3769 帧 PointRCNN 与 CenterPoint 主结果显示，大多数上采样条件低于同线 baseline；Line A 没有任何 CenterPoint 类别改善。
2. **Line A 与 Line B 必须分开解释。** Line A 测边际增密，Line B 测稀疏损失恢复。Line B 的少量正 AP 只出现在 CenterPoint 的 Pedestrian/Cyclist，且只恢复原始—稀疏缺口的一小部分。
3. **保留观测点是必要但不充分的。** PointRCNN Line B observed-first 比 generated-only 提高 9.224 AP，但仍比 baseline 低 11.004 AP。
4. **固定输入预算不是主要根因。** PointRCNN 统一到 16384 点未恢复 Line B；CenterPoint Line B 没有触发体素上限仍下降。
5. **非局部 patch 是已定位的上游故障。** Line B patch 的 XY 对角线中位数 30.46 m、p90 124.10 m，与物体级局部表面假设不相容。
6. **几何真实性与任务排序一致。** 两条线内，0.2 m 生成体素真实精度与 PointRCNN AP 的四方法秩相关均为 \(\rho_s=1.00\)。
7. **检测器决定损失幅度，不改变共同方向。** 全训练适配后 PU-GCN 在 CenterPoint 上下降约 4.9–5.9 AP，在 PointRCNN 上下降约 11.0–11.4 AP。
8. **更多框内点不保证检测。** 恢复与失败点云案例连同 527 个 GT 的对象审计表明，空间位置、表面支持和来源顺序比裸点数更有解释力。
9. **全方法平行点云与分布统计排除了展示偏差。** 两条输入线的五输入网格、32 帧径向点数以及 527 个 paired IoU/距离召回表明，生成预算确实到达中远距，但目标级差值中退化占比更高，且 PointRCNN 的 30 m 后损失有明确 Wilson 区间支持。

这些结果不支持“当前上采样流水线普遍提升 KITTI 三维检测”的强命题；它们支持更细致的结论：在严格、无标签的场景适配下，局部性、几何真实性、观测保护和 detector 表示共同决定增密是否有任务价值，其中当前最主要的限制发生在生成器之前和生成几何本身。

# 6. Conclusion and Outlook

## 6.1 Answers to the Research Questions

### 6.1.1 RQ1：点云上采样能否提高 KITTI 三维目标检测？

在本研究已经完成并通过完整性检查的协议下，答案是：**不能把上采样视为普遍有效的检测增强；只在特定检测器、类别和稀疏输入条件下观察到有限恢复。**

证据来自三个互补层面。第一，3769 帧 PointRCNN E1 中，四种方法在 Line A 和 Line B 的 Car Moderate AP 都低于同线 baseline。第二，3769 帧 CenterPoint 中，Line A 的所有“方法×类别”组合均为负；Line B 的 PDANS/PU-GCN 只在 Pedestrian 或 Cyclist 上取得 +0.061 至 +3.632 AP，而 Car 明显下降。第三，经过 3712 帧输入域适配后，PU-GCN 仍在两个检测器、两条输入线上低于 baseline。

因此，论文结论必须写成条件命题：

\[
\exists(d,c,l,m):\Delta AP_{d,m,c}^{(l)}>0
\quad\not\Rightarrow\quad
\forall(d,c,l,m):\Delta AP_{d,m,c}^{(l)}>0.
\tag{6.1}
\]

本实验只证明左侧存在于少量 CenterPoint Line B 小类别组合，不满足右侧的普遍改善。相应地，“上采样恢复了 KITTI 检测”并不是数据支持的总结；“上采样在部分小目标条件下显示有限恢复，但总体受几何适配约束”才是准确表述。

### 6.1.2 RQ2：几何质量与检测性能有什么关系？

在当前四方法比较中，几何真实性与检测排序具有强一致性。Line A 和 Line B 的 0.2 m 生成体素真实精度排序均为 PDANS > PU-GCN > PU-EdgeFormer > PU-Net*，与 PointRCNN Moderate AP 排序完全相同，Spearman \(\rho_s=1.00\)。车辆框内近参考表面比例也呈相同方向，额外体素和外壳污染则呈反向排序。

然而，该结论应被解释为“必要候选条件”，而不是充分因果定律。几何评价使用原始扫描作为有限参考，无法观测所有真实物理表面；四个方法不足以稳定估计连续效应曲线；同一几何点集进入不同 detector 后还会经过采样、体素化和特征聚合。更严格的结论是：

\[
\Delta AP
=F\!\left(
P_{\mathrm{vox}},
C_{\tau},
E_{\mathrm{vox}},
\eta_{\mathrm{obs}},
A_d,
\phi_d
\right),
\tag{6.2}
\]

其中 \(P_{\mathrm{vox}}\) 是生成体素精度，\(C_\tau\) 是参考覆盖，\(E_{\mathrm{vox}}\) 是额外体素比例，\(\eta_{\mathrm{obs}}\) 是实际保留观测比例，\(A_d\) 与 \(\phi_d\) 分别是检测器适配和权重。现有结果能够确认这些变量共同重要，但不能从四个方法反演出函数 \(F\) 的通用形式。

### 6.1.3 RQ3：结果在不同检测器之间是否一致？

方向一致、幅度不一致。PU-GCN 全训练适配实验中，PointRCNN 在 Line A/B 分别下降 11.400/11.004 AP，CenterPoint 分别下降 5.890/4.852 AP。两者都没有超过 baseline，因此共同根因不能归结为某一个 detector；CenterPoint 的较小损失说明体素聚合对部分点级扰动更稳定。

输入顺序消融进一步揭示 detector-specific 机制。PointRCNN Line B 的 observed-first 相对 generated-only 增加 9.224 AP；CenterPoint 的顺序收益主要发生在 Line A 达到 40000 体素上限时，Line B 几乎为零。换言之，PointRCNN 的关键预算是点采样与局部特征，CenterPoint 的关键预算是体素内点数和非空体素数。任何关于“上采样有效性”的论文结论都应注明下游 detector，而不能把一个模型的响应推广为整个检测任务。

### 6.1.4 研究问题的量化证据汇总

表 6.1 不再逐方法重复第五章，而是为三个研究问题建立“主结果—机制证据—结论强度”对应关系。数值前的正负号始终以 paired line baseline 为参照；不同规模的实验不混合求平均。

**表 6.1　三个研究问题的量化答案与证据强度**

| 研究问题 | 全验证集主结果 | 补充/机制证据 | 支持的结论 | 强度 |
|---|---|---|---|---|
| RQ1：上采样是否提高检测 | PointRCNN 最优 PDANS：Line A/B 为 -17.741/-20.620 AP；CenterPoint Line B 只有少量小类别正值 | 点数范围恢复到原始规模，仍未恢复 Car AP | 不支持普遍提升；仅支持条件化、小幅类别恢复 | 强 |
| RQ2：几何与任务如何关联 | 两条线的生成体素精度排序与 PointRCNN AP 排序完全一致，\(\rho_s=1.00\) | 额外体素 41.74%–62.45%；外壳比例与 AP 反向排序 | 位置真实性比名义点数更能解释当前方法差异 | 中强；方法数仅 4 |
| RQ3：是否依赖检测器 | 全训练适配后 PU-GCN：PointRCNN -11.400/-11.004 AP，CenterPoint -5.890/-4.852 AP | 对象净 TP 变化 PointRCNN -59/-66，CenterPoint -16/-17 | 两 detector 方向一致，体素式 detector 损失较小 | 强补充 |

图 6.1 用四个彼此独立的量化面板压缩结论。左上是全验证集 PointRCNN 最优方法仍低于 baseline；右上是 CenterPoint PDANS 只有 Line B Pedestrian/Cyclist 局部为正；左下是 3712 帧适配后的残余 detector gap；右下是 527 个 GT 上 observed-first 相对 baseline 的对象净 TP 变化。四个面板分别来自全量 AP、类别差值、域适配和对象匹配，因而不是同一份数字的重复绘制。

![图 6.1　第五章主要结论的定量仪表板：全量 AP、类别差值、适配残差与对象净 TP。](figures/fig6_01_quantitative_conclusion_dashboard.pdf)

这个汇总允许区分“确定结论”和“待验证解释”。可以确定的是：当前 strict-\(4N\) 链没有取得普遍 AP 改善；有限点/体素预算和 detector 域偏移都只解释部分损失；几何真实性与任务排序一致。仍待验证的是：修复 patch 与 PU-Net* 规范化后，各方法的绝对 AP 会恢复多少，以及使用真实低线束传感器是否产生相同幅度。

## 6.2 Main Contributions of the KITTI Study

本研究的贡献不在于宣称某个方法取得新的 KITTI 最优 AP，而在于建立并实证了一套能够解释负结果的任务导向评价框架。

**第一，建立两条互不混淆的评价线。** Line A 检查完整扫描上的边际增密，Line B 检查稀疏化后的性能恢复。该设计避免把“稀疏 baseline 很低后略有回升”与“超过原始扫描”混为一谈。

**第二，执行 strict-\(4N\) 与观测可追踪。** 主输出被定义为 \(N\) 个真实观测加 \(3N\) 个生成点，原始观测不被网络回归点冒充。输入数量、观测来源和生成来源可以沿 evaluator 全链追踪。

**第三，使用点式与体素式检测器交叉验证。** PointRCNN [11] 和 CenterPoint [12] 共享同一上采样输入，却具有不同前端表示。两者共同下降把问题定位到检测器之前；差值幅度又揭示下游输入预算的调节作用。

**第四，把任务指标与几何机制连接。** 全量 AP 与 patch 跨度、生成体素精度、参考覆盖、额外体素、距离分层以及对象 TP/FN 转换共同分析，使“AP 下降”从黑盒现象变成可定位的因果候选链。

**第五，保留失败证据而不删除异常方法。** PU-Net* 结果因已确认的尺度包装错误而被标记为流水线诊断，不用于架构排名。这个处理既保留工程事实，也避免不公平学术结论。

## 6.3 Limitations

### 6.3.1 数据与稀疏化模型

Line B 的四分之一下采样是确定性压力测试，不等同于真实低线束 LiDAR。真实传感器变化还涉及垂直角分辨率、扫描相位、运动畸变、反射材料与遮挡次序。当前结果能够说明“在本下采样算子下能否恢复”，不能直接预测从 64 线到 16 线等硬件替换的实际收益。

原始 KITTI 扫描被用作几何参考，但它不是连续、无噪声的地面真值表面。网络在两个观测点之间合理插值时，也可能落入未被原始扫描占据的体素。因此，额外体素比例应与 AP、对象边界和多阈值距离共同解释，不能单独作为错误率。

### 6.3.2 方法权重与适配器

各上采样方法的公开权重、训练数据与输入预处理不完全相同。统一 exact-\(4N\) 能控制输出数量，不能使网络先验完全一致。共同 patch 提取器的非局部性是当前最明确的工程限制；修复后数值可能变化。

PU-Net* 包装器缺少与训练一致的中心化、尺度归一化和逆变换。因此它的当前 AP 只能用于故障诊断。最终答辩稿若保留该方法，必须在修复后完整重跑 3769 帧两 detector 主协议；否则应从方法能力对比表中移出，仅放入实现限制。

### 6.3.3 检测器适配与统计不确定性

全数据适配目前集中于 PU-GCN；四方法的完整 detector-adaptation 矩阵尚未全部建立。因此，不能断言其他方法在充分适配后仍保持完全相同的 AP 差值。64 帧微调只能提供筛查证据。

正式 KITTI evaluator 提供数据集级 AP，但当前报告没有对全部条件执行多随机种子训练或帧级 bootstrap 置信区间。固定权重推理本身是确定的，训练适配仍可能受初始化和 mini-batch 次序影响。对未来的适配实验，应以帧为重采样单位计算配对 bootstrap：

\[
\Delta AP^{*(b)}
=AP\!\left(\{s_i^{*(b)}\}_{i=1}^{n};m\right)
-AP\!\left(\{s_i^{*(b)}\}_{i=1}^{n};\mathrm{base}\right),
\tag{6.3}
\]

并用 \(B\) 次重采样的 2.5% 与 97.5% 分位数形成 95% 区间。这里必须对同一重采样帧同时计算方法与 baseline，以保留配对结构。

### 6.3.4 对象诊断与归因边界

527 个 Moderate Car GT 的对象审计采用贪心同类匹配，只用于解释 IoU 与 TP/FN 转换；它没有按预测置信度积分，也没有完全复制官方 ignore 区域与难度处理，不能替代 AP。选出的恢复/失败案例是可视化证据，不是总体频率估计。

几何精度与 AP 的 \(\rho_s=1.00\) 建立在四个方法上。其排序一致性很强，但样本数太小，不适合宣称普适的统计关系。更大规模研究应加入不同倍率、不同 patch 尺度、不同候选筛选器和多个训练种子，以形成足够的条件点。

## 6.4 Recommended Corrected Experimental Pipeline

现有结果已经指出下一轮实验最有价值的修正顺序。该顺序不是为了追求更高数字而改变协议，而是为了逐项移除已识别的混杂因素。

### 6.4.1 真正局部的无标签 patch 提取

用 FPS seed 加 kNN 或 ball query 替换粗 bin 连续切块。对种子 \(\mathbf c_j\)，kNN patch 定义为

\[
\mathcal P_j^{\mathrm{kNN}}
=\operatorname*{arg\,min}_{\substack{\mathcal P\subset\mathcal X\\|\mathcal P|=K}}
\sum_{\mathbf x\in\mathcal P}\|\mathbf x-\mathbf c_j\|_2^2,
\tag{6.4}
\]

ball-query patch 为

\[
\mathcal P_j^{\mathrm{ball}}
=\{\mathbf x\in\mathcal X:\|\mathbf x-\mathbf c_j\|_2\le r_j\}.
\tag{6.5}
\]

半径 \(r_j\) 可以依局部点间距调整，但规则必须只依赖输入点，不得访问 GT 框、类别或 evaluator。若点数不足，应使用确定性重复/邻近填充并记录 padding 比例；不能扩大到几十米只为凑满 2048 点。

### 6.4.2 与训练一致的规范化和逆变换

每个 patch 保存中心 \(\boldsymbol\mu_j\) 与尺度 \(a_j>0\)：

\[
\widetilde{\mathbf x}_i
=\frac{\mathbf x_i-\boldsymbol\mu_j}{a_j},
\qquad
\widehat{\mathbf g}_k
=a_j\widetilde{\mathbf g}_k+\boldsymbol\mu_j.
\tag{6.6}
\]

网络输入与输出必须分别执行正变换和逆变换，并对每个 patch 记录 \(\boldsymbol\mu_j,a_j\)。PU-Net* 只有在此环节与原训练代码逐项一致后，才可重新进入主方法表。

### 6.4.3 观测保留、候选合并与 exact-\(4N\)

重叠 patch 会产生多于 \(3N\) 个候选。统一、无标签的选择器应求解

\[
\mathcal G^*
=\operatorname*{arg\,max}_{\mathcal G\subseteq\mathcal C}
\left[
\lambda_1\operatorname{Cov}(\mathcal G,\mathcal X)
-\lambda_2\operatorname{Dup}(\mathcal G)
-\lambda_3\operatorname{Out}(\mathcal G,\mathcal X)
\right],
\quad |\mathcal G|=3N,
\tag{6.7}
\]

其中 coverage、duplicate 与 outlier 均从输入几何计算，不使用测试 GT。最终按

\[
\mathcal Y=[\mathcal X;\mathcal G^*],\qquad|\mathcal Y|=4N
\tag{6.8}
\]

组装，并保存观测/生成来源掩码。对 PointRCNN，进一步记录真正进入 16384 点采样的两类比例；对 CenterPoint，记录范围过滤后体素数、达到 5 点上限的体素比例和 40000 体素上限命中情况。

### 6.4.4 分阶段验收条件

在重新运行两个 detector 的 3769 帧全量评价前，建议先通过以下预注册式门槛：

| 阶段 | 验收量 | 建议要求 |
|---|---|---|
| patch | p90 半径、XY 对角线、跨组件比例 | Line B 中位跨度不再达到多目标/全场景尺度 |
| 坐标 | finite、范围、逆变换误差 | 无 NaN/Inf；米制范围合理；往返误差接近数值精度 |
| strict 输出 | 点数、观测哈希、来源掩码 | 每帧恰好 \(4N\)；\(N\) 个观测逐点保留 |
| 几何 | \(P_{\mathrm{vox}}\)、\(C_\tau\)、\(E_{\mathrm{vox}}\) | 精度上升且额外体素下降，不能只提高覆盖 |
| 目标 | 近/中/远车辆近表面比例 | 不随距离出现不可接受的系统性崩溃 |
| detector | 预测文件、空预测、AP 与配对差值 | 全 3769 帧通过；同线 baseline 同时运行 |

“建议要求”中的定量阈值应在查看最终 AP 之前根据开发集确定，避免用测试结果反向挑选最有利阈值。

### 6.4.5 证据门控的重跑顺序与参数登记

图 6.2 把下一轮工作分成五个不能倒置的 gate。Gate 0 已由当前 10 组 CenterPoint 输入审计和 PointRCNN 全量文件检查通过；Gate 1 当前未通过，因为共同 patch 在 Line B 的 XY 对角线中位数达到 30.46 m、p90 达 124.10 m，并且 PU-Net* 缺少与训练一致的尺度变换；Gate 2 当前未通过，因为 exact-\(4N\) 输入含 41.74%–62.45% 的参考扫描不支持体素；Gate 3 显示 observed-first 有局部收益但 baseline gap 尚未闭合。只有 Gate 1–3 修正并预注册后，Gate 4 的 3769 帧全量重跑才具有比较新架构能力的意义。

![图 6.2　修正后 KITTI 实验的证据门控时间线；圆点颜色表示通过、失败、部分完成与待执行，后阶段不能掩盖前阶段失败。](figures/fig6_02_evidence_gated_roadmap.pdf)

**表 6.2　当前证据状态与进入下一阶段所需产物**

| Gate | 当前状态 | 已有证据 | 下一阶段前必须产生的产物 |
|---|---|---|---|
| 0 协议锁定 | PASS | 每帧 exact-\(4N\)、有限坐标、3769 文件齐全 | 冻结帧列表、\(D_4\)、intensity 和 evaluator 哈希 |
| 1 适配器单元测试 | FAIL | 非局部 patch；PU-Net* 尺度错误 | patch 局部性报告、中心/尺度往返误差、来源掩码审计 |
| 2 几何开发集 | FAIL | 生成体素精度 23.11%–50.68%；额外体素 41.74%–62.45% | 在固定开发集报告 precision、recall、extra、near-surface、shell |
| 3 detector pilot | PARTIAL | observed-first 恢复部分 AP/TP，但仍低于 baseline | 同一 256 帧、同线 baseline、两 detector、对象/距离明细 |
| 4 全验证集 | 当前参考已完成 | 3769 帧 PointRCNN 与 CenterPoint 主表 | Gate 1–3 通过后重跑 3 类×3 难度×两线完整矩阵 |

表 6.3 给出下一轮实验需要在运行前登记的参数。表中“已锁定”意味着保持当前定义以便前后可比；“开发集选择后锁定”意味着可以在独立开发集上选择，但一旦查看正式验证 AP 就不得再修改；“逐方法读取训练配置”意味着参数必须忠实于原方法，而不能为了输出更好数字统一成错误尺度。

**表 6.3　修正实验的参数登记表**

| 参数组 | 参数 | 登记规则 | 是否允许查看验证 AP 后修改 |
|---|---|---|---|
| 数据 | train/val/256 pilot 帧列表 | 使用已保存列表与哈希 | 否 |
| 稀疏化 | \(D_4\) 算子、随机种子 | 保持当前 Line B 定义；另加真实低线束实验时单列 | 否 |
| patch | seed 算法 | FPS 或确定性覆盖采样；在开发集选择后锁定 | 否 |
| patch | 邻域 | 2048 点 kNN 或预注册 ball query；记录 padding、半径与重叠率 | 否 |
| 坐标 | center/scale | 逐方法读取训练时定义；保存 \(\boldsymbol\mu_j,a_j\) 并执行逆变换 | 否 |
| 生成 | 倍率与合并 | 每 patch 方法原生倍率；全帧统一选出 \(3N\) generated | 否 |
| 输出 | 观测保护 | \(N\) observed 逐点保留并置前；保存 source mask | 否 |
| 强度 | 主策略 | nearest-input 作为可复现主策略 | 否 |
| 强度 | 敏感性 | 局部插值或 intensity mask，所有方法使用同一规则 | 否 |
| PointRCNN | E1/E2 | E1 为 detector-native exact-\(4N\)；E2 固定 16384，只作敏感性 | 否 |
| CenterPoint | range/voxel/cap | \([0,70.4)\times[-40,40)\times[-3,1)\) m；0.05/0.05/0.10 m；5 点/体素；40000 体素 | 否 |
| 评价 | AP | 固定 KITTI \(AP_{R40}\)，保留 BBox/BEV/3D、三难度、三类别 | 否 |
| 统计 | 训练种子/区间 | 在开跑前登记种子数；用配对 bootstrap 报告 95% 区间 | 否 |

门控逻辑可以形式化为

\[
\operatorname{RunFull}
=\mathbb I(G_0=1)\mathbb I(G_1=1)\mathbb I(G_2=1)\mathbb I(G_3=1),
\tag{6.9}
\]

其中 \(G_k\) 是第 \(k\) 个 gate 是否通过。该式的目的不是把复杂实验简化成一个分数，而是防止“全量 AP 已经跑完”被误认为上游适配器已经正确。只要局部性、坐标逆变换或 strict 输出任一项未通过，新的全量 AP 就只能继续诊断流水线，不能作为公平架构结论。

## 6.5 Outlook and Future Work

### 6.5.1 从“均匀增密”转向“任务不确定性驱动的增密”

当前协议把每个输入点的点数倍率固定为四倍，但 LiDAR 场景的信息缺失并不均匀。道路平面通常不需要与远距车辆边缘相同的生成预算。未来可定义无标签不确定性 \(u_i\)，结合局部密度、曲率、遮挡边界和距离，为 patch 分配预算：

\[
n_j
=3N\cdot
\frac{\exp(u_j/\tau_u)}{\sum_k\exp(u_k/\tau_u)},
\qquad \sum_j n_j=3N.
\tag{6.10}
\]

该策略仍保持全帧 exact-\(4N\)，但把新增点从大面积平坦背景转移到真正缺失且可恢复的局部区域。为了防止检测标签泄漏，\(u_j\) 的主协议应只由输入几何产生；使用 detector feature 的版本应单列为任务感知扩展。

### 6.5.2 显式建模 LiDAR 扫描结构与距离

物体级欧氏邻域无法完整表示 LiDAR 的角度采样。未来可在 range image 中建模水平/垂直邻接，类似 TULIP 对 LiDAR 上采样的传感器结构考虑 [32]，再映射回三维坐标。对点 \((x,y,z)\)，球坐标为

\[
r=\sqrt{x^2+y^2+z^2},\qquad
\theta=\operatorname{atan2}(y,x),\qquad
\varphi=\arcsin(z/r).
\tag{6.11}
\]

在 \((\theta,\varphi)\) 网格中插值能够保持扫描线拓扑，并允许对远距量化误差显式建模。该路线需要同时预测 range 与有效性/遮挡概率，避免在被前景遮挡的射线上补出后方表面。

### 6.5.3 几何—强度联合生成

最近邻强度复制虽然可复现，却不满足坐标位移后的物理一致性。未来模型可联合输出位置与强度分布：

\[
p(\mathbf g,I_g\mid\mathcal P)
=p(\mathbf g\mid\mathcal P)\,
p(I_g\mid\mathbf g,\mathcal P),
\tag{6.12}
\]

并以异方差形式预测不确定性：

\[
\mathcal L_I
=\sum_g
\left(
\frac{|I_g-\widehat I_g|}{\sigma_g}
+\log\sigma_g
\right).
\tag{6.13}
\]

这种设计允许 detector 降低对高不确定生成点的权重，而不是把每个生成点视为与实测回波同等可靠。

### 6.5.4 来源感知与置信度感知的检测器

当前输出虽保存来源掩码，检测器通常只消费 \((x,y,z,I)\)。未来可加入观测标志 \(o_i\in\{0,1\}\) 和生成置信度 \(w_i\in[0,1]\)：

\[
\mathbf f_i^{(0)}=[x_i,y_i,z_i,I_i,o_i,w_i].
\tag{6.14}
\]

这样 detector 能学习“真实观测优先、生成点作为软证据”，而不是让两者无差别竞争。为保持公平，应比较三种设置：冻结 detector 不使用来源、使用来源后仅训练 detector、生成器与 detector 联合训练。联合训练的结果回答系统最优能力，冻结设置回答即插即用迁移能力，两者不能放在同一列而不注明。

### 6.5.5 多帧信息与单帧生成的比较

如果应用允许时间上下文，真实相邻帧配准可能比单帧 hallucination 提供更可靠的缺失表面。未来应加入 multi-sweep baseline：

\[
\mathcal X_t^{\mathrm{multi}}
=\bigcup_{\delta=-K}^{K}
\mathbf T_{t\leftarrow t+\delta}
\mathcal X_{t+\delta},
\tag{6.15}
\]

其中 \(\mathbf T\) 是自车位姿变换。动态目标需要运动补偿，否则多帧叠加会产生另一种几何拖影。把学习上采样与真实多帧观测比较，可以回答生成点在什么计算/延迟约束下才具有实际价值。

### 6.5.6 更完整的统计设计

未来实验应至少包含三类重复：上采样网络训练种子、detector 适配种子和下采样种子。对条件 \(c\) 的层次模型可以写为

\[
T_{c,r,s}
=\mu+\alpha_c+u_r+v_s+\epsilon_{c,r,s},
\tag{6.16}
\]

其中 \(u_r\) 表示训练重复效应，\(v_s\) 表示数据稀疏化重复效应。即使最终仍以官方 AP 为主，也应报告均值、标准差、配对置信区间和效应量，避免把一次训练的 0.1–0.5 AP 波动写成稳定改善。

## 6.6 Final Conclusion

本研究从 KITTI 全场景出发，对四类上采样方法、两条输入线和两种三维检测器建立了可追踪的评价链。结果显示，当前流水线能够稳定生成 exact-\(4N\) 文件，却不能稳定生成等价的真实几何信息。PointRCNN 和 CenterPoint 的共同退化、E1/E2 输入预算消融、非局部 patch 测量、生成体素精度与 AP 排名一致性、距离分层以及对象级 TP/FN 转换，共同把主要问题定位到场景 patch 构造与生成几何，而不是单一 evaluator 或单一 detector。

结论并不是“点云上采样对检测永远无用”。CenterPoint Line B 在 Pedestrian/Cyclist 上的有限正值、observed-first 的部分恢复和 detector 适配的改善都说明上采样存在可利用空间。真正受到否定的是更简单的假设：把物体级网络直接包装到全场景、令点数变为四倍，就会自动提高检测。

后续工作的优先级由证据明确给出：首先修复真正局部的 patch 与 PU-Net 尺度变换；其次在 strict-\(4N\) 下同时提高参考覆盖和生成精度、减少额外体素；再次使用来源/置信度感知的 detector 输入；最后才扩展到多倍率、多传感器与联合训练。只有当这些修正同时在 PointRCNN 与 CenterPoint、全验证集和距离/类别分层中通过，才能把“几何上更密”升级为“任务上更有用”。

# 引用衔接、新增文献与本地实验依据

## 沿用前文的学术文献

以下编号完全继承 `thesis.pdf`，最终合并时不要重新编号。本章实际引用的条目列于此，未引用但已存在的 [3]、[8]–[10]、[13]–[31]、[33] 仍保留在整篇论文总参考文献表中。

[1] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? The KITTI vision benchmark suite,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2012, pp. 3354–3361.

[2] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The KITTI dataset,” *Int. J. Robot. Res.*, vol. 32, no. 11, pp. 1231–1237, 2013.

[4] L. Yu, X. Li, C.-W. Fu, D. Cohen-Or, and P.-A. Heng, “PU-Net: Point cloud upsampling network,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 2790–2799.

[5] G. Qian, A. Abualshour, G. Li, A. Thabet, and B. Ghanem, “PU-GCN: Point cloud upsampling using graph convolutional networks,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 11683–11692.

[6] D. Kim, M. Shin, and J. Paik, “PU-EdgeFormer: Edge transformer for dense prediction in point cloud upsampling,” in *Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP)*, 2023, pp. 1–5.

[7] B. Zhang, S. Yang, H. Chen, C. Yang, J. Jia, and G. Jiang, “Point cloud upsampling using conditional diffusion module with adaptive noise suppression,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2025.

[11] S. Shi, X. Wang, and H. Li, “PointRCNN: 3D object proposal generation and detection from point cloud,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 770–779.

[12] T. Yin, X. Zhou, and P. Krähenbühl, “Center-based 3D object detection and tracking,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 11784–11793.

[15] Y. Zhou and O. Tuzel, “VoxelNet: End-to-end learning for point cloud based 3D object detection,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 4490–4499.

[16] Y. Yan, Y. Mao, and B. Li, “SECOND: Sparsely embedded convolutional detection,” *Sensors*, vol. 18, no. 10, p. 3337, 2018.

[32] B. Yang, P. Pfreundschuh, R. Siegwart, M. Hutter, P. Moghadam, and V. Patil, “TULIP: Transformer for upsampling of LiDAR point clouds,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024, pp. 15354–15364.

[34] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL visual object classes (VOC) challenge,” *Int. J. Comput. Vis.*, vol. 88, no. 2, pp. 303–338, 2010.

[35] A. Simonelli, S. R. Bulò, L. Porzi, M. López-Antequera, and P. Kontschieder, “Disentangling monocular 3D object detection,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 1991–1999.

## 新增学术文献

**无。** 本稿所有学术编号均与前文一致，没有新增 [36] 之后的条目。若最终论文第 3、4 章已经新增 [36]–[37]，保留其既有编号即可；本章没有引用它们，也不再次登记。

## 本地实验依据（不计入学术参考文献）

[L14] `results/pugcn_full_retrain_20260824/reports/full_retraining_report.md`、`full_retraining_summary.csv` 与 `pointrcnn_observed_first_report.md`：3712 帧适配训练、固定 256 帧 PointRCNN/CenterPoint PU-GCN 评价。

[L15] `results/centerpoint_exact4n_e1_20260729/CENTERPOINT_LINE_A_B_REPORT.md` 与 `results/centerpoint_exact4n_reconstructed_order_safe_20260729/full_ap_summary.csv`：CenterPoint 3769 帧、10 组 exact-\(4N\) 主评价和输入顺序消融。

[L16] `results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_live_ap_summary.csv`：PointRCNN 3769 帧 E1/E2 评价。

[L17] `results/kitti_unified_x4_input_preserving_e1_e2_20260718/reports/e1_e2_concrete_root_cause_report.md` 及 `geometry_root_cause_v1/`：64 帧 patch 局部性、生成体素、车辆表面和距离分层审计。

[L18] `results/pugcn_full_retrain_presentation_20260828/evidence/current_object_evidence.json` 与 `current_object_records.csv`：固定 256 帧、527 个 Moderate Car GT 的对象级 IoU 与案例证据。

[L19] `results/current_original_downsampled_upsampling_comparison/current_comparison_report.md`：较早跨输入比较；因协议与当前主实验不同，仅用于历史核对，不进入主表结论。

[L20] `results/pointrcnn_finetune64_six_arms_20260824/reports/eval256_ap_report.md`：64 帧微调筛查结果，不作为最终适配结论。
