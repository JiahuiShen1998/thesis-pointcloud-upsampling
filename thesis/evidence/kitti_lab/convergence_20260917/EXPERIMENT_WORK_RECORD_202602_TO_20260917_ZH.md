# 从二、三月到现在的完整实验工作记录

更新日期：2026-09-17（Europe/Berlin）。范围：本地工作区现存证据；不是对未保留日志的逐日活动作补写。

## 阅读说明与证据边界

这份记录分为两部分：

- **第一部分：截至 09-17 的统一时间线、最新结果、问题与处理、代码及证据入口。** 当前结论以这一部分为准。
- **第二部分：09-10 历史总记录全文。** 保留全部已记录的早期实验、失败、消融、子集结果、代码改动、文档和清理记录；正文原样保留，仅降低标题层级。里面的“当前”“最新”“未证明收敛”均指 09-10 当时，不是 09-17 的状态。

“完整”指尽可能覆盖现存证据中的全部工作类别，不能保证还原没有日志的活动。二、三月可从历史回溯记录及仍在的工程文件确认准备工作，但本次未找到可逐日核验的同期训练日志；四月也未找到足够独立的按日记录。不能把文件存在或修改时间当作当月实验完成的证明。

状态区分：完成／部分完成／失败或中止／代码完成但无正式实验／审计分析／未执行。pilot、smoke、计划和空目录均不冒充 full-val。

历史指标必须按原协议解读：早期 PointRCNN 存在旧 AP_R11 风格 evaluator、不同输入采样和 proposal 配置；不能与 09 月 AP_R40 主表横向拼接。不同子集、不同随机种子或不同权重的 AP 也不直接比较。

## 1. 当前一句话结论

多方法严格上采样没有得到稳定的检测提升；observed-first 加 detector adaptation 显著恢复了性能。现在 PointRCNN 和 CenterPoint 的 Line A/B 都已有完整逐 epoch 验证并满足设定的平台期判据，但最佳结果仍低于各自参照基线。PU-GCN 网络本身始终固定，没有进行 KITTI 上的上采样网络重训。

## 2. 从头到尾的时间线

| 阶段 | 目的与实际工作 | 问题、结果与状态 | 证据定位 |
|---|---|---|---|
| 2026-02 至 03，回溯记录 | 建立 PointNet/ModelNet/KITTI 工程、数据预处理、分类训练与上采样接口 | 工程准备；没有可确认的完整 ModelNet40 分类准确率。日期仅能按历史总记录定位，不能细分每一天 | 第二部分第 10、17 节；thesis_demo |
| 2026-04 | 早期工程到五月实验的过渡 | 未找到足够独立同期记录，不补造实验和成绩 | 证据缺口 |
| 05-03 至 05-08 | KITTI baseline、EAR、PU-Net x2；PU-GCN 环境和推理恢复 | 形成早期完整评估，也暴露出严重掉点和环境兼容问题。05-04 PU-Net 旧协议 Moderate 3D AP 1.1364 | 05-04 工作日志；第二部分第 9 节 |
| 05-12 至 05-19 | fair comparison、RPN4096、TULIP、PU-EdgeFormer 初次接入 | 配置未统一；PointRCNN 负采样数、部分文件缺失、旧 TF/CUDA 编译及权重问题。不能将这些混合协议结果并作正式排名 | 第二部分第 6、9 节 |
| 06-06 至 06-15 | TULIP 全量与输出审计；PDANS/SPU-PMD 接入；HPC 迁移准备 | TULIP 已跑但非统一 exact-4×；SPU-PMD 仅 4-frame smoke；HPC 受 DNS/认证阻塞 | 第二部分第 5、9、11 节 |
| 06-20 至 06-30 | 审计真实点数倍率，设计严格 x4 Line A/B，补充可视化与清理 | 发现旧输出不总是 x4、部分方法有点数上限；大中间文件迁移和缓存清理有清单 | 第二部分第 3、9、12 节 |
| 07-03 至 07-18 | PDANS、PU-GCN、PU-EdgeFormer、PU-Net 严格 Line A/B；PointRCNN 全量；E1/E2/E3 输入控制 | 主实验完成；上采样未超过 baseline；增加 sampler-safe，区分文件层点数和检测器真正读入点数 | 第二部分第 5–7 节；master inventory |
| 07-19 至 07-31 | CenterPoint 全量；dose、direct no-resample、patch 根因、PU-Net 归一化 2×2 消融、surface patch | 找到重复点、非局部 patch、归一化及 voxel 激活问题。修复可恢复部分 AP，但不等于超过 baseline；部分全量 dose 未完成 | 第二部分第 5–7、12 节 |
| 08-04 至 08-11 | 多方法局部 patch pilot、region split、detector-aware PDANS V2–V5、Line B 选择策略 | 256-frame 开发和部分 holdout64 完成；局部正结果没有成为稳定的跨协议/跨检测器增益 | 第二部分第 7 节 |
| 08-12 至 08-18 | 方法可行性调查、采样方差、系统汇报、目标漏检案例 | 16-seed 检查削弱单次正增益解释；新方法调查不计作已运行实验；生成报告和 lost-car 图 | 第二部分第 7、14、15 节 |
| 08-24 至 08-28 | finetune64 六分支筛选；完整 train 来源数据的 detector adaptation；observed-first pilot | 先子集试验再完整训练，256-val pilot 得到改善；3 epochs 是固定长度，不是由验证收敛选出 | 第二部分第 8 节；full_retraining_report |
| 09-02 至 09-08 | 论文章节 3–6、ModelNet/KITTI 整合、图表与来源清单 | 文档产出完成；部分表仍是旧 pilot，不能自动当作后来的 full-val 结果 | 第二部分第 14 节 |
| 09-08 至 09-10 | 严格 PU-GCN A/B 各 3,769 帧；双检测器 20-arm 全量矩阵；全类别 AP 和源码封存 | 20/20 PASS；明确 3 epochs 没有收敛证据；固定上采样网络与 detector adaptation 的含义被澄清 | full_val_detector_matrix；convergence_audit |
| 09-14 起收敛补充 | CenterPoint A/B 各 12 epochs，每轮保存和全量验证 | 两线满足 0.2 AP / patience 3 判据；最佳 A 75.9661、B 62.9007，均 e12 | CenterPoint convergence summary / CSV |
| 09-14 至 09-15 | PointRCNN 最初 12-epoch RPN 收敛实验 | A RPN e12 又显著提高到 69.5780，末尾未到平台期，不能宣布收敛 | 12→18 schedule_extension_protocol |
| 09-15 至 09-16 | 新 18-epoch PointRCNN schedule，A/B RPN 与 RCNN 分阶段全量验证 | A 两阶段、B RPN 达标；B RCNN e16 到 56.8876，e17/e18 仅 2 轮未显著改善，未达 patience 3 | 18-epoch 逐阶段 CSV/JSON |
| 09-16 至 09-17 | 保持 B RPN e14，另开 24-epoch RCNN 完整 schedule | 最佳 e19 为 57.0485；e20–24 连续 5 轮未显著改善，状态 CONVERGED；24 已满足判据，无需执行后备 30/36 schedule | final_comparison.json；status.json |
| 09-17 本次整理 | 合并最新收敛证据和此前全部工作记录 | 新建简报与本文件，保留 09-10 原记录；没有启动新训练，没有修改实验权重 | 本文件、简报、最新结果 CSV |

早期工程证据：[thesis_demo 分类训练](/home/ra87racy/projects/thesis_demo/train_cls.py)、[PointNet 模型](/home/ra87racy/projects/thesis_demo/models/pointnet_cls.py)、[ModelNet 数据预处理](/home/ra87racy/projects/thesis_demo/scripts/preprocess_modelnet40_off.py)。其中 train_upsampling.py、dgcnn_cls.py 目前仍为空文件；接口/文件名存在不代表对应模型训练完成。

五月直接日志：[WORKLOG_2026-05-04](/home/ra87racy/projects/baseline_detectors/PointRCNN/WORKLOG_2026-05-04.md)。较早总清单：[master inventory](/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_EXPERIMENT_MASTER_INVENTORY_20260804_ZH.md)。历史日期主要依据当时报告的阶段归属，文件名日期不一定等于完成日期。

## 3. 当前正式结果：未适配、原 3-epoch、平台期结果与 baseline

统一指标：Car 3D Moderate AP_R40；每个评估使用 3,769 帧。这里的“未适配”指相同 observed-first 点云输入搭配官方 detector 权重，不是原始点云 baseline。

| 检测器 | Line | 同输入未适配 | 原 3-epoch | 延长训练后最佳 | 参照基线 | 与基线差值 |
|---|---|---:|---:|---:|---:|---:|
| PointRCNN | A | 64.2748 | 71.0075 | **71.5920** | 81.9528 | -10.3608 |
| PointRCNN | B | 35.5289 | 55.2145 | **57.0485** | 68.3312 | -11.2827 |
| CenterPoint | A | 63.2064 | 74.7012 | **75.9661** | 79.2773 | -3.3112 |
| CenterPoint | B | 44.0929 | 61.6639 | **62.9007** | 68.0490 | -5.1483 |

| 检测器 | Line | 相对同输入未适配提升 | 相对旧 3-epoch 提升 | 最佳 checkpoint |
|---|---|---:|---:|---|
| PointRCNN | A | +7.3172 | +0.5845 | RPN e14 + RCNN e2，18-epoch schedule |
| PointRCNN | B | +21.5197 | +1.8340 | RPN e14 + RCNN e19，24-epoch schedule |
| CenterPoint | A | +12.7597 | +1.2649 | detector e12，12-epoch schedule |
| CenterPoint | B | +18.8078 | +1.2368 | detector e12，12-epoch schedule |

证据：[PointRCNN 最终汇总](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/FINAL_RESULTS.md)、[PointRCNN 逐阶段证据](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/final_comparison.json)、[CenterPoint 收敛汇总](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_20260914/reports/centerpoint_convergence_summary.md)。

比较限制：

1. Line A baseline 是原始 N 点、官方 detector 权重；没有单独重训到收敛的 original-N baseline。
2. Line B baseline 是稀疏 M 点、既有 3-epoch adapted detector；也没有在此次延长成收敛 baseline。
3. 这些差值是对已有参照的结果差，不是所有分支训练预算完全一致的因果效应。
4. checkpoint 按同一验证集的 Car Moderate AP 选择，再报告同一验证集结果；应称 validation-selected 结果，不是独立 test 泛化结论，存在选择偏差风险。
5. 对 Car 设置的平台期规则不能自动推广为所有类别、所有指标都已收敛。旧全类别结果见第二部分第 4.3 节，不能把旧 checkpoint 的 Pedestrian/Cyclist 数字嫁接给新最佳 checkpoint。
6. 09-10 的 20-arm 矩阵仍是原 3-epoch 实验，不能将其中 adapted 行静默替换后继续称为当时的原始矩阵。

## 4. 收敛究竟验证到了哪一步

### 4.1 使用的规则

显著改善条件为当前 AP **严格大于**记录的显著最佳 AP + 0.2；否则连续未显著改善计数加 1。末尾计数至少为 3，标为达到平台期。报告遍历完整 schedule；中途曾触发 patience 后若又显著反弹，会重新计数。

全局数值最佳 checkpoint 和“显著改善”的参考 checkpoint 可以不同。比如 A RCNN e2 比 e1 小幅更高，但没高过 0.2，所以不重置显著改善计数；CenterPoint e12 同理。

该规则提供本次训练设置下的操作性收敛证据，不证明参数梯度为零、不证明全局最优，也不证明换学习率或训练方案不会继续改善。

### 4.2 PointRCNN 两阶段证据

RPN 阶段的 AP 是对应组合检测评估值，不应与最终 RCNN 模型的 AP 当作同一训练阶段。

| Line | 阶段 | 完成 epochs | 数值最佳 epoch | 最佳 AP | 末尾连续未显著改善 epochs | 状态 |
|---|---|---:|---:|---:|---:|---|
| A | RPN | 18 | 14 | 70.9712 | 4 | CONVERGED |
| A | RCNN | 18 | 2 | 71.5920 | 17 | CONVERGED |
| B | RPN | 18 | 14 | 52.7646 | 4 | CONVERGED |
| B | RCNN | 24 | 19 | 57.0485 | 5 | CONVERGED |

最终选用的四条曲线共 78 个 epoch 评估（18+18+18+24），不是 78 个独立随机种子实验。旧 18-epoch B RCNN 和更早 12-epoch A RPN 是另外的 schedule，保留作历史证据，不拼接成一条连续曲线。

B RCNN 24-epoch 尾段：

| epoch | Car Moderate AP_R40 |
|---:|---:|
| 19 | 57.0485 |
| 20 | 55.6280 |
| 21 | 55.3521 |
| 22 | 55.3774 |
| 23 | 55.6153 |
| 24 | 55.5902 |

所以最终使用 e19，而不是仅因最后保存就使用 e24。

### 4.3 CenterPoint 两线证据

| Line | 完成 epochs | 最佳 epoch | 最佳 AP | 显著改善参考 epoch / AP | 末尾未显著改善轮数 |
|---|---:|---:|---:|---|---:|
| A | 12 | 12 | 75.9661 | e8 / 75.7998 | 4 |
| B | 12 | 12 | 62.9007 | e9 / 62.8977 | 3 |

A 最后的数值增量 0.1663、B 为 0.0030，均不超过 0.2。因此“e12 最高”与“满足此平台期阈值”不矛盾；也不应说末尾 AP 完全不再上涨。本次核验 CSV 共 24 行，均标记 3,769 帧，24 个 checkpoint 路径均存在。

### 4.4 为什么不能直接把 18 后面接成 24

OneCycle 学习率轨迹依赖总 epochs。此次 24-epoch B RCNN 是从同一 RCNN 初始化、固定同一 RPN e14 开始的完整新 schedule，不是旧 e18 后再追加 6 个等价 epoch。相同总 schedule 内的中断可以从 checkpoint 恢复；改变总 schedule 时需明确记作新实验。

证据：[12→18 协议](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_e18_20260915/schedule_extension_protocol.json)、[B RCNN 24 schedule 与初始化哈希](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/final_comparison.json)。

训练过程中有过进程中断和阶段衔接恢复，不应把一次中断说成最终失败；目前结果文件明确给出 CONVERGED。本次受沙箱限制不能连接用户 systemd bus，因此不以本次隔离环境进程列表证明整机无训练任务。

## 5. 两个核心问题，以及此前反复追问的细节

### 问题一：上采样为什么一直低于 baseline，是否只要增加点就能恢复信息？

现有结果不支持“点更多就检测更准”。四方法主实验、双检测器、输入控制、patch/归一化消融、采样/voxel 审计都提示，几何质量与输入分布同样重要。生成点不是新的传感器观测；增加点数不保证恢复被降采样丢掉的目标证据。

但并非已经将每种机制的贡献完全因果分解。比如某个 patch 修复导致 AP 上升，只能支持该配置下修复有帮助，不能量化所有剩余差距的唯一原因。详细数值全部在第二部分第 5–7 节。

### 问题二：是不是因为只跑了 3 epochs，所以还不能判断？

以前确实不能宣称 3 epochs 已收敛；现在已补上逐 epoch checkpoint、3,769 帧验证和明确平台期规则。延长后四组最佳 AP 都提高，但仍未达到参照 baseline。因此“仅因三轮训练不足”不能完整解释目前差距；这仍不等于排除其他优化方案。

**最初为什么选 3？** 代码证明它是预先固定的 adaptation 长度；现存日志没有记录选择 3 而非其他值的实验依据。不能把“预算有限”“三轮足够”等猜测补写为事实。

### 3N/3M 如何选，原理是什么？

设原始点数 N；Line A 直接从 N 上采样到 4N。Line B 先固定 seed 无放回保留 M=floor(N/4)，再生成 4M。

- direct：只使用网络生成的 strict 4N/4M rows。
- observed-first：保留全部真实 observed N/M rows，再从生成 rows 中稳定 seed 均匀无放回抽 3N/3M，合起来仍为 4N/4M。
- 因而“3N”是为了保持总数 4N 同时保留 N 个真实点，并不是网络标出的三组高置信度新点。
- 没有按置信度、曲率、FPS 或几何得分选 3N；也没有保证坐标与 observed 完全不同。无放回保证 row index 不重复，不等于几何坐标必然唯一。
- PU-GCN patch 输入 2048、输出 8192，单 patch 归一化、推理后反归一化；生成点 intensity 用 observed 1-NN 继承。这些是实际流程，不追加未涉及的筛点机制。

下面摘录实际构造代码的关键逻辑，不是新提出的算法：

```python
BASE_SEED = 20260718

def stable_seed(*parts):
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF

expected = 4 * observed.shape[0]  # predicted 行数先被检查等于 expected
seed = stable_seed(BASE_SEED, "e1", args.reference_token, frame)
rng = np.random.default_rng(seed)
selected = rng.choice(expected, size=3 * observed.shape[0], replace=False)
final = np.concatenate((observed, predicted[selected]), axis=0).astype(
    np.float32, copy=False
)
```

完整代码：[prepare_centerpoint_observed_first_train.py](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/prepare_centerpoint_observed_first_train.py)。脚本同时审计 observed 前缀与生成后缀精确一致，并记录选择索引和输出的 SHA-256。虽然文件名有 CenterPoint，所得 observed-first 输入亦用于相应 PointRCNN 实验；具体输入路径见 final_comparison.json。

详细协议：[protocol_and_code_zh.md](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/protocol_and_code_zh.md)。其末尾旧运行状态只反映当时进度，不能覆盖已完成的主表或最新结果。

### 究竟重训了谁？训练帧数为什么有不同数字？

PU-GCN 固定作者 PU1K model-100；重训的是 PointRCNN 的 RPN/RCNN 和 CenterPoint。KITTI train split 来源 3,712 帧，PointRCNN 训练筛选后实际保留 3,265 帧，与训练对象/范围过滤有关；val 为完整 3,769 帧。这几个数字分属不同阶段，不能将 3,265 写成验证集缺失，也不能将 detector epochs 写成 PU-GCN epochs。

## 6. 主要问题—处理—结果台账

历史数值保留各自子集和协议，不与第 3 节的新 AP_R40 表混用。

| 问题 | 做了什么 | 结果与可支持结论 |
|---|---|---|
| 旧环境/自定义 CUDA ops 不兼容 | 修复 TF 路径、扩展编译/依赖、设备 tensor、可选导入；不同项目使用兼容环境 | 多方法能运行；PU-EdgeFormer 正式结果来自 ops-reuse 兼容路径，不能说原环境一键复现 |
| 上采样结果没有严格 4× | ratio audit；重新定义 A=N→4N、B=M→4M；逐帧审计 | 09 月 A/B 各 3,769 帧 strict 输入完成；TULIP 保留独立协议 |
| 旧 patch 空间不局部 | 对空间跨度做审计，改局部/覆盖 patch | 旧 Line B patch XY p90 可达 124.10 m；局部修复恢复部分性能，但未消除全部掉点 |
| 第一版局部 patch 复制/聚集 | 统计重复率、voxel 占用并修订构建 | 一批输出重复 rows 59.4%；PointRCNN 与 CenterPoint 反应不同，说明单看几何图不足 |
| PU-Net 归一化错误 | 旧/正确归一化 × 旧/局部 patch 的 2×2 消融 | 256-frame PointRCNN 从旧组合 7.9823 到正确+局部 43.5424，仍低于同批 baseline 81.3033 |
| PointRCNN 固定采样及负 sample size | sampler-safe、direct no-resample、E1/E2/E3 对照 | 修复运行错误；取消采样并未保证改善，连 baseline 也变，因此不能混合两种协议解释 |
| CenterPoint voxel 截断/激活变化 | 统计 voxel 数量、cap 命中与密度分布 | A 中生成方法更常触发 cap；B 不触发也掉点，cap 不是唯一解释 |
| 生成比例过高是否破坏检测 | dose pilot、全量 g10、小比例真实点对照 | 小子集偶有正值，全量 g10 未稳定胜 baseline；全量 g25/g50 未完成，勿与多比例 pilot 混淆 |
| detector-aware PDANS V1–V5 的局部正增益 | 组件消融、双检测器、holdout64、16-seed | V1 单次 +1.2227 未成为稳健优势；16-seed 平均差 -0.438，9/16 为负；V5 holdout CP Car -0.1369 |
| region split 是否保住更多目标 | 三类 PointRCNN 分区评估、holdout 和采样敏感性检查 | 个别 holdout 有增益，但全方法 pilot 与 worker/采样变体不稳定；不能宣称稳定 detector-wide 改善 |
| 从 direct 生成输入改为 observed-first | 保留 N/M 真实点并抽 3N/3M 生成点 | full-val 中 observed-first 优于对应 direct；配合 adaptation 恢复更明显，仍未胜 baseline |
| 3 epochs 是否足够 | 原收敛审计；后来每轮存 checkpoint 和全量验证 | 原来无证据；现在四组按设定阈值达到平台期，最新数值见第 3–4 节 |
| 中途短平台是否等于结束 | 检查完整曲线和末尾 patience | 12-epoch A RPN、18-epoch B RCNN 都出现后期改善，最终扩展到合适的完整 schedule |
| baseline 是否同训练预算 | 明确列出原始/稀疏、official/adapted 参照 | 未补做所有 baseline 的收敛训练；报告保留该比较限制，不虚构公平性 |
| 小子集和随机采样误差 | 16-seed、重复子采样不确定性估计 | 20-frame smoke/pilot 不能支持小幅增益；选择依据要与独立泛化结论分开 |
| EAR strict 全量太慢 | 少量帧实测并估算 | 当时估算 full A/B 需约 92–103 天，未完成；不把旧 EAR 评估冒充新 strict full |
| SPU-PMD 接入困难 | 修复导入/环境并跑 4-frame smoke | finite 输出与局部检测可用，但无完整 AP |
| ModelNet40 分类链未完成 | 准备工程、协议；PU-GCN 每线 8 样本 smoke | 无本地完整 12,311 × 两线结果，无 PointNet++ 分类准确率；已有 HPC 文件不能无日志归为新实验 |
| HPC 迁移失败 | bundle、校验清单、dry-run、连接检查 | DNS/认证阻塞，未完成远程传输/作业，不能写成 HPC 全量运行 |
| 磁盘压力 | 多轮获授权清理、保留最终产物和清单 | 历史清理容量见第二部分第 12 节；那些容量与空闲空间不是今天的实时值 |
| 实验代码大量未跟踪 | 源码快照和 SHA-256 manifest | 09 月 151-file snapshot 有据；旧 snapshot 不包含后来新增的收敛脚本，需分别引用 |
| 论文与报告版本滞后 | 生成本次统一记录，保留原版本 | 本次没有重写所有既有 PPT/论文章节，旧材料仍需人工同步最新表 |

## 7. 已完成、部分完成与没有做的边界

完成：历史多方法主实验和多批诊断（按各自协议）；PU-GCN 09-10 双检测器 20-arm 全量矩阵；observed-first 的 CenterPoint 12-epoch A/B 验证；PointRCNN A/B 分阶段平台期验证；本次中文简报与综合记录。

未被本次工作完成或不能证明：PU-GCN 网络的 KITTI 重训、ModelNet40 正式分类、EAR 新 strict full-val、SPU-PMD full-val、HPC 成功迁移、全部方法都进行完整 detector adaptation、所有 baseline 等预算收敛、每个类别都已收敛、独立 test 改善、所有既有 PPT/章节已同步。

“满足本次判据”不等于“所有实验都完成”，也不等于“算法已经超过 baseline”。

## 8. 结果、图表、代码与证据导航

### 最新结果与现成曲线

- [PointRCNN 最终结果](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/FINAL_RESULTS.md)
- [最终完整 JSON（含每轮轨迹、旧结果和源码/权重哈希）](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/final_comparison.json)
- [最终任务状态](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/status.json)
- [A RPN 18-epoch 曲线](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_e18_20260915/reports/pointrcnn_line_a_pugcn_observed_first_rpn_convergence.png)
- [A RCNN 18-epoch 曲线](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_e18_20260915/reports/pointrcnn_line_a_pugcn_observed_first_rcnn_convergence.png)
- [B RPN 18-epoch 曲线](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_e18_20260915/reports/pointrcnn_line_b_pugcn_observed_first_rpn_convergence.png)
- [B RCNN 24-epoch 曲线](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/schedule_24/reports/pointrcnn_line_b_pugcn_observed_first_rcnn_convergence.png)
- [B RCNN 24-epoch 逐轮 CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_line_b_rcnn_to_convergence_20260916/schedule_24/reports/pointrcnn_line_b_pugcn_observed_first_rcnn_convergence.csv)
- [CenterPoint A/B 逐轮曲线](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_20260914/reports/centerpoint_epoch_validation_curve.png)
- [CenterPoint A/B 逐轮 CSV](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_convergence_20260914/reports/centerpoint_epoch_validation_curve.csv)
- [09-10 原始 20-arm full-val 表](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/full_val_detector_matrix.md)
- [09-10 全类别 AP](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/all_classes_ap_r40.csv)
- [本次整理的四组最新对比 CSV](/home/ra87racy/reports/LATEST_DETECTOR_COMPARISON_20260917.csv)

### 实际使用的关键代码

- [observed-first 输入构造及逐帧审计](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/prepare_centerpoint_observed_first_train.py)
- [原 full-val 主流程](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pugcn_detector_adaptation_full_val_20260908.sh)
- [PointRCNN 分阶段训练](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pointrcnn_full_train_stage.py)
- [PointRCNN checkpoint 验证](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pointrcnn_checkpoint_split_eval.py)
- [PointRCNN 收敛主流程](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pugcn_pointrcnn_convergence_20260914.sh)
- [PointRCNN 平台期判定与汇总](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/summarize_pugcn_pointrcnn_convergence_20260914.py)
- [B RCNN 延长到满足判据的调度器](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_line_b_rcnn_until_converged_20260916.py)
- [CenterPoint 收敛主流程](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/run_pugcn_centerpoint_convergence_20260914.sh)
- [CenterPoint 平台期判定与汇总](/home/ra87racy/projects/baseline_detectors/PointRCNN/scripts/summarize_pugcn_centerpoint_convergence_20260914.py)
- [旧 3-epoch 收敛审计](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/convergence_audit.md)
- [09-10 源码快照说明](/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/experiment_source_full.md)
- [ModelNet40 状态](/home/ra87racy/projects/modelnet40_pointnet2_upsampling/reports/modelnet40_x4_final_protocol_report.md)

## 9. 可用于论文/答辩的结论

> 在本实验严格点数控制的 KITTI Line A/B 协议中，通用预训练上采样方法未带来稳定的 3D 检测增益。保留真实观测点并进行检测器适配显著缩小了与参照 baseline 的差距。进一步逐 epoch 验证显示，PointRCNN 和 CenterPoint 的当前训练设置均满足事先定义的 AP 平台期判据，但最佳验证结果仍低于现有 baseline。结果说明仅增加点数及延长当前适配 schedule 不足以消除差距；同时，baseline 训练预算未完全匹配、验证集用于 checkpoint 选择，限制了结论的因果解释与泛化范围。

---

## 第二部分：截至 2026-09-10 的历史总记录全文

**以下是历史快照，不是最新状态。特别注意：其中“3-epoch 未证明收敛”保留为当时的事实；09-14 至 09-17 的延长 schedule 是后来新增实验，不会让原三轮实验追溯性地变成已收敛。旧的运行状态、可用空间、当前最优等均只对 09-10 有效。**

保留原有结果、失败和未执行事项，不删除负结果。原始文件：[09-10 历史总记录](/home/ra87racy/reports/ALL_EXECUTED_WORK_EXPERIMENT_CHANGE_RESULT_SUMMARY_20260910_ZH.md)。

### 全部已执行工作、实验调整与结果总记录

**核验截止时间：2026-09-10（Europe/Berlin）**  
**核验范围：`/home/ra87racy` 本地工作区**  
**原则：只记录有代码改动、命令输出、日志、manifest、预测文件、评估表、checkpoint、报告或清理清单能够证明已经实际发生的工作。**

---

#### 1. 这份总记录怎样区分“做过”和“没做过”

状态定义：

- **完成**：目标数据量和评估均完成，且有完整输出或结果表。
- **部分完成**：只完成生成、smoke、子集、训练或某一阶段，没有完成原定完整评估。
- **失败/中止**：实际启动过，但因环境、CUDA、数据、采样、运行时间或资源问题失败或被停止。
- **修改完成但无最终结果**：代码/环境/脚本确实修改过，但没有足够证据说明完整实验已经跑完。
- **审计/分析完成**：实际读取输出并计算了几何、体素、检测转移、采样方差等分析；不是新的模型训练结果。
- **未执行**：只出现在计划、空目录或可行性列表中，不计作实验结果。

本次扫描到 PointRCNN 主项目 `results/` 下 **132 个一级结果目录**。其中既有正式全量实验，也有 smoke、pilot、消融、故障诊断、可视化和报告生成。本文按“实验族”合并记录，避免把同一批预测的多个图表版本误算成多个独立实验。

##### 1.1 关键评价口径警告

1. 2026-09-08 至 09-10 的最终矩阵显式使用 **KITTI AP_R40**，每个 arm 必须有 **3,769/3,769** 帧才进入主表。
2. 较早的 PointRCNN 本地 evaluator 默认用 41 个 precision sample 中每隔 4 个取一个，实质是旧 **AP_R11 风格口径**。旧表保留为历史证据，但不能与最新 AP_R40 表直接比较。
3. 早期实验曾使用不同的 `RPN.NUM_POINTS`、distance proposal、输入采样、整帧/patch、点数上限和 checkpoint；它们不能被拼成一张“同协议排行榜”。
4. CenterPoint 以 voxel 为输入，PointRCNN 以固定点数采样为输入；文件层面的 exact 4× 不等于检测器内部真正使用了全部 4× 点。

---

#### 2. 当前工作的总体结论

截至 2026-09-10，最完整、证据最强的结论是：

1. **严格把点数扩大到 4×，没有自动恢复甚至提高 3D 检测精度。** 在冻结检测器下，所有正式上采样方法总体都低于对应 baseline，Line B（先降采样再上采样）尤其明显。
2. **主要问题不是单一的“点数不足”。** 已验证的原因包括：旧 patch 不局部、归一化错误、重复点、生成点激活大量新 voxel、PointRCNN 16,384 点入口采样、CenterPoint 每 voxel 最多 5 点/最多 40,000 voxels、以及预训练上采样分布与检测器训练分布不匹配。
3. **保留 observed 点并做 detector adaptation 能明显恢复性能，但仍没有超过配对 baseline。** 最新 full-val 最接近 baseline 的结果是 CenterPoint：
   - Line A，observed `N` + predicted `3N`，adapted：Car Moderate 3D AP_R40 **74.7012**，对 official baseline **79.2773**，差 **-4.5761**。
   - Line B，observed `M` + predicted `3M`，adapted：Car Moderate **61.6639**，对 adapted baseline **68.0490**，差 **-6.3851**。
4. **训练损失下降，但不能声称 3 epochs 已收敛。** 只有 epoch 3 checkpoint，没有 epoch 1/2 validation AP，也没有 plateau 或预先定义的 early stopping 证据。
5. **ModelNet40 目前只有 PU-GCN 小规模 exact-4× smoke；没有完整 12,311 样本本地结果，也没有 PointNet++ 分类准确率。** 这部分不能写成完整实验完成。

当前最终流水线状态：Line A 和 Line B 都是 **3,769/3,769** 点云完成，检测矩阵 **20/20 PASS**；核验时没有正在运行的 PU-GCN、PDANS、PointRCNN、CenterPoint、OpenPCDet 或 TULIP 研究进程。磁盘剩余约 **31–32 GiB**。

---

#### 3. 最终采用的严格 4× 协议

令原始 KITTI 点云为 `N` 点：

- **Line A**：原始 `N` 点直接上采样，最终严格为 `4N`。该线测试“对已有测量做增密”，不是信息恢复。
- **Line B**：先用固定 seed、无放回方式降为 `M=floor(N/4)`，再上采样到严格 `4M`。该线才是接近“稀疏输入恢复原始密度”的协议。
- **direct**：检测器直接读取严格 `4N`/`4M` 的网络输出。
- **observed-first**：完整保留 observed `N`/`M`，再从 strict 预测点中用稳定 seed 无放回抽 `3N`/`3M`，得到总计 `4N`/`4M`。

最终 PU-GCN patch 流程：

- patch 大小 `2048`，输出 `8192`；
- `fps_ball_cover_knn_v3`；
- Line A primary radius 2 m，Line B 4 m；cover radius 6 m；支持点阈值 2,048；
- patch 单独做中心化和单位球归一化，推理后反归一化；
- 合并所有 patch raw candidates 后，以固定 seed 无放回抽到 exact `4N`/`4M`；不足 exact target 时直接失败，禁止复制点补足；
- intensity 用预测 XYZ 到 observed XYZ 的 1-NN 继承；
- PU-GCN 固定使用作者 PU1K `model-100`，没有在 KITTI 上重新训练上采样网络。

---

#### 4. 最新完整结果：PU-GCN × detector adaptation × full KITTI val

##### 4.1 PointRCNN，Car 3D AP_R40，3,769 帧/arm

| Line | 输入 | 权重 | Easy / Moderate / Hard | 对配对基线的 Moderate 差值 | 状态 |
|---|---|---|---:|---:|---|
| A | baseline `N` | official | 92.4325 / **81.9528** / 77.8468 | 0 | 完成 |
| A | direct `4N` | official | 84.7443 / **61.9294** / 57.7869 | -20.0235 | 完成 |
| A | direct `4N` | adapted | 85.1671 / **68.8800** / 64.6697 | -13.0728¹ | 完成 |
| A | observed `N` + predicted `3N` | official | 85.0374 / **64.2748** / 60.0911 | -17.6780 | 完成 |
| A | observed `N` + predicted `3N` | adapted | 86.7068 / **71.0075** / 66.7939 | -10.9453¹ | 完成 |
| B | baseline `M` | official | 85.2016 / **65.5757** / 61.2831 | 0 | 完成 |
| B | baseline `M` | adapted | 84.3920 / **68.3312** / 64.2613 | 0 | 完成 |
| B | direct `4M` | official | 46.4657 / **30.0909** / 25.8000 | -35.4848 | 完成 |
| B | direct `4M` | adapted | 68.2592 / **46.6150** / 40.2487 | -21.7163 | 完成 |
| B | observed `M` + predicted `3M` | official | 55.0177 / **35.5289** / 30.9359 | -30.0469 | 完成 |
| B | observed `M` + predicted `3M` | adapted | 74.1970 / **55.2145** / 49.0583 | -13.1167 | 完成 |

¹ Line A 没有另外训练 adapted original-`N` baseline，所以这两个 adapted 差值以 official Line A baseline 为参照，同时改变了输入和权重，解释时应保守。

##### 4.2 CenterPoint，Car 3D AP_R40，3,769 帧/arm

| Line | 输入 | 权重 | Easy / Moderate / Hard | 对配对基线的 Moderate 差值 | 状态 |
|---|---|---|---:|---:|---|
| A | baseline `N` | official | 88.3907 / **79.2773** / 76.7371 | 0 | 完成 |
| A | direct `4N` | official | 81.8888 / **60.6081** / 57.9061 | -18.6692 | 完成 |
| A | observed `N` + predicted `3N` | official | 83.3121 / **63.2064** / 61.0418 | -16.0709 | 完成 |
| A | observed `N` + predicted `3N` | adapted | 86.8154 / **74.7012** / 72.6570 | **-4.5761** | 完成 |
| B | baseline `M` | official | 81.4003 / **64.5979** / 59.9701 | 0 | 完成 |
| B | baseline `M` | adapted | 83.7514 / **68.0490** / 64.6309 | 0 | 完成 |
| B | direct `4M` | official | 56.4211 / **35.3530** / 31.2043 | -29.2449 | 完成 |
| B | observed `M` + predicted `3M` | official | 66.9217 / **44.0929** / 40.0268 | -20.5050 | 完成 |
| B | observed `M` + predicted `3M` | adapted | 80.4109 / **61.6639** / 57.5252 | **-6.3851** | 完成 |

##### 4.3 CenterPoint 全类别复核

最优 observed-first adapted 也没有在 Pedestrian/Cyclist 上超过配对 baseline：

| Line | 类别 | baseline Moderate 3D AP_R40 | observed-first adapted | 差值 |
|---|---|---:|---:|---:|
| A | Pedestrian | 50.6533 | 48.4156 | -2.2377 |
| A | Cyclist | 64.6054 | 57.6989 | -6.9065 |
| B | Pedestrian | 47.4395（adapted baseline） | 44.0568 | -3.3827 |
| B | Cyclist | 42.0672（adapted baseline） | 33.5610 | -8.5062 |

##### 4.4 实际训练调整

**PointRCNN adaptation：**完整 3,712 个 train frame；每个 arm 独立训练；RPN 3 epochs + offline RCNN 3 epochs；batch size 1；workers 0；Adam one-cycle；LR `0.0002`；seed `20260823`；关闭 GT database augmentation；只保存 epoch 3。

**CenterPoint adaptation：**官方 80-epoch checkpoint 初始化；完整 3,712 个 train frame；batch size 2；workers 0；3 epochs；Adam one-cycle；LR `0.0003`；weight decay `0.01`；seed `666`；关闭 GT sampling；只保存 epoch 3。

训练 loss 的 E1→E3 变化：

- PointRCNN Line A direct：RPN `1.624055→1.347459`（-17.03%），RCNN `1.043027→0.959636`（-8.00%）。
- PointRCNN Line B direct：RPN -27.61%，RCNN -7.65%。
- PointRCNN Line B baseline：RPN -23.95%，RCNN -4.16%。
- PointRCNN Line A observed-first：RPN -16.46%，RCNN -4.42%。
- PointRCNN Line B observed-first：RPN -24.82%，RCNN -6.86%。
- CenterPoint Line A：`2.33→2.11`（-9.44%）；Line B PU-GCN：`3.69→3.10`（-15.99%）；Line B baseline：`3.00→2.65`（-11.67%）。

这些数字只证明优化过程正常完成且 loss 下降，**不证明 validation 已收敛**。

---

#### 5. 全量冻结检测器的统一 exact-4× 实验（2026-07）

##### 5.1 PointRCNN 全 3,769 帧旧统一表

以下是当时报告保存的数值；由于 evaluator/输入协议早于 09 月最终 AP_R40 runner，应当作为历史同批对照，不与第 4 节直接混比。

| 输入 | Easy / Moderate / Hard 3D AP | 结论 |
|---|---:|---|
| Original baseline | 92.2731 / **82.2554** / 77.9454 | 上界 |
| Downsampled baseline | 85.1766 / **65.7460** / 61.3818 | 降采样损失明显 |
| Line A PDANS | 84.6118 / **67.2235** / 62.3981 | 四方法中最好，但低于 original |
| Line A PU-GCN | 80.1727 / **58.2759** / 53.4719 | 下降 |
| Line A PU-EdgeFormer | 65.6774 / **44.9596** / 40.1919 | 大幅下降 |
| Line A PU-Net old | 11.7639 / **8.3802** / 7.4407 | 严重失效 |
| Line B PDANS | 65.0123 / **44.0752** / 39.1729 | 未恢复 downsample baseline |
| Line B PU-GCN | 45.9303 / **28.6803** / 24.1310 | 下降 |
| Line B PU-EdgeFormer | 33.7635 / **20.5693** / 17.6153 | 下降 |
| Line B PU-Net old | 12.4097 / **8.7807** / 7.7637 | 严重失效 |

##### 5.2 CenterPoint 全 3,769 帧，Moderate 3D AP_R40

| 输入 | Car | Pedestrian | Cyclist | 结论 |
|---|---:|---:|---:|---|
| Original baseline | 79.2773 | 50.6533 | 64.6054 | 原始基线 |
| Downsampled baseline | 64.5979 | 24.5690 | 15.4577 | 稀疏化显著损失 |
| Line A PDANS | 64.3574 | 45.1910 | 45.8953 | 四方法中总体最好，但仍低于 original |
| Line A PU-GCN | 58.8955 | 44.1470 | 44.3082 | 下降 |
| Line A PU-EdgeFormer | 45.7342 | 36.8205 | 35.2235 | 下降 |
| Line A PU-Net old | 10.7512 | 19.4930 | 13.1928 | 严重失效 |
| Line B PDANS | 46.7512 | 28.1790 | 15.5186 | Car 未恢复；Pedestrian 有相对稀疏基线改善 |
| Line B PU-GCN | 39.0714 | 24.9302 | 16.8458 | 总体未恢复 |
| Line B PU-EdgeFormer | 24.7476 | 15.5896 | 7.0374 | 下降 |
| Line B PU-Net old | 11.7178 | 11.9174 | 5.0172 | 严重失效 |

##### 5.3 全量方法执行状态

- **PDANS**：最初因 extension source / `pytorch3d` 问题失败，后修复环境和设备相关代码，最终完成 Line A、Line B strict exact-4× 与双检测器评估。
- **PU-GCN**：修复 TensorFlow/CUDA op 编译，完成正式两条线全量生成和评估；之后又完成 detector adaptation 全矩阵。
- **PU-EdgeFormer**：原仓库环境直接复现失败；后来通过复用兼容的 PU-GCN ops/checkpoint 推理路径完成统一协议结果。该事实必须写清，不能声称是未经适配的原仓库一键复现。
- **PU-Net**：进行了 Python 2→3、TF custom op、CPU fallback、编译与归一化修复；旧适配器结果极差，修复后小规模结果改善，但仍未成为最佳方法。
- **TULIP**：完成 Line A/Line B 全 3,769 帧推理与 PointRCNN/CenterPoint 评估，但输出不是 strict exact-4×，且 PointRCNN 使用过 `RPN=4096`、关闭 distance proposal，不能并入统一四方法主表。
- **EAR**：只完成早期全量/历史输入评估和后来的 5 帧 strict smoke；没有完成新的 strict Line A+Line B 全量重生成。
- **SPU-PMD**：只完成 4 帧 smoke 和 PointRCNN 小样本尝试，无 full-val AP。

---

#### 6. 关键实验调整、为什么改、改后发生了什么

| 调整 | 实际修改/运行 | 结果 |
|---|---|---|
| 从“点数大约增加”改成 strict exact 4× | Line A `N→4N`；Line B `floor(N/4)→4M`；manifest 逐帧核验 | 消除了方法输出点数不一致，但检测性能仍未随点数提升 |
| 旧 sequential/coarse-bin patch → 局部 FPS/ball/kNN | 先后测试 FPS、ball、cover、`fps_ball_cover_knn_v3` | patch 局部性提高；发现首个 local 版本有 59.4% 重复 row |
| 加 cover 和 unique kNN floor | surface-c32 / cover radius 6 m / unique 2048 support | 2048 支持唯一、重复归零，几何覆盖改善；检测仍未全面超过 baseline |
| PU-Net 单位球归一化修复 | 2×2：wrong/fixed normalization × old/local patch | PointRCNN 最好从错误配置的个位数提高到 43.5424；CenterPoint 提高到 46.2047，但仍差 |
| PointRCNN sampler-safe | far points ≥16,384 时改为从全部候选无放回采样；稀疏时允许 replacement padding | 所有曾因负 sample size 失败的 Line A 完成；fallback 占比 <0.4%；修复崩溃但不能解释全部 AP 下降 |
| PointRCNN E3 observed-first 32,768 | observed 点优先，保留实测点后再放预测点 | Line A 有一定改善，Line B 几乎没有；说明入口 cap/顺序不是唯一原因 |
| CenterPoint voxel 审计 | 统计 occupied voxels、max-5 retention、40,000 cap | Line A 中 PU-GCN/PUEF 等大量触发 voxel cap；Line B 不触发 cap，仍下降，故 cap 不是唯一原因 |
| direct no-resample | 在独立 PointRCNN 副本中让网络读取所有有效点，不固定 16,384 | 连 original 都从 Moderate 81.95 降至 74.91；上采样方法仍差，证明简单去掉 resample 不是修复 |
| region split + 3 m halo | 2,067 个共享 core region，核心点无 16,384 截断，三类 OpenPCDet PointRCNN | 某些 Ped/Cyc 指标改善，但全方法 exact-4× pilot 仍全部低于 baseline 主指标 |
| detector-aware PDANS V2 | voxel-only、proposal-only、full；PointRCNN 与 CenterPoint 256 帧 | 跨检测器 gate 失败；CenterPoint Ped 改善但 Car/Cyc 退化 |
| V3 true internal proposals | PointRCNN pre-RCNN RPN proposals；CenterPoint pre-NMS heatmap top-K | 与 V2 接近，证明 final-box bias 不是主要问题 |
| V4 measured-voxel anchor | 禁止 generated-only voxel 激活 | PointRCNN Car Moderate 3D +0.3093；CenterPoint Car -0.0192，但 Ped/Cyc 仍略退化 |
| V5 predicted-class adaptive gate | 只给 CenterPoint 预测为 Car 的 proposals 增点，PointRCNN 沿用 V4 | pilot256 基本保持 CenterPoint 全类别 baseline；独立 holdout64 的 CenterPoint Car 轻微下降，未形成稳定跨检测器提升 |
| observed-first | 完整保留 observed，再补 3N/3M predicted | 冻结模型有改善但仍差；配合 full-train adaptation 后恢复最明显 |
| detector adaptation | 先 64 帧筛选，再 3,712 帧训练，最后 3,769 帧完整验证 | 大幅缩小差距，但最终无 arm 超过配对 baseline |

---

#### 7. 根因实验和消融分析

##### 7.1 旧 patch 的非局部问题

- 旧 extractor 使用粗空间 bin 后按行切块，不保证同一 patch 是局部表面。
- Line B patch 的 XY 对角线中位数 **30.46 m**，p90 **124.10 m**，明显违背训练时局部 object patch 的分布。
- 第一个 local patch 版本改善局部性，但出现 **59.4% duplicate rows**。
- 对该版本：PointRCNN Moderate 从 **57.1829 降到 43.6976**；CenterPoint 从 **61.1172 提高到 77.4150**。同时 occupied voxels 中位数从 **11,584.5 降到 2,124.5**，说明不同检测器对点分布的偏好不同。

##### 7.2 PU-Net 归一化 × patch 的 2×2 因果消融（256 帧）

| 配置 | PointRCNN Moderate 3D | CenterPoint Moderate 3D |
|---|---:|---:|
| baseline | 81.3033 | 79.8368 |
| wrong norm + old patch | 7.9823 | 15.9788 |
| fixed norm + old patch | 18.2742 | 41.2227 |
| wrong norm + local patch | 6.3008 | 16.4119 |
| fixed norm + local patch | **43.5424** | **46.2047** |

结论：归一化错误与 patch 非局部都是真实问题，修复二者能显著恢复，但仍无法接近 baseline。

##### 7.3 PointRCNN / CenterPoint 输入瓶颈

- PointRCNN 默认入口固定 16,384 点。
- CenterPoint voxel size `[0.05,0.05,0.1]`，每 voxel 最多 5 点，test 最多 40,000 voxels。
- 32 帧 Line A 审计的 median voxels / 触发 40k cap 帧数：original `14,944 / 0`；PDANS `36,913 / 6`；PU-GCN `45,108 / 29`；PU-EdgeFormer `46,182 / 29`；PU-Net `55,384 / 32`。
- Line B 没有触发 voxel cap，却仍有明显性能下降。因此 cap 是 Line A 的重要问题，但不是全局唯一解释。

##### 7.4 direct no-resample 全量实验

| 输入 | PointRCNN Car 3D Easy / Moderate / Hard |
|---|---:|
| standard original，固定 16,384 | 92.26 / **81.95** / 77.75 |
| original full all-valid | 79.83 / **74.91** / 72.73 |
| Line A PDANS all-valid | 83.59 / **63.49** / 54.34 |
| Line A PU-GCN all-valid | 78.21 / **54.26** / 47.11 |
| Line A PU-EdgeFormer all-valid | 67.11 / **43.85** / 37.10 |
| Line A PU-Net all-valid | 12.23 / **8.82** / 8.17 |

结论：网络训练时依赖固定采样分布；把所有点直接塞入模型甚至损害 original，不能把“取消采样”当作恢复方案。

##### 7.5 dose / 生成点比例

在 256 帧上测试 `g2.5/g5/g7.5/g10`：

- Line A PDANS Moderate 约 `85.10 / 82.37 / 82.25 / 82.42`，对应 baseline `82.66`；小 dose 曾出现子集增益。
- Line B 各 dose 均低于对应稀疏 baseline。
- 全 3,769 帧 `g10`：Line A PDANS/PU-GCN/PU-EdgeFormer/PU-Net Moderate 分别约 `79.951/79.478/76.433/72.371`；Line B `59.048/55.954/52.415/45.700`。
- `g25/g50` 没有完成，不能报告为已验证。

##### 7.6 detector-aware PDANS V2–V5（均为 256 帧开发实验）

**V2：**

- PointRCNN baseline 81.1768；V1 full 82.3995（+1.2227），但 BEV -1.1421；V2 voxel-only 79.3805，proposal-only 75.7533，full 79.2269。
- CenterPoint V2 full：Car 76.7736（-3.0632），Ped 48.5569（+5.0476），Cyc 72.2834（-3.8283）。
- 结论：只改善 Ped，不是 detector-wide 改善；不扩到 3,769。

**V3：**

- PointRCNN V3 full 79.2933（相对 baseline -1.8835）。
- CenterPoint V3 full：Car 76.7396（-3.0972），Ped 48.5887（+5.0794），Cyc 72.2856（-3.8261）。
- V2/V3 的 voxel 与检测转移几乎相同；internal proposal 替代 final boxes 没有解决问题。

**V4：**

- PointRCNN Car 81.4861（+0.3093），BEV +0.2093。
- CenterPoint Car 79.8176（-0.0192），Ped 43.0727（-0.4366），Cyc 75.7678（-0.3439）。
- 新 active voxel 中位数降到 0，证明 generated-only voxel activation 是可控因果因素；但严格跨类别 gate 未通过。

**V5：**

- PointRCNN 沿用 V4：Car +0.3093。
- CenterPoint：Car -0.0184，Ped +0.0004，Cyc +0.0000，基本回到 baseline。
- V5 对应 256 个输入文件真实存在并评估完成，但它是方法开发集。
- 独立 holdout64 的 PDANS surface-c32 **64/64 点云生成完成**，CenterPoint baseline/V5 的检测与 AP_R40 也在单独结果目录完成：Car Moderate `74.1881→74.0512`（-0.1369），BEV `86.3893→85.2910`（-1.0983）；Pedestrian、Cyclist 与 baseline 完全相同。结果没有复现正向 detector-wide 增益。

##### 7.7 object-preserving 与方法组件消融（PointRCNN，pilot256）

V1 workspace 中实际跑了 8 个 selector/component 变体。Car Moderate 3D AP_R40：

| 变体 | Moderate |
|---|---:|
| baseline | 81.1768 |
| density-only | 79.8568 |
| fixed 2.5% PDANS | 81.4430 |
| full | **82.3995** |
| matched-real dynamic | 81.7996 |
| NN-only | 81.5914 |
| no-adaptive | 80.1478 |
| no-confidence | 81.6421 |
| no-sparse | 80.9414 |

单一 seed 下 full 看似 +1.2227；后来的 16-seed 探针证明该增益不稳定，见 7.10。

另两组 patch-pair 对照也实际完成：

- PDANS：old patch 57.9933，surface-c32 68.4744；PU-GCN：old patch 55.8076，surface-c32 63.4482；同批 baseline 81.5569。
- PU-EdgeFormer：old patch 41.7379，surface-c32 58.0212；PU-Net fixed：old patch 16.8097，surface-c32 22.9283，cover-kNN 44.0985；同批 baseline 81.7441。

结论：surface/local patch 对所有方法都有实质改善，但不足以达到 baseline。

##### 7.8 region-split 三类别 exact-4× pilot256

共同使用 2,067 个 core regions、3 m halo、零 core point loss、每个 detector input 不超过 16,384。Moderate 3D AP_R40：

| 方法 | Car | Pedestrian | Cyclist |
|---|---:|---:|---:|
| baseline | 77.8161 | 58.2090 | 78.2919 |
| PDANS surface-c32 | 68.8726 | 58.0672 | 71.5795 |
| PU-GCN surface-c32 | 61.5081 | 55.4667 | 57.4759 |
| PU-EdgeFormer surface-c32 | 61.2790 | 56.1431 | 56.1780 |
| PU-Net fixed surface-c32 | 26.1774 | 25.0164 | 28.9221 |

所有方法的主要 3D 指标仍未全面超过 baseline；PDANS 最接近。

V4 的独立 holdout64 region-split 也实际评估完成：Car/Pedestrian/Cyclist Moderate 3D 分别 `75.7267→77.4777`、`38.1070→45.8565`、`15.0000→17.2222`。这些差值只适用于相同 region-split 协议；holdout 中 Ped/Cyc GT 很少，不能替代整帧 full-val。

同一 V4 在原生整帧 OpenPCDet PointRCNN 下显示明显采样敏感性：确定性 `workers=0` 的 holdout64 Car +4.0651、Ped -0.8332、Cyc -3.9286；原生 `workers=2` 则 Car -2.2718、Ped +2.5279、Cyc +1.6234。因而不能声称跨采样实现稳定提升。

##### 7.9 Line B c2048 / E1 / consensus / random 选择实验（pilot256）

PointRCNN Car Moderate 3D：

| 选择协议 | baseline | PDANS | PU-GCN | PU-EdgeFormer | PU-Net fixed |
|---|---:|---:|---:|---:|---:|
| c2048-r4 | 64.0416 | 36.6162 | 24.9619 | 21.2409 | 2.3971 |
| E1 | 64.0362 | **44.0487** | 32.8364 | 28.2585 | 10.5793 |
| consensus | 63.4197 | 39.2524 | **33.1648** | 22.4376 | **18.8792** |

CenterPoint 同一 Line B pilot 的 Car Moderate：baseline 61.6558；random PDANS/PU-GCN/PU-EdgeFormer/PU-Net 为 `47.7441/41.3784/35.4329/15.1487`；consensus 为 `43.6085/37.3131/27.3251/19.8041`。选择策略会改变方法排序，但没有一组恢复 baseline。

##### 7.10 PointRCNN 16-seed 采样噪声探针

- 固定输入、checkpoint、split 和合并规则，只改变 evaluator sampling seed，运行 seed 0–15。
- 新增的 24 个 slot 全部 PASS；6 次 CUDA 偶发 segmentation fault 被内置重试吸收。
- Car Moderate 3D：baseline `80.998±1.198`；V1 full `80.560±1.086`；配对差值均值 **-0.438**、sd **1.769**，范围 `-3.51…+2.23`。
- 原先单 seed 的 +1.2227 是 full 臂 16 次中的最大值；9/16 差值为负，median -0.75。
- `MDE95≈3.47 AP`。这证明 split-region 路径下约 0.4–1.2 AP 的排序不可读；不能把该噪声数值直接套到 native E2 full-val。

##### 7.11 exact-4× quality selection 小探针

- 实际对 Line A PDANS 做过 confidence selection：no-quota 3 帧、voxel quota=6 为 5 帧。
- no-quota：quality precision 中位数 0.6821，低于 random 0.8155；unsupported 比例 0.5825，高于 random 0.4940。
- quota=6：quality precision 0.5965，低于 random 0.8024；unsupported 0.6329，高于 random 0.5337。
- 这是 3/5 帧的机制探针，没有 detector AP；结果否定了当时的 confidence quality selector，未扩展。

---

#### 8. 训练规模递进：64 帧筛选 → 全 train pilot → full val

##### 8.1 PointRCNN finetune64 六臂筛选

- 固定 64 个 train frame，实际 63 个有效；seed `20260823`。
- RPN 3 epochs + RCNN 3 epochs，每阶段每 epoch 93 steps。
- 6 个输入 arm × pretrained/finetuned，共 12 个 256-frame eval，全部生成了 256 份预测并 PASS。
- 新版汇总中的 Moderate 变化：Line A PU-GCN `60.2662→64.4408`（+4.1746）；Line B PU-GCN `32.9201→37.5150`（+4.5949）；Line B PDANS `43.0067→47.4327`（+4.4260）；Line B baseline `66.6188→62.9139`（-3.7049）。
- PDANS Line A 在两个报告版本中数值有差异；应以较晚的 eval256 汇总为准，不能挑选更有利的旧数。
- 结论：极小训练集能显示“域适配有恢复潜力”，但 baseline 也退化，不能作最终结论。

##### 8.2 完整 3,712 train 的 256-val pilot

训练使用完整 train split，但当时只评估固定 256 val：

- PointRCNN A PU-GCN adapted 67.2345，对 baseline 79.1910，差 -11.9565。
- PointRCNN B PU-GCN adapted 47.0204，对 adapted baseline 67.2485，差 -20.2281。
- CenterPoint A adapted 73.9468，对 79.8368，差 -5.8900。
- CenterPoint B adapted 60.6978，对 65.5501，差 -4.8523。
- PointRCNN observed-first：A 67.7909，比 direct +0.5564；B 56.2443，比 direct +9.2239。

这些是 256-frame pilot，已被第 4 节的 full 3,769-frame 结果替代，但仍是实际跑过的递进证据。

---

#### 9. 早期基线、方法接入和失败记录（2026-05 至 06）

##### 9.1 PointRCNN / EAR / PU-Net 早期工作

- 历史 PointRCNN baseline 记录：3D AP `89.19/78.85/77.91`。
- 一次 clean original rerun：`89.2040/78.6795/77.8099`。
- EAR 历史报告：`88.5558/77.9565/76.7821`，略低于 baseline。
- 早期 PU-Net x2 全量生成与评估完成一次：3D AP `0.1976/1.1364/1.1364`，说明旧适配完全不兼容。
- 后续 recovered/fullframe 报告又出现 PU-Net Moderate 约 59.2885；由于输入和恢复流程不同，与上面的极低结果不是同协议，已作为 provenance 风险保留。
- 为 PU-Net 增加了 resume/chunk、Python 3、TensorFlow op 编译、GPU bootstrap、CPU fallback、数据读取和归一化修复。

##### 9.2 May 16 一批非统一配置结果

记录的 Moderate 3D：original 77.93，downsample50 76.99，EAR 73.89，PU-Net 35.26，PU-GCN 50.65。PU-GCN 当时用过 `RPN=26000`，配置不统一，因此只保留为早期探索。

##### 9.3 RPN4096 / no-distance-propose 比较

- original 在 4,096 点配置下因负维度采样失败。
- downsample 完成：3D `83.6938/65.1662/60.4608`。
- EAR 完成：`81.5120/62.6687/57.8478`。
- PU-Net 在 16 个样本后失败。
- PU-GCN 当时缺 8 帧，未完成。
- TULIP 完成：`27.5045/16.9675/13.7001`。

##### 9.4 TULIP

- 完成 full Line A 解析结果约 `54.3492/35.0242/30.3302`；另有 Line B 全量和 CenterPoint 全量结果。
- CenterPoint Moderate：Line A Car/Ped/Cyc `31.77/15.98/4.59`；Line B `17.68/2.99/0.08`，远低于 original `79.28/50.65/64.61`。
- 原生输出是 range-image 垂直 4×，但转换回 XYZ 后不保证 exact 4×；实际输出比约 0.41× 和 0.73–0.78×。
- 8 个已知帧在 4,096/2,048/1,024 配置下会因 distance proposal 的 far/near bucket 空集进入 CUDA NMS 而中止。后续加入 sole-nonempty-bucket guard，并使用 score-only proposal 绕过。

##### 9.5 EAR strict-4× 可行性

- 5 帧 × Line A/B，共 10 个条目 smoke 全部 PASS。
- CPU 整帧实现包含全局 kNN/PCA 与 Python 循环；Line A 约 26–29 分钟/帧，Line B 约 9–10 分钟/帧。
- 估计两条线串行需约 92–103 天；还观察到一帧 byte-identical duplication。
- 因运行代价没有完成新的 strict full-val 重生成；这是**运行时间中止/不可行**，不是模型最终负结果。

##### 9.6 SPU-PMD

- 第一次因缺 `pyvista` 失败；第二次因 `knn_cuda` 失败。
- 修改 `utils/MeshUtil.py` 与 `main.py` 为 lazy import 后，4 帧 inference 成功；每个完整帧内部先降到 2,048，再输出 8,192，finite，即相对内部输入新增 3× 点。
- PointRCNN `RPN=4096` 评估因 sampler underflow 失败；改为 `RPN=2048` 后 4 帧跑完，detections 为 `1/0/0/2`。
- 样本太少，没有 AP；无 full-val。

##### 9.7 PDANS 初期失败与后续修复

- 初期 inference 因缺 extension source 和 `pytorch3d` 未跑通；`pointops` 曾单独编译成功，准备了 4 个 XYZ 输入。
- 修改 pointnet2 的 device/tensor 处理、编译产物和 CUDA 兼容后，后续完成正式 Line A/Line B 全量。

##### 9.8 PU-EdgeFormer 初期直接复现失败

- 建立 Python 3.6.8 / TensorFlow 1.13.1 环境。
- custom ops 因 CUDA 10 路径硬编码、无 `nvcc` 和 checkpoint 缺失失败。
- 只把 3 个 KITTI frame 转成 2,048 点并生成 Plotly，可视化完成，但当时没有模型 inference。
- 后来正式统一结果来自兼容 ops/reuse 路径；两阶段必须分开表述。

##### 9.9 ratio audit

实际审计发现：

- EAR 约 `1.008×`，不是目标 x2。
- PU-Net 输出比例随帧变化。
- PU-GCN/PDANS 旧流程受 100k cap 影响。
- TULIP 在 range image 内垂直 4×，但回到 XYZ 后可能比输入点还少。

该审计直接推动了 strict exact-4× Line A/B 协议。

---

#### 10. ModelNet40 工作记录

##### 10.1 已实际完成

- 建立 `modelnet40_pointnet2_upsampling` 协议与报告结构。
- PU-GCN 本地 smoke：Line A 8/8，`1024→4096`；Line B 8/8，`256→1024`；输出 finite，点数 exact。
- 已编写/修改 ModelNet40 数据预处理、上采样适配、分类训练入口、审计和报告文件。
- `thesis_demo` 中建立了 `train_cls.py`、`pointnet_cls.py`、dataset/preprocess scaffold。

##### 10.2 没有完成，不能写成结果

- 两条线各 12,311 个样本的完整本地 PU-GCN 生成没有提交/没有完成证据。
- PointNet++ 5 个分类分支为 `NOT_SUBMITTED`，没有分类 accuracy。
- 没有完整的 geometry benchmark、分类训练曲线或最终 ModelNet40 对照表。
- `dgcnn_cls.py`、`train_upsampling.py`、部分 metrics/visualization/README 曾为空文件；属于 scaffold，不属于已运行实验。
- HPC 上可见的 EAR/PDANS/PU-Net 12,311 文件若没有本地作业日志，只能记作“已有产物”，不能在本记录中断言是当前工作区本次亲自跑出的任务。

---

#### 11. HPC 迁移、运行和资源工作

##### 11.1 KITTI 上采样迁移包

- 准备过 10 个 variant × 3,769 帧、约 42 GiB 的迁移/提交结构、checksum、job script 和说明。
- dry run 真实执行：`tinyx` DNS 解析失败；`tinyx.nhr.fau.de` SSH authentication 失败。
- 因此没有实际完成远端传输，也没有远端正式作业结果。

##### 11.2 Conda/运行环境

实际建立过独立环境：`ear`、`openpcdet_centerpoint`、`pointrcnn_old`、`puedgeformer`、`pugcn`、`punet_tf`、`tulip`、`upsampling_basic`。这些环境支撑了不同年代 TensorFlow/PyTorch/CUDA 依赖的复现与修复。

---

#### 12. 磁盘清理和数据治理记录

##### 12.1 2026-05-18 安全清理

- 删除空/smoke 目录、PU-GCN evaluation log、PU-Net preparation log 和 `__pycache__`。
- 实际磁盘变化约 420 MiB；有执行日志。

##### 12.2 2026-06-26 第一阶段

- 清理 pip cache 约 13 GiB、conda clean 约 4.1 GiB；`df` 显示已用空间约减少 16 GiB。
- 将 6 个 PU-GCN 巨型中间目录移到 `~/TO_DELETE_REVIEW`，约 193 GiB；该阶段是移动待复核，不等同于立刻永久删除。
- 后续删除清单记录了约 1,487,400 个 `.xyz`、29,128 个 `.png`、22,573 个 `.json`、22,349 个 `.bin`、21,966 个 `.csv`、7,201 个 `.md` 中间文件路径。

##### 12.3 2026-07-31 用户授权的大清理

- 清理旧 strict line trees、CenterPoint reconstructed inputs、8 组 raw patch outputs、旧 Line B cap100k、`thesis_demo` venv、cache/VSCode backup 等。
- 报告记录释放 **459 GiB**（492,379,832,320 bytes）。
- 核验保留了 8 组 `merged_raw`、8 组 `final_bin`、checkpoint、源码和当前结果。

当前文件系统为约 1.0 TiB，总用量约 993 GiB，剩余约 31–32 GiB，仍处于高占用状态。

---

#### 13. 实际代码修改记录

##### 13.1 PointRCNN 主仓库

当前分支：`experiment/centerpoint-unified-line-a-b`。跟踪文件仍有本地未提交修改：**4 files，72 insertions，8 deletions**。

- `lib/config.py`：`yaml.load` 改为 `yaml.safe_load`。
- `lib/datasets/kitti_dataset.py`：NFS/并发读取增加最多 20 次重试、字节数与 short-read 检查。
- `lib/datasets/kitti_rcnn_dataset.py`：修复 far points 超过固定采样数时的负 sample size；稀疏输入安全补点。
- `lib/rpn/proposal_layer.py`：far-only / near-only 空 bucket guard，避免 TULIP/region-split 情况下对空张量做 CUDA NMS。
- 新增大量未跟踪脚本：strict-x4、patch extraction、各方法 wrapper、E1/E2/E3、CenterPoint、detector-aware V2–V5、region split、训练、全量评估、分析、可视化、打包和恢复脚本。

##### 13.2 PU-Net

当前 diff：**35 files，226 insertions，122 deletions**，另有重新编译的 `.so`/`.o`。

- Python 2→3 兼容；TensorFlow API 调整。
- sampling/grouping/interpolation/CD/EMD custom op ABI 与 CUDA 编译脚本修复。
- CPU fallback 与 GPU bootstrap。
- 数据读取、provider、model utils、归一化和 full-frame adapter 修复。

##### 13.3 PDANS

当前跟踪 diff：**2 files，5 insertions，4 deletions**。

- `pointnet2/util.py`、`pointnet2_utils.py` 的 device-aware tensor/CUDA 处理。
- 增加 checkpoint 和 pointops 编译产物。

##### 13.4 SPU-PMD

当前 diff：**9 files，18 insertions，42 deletions**。

- lazy imports，移除/替代不可用依赖路径。
- pointnet2 C++/CUDA 扩展头文件与 source 兼容修复。
- operations 和 utility 调整；保存了 model/extension 构建产物。

##### 13.5 OpenPCDet / CenterPoint

当前 diff：**3 files，21 insertions，7 deletions**，新增 1 个工具脚本。

- dataset import 的可选依赖处理。
- data processor 的点输入/voxel 相关兼容。
- detector template 的 checkpoint `weights_only=False` 等恢复兼容。
- 新增 `create_kitti_val_infos_only.py`。

##### 13.6 PU-GCN

- `tf_ops/compile.sh`：**14 insertions，3 deletions**，修复本机 TensorFlow/CUDA op 编译路径。
- 增加 PU1K pretrained checkpoint、本地备份和 strict/full-frame wrapper。

##### 13.7 TULIP / PU-EdgeFormer

- TULIP 源仓库跟踪文件基本未改，但新增本地 `scripts/`、`results/` 和缓存。
- PU-EdgeFormer 的正式实验主要通过 ops-reuse 兼容副本接入；原仓库初次 custom-op 编译失败的证据仍保留。

##### 13.8 版本控制状态风险

PointRCNN 主仓库的大部分实验脚本、报告、结果、权重、备份和工具仍是 **untracked**；Git 历史主要是上游旧提交，不能仅依赖 `git log` 还原这段工作。当前可复现性依赖工作区文件、结果 manifest 和报告。09 月实验已另外打包 **151 个源码/CUDA/配置/split 文件**，并生成 SHA-256 manifest，降低了这一风险。

---

#### 14. 分析、可视化、汇报和论文材料

以下均有实际生成文件：

- 2026-08-04：全实验 master inventory（中文）。
- 2026-08-08：上采样失败研究汇报，中文、英文和详细英文版本。
- 2026-08-09/10：systematic report 中英文 deck。
- 2026-08-13：英文 progress report PDF/PPT。
- 2026-08-17：42 页 advisor complete PPT/PDF、详细 defense report 和 talk track。
- 2026-08-18：lost-car detected→missed 截图证据。
- 2026-08-28：PU-GCN full retraining presentation。
- 2026-08-26：Chapter 2 expanded，MD/HTML/PDF。
- 2026-09-02：Chapter 3/4 KITTI，MD/HTML/PDF。
- 2026-09-05：Chapter 3/4 ModelNet40+KITTI integrated，MD/PDF。
- 2026-09-08：Chapter 5/6 detailed，MD/PDF，约 35 张图与 figure manifest。
- 建立过 object crop、Open3D、Plotly、BEV、检测框、帧转移、lost-car、dual-detector、几何/体素分布等多批可视化。

注意：09-08 以前的 thesis chapter 使用的是较早 256-frame adaptation 结果；它们**尚未自动更新成 09-10 的 3,769-frame 20/20 最终矩阵**。

---

#### 15. 方法可行性调查：做过调查，但没有模型实验

2026-08-12 的 detection-oriented triage 实际检查了 PUDet、GFAS、PDANet、DAPU、TULIP 等：

- PUDet/GFAS：没有找到可执行公开代码。
- PDANet：有代码但没有可用权重。
- DAPU：没有公开可直接运行代码，只能自行实现。
- TULIP：已经实际跑过，不再作为新候选。

同时对现有预测做了 200 次重复子采样，估计配对差值的不确定性：

| 帧数 | 差值标准差 | 约 95% 最小可检测差异 |
|---:|---:|---:|
| 20 | 6.02 | 11.80 |
| 50 | 3.71 | 7.27 |
| 100 | 2.87 | 5.63 |
| 256 | 1.75 | 3.44 |
| 512 | 1.26 | 2.46 |

结论：20-frame 只能排除很大的失败，不能可靠证明小幅提升。PUDet/GFAS/PDANet/DAPU 在本工作区**没有正式 inference/AP**，不得列入已跑方法表。

---

#### 16. 明确失败、未完成或不能证明的事项

1. ModelNet40 full 12,311 × 两条线：未完成。
2. PointNet++ 五分支分类训练和 accuracy：未提交、无结果。
3. EAR 新 strict Line A/B full-val：因 CPU 运行时间不可接受未完成。
4. SPU-PMD full-val：未完成，只有 4-frame smoke。
5. PU-EdgeFormer 原环境一键复现：custom ops/checkpoint 阻塞；后续是兼容路径结果。
6. HPC KITTI 迁移：DNS/SSH authentication 阻塞，没有完成远端传输。
7. dose `g25/g50`：未完成。
8. V5 disjoint holdout64：64/64 点云、CenterPoint baseline/V5 AP 都已完成，但没有复现正向增益；OpenPCDet 三类 holdout 对采样实现敏感，未通过“稳定 detector-wide 改善”判据。
9. 当前唯一确认的空一级结果目录是 `pugcn_metrics_batch_filter_range_dedup_voxel_003`；空目录不计作实验。名称带错误日期后缀、实际不存在的路径也不计入记录。
10. PUDet、GFAS、PDANet、DAPU：只做过可行性调查，没有正式模型结果。
11. 3-epoch detector adaptation convergence：没有被证明。
12. 最新 `protocol_and_code_zh.md` 末尾仍保留“Line A 运行中”的旧状态段；应以同目录 09-10 更新的 `current_progress.md` 和 `full_val_detector_matrix.md` 为准，后者已经 20/20 完成。

---

#### 17. 按时间的总工作线

| 时间 | 已发生的工作 | 状态 |
|---|---|---|
| 2026-02 至 03 | PointNet/ModelNet/KITTI scaffold、数据脚本、初始检测/上采样工程结构 | 修改完成；多数无最终实验 |
| 2026-05-03 至 05-08 | PU-Net、EAR、PointRCNN baseline；PU-GCN 环境、smoke、full-frame 恢复 | 完成/失败混合，形成首批 AP 和兼容性问题 |
| 2026-05-12 至 05-19 | fair comparison、RPN4096、TULIP、PU-EdgeFormer 可行性、环境审计 | 部分完成；多项配置不统一或失败 |
| 2026-06-06 至 06-15 | TULIP full/audit、PDANS/SPU-PMD 接入、HPC 迁移尝试 | TULIP 完成；SPU-PMD smoke；迁移失败 |
| 2026-06-20 至 06-30 | ratio audit、严格 x4 设计、可视化、第一轮大清理 | 审计/修改完成 |
| 2026-07-03 至 07-18 | 四方法 strict Line A/B、PointRCNN full eval、sampler-safe、E1/E2/E3 | 主实验完成 |
| 2026-07-19 至 07-31 | CenterPoint full、dose、direct no-resample、patch 根因、PU-Net 2×2、surface-c32 | 完成/部分完成；获得主要因果证据 |
| 2026-08-04 至 08-11 | surface all-method pilot、region split、detector-aware PDANS V2–V5、Line B c2048 系列 | 256-frame 开发/诊断完成；V5/region-split holdout64 也完成，但未得到稳定跨协议增益 |
| 2026-08-12 至 08-18 | 新方法 triage、采样方差、系统汇报、lost-car 证据 | 分析/汇报完成 |
| 2026-08-24 至 08-28 | finetune64、完整 3,712-train 的 PU-GCN detector adaptation、observed-first pilot | 训练完成，256-val pilot 完成 |
| 2026-09-02 至 09-08 | 论文章节 3–6、图表和来源清单 | 文档完成，但部分引用旧 pilot |
| 2026-09-08 至 09-10 | PU-GCN 两条线 full 3,769 生成、20-arm 双检测器 full-val、全类别 AP、收敛审计、源码打包 | **20/20 完成** |

---

#### 18. 最终可写入论文的严谨结论

可以写：

> 在严格点数控制的 KITTI Line A/Line B 协议下，通用预训练点云上采样模型能够生成 exact-4× 输入，但冻结的 PointRCNN 和 CenterPoint 并未因此获得稳定检测增益。局部 patch、归一化、输入采样、voxel activation 和检测器训练分布都会显著影响结果。保留 observed 点并针对新输入分布进行 detector adaptation 能明显缩小性能差距；然而在 3,769-frame full validation 上，当前最佳配置仍低于其配对 baseline。因此，本实验支持“点数增加不等于任务信息恢复”，而不支持“上采样已经提升最终检测精度”。

不能写：

- “所有上采样方法都在相同原生代码、相同 checkpoint、相同 evaluator 下直接公平运行”——PU-EdgeFormer、PU-Net、TULIP 都有兼容路径或协议差异。
- “3 epochs 已收敛”——没有证据。
- “V5 在独立 holdout 上稳定提升”——AP 已跑，但 CenterPoint Car 轻微下降，三类 OpenPCDet 结果又对采样协议敏感。
- “ModelNet40 分类实验已完成”——没有 accuracy。
- “PU-GCN 在 KITTI 上训练了 3 epochs”——3 epochs 是 detector adaptation；PU-GCN 固定作者 PU1K model-100。
- “3N/3M 是模型置信度最高或几何最好的新点”——目前只是稳定 seed 的均匀无放回抽样。

---

#### 19. 主要证据入口

- 08-04 总清单：`/home/ra87racy/projects/baseline_detectors/PointRCNN/reports/THESIS_EXPERIMENT_MASTER_INVENTORY_20260804_ZH.md`
- 最新进度：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/current_progress.md`
- 最新 20-arm 主表：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/full_val_detector_matrix.md`
- 最新全类别 AP：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/all_classes_ap_r40.csv`
- 收敛审计：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/convergence_audit.md`
- 完整协议：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/protocol_and_code_zh.md`
- 151-file 源码快照说明：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_detector_adaptation_full_val_20260908/reports/experiment_source_full.md`
- 256-frame full-train pilot：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pugcn_full_retrain_20260824/reports/full_retraining_report.md`
- finetune64：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pointrcnn_finetune64_six_arms_20260824/reports/finetune64_screen_report.md`
- direct no-resample：`/home/ra87racy/projects/baseline_detectors/PointRCNN_direct4n_noresample/experiments/direct4n_noresample/runs/full_val_frozen_direct4n_v1/direct4n_comparison.md`
- region-split all methods：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/openpcdet_pointrcnn_three_class_region_split_exact4n_all_methods_pilot256_20260808/EXACT4N_REGION_SPLIT_ALL_METHODS_RESULT_ZH.md`
- detector-aware V2/V3/V4/V5：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detector_aware_pdans_v2_20260805/reports/v2_decision_report.md`、`.../detector_aware_pdans_v3_20260805/reports/v3_decision_report.md`、`.../detector_aware_pdans_v4_20260805/reports/v4_decision_report.md`、`.../detector_aware_pdans_v5_20260805/reports/v5_pilot256_decision_report.md`
- V5 holdout64：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/detector_aware_pdans_v5_holdout64_20260805/`
- PointRCNN 三类 holdout 与采样敏感性：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/openpcdet_pointrcnn_three_class_holdout64_20260808/POINT_RCNN_THREE_CLASS_RUNTIME_AND_RESULT_ZH.md`
- 16-seed 采样噪声：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/pointrcnn_sampling_variance_probe_20260810/SIGMA_16SEED_VERDICT_ZH.md`
- PU-Net 因果消融：`/home/ra87racy/projects/baseline_detectors/PointRCNN/results/kitti_patch_punet_causal_ablation_v1_20260731`
- ModelNet40 协议状态：`/home/ra87racy/projects/modelnet40_pointnet2_upsampling/reports/modelnet40_x4_final_protocol_report.md`
- HPC 迁移记录：`/home/ra87racy/projects/kitti_upsampling_lab_to_hpc_migration/reports/`
- 05-04 工作日志：`/home/ra87racy/projects/baseline_detectors/PointRCNN/WORKLOG_2026-05-04.md`
- 磁盘清理：`/home/ra87racy/projects/disk_cleanup_execution_log.md`、`/home/ra87racy/disk_cleanup_stage1_report.md`、`/home/ra87racy/disk_cleanup_20260731_456g_manifest.md`
- 删除文件清单：`/home/ra87racy/deleted_pugcn_intermediate_manifest.txt`、`/home/ra87racy/deleted_pugcn_filetype_counts.txt`

---

#### 20. 当前收尾状态

- 最新 full-val 核心实验：**完成**。
- 两条 PU-GCN validation 输入：**3,769/3,769 + 3,769/3,769 完成**。
- 检测评估：**20/20 PASS**。
- 当前训练/推理进程：**无**。
- 最终结论：**adaptation 显著恢复，但无配置超过配对 baseline**。
- 仍需人工同步的材料：09-08 以前的论文章节和汇报中的 256-frame adaptation 表，应替换为 09-10 full-val 表；这项同步目前没有完成，故本记录不把它写成已更新。

