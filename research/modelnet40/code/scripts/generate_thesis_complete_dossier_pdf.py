#!/usr/bin/env python3
"""Generate a complete Chinese PDF dossier for ModelNet40 thesis writing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentations" / "ModelNet40_Thesis_Complete_Dossier.pdf"
FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"

pdfmetrics.registerFont(TTFont("CN", FONT))

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm


def styles():
    base = getSampleStyleSheet()
    s = {
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="CN",
            fontSize=20,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            fontName="CN",
            fontSize=11,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="CN",
            fontSize=14,
            leading=20,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#111111"),
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="CN",
            fontSize=12,
            leading=17,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#1a1a1a"),
        ),
        "h3": ParagraphStyle(
            "h3",
            fontName="CN",
            fontSize=10.5,
            leading=15,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.HexColor("#222222"),
        ),
        "body": ParagraphStyle(
            "body",
            fontName="CN",
            fontSize=9.5,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="CN",
            fontSize=9.5,
            leading=13.5,
            leftIndent=8,
            spaceAfter=2,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="CN",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#444444"),
            spaceAfter=3,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            fontName="CN",
            fontSize=7.5,
            leading=10,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            fontName="CN",
            fontSize=7.5,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="CN",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#555555"),
            spaceBefore=2,
            spaceAfter=8,
            alignment=TA_CENTER,
        ),
        "toc": ParagraphStyle(
            "toc",
            fontName="CN",
            fontSize=10,
            leading=16,
            spaceAfter=3,
        ),
    }
    return s


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def bullets(items, style):
    flow = []
    for it in items:
        flow.append(P(f"• {it}", style))
    return flow


def make_table(headers, rows, col_widths=None, s=None):
    cell = s["table_cell"]
    head = s["table_header"]
    data = [[P(h, head) for h in headers]]
    for row in rows:
        data.append([P(str(c), cell) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "CN"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("CN", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(MARGIN, 1.0 * cm, "ModelNet40 Upsampling → PointNet++ 完整实验档案")
    canvas.drawRightString(PAGE_W - MARGIN, 1.0 * cm, f"{doc.page}")
    canvas.restoreState()


def build():
    s = styles()
    story = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    usable = PAGE_W - 2 * MARGIN

    # -------- Cover --------
    story.append(Spacer(1, 3.2 * cm))
    story.append(P("ModelNet40 点云上采样 + PointNet++ 分类", s["cover_title"]))
    story.append(P("完整实验档案 / Thesis Writing Dossier", s["cover_title"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(P("实验设置 · 结果 · 协议演变 · 方法与论文 · 工作记录 · 文件索引", s["cover_sub"]))
    story.append(Spacer(1, 1.2 * cm))
    story.append(
        P(
            f"项目路径：<br/>{ROOT}<br/><br/>"
            f"生成时间：{now}<br/>"
            "数据来源：pointnet2_results / geometry equal-N / Full Report notes<br/>"
            "用途：整理思路、撰写论文 / 答辩材料",
            s["cover_sub"],
        )
    )
    story.append(Spacer(1, 2.0 * cm))
    story.append(
        P(
            "核心结论：点云上采样不普遍提升下游分类；几何质量与分类精度不对齐；"
            "Line A 全部低于 Original 1024；Line B 仅 PU-Net 略超下采样基线。",
            s["body"],
        )
    )
    story.append(PageBreak())

    # -------- TOC --------
    story.append(P("目录", s["h1"]))
    toc = [
        "1. 研究背景与问题陈述",
        "2. 核心原理与评价逻辑",
        "3. 最终实验协议（Canonical）",
        "4. 方法、论文与技术原理",
        "5. 分类实验结果（完整数值）",
        "6. 几何质量结果（Equal-N）",
        "7. 几何 vs 分类讨论",
        "8. Mesh-ref 对照实验",
        "9. 协议演变与历史修改记录",
        "10. 工程实施与工作记录",
        "11. Ablation / 非主结果清单",
        "12. 论文写作建议与可用结论句",
        "13. 关键文件与材料索引",
        "附录 A. 训练超参与数据路径",
        "附录 B. 方法代码与 checkpoint 来源",
    ]
    for t in toc:
        story.append(P(t, s["toc"]))
    story.append(PageBreak())

    # -------- 1 --------
    story.append(P("1. 研究背景与问题陈述", s["h1"]))
    story.append(
        P(
            "点云上采样（point cloud upsampling）旨在从稀疏点云生成更密集、几何更完整的点集。"
            "大量工作以 Chamfer Distance（CD）、Hausdorff Distance（HD）、点到面距离（P2F）等几何指标评价上采样质量，"
            "并默认“更密、几何更好”有利于下游三维任务。本课题在 ModelNet40 官方分类设定下，"
            "系统检验该假设是否成立：当上采样输出送入 PointNet++ 分类器时，分类 Overall Accuracy（OA）是否提升。",
            s["body"],
        )
    )
    story.append(P("研究问题（Research Questions）", s["h2"]))
    story.extend(
        bullets(
            [
                "RQ1：在已较充分的 Original 1024 上再 ×4 加密到 4096（Densification），能否提升 PointNet++ 分类？",
                "RQ2：在 ×4 下采样到 256 后再恢复到 1024（Recovery），上采样能否挽回分类损失？",
                "RQ3：几何指标最优的方法，是否也是下游分类最优的方法？",
                "RQ4：点数增加本身（mesh 直接采 4096）是否足以解释分类变化？",
            ],
            s["bullet"],
        )
    )
    story.append(P("实验定位", s["h2"]))
    story.extend(
        bullets(
            [
                "任务：ModelNet40 形状分类（非 KITTI 检测、非 AP）。",
                "下游模型：PointNet++ SSG（pointnet2_cls_ssg），输入仅 XYZ。",
                "上采样器：使用已有预训练权重做推理，本实验不重新训练上采样网络。",
                "分类器：每个点数设置从零训练（from scratch），不是在 1024 预训练权重上 finetune。",
                "主倍率：R=×4。TULIP（range-image / LiDAR 风格）仅 supplementary，不进入主 xyz 对比表。",
            ],
            s["bullet"],
        )
    )
    story.append(
        P(
            "一句话贡献：在统一 ×4 two-line 协议下比较五种上采样方法，证明几何恢复与分类效用必须分开报告；"
            "稠密化普遍无益，稀疏恢复仅 PU-Net 有边际收益且仍低于 Original。",
            s["body"],
        )
    )

    # -------- 2 --------
    story.append(P("2. 核心原理与评价逻辑", s["h1"]))
    story.append(P("2.1 Two-line 实验设计原理", s["h2"]))
    story.append(
        P(
            "设计两条独立实验线，避免把“加密好数据”与“恢复坏数据”混为一谈。"
            "跨线比较（例如直接拿 Line A 上采样 vs Line B 上采样）不作为主结论。",
            s["body"],
        )
    )
    story.append(
        make_table(
            ["实验线", "科学含义", "输入路径", "PointNet++ 输入点数"],
            [
                ["Line A Densification", "好数据再加密", "Original 1024 → Up ×4", "baseline 1024 / up 4096"],
                ["Line B Recovery", "稀疏后再恢复", "Original→Down×4(256)→Up×4", "baseline 256 / up 1024"],
            ],
            col_widths=[3.2 * cm, 3.5 * cm, 5.0 * cm, 4.5 * cm],
            s=s,
        )
    )
    story.append(P("表 1. 双线协议概览", s["caption"]))

    story.append(P("2.2 点数协议原则（不可违反）", s["h2"]))
    story.extend(
        bullets(
            [
                "Baseline 保留原生点数：不上采样对齐，不 pad/resample 到上采样输出点数。",
                "Upsampling 保留输出点数：不裁回 baseline 原生点数。",
                "禁止 baseline↔upsampling 人为点数匹配（否则混淆“上采样效应”与“点数效应”）。",
                "同线同分支：同一上采样倍率、同一输出点数。",
                "Original line 与 Downsampled line 独立，不共享点数一致性要求。",
            ],
            s["bullet"],
        )
    )

    story.append(P("2.3 分类指标原理", s["h2"]))
    story.extend(
        bullets(
            [
                "数据集：ModelNet40 official split，9843 train / 2468 test；分类任务无独立 validation set。",
                "Best Overall Accuracy：训练过程中记录到的最高 overall（instance）accuracy；本实现在 test 提升时保存 best checkpoint。",
                "Final Overall：第 200 epoch 的 overall accuracy，作为补充。",
                "主文以 Best OA 为主；差值用 percentage points（pp）。",
                "Line B 主 Δ：相对 Downsampled ×4 baseline；次级 gap：相对 Original baseline。",
            ],
            s["bullet"],
        )
    )

    story.append(P("2.4 几何指标原理（两套，勿混用）", s["h2"]))
    story.append(
        make_table(
            ["版本", "参考对象", "用途", "写论文注意"],
            [
                [
                    "Equal-N（现行推荐）",
                    "A: mesh-ref 4096；B down: mesh-ref 256；B up: Original 1024",
                    "主几何表",
                    "等基数 CD/HD 才有可比意义",
                ],
                [
                    "vs Original（不等基数）",
                    "4096 vs 1024 等",
                    "历史/辅助",
                    "表示离散集一致性，非连续曲面精度",
                ],
                [
                    "旧 mesh / exact P2F",
                    "dense mesh surface",
                    "早期计算过",
                    "2026-08 PPT 修订已删除主叙述",
                ],
            ],
            col_widths=[3.2 * cm, 5.2 * cm, 2.8 * cm, 5.0 * cm],
            s=s,
        )
    )
    story.append(P("表 2. 几何评价协议", s["caption"]))
    story.extend(
        bullets(
            [
                "CD（Chamfer）：双向最近邻距离均值之和，衡量整体匹配。",
                "HD（Hausdorff）：最坏最近邻匹配，对离群点敏感。",
                "NUC：点分布均匀性；越低通常越均匀（协议内相对比较）。",
                "Mesh-ref 采样：对 .off 做三角形面积加权采样 + unit-sphere 归一化；seed=stable_seed(42,...)。",
                "Mesh-ref 不是从 Original 1024 下采样得到，而是从同一 mesh 家族独立采样。",
            ],
            s["bullet"],
        )
    )

    # -------- 3 --------
    story.append(P("3. 最终实验协议（Canonical）", s["h1"]))
    story.append(
        P(
            "以下为 2026-07 起定稿、并经 2026-08 equal-N / mesh-ref 基线补充后的最终主协议。"
            "历史 ×2 / 512 点线仅作 ablation，见第 11 节。",
            s["body"],
        )
    )
    story.append(
        make_table(
            ["线", "分支", "磁盘点数", "num_point", "allow_resample", "角色"],
            [
                ["A", "Original baseline", "1024", "1024", "false", "主基线"],
                ["A", "Original + EAR/PDANS/PU-Net/PU-GCN/EdgeFormer", "4096", "4096", "false", "主上采样"],
                ["B", "Downsampled ×4 baseline", "256", "256", "false", "主基线"],
                ["B", "Down ×4 + 同上五种方法", "1024", "1024", "false", "主上采样"],
                ["—", "Mesh-ref 256", "256", "256", "false", "采样过程对照"],
                ["—", "Mesh-ref 4096", "4096", "4096", "false", "稠密 mesh 采样对照"],
            ],
            col_widths=[1.2 * cm, 6.2 * cm, 2.0 * cm, 2.0 * cm, 2.4 * cm, 2.4 * cm],
            s=s,
        )
    )
    story.append(P("表 3. Canonical 分支与点数", s["caption"]))

    story.append(P("训练统一设置", s["h2"]))
    story.extend(
        bullets(
            [
                "模型：pointnet2_cls_ssg；类别数 40；输入 XYZ only。",
                "epoch=200；optimizer=Adam；lr=0.001；decay_rate=0.0001；batch_size=24；seed=42；num_workers=4。",
                "配置目录：configs/pointnet2_x4_two_line/*.yaml",
                "结果目录：pointnet2_results/x4_two_line_final/**/metrics.json",
            ],
            s["bullet"],
        )
    )

    # -------- 4 --------
    story.append(PageBreak())
    story.append(P("4. 方法、论文与技术原理", s["h1"]))
    story.append(
        make_table(
            ["方法", "类型", "核心原理", "论文 / 出处"],
            [
                [
                    "EAR",
                    "经典几何\n非学习",
                    "先远离尖锐特征采样以获得可靠法向，再渐进向边附近上采样；保尖边、可调密度",
                    "Huang et al., ACM TOG 2013\nEdge-aware point set resampling",
                ],
                [
                    "PU-Net",
                    "学习\nTF1",
                    "多层点特征提取 + feature-space 多分支扩展，开创学习式点云上采样",
                    "Yu et al., CVPR 2018\narXiv:1801.06761",
                ],
                [
                    "PU-GCN",
                    "学习\nTF1",
                    "用图卷积建模局部几何关系进行上采样；常在 CD 上表现强",
                    "Qian et al., CVPR 2021\narXiv:1912.03264",
                ],
                [
                    "PDANS",
                    "学习\nPyTorch",
                    "条件扩散上采样 + Adaptive Noise Suppression / Tree-Trans；强调噪声鲁棒与细节",
                    "CVPR 2025\ngithub.com/Baty2023/PDANS",
                ],
                [
                    "PU-EdgeFormer",
                    "学习\nTF",
                    "EdgeFormer：EdgeConv（局部）+ multi-head self-attention（全局）",
                    "Kim et al., ICASSP/arXiv 2023\narXiv:2305.01148",
                ],
                [
                    "PointNet++",
                    "下游分类",
                    "层次化采样与局部 PointNet 聚合；本实验采用 SSG",
                    "Qi et al., NeurIPS 2017\nyanx27 PyTorch 实现",
                ],
                [
                    "ModelNet40",
                    "数据集",
                    "CAD mesh 采样点云；官方 train/test split",
                    "Wu et al., CVPR 2015",
                ],
                [
                    "TULIP",
                    "补充 only",
                    "LiDAR range-image 上采样；与 xyz 点数协议不兼容主对比",
                    "ETHZ ASL TULIP\n不进主表",
                ],
            ],
            col_widths=[2.4 * cm, 2.0 * cm, 6.2 * cm, 5.6 * cm],
            s=s,
        )
    )
    story.append(P("表 4. 方法与文献", s["caption"]))

    story.append(P("实现与推理备注", s["h2"]))
    story.extend(
        bullets(
            [
                "EAR：确定性脚本，无 checkpoint；从 lab 迁移到 HPC。",
                "PU-Net / PU-GCN / PU-EdgeFormer：TensorFlow 系 + 自定义 CUDA/tf_ops；需匹配编译环境。",
                "PDANS：PyTorch + PyTorch3D / Chamfer / pointops；大体积 .pkl checkpoint（PU1K/PUGAN）。",
                "PU-EdgeFormer：2026-07-16 接入并完成真实性审计（模型路径、checkpoint、与 PU-GCN 输出差异、无 leakage）。",
                "所有主方法在两条线上均生成完整 12311 样本（9843+2468），再经点数/NaN 审计后送入 PointNet++。",
            ],
            s["bullet"],
        )
    )

    # -------- 5 --------
    story.append(P("5. 分类实验结果（完整数值）", s["h1"]))
    story.append(P("5.1 Line A — Densification（相对 Original 1024）", s["h2"]))
    story.append(
        make_table(
            ["方法", "点数", "Best OA", "Final OA", "Best Class", "Δ Best vs Orig (pp)"],
            [
                ["Original baseline", "1024", "91.95%", "91.31%", "88.00%", "0.00"],
                ["PU-GCN", "4096", "91.63%", "91.00%", "87.74%", "-0.32"],
                ["PDANS", "4096", "91.62%", "90.87%", "88.44%", "-0.33"],
                ["EAR", "4096", "91.48%", "90.85%", "88.19%", "-0.47"],
                ["PU-Net", "4096", "90.95%", "90.55%", "87.46%", "-1.00"],
                ["PU-EdgeFormer", "4096", "90.54%", "90.13%", "86.35%", "-1.41"],
            ],
            col_widths=[3.2 * cm, 1.6 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 4.6 * cm],
            s=s,
        )
    )
    story.append(P("表 5. Line A 分类完整结果（Best 为主指标）", s["caption"]))
    story.append(
        P(
            "解读：没有任何 4096 上采样分支超过 Original 1024。"
            "最接近的是 PU-GCN（−0.32 pp）与 PDANS（−0.33 pp）；"
            "PU-EdgeFormer 下降最大（−1.41 pp）。说明在本设定下，对已充分采样的 CAD 点云再加密，"
            "并不能稳定转化为 PointNet++ 分类收益。",
            s["body"],
        )
    )

    story.append(P("5.2 Line B — Recovery（相对 Down 256；gap 相对 Original）", s["h2"]))
    story.append(
        make_table(
            ["方法", "点数", "Best OA", "Final OA", "Δ vs Down (pp)", "gap vs Orig (pp)"],
            [
                ["Downsampled ×4", "256", "90.85%", "90.46%", "0.00", "-1.10"],
                ["PU-Net", "1024", "91.27%", "90.65%", "+0.42", "-0.68"],
                ["PDANS", "1024", "90.34%", "89.46%", "-0.51", "-1.61"],
                ["PU-GCN", "1024", "90.06%", "89.63%", "-0.79", "-1.89"],
                ["PU-EdgeFormer", "1024", "89.32%", "88.58%", "-1.53", "-2.63"],
                ["EAR", "1024", "88.81%", "88.02%", "-2.04", "-3.14"],
            ],
            col_widths=[3.2 * cm, 1.6 * cm, 2.2 * cm, 2.2 * cm, 3.4 * cm, 3.6 * cm],
            s=s,
        )
    )
    story.append(P("表 6. Line B 分类完整结果", s["caption"]))
    story.append(
        P(
            "解读：恢复点数并不自动恢复分类性能。仅 PU-Net 相对稀疏基线有 +0.42 pp 的边际提升，"
            "但仍比 Original 低 0.68 pp。EAR 作为经典几何方法在分类上最差（−2.04 pp）。"
            "这说明“把点数补回 1024”不等于“把语义可分性补回”。",
            s["body"],
        )
    )

    # -------- 6 --------
    story.append(P("6. 几何质量结果（Equal-N，test N=2468）", s["h1"]))
    story.append(P("6.1 Line A：4096 vs Mesh-ref 4096（越低越好）", s["h2"]))
    story.append(
        make_table(
            ["方法", "CD", "HD", "NUC", "ΔNUC"],
            [
                ["Mesh-ref 4096 (self)", "0.000000", "0.000000", "0.641852", "0.000"],
                ["PU-GCN", "0.042306", "0.093219", "0.520398", "-0.121"],
                ["PU-EdgeFormer", "0.046666", "0.110932", "0.675435", "+0.034"],
                ["PDANS", "0.046821", "0.113510", "0.661176", "+0.019"],
                ["PU-Net", "0.049719", "0.111177", "0.605211", "-0.037"],
                ["EAR", "0.054722", "0.115396", "0.815631", "+0.174"],
            ],
            col_widths=[4.0 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm],
            s=s,
        )
    )
    story.append(P("表 7. Line A equal-N 几何", s["caption"]))

    story.append(P("6.2 Line B：Down 256 vs Mesh-ref 256；Ups 1024 vs Original 1024", s["h2"]))
    story.append(
        make_table(
            ["方法", "点数", "参考", "CD", "HD", "NUC"],
            [
                ["Mesh-ref 256 (self)", "256", "Mesh-ref 256", "0.000", "0.000", "2.126"],
                ["Downsampled ×4", "256", "Mesh-ref 256", "0.133", "0.205", "2.093"],
                ["Original (self)", "1024", "Original 1024", "0.000", "0.000", "1.084"],
                ["PU-GCN", "1024", "Original 1024", "0.0558", "0.1514", "1.524"],
                ["PU-EdgeFormer", "1024", "Original 1024", "0.0619", "0.1776", "1.123"],
                ["PDANS", "1024", "Original 1024", "0.0627", "0.1905", "1.053"],
                ["PU-Net", "1024", "Original 1024", "0.0646", "0.1083", "1.080"],
                ["EAR", "1024", "Original 1024", "0.0767", "0.1932", "0.965"],
            ],
            col_widths=[3.4 * cm, 1.6 * cm, 3.4 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm],
            s=s,
        )
    )
    story.append(P("表 8. Line B equal-N 几何", s["caption"]))
    story.append(
        P(
            "关键观察：PU-GCN 在两条线的 CD 往往最好，但 Line B 的 NUC 明显变差（1.524）；"
            "PU-Net 的 HD 在 Line B 最好（0.108），NUC 接近 Original，与其唯一的分类增益相一致。"
            "EAR 的 NUC 可较低，但 CD/HD 与分类都偏弱。",
            s["body"],
        )
    )

    # -------- 7 --------
    story.append(PageBreak())
    story.append(P("7. 几何 vs 分类讨论", s["h1"]))
    story.append(
        make_table(
            ["现象", "证据", "论文含义"],
            [
                [
                    "几何好 ≠ 分类好",
                    "Line B：PU-GCN CD 最小，但 Best OA 低于 PU-Net",
                    "必须分表报告几何与任务指标",
                ],
                [
                    "加密无益",
                    "Line A：全部 4096 方法 < Original 1024",
                    "Densification 假设不成立（本设定）",
                ],
                [
                    "恢复有限",
                    "仅 PU-Net +0.42 pp，且仍低于 Original",
                    "Upsampling 不能完全抵消 ×4 信息损失",
                ],
                [
                    "分布重要",
                    "PU-Net HD/NUC 更均衡；PU-GCN NUC 恶化",
                    "PointNet++ 对局部结构/分布敏感",
                ],
                [
                    "点数本身弱解释",
                    "Mesh-ref 4096 仅 +0.05 pp vs Original",
                    "“更密”不是主因果",
                ],
            ],
            col_widths=[3.2 * cm, 6.5 * cm, 6.5 * cm],
            s=s,
        )
    )
    story.append(P("表 9. 讨论要点", s["caption"]))

    # -------- 8 --------
    story.append(P("8. Mesh-ref 对照实验", s["h1"]))
    story.append(
        P(
            "为回答“分类变化是否只是因为点数不同 / 采样过程不同”，额外从 .off 独立面积加权采样 "
            "mesh-ref 256 与 mesh-ref 4096，并用同一套 PointNet++ 超参从零训练。",
            s["body"],
        )
    )
    story.append(
        make_table(
            ["方法", "点数", "Best OA", "Final OA", "best epoch", "对照", "对照 Best", "Δ pp"],
            [
                ["Mesh-ref 256", "256", "90.88%", "90.38%", "129", "Down ×4 256", "90.85%", "+0.03"],
                ["Mesh-ref 4096", "4096", "92.00%", "91.18%", "148", "Original 1024", "91.95%", "+0.05"],
            ],
            col_widths=[2.6 * cm, 1.4 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.8 * cm, 2.0 * cm, 1.6 * cm],
            s=s,
        )
    )
    story.append(P("表 10. Mesh-ref PointNet++ 基线", s["caption"]))
    story.extend(
        bullets(
            [
                "同点数下，mesh 直接采样 vs 从 Original 下采样，分类几乎相同（+0.03 pp）。",
                "更密的 mesh-ref 4096 相对 Original 1024 仅 +0.05 pp，无实质优势。",
                "因此 Line A 上采样分支全面落后，更应归因于上采样几何/分布特性，而非“4096 这个数字”。",
            ],
            s["bullet"],
        )
    )

    # -------- 9 --------
    story.append(P("9. 协议演变与历史修改记录", s["h1"]))
    story.append(
        make_table(
            ["阶段", "变更内容", "现在状态"],
            [
                ["前期", "KITTI + CenterPoint/PointRCNN 检测线；lab→HPC 迁移方法与数据", "并行工程背景，非本主结果"],
                ["2026-06-15", "建立 modelnet40_experiments 结构与 method_registry", "资产索引保留"],
                ["2026-06-20+", "搭建 modelnet40_pointnet2_upsampling 主 pipeline", "现行项目根"],
                ["2026-06-25", "主倍率 ×2→×4；旧 512→1024 EAR 降为 ablation", "已生效"],
                ["随后定稿", "Line B 改为 256→1024（对齐 N 的恢复），不再用 512→2048 作主线", "Canonical"],
                ["实施期", "修复 CUDA/TF ops、PDANS 环境、PU-Net 超时 resume、PU-GCN stall 等", "见 reports/"],
                ["2026-07-16", "接入 PU-EdgeFormer；真实性审计 PASS 后入主表", "第五方法"],
                ["2026-07-12~17", "thesis 表/图/解读定稿；geometry vs classification 结论明确", "结果章骨架"],
                ["2026-08-03", "equal-N mesh-ref 几何重算；mesh-ref 分类基线训练完成", "几何主协议"],
                ["2026-08-04", "PPT 去除 mesh/P2F 主叙述；Full Report；强调 from-scratch", "对外汇报版本"],
            ],
            col_widths=[2.6 * cm, 8.5 * cm, 5.1 * cm],
            s=s,
        )
    )
    story.append(P("表 11. 协议与工程时间线", s["caption"]))

    story.append(P("关键“纠错”说明（写论文 Limitations / Appendix 很有用）", s["h2"]))
    story.extend(
        bullets(
            [
                "早期曾短暂以 R=×2 为推荐，后因与 PU 系方法默认 ×4、以及实验目标一致性而改为 ×4。",
                "早期 Line B 使用 downsampled50（约 512 点）；后改为 downsampled_x4（256 点）以形成严格 ×4 恢复闭环。",
                "曾出现把 baseline loader 重采样到 1024 的做法（Step 6），现明确为 naive ablation，禁止当主基线。",
                "几何叙事从“dense mesh + P2F”调整为 equal-N；避免不等基数 CD/HD 误导排序。",
                "Best OA 的含义在报告中明确：无 val set，是 checkpoint 历史最高 overall。",
            ],
            s["bullet"],
        )
    )

    # -------- 10 --------
    story.append(PageBreak())
    story.append(P("10. 工程实施与工作记录（摘要）", s["h1"]))
    story.append(P("10.1 数据与生成", s["h2"]))
    story.extend(
        bullets(
            [
                "Original 1024 点云由 ModelNet40 mesh 采样并 unit-sphere 归一化。",
                "Downsampled ×4：datasets/modelnet40_downsampled_x4/（256 点，12311 shapes）。",
                "各方法 Line A/B 上采样输出经 strict 点数审计（期望 4096 或 1024），零 NaN/Inf。",
                "PointNet++ 输入通过 symlink/准备脚本写入 pointnet2_inputs/。",
            ],
            s["bullet"],
        )
    )
    story.append(P("10.2 典型故障与修复（可写工程附录）", s["h2"]))
    story.extend(
        bullets(
            [
                "PDANS：早期 job 因 CUDA/torch.cuda.is_available=False 失败；对齐 module/LD_LIBRARY_PATH 后修复。",
                "PU-Net / PU-GCN：TF ops / CUDA toolchain 编译与运行环境敏感。",
                "PU-Net Line B full：多 chunk 24h TIMEOUT；resume 缺失 3161 样本后最终 12311/12311 PASS。",
                "PU-GCN：出现过 RTX 3080 stall / SyntaxError，经 resume 与修复完成。",
                "PU-EdgeFormer：单独做 repeat-copy、call-path、checkpoint、与 PU-GCN 差异、leakage、freshness 审计。",
            ],
            s["bullet"],
        )
    )
    story.append(P("10.3 可视化与汇报材料", s["h2"]))
    story.extend(
        bullets(
            [
                "静态图：figures/modelnet40/pointnet2_final*_with_pu_edgeformer/、geometry_delta_*、final_combined/",
                "交互 HTML：figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/（10 个 dropdown）",
                "PPT：ModelNet40_PointNet2_Full_Report.pptx（28 页）与 Original_Reference_Revised.pptx（21 页）",
                "相机统一：同类方法同 elev/azim/轴范围/点大小规则，保证定性公平。",
            ],
            s["bullet"],
        )
    )

    # -------- 11 --------
    story.append(P("11. Ablation / 非主结果清单", s["h1"]))
    story.append(
        make_table(
            ["项目", "内容", "论文用法"],
            [
                ["×2 EAR Line B", "512→1024（旧 Step 7b/8）", "倍率 ablation；非主 ×4"],
                ["naive resample baseline", "磁盘 512，loader 采到 1024（Step 6）", "证明点数对齐会污染对比"],
                ["旧 downsampled50 主线", "512 点协议", "被 ×4/256 取代"],
                ["TULIP", "range-image 上采样", "supplementary only"],
                ["不等基数 CD/HD 排名", "4096 vs 1024 等", "不作为主几何排序"],
                ["旧 P2F 主叙事", "dense mesh exact P2F", "可附录，勿作现行主表"],
                ["KITTI/CenterPoint", "检测 AP 线", "另章；本档案不展开主表"],
            ],
            col_widths=[3.6 * cm, 6.0 * cm, 6.6 * cm],
            s=s,
        )
    )
    story.append(P("表 12. 非主结果与建议写法", s["caption"]))

    # -------- 12 --------
    story.append(P("12. 论文写作建议与可用结论句", s["h1"]))
    story.append(P("建议章节结构", s["h2"]))
    story.extend(
        bullets(
            [
                "Introduction：动机——上采样文献重几何，缺系统下游分类验证；提出 two-line ×4 问题。",
                "Related Work：EAR → PU-Net → PU-GCN → EdgeFormer/扩散（PDANS）；PointNet++；ModelNet40。",
                "Method：协议原则、双线流程、from-scratch 分类、equal-N 几何。",
                "Experimental Setup：数据、超参、指标定义、实现细节。",
                "Results：表 5–8 + mesh-ref 表 10；绝对 OA 与 Δ pp 同报。",
                "Discussion：几何–任务不对齐；Densification vs Recovery；分布/HD/NUC 角色。",
                "Ablation/Appendix：×2、naive resample、mesh-ref、工程审计。",
                "Conclusion：四条主结论（见下）。",
            ],
            s["bullet"],
        )
    )
    story.append(P("可直接改写进论文的结论句", s["h2"]))
    story.extend(
        bullets(
            [
                "点云上采样并不普遍提升 PointNet++ 在 ModelNet40 上的分类精度。",
                "在 Densification 设定（1024→4096）下，五种方法的 Best OA 均低于原生 Original 1024（91.95%）。",
                "在 Recovery 设定（256→1024）下，仅 PU-Net 相对稀疏基线提升 +0.42 pp，但仍低于 Original 0.68 pp。",
                "几何指标与分类效用不对齐：例如 PU-GCN 常具更优 CD，但分类不及 PU-Net。",
                "Mesh-ref 4096 相对 Original 仅 +0.05 pp，表明单纯增加点数不足以解释或保证分类增益。",
                "因此，上采样方法应同时报告几何质量与下游任务效用，并明确实验线（加密 vs 恢复）。",
            ],
            s["bullet"],
        )
    )

    # -------- 13 --------
    story.append(PageBreak())
    story.append(P("13. 关键文件与材料索引", s["h1"]))
    story.append(
        make_table(
            ["用途", "路径（相对项目根）"],
            [
                ["本 PDF", "presentations/ModelNet40_Thesis_Complete_Dossier.pdf"],
                ["完整 PPT/PDF", "presentations/ModelNet40_PointNet2_Full_Report.*"],
                ["修订版 PPT", "presentations/ModelNet40_PointNet2_Original_Reference_Revised.*"],
                ["分类总表", "reports/modelnet40_pointnet2_final_two_line_classification_summary_with_pu_edgeformer.*"],
                ["几何+分类总表", "reports/modelnet40_final_thesis_summary_table_with_pu_edgeformer.*"],
                ["结果解读", "reports/modelnet40_thesis_result_analysis.md"],
                ["Equal-N 几何", "reports/modelnet40_geometry_equal_n_*.{csv,md,json}"],
                ["Mesh-ref 分类", "reports/modelnet40_pointnet2_mesh_ref_baseline_results.*"],
                ["×4 协议报告", "reports/modelnet40_x4_final_protocol_report.md"],
                ["协议纠错日志", "reports/modelnet40_pointnet2_final_protocol_corrected.md"],
                ["EdgeFormer 审计", "reports/pu_edgeformer_modelnet40_authenticity_verification_20260716.md"],
                ["图注", "reports/modelnet40_thesis_figure_captions.md"],
                ["Figure index", "figures/modelnet40/figure_index.md"],
                ["方法注册表", "../modelnet40_experiments/configs/method_registry.yaml"],
                ["训练配置", "configs/pointnet2_x4_two_line/*.yaml"],
                ["指标 JSON", "pointnet2_results/x4_two_line_final/**/metrics.json"],
            ],
            col_widths=[3.8 * cm, 12.4 * cm],
            s=s,
        )
    )
    story.append(P("表 13. 文件索引", s["caption"]))

    # -------- Appendix A --------
    story.append(P("附录 A. 训练超参与数据路径", s["h1"]))
    story.append(
        make_table(
            ["项", "值"],
            [
                ["数据 split", "9843 train / 2468 test（官方）"],
                ["模型", "pointnet2_cls_ssg"],
                ["epoch / batch / lr / seed", "200 / 24 / 0.001 / 42"],
                ["optimizer / decay", "Adam / 0.0001"],
                ["特征", "XYZ only"],
                ["Line A data", "pointnet2_inputs/lineA_*"],
                ["Line B data", "pointnet2_inputs/lineB_*"],
                ["Downsampled root", "datasets/modelnet40_downsampled_x4/"],
                ["Mesh-ref roots", "datasets/modelnet40_mesh_ref_{256,4096}/"],
                ["结果 root", "pointnet2_results/x4_two_line_final/"],
            ],
            col_widths=[5.0 * cm, 11.2 * cm],
            s=s,
        )
    )
    story.append(P("表 A1. 超参与路径", s["caption"]))

    # -------- Appendix B --------
    story.append(P("附录 B. 方法代码与 checkpoint 来源", s["h1"]))
    story.append(
        make_table(
            ["方法", "代码来源", "权重/备注"],
            [
                ["EAR", "code/upsampling_methods/EAR（lab 迁移）", "无 checkpoint（非学习）"],
                ["PU-Net", "yulequan/PU-Net（迁移）", "TF1 model-120 等"],
                ["PU-GCN", "guochengqian/PU-GCN（迁移）", "TF1 model-120 等"],
                ["PDANS", "Baty2023/PDANS（迁移）", "PU1K_PDANS.pkl / PUGAN_PDANS.pkl"],
                ["PU-EdgeFormer", "external/pu_edgeformer_ops_reuse", "checkpoint_model100；2026-07-16 审计"],
                ["TULIP", "ethz-asl/TULIP（迁移）", "tulip_kitti.pth；supplementary"],
                ["PointNet++", "external/Pointnet_Pointnet2_pytorch", "本实验全部 from-scratch 训练"],
            ],
            col_widths=[3.0 * cm, 6.5 * cm, 6.7 * cm],
            s=s,
        )
    )
    story.append(P("表 B1. 代码与权重", s["caption"]))

    story.append(Spacer(1, 0.6 * cm))
    story.append(
        P(
            "文档结束。数值以 metrics.json 与 equal-N CSV 为准；若与旧 markdown 冲突，以 2026-08 Full Report / equal-N / "
            "classification_summary_with_pu_edgeformer 为准。",
            s["small"],
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=1.5 * cm,
        bottomMargin=1.6 * cm,
        title="ModelNet40 Thesis Complete Dossier",
        author="thesis_pointcloud",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")
    print(f"Size bytes: {OUT.stat().st_size}")


if __name__ == "__main__":
    build()
