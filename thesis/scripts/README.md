# 当前论文的编译、图源与核查

主文件为 thesis.tex，交付 PDF 为同目录 thesis.pdf。
沿用 FAU/LMS 模板和 12 TeX pt 正文字号；无 Appendix。
2026-09-17 修订以老师邮件及用户确认的“第一、二章不动”为范围。
具体前后位置见 audit/teacher_email_revision_20260917/CHANGES_CN.md。

## 日常编译

在 thesis/ 目录、已安装 TeX Live 且 latexmk 位于 PATH 的环境运行：

    python scripts/build_current_pdf.py

中间文件在 build/current/，成功后更新根目录 thesis.pdf。
覆盖前请保存批注并关闭阅读器中的旧 PDF。仅编译不需要重新生成实验数据。

## 当前核查

    python scripts/check_teacher_revision_citations.py
    python scripts/check_teacher_revision_layout.py
    python scripts/update_active_figure_manifest.py

引用检查覆盖实际入文源码、BibTeX 编号及 PDF 链接；版面检查从实际入文清单
获取图数，检查页顶放置与最终字号。若旧 PDF 被锁定，可在前两条命令末尾加
--built 检查 build/current/thesis.pdf；这不表示根目录已更新。

## 科研图片

- figure/active_manifest.json：实际进入正文的 38 幅科研图及哈希。
- figure/cv_portrait.png：作者提供的原始照片，无裁剪和修图。
- scripts/make_teacher_revision_figures.py：本次两幅 ModelNet 点云图。
- audit/teacher_email_revision_20260917/figure_manifest.json：点云数组来源哈希、全部点数、
  各方法 CD/HD 回算核验及公共色标范围。
- audit/teacher_email_revision_20260917/figure_qa/：科研图对齐、字号和碰撞报告。

只重绘此次两幅图：

    python scripts/make_teacher_revision_figures.py

沿用 Python/matplotlib 与 .python-deps/ 依赖。对齐检查使用已安装的
nature-figure 技能。每幅图含输入、参考和五种方法；所有输出均使用距离色标，
不生成新实验预测、不凭外观添加分类成功或失败标签。
完整扩展图脚本 expand_results_figures.py --part all 已调用新的两图函数，
不会将这两幅图恢复为旧九面板版本。
其他旧图保留在文件夹内，但是否入文以正文和 active_manifest 为准。

## 数据与文献

evidence/modelnet40_hpc/ 仅采用 HPC 的 ModelNet 结果；
evidence/kitti_lab/ 仅采用 lab 的 KITTI 结果。
bibfiles/references.bib 和 bibfiles/papers/ 保留真实文献及归档 PDF。
实验记录只用于核查，不作为参考文献条目。

P2F 与 CD/HD 同表展示，但 P2F 沿用独立 cloud/mesh 归一化的次级协议。
prepare_evidence.py 已同步生成 P2F 列；日常编译无需运行它。
KITTI 后续训练是检测器适配，PU-GCN 本身仍固定。
历史子集、最终全验证集和匹配事件不混作同一种实验指标。

## 一次性历史脚本

restore_*、clean_recovered_prose.py、apply_record_revision.py、
revise_citation_positions.py 以及 audit 下的 revise_*.py 是历次修改记录。
不要在当前稿件上再次运行这些一次性修改脚本。
此前 review PDF 由整本 PDF 抽取；内容以当前 thesis.pdf 为准。
