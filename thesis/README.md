# 完整论文源码

源自 `D:\Thesis\thesis` 的当前正式工程，包含全部章节、模板包、BibTeX 数据库和样式、任务书、现有图片、可编辑图源、表格、图表生成脚本及其本地 evidence。

- `thesis.tex`：唯一主入口；`texfiles/` 包括第 1–6 章及前后置部分。
- `figure/`：正文图片，以及 PNG/SVG/PDF 编辑与展示版本。
- `bibfiles/references.bib`、`bibfiles/IEEEtran_thesis.bst`：文献库与本地样式。
- `topic/description.pdf`：签字任务书。
- `scripts/`、`evidence/`：论文图表生成与来源记录；历史脚本的外部 skill、字体或原工作区路径不等于编译现有论文所必需的依赖。
- `../thesis.pdf`：本次原样保留的本机当前正式 PDF。

在安装 TeX Live（原工程使用 2025）或兼容发行版的机器上，于本目录执行：

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/current thesis.tex
```

生成文件在 `build/current/thesis.pdf`。直接编译现有插图不需要 Python 绘图库、原研究数据集或 Codex skill。`python scripts/build_current_pdf.py` 也保留供沿用原工作流，会生成本副本下的 `audit/` 和 `thesis.pdf`；原样归档 PDF 保存在上一级。

未复制本地 Python 依赖安装目录、编译缓存、历史 audit 备份、审阅裁剪 PDF、过时 figures 目录及 `bibfiles/papers` 下 30 篇外部论文 PDF。这些不属于当前论文的 88 个本地编译输入；文献 .bib 和 .bst 已保留。原资料仍在原工作目录。
