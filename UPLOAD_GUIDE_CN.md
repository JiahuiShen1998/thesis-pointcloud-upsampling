# 完整论文项目：下载、验证与后续更新

本地归档目录：`D:\Thesis\Git\thesis_handover_prepared_20260927`。

## 实际仓库与分支

| 平台 | 项目 | 本次归档分支 |
|---|---|---|
| 学校 GitLab | [PreprocessingPC](https://gitlab.lms.tf.fau.de/marina.ritthaler/preprocessingpc/-/tree/codex/thesis-archive) | `codex/thesis-archive` |
| 个人 GitHub | [thesis-pointcloud-upsampling](https://github.com/JiahuiShen1998/thesis-pointcloud-upsampling) | `main` |

学校账号个人项目配额为 0；作者已明确选择现有 PreprocessingPC 私有项目。本次使用独立归档分支，保留学校项目原来的 `main` 和 `share/modelnet40-pointnet2-presentation`。打开学校项目时请选择上述归档分支；默认 `main` 不是本次完整论文归档。

当前提交与下载验证状态见 [UPLOAD_STATUS_CN.md](UPLOAD_STATUS_CN.md)。目录已经初始化并配置远程，无需再次建库或初始化。完整交付仍待补最终答辩稿。

## 下载完整归档

安装 Git、Git LFS 和 Python 3，完成学校账号的 SSH 认证后：

```sh
git clone --single-branch --branch codex/thesis-archive git@gitlab.lms.tf.fau.de:marina.ritthaler/preprocessingpc.git thesis-archive
cd thesis-archive
git lfs install --local
git lfs pull
python tools/check_archive.py
git lfs fsck
```

HTTPS 地址为 `https://gitlab.lms.tf.fau.de/marina.ritthaler/preprocessingpc.git`，分支不变。不要把密码或令牌写入 URL、脚本、README 或提交。

PDF、全部 LaTeX 编译输入、研究源码/结果和现有演示素材均在目录中。权重、PPTX、部分二进制证据使用 LFS，必须下载实际对象，不能只保存指针文本。校验脚本应报告所有清单文件一致，LFS 检查应通过。

## 编译论文

在已安装 TeX Live（本次验证版本 2025）和 latexmk 的环境中：

```sh
cd thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/current thesis.tex
```

输出 `thesis/build/current/thesis.pdf`。仓库根目录的 `thesis.pdf` 是原样保留的本机论文 PDF。

## 后续补最终答辩稿

在现有本地归档目录进行更新，将最终 PPTX/源文件、PDF 导出及全部必要素材放入 `presentation/final/`。更新 README、presentation/README.txt、DELIVERY_CHECKLIST_CN.md 和上传状态中的待补标记后，在仓库根目录执行：

```powershell
git status --short
git branch --show-current
git remote -v
python tools/check_archive.py --refresh
python tools/check_archive.py
git add presentation README.md DELIVERY_CHECKLIST_CN.md UPLOAD_STATUS_CN.md SHA256SUMS
git diff --cached --stat
git commit -m 'Add final defence presentation and assets'
git push github HEAD:main
git push gitlab HEAD:codex/thesis-archive
```

以上推送命令用于本次已配置的原始归档目录；新克隆目录通常只有 `origin`，应推送到对应的实际远程。若出现非快进错误，先检查新提交，不要强制推送。不要在 `D:\Thesis` 根目录执行 `git add .`，该目录还有原始数据集、环境和历史备份。

本机验证可用的是 Codex 随附 Windows Git / Git LFS。若 PowerShell 找不到 Git，可在当前会话临时加入其目录：

```powershell
$env:Path = 'C:\Users\shenj\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd;' + $env:Path
```

更新后从远端取回新提交，运行同一校验脚本，确认 PDF、权重与 PPTX 均完整。`SOURCE_PROVENANCE.json` 保留初始来源记录；新增文件由更新后的 `SHA256SUMS` 记录。

## 交付范围

现有归档包含完整论文编译工程，以及已整理的 ModelNet40 与 KITTI 源码和研究证据。全量原始数据、KITTI 检测器权重、全部生成点云和逐帧预测不在此 Git 仓库中，不能把本包称为全量服务器备份或一条命令重跑所有实验的环境。

上传到私有仓库后仍需核实最终接收者的访问权限。老师要求的可答辩星期一需作者另行确认和回复；本次上传不会替作者选择日期或发送邮件。
