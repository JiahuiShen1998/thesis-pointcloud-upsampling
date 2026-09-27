# Windows 上上传到 GitHub 和 GitLab

本次已准备的目录：

`D:\Thesis\Git\thesis_handover_prepared_20260927`

此文件保留首次整理时的手动上传步骤；当前已由助手直接处理上传，实际状态见 `UPLOAD_STATUS_CN.md`。不要重复初始化。可以先上传这份研究归档快照，最终答辩稿完成后再补交；在此之前保留 README 的“最终答辩稿待补”状态。

## 1. 建立两个空仓库

- GitHub：新建 repository，建议 Private。
- GitLab：在你实际使用的 GitLab 网站新建 blank project，建议 Private。若学校使用自己的 GitLab 域名，就使用学校项目的 Clone URL；不要直接套用 gitlab.com。
- 两端均先不自动生成 README、License、.gitignore，避免与本地第一次提交产生分叉。
- 复制两个项目的 SSH 或 HTTPS Clone URL，并完成相应登录/SSH 配置。不要把访问令牌写进归档文件。

老师邮件本身没有指定上传渠道。若老师提供了学校存储链接，应以那个入口为交付渠道；GitHub/GitLab 链接还需确保老师能访问。

## 2. 在 PowerShell 进入正确目录

```powershell
Set-Location 'D:\Thesis\Git\thesis_handover_prepared_20260927'
git --version
git lfs version
```

本次核实 Codex 自带的 Windows Git LFS 可用（3.7.1），但当前 WSL 的 git-lfs 无法执行。推荐使用 Windows Git。若 PowerShell 的 git 命令不可用，可仅在本次终端加入已存在的路径：

```powershell
$env:Path = 'C:\Users\shenj\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd;' + $env:Path
```

不要在 D:\Thesis 根目录执行 git add .；那里含数据集、环境和大量历史备份。

## 3. 初始化、校验并提交

以下只适用于这份新建、尚未初始化的准备目录。姓名、邮箱和 URL 需要换成你的值。

```powershell
git init -b codex/thesis-archive
git config user.name '你的姓名'
git config user.email '你的 Git 提交邮箱'
git config core.longpaths true
git lfs install --local
python tools/check_archive.py

git add .
git status --short
git lfs ls-files
git ls-files thesis.pdf research/README.txt
# 确认状态和文件列表正确后提交
git commit -m 'Archive thesis source and research results; final slides pending'
```

`.gitattributes` 已把 checkpoint、PPTX 和点云二进制证据交给 LFS；无需再次手动 track。若 python 不在 PATH，使用已安装的 Python 3 或 Codex 随附解释器执行同一脚本。

## 4. 推送两端

```powershell
git remote add github '替换成 GitHub Clone URL'
git remote add gitlab '替换成 GitLab Clone URL'
git remote -v
git push -u github codex/thesis-archive
git push gitlab codex/thesis-archive
```

GitLab 的 LFS 开关、存储配额和访问权限取决于实际实例与项目，必须在所用项目核实。不要把原服务器 .git 文件夹覆盖到本包；Windows 文件名映射已经发生变化，新初始化更清楚。

## 5. 验证下载并补最终 PPT

推送成功后，从两端分别克隆到新的空目录，各运行 `git lfs pull` 和 `python tools/check_archive.py`，确认下载到真实权重/PPTX，不是只含几行文本的 LFS 指针。检查 thesis.pdf、thesis/thesis.tex、research/README.txt 和两个研究子目录。

最终答辩稿准备好后放入 presentation/final/，补齐 PDF 和素材，更新 README/DELIVERY_CHECKLIST 的待办状态，然后：

```powershell
python tools/check_archive.py --refresh
python tools/check_archive.py
git add .
git commit -m 'Add final defence presentation and assets'
git push github codex/thesis-archive
git push gitlab codex/thesis-archive
```

添加最终稿后的 SOURCE_PROVENANCE.json 仍只描述首次整理来源；新文件由更新后的 SHA256SUMS 记录。

## 为什么使用 Git/LFS

GitHub 网页上传单文件上限 25 MiB，普通 Git 阻止超过 100 MiB 的单文件。原 lab ZIP 约 45.4 MiB，已超过网页上传上限；整包 ZIP 也不适合作为源代码仓库内容。这里按目录提交，PPTX/权重等由 LFS 管理。

本包含个人 CV、签字任务书和论文研究素材，推荐私有上传。公开展示应另外准备不含这些材料的精简仓库；本包不是已经审查完成的公开发行版。

官方依据（2026-09-27 查阅）：

- [GitHub 本地代码首次上传](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [GitHub 文件大小限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitLab 项目推送与实例地址](https://docs.gitlab.com/topics/git/project/)
- [GitLab LFS](https://docs.gitlab.com/topics/git/lfs/)
