# 论文归档与上传准备包

此目录整合本机最新整本论文、HPC ModelNet40 实验及 lab KITTI 实验，整理日期为 2026-09-27。当前可以作为研究材料初始归档上传；**最终答辩演示稿尚待补入，不能标记为老师要求的最终完整交付。**

```text
thesis_handover_prepared_20260927/
├── thesis/                    当前整本论文 LaTeX 工程及图源、证据
├── thesis.pdf                 本机当前正式 PDF（163 页）
├── research/
│   ├── README.txt             两组研究代码与结果的说明
│   ├── modelnet40/            HPC 分类实验、配置、日志、14 个权重
│   └── kitti/                 lab 检测实验代码、协议和汇总结果
├── presentation/
│   ├── README.txt             当前素材与待补最终稿的说明
│   ├── modelnet40_materials/  现有分类实验演示及素材
│   └── kitti_materials/       现有检测实验演示及生成素材
├── UPLOAD_GUIDE_CN.md         Windows 上上传 GitHub/GitLab 的步骤
├── DELIVERY_CHECKLIST_CN.md   按老师邮件逐项对照
├── SOURCE_PROVENANCE.json     来源文件与字节校验记录
├── SHA256SUMS                本包文件的 SHA-256 清单
└── tools/check_archive.py    校验或更新清单
```

`thesis/thesis.tex` 是当前论文入口；不要用服务器上的旧第 3–6 章草稿替换它。子目录中的历史 README 反映各原工作区的范围；例如 lab 没有最终 ModelNet40 分类结果，但本包 `research/modelnet40` 已补入 HPC 的 14 个最终分支，二者不能混淆。

原始资料未改动。仓库的实际提交与上传状态见 `UPLOAD_STATUS_CN.md`。老师邮件只要求归档内容，没有指定必须上传 GitHub/GitLab，也没有给出上传网址。GitHub/GitLab 是作者计划采用的传递方式。

本包含 CV、签字任务书和研究素材，建议两个平台先采用私有仓库。开源展示可另建精简版本；不要直接改动此份老师归档中的事实和实验边界。
