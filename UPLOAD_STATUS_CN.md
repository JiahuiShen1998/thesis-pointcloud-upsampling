# 上传状态

更新日期：2026-09-28。整套现有论文归档已经上传；**最终答辩 PPT 仍待补**。

| 平台 | 归档位置 | 状态 |
|---|---|---|
| 学校 GitLab | [PreprocessingPC / codex/thesis-archive](https://gitlab.lms.tf.fau.de/marina.ritthaler/preprocessingpc/-/tree/codex/thesis-archive) | 私有项目；整套归档和详细 README 已上传，重新下载验证通过 |
| 个人 GitHub | [thesis-pointcloud-upsampling / main](https://github.com/JiahuiShen1998/thesis-pointcloud-upsampling) | 私有仓库；归档和完整项目说明已上传 |

## 已完成的验证

- GitLab 归档提交 `495f4d9cffee2c92d72e8dccde2ab5a82f4c5ff4` 已从学校服务器独立克隆，跟踪 1,877 个文件；清单中的 1,876 个文件全部通过 SHA-256 校验，Git LFS fsck 通过，克隆工作区干净。
- GitHub 此前的归档提交 `07712dfeb1f2d32a7702ce2a6137d0c6515c2f04` 也已独立下载，1,876 个清单文件及 LFS 检查全部通过；之后已同步完整 README 和学校项目入口。
- 187 个唯一 LFS 对象（约 408 MB）上传到学校 GitLab，覆盖 209 个受 LFS 管理的文件路径。
- 学校项目原 `main` 保持 `871e82d33781b70d852678c0fc70ac7a7503820b`，原 `share/modelnet40-pointnet2-presentation` 保持 `daacfaf1a76be4c42d01431c225a69a715818db2`；未覆盖原分支。
- 学校账号为 `Stud_Jiahui_Shen`。作者明确授权使用现有 PreprocessingPC 项目，权限为 Maintainer，LFS 已启用。临时 API 凭据用完即撤销，未写入文件或仓库。

## 打开与下载

学校项目完整论文内容在 `codex/thesis-archive` 分支，默认 `main` 不是本次归档入口。直接打开上表中的分支链接即可浏览完整 README、论文及研究材料。下载和更新命令见 [UPLOAD_GUIDE_CN.md](UPLOAD_GUIDE_CN.md)。

后续提交可能更新状态记录；下载当前版本后运行 `python tools/check_archive.py` 和 `git lfs fsck`，核对当前清单。提交历史保留上述具体校验时点。

## 交付余项

- 最终答辩稿、PDF 导出和素材待加入 `presentation/final/`；现有五份 PPTX 为进展稿/候选材料。
- 全量原始数据、KITTI 检测器权重和逐帧预测不在本次 Git 归档中，范围详见项目 README。
- 学校项目原有成员权限保持不变；GitHub 未新增协作者。未替作者发送邮件或选择答辩日期。
