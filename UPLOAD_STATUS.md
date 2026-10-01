# Upload and verification record

Updated: 2026-10-01. The available thesis archive is uploaded; the final defence presentation is pending.

| Platform | Archive location |
|---|---|
| School GitLab | [PreprocessingPC / thesis-archive](https://gitlab.lms.tf.fau.de/marina.ritthaler/preprocessingpc/-/tree/thesis-archive) |
| Personal GitHub | [thesis-pointcloud-upsampling / main](https://github.com/JiahuiShen1998/thesis-pointcloud-upsampling) |

Both repositories are private. The school account is Stud_Jiahui_Shen and has Maintainer access to the selected project; Git LFS is enabled. Temporary API credentials used for upload administration were revoked and were not saved in the archive.

## Verified upload baseline

Commit `234b3a84f150abce61e6aed250908ab1c732b461` was independently retrieved from both platforms. Each checkout contained 1,877 tracked files; all 1,876 entries in SHA256SUMS matched, Git LFS fsck passed, and both working trees were clean.

The school upload included 187 unique LFS objects, approximately 408 MB, covering 209 tracked file paths. The original school `main` remained at `871e82d33781b70d852678c0fc70ac7a7503820b`, and `share/modelnet40-pointnet2-presentation` remained at `daacfaf1a76be4c42d01431c225a69a715818db2`.

The subsequent English localization translates archived documentation and presentation text and updates paths and checksums. The baseline hashes above describe the earlier verified snapshot. Verify the current version with `python tools/check_archive.py` and `git lfs fsck` after downloading the actual LFS objects.

## Portfolio and reproducibility update (2026-10-01)

The homepage now presents the research question, contributions, result tables and original figures, with separate code-reading routes for ML/CV engineering, research and LiDAR perception. Generated HTML exports are marked for exclusion from source-language statistics; all interactive reports remain in the archive.

New portable entry points rebuild the recorded results and execute one real archived classifier checkpoint on an included cloud. Local Windows verification passed all 11 evidence/asset tests and produced the expected airplane prediction with Python 3.12, PyTorch 2.6.0+cpu and NumPy 2.2.6. Full GPU retraining was not performed. See [the reproduction guide](docs/REPRODUCING.md) for exact commands, external assets and limitations.

The [GitHub Actions workflow](.github/workflows/reproducibility.yml) checks evidence and CPU inference independently. Its live run status is in the GitHub Actions tab. File counts from older milestones above describe those snapshots; the current `SHA256SUMS` is authoritative after a complete download.

## Access and updates

The complete school archive is on `thesis-archive`; use the direct branch link above. The original school `main` is retained. Instructions for downloading, compiling and updating the archive are in UPLOAD_GUIDE.md.

## Remaining work

- Add the final defence deck, PDF export and supporting assets under presentation/final/. The existing five PPTX files are progress/candidate materials.
- Full raw datasets, KITTI detector weights and raw predictions remain outside this archive; see the research README files for the exact scope.
- Existing school project membership is unchanged. No GitHub collaborators were added, and no defence dates or emails were sent on the author's behalf.
