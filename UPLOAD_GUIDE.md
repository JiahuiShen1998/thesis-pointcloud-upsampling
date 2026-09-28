# Download, verify and update the complete thesis archive

Local archive directory: `D:\Thesis\Git\thesis_handover_prepared_20260927`.

## Repositories and branches

| Platform | Project | Archive branch |
|---|---|---|
| School GitLab | [PreprocessingPC](https://gitlab.lms.tf.fau.de/marina.ritthaler/preprocessingpc/-/tree/thesis-archive) | `thesis-archive` |
| Personal GitHub | [thesis-pointcloud-upsampling](https://github.com/JiahuiShen1998/thesis-pointcloud-upsampling) | `main` |

The school account has a personal project quota of 0. The author selected the existing private PreprocessingPC project. Its existing `main` and `share/modelnet40-pointnet2-presentation` branches are retained. Open the archive branch linked above to browse the complete thesis.

See [UPLOAD_STATUS.md](UPLOAD_STATUS.md) for upload and verification records. The original local archive already has Git and remotes configured; do not initialize it again. The final defence deck remains pending.

## Download and verify

Install Git, Git LFS and Python 3. After configuring SSH access to the school account:

```sh
git clone --single-branch --branch thesis-archive git@gitlab.lms.tf.fau.de:marina.ritthaler/preprocessingpc.git thesis-archive
cd thesis-archive
git lfs install --local
git lfs pull
python tools/check_archive.py
git lfs fsck
```

The HTTPS clone URL is `https://gitlab.lms.tf.fau.de/marina.ritthaler/preprocessingpc.git`; use the same branch. Keep passwords and access tokens out of repository URLs, scripts, documentation and commits.

The archive includes the PDF, all manuscript compilation inputs, research source/results and current presentation assets. Checkpoints, PPTX files and selected binary evidence use Git LFS. Download the actual objects, not just pointer text. Both checksum verification and the LFS check must pass.

## Compile the manuscript

With TeX Live (2025 was used for the verified build) and latexmk installed:

```sh
cd thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/current thesis.tex
```

The output is `thesis/build/current/thesis.pdf`. The repository-root `thesis.pdf` is the preserved local manuscript PDF.

## Add the final defence presentation

In the existing local archive, add the final PPTX or equivalent source, PDF export and all necessary assets under `presentation/final/`. Update README.md, presentation/README.txt, DELIVERY_CHECKLIST.md and the upload status. Then run from the repository root:

```powershell
git status --short
git branch --show-current
git remote -v
python tools/check_archive.py --refresh
python tools/check_archive.py
git add presentation README.md DELIVERY_CHECKLIST.md UPLOAD_STATUS.md SHA256SUMS
git diff --cached --stat
git commit -m 'Add final defence presentation and assets'
git push github HEAD:main
git push gitlab HEAD:thesis-archive
```

These remote names apply to the original prepared archive. A fresh clone usually has only `origin`; use its actual remote and branch. Inspect new remote commits before resolving any non-fast-forward error. Do not force-push. Do not run `git add .` in the parent `D:\Thesis` directory, which also contains datasets, environments and historical backups.

On Windows, ensure Git for Windows and Git LFS are installed and available on PATH. Retrieve each uploaded update and repeat the checks to confirm that PDFs, checkpoints and decks are complete. SOURCE_PROVENANCE.json records the initial copied/recovered material; the current SHA256SUMS records later documentation and file changes.

## Handover scope

This repository contains the complete manuscript build project and the assembled ModelNet40/KITTI source and evidence archives. Full raw datasets, KITTI detector checkpoints, all generated point clouds and raw per-frame predictions are outside this Git archive. It is not a complete server backup or a one-command environment for rerunning all experiments.

Confirm recipient access before final handover. The author must separately select and communicate available Mondays for the defence; repository uploads do not choose dates or send email.
