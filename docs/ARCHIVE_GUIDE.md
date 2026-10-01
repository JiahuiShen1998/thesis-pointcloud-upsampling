# Investigating the Impact of Point Cloud Upsampling on 3D Object Detection Performance

Master's thesis by **Jiahui Shen**, Friedrich-Alexander-Universität Erlangen-Nürnberg, Chair of Multimedia Communications and Signal Processing (LMS). Supervisor: **Marina Ritthaler, M.Sc.** Manuscript: September 2026.

This repository brings together the complete current thesis LaTeX project and PDF, the ModelNet40 and KITTI research archives, and the available presentation materials. It supports inspection of the thesis, its experimental protocols, and the recorded results.

**Handover status: the final defence presentation is still pending.** The existing slide decks are progress/candidate materials. Raw datasets and several external assets needed for a complete GPU rerun are not part of this archive; see the scope below.

## Start here

- [Read the thesis PDF](../thesis.pdf) — 163 pages, preserved from the current local manuscript.
- [Browse the thesis source](../thesis/) — main entry: [thesis.tex](../thesis/thesis.tex).
- [Research archive overview](../research/README.txt) — how the two experimental parts fit together.
- [Presentation status](../presentation/README.txt) — existing material and the remaining final deck.
- [Delivery checklist](../DELIVERY_CHECKLIST.md), [upload status](../UPLOAD_STATUS.md), and [download and update guide](../UPLOAD_GUIDE.md).

The school GitLab archive is on **`thesis-archive`** in [PreprocessingPC](https://gitlab.lms.tf.fau.de/marina.ritthaler/preprocessingpc/-/tree/thesis-archive). The project's existing branches are retained. The author's public [GitHub copy](https://github.com/JiahuiShen1998/thesis-pointcloud-upsampling) uses **`main`**.

## Research scope

The thesis studies whether point-cloud upsampling benefits downstream perception, comparing densification of an original input with recovery after fourfold sparsification.

- **ModelNet40:** EAR, PDANS, PU-Net, PU-GCN and PU-EdgeFormer are evaluated with separately trained PointNet++ classifiers and equal-cardinality geometric references. Fourteen final/control classifier runs are archived with metrics, full 200-epoch training logs, source snapshots and best-model checkpoints.
- **KITTI:** PointRCNN and CenterPoint are evaluated with PU-GCN inputs, detector adaptation, and extended observed-first training. The archive includes the full-validation summaries, protocol records, convergence evidence, and selected diagnostics used in the manuscript.

Geometric quality and downstream performance are separate outcomes. Use the original metric, split, input-line and training-budget labels when reading or reusing results.

## Repository layout

```text
.
├── thesis.pdf                       Preserved manuscript PDF
├── thesis/
│   ├── thesis.tex                   Main LaTeX entry
│   ├── texfiles/                    Chapters 1–6 and front/back matter
│   ├── bibfiles/                    Bibliography and local BibTeX style
│   ├── packages/                    Local LaTeX packages
│   ├── figure/                      Figures, including editable sources
│   ├── topic/                       Signed thesis task
│   ├── scripts/                     Figure/table and manuscript utilities
│   └── evidence/                    Supporting local evidence
├── research/
│   ├── README.txt                   Combined research handover notes
│   ├── modelnet40/
│   │   ├── code/                    Scripts, configurations and Slurm jobs
│   │   └── results/                 Reports, figures and 14 classifier runs
│   └── kitti/
│       ├── source/                  Integration, training and evaluation code
│       ├── results/                 Validation and convergence evidence
│       └── environment/             Dependency and omitted-asset notes
├── presentation/                    Existing decks, exports and assets
├── tools/check_archive.py            SHA-256 verification utility
├── SHA256SUMS                       Current archive file checksums
├── SOURCE_PROVENANCE.json            Original copy/recovery provenance
├── DELIVERY_CHECKLIST.md          Supervisor requirement checklist
├── UPLOAD_STATUS.md               Upload and verification record
└── UPLOAD_GUIDE.md                Download and future update instructions
```

See [English edition and provenance](../docs/ENGLISH_EDITION.md) for the language update and filename mapping. Earlier project-planning files are retained for history. See [docs/LEGACY_NOTES_README.md](../docs/LEGACY_NOTES_README.md); they do not override the final experimental protocols or define a universal runtime environment.

## Download the complete archive

Requirements: access to the private GitLab project, Git, Git LFS, and Python 3 for checksum verification. An SSH key must be registered with the school GitLab account for the SSH command below.

```sh
git lfs version
git clone --single-branch --branch thesis-archive git@gitlab.lms.tf.fau.de:marina.ritthaler/preprocessingpc.git thesis-archive
cd thesis-archive
git lfs install --local
git lfs pull
python tools/check_archive.py
git lfs fsck
```

If SSH is unavailable, use the project's HTTPS clone URL with the same branch:

```text
https://gitlab.lms.tf.fau.de/marina.ritthaler/preprocessingpc.git
```

Checkpoints, PPTX decks and selected binary evidence use Git LFS. A repository containing only LFS pointer text is not a complete download. Verification must report that every file in `SHA256SUMS` matches. Downloading a source ZIP alone is not the verified handover procedure.

The archived file content is about 750 MiB; allow additional space for Git history and the local LFS object cache.

## Compile the thesis

The manuscript was independently compiled with TeX Live 2025 during the handover audit. Install TeX Live with the required LaTeX packages and `latexmk`, then run from the repository root:

```sh
cd thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/current thesis.tex
```

The output is `thesis/build/current/thesis.pdf`. The original archived PDF remains at the repository root. Rendering the existing manuscript uses the supplied figures and bibliography; it does not require the original datasets or a GPU. See [the manuscript build notes](../thesis/README.md).

The independently compiled PDF matched the archived 163-page PDF in page text and page content streams. PDF metadata can differ, so a fresh build is not claimed to have the same whole-file hash.

## Inspect the experimental evidence

### ModelNet40

Start with [research/modelnet40/README.txt](../research/modelnet40/README.txt), then inspect:

- [Final classification report including PU-EdgeFormer](../research/modelnet40/results/reports/modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.csv).
- [Equal-cardinality geometry protocol](../research/modelnet40/results/reports/modelnet40_geometry_equal_n_protocol.md), [summary](../research/modelnet40/results/reports/modelnet40_geometry_equal_n_summary.csv), and [per-sample records](../research/modelnet40/results/reports/modelnet40_geometry_equal_n_per_sample.csv).
- [Final classifier runs](../research/modelnet40/results/pointnet2_final_runs/) — `metrics.json`, `train.log`, best checkpoints and saved source.
- [Research code](../research/modelnet40/code/) and [result figures](../research/modelnet40/results/figures/).

Interpretation limits: final classifier runs use seed 42; best checkpoints were selected using the test set; reported OA/mAcc average batch-level quantities. Implemented Chamfer distance uses unsquared Euclidean nearest-neighbour distances. Downstream classifiers were trained from scratch; upsamplers use existing pretrained weights.

### KITTI

Start with [research/kitti/README.txt](../research/kitti/README.txt), then inspect:

- [Final experiment brief, 2026-09-17](../research/kitti/results/EXPERIMENT_BRIEF_20260917_EN.md).
- [Full-validation results](../research/kitti/results/latest_full_validation/) and [extended-training convergence evidence](../research/kitti/results/convergence/).
- [Source code](../research/kitti/source/) and [dependency notes](../research/kitti/environment/).
- [Data and large-file scope](../research/kitti/DATA_AND_LARGE_FILES.md).

Full-validation metrics, selected-frame diagnostics and per-object examples have different scopes. Detector adaptation budgets and checkpoint selection must remain explicit when comparing results. The historical ModelNet40 content inside the KITTI source package is lab protocol/smoke work; final HPC classification runs are in the separate `research/modelnet40/` archive.

## Reproduction boundaries and provenance

This is an archive of the available authored source and result evidence, not a complete image of the original training servers. Full numerical reruns require separately obtained datasets, upstream code, compatible dependencies and custom CUDA/TensorFlow/PyTorch operations, external upsampler weights, and substantial GPU resources. Historical scripts retain machine-specific paths that must be adapted. No full GPU rerun was performed as part of the upload verification.

Included are the fourteen ModelNet40 classifier checkpoints. Omitted are the complete raw datasets, full generated point-cloud trees, most external method repositories and pretrained upsampler assets, KITTI detector checkpoints, and raw per-frame detector predictions. Selected supporting evidence is retained under `thesis/evidence/`.

`SOURCE_PROVENANCE.json` records the initial copied/recovered material. Seventeen historical ModelNet40 note filenames containing colons were renamed for Windows compatibility; their bytes match the original Git objects. The mapping is in [WINDOWS_FILENAME_MAP.json](../research/modelnet40/WINDOWS_FILENAME_MAP.json). `SHA256SUMS` records the current archive contents, including later documentation updates.

Original third-party licence files and notices are retained. This handover archive does not grant a new blanket licence to third-party code, datasets or weights.

## Remaining handover work

1. Add the actual final defence deck, PDF export and required assets under `presentation/final/`. The current five PPTX files are supporting progress/candidate decks.
2. Update the presentation and delivery status, refresh checksums, commit and upload the added material using [the update guide](../UPLOAD_GUIDE.md).
3. Use the public GitHub link for portfolio readers. Confirm the supervisor can access the private school GitLab archive, and separately communicate actual available defence dates.

The exact PDF previously sent to the university was not independently retrieved during this audit; the archived PDF is the current verified local version.
