# Handover requirements and archive coverage

This checklist follows the supervisor's archive email.

| Requirement | Status | Location |
|---|---|---|
| All files needed to generate the thesis | Included; all 88 local compilation inputs were checked against the successful build record | thesis/ |
| Research source code and results | ModelNet40 and KITTI archives combined; fourteen classifier runs include metrics, 200-epoch logs and checkpoints | research/ |
| A research README.txt | Included, with the original subproject notes retained | research/README.txt |
| Final defence presentation and its source/assets | Pending; five existing PPTX files are progress/candidate materials | presentation/ |
| Thesis PDF | Current local 163-page PDF included; the author reported submitting the final thesis separately | thesis.pdf |
| Available Mondays for the defence | The author must select actual available dates and reply separately | Outside the file archive |

## Issues resolved during archive preparation

- The original ModelNet40 manifest contained 1,663 entries: 1,646 files were present and passed their hashes; 17 timestamped historical notes were missing because their filenames contained colons. Their original bytes were recovered from the supplied Git index, matched against the original hashes, and saved under Windows-compatible names. See research/modelnet40/WINDOWS_FILENAME_MAP.json.
- The lab ZIP matched its recorded SHA-256 and passed CRC checks. All 780 extracted file entries matched. Seven Python cache files explain the difference from the original compact-package count of 773.
- Statements in the server archives that the complete thesis was unavailable describe those servers. The complete current manuscript was found locally and included. Historical chapter drafts do not replace the current chapters 1–2.
- The old manuscript .gitignore ignored thesis.pdf. The combined archive keeps its original PDF at the repository root with separate ignore rules.
- The old compact lab release omitted final HPC classification results. The combined archive includes the fourteen final/control ModelNet40 classifier runs.

## Remaining handover work

1. Complete the final defence presentation covering both ModelNet40 and KITTI. Add its editable source, PDF and required assets, then update this checklist.
2. Preserve the archived thesis.pdf. If the exact PDF submitted to the university differs, archive and label that exact version separately. A submission statement alone does not establish byte identity.
3. The archive is uploaded to GitHub and to the school GitLab project's thesis-archive branch. See UPLOAD_STATUS.md for verified upload milestones. Confirm that the intended recipient can access the archive branch.
4. Select Mondays on which the author is actually available and reply separately. The supervisor's email did not specify a Git hosting platform, repository URL or upload deadline.

The full raw datasets, KITTI detector weights and raw per-frame predictions are not included. The email did not explicitly request these assets. If a separate institutional policy requires complete raw research data, archive those assets from the original servers in controlled storage. This compact archive is not a complete server backup.
