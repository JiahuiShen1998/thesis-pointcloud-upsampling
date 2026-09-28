# Validation performed for this archive

- The copied thesis project compiled independently with TeX Live 2025 and latexmk 4.86a, producing 163 pages.
- Extracted page text and page content streams matched the original PDF on all 163 pages. Whole-file hashes differ after regeneration because PDF metadata can change; the repository-root thesis.pdf preserves the original local PDF.
- No undefined references, undefined citations or duplicate labels were found. The build retained three local-package naming notices and the existing float-only page notice on page 92.
- All 88 local manuscript compilation inputs were present and matched the latest successful build record under the Windows latexmk CRLF hashing convention.
- All 1,646 originally present ModelNet40 files matched the source manifest. The 17 missing historical notes were recovered from original Git objects and matched their original SHA-256 hashes; only their filenames were changed for Windows compatibility.
- The lab ZIP passed CRC verification, matched its recorded SHA-256, and matched all 780 extracted file entries.
- All fourteen final/control ModelNet40 runs contained valid metrics.json, a training log reaching Epoch 200/200, and best_model.pth.
- The five PPTX files passed ZIP-integrity checks. One older deck has six historical HPC hyperlinks on slide 17, as documented in presentation/README.txt. Archive integrity does not establish final defence-content approval.
- No complete GPU reproduction was run, and no email was sent. The archive was uploaded to GitHub and school GitLab and independently downloaded. SHA-256 and Git LFS checks passed; UPLOAD_STATUS.md records the exact verified baseline.

Detailed local audit logs are retained under Git/upload_audit_20260927 in the original workspace and are not distributed with this repository. English localization changes are recorded separately from the original copy/build validation. SHA256SUMS is authoritative for the current archived files.

## English edition checks (2026-09-28)

- Checked 1,347 readable UTF-8 files and every XML part in all five PPTX files: no Chinese characters or excluded tool branding remained in the current file contents.
- Scanned extracted text and metadata in all 149 PDFs: no Chinese text was found and all files were readable.
- OCR covered 158 unique raster images. The six flagged snippets in five images were inspected visually and were axis marks, line symbols, or a plot marker rather than Chinese text.
- Parsed all 355 Python files successfully and checked the local links in the main handover guides.
- Preserved the mathematical notation in the historical chapter draft; translated labels inside formulas are English.
- The main thesis PDF remains byte-identical, with SHA-256 `ef18560e78656b0e1e390c0b733fa68ce6956817beaff66515d6d4fb5a3bafb6`.
- Existing numerical result files, arrays, checkpoints and LaTeX sources were unchanged. The English dossier was regenerated from its reviewed Markdown; all ten pages were visually checked.
- Sixteen previous document paths map to English editions, including four existing English companions that replace duplicate language copies. See `docs/ENGLISH_FILENAME_MAP.json`.

These checks cover language presence, source integrity and archive readability. Longer historical working translations are not a substitute for the final manuscript or a new scientific review.
