# Manuscript build, figure sources and verification

The main entry is thesis.tex. In the original manuscript workflow, thesis.pdf was generated in the manuscript directory; the preserved handover PDF is now at the repository root, one level above thesis/.

The manuscript uses the FAU/LMS template with 12 TeX pt body text and no Appendix. The 2026-09-17 revision followed the supervisor's email and the author's instruction to leave chapters 1 and 2 unchanged. The original detailed change record was audit/teacher_email_revision_20260917/CHANGES_CN.md; historical audit folders are not included in this handover copy.

## Routine compilation

From thesis/, with TeX Live installed and latexmk on PATH:

```sh
python scripts/build_current_pdf.py
```

Intermediate files are written under build/current/. After a successful build, the script updates thesis/thesis.pdf. Save annotations and close an open old PDF before replacing it. Compilation does not regenerate experimental data. The preserved repository-root PDF remains separate.

## Current verification utilities

```sh
python scripts/check_teacher_revision_citations.py
python scripts/check_teacher_revision_layout.py
python scripts/update_active_figure_manifest.py
```

Citation checks cover included source, BibTeX numbering and PDF links. Layout checks use the active figure manifest and check top placement and final font sizes. If the old PDF is locked, add --built to the first two commands to inspect build/current/thesis.pdf; this does not imply that another PDF copy was updated.

## Scientific figures

- figure/active_manifest.json records the 38 scientific figures actually included in the manuscript and their hashes.
- figure/cv_portrait.png is the author-supplied original photograph, without cropping or retouching.
- scripts/make_teacher_revision_figures.py generates the two revised ModelNet point-cloud figures.
- The original audit/teacher_email_revision_20260917/figure_manifest.json recorded source-array hashes, point counts, recomputed CD/HD and the shared colour scale. The original figure_qa/ folder held alignment, font-size and collision reports. These historical audit folders are not included here.

To regenerate only those two figures:

```sh
python scripts/make_teacher_revision_figures.py
```

Regeneration uses Python/matplotlib and requires the original data and compatible dependencies. External figure helpers are selected through FIGURE_HELPERS_DIR; optional research-review helpers use RESEARCH_REVIEW_HELPERS_DIR. These external helper modules are not needed to compile the supplied PDF figures.

Each point-cloud figure contains the input, reference and five methods. Output panels use a distance colour scale. The scripts do not generate new experimental predictions or assign classification success/failure from appearance. expand_results_figures.py --part all calls the revised two-figure functions instead of restoring the old nine-panel versions. Other historical figures remain available; the manuscript and active manifest define which are included.

## Data and bibliography

ModelNet evidence under evidence/modelnet40_hpc/ comes from the HPC experiment. KITTI evidence under evidence/kitti_lab/ comes from the lab experiment. bibfiles/references.bib is included; external literature PDFs from the original bibfiles/papers/ directory are not part of this handover. Experiment logs support verification and are not bibliography entries.

P2F is shown alongside CD/HD but uses a separate secondary protocol with independent cloud/mesh normalization. prepare_evidence.py generated the P2F columns and is unnecessary for routine compilation. Later KITTI training adapts the detectors; PU-GCN itself remains fixed. Historical subsets, final full-validation results and matched events must retain their different metric scopes.

## One-time historical scripts

restore_*, clean_recovered_prose.py, apply_record_revision.py, revise_citation_positions.py and historical audit/revise_*.py scripts record earlier edits. Do not rerun one-time editing scripts on the current manuscript. Earlier review PDFs were extracted from the full PDF; the current manuscript is authoritative.
