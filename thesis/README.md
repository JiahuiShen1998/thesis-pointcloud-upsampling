# Complete manuscript source

This is the current complete manuscript project copied from `D:\Thesis\thesis`. It includes all chapters, local template packages, BibTeX data and style, the signed thesis task, figures and editable figure sources, tables, generation scripts and local supporting evidence.

- `thesis.tex`: main entry; `texfiles/` contains chapters 1–6 and front/back matter.
- `figure/`: manuscript figures and their PNG/SVG/PDF editing or display versions.
- `bibfiles/references.bib` and `bibfiles/IEEEtran_thesis.bst`: bibliography database and local style.
- `topic/description.pdf`: signed thesis task.
- `scripts/` and `evidence/`: figure/table generation and provenance. External helpers, fonts or original machine paths required by historical regeneration scripts are separate from the dependencies needed to compile the existing manuscript.
- `../thesis.pdf`: the preserved current local manuscript PDF.

With TeX Live (2025 was used in the source environment) or a compatible distribution installed, run from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/current thesis.tex
```

The output is `build/current/thesis.pdf`. Compiling with the supplied figures does not require Python plotting libraries, original research datasets or external figure-generation helpers. The historical `python scripts/build_current_pdf.py` workflow is also retained; it generates audit/ and thesis.pdf in this copy. The preserved handover PDF is one directory above.

Local Python dependency installations, build caches, historical audit backups, review-crop PDFs, obsolete figure directories and thirty external paper PDFs from bibfiles/papers were not copied. They are outside the current manuscript's 88 local compilation inputs. The bibliography and local style are included; original material remains in the source workspace.
