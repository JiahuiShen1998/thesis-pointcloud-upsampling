# English archive edition

The current archive contains English project documentation, historical working records, script comments, presentation text, and the complete ModelNet40 experimental dossier. The dossier PDF is generated from the reviewed English Markdown so that its tables have one source. Existing university forms and formal front matter retain their original official wording.

The main README, handover checklist, upload instructions, ModelNet40 dossier, and experimental progress brief were reviewed against their sources. Longer historical records are English working translations retained for traceability; they are not replacement thesis chapters or newly validated research claims. The final manuscript and the numerical evidence retain their authority over older working notes.

## Provenance and integrity

- `SOURCE_PROVENANCE.json` records the initial copy and recovery operation. Its original paths and hashes intentionally describe that earlier snapshot.
- [ENGLISH_FILENAME_MAP.json](ENGLISH_FILENAME_MAP.json) maps renamed documents to their English-edition paths.
- `SHA256SUMS` records the current file contents. Run `python tools/check_archive.py` after pulling all Git LFS objects.
- Numerical result files, arrays, model checkpoints, LaTeX manuscript sources, and the archived main thesis PDF were not changed by this language update.
- Mathematical expressions in the historical chapter draft retain their original notation; descriptive labels inside formulas were translated.
- External figure/review helper directories used by selected maintenance scripts can be supplied through `FIGURE_HELPERS_DIR` and `RESEARCH_REVIEW_HELPERS_DIR`; see the manuscript script README.

This edition changes documentation and presentation language. It does not claim that unfinished experiments or the final defence presentation have been completed. See the root delivery checklist for remaining handover work.
