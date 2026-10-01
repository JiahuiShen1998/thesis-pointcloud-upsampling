# Environment and reproducibility notes

The original experiments ran on an HPC/Slurm environment with method-specific software stacks. No single complete lock file or container image was found, so this archive does not claim one-command reproduction.

Known components include:

- Python, NumPy, SciPy, pandas, PyYAML, Matplotlib, and scikit-learn for data, analysis, and figures.
- PyTorch/CUDA for PointNet++ and PDANS-related code.
- TensorFlow 1 plus compiled CUDA custom operations for the archived PU-Net and PU-GCN integrations.
- Open3D and PyTorch3D in method-specific environments.
- TeX Live and pdfLaTeX/latexmk for the archived manuscript; see [the verified build instructions](../../thesis/README.md).
- Slurm for the supplied `.sbatch` jobs.

The exact CUDA, compiler, framework, pretrained-weight, and GPU compatibility requirements differ by method. Historical fixes and failures are documented in `results/reports/` and the [archived experiment reports](results/reports/).

For inspection without retraining, use the archived CSV/JSON/Markdown reports and final run metrics. For a full rerun, first replace cluster-specific absolute paths, acquire the raw ModelNet40 data and separately licensed pretrained method weights, reconstruct each method's environment, and run smoke validation before scheduling full jobs.


For the tested CPU inference environment and evidence reconstruction commands, see [the current reproduction guide](../../docs/REPRODUCING.md).
