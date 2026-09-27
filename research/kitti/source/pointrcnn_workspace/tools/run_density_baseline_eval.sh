#!/usr/bin/env bash
set -euo pipefail

# Generic runner for method-agnostic density evaluations.
#
# This script does not generate point clouds. It simply evaluates already
# prepared velodyne folders with the unified density framework.
#
# Provide a manifest JSON file via DENSITY_METHODS_JSON or --manifest.
# Each manifest entry must include:
#   experiment_name, method_name, processing_type,
#   reference_velodyne, processed_velodyne, output_root

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv_pointrcnn/bin/python}"
MANIFEST="${1:-${DENSITY_METHODS_JSON:-}}"

if [[ -z "$MANIFEST" ]]; then
  cat <<'EOF'
Usage:
  DENSITY_METHODS_JSON=path/to/manifest.json tools/run_density_baseline_eval.sh
or
  tools/run_density_baseline_eval.sh path/to/manifest.json

The manifest must be a JSON array of experiment records with:
  experiment_name, method_name, processing_type,
  reference_velodyne, processed_velodyne, output_root
EOF
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

"$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import shlex
import subprocess
import sys
from pathlib import Path

root = Path('/home/ra87racy/projects/baseline_detectors/PointRCNN')
manifest_path = Path(sys.argv[1])
entries = json.loads(manifest_path.read_text(encoding='utf-8'))
if not isinstance(entries, list):
    raise SystemExit('Manifest must be a JSON array.')

for entry in entries:
    required = ['experiment_name', 'method_name', 'processing_type', 'reference_velodyne', 'processed_velodyne', 'output_root']
    missing = [key for key in required if key not in entry]
    if missing:
        raise SystemExit(f"Manifest entry missing keys: {missing}")

    cmd = [
        sys.executable,
        str(root / 'tools' / 'evaluate_density_pointclouds.py'),
        '--training-dir', str(root / 'data/KITTI/object/training'),
        '--reference-velodyne', entry['reference_velodyne'],
        '--processed-velodyne', entry['processed_velodyne'],
        '--output-root', entry['output_root'],
        '--experiment-name', entry['experiment_name'],
        '--method-name', entry['method_name'],
        '--processing-type', entry['processing_type'],
    ]
    if entry.get('detector_results_root'):
        cmd.extend(['--detector-results-root', entry['detector_results_root']])
    if entry.get('dataset'):
        cmd.extend(['--dataset', entry['dataset']])
    if entry.get('split'):
        cmd.extend(['--split', entry['split']])
    if entry.get('max_visualized_samples') is not None:
        cmd.extend(['--max-visualized-samples', str(entry['max_visualized_samples'])])
    if entry.get('bev_cell_size') is not None:
        cmd.extend(['--bev-cell-size', str(entry['bev_cell_size'])])
    if entry.get('seed') is not None:
        cmd.extend(['--seed', str(entry['seed'])])

    print('Running:', ' '.join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=str(root), check=True)
PY

