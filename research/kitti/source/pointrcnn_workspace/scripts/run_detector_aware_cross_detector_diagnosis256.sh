#!/usr/bin/env bash
set -euo pipefail

repo="/home/ra87racy/projects/baseline_detectors/PointRCNN"
python_bin="/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python"

cd "$repo"
echo "stage=transitions status=started utc=$(date -u +%FT%TZ)"
"$python_bin" -u scripts/analyze_detector_aware_transitions256.py
echo "stage=transitions status=completed utc=$(date -u +%FT%TZ)"
echo "stage=voxels status=started utc=$(date -u +%FT%TZ)"
"$python_bin" -u scripts/analyze_detector_aware_centerpoint_voxels256.py
echo "stage=voxels status=completed utc=$(date -u +%FT%TZ)"
echo "pipeline_status=completed utc=$(date -u +%FT%TZ)"
