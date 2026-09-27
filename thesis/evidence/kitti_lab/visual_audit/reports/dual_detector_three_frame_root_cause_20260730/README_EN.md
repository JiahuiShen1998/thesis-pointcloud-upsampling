# Dual-detector three-frame evidence package

Main report: `THESIS_ROOT_CAUSE_REPORT_EN.md`

Final three frames: `000590`, `005625`, `006682`.

## Quick entry points

- 48 cases: `case_index.csv`
- All boxes: `analysis/all_selected_frame_boxes.csv`
- All per-GT transitions: `analysis/selected_frame_gt_transitions.csv`
- Full BBox/BEV/3D AP: `analysis/all_detector_ap_r40_bbox_bev_3d.csv`
- Full transition statistics: `analysis/transition_summary_full_val.csv`
- Three-frame directory: `frames/<frame>/line_<a|b>/<method>/<detector>/`

> Note for the portable bundle: `analysis/` and `scripts/` are **not** included here — they live on the experiment host. What ships in this bundle is `case_index.csv`, `selected_frames.txt`, the `frames/` tree, and the per-detector `tables/` under `../detector_separated_difficulty_root_cause_20260730/`.

## Reproduction

```bash
PY=/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python

$PY scripts/analyze_dual_detector_frame_transitions.py

$PY scripts/generate_dual_detector_three_frame_evidence.py \
  --frames 000590 005625 006682

$PY scripts/summarize_dual_detector_evidence.py
```

These commands only read existing experiment results and write into this directory. They do not run upsampling, detector training, or detector inference.

## Non-HTML interactive 3D

```bash
$PY scripts/view_dual_detector_evidence_open3d.py \
  --frame 005625 --line B --method pdans --detector centerpoint \
  --state upsampled --effective --gt-index 4 --point-size 1.0
```

Drop `--gt-index` to view the whole frame; drop `--effective` to view the complete observed/E1 cloud; `--state overlay` overlays gray observed points with blue generated ones.

> In the portable bundle, use `viewer/gui_viewer.py` (button-driven GUI) or `viewer/portable_open3d_viewer.py` (same command-line arguments as above) instead of `scripts/view_dual_detector_evidence_open3d.py`.
