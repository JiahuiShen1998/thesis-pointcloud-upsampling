# Dual-detector three-frame evidence package

Main report:`THESIS_ROOT_CAUSE_REPORT_EN.md`

 Final 3 frame: `000590`, `005625`, `006682`.

## Fast-track entrance.

- 48 case:`case_index.csv`
- All boxes:`analysis/all_selected_frame_boxes.csv`
- All GT transfers:`analysis/selected_frame_gt_transitions.csv`
- BBox/BEV/3D Full AP:`analysis/all_detector_ap_r40_bbox_bev_3d.csv`
- Total transition Statistics:`analysis/transition_summary_full_val.csv`
- Three frame directories:`frames/<frame>/line_<a|b>/<method>/<detector>/`

## Revert

```bash
PY=/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python

$PY scripts/analyze_dual_detector_frame_transitions.py

$PY scripts/generate_dual_detector_three_frame_evidence.py \
  --frames 000590 005625 006682

$PY scripts/summarize_dual_detector_evidence.py
```

The above commands only read the results of existing experiments and write them to this directory;Do not run upsampling, detector training or detector reasoning.

## Non-HTML Interactive 3D

```bash
$PY scripts/view_dual_detector_evidence_open3d.py \
  --frame 005625 --line B --method pdans --detector centerpoint \
  --state upsampled --effective --gt-index 4 --point-size 1.0
```

Get rid of it. `--gt-index` View the full frame;Get rid of it. `--effective` View the full observed/E1 point cloud;`--state overlay` Use the grey observed and the blue generated superimpose.
