#!/usr/bin/env python3
"""Audit GT object/core intersections after label-free split boundaries are frozen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


CLASSES = ("Car", "Pedestrian", "Cyclist")


def read_regions(path: Path) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {
                "slot": int(row["slot"]),
                "x_min": float(row["x_min_rect_m"]),
                "x_max": float(row["x_max_rect_m"]),
                "z_min": float(row["z_min_rect_m"]),
                "z_max": float(row["z_max_rect_m"]),
                "halo_m": float(row.get("halo_m", 0.0) or 0.0),
            }
            output.setdefault(row["frame_id"], []).append(parsed)
    return output


def corners_bev(fields: list[str]) -> np.ndarray:
    width = float(fields[9])
    length = float(fields[10])
    center_x = float(fields[11])
    center_z = float(fields[13])
    yaw = float(fields[14])
    local = np.asarray(
        [
            [length * 0.5, width * 0.5],
            [-length * 0.5, width * 0.5],
            [-length * 0.5, -width * 0.5],
            [length * 0.5, -width * 0.5],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return local @ rotation.T + np.asarray([center_x, center_z])


def contains(region: dict, point: np.ndarray, margin: float = 0.0) -> bool:
    return (
        region["x_min"] - margin <= point[0] < region["x_max"] + margin
        and region["z_min"] - margin <= point[1] < region["z_max"] + margin
    )


def audit_workspace(workspace: Path, label_dir: Path, selected_frames: set[str] | None) -> dict:
    regions = read_regions(workspace / "manifests/regions.csv")
    rows: list[dict] = []
    for frame, frame_regions in sorted(regions.items()):
        if selected_frames is not None and frame not in selected_frames:
            continue
        label_path = label_dir / f"{frame}.txt"
        if not label_path.is_file():
            continue
        for label_index, line in enumerate(label_path.read_text().splitlines()):
            fields = line.split()
            if len(fields) < 15 or fields[0] not in CLASSES:
                continue
            center = np.asarray([float(fields[11]), float(fields[13])], dtype=np.float64)
            owners = [region for region in frame_regions if contains(region, center)]
            if len(owners) != 1:
                continue
            owner = owners[0]
            corners = corners_bev(fields)
            core_complete = all(contains(owner, corner) for corner in corners)
            halo_complete = all(
                contains(owner, corner, owner["halo_m"]) for corner in corners
            )
            rows.append(
                {
                    "frame_id": frame,
                    "label_index": label_index,
                    "class": fields[0],
                    "slot": owner["slot"],
                    "core_complete": core_complete,
                    "core_boundary_crossed": not core_complete,
                    "halo_complete": halo_complete,
                    "halo_m": owner["halo_m"],
                }
            )

    reports = workspace / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "object_boundary_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frame_id"])
        writer.writeheader()
        writer.writerows(rows)

    def aggregate(group: list[dict]) -> dict:
        total = len(group)
        crossed = sum(bool(row["core_boundary_crossed"]) for row in group)
        halo_complete = sum(bool(row["halo_complete"]) for row in group)
        crossed_halo_complete = sum(
            bool(row["core_boundary_crossed"] and row["halo_complete"]) for row in group
        )
        return {
            "objects": total,
            "core_boundary_crossed": crossed,
            "core_boundary_crossed_fraction": crossed / total if total else None,
            "halo_complete": halo_complete,
            "halo_complete_fraction": halo_complete / total if total else None,
            "crossed_but_complete_in_halo": crossed_halo_complete,
            "crossed_but_complete_in_halo_fraction": (
                crossed_halo_complete / crossed if crossed else None
            ),
        }

    summary = {
        "workspace": str(workspace),
        "boundaries_were_frozen_without_gt": True,
        "gt_usage": "post-hoc audit only",
        "all": aggregate(rows),
        "by_class": {
            name: aggregate([row for row in rows if row["class"] == name])
            for name in CLASSES
        },
        "status": "PASS",
    }
    (reports / "object_boundary_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, action="append", required=True)
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("data/KITTI/object/training/label_2"),
    )
    parser.add_argument("--split-file", type=Path)
    args = parser.parse_args()
    selected = None
    if args.split_file:
        selected = {line.strip() for line in args.split_file.read_text().splitlines() if line.strip()}
    for workspace in args.workspace:
        summary = audit_workspace(workspace.resolve(), args.label_dir.resolve(), selected)
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
