#!/usr/bin/env python3
"""Audit old versus local patch geometry for the frozen 20-frame ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LINES = ("line_a_original_x4_up", "line_b_downsampled_x4_up")


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-kind", default="geometry20")
    parser.add_argument("--old-run-kind", default=None)
    parser.add_argument("--old-variant", default="patch_old_spatial_chunk")
    parser.add_argument("--local-variant", default="patch_local_ball_r6_pr3")
    parser.add_argument("--local-variant-line-a", default=None)
    parser.add_argument("--local-variant-line-b", default=None)
    args = parser.parse_args()

    patch_rows: list[dict] = []
    frame_rows: list[dict] = []
    summary_rows: list[dict] = []
    selected_pairs = [(args.old_variant, line) for line in LINES]
    local_by_line = {
        "line_a_original_x4_up": args.local_variant_line_a or args.local_variant,
        "line_b_downsampled_x4_up": args.local_variant_line_b or args.local_variant,
    }
    selected_pairs.extend((local_by_line[line], line) for line in LINES)
    for variant, line in selected_pairs:
            run_kind = (
                args.old_run_kind or args.run_kind
                if variant == args.old_variant
                else args.run_kind
            )
            manifest_dir = args.root / run_kind / variant / line / "manifests"
            xy_values: list[float] = []
            xyz_values: list[float] = []
            unique_values: list[float] = []
            repeat_values: list[float] = []
            coverage_values: list[float] = []
            metas = sorted(manifest_dir.glob("*_patch_metadata.json"))
            if len(metas) != 20:
                raise RuntimeError(f"expected 20 metadata files, found {len(metas)}: {manifest_dir}")
            for meta_path in metas:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                frame = str(meta["frame_id"])
                input_count = int(meta["input_point_count"])
                covered: set[int] = set()
                frame_xy: list[float] = []
                frame_xyz: list[float] = []
                for patch_meta in meta["patches"]:
                    patch_path = Path(meta["output_patch_dir"]) / patch_meta["file"]
                    patch = np.load(patch_path).astype(np.float64)
                    span = patch.max(axis=0) - patch.min(axis=0)
                    xy = float(np.linalg.norm(span[:2]))
                    xyz = float(np.linalg.norm(span))
                    indices = [int(value) for value in patch_meta["original_indices"]]
                    covered.update(indices)
                    unique = int(np.unique(indices).shape[0])
                    repeats = int(len(indices) - unique)
                    frame_xy.append(xy)
                    frame_xyz.append(xyz)
                    xy_values.append(xy)
                    xyz_values.append(xyz)
                    unique_values.append(unique)
                    repeat_values.append(repeats / len(indices))
                    patch_rows.append(
                        {
                            "variant": variant,
                            "line": line,
                            "frame_id": frame,
                            "patch_id": patch_meta["patch_id"],
                            "xy_bbox_diagonal_m": xy,
                            "xyz_bbox_diagonal_m": xyz,
                            "unique_support_points": unique,
                            "repeat_fraction": repeats / len(indices),
                        }
                    )
                coverage = len(covered) / input_count
                coverage_values.append(coverage)
                frame_rows.append(
                    {
                        "variant": variant,
                        "line": line,
                        "frame_id": frame,
                        "patch_count": len(meta["patches"]),
                        "coverage_fraction": coverage,
                        "xy_p50_m": float(np.quantile(frame_xy, 0.5)),
                        "xy_p90_m": float(np.quantile(frame_xy, 0.9)),
                        "xyz_p50_m": float(np.quantile(frame_xyz, 0.5)),
                        "xyz_p90_m": float(np.quantile(frame_xyz, 0.9)),
                    }
                )
            xy_q = quantiles(xy_values)
            xyz_q = quantiles(xyz_values)
            unique_q = quantiles(unique_values)
            repeat_q = quantiles(repeat_values)
            coverage_q = quantiles(coverage_values)
            summary_rows.append(
                {
                    "variant": variant,
                    "line": line,
                    "frames": len(metas),
                    "patches": len(xy_values),
                    "xy_p50_m": xy_q["p50"],
                    "xy_p90_m": xy_q["p90"],
                    "xy_max_m": xy_q["max"],
                    "xyz_p50_m": xyz_q["p50"],
                    "xyz_p90_m": xyz_q["p90"],
                    "unique_support_p10": float(np.quantile(unique_values, 0.1)),
                    "unique_support_p50": unique_q["p50"],
                    "repeat_fraction_p50": repeat_q["p50"],
                    "coverage_p50": coverage_q["p50"],
                    "coverage_min": coverage_q["min"],
                }
            )

    report_dir = args.root / args.run_kind / "reports"
    write_csv(report_dir / "patch_geometry_per_patch.csv", patch_rows)
    write_csv(report_dir / "patch_geometry_per_frame.csv", frame_rows)
    write_csv(report_dir / "patch_geometry_summary.csv", summary_rows)

    by_key = {(row["variant"], row["line"]): row for row in summary_rows}
    markdown = [
        "# Geometry20 Patch Locality Audit",
        "",
        "| Line | Variant | patches | XY p50 m | XY p90 m | coverage p50 | unique support p10 | repeat p50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        markdown.append(
            f"| {row['line']} | {row['variant']} | {row['patches']} | "
            f"{row['xy_p50_m']:.3f} | {row['xy_p90_m']:.3f} | {row['coverage_p50']:.3f} | "
            f"{row['unique_support_p10']:.0f} | {row['repeat_fraction_p50']:.3f} |"
        )
    markdown.extend(["", "## Paired aggregate reductions", ""])
    gates = []
    for line in LINES:
        old = by_key[(args.old_variant, line)]
        local = by_key[(local_by_line[line], line)]
        p50_reduction = 1.0 - local["xy_p50_m"] / old["xy_p50_m"]
        p90_reduction = 1.0 - local["xy_p90_m"] / old["xy_p90_m"]
        gate = bool(
            p50_reduction >= 0.40
            and p90_reduction >= 0.50
            and local["coverage_p50"] >= 0.75
            and local["unique_support_p10"] >= 256
        )
        gates.append(gate)
        markdown.append(
            f"- `{line}`: XY p50 reduction `{p50_reduction:.1%}`, p90 reduction `{p90_reduction:.1%}`, "
            f"geometry gate `{'PASS' if gate else 'FAIL'}`."
        )
    overall = all(gates)
    markdown.extend(
        [
            "",
            f"- Overall geometry gate: `{'PASS' if overall else 'FAIL'}`.",
            "- Gate fixed before detector AP: p50 reduction >=40%, p90 reduction >=50%, "
            "median frame coverage >=75%, unique-support p10 >=256.",
        ]
    )
    (report_dir / "GEOMETRY20_AUDIT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "geometry_gate": overall, "report": str(report_dir / "GEOMETRY20_AUDIT.md")}, sort_keys=True))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
