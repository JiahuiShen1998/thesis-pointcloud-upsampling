#!/usr/bin/env python3
"""Read-only audit for ModelNet40 ×4 two-line protocol data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    EXPECTED_CLASSES,
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    LEGACY_DOWNSAMPLED50_ROOT,
    LEGACY_LINEA_UP_ROOT,
    LEGACY_LINEB_UP_ROOT,
    MESH_ROOT,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    UP_FACTOR,
    detect_original_point_count,
    legacy_method_subdir,
    load_class_to_idx,
    protocol_counts,
    reports_dir,
)

SPLITS = ("train", "test")


def count_npy(root: Path) -> dict[str, int]:
    counts = {s: 0 for s in SPLITS}
    if not root.is_dir():
        return counts
    for split in SPLITS:
        split_dir = root / split
        if split_dir.is_dir():
            counts[split] = sum(1 for _ in split_dir.rglob("*.npy"))
    return counts


def sample_point_stats(root: Path, max_samples: int = 200) -> dict:
    shapes: Counter[tuple[int, int]] = Counter()
    radii: list[float] = []
    nan_count = inf_count = 0
    n = 0
    for npy in sorted(root.rglob("*.npy")):
        if n >= max_samples:
            break
        arr = np.load(npy)
        shapes[(int(arr.shape[0]), int(arr.shape[1]))] += 1
        if not np.isfinite(arr).all():
            if np.isnan(arr).any():
                nan_count += 1
            if np.isinf(arr).any():
                inf_count += 1
        radii.append(float(np.max(np.linalg.norm(arr, axis=1))))
        n += 1
    return {
        "sampled_files": n,
        "shape_histogram": dict(shapes),
        "dominant_shape": shapes.most_common(1)[0][0] if shapes else None,
        "radius_min": min(radii) if radii else None,
        "radius_max": max(radii) if radii else None,
        "nan_files": nan_count,
        "inf_files": inf_count,
    }


def audit_manifest_alignment(original_root: Path, other_root: Path) -> dict:
    issues = []
    matched = 0
    for split in SPLITS:
        orig_manifest = original_root / "metadata" / f"{split}_manifest.csv"
        if not orig_manifest.is_file():
            issues.append(f"missing original manifest {split}")
            continue
        with open(orig_manifest, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rel = Path(split) / row["class_name"] / f"{row['shape_id']}.npy"
                orig = original_root / rel
                other = other_root / rel
                if not orig.is_file():
                    issues.append(f"missing original {rel}")
                elif not other.is_file():
                    issues.append(f"missing counterpart {rel}")
                else:
                    matched += 1
    return {"matched_pairs": matched, "issues_count": len(issues), "sample_issues": issues[:20]}


def audit_downsampled_x4_candidate(root: Path, n: int) -> dict:
    target = n // UP_FACTOR
    counts = count_npy(root)
    stats = sample_point_stats(root)
    dom = stats.get("dominant_shape")
    point_ok = dom == (target, 3) if dom else False
    count_ok = counts["train"] == EXPECTED_TRAIN and counts["test"] == EXPECTED_TEST
    align = audit_manifest_alignment(ORIGINAL_ROOT, root) if root.is_dir() else {"matched_pairs": 0}
    align_ok = align["matched_pairs"] == EXPECTED_TRAIN + EXPECTED_TEST
    valid = count_ok and point_ok and align_ok and stats["nan_files"] == 0 and stats["inf_files"] == 0
    return {
        "path": str(root),
        "exists": root.is_dir(),
        "train_count": counts["train"],
        "test_count": counts["test"],
        "expected_point_count": target,
        "dominant_shape": dom,
        "count_ok": count_ok,
        "point_count_ok": point_ok,
        "alignment_ok": align_ok,
        "valid_for_reuse": valid,
        "stats": stats,
        "alignment": align,
    }


def audit_legacy_upsampling(root: Path, expected_in: int, expected_out: int) -> list[dict]:
    rows = []
    for method in ("ear", "punet", "pugcn", "pdans"):
        sub = root / f"{method}_x4"
        counts = count_npy(sub)
        stats = sample_point_stats(sub) if sub.is_dir() else {"dominant_shape": None}
        dom = stats.get("dominant_shape")
        rows.append(
            {
                "method": method,
                "path": str(sub.relative_to(PROJECT_ROOT)) if sub.is_dir() else str(sub),
                "exists": sub.is_dir(),
                "train_count": counts["train"],
                "test_count": counts["test"],
                "expected_input": expected_in,
                "expected_output": expected_out,
                "dominant_shape": dom,
                "count_ok": counts["train"] == EXPECTED_TRAIN and counts["test"] == EXPECTED_TEST,
                "shape_ok": dom == (expected_out, 3),
                "protocol_note": (
                    "Line A compatible (1024→4096)" if expected_out == 4096
                    else "OLD protocol (512→2048); NOT valid for new Line B (256→1024)"
                ),
            }
        )
    return rows


def audit_mesh_availability() -> dict:
    off_in_mesh_root = sum(1 for _ in MESH_ROOT.rglob("*.off")) if MESH_ROOT.is_dir() else 0
    manifest_off = 0
    manifest_missing = 0
    for split in SPLITS:
        manifest = ORIGINAL_ROOT / "metadata" / f"{split}_manifest.csv"
        if not manifest.is_file():
            continue
        with open(manifest, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                off = Path(row.get("source_off", ""))
                if off.is_file():
                    manifest_off += 1
                else:
                    manifest_missing += 1
    return {
        "mesh_root": str(MESH_ROOT),
        "mesh_root_off_count": off_in_mesh_root,
        "manifest_off_available": manifest_off,
        "manifest_off_missing": manifest_missing,
        "p2f_available": manifest_off > 0 and manifest_missing == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n = detect_original_point_count()
    counts = protocol_counts(n)
    class_to_idx = load_class_to_idx()

    original_counts = count_npy(ORIGINAL_ROOT)
    original_stats = sample_point_stats(ORIGINAL_ROOT, max_samples=500)

    downsampled_x4 = audit_downsampled_x4_candidate(DOWNSAMPLED_X4_ROOT, n)
    legacy_down50 = audit_downsampled_x4_candidate(LEGACY_DOWNSAMPLED50_ROOT, n)
    # relabel legacy down50 with actual point count
    legacy_down50["expected_point_count"] = 512
    legacy_down50["valid_for_reuse"] = False
    legacy_down50["note"] = "Legacy N/2 downsample (512 pts); NOT x4 downsample"

    lineA_legacy = audit_legacy_upsampling(LEGACY_LINEA_UP_ROOT, n, counts["four_N"])
    lineB_legacy = audit_legacy_upsampling(LEGACY_LINEB_UP_ROOT, n // 2, n * 2)

    mesh = audit_mesh_availability()

    rep = reports_dir()
    rep.mkdir(parents=True, exist_ok=True)
    csv_path = rep / "modelnet40_x4_data_audit.csv"
    md_path = rep / "modelnet40_x4_data_audit.md"

    rows = [
        {"section": "protocol", "key": "N", "value": n},
        {"section": "protocol", "key": "N_div_4", "value": counts["N_div_4"]},
        {"section": "protocol", "key": "four_N", "value": counts["four_N"]},
        {"section": "original", "key": "train_count", "value": original_counts["train"]},
        {"section": "original", "key": "test_count", "value": original_counts["test"]},
        {"section": "original", "key": "class_count", "value": len(class_to_idx)},
        {"section": "original", "key": "dominant_shape", "value": original_stats["dominant_shape"]},
        {"section": "downsampled_x4", "key": "path", "value": str(DOWNSAMPLED_X4_ROOT)},
        {"section": "downsampled_x4", "key": "valid_for_reuse", "value": downsampled_x4["valid_for_reuse"]},
        {"section": "downsampled_x4", "key": "train_count", "value": downsampled_x4["train_count"]},
        {"section": "downsampled_x4", "key": "test_count", "value": downsampled_x4["test_count"]},
        {"section": "legacy_downsampled50", "key": "path", "value": str(LEGACY_DOWNSAMPLED50_ROOT)},
        {"section": "legacy_downsampled50", "key": "dominant_shape", "value": legacy_down50["stats"].get("dominant_shape")},
        {"section": "legacy_downsampled50", "key": "note", "value": "N/2=512; old Line B baseline"},
        {"section": "mesh", "key": "p2f_available", "value": mesh["p2f_available"]},
        {"section": "mesh", "key": "manifest_off_available", "value": mesh["manifest_off_available"]},
    ]
    for r in lineA_legacy:
        rows.append({"section": "legacy_lineA_up", "key": r["method"], "value": json.dumps(r)})
    for r in lineB_legacy:
        rows.append({"section": "legacy_lineB_up", "key": r["method"], "value": json.dumps(r)})

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "key", "value"])
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# ModelNet40 ×4 Protocol — Data Audit",
        "",
        f"- Generated at: {ts}",
        f"- PROJECT_ROOT: `{PROJECT_ROOT}`",
        "",
        "## Protocol point counts (detected from data)",
        "",
        f"| Symbol | Value |",
        f"| --- | ---: |",
        f"| N (original baseline) | {n} |",
        f"| N/4 (downsampled ×4) | {counts['N_div_4']} |",
        f"| 4N (Line A upsampling output) | {counts['four_N']} |",
        f"| N (Line B upsampling output) | {n} |",
        "",
        "## Original ModelNet40",
        "",
        f"- Train: {original_counts['train']} (expected {EXPECTED_TRAIN})",
        f"- Test: {original_counts['test']} (expected {EXPECTED_TEST})",
        f"- Classes: {len(class_to_idx)} (expected {EXPECTED_CLASSES})",
        f"- Dominant shape: {original_stats['dominant_shape']}",
        "",
        "## Downsampled ×4 database (`modelnet40_downsampled_x4`)",
        "",
        f"- Path: `{DOWNSAMPLED_X4_ROOT}`",
        f"- Exists: {downsampled_x4['exists']}",
        f"- Valid for reuse: **{downsampled_x4['valid_for_reuse']}**",
        f"- Counts: train={downsampled_x4['train_count']}, test={downsampled_x4['test_count']}",
        f"- Expected point count: {counts['N_div_4']}",
        f"- Dominant shape: {downsampled_x4['stats'].get('dominant_shape')}",
        "",
        "## Legacy `modelnet40_downsampled50` (preserved, NOT x4)",
        "",
        f"- Path: `{LEGACY_DOWNSAMPLED50_ROOT}`",
        f"- Dominant shape: {legacy_down50['stats'].get('dominant_shape')} (N/2, old protocol)",
        f"- **Do not use as x4 downsampled database.**",
        "",
        "## Legacy Line A upsampling outputs (1024→4096)",
        "",
        "| method | exists | train | test | shape | reusable for Line A |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for r in lineA_legacy:
        md_lines.append(
            f"| {r['method']} | {r['exists']} | {r['train_count']} | {r['test_count']} | "
            f"{r['dominant_shape']} | {r['count_ok'] and r['shape_ok']} |"
        )

    md_lines += [
        "",
        "## Legacy Line B upsampling outputs (512→2048, OLD protocol)",
        "",
        "| method | exists | train | test | shape | valid for NEW Line B |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for r in lineB_legacy:
        md_lines.append(
            f"| {r['method']} | {r['exists']} | {r['train_count']} | {r['test_count']} | "
            f"{r['dominant_shape']} | **NO** (need 256→1024) |"
        )

    md_lines += [
        "",
        "## Mesh / P2F availability",
        "",
        f"- Mesh root: `{mesh['mesh_root']}`",
        f"- .off files in mesh root: {mesh['mesh_root_off_count']}",
        f"- Manifest-linked .off available: {mesh['manifest_off_available']}",
        f"- Manifest-linked .off missing: {mesh['manifest_off_missing']}",
        f"- P2F status: **{'available' if mesh['p2f_available'] else 'unavailable'}**",
        "",
        "## Old protocol artifacts (preserved)",
        "",
        "- `downsampled50` (512 pts) — old Line B baseline",
        "- `downsampled50_up/*_x4` (2048 pts) — old Line B upsampling",
        "- `512→2048` PointNet++ results — superseded by new Line B (256→1024)",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"N={n}, N/4={counts['N_div_4']}, 4N={counts['four_N']}")
    print(f"downsampled_x4 valid_for_reuse={downsampled_x4['valid_for_reuse']}")
    print(f"P2F available={mesh['p2f_available']}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
