#!/usr/bin/env python3
"""Select, cap, and precheck PU-GCN cap_100k detector-facing subsets."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_BASE = 20260508
QUOTA = 100000
RPN_NUM_POINTS = 16384
FAR_CAP_FOR_DEFAULT = 16000
METRIC_RANGE = {
    "x_min": 0.0,
    "x_max": 80.0,
    "y_min": -40.0,
    "y_max": 40.0,
    "z_min": -3.5,
    "z_max": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--frame-list", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected frames from start-index onward.")
    parser.add_argument("--base-frame-list", type=Path, default=PROJECT_ROOT / "results/pointrcnn_larger_default_validation_cap100k/shared_20_frame_list.txt")
    parser.add_argument("--validation-split", type=Path, default=PROJECT_ROOT / "data/KITTI/ImageSets/val.txt")
    parser.add_argument("--original-folder", type=Path, default=PROJECT_ROOT / "data/KITTI/object/training/velodyne_original_val")
    parser.add_argument("--ear-folder", type=Path, default=PROJECT_ROOT / "data/KITTI/object/training/velodyne_ear_val")
    parser.add_argument("--punet-folder", type=Path, default=PROJECT_ROOT / "data/KITTI/object/training/velodyne_punet_x2_fullframe")
    parser.add_argument("--smoke-dense-folder", type=Path, default=PROJECT_ROOT / "data/KITTI/object/training/velodyne_pugcn_filter_range_dedup_voxel_003_smoke5")
    parser.add_argument("--existing-dense-folder", action="append", type=Path, default=[PROJECT_ROOT / "results/pointrcnn_larger_default_validation_cap100k/pugcn_dense_filter_range_dedup_voxel_003"])
    parser.add_argument("--new-dense-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, default=PROJECT_ROOT / "data/KITTI/object/training/pugcn_cap_100k")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def load_ids(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def slice_ids(frame_ids: Sequence[str], start_index: int, limit: int) -> List[str]:
    if start_index < 0:
        raise ValueError("start-index must be >= 0")
    sliced = list(frame_ids[start_index:])
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


def load_points(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("%s does not reshape to N x 4" % path)
    return raw.reshape(-1, 4)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def select_frames(args: argparse.Namespace) -> List[str]:
    if args.frame_list is not None:
        return slice_ids(load_ids(args.frame_list), args.start_index, args.limit)
    base_ids = load_ids(args.base_frame_list)
    val_ids = load_ids(args.validation_split)
    selected: List[str] = []
    seen = set()
    for frame_id in base_ids:
        selected.append(frame_id)
        seen.add(frame_id)
    for frame_id in val_ids:
        if frame_id in seen:
            continue
        if all((folder / ("%s.bin" % frame_id)).exists() for folder in (args.original_folder, args.ear_folder, args.punet_folder)):
            selected.append(frame_id)
            seen.add(frame_id)
        if len(selected) >= args.target_count:
            break
    if len(selected) != args.target_count:
        raise RuntimeError("Could only select %d frames, expected %d" % (len(selected), args.target_count))
    return slice_ids(selected, args.start_index, args.limit)


def source_for(args: argparse.Namespace, frame_id: str) -> Tuple[Path, str]:
    candidates = [(args.smoke_dense_folder, "existing_smoke5_dense_filter_range_dedup_voxel_003")]
    candidates.extend((folder, "existing_dense_filter_range_dedup_voxel_003") for folder in args.existing_dense_folder)
    candidates.append((args.new_dense_folder, "new_dense_filter_range_dedup_voxel_003"))
    for folder, kind in candidates:
        path = folder / ("%s.bin" % frame_id)
        if path.exists():
            return path, kind
    raise FileNotFoundError("Missing dense source for %s" % frame_id)


def metric_mask(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3]
    return (
        np.isfinite(points).all(axis=1)
        & (xyz[:, 0] >= METRIC_RANGE["x_min"])
        & (xyz[:, 0] <= METRIC_RANGE["x_max"])
        & (xyz[:, 1] >= METRIC_RANGE["y_min"])
        & (xyz[:, 1] <= METRIC_RANGE["y_max"])
        & (xyz[:, 2] >= METRIC_RANGE["z_min"])
        & (xyz[:, 2] <= METRIC_RANGE["z_max"])
    )


def choose_priority(rng: np.random.Generator, idxs: np.ndarray, original_mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(idxs) == 0:
        return np.empty((0,), dtype=np.int64)
    idxs = np.asarray(idxs, dtype=np.int64)
    original_idxs = idxs[original_mask[idxs]]
    generated_idxs = idxs[~original_mask[idxs]]
    picked = []
    if len(original_idxs):
        take = min(n, len(original_idxs))
        picked.append(rng.choice(original_idxs, size=take, replace=False))
        n -= take
    if n > 0 and len(generated_idxs):
        take = min(n, len(generated_idxs))
        picked.append(rng.choice(generated_idxs, size=take, replace=False))
    if not picked:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(picked).astype(np.int64)


def constrained_default_cap(compat, dense: np.ndarray, original: np.ndarray, frame_id: str, rng: np.random.Generator) -> np.ndarray:
    candidate = dense[metric_mask(dense)]
    if len(candidate) <= QUOTA:
        return candidate.copy()
    original_mask = compat.exact_overlap_mask(candidate, original)
    masks = compat.detector_masks(candidate, frame_id)
    far_idxs = np.where(masks["far"])[0]
    near_idxs = np.where(masks["near"])[0]
    other_idxs = np.where(~masks["far"] & ~masks["near"])[0]
    far_quota = min(FAR_CAP_FOR_DEFAULT, len(far_idxs))
    near_min = max(0, RPN_NUM_POINTS - far_quota)

    selected_far = choose_priority(rng, far_idxs, original_mask, far_quota)
    selected_near = choose_priority(rng, near_idxs, original_mask, near_min)
    selected_set = set(int(x) for x in np.concatenate([selected_far, selected_near]))
    remaining_quota = QUOTA - len(selected_set)
    for group in (other_idxs, near_idxs):
        rest = np.array([int(x) for x in group if int(x) not in selected_set], dtype=np.int64)
        take = choose_priority(rng, rest, original_mask, remaining_quota)
        selected_set.update(int(x) for x in take)
        remaining_quota = QUOTA - len(selected_set)
        if remaining_quota <= 0:
            break
    if remaining_quota > 0:
        rest = np.array([int(x) for x in far_idxs if int(x) not in selected_set], dtype=np.int64)
        take = choose_priority(rng, rest, original_mask, remaining_quota)
        selected_set.update(int(x) for x in take)
    keep = np.array(sorted(selected_set), dtype=np.int64)
    if len(keep) > QUOTA:
        keep = rng.choice(keep, size=QUOTA, replace=False)
    rng.shuffle(keep)
    return candidate[keep].copy()


def generation_runtime(args: argparse.Namespace, frame_id: str) -> object:
    manifest = args.result_root / "pugcn_dense_generation" / frame_id / "batch_manifest.json"
    if manifest.exists():
        data = read_json(manifest)
        return data.get("total_runtime_sec") or data.get("refresh_runtime_sec") or ""
    raw_manifest = args.result_root / "pugcn_dense_generation" / frame_id / "raw_100" / "work" / "manifest.json"
    if raw_manifest.exists():
        return read_json(raw_manifest).get("total_runtime_sec", "")
    old_manifest = PROJECT_ROOT / "results/pointrcnn_larger_default_validation_cap100k/pugcn_dense_generation" / frame_id / "batch_manifest.json"
    if old_manifest.exists():
        data = read_json(old_manifest)
        return data.get("total_runtime_sec") or data.get("refresh_runtime_sec") or ""
    return ""


def batch_warning(args: argparse.Namespace, frame_id: str) -> str:
    failure = args.result_root / "pugcn_dense_generation" / frame_id / "batch_failure.json"
    if failure.exists():
        data = read_json(failure)
        return "Dense output exists, but batch wrapper recorded failure: %s" % data.get("error", "unknown")
    old_failure = PROJECT_ROOT / "results/pointrcnn_larger_default_validation_cap100k/pugcn_dense_generation" / frame_id / "batch_failure.json"
    if old_failure.exists():
        data = read_json(old_failure)
        return "Dense output exists, but earlier batch wrapper recorded failure: %s" % data.get("error", "unknown")
    return ""


def precheck_row(compat, args: argparse.Namespace, frame_id: str, capped: np.ndarray, original: np.ndarray, output_path: Path, source_path: Path, source_kind: str, action: str, dense_count: int, warning: str) -> Dict[str, object]:
    masks = compat.detector_masks(capped, frame_id)
    valid = int(masks["valid"].sum())
    near = int(masks["near"].sum())
    far = int(masks["far"].sum())
    near_request = RPN_NUM_POINTS - far if valid > RPN_NUM_POINTS else "N/A"
    likely = valid >= RPN_NUM_POINTS and far <= RPN_NUM_POINTS and (near_request == "N/A" or near >= near_request)
    xyz = capped[:, :3]
    return {
        "frame_id": frame_id,
        "output_path": str(output_path),
        "source_dense_path": str(source_path),
        "source_kind": source_kind,
        "action": action,
        "input_original_points": int(len(original)),
        "dense_source_points": int(dense_count),
        "output_points": int(len(capped)),
        "dtype": "float32",
        "shape": "%dx4" % len(capped),
        "x_min": float(xyz[:, 0].min()),
        "x_max": float(xyz[:, 0].max()),
        "y_min": float(xyz[:, 1].min()),
        "y_max": float(xyz[:, 1].max()),
        "z_min": float(xyz[:, 2].min()),
        "z_max": float(xyz[:, 2].max()),
        "intensity_min": float(capped[:, 3].min()),
        "intensity_max": float(capped[:, 3].max()),
        "nan_points": int(np.isnan(capped).any(axis=1).sum()),
        "inf_points": int(np.isinf(capped).any(axis=1).sum()),
        "invalid_points": int((~np.isfinite(capped).all(axis=1)).sum()),
        "out_of_range_points": int(compat.metric_out_of_range(capped)),
        "detector_valid_points": valid,
        "detector_near_points_depth_lt_40": near,
        "detector_far_points_depth_ge_40": far,
        "default_sampler_near_request": near_request,
        "likely_pass_default_sampling": bool(likely),
        "near_duplicate_ratio_voxel_003": float(compat.near_duplicate_ratio(capped, 0.03)),
        "generation_runtime_sec": generation_runtime(args, frame_id),
        "sampling_seed": SEED_BASE + int(frame_id) + QUOTA,
        "warning": warning,
    }


def write_markdown(args: argparse.Namespace, frame_ids: Sequence[str], rows: Sequence[Dict[str, object]], failed: Sequence[Tuple[str, str]], precheck_rows: Sequence[Dict[str, object]]) -> None:
    reused = [row["frame_id"] for row in rows if row["action"] == "verified_existing"]
    generated = [row["frame_id"] for row in rows if row["action"] == "generated_new"]
    regenerated = [row["frame_id"] for row in rows if row["action"] == "regenerated_default_compatible_cap_100k"]
    warnings = [row for row in rows if row["warning"]]
    expected_count = len(frame_ids)
    ready = (
        len(precheck_rows) == expected_count
        and not failed
        and all(row["likely_pass_default_sampling"] for row in precheck_rows)
        and all(row["invalid_points"] == 0 and row["out_of_range_points"] == 0 for row in precheck_rows)
    )
    failed_text = "none" if not failed else "; ".join("%s: %s" % item for item in failed)
    subset_name = "full-validation" if args.frame_list is not None else ("%d-frame" % args.target_count)
    stem = "pugcn_cap100k_full_validation" if args.frame_list is not None else ("pugcn_cap100k_%dframe" % args.target_count)
    summary_name = "%s_generation_summary.md" % stem
    precheck_csv_name = "%s_input_precheck.csv" % stem
    precheck_md_name = "%s_input_precheck.md" % stem
    missing_name = "missing_full_validation_pugcn_frames.txt" if args.frame_list is not None else ("shared_%d_missing_pugcn_frames.txt" % args.target_count)
    frame_list_name = "full_validation_frame_list.txt" if args.frame_list is not None else ("shared_%d_frame_list.txt" % args.target_count)
    lines = [
        "# PU-GCN cap_100k %s Generation Summary" % subset_name.title(),
        "",
        "## Selected Frames",
        "",
        "```text",
        "\n".join(frame_ids),
        "```",
        "",
        "## Policy",
        "",
        "- Dense source: selected PU-GCN `filter_range_dedup_voxel_003` full-frame output.",
        "- Detector-facing normalization: deterministic `pugcn_cap_100k` cap.",
        "- Quota: at most `100000` points per frame.",
        "- Seed formula: `%d + int(frame_id) + 100000`." % SEED_BASE,
        "- Sampling policy: preserve exact original-overlap rows first when present, then sample generated/non-overlap rows without replacement.",
        "- Default-compatibility guard: regenerate a cap if invalid, out-of-range, or likely to fail default PointRCNN sampling.",
        "- Point format: KITTI-compatible `N x 4 float32 [x, y, z, intensity]`.",
        "- PointRCNN inference and AP evaluation were not run.",
        "",
        "## Generation Status",
        "",
        "- reused existing cap_100k frames: %d (%s)" % (len(reused), ", ".join(reused)),
        "- newly generated cap_100k frames: %d (%s)" % (len(generated), ", ".join(generated)),
        "- regenerated by default-compatibility guard: %d (%s)" % (len(regenerated), ", ".join(regenerated) if regenerated else "none"),
        "- failed frames: %s" % failed_text,
        "- warnings: %d" % len(warnings),
        "",
        "| Frame | Action | Original points | Dense source points | cap_100k points | Runtime sec | Warning |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| `%s` | %s | %s | %s | %s | %s | %s |"
            % (
                row["frame_id"],
                row["action"],
                row["input_original_points"],
                row["dense_source_points"],
                row["output_points"],
                row["generation_runtime_sec"],
                row["warning"],
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            (
                "The `%s` `pugcn_cap_100k` detector-facing set is ready for default PointRCNN inference and AP evaluation."
                % subset_name
                if ready
                else "The `%s` set was prepared, but review failures or precheck warnings before detector use." % subset_name
            ),
            "",
            "## Output Files",
            "",
            "- Detector-facing folder: `%s/`" % args.output_folder,
            "- Frame list: `%s`" % (args.result_root / frame_list_name),
            "- Missing-frame list: `%s`" % (args.result_root / missing_name),
            "- Input precheck CSV: `%s`" % (args.result_root / precheck_csv_name),
            "- Input precheck Markdown: `%s`" % (args.result_root / precheck_md_name),
        ]
    )
    (args.result_root / summary_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_precheck_md(args: argparse.Namespace, precheck_rows: Sequence[Dict[str, object]], failed: Sequence[Tuple[str, str]]) -> None:
    subset_name = "full-validation" if args.frame_list is not None else ("%d-frame" % args.target_count)
    stem = "pugcn_cap100k_full_validation" if args.frame_list is not None else ("pugcn_cap100k_%dframe" % args.target_count)
    lines = [
        "# PU-GCN cap_100k %s Input Precheck" % subset_name.title(),
        "",
        "Frames checked: %d" % len(precheck_rows),
        "Failed precheck rows: %d" % len(failed),
        "",
        "| Frame | Points | Valid | Near | Far | Near request | Likely default sampler pass | Invalid | Out-of-range | Near-dup 0.03 | Warning |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in precheck_rows:
        lines.append(
            "| {frame_id} | {output_points} | {detector_valid_points} | {detector_near_points_depth_lt_40} | "
            "{detector_far_points_depth_ge_40} | {default_sampler_near_request} | {likely_pass_default_sampling} | "
            "{invalid_points} | {out_of_range_points} | {near_duplicate_ratio_voxel_003:.6f} | {warning} |".format(**row)
        )
    (args.result_root / ("%s_input_precheck.md" % stem)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    frame_ids = select_frames(args)
    stem = "pugcn_cap100k_full_validation" if args.frame_list is not None else ("pugcn_cap100k_%dframe" % args.target_count)
    frame_list_name = "full_validation_frame_list.txt" if args.frame_list is not None else ("shared_%d_frame_list.txt" % args.target_count)
    missing_name = "missing_full_validation_pugcn_frames.txt" if args.frame_list is not None else ("shared_%d_missing_pugcn_frames.txt" % args.target_count)
    (args.result_root / frame_list_name).write_text("\n".join(frame_ids) + "\n", encoding="utf-8")
    missing = [frame_id for frame_id in frame_ids if not (args.output_folder / ("%s.bin" % frame_id)).exists()]
    (args.result_root / missing_name).write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    if args.list_only:
        print("selected", len(frame_ids))
        print("missing", len(missing))
        return

    spec = importlib.util.spec_from_file_location("compat", PROJECT_ROOT / "scripts/prepare_pugcn_default_compat_variants.py")
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)

    rows: List[Dict[str, object]] = []
    precheck_rows: List[Dict[str, object]] = []
    failed: List[Tuple[str, str]] = []
    for frame_id in frame_ids:
        output_path = args.output_folder / ("%s.bin" % frame_id)
        warning = batch_warning(args, frame_id)
        try:
            source_path, source_kind = source_for(args, frame_id)
            original = load_points(args.original_folder / ("%s.bin" % frame_id))
            dense = load_points(source_path)
            if output_path.exists():
                capped = load_points(output_path)
                action = "verified_existing"
            else:
                rng = np.random.default_rng(SEED_BASE + int(frame_id) + QUOTA)
                capped = compat.random_cap(dense, original, QUOTA, rng)
                capped.astype(np.float32, copy=False).tofile(output_path)
                action = "generated_new"
            pre_row = precheck_row(compat, args, frame_id, capped, original, output_path, source_path, source_kind, action, len(dense), warning)
            if (
                not pre_row["likely_pass_default_sampling"]
                or pre_row["invalid_points"] > 0
                or pre_row["out_of_range_points"] > 0
            ):
                rng = np.random.default_rng(SEED_BASE + int(frame_id) + QUOTA + 17)
                capped = constrained_default_cap(compat, dense, original, frame_id, rng)
                capped.astype(np.float32, copy=False).tofile(output_path)
                action = "regenerated_default_compatible_cap_100k"
                pre_row = precheck_row(compat, args, frame_id, capped, original, output_path, source_path, source_kind, action, len(dense), warning)
            precheck_rows.append(pre_row)
            rows.append(
                {
                    "frame_id": frame_id,
                    "action": action,
                    "input_original_points": pre_row["input_original_points"],
                    "dense_source_points": pre_row["dense_source_points"],
                    "output_points": pre_row["output_points"],
                    "generation_runtime_sec": pre_row["generation_runtime_sec"],
                    "source_dense_path": str(source_path),
                    "output_path": str(output_path),
                    "warning": warning,
                }
            )
        except Exception as exc:
            failed.append((frame_id, str(exc)))
    write_csv(args.result_root / ("%s_input_precheck.csv" % stem), precheck_rows)
    write_csv(args.result_root / ("%s_generation_rows.csv" % stem), rows)
    write_precheck_md(args, precheck_rows, failed)
    write_markdown(args, frame_ids, rows, failed, precheck_rows)
    ready = len(precheck_rows) == len(frame_ids) and not failed and all(row["likely_pass_default_sampling"] for row in precheck_rows)
    print("selected", len(frame_ids))
    print("missing_before", len(missing))
    print("checked", len(precheck_rows))
    print("failed", "none" if not failed else failed)
    print("ready", ready)


if __name__ == "__main__":
    main()
