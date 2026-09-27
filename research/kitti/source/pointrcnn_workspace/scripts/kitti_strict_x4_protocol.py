#!/usr/bin/env python3
"""KITTI strict x4 upsampling protocol preparation and audit.

This script intentionally separates detector-input point-budget checks from
detector-internal sampling/voxelization. It does not start PointRCNN,
CenterPoint, KITTI AP evaluation, detector inference, or detector training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
RESULT_ROOT = PROJECT_ROOT / "results" / "kitti_strict_x4_upsampling_protocol"
KITTI_TRAINING = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
VAL_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
ORIGINAL_INPUT = KITTI_TRAINING / "velodyne_original_val"
DOWNSAMPLED50_INPUT = KITTI_TRAINING / "velodyne_downsampled_50_val"

METHODS = ["EAR", "PU-Net", "PU-GCN", "PDANS", "PU-EdgeFormer"]
PREFERRED_SMOKE = ["000001", "000093", "000242", "003219", "006833", "007458"]

METHOD_AUDIT = {
    "EAR": {
        "code_dir": "/home/ra87racy/projects/upsampling/EAR",
        "entry": "/home/ra87racy/projects/upsampling/EAR/ear_upsampling.py",
        "checkpoint": "none",
        "input_format": "KITTI .bin N x 4 float32",
        "output_format": "KITTI .bin N x 4 float32",
        "supports_x4": "yes via --up_factor 4.0 and target_n in helper",
        "patch_based": "no",
        "full_frame_strategy": "full-frame PCA/edge-aware insertion; may be slow for full KITTI frames",
        "adapter_needed": "strict wrapper recommended for split/frame control and final 4N audit",
        "intensity_policy": "native EAR midpoint intensity; strict protocol will audit as xyzi",
        "risk": "old results were not clean x4; full-frame EAR can be expensive",
    },
    "PU-Net": {
        "code_dir": "/home/ra87racy/projects/upsampling/PU-Net",
        "entry": "scripts/punet_kitti_adapter.py",
        "checkpoint": "/home/ra87racy/projects/upsampling/PU-Net/model/generator2_new6",
        "input_format": "patch xyz text from KITTI .bin",
        "output_format": "patch xyz text merged to KITTI .bin",
        "supports_x4": "official tester supports --up_ratio 4 if checkpoint/log_dir supports it",
        "patch_based": "yes",
        "full_frame_strategy": "BEV patches, official PU-Net inference, merge, then final strict 4N normalization",
        "adapter_needed": "yes; old adapter defaults to x2 and voxel fusion can change point count",
        "intensity_policy": "nearest-neighbor intensity from source/input frame",
        "risk": "old results are not clean x4; checkpoint/runtime must be verified before smoke",
    },
    "PU-GCN": {
        "code_dir": "external/PU-GCN",
        "entry": "scripts/pugcn_kitti_full_frame_reconstruct.py",
        "checkpoint": "external/PU-GCN/pretrained/pu1k-pugcn/model-100",
        "input_format": "patch xyz text from KITTI .bin",
        "output_format": "patch xyz text merged to KITTI .bin",
        "supports_x4": "yes via --upsample_ratio 4",
        "patch_based": "yes",
        "full_frame_strategy": "KNN patches, official PU-GCN inference, merge, then final strict 4N normalization",
        "adapter_needed": "yes; set --max_output_points 0, avoid cap_100k, then normalize to 4N",
        "intensity_policy": "nearest-neighbor intensity from source/input frame",
        "risk": "old results used cap_100k/post-processing and cannot be reused",
    },
    "PDANS": {
        "code_dir": "external/PDANS",
        "entry": "scripts/pdans_kitti_adapter.py",
        "checkpoint": "external/PDANS/checkpoints/PUGAN_PDANS.pkl or PU1K_PDANS.pkl",
        "input_format": "KITTI .bin N x 4 float32 to patch xyz tensors",
        "output_format": "KITTI .bin N x 4 float32",
        "supports_x4": "yes at diffusion sampling label/up_ratio level if checkpoint available",
        "patch_based": "yes",
        "full_frame_strategy": "patch diffusion generation, merge, nearest intensity, final strict 4N normalization",
        "adapter_needed": "yes; disable 100k cap and original+generated policy must be normalized",
        "intensity_policy": "nearest-neighbor intensity from source/input frame",
        "risk": "old results include original + generated + 100k cap and cannot be reused",
    },
    "PU-EdgeFormer": {
        "code_dir": "external/reproducibility_check/puedgeformer_minimal_feasibility/PU-EdgeFormer_ops_reuse",
        "entry": "external/reproducibility_check/puedgeformer_minimal_feasibility/PU-EdgeFormer_ops_reuse/main.py --model edgetransformer --phase test",
        "checkpoint": "external/reproducibility_check/puedgeformer_minimal_feasibility/PU-EdgeFormer_ops_reuse/log/pu-edgeformer_full_pu1k_20260625_2314/.../model-100",
        "input_format": "patch xyz text; PU1K model expects 2048-point patches",
        "output_format": "patch xyz text; 2048 -> 8192 x3",
        "supports_x4": "yes, verified on PU1K input_2048",
        "patch_based": "yes",
        "full_frame_strategy": "KITTI patch extraction, model-100 inference, merge, final strict 4N normalization",
        "adapter_needed": "yes; KITTI full-frame wrapper still required",
        "intensity_policy": "nearest-neighbor intensity from source/input frame",
        "risk": "CD/Hausdorff verified; EMD_NOT_VERIFIED; KITTI smoke still required",
    },
}


def read_split(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path} does not reshape to N x 4 float32")
    return raw.reshape(-1, 4)


def point_count_from_size(path: Path) -> int:
    if not path.exists():
        return 0
    size = path.stat().st_size
    if size % 16 != 0:
        return 0
    return size // 16


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def shape_stats(points: np.ndarray, duplicate_check: bool = False) -> Dict[str, object]:
    xyz = points[:, :3]
    intensity = points[:, 3]
    finite = np.isfinite(points).all()
    row = {
        "points": int(len(points)),
        "shape": f"{len(points)}x{points.shape[1]}",
        "finite": bool(finite),
        "nan_inf": bool(not finite),
        "x_min": float(np.min(xyz[:, 0])) if len(points) else "",
        "x_max": float(np.max(xyz[:, 0])) if len(points) else "",
        "y_min": float(np.min(xyz[:, 1])) if len(points) else "",
        "y_max": float(np.max(xyz[:, 1])) if len(points) else "",
        "z_min": float(np.min(xyz[:, 2])) if len(points) else "",
        "z_max": float(np.max(xyz[:, 2])) if len(points) else "",
        "intensity_min": float(np.min(intensity)) if len(points) else "",
        "intensity_max": float(np.max(intensity)) if len(points) else "",
        "intensity_mean": float(np.mean(intensity)) if len(points) else "",
    }
    if duplicate_check:
        rounded_xyz = np.round(xyz, decimals=4)
        unique = len(np.unique(rounded_xyz, axis=0)) if len(points) else 0
        row["unique_xyz_rounded4"] = int(unique)
        row["duplicate_xyz_rounded4_ratio"] = float(1.0 - unique / len(points)) if len(points) else ""
    return row


def summarize_input(name: str, folder: Path, frame_ids: Sequence[str]) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    counts: List[int] = []
    missing: List[str] = []
    malformed: List[str] = []
    for frame_id in frame_ids:
        path = folder / f"{frame_id}.bin"
        if not path.exists():
            missing.append(frame_id)
            continue
        size = path.stat().st_size
        if size % 16 != 0:
            malformed.append(frame_id)
            continue
        points = size // 16
        counts.append(points)
        row = {
            "line": name,
            "frame": frame_id,
            "path": str(path),
            "file_size_mod16": size % 16,
            "points": int(points),
            "shape": f"{points}x4",
            "finite": "not_read_input_fast_audit",
            "nan_inf": "not_read_input_fast_audit",
        }
        rows.append(row)
    summary = {
        "line": name,
        "input_dir": str(folder),
        "val_split_frames": len(frame_ids),
        "bin_files_in_dir": len(list(folder.glob("*.bin"))),
        "matched_val_files": len(rows),
        "missing_count": len(missing),
        "missing_first10": ",".join(missing[:10]) if missing else "none",
        "malformed_count": len(malformed),
        "malformed_first10": ",".join(malformed[:10]) if malformed else "none",
        "min_points": min(counts) if counts else "",
        "max_points": max(counts) if counts else "",
        "mean_points": round(mean(counts), 3) if counts else "",
    }
    return summary, rows


def choose_smoke_frames(frame_ids: Sequence[str]) -> tuple[List[str], List[str]]:
    available = set(frame_ids)
    available_preferred = [fid for fid in PREFERRED_SMOKE if fid in available]
    chosen = available_preferred[:5]
    replacements: List[str] = []
    for fid in frame_ids:
        if len(chosen) >= 5:
            break
        if fid not in chosen:
            chosen.append(fid)
            replacements.append(fid)
    if len(chosen) > 5:
        extra = chosen[5:]
        chosen = chosen[:5]
        replacements.extend(extra)
    missing_preferred = [fid for fid in PREFERRED_SMOKE if fid not in available]
    reasons = [f"{fid}: not in val split" for fid in missing_preferred]
    unused_preferred = available_preferred[5:]
    if unused_preferred:
        reasons.append(f"unused preferred frames because smoke is fixed to 5 frames: {','.join(unused_preferred)}")
    if replacements:
        reasons.append(f"replacement frames selected from val split: {','.join(replacements)}")
    return chosen, reasons


def ensure_dirs() -> None:
    (RESULT_ROOT / "lineA_original_baseline_native").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "lineB_downsampled50_baseline_native").mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        (RESULT_ROOT / "lineA_original_up_x4" / method).mkdir(parents=True, exist_ok=True)
        (RESULT_ROOT / "lineB_downsampled50_up_x4" / method).mkdir(parents=True, exist_ok=True)


def write_baseline_manifests(frame_ids: Sequence[str]) -> None:
    manifests = [
        ("lineA_original_baseline_native", ORIGINAL_INPUT),
        ("lineB_downsampled50_baseline_native", DOWNSAMPLED50_INPUT),
    ]
    for line_name, src in manifests:
        rows = []
        for frame_id in frame_ids:
            source = src / f"{frame_id}.bin"
            rows.append(
                {
                    "line": line_name,
                    "frame": frame_id,
                    "source_path": str(source),
                    "exists": source.exists(),
                    "detector_input_policy": "baseline_native_no_point_count_matching_to_upsampling",
                }
            )
        write_csv(RESULT_ROOT / line_name / "manifest.csv", rows)


def existing_reports_rows() -> List[Dict[str, object]]:
    names = [
        "upsampling_ratio_audit.md",
        "upsampling_ratio_audit.csv",
        "upsampling_ratio_sample_stats.csv",
        "pu_edgeformer_pu1k_sanity_gate_result.md",
        "pu_edgeformer_pu1k_eval_report.md",
        "pu_edgeformer_model100_restore_report.md",
    ]
    return [{"report": f"reports/{name}", "exists": (REPORT_DIR / name).exists()} for name in names]


def method_runtime_rows() -> List[Dict[str, object]]:
    rows = []
    for method, info in METHOD_AUDIT.items():
        row = {"method": method, **info}
        entry = info["entry"]
        ckpt = info["checkpoint"]
        entry_path = Path(entry.split()[0])
        row["entry_exists"] = entry_path.exists() if entry_path.is_absolute() else (PROJECT_ROOT / entry_path).exists()
        if ckpt == "none":
            row["checkpoint_status"] = "not_required"
        elif " or " in ckpt or "..." in ckpt:
            row["checkpoint_status"] = "see_report_or_multiple_candidates"
        else:
            ckpt_path = PROJECT_ROOT / ckpt
            if ckpt_path.exists():
                row["checkpoint_status"] = "exists"
            elif any(PROJECT_ROOT.glob(f"{ckpt}.*")):
                row["checkpoint_status"] = "exists_prefix"
            else:
                row["checkpoint_status"] = "missing"
        rows.append(row)
    return rows


def audit_outputs(frame_ids: Sequence[str], smoke_only: bool) -> List[Dict[str, object]]:
    ids = list(frame_ids)
    if smoke_only:
        ids, _ = choose_smoke_frames(ids)
    rows: List[Dict[str, object]] = []
    line_defs = [
        ("Line A", ORIGINAL_INPUT, RESULT_ROOT / "lineA_original_up_x4"),
        ("Line B", DOWNSAMPLED50_INPUT, RESULT_ROOT / "lineB_downsampled50_up_x4"),
    ]
    for line, input_dir, root in line_defs:
        for method in METHODS:
            out_dir = root / method
            for frame_id in ids:
                input_path = input_dir / f"{frame_id}.bin"
                output_path = out_dir / f"{frame_id}.bin"
                input_points = point_count_from_size(input_path)
                target = input_points * 4
                row: Dict[str, object] = {
                    "Line": line,
                    "Method": method,
                    "Frame": frame_id,
                    "Input Points": input_points,
                    "Target Output Points": target,
                    "output_path": str(output_path),
                    "exists": output_path.exists(),
                    "Intensity Policy": "nearest-neighbor input intensity for xyz-only model outputs; native intensity audited if method outputs xyzi",
                }
                if not output_path.exists():
                    row.update({"Actual Output Points": "", "Ratio": "", "NaN/Inf": "", "Status": "NOT_RUN"})
                    rows.append(row)
                    continue
                if output_path.stat().st_size % 16 != 0:
                    row.update({"Actual Output Points": "", "Ratio": "", "NaN/Inf": "", "Status": "FAIL_SIZE_NOT_DIVISIBLE_BY_16"})
                    rows.append(row)
                    continue
                out = read_bin(output_path)
                stats = shape_stats(out, duplicate_check=smoke_only)
                row.update(stats)
                row["Actual Output Points"] = len(out)
                row["Ratio"] = round(len(out) / input_points, 8) if input_points else ""
                row["NaN/Inf"] = bool(not np.isfinite(out).all())
                row["Status"] = "PASS" if len(out) == target and np.isfinite(out).all() else "FAIL"
                rows.append(row)
    return rows


def estimate_disk(rows: Sequence[Dict[str, object]]) -> str:
    total_bytes = 0
    for row in rows:
        pts = row.get("points") or row.get("Input Points") or 0
        try:
            total_bytes += int(pts) * 4 * 4
        except Exception:
            pass
    gb = total_bytes * 4 * len(METHODS) / (1024**3)
    return f"approximately {gb:.2f} GiB for all strict x4 upsampling outputs before method logs/work dirs"


def write_reports(args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_dirs()
    frame_ids = read_split(VAL_SPLIT)
    write_baseline_manifests(frame_ids)

    original_summary, original_rows = summarize_input("Line A Original", ORIGINAL_INPUT, frame_ids)
    down_summary, down_rows = summarize_input("Line B Downsampled50", DOWNSAMPLED50_INPUT, frame_ids)
    frame_ids_original = {Path(r["path"]).stem for r in original_rows}
    frame_ids_down = {Path(r["path"]).stem for r in down_rows}
    same_ids = frame_ids_original == frame_ids_down == set(frame_ids)
    smoke_frames, smoke_reasons = choose_smoke_frames(frame_ids)

    write_csv(REPORT_DIR / "kitti_strict_x4_protocol_input_sample_stats.csv", original_rows + down_rows)
    write_csv(REPORT_DIR / "kitti_strict_x4_protocol_existing_reports.csv", existing_reports_rows())
    write_csv(REPORT_DIR / "kitti_strict_x4_protocol_method_audit.csv", method_runtime_rows())

    smoke_rows = audit_outputs(frame_ids, smoke_only=True)
    write_csv(REPORT_DIR / "kitti_strict_x4_protocol_smoke_stats.csv", smoke_rows)
    full_audit_rows = audit_outputs(frame_ids, smoke_only=False)
    write_csv(REPORT_DIR / "kitti_strict_x4_protocol_ratio_audit.csv", full_audit_rows)

    manifest_rows = []
    for line, input_dir, out_root in [
        ("Line A", ORIGINAL_INPUT, RESULT_ROOT / "lineA_original_up_x4"),
        ("Line B", DOWNSAMPLED50_INPUT, RESULT_ROOT / "lineB_downsampled50_up_x4"),
    ]:
        for method in METHODS:
            for fid in frame_ids:
                src = input_dir / f"{fid}.bin"
                out = out_root / method / f"{fid}.bin"
                n = point_count_from_size(src)
                manifest_rows.append(
                    {
                        "line": line,
                        "method": method,
                        "frame": fid,
                        "input_path": str(src),
                        "output_path": str(out),
                        "input_points": n,
                        "target_output_points": 4 * n,
                        "output_exists": out.exists(),
                    }
                )
    write_csv(REPORT_DIR / "kitti_strict_x4_protocol_manifest.csv", manifest_rows)

    md = [
        "# KITTI Strict x4 Protocol Smoke Report",
        "",
        "## Existing Report Check",
    ]
    for row in existing_reports_rows():
        md.append(f"- `{row['report']}`: {'FOUND' if row['exists'] else 'MISSING'}")
    md.extend(
        [
            "",
            "## Historical Result Reuse Decision",
            "- EAR old results: NOT_REUSABLE, not clean x4.",
            "- PU-Net old results: NOT_REUSABLE, not clean x4.",
            "- PU-GCN old results: NOT_REUSABLE, cap_100k/post-processing issue.",
            "- PDANS old results: NOT_REUSABLE, original+generated+100k cap issue.",
            "- PU-EdgeFormer: PU1K model-100 accepted for KITTI integration candidate; CD/Hausdorff-verified, EMD not verified.",
            "- TULIP: historical/extended only, excluded from strict main table.",
            "- SPU-PMD: excluded.",
            "",
            "## Input Split",
            f"- Original input directory: `{ORIGINAL_INPUT}`",
            f"- Downsampled50 input directory: `{DOWNSAMPLED50_INPUT}`",
            f"- Val split file: `{VAL_SPLIT}`",
            f"- Val split frame count: `{len(frame_ids)}`",
            f"- Original point stats min/max/mean: `{original_summary['min_points']}` / `{original_summary['max_points']}` / `{original_summary['mean_points']}`",
            f"- Downsampled50 point stats min/max/mean: `{down_summary['min_points']}` / `{down_summary['max_points']}` / `{down_summary['mean_points']}`",
            f"- Frame id lists identical across split/original/downsampled50: `{same_ids}`",
            "",
            "## Output Root",
            f"- `{RESULT_ROOT}`",
            "- Baseline native directories contain manifests only; no large data copy was made.",
            "",
            "## Smoke Frames",
            f"- Selected: `{', '.join(smoke_frames)}`",
            f"- Selection notes: `{'; '.join(smoke_reasons) if smoke_reasons else 'all selected from preferred list'}`",
            "",
            "## Smoke Table",
            "",
            "| Line | Method | Frame | Input Points | Target Output Points | Actual Output Points | Ratio | NaN/Inf | Intensity Policy | Status |",
            "| ---- | ------ | ----- | ------------ | -------------------- | -------------------- | ----- | ------- | ---------------- | ------ |",
        ]
    )
    for row in smoke_rows:
        md.append(
            f"| {row['Line']} | {row['Method']} | {row['Frame']} | {row['Input Points']} | "
            f"{row['Target Output Points']} | {row.get('Actual Output Points', '')} | {row.get('Ratio', '')} | "
            f"{row.get('NaN/Inf', '')} | {row['Intensity Policy']} | {row['Status']} |"
        )
    md.extend(
        [
            "",
            "## Interpretation",
            "No strict method smoke outputs were found under the new isolated output root yet, so smoke status is NOT_RUN rather than PASS.",
            "The directory structure, input audit, manifest, method audit, and full-run plan are ready. Run method-specific KITTI inference first, then rerun this script to re-audit outputs.",
        ]
    )
    (REPORT_DIR / "kitti_strict_x4_protocol_smoke_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    command_lines = []
    for line_name, input_dir, out_root in [
        ("lineA_original", ORIGINAL_INPUT, RESULT_ROOT / "lineA_original_up_x4"),
        ("lineB_downsampled50", DOWNSAMPLED50_INPUT, RESULT_ROOT / "lineB_downsampled50_up_x4"),
    ]:
        command_lines.extend(
            [
                f"# {line_name} EAR: run EAR per frame with target 4N, then audit",
                f"python /home/ra87racy/projects/upsampling/EAR/ear_upsampling.py --input_dir {input_dir} --output_dir {out_root / 'EAR'} --keep_ratio 1.0 --up_factor 4.0 --k_neighbors 20 --max_iter 5 --seed 20260629",
                f"# {line_name} PU-Net: set --up_ratio 4; final strict 4N normalization wrapper still required after merge",
                f"python scripts/punet_kitti_adapter.py --input_subdir {input_dir.name} --output_subdir {out_root.relative_to(KITTI_TRAINING) / 'PU-Net' if str(out_root).startswith(str(KITTI_TRAINING)) else out_root / 'PU-Net'} --up_ratio 4 --num_point 1024 --max_input_points 1024 --skip_existing",
                f"# {line_name} PU-GCN: avoid cap_100k; normalize final detector-input bins to 4N",
                f"python scripts/pugcn_kitti_full_frame_reconstruct.py --input_bin {input_dir}/FRAME.bin --output_bin {out_root}/PU-GCN/FRAME.bin --upsample_ratio 4 --max_output_points 0 --include_original true --seed 20260629",
                f"# {line_name} PDANS: disable old 100k cap semantics by using a strict 4N final normalizer after raw generation",
                f"python scripts/pdans_kitti_adapter.py infer --input-folder {input_dir} --output-folder {out_root / 'PDANS'} --line-name {line_name}_pdans_strict_x4 --upsample-ratio 4 --cap-points 0 --seed 20260629",
                f"# {line_name} PU-EdgeFormer: model edgetransformer model-100; KITTI patch wrapper still required",
                f"cd external/reproducibility_check/puedgeformer_minimal_feasibility/PU-EdgeFormer_ops_reuse && python main.py --phase test --model edgetransformer --restore <model-100-dir> --data_dir <kitti-patch-dir> --up_ratio 4",
            ]
        )
    plan = [
        "# KITTI Strict x4 Protocol Full Run Plan",
        "",
        f"- Line A input: `{ORIGINAL_INPUT}`",
        f"- Line B input: `{DOWNSAMPLED50_INPUT}`",
        f"- Output root: `{RESULT_ROOT}`",
        f"- Expected val outputs per method per line: `{len(frame_ids)}`",
        f"- Estimated disk: `{estimate_disk(original_rows + down_rows)}`",
        "- Estimated runtime: EAR likely hours-days CPU; TF1 patch methods depend on GPU/TF ops; PDANS diffusion likely slowest unless batching is tuned.",
        "- Logs: `results/kitti_strict_x4_upsampling_protocol/logs/<line>/<method>/`.",
        "- Retry: frame-level rerun with `--skip_existing`; failed frames listed in method summary CSV.",
        "- Resume: skip output files that already pass size/shape/4N audit.",
        "- Ratio check: rerun `python scripts/kitti_strict_x4_protocol.py` and inspect `reports/kitti_strict_x4_protocol_ratio_audit.csv`.",
        "- Final manifest: `reports/kitti_strict_x4_protocol_manifest.csv`.",
        "- Detector evaluation: DO NOT START until explicitly confirmed.",
        "",
        "## Commands",
        "",
        "```bash",
        *command_lines,
        "```",
        "",
        "## Detector Integration Plan After Full Audit Passes",
        "- Baseline branches keep native point counts: Original baseline native and Downsampled50 baseline native.",
        "- Upsampling branches use strict x4 point counts: Original + upsampling = 4N, Downsampled50 + upsampling = 4N'.",
        "- Baseline does not need to match upsampling point counts.",
        "- Within each upsampling group, methods must match frame-level output counts exactly.",
        "- PointRCNN and CenterPoint must use fixed detector config/checkpoint/split/eval script; detector-internal RPN sampling or voxelization is recorded separately.",
    ]
    (REPORT_DIR / "kitti_strict_x4_protocol_full_run_plan.md").write_text("\n".join(plan) + "\n", encoding="utf-8")

    summary = [
        "# KITTI Strict x4 Protocol Ratio Audit",
        "",
        f"- Full audit rows: `{len(full_audit_rows)}`",
        f"- PASS rows: `{sum(1 for r in full_audit_rows if r.get('Status') == 'PASS')}`",
        f"- NOT_RUN rows: `{sum(1 for r in full_audit_rows if r.get('Status') == 'NOT_RUN')}`",
        f"- FAIL rows: `{sum(1 for r in full_audit_rows if str(r.get('Status', '')).startswith('FAIL'))}`",
        "- Required pass condition: output count equals 4 x line input count for every method/frame, no NaN/Inf, no malformed size.",
    ]
    (REPORT_DIR / "kitti_strict_x4_protocol_ratio_audit.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-op", action="store_true", help="Reserved for compatibility; reports are always written.")
    args = parser.parse_args()
    write_reports(args)
    print(f"Wrote strict protocol reports under {REPORT_DIR}")
    print(f"Prepared output root {RESULT_ROOT}")


if __name__ == "__main__":
    main()
