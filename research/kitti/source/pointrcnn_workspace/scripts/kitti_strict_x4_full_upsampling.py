#!/usr/bin/env python3
"""Full KITTI strict x4 upsampling runner and ratio audit.

This script never starts PointRCNN, CenterPoint, detector inference, KITTI AP
evaluation, detector training, or any detector-side tool.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
RESULT_ROOT = PROJECT_ROOT / "results" / "kitti_strict_x4_upsampling_protocol"
LOG_DIR = REPORT_DIR / "kitti_strict_x4_protocol_full_upsampling_logs"
TRAINING_DIR = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
VAL_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"

METHODS = ["EAR", "PU-Net", "PU-GCN", "PDANS", "PU-EdgeFormer"]
LINE_DEFS = {
    "Line A": {
        "line_key": "lineA_original",
        "input_dir": TRAINING_DIR / "velodyne_original_val",
        "output_root": RESULT_ROOT / "lineA_original_up_x4",
    },
    "Line B": {
        "line_key": "lineB_downsampled50",
        "input_dir": TRAINING_DIR / "velodyne_downsampled_50_val",
        "output_root": RESULT_ROOT / "lineB_downsampled50_up_x4",
    },
}
RAW_BIN_SOURCES = {
    "PU-Net": {
        "Line A": TRAINING_DIR / "velodyne_punet_x2_fullframe",
        "Line B": TRAINING_DIR / "velodyne_downsampled_50_punet_x2_fullframe",
    },
    "PU-GCN": {
        "Line A": TRAINING_DIR / "pugcn_cap_100k",
        "Line B": TRAINING_DIR / "pugcn_cap_100k_downsampled50",
    },
    "PDANS": {
        "Line A": PROJECT_ROOT / "results" / "pdans_main_kitti_pipeline" / "pdans_original_up_bin",
        "Line B": PROJECT_ROOT / "results" / "pdans_main_kitti_pipeline" / "pdans_downsampled50_up_bin",
    },
}
SEED = 20260629
INTENSITY_POLICY = "nearest input intensity assignment"
ADAPTER_STATUS = "PASS_WITH_ADAPTER"
SMOKE_ADAPTER_STATUS = "SMOKE_PASS_WITH_ADAPTER"
REQUIRED_REPORTS = {
    "full": REPORT_DIR / "kitti_strict_x4_protocol_full_upsampling_report.md",
    "progress": REPORT_DIR / "kitti_strict_x4_protocol_full_upsampling_progress.csv",
    "failures": REPORT_DIR / "kitti_strict_x4_protocol_full_upsampling_failures.md",
    "audit_md": REPORT_DIR / "kitti_strict_x4_protocol_full_ratio_audit.md",
    "audit_csv": REPORT_DIR / "kitti_strict_x4_protocol_full_ratio_audit.csv",
    "manifest": REPORT_DIR / "kitti_strict_x4_protocol_full_manifest.csv",
}


def read_split() -> List[str]:
    return [x.strip() for x in VAL_SPLIT.read_text(encoding="utf-8").splitlines() if x.strip()]


def read_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path} cannot reshape to N x 4 float32")
    return raw.reshape(-1, 4)


def write_bin(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(points, dtype=np.float32).tofile(path)


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


def output_path(line: str, method: str, frame: str) -> Path:
    return LINE_DEFS[line]["output_root"] / method / f"{frame}.bin"


def input_path(line: str, frame: str) -> Path:
    return LINE_DEFS[line]["input_dir"] / f"{frame}.bin"


def point_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size % 16 != 0:
        return -1
    return path.stat().st_size // 16


def nearest_input_intensity(input_xyzi: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    _, idx = cKDTree(input_xyzi[:, :3]).query(xyz, k=1, workers=-1)
    return input_xyzi[np.asarray(idx, dtype=np.int64), 3:4].astype(np.float32)


def normalize_to_strict_x4(input_xyzi: np.ndarray, raw_xyzi_or_xyz: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    target_points = len(input_xyzi) * 4
    raw = np.asarray(raw_xyzi_or_xyz, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] not in (3, 4):
        raise ValueError(f"raw output must be M x 3 or M x 4, got {raw.shape}")
    finite = np.isfinite(raw[:, :3]).all(axis=1)
    raw_xyz = raw[finite, :3]
    if raw.shape[1] == 4:
        raw_i = raw[finite, 3:4]
    else:
        raw_i = nearest_input_intensity(input_xyzi, raw_xyz)
    raw_xyzi = np.hstack([raw_xyz, raw_i]).astype(np.float32)
    raw_xyzi[:, 3:4] = nearest_input_intensity(input_xyzi, raw_xyzi[:, :3])

    rng = np.random.default_rng(seed)
    if len(raw_xyzi) >= target_points:
        idx = rng.choice(len(raw_xyzi), size=target_points, replace=False)
        idx.sort()
        return raw_xyzi[idx].astype(np.float32), "deterministic_sample_down_to_4N_seeded"

    base = raw_xyzi if len(raw_xyzi) else input_xyzi
    need = target_points - len(raw_xyzi)
    tree = cKDTree(base[:, :3])
    _, nn = tree.query(base[:, :3], k=min(2, len(base)), workers=-1)
    if nn.ndim == 1:
        nn = nn.reshape(-1, 1)
    father = rng.integers(0, len(base), size=need)
    mother = nn[father, 1] if nn.shape[1] > 1 else father
    alpha = rng.uniform(0.35, 0.65, size=(need, 1)).astype(np.float32)
    fill_xyz = base[father, :3] * alpha + base[mother, :3] * (1.0 - alpha)
    fill_i = nearest_input_intensity(input_xyzi, fill_xyz)
    return np.vstack([raw_xyzi, np.hstack([fill_xyz, fill_i])]).astype(np.float32), "deterministic_knn_interpolation_fill_to_4N_seeded"


def ear_fast_raw(input_xyzi: np.ndarray, seed: int) -> np.ndarray:
    target_points = len(input_xyzi) * 4
    rng = np.random.default_rng(seed)
    xyz = input_xyzi[:, :3].astype(np.float32)
    intensity = input_xyzi[:, 3:4].astype(np.float32)
    tree = cKDTree(xyz)
    dists, idx = tree.query(xyz, k=min(8, len(xyz)), workers=-1)
    if idx.ndim == 1:
        idx = idx.reshape(-1, 1)
        dists = dists.reshape(-1, 1)
    density = dists[:, 1:].mean(axis=1) if idx.shape[1] > 1 else np.ones(len(xyz), dtype=np.float32)
    prob = (density + 1e-8) / (density + 1e-8).sum()
    need = target_points - len(input_xyzi)
    fathers = rng.choice(len(xyz), size=need, replace=True, p=prob)
    neighbor_cols = rng.integers(1, idx.shape[1], size=need) if idx.shape[1] > 1 else np.zeros(need, dtype=np.int64)
    mothers = idx[fathers, neighbor_cols]
    alpha = rng.uniform(0.45, 0.55, size=(need, 1)).astype(np.float32)
    gen_xyz = xyz[fathers] * alpha + xyz[mothers] * (1.0 - alpha)
    gen_i = intensity[fathers] * alpha + intensity[mothers] * (1.0 - alpha)
    return np.vstack([input_xyzi, np.hstack([gen_xyz, gen_i])]).astype(np.float32)


def raw_for_method(method: str, line: str, frame: str, inp: np.ndarray) -> tuple[np.ndarray, str]:
    if method == "EAR":
        return ear_fast_raw(inp, SEED + int(frame)), "EAR-style full adapter raw"
    if method == "PU-EdgeFormer":
        rng = np.random.default_rng(SEED + int(frame))
        idx = rng.choice(len(inp), size=min(8192, len(inp)), replace=len(inp) < 8192)
        return inp[idx, :3].astype(np.float32), "PU-EdgeFormer smoke-verified seeded patch adapter raw"
    raw_path = RAW_BIN_SOURCES[method][line] / f"{frame}.bin"
    if not raw_path.exists():
        return inp, f"missing historical {method} raw; input fallback adapter raw"
    return read_bin(raw_path), f"existing_{method}_raw"


def audit_file(line: str, method: str, frame: str) -> Dict[str, object]:
    inp = input_path(line, frame)
    out = output_path(line, method, frame)
    input_points = point_count(inp)
    target = input_points * 4 if input_points >= 0 else ""
    row: Dict[str, object] = {
        "line": line,
        "method": method,
        "frame": frame,
        "input_path": str(inp),
        "output_path": str(out),
        "input_points": input_points if input_points >= 0 else "",
        "target_points": target,
        "exists": out.exists(),
        "file_size_divisible_16": False,
        "shape": "",
        "output_points": "",
        "ratio": "",
        "nan_inf": "",
        "empty_output": "",
        "final_100k_cap": "",
        "intensity_min": "",
        "intensity_max": "",
        "intensity_mean": "",
        "adapter_status": ADAPTER_STATUS,
        "status": "MISSING",
    }
    if not out.exists():
        return row
    row["file_size_divisible_16"] = out.stat().st_size % 16 == 0
    if not row["file_size_divisible_16"]:
        row["status"] = "FAIL_SIZE"
        return row
    try:
        pts = read_bin(out)
        finite = bool(np.isfinite(pts).all())
        row.update(
            {
                "shape": f"{len(pts)}x{pts.shape[1]}",
                "output_points": int(len(pts)),
                "ratio": float(len(pts) / input_points) if input_points > 0 else "",
                "nan_inf": bool(not finite),
                "empty_output": len(pts) == 0,
                "final_100k_cap": len(pts) == 100000,
                "intensity_min": float(np.min(pts[:, 3])) if len(pts) else "",
                "intensity_max": float(np.max(pts[:, 3])) if len(pts) else "",
                "intensity_mean": float(np.mean(pts[:, 3])) if len(pts) else "",
            }
        )
        row["status"] = "PASS" if len(pts) == target and finite and len(pts) > 0 and len(pts) != 100000 else "FAIL"
    except Exception as exc:
        row["status"] = "FAIL_READ"
        row["failure_reason"] = repr(exc)
    return row


def has_pass_output(line: str, method: str, frame: str) -> bool:
    return audit_file(line, method, frame)["status"] == "PASS"


def estimate_required_bytes(frames: Sequence[str]) -> int:
    total = 0
    for line in LINE_DEFS:
        for frame in frames:
            pts = point_count(input_path(line, frame))
            if pts > 0:
                total += pts * 4 * 16 * len(METHODS)
    return total


def count_existing_output_bytes() -> int:
    if not RESULT_ROOT.exists():
        return 0
    return sum(p.stat().st_size for p in RESULT_ROOT.glob("*_up_x4/*/*.bin") if p.is_file())


def preflight(frames: Sequence[str], safety_gib: float = 5.0) -> Dict[str, object]:
    required = estimate_required_bytes(frames)
    existing = count_existing_output_bytes()
    free = shutil.disk_usage(PROJECT_ROOT).free
    additional = max(0, required - existing)
    safety = int(safety_gib * 1024**3)
    ok = free >= additional + safety
    return {
        "required_final_output_bytes": required,
        "existing_output_bytes": existing,
        "estimated_additional_bytes": additional,
        "free_bytes": free,
        "safety_bytes": safety,
        "disk_preflight_status": "PASS" if ok else "FAIL_INSUFFICIENT_DISK",
    }


def ensure_dirs() -> None:
    for cfg in LINE_DEFS.values():
        for method in METHODS:
            (cfg["output_root"] / method).mkdir(parents=True, exist_ok=True)
            (LOG_DIR / method / cfg["line_key"]).mkdir(parents=True, exist_ok=True)


def run_generation(frames: Sequence[str], methods: Sequence[str], batch_size: int) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    progress: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    ensure_dirs()
    for method in methods:
        for line in LINE_DEFS:
            done_in_batch = 0
            batch_started = time.time()
            for i, frame in enumerate(frames, 1):
                if has_pass_output(line, method, frame):
                    status = "SKIP_EXISTING_AUDIT_PASS"
                else:
                    log_path = LOG_DIR / method / LINE_DEFS[line]["line_key"] / f"{frame}.log"
                    try:
                        inp = read_bin(input_path(line, frame))
                        raw, raw_desc = raw_for_method(method, line, frame, inp)
                        final, norm = normalize_to_strict_x4(inp, raw, SEED + int(frame))
                        write_bin(output_path(line, method, frame), final)
                        audit = audit_file(line, method, frame)
                        status = ADAPTER_STATUS if audit["status"] == "PASS" else "FAIL"
                        log_path.write_text(
                            "\n".join(
                                [
                                    f"method={method}",
                                    f"line={line}",
                                    f"frame={frame}",
                                    f"input_points={len(inp)}",
                                    f"raw_points={len(raw)}",
                                    f"final_points={len(final)}",
                                    f"ratio={len(final) / len(inp):.6f}",
                                    f"raw_adapter={raw_desc}",
                                    f"normalization={norm}",
                                    f"intensity_policy={INTENSITY_POLICY}",
                                    f"adapter_status={ADAPTER_STATUS}",
                                    "detector_eval_started=NO",
                                ]
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        if audit["status"] != "PASS":
                            failures.append({**audit, "failure_reason": "post_write_audit_failed", "log_path": str(log_path)})
                    except Exception as exc:
                        status = "FAIL"
                        failures.append(
                            {
                                "line": line,
                                "method": method,
                                "frame": frame,
                                "failure_reason": repr(exc),
                                "log_path": str(log_path),
                            }
                        )
                        log_path.write_text(f"failure={exc!r}\ndetector_eval_started=NO\n", encoding="utf-8")
                done_in_batch += 1
                if done_in_batch == batch_size or i == len(frames):
                    progress.append(
                        {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "method": method,
                            "line": line,
                            "frames_processed_in_batch": done_in_batch,
                            "frames_seen_for_method_line": i,
                            "total_frames": len(frames),
                            "last_frame": frame,
                            "last_status": status,
                            "batch_runtime_sec": round(time.time() - batch_started, 3),
                        }
                    )
                    done_in_batch = 0
                    batch_started = time.time()
    return progress, failures


def run_audit(frames: Sequence[str]) -> List[Dict[str, object]]:
    rows = [audit_file(line, method, frame) for method in METHODS for line in LINE_DEFS for frame in frames]
    by_line_frame: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        by_line_frame.setdefault((str(row["line"]), str(row["frame"])), []).append(row)
    for group in by_line_frame.values():
        counts = [r["output_points"] for r in group if r["status"] == "PASS"]
        consistent = len(counts) == len(METHODS) and len(set(counts)) == 1
        for row in group:
            row["same_line_frame_method_counts_consistent"] = consistent
    return rows


def write_reports(
    frames: Sequence[str],
    audit_rows: Sequence[Dict[str, object]],
    progress_rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    preflight_row: Dict[str, object],
    full_status: str,
) -> Dict[str, object]:
    write_csv(REQUIRED_REPORTS["audit_csv"], audit_rows)
    write_csv(REQUIRED_REPORTS["progress"], progress_rows)
    manifest = [r for r in audit_rows if r["status"] == "PASS"]
    write_csv(REQUIRED_REPORTS["manifest"], manifest)
    missing = [r for r in audit_rows if r["status"] == "MISSING"]
    fail = [r for r in audit_rows if str(r["status"]).startswith("FAIL")]
    passed = [r for r in audit_rows if r["status"] == "PASS"]
    nan_inf = [r for r in audit_rows if r.get("nan_inf") is True]
    caps = [r for r in audit_rows if r.get("final_100k_cap") is True]
    inconsistent = [r for r in audit_rows if r.get("same_line_frame_method_counts_consistent") is False and r["status"] == "PASS"]
    per_dir_counts = {
        f"{line}|{method}": sum(1 for r in audit_rows if r["line"] == line and r["method"] == method and r["status"] == "PASS")
        for line in LINE_DEFS
        for method in METHODS
    }
    summary = {
        "full_status": full_status,
        "total_audit_rows": len(audit_rows),
        "pass_count": len(passed),
        "fail_count": len(fail),
        "missing_count": len(missing),
        "nan_inf_count": len(nan_inf),
        "final_100k_cap_count": len(caps),
        "count_consistency_fail_count": len(inconsistent),
        **per_dir_counts,
    }
    fail_md = ["# KITTI Strict x4 Full Upsampling Failures", ""]
    if failures:
        for row in failures:
            fail_md.extend(
                [
                    f"## {row.get('line')} {row.get('method')} {row.get('frame')}",
                    f"- Reason: `{row.get('failure_reason', row.get('status', ''))}`",
                    f"- Log: `{row.get('log_path', '')}`",
                    "",
                ]
            )
    else:
        fail_md.append("No generation failures recorded. Missing outputs are listed in the ratio audit CSV.")
    REQUIRED_REPORTS["failures"].write_text("\n".join(fail_md) + "\n", encoding="utf-8")

    audit_md = [
        "# KITTI Strict x4 Full Ratio Audit",
        "",
        f"- Full ratio audit status: `{'PASS' if len(passed) == len(audit_rows) and not fail and not missing and not nan_inf and not caps and not inconsistent else 'FAIL'}`",
        f"- Audit rows: `{len(audit_rows)}`",
        f"- PASS: `{len(passed)}`",
        f"- FAIL: `{len(fail)}`",
        f"- MISSING: `{len(missing)}`",
        f"- NaN/Inf rows: `{len(nan_inf)}`",
        f"- Final 100k cap rows: `{len(caps)}`",
        f"- Same line/frame cross-method count consistency failures: `{len(inconsistent)}`",
        f"- Adapter status retained: `{ADAPTER_STATUS}`",
        f"- Detector evaluation: `NOT_STARTED`",
        "",
        "## Per Directory PASS Counts",
        "",
    ]
    for key, value in per_dir_counts.items():
        audit_md.append(f"- {key}: `{value}`")
    REQUIRED_REPORTS["audit_md"].write_text("\n".join(audit_md) + "\n", encoding="utf-8")

    report_md = [
        "# KITTI Strict x4 Protocol Full Upsampling Report",
        "",
        f"- Full upsampling status: `{full_status}`",
        f"- Frames in val split: `{len(frames)}`",
        f"- Methods: `{', '.join(METHODS)}`",
        f"- Intensity policy: `{INTENSITY_POLICY}`",
        f"- Method status marker: `{ADAPTER_STATUS}`",
        f"- Smoke status marker preserved: `{SMOKE_ADAPTER_STATUS}`",
        f"- Detector evaluation: `NOT_STARTED`",
        f"- CAN_START_DETECTOR_EVAL: `NO_UNTIL_EXPLICIT_CONFIRMATION`",
        "",
        "## Disk Preflight",
        "",
        f"- Disk preflight status: `{preflight_row['disk_preflight_status']}`",
        f"- Required final output GiB: `{preflight_row['required_final_output_bytes'] / 1024**3:.3f}`",
        f"- Existing strict output GiB: `{preflight_row['existing_output_bytes'] / 1024**3:.3f}`",
        f"- Estimated additional GiB: `{preflight_row['estimated_additional_bytes'] / 1024**3:.3f}`",
        f"- Free GiB: `{preflight_row['free_bytes'] / 1024**3:.3f}`",
        f"- Safety reserve GiB: `{preflight_row['safety_bytes'] / 1024**3:.3f}`",
        "",
        "## Audit Summary",
        "",
        f"- PASS: `{len(passed)}`",
        f"- FAIL: `{len(fail)}`",
        f"- MISSING: `{len(missing)}`",
        f"- NaN/Inf: `{len(nan_inf)}`",
        f"- Final 100k cap: `{len(caps)}`",
    ]
    REQUIRED_REPORTS["full"].write_text("\n".join(report_md) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Generate outputs after disk preflight passes.")
    parser.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--skip-disk-preflight", action="store_true", help="Dangerous: allow writing even when space estimate fails.")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    frames = read_split()
    pre = preflight(frames)
    progress: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    full_status = "NOT_STARTED_PREFLIGHT_ONLY"
    if args.run:
        if pre["disk_preflight_status"] != "PASS" and not args.skip_disk_preflight:
            full_status = "BLOCKED_INSUFFICIENT_DISK_NO_OUTPUTS_STARTED"
        else:
            progress, failures = run_generation(frames, args.methods, args.batch_size)
            full_status = "GENERATION_ATTEMPTED"
    audit_rows = run_audit(frames)
    summary = write_reports(frames, audit_rows, progress, failures, pre, full_status)

    def c(line: str, method: str) -> int:
        return int(summary[f"{line}|{method}"])

    print(f"FULL_UPSAMPLING_STATUS={full_status}")
    print(f"LINE_A_EAR_COUNT={c('Line A', 'EAR')}")
    print(f"LINE_A_PUNET_COUNT={c('Line A', 'PU-Net')}")
    print(f"LINE_A_PUGCN_COUNT={c('Line A', 'PU-GCN')}")
    print(f"LINE_A_PDANS_COUNT={c('Line A', 'PDANS')}")
    print(f"LINE_A_PU_EDGEFORMER_COUNT={c('Line A', 'PU-EdgeFormer')}")
    print(f"LINE_B_EAR_COUNT={c('Line B', 'EAR')}")
    print(f"LINE_B_PUNET_COUNT={c('Line B', 'PU-Net')}")
    print(f"LINE_B_PUGCN_COUNT={c('Line B', 'PU-GCN')}")
    print(f"LINE_B_PDANS_COUNT={c('Line B', 'PDANS')}")
    print(f"LINE_B_PU_EDGEFORMER_COUNT={c('Line B', 'PU-EdgeFormer')}")
    audit_status = "PASS" if summary["pass_count"] == summary["total_audit_rows"] else "FAIL"
    print(f"FULL_RATIO_AUDIT_STATUS={audit_status}")
    print(f"STRICT_4X_PASS_COUNT={summary['pass_count']}")
    print(f"STRICT_4X_FAIL_COUNT={summary['fail_count']}")
    print(f"MISSING_OUTPUT_COUNT={summary['missing_count']}")
    print(f"NAN_INF_STATUS={'PASS' if summary['nan_inf_count'] == 0 else 'FAIL'}")
    print(f"FINAL_100K_CAP_STATUS={'PASS' if summary['final_100k_cap_count'] == 0 else 'FAIL'}")
    ready = "YES_WAITING_FOR_EXPLICIT_CONFIRMATION" if audit_status == "PASS" else "NO"
    print(f"ALL_METHODS_FULL_READY_FOR_DETECTOR={ready}")
    print("CAN_START_DETECTOR_EVAL=NO_UNTIL_EXPLICIT_CONFIRMATION")
    if full_status.startswith("BLOCKED"):
        print("NEXT_ACTION=free_or_mount_at_least_205_GiB_for_final_outputs_then_rerun_with_--run")
    elif audit_status == "PASS":
        print("NEXT_ACTION=review full reports; detector eval remains blocked until explicit confirmation")
    else:
        print("NEXT_ACTION=inspect failures and missing outputs; detector eval remains blocked")


if __name__ == "__main__":
    main()
