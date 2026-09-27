#!/usr/bin/env python3
"""Audit ModelNet40 upsampling ratios: config + on-disk outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
KITTI_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/datasets/KITTI")

MAIN_METHODS = {"EAR", "PDANS", "PU-Net", "PU-GCN"}
MAIN_PROTOCOL_R = 4
MAIN_PROTOCOL_TARGET = {"A": 4096, "B": 2048}
MAIN_PROTOCOL_INPUT = {"A": 1024, "B": 512}

METHOD_REGISTRY = [
    {
        "line": "A",
        "method": "EAR",
        "method_group": "main",
        "output_subdir": "ear_x4",
        "input_dataset": "modelnet40_original",
        "configured_ratio": "target_n=4096 / up_factor=4.0 (main protocol R=×4)",
        "ratio_source_file": "scripts/ear_modelnet40_utils.py:TARGET_POINTS_X4_ORIGINAL=4096; UP_FACTOR_X4=4.0",
        "scripts": "scripts/run_ear_x4_smoke.py; scripts/run_ear_x4_chunk.py",
        "jobs": "jobs/run_ear_x4_lineA_cpu_array.sbatch",
        "protocol_role": "main_pending",
    },
    {
        "line": "A",
        "method": "EAR",
        "method_group": "main",
        "output_subdir": "ear_x4_smoke",
        "input_dataset": "smoke_inputs/original",
        "configured_ratio": "target_n=4096 / up_factor=4.0 → 1024→4096",
        "ratio_source_file": "scripts/ear_modelnet40_utils.py:TARGET_POINTS_X4_ORIGINAL=4096",
        "scripts": "scripts/run_ear_x4_smoke.py",
        "jobs": "(smoke only)",
        "protocol_role": "main_smoke",
    },
    {
        "line": "B",
        "method": "EAR",
        "method_group": "main",
        "output_subdir": "ear_x4",
        "input_dataset": "modelnet40_downsampled50",
        "configured_ratio": "target_n=2048 / up_factor=4.0 (main protocol R=×4)",
        "ratio_source_file": "scripts/ear_modelnet40_utils.py:TARGET_POINTS_X4_DOWN=2048; UP_FACTOR_X4=4.0",
        "scripts": "scripts/run_ear_x4_smoke.py; scripts/run_ear_x4_chunk.py",
        "jobs": "jobs/run_ear_x4_lineB_cpu_array.sbatch",
        "protocol_role": "main_pending",
    },
    {
        "line": "B",
        "method": "EAR",
        "method_group": "main",
        "output_subdir": "ear_x4_smoke",
        "input_dataset": "smoke_inputs/downsampled50",
        "configured_ratio": "target_n=2048 / up_factor=4.0 → 512→2048",
        "ratio_source_file": "scripts/ear_modelnet40_utils.py:TARGET_POINTS_X4_DOWN=2048",
        "scripts": "scripts/run_ear_x4_smoke.py",
        "jobs": "(smoke only)",
        "protocol_role": "main_smoke",
    },
    {
        "line": "A",
        "method": "EAR",
        "method_group": "main",
        "output_subdir": "ear",
        "input_dataset": "modelnet40_original",
        "configured_ratio": "legacy ×2 wrapper: target_n=1024 / up_factor=2.0 (identity at 1024 input)",
        "ratio_source_file": "scripts/ear_modelnet40_utils.py:TARGET_POINTS=1024 (legacy)",
        "scripts": "scripts/run_ear_smoke.py (legacy)",
        "jobs": "(not used in main protocol)",
        "protocol_role": "legacy_x2",
    },
    {
        "line": "B",
        "method": "EAR",
        "method_group": "main",
        "output_subdir": "ear",
        "input_dataset": "modelnet40_downsampled50",
        "configured_ratio": "legacy ×2: target_n=1024 / up_factor=2.0 → 512→1024",
        "ratio_source_file": "scripts/ear_modelnet40_utils.py; scripts/run_ear_full_lineB_chunk.py",
        "scripts": "scripts/run_ear_full_lineB_chunk.py",
        "jobs": "jobs/run_ear_full_lineB_cpu_array.sbatch",
        "protocol_role": "ablation_preliminary_x2",
    },
    {
        "line": "A",
        "method": "PDANS",
        "method_group": "main",
        "output_subdir": "pdans",
        "input_dataset": "modelnet40_original",
        "configured_ratio": "R=4 (default, configurable via --R)",
        "ratio_source_file": "external_lab_migrated/PDANS/pointnet2/samples.py:--R default 4",
        "scripts": "(not yet in project; planned step7)",
        "jobs": "(none)",
    },
    {
        "line": "B",
        "method": "PDANS",
        "method_group": "main",
        "output_subdir": "pdans",
        "input_dataset": "modelnet40_downsampled50",
        "configured_ratio": "R=4 (default) or R=2 (planned)",
        "ratio_source_file": "external_lab_migrated/PDANS/pointnet2/example_eval.py:R=4",
        "scripts": "(not yet in project)",
        "jobs": "(none)",
    },
    {
        "line": "A",
        "method": "PU-Net",
        "method_group": "main",
        "output_subdir": "punet",
        "input_dataset": "modelnet40_original",
        "configured_ratio": "up_ratio=4 (default in main.py)",
        "ratio_source_file": "external_lab_migrated/PU-Net/code/main.py:--up_ratio default 4",
        "scripts": "(not yet in project)",
        "jobs": "(none)",
    },
    {
        "line": "B",
        "method": "PU-Net",
        "method_group": "main",
        "output_subdir": "punet",
        "input_dataset": "modelnet40_downsampled50",
        "configured_ratio": "up_ratio=4 (default); configurable",
        "ratio_source_file": "external_lab_migrated/PU-Net/code/main.py:--up_ratio default 4",
        "scripts": "(not yet in project)",
        "jobs": "(none)",
    },
    {
        "line": "A",
        "method": "PU-GCN",
        "method_group": "main",
        "output_subdir": "pugcn",
        "input_dataset": "modelnet40_original",
        "configured_ratio": "up_ratio=4 (default in configs.py)",
        "ratio_source_file": "external_lab_migrated/PU-GCN/Upsampling/configs.py:--up_ratio default 4",
        "scripts": "(not yet in project)",
        "jobs": "(none)",
    },
    {
        "line": "B",
        "method": "PU-GCN",
        "method_group": "main",
        "output_subdir": "pugcn",
        "input_dataset": "modelnet40_downsampled50",
        "configured_ratio": "up_ratio=4 (default); configurable",
        "ratio_source_file": "external_lab_migrated/PU-GCN/Upsampling/configs.py:--up_ratio default 4",
        "scripts": "(not yet in project)",
        "jobs": "(none)",
    },
    {
        "line": "A",
        "method": "TULIP",
        "method_group": "supplementary",
        "output_subdir": "tulip",
        "input_dataset": "modelnet40_original",
        "configured_ratio": "range-image ×4 (16×1024 → 64×1024); KITTI-trained tulip_base",
        "ratio_source_file": "external_lab_migrated/TULIP/scripts/tulip_kitti_smoke_infer.py: img_size=(16,1024) target_img_size=(64,1024)",
        "scripts": "external_lab_migrated/TULIP/scripts/tulip_kitti_smoke_infer.py; tulip/main_lidar_upsampling.py",
        "jobs": "(none — no ModelNet40 wrapper/job)",
    },
    {
        "line": "B",
        "method": "TULIP",
        "method_group": "supplementary",
        "output_subdir": "tulip",
        "input_dataset": "modelnet40_downsampled50",
        "configured_ratio": "range-image ×4 (16×1024 → 64×1024); xyz count not fixed",
        "ratio_source_file": "external_lab_migrated/TULIP/tulip/model/tulip.py; TULIP_pointRCNN_consistency.md",
        "scripts": "external_lab_migrated/TULIP/scripts/convert_tulip_ply_to_kitti_bin.py",
        "jobs": "(none — no ModelNet40 wrapper/job)",
    },
]

TULIP_TAGS = "range-image-based / LiDAR-style upsampling candidate / supplementary candidate"


@dataclass
class ScanResult:
    train_count: int = 0
    test_count: int = 0
    output_min: int | None = None
    output_max: int | None = None
    output_mean: float | None = None
    ratio_min: float | None = None
    ratio_max: float | None = None
    ratio_mean: float | None = None
    nan_inf: bool = False
    sample_shape: str = ""
    input_points: int | None = None
    has_data: bool = False
    file_types: str = ""


@dataclass
class KittiTulipRef:
    n: int
    input_min: int
    input_max: int
    input_mean: float
    output_min: int
    output_max: int
    output_mean: float
    ratio_min: float
    ratio_max: float
    ratio_mean: float
    ratio_std: float


def expected_input_points(line: str, input_dataset: str) -> int:
    if "downsampled" in input_dataset:
        return 512
    if "original" in input_dataset:
        return 1024
    return 0


def point_count_from_file(path: Path) -> int | None:
    try:
        if path.suffix == ".npy":
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return int(arr.shape[0])
            if arr.ndim == 3:
                return int(np.count_nonzero(arr[..., 0] > 0))
        elif path.suffix == ".bin":
            arr = np.fromfile(path, dtype=np.float32)
            if arr.size % 4 == 0:
                return int(arr.reshape(-1, 4).shape[0])
        elif path.suffix == ".ply":
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("element vertex"):
                        return int(line.split()[-1])
    except Exception:
        return None
    return None


def scan_output_dir(output_root: Path, input_pts: int) -> ScanResult:
    res = ScanResult(input_points=input_pts)
    if not output_root.is_dir():
        return res

    counts: list[int] = []
    exts_seen: set[str] = set()
    for split in ("train", "test"):
        split_dir = output_root / split
        if not split_dir.is_dir():
            continue
        files = [
            *split_dir.rglob("*.npy"),
            *split_dir.rglob("*.ply"),
            *split_dir.rglob("*.bin"),
        ]
        if split == "train":
            res.train_count = len(files)
        else:
            res.test_count = len(files)
        for fpath in files:
            exts_seen.add(fpath.suffix)
            n = point_count_from_file(fpath)
            if n is None:
                continue
            counts.append(n)
            if fpath.suffix == ".npy":
                arr = np.load(fpath)
                if not np.isfinite(arr).all():
                    res.nan_inf = True
            if not res.sample_shape:
                res.sample_shape = f"({n}, *) via {fpath.suffix}"

    res.file_types = ",".join(sorted(exts_seen)) if exts_seen else ""
    if not counts:
        return res

    res.has_data = True
    res.output_min = min(counts)
    res.output_max = max(counts)
    res.output_mean = float(np.mean(counts))
    if input_pts > 0:
        ratios = [c / input_pts for c in counts]
        res.ratio_min = min(ratios)
        res.ratio_max = max(ratios)
        res.ratio_mean = float(np.mean(ratios))
    return res


def kitti_tulip_reference(line: str) -> KittiTulipRef | None:
    """Paired KITTI .bin reference — supplementary evidence for TULIP xyz behavior."""
    val_path = Path(
        "/home/woody/iwnt/iwnt189h/thesis_pointcloud/code/detectors/OpenPCDet/data/kitti/ImageSets/val.txt"
    )
    if not val_path.is_file():
        return None
    ids = [x.strip() for x in val_path.read_text().splitlines() if x.strip()]
    if line == "A":
        base = KITTI_ROOT / "object/training/velodyne"
        tulip = KITTI_ROOT / "upsampled_variants/original_tulip"
    else:
        base = KITTI_ROOT / "object/training/velodyne_downsampled_50"
        tulip = KITTI_ROOT / "upsampled_variants/downsampled50_tulip"

    ratios: list[float] = []
    ins: list[int] = []
    outs: list[int] = []
    for fid in ids:
        bp, tp = base / f"{fid}.bin", tulip / f"{fid}.bin"
        if not bp.is_file() or not tp.is_file():
            continue
        ni = point_count_from_file(bp)
        no = point_count_from_file(tp)
        if ni and no:
            ins.append(ni)
            outs.append(no)
            ratios.append(no / ni)
    if not ratios:
        return None
    r = np.array(ratios)
    ic = np.array(ins)
    oc = np.array(outs)
    return KittiTulipRef(
        n=len(r),
        input_min=int(ic.min()),
        input_max=int(ic.max()),
        input_mean=float(ic.mean()),
        output_min=int(oc.min()),
        output_max=int(oc.max()),
        output_mean=float(oc.mean()),
        ratio_min=float(r.min()),
        ratio_max=float(r.max()),
        ratio_mean=float(r.mean()),
        ratio_std=float(r.std()),
    )


def classify_tulip(line: str, scan: ScanResult, kref: KittiTulipRef | None = None) -> tuple[str, str]:
    if scan.has_data:
        stable = scan.output_min == scan.output_max
        if stable and scan.ratio_mean is not None:
            return (
                "supplementary_only",
                f"ModelNet40 TULIP outputs exist but method is range-image/LiDAR-style; "
                f"keep as supplementary, not main xyz upsampling group",
            )
        return (
            "supplementary_only",
            "ModelNet40 TULIP outputs exist but point counts vary; "
            "range-image projection/filtering prevents fixed xyz ratio",
        )

    kref = kref if kref is not None else kitti_tulip_reference(line)
    base_note = (
        "No ModelNet40 .npy/.ply/.bin outputs; output dir not created. "
        "TULIP is range-image-based / LiDAR-style — requires xyz→range-image adapter for objects. "
        f"Tags: {TULIP_TAGS}."
    )
    if kref is None:
        return "not_main_for_modelnet40", base_note

    stable_xyz = False
    theory_x4 = True
    note = (
        f"{base_note} KITTI supplementary reference (n={kref.n}): "
        f"xyz output min/max/mean={kref.output_min}/{kref.output_max}/{kref.output_mean:.1f}; "
        f"actual xyz ratio min/max/mean={kref.ratio_min:.4f}/{kref.ratio_max:.4f}/{kref.ratio_mean:.4f} "
        f"(std={kref.ratio_std:.4f}). "
        f"Theoretical range-image upsampling is ×4 (16-line→64-line), but xyz point ratio is "
        f"{'not ×4' if theory_x4 else 'unknown'} and varies per frame due to valid-pixel mask / projection / filtering."
    )
    if not stable_xyz:
        return "supplementary_only", note + " Status: supplementary_only — not aligned with fixed-R xyz main protocol."
    return "not_main_for_modelnet40", note


def classify_status(
    method: str,
    line: str,
    subdir: str,
    scan: ScanResult,
    input_pts: int,
    protocol_role: str = "",
    kitti_ref: KittiTulipRef | None = None,
) -> tuple[str, str]:
    if method == "TULIP":
        return classify_tulip(line, scan, kitti_ref)

    main_target = MAIN_PROTOCOL_TARGET.get(line, 0)
    role = protocol_role or ""

    if role == "ablation_preliminary_x2":
        if scan.has_data and scan.output_min == 1024 and scan.ratio_mean is not None and abs(scan.ratio_mean - 2.0) < 0.05:
            return (
                "ablation_preliminary_x2",
                f"Legacy EAR ×2 full output preserved: {input_pts}→1024; not main protocol (main target={main_target})",
            )
        if not scan.has_data:
            return "ablation_preliminary_x2", "Legacy ×2 path; no full data (ablation slot)"
        return "ablation_preliminary_x2", f"Legacy ×2 artifact; main protocol uses ear_x4 → {main_target} pts"

    if role == "legacy_x2":
        return "ablation_preliminary_x2", f"Legacy ×2 wrapper (target_n=1024); main protocol target={main_target}"

    if role in ("main_pending", "main_smoke") and subdir.startswith("ear_x4"):
        if not scan.has_data:
            return "pending_generation", f"EAR ×4 main output not yet generated; target={main_target} pts (R=×{MAIN_PROTOCOL_R})"
        stable = scan.output_min == scan.output_max == main_target
        if scan.nan_inf:
            return "needs_fix_or_supplementary", "NaN/Inf detected in EAR ×4 outputs"
        if stable and scan.ratio_mean is not None and abs(scan.ratio_mean - MAIN_PROTOCOL_R) < 0.05:
            if role == "main_smoke":
                return "main_smoke_pass", f"Smoke EAR ×4: {input_pts}→{main_target}, ratio≈{MAIN_PROTOCOL_R}"
            return "main_compatible", f"EAR ×4 main: {input_pts}→{main_target}, ratio≈{MAIN_PROTOCOL_R}"
        return (
            "needs_fix_or_supplementary",
            f"EAR ×4 output mismatch: expected {main_target} pts ratio {MAIN_PROTOCOL_R}, "
            f"got min/max={scan.output_min}/{scan.output_max} ratio μ={scan.ratio_mean}",
        )

    if not scan.has_data:
        if method in MAIN_METHODS - {"EAR"}:
            return (
                "pending_generation",
                f"No ModelNet40 outputs yet; main protocol R=×{MAIN_PROTOCOL_R}, target={main_target}",
            )
        return "no_data", "Output directory empty or missing"

    stable = scan.output_min == scan.output_max
    if not stable:
        return "needs_fix_or_supplementary", f"Output point count varies: min={scan.output_min} max={scan.output_max}"
    if scan.nan_inf:
        return "needs_fix_or_supplementary", "NaN/Inf detected in outputs"

    ratio = scan.ratio_mean
    out_pts = scan.output_min

    if method == "EAR" and line == "B" and subdir == "ear" and out_pts == 1024:
        return "ablation_preliminary_x2", "Legacy Line B EAR ×2 (512→1024); superseded by ear_x4 → 2048"
    if stable and out_pts == main_target and ratio is not None and abs(ratio - MAIN_PROTOCOL_R) < 0.05:
        return "main_compatible", f"Stable ×{MAIN_PROTOCOL_R}: {input_pts}→{out_pts}"
    if stable and ratio is not None:
        return "needs_fix_or_supplementary", f"Stable output {out_pts} pts but ratio={ratio:.4f} ≠ main R={MAIN_PROTOCOL_R}"
    return "needs_fix_or_supplementary", "Unable to verify ratio stability"


def build_rows_with_cache(kitti_cache: dict[str, KittiTulipRef | None]) -> list[dict]:
    rows = []
    for entry in METHOD_REGISTRY:
        line = entry["line"]
        subdir = entry["output_subdir"]
        parent = "modelnet40_original_up" if line == "A" else "modelnet40_downsampled50_up"
        output_root = PROJECT_ROOT / "datasets" / parent / subdir
        input_pts = expected_input_points(line, entry["input_dataset"])
        scan = scan_output_dir(output_root, input_pts)
        status, notes = classify_status(
            entry["method"],
            line,
            subdir,
            scan,
            input_pts,
            entry.get("protocol_role", ""),
            kitti_cache.get(line),
        )
        if not scan.has_data and entry["method"] in MAIN_METHODS and entry["method"] != "EAR":
            notes += f"; main protocol R=×{MAIN_PROTOCOL_R}, target={MAIN_PROTOCOL_TARGET[line]}"

        row = {
            "method_group": entry["method_group"],
            "line": f"Line {line}",
            "method": entry["method"],
            "input_dataset": entry["input_dataset"],
            "output_dataset": str(output_root.relative_to(PROJECT_ROOT)),
            "main_protocol_R": MAIN_PROTOCOL_R,
            "main_protocol_target_points": MAIN_PROTOCOL_TARGET[line],
            "train_count": scan.train_count,
            "test_count": scan.test_count,
            "input_points": input_pts,
            "output_min_points": scan.output_min if scan.has_data else "",
            "output_max_points": scan.output_max if scan.has_data else "",
            "output_mean_points": f"{scan.output_mean:.1f}" if scan.output_mean is not None else "",
            "actual_ratio_min": f"{scan.ratio_min:.4f}" if scan.ratio_min is not None else "",
            "actual_ratio_max": f"{scan.ratio_max:.4f}" if scan.ratio_max is not None else "",
            "actual_ratio_mean": f"{scan.ratio_mean:.4f}" if scan.ratio_mean is not None else "",
            "configured_ratio": entry["configured_ratio"],
            "ratio_source_file": entry["ratio_source_file"],
            "status": status,
            "notes": notes,
            "tags": TULIP_TAGS if entry["method"] == "TULIP" else "",
            "_scripts": entry.get("scripts", ""),
            "_jobs": entry.get("jobs", ""),
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "method_group", "line", "method", "input_dataset", "output_dataset",
        "main_protocol_R", "main_protocol_target_points",
        "train_count", "test_count", "input_points",
        "output_min_points", "output_max_points", "output_mean_points",
        "actual_ratio_min", "actual_ratio_max", "actual_ratio_mean",
        "configured_ratio", "ratio_source_file", "status", "tags", "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _table_row(r: dict) -> str:
    out_str = (
        f"{r['output_min_points']}/{r['output_max_points']}/{r['output_mean_points']}"
        if r["output_min_points"] != ""
        else "—"
    )
    ratio_str = (
        f"{r['actual_ratio_min']}–{r['actual_ratio_max']} (μ={r['actual_ratio_mean']})"
        if r["actual_ratio_mean"] != ""
        else "—"
    )
    return (
        f"| {r['line']} | {r['method']} | `{r['output_dataset']}` | {r['input_points']} "
        f"| {out_str} | {ratio_str} | {r['configured_ratio']} | **{r['status']}** |"
    )


def write_md(rows: list[dict], path: Path, kitti_cache: dict[str, KittiTulipRef | None]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    main_rows = [r for r in rows if r["method_group"] == "main"]
    supp_rows = [r for r in rows if r["method_group"] == "supplementary"]

    lines = [
        "# ModelNet40 Upsampling Ratio Audit",
        "",
        f"- Generated at: {ts}",
        f"- Project: `{PROJECT_ROOT}`",
        "",
        "## Main Methods (xyz upsampling)",
        "",
        "EAR, PU-Net, PU-GCN, PDANS — main protocol **R=×4**.",
        "",
        "- Line A upsampling target: **4096** (1024 → ×4)",
        "- Line B upsampling target: **2048** (512 → ×4)",
        "",
        "| Line | Method | Dataset | Input | Target | Output (min/max/mean) | Actual ratio | Configured | Status |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for r in main_rows:
        out_str = (
            f"{r['output_min_points']}/{r['output_max_points']}/{r['output_mean_points']}"
            if r["output_min_points"] != ""
            else "—"
        )
        ratio_str = (
            f"{r['actual_ratio_min']}–{r['actual_ratio_max']} (μ={r['actual_ratio_mean']})"
            if r["actual_ratio_mean"] != ""
            else "—"
        )
        lines.append(
            f"| {r['line']} | {r['method']} | `{r['output_dataset']}` | {r['input_points']} "
            f"| {r['main_protocol_target_points']} | {out_str} | {ratio_str} | {r['configured_ratio']} | **{r['status']}** |"
        )

    lines.extend([
        "",
        "## Supplementary / Special Method",
        "",
        "### TULIP",
        "",
        f"**Tags:** {TULIP_TAGS}",
        "",
        "TULIP is **not** grouped with EAR / PU-Net / PU-GCN / PDANS for main-protocol ratio alignment.",
        "It uses KITTI range-image upsampling (16×1024 low-res → 64×1024 high-res); theoretical image upsampling is **×4**.",
        "",
        "| Line | Method | Dataset | Input | Output (min/max/mean) | Actual ratio | Configured | Status |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ])
    lines.extend(_table_row(r) for r in supp_rows)

    # KITTI supplementary reference block
    k_a = kitti_cache.get("A")
    k_b = kitti_cache.get("B")
    lines.extend([
        "",
        "#### ModelNet40 TULIP scan result",
        "",
        "- `datasets/modelnet40_original_up/tulip/` — **not created** (0 `.npy` / `.ply` / `.bin`)",
        "- `datasets/modelnet40_downsampled50_up/tulip/` — **not created** (0 `.npy` / `.ply` / `.bin`)",
        "- No ModelNet40 wrapper or generation job exists in `modelnet40_pointnet2_upsampling`.",
        "",
        "#### KITTI supplementary reference (xyz point counts, not ModelNet40)",
        "",
        "Used only to characterize TULIP xyz behavior when back-projected from range images.",
        "",
    ])
    if k_a:
        lines.append(
            f"- **Line A (KITTI original→tulip, n={k_a.n}):** input {k_a.input_min}/{k_a.input_max}/{k_a.input_mean:.0f} → "
            f"output {k_a.output_min}/{k_a.output_max}/{k_a.output_mean:.0f}; "
            f"ratio {k_a.ratio_min:.4f}–{k_a.ratio_max:.4f} (μ={k_a.ratio_mean:.4f}, σ={k_a.ratio_std:.4f})"
        )
    if k_b:
        lines.append(
            f"- **Line B (KITTI downsampled50→tulip, n={k_b.n}):** input {k_b.input_min}/{k_b.input_max}/{k_b.input_mean:.0f} → "
            f"output {k_b.output_min}/{k_b.output_max}/{k_b.output_mean:.0f}; "
            f"ratio {k_b.ratio_min:.4f}–{k_b.ratio_max:.4f} (μ={k_b.ratio_mean:.4f}, σ={k_b.ratio_std:.4f})"
        )

    lines.extend(["", "## Per-Method Notes", ""])
    for r in rows:
        tag_line = f"- Tags: {r['tags']}\n" if r.get("tags") else ""
        lines.extend([
            f"### {r['line']} — {r['method']} (`{r['output_dataset']}`)",
            "",
            f"- Group: **{r['method_group']}**",
            f"- Status: **{r['status']}**",
            tag_line.rstrip(),
            f"- Train/test file count: {r['train_count']} / {r['test_count']}",
            f"- Ratio source: `{r['ratio_source_file']}`",
            f"- Scripts: {r['_scripts']}",
            f"- Notes: {r['notes']}",
            "",
        ])

    lines.extend([
        "## Baseline Native Counts (no upsampling)",
        "",
        "| Line | Dataset | Native points |",
        "| --- | --- | ---: |",
        "| A | `datasets/modelnet40_original` | 1024 |",
        "| B | `datasets/modelnet40_downsampled50` | 512 |",
        "",
        "See `reports/modelnet40_upsampling_ratio_recommendation.md` for unified R=×4 main protocol.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    kitti_cache = {"A": kitti_tulip_reference("A"), "B": kitti_tulip_reference("B")}
    rows = build_rows_with_cache(kitti_cache)
    csv_path = REPORTS_DIR / "modelnet40_upsampling_ratio_audit.csv"
    md_path = REPORTS_DIR / "modelnet40_upsampling_ratio_audit.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, csv_path)
    write_md(rows, md_path, kitti_cache)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(json.dumps({f"{r['line']}_{r['method']}": r['status'] for r in rows if r['method'] == 'TULIP'}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
