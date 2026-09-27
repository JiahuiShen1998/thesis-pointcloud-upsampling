#!/usr/bin/env python3
"""Audit method source validity and identity for KITTI strict x4 outputs.

Read-only with respect to model outputs: this script only reads existing bins,
logs, manifests, and scripts, then writes audit reports.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
RESULT_ROOT = PROJECT_ROOT / "results" / "kitti_strict_x4_upsampling_protocol"
TRAINING_DIR = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
VAL_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
FULL_LOG_DIR = REPORT_DIR / "kitti_strict_x4_protocol_full_upsampling_logs"
SMOKE_LOG_DIR = REPORT_DIR / "kitti_strict_x4_protocol_method_smoke_logs"
FIX_LOG_DIR = REPORT_DIR / "kitti_strict_x4_protocol_method_runtime_fix_logs"

METHODS = ["EAR", "PU-Net", "PU-GCN", "PDANS", "PU-EdgeFormer"]
LINES = {
    "Line A": {
        "key": "lineA_original",
        "input": TRAINING_DIR / "velodyne_original_val",
        "output": RESULT_ROOT / "lineA_original_up_x4",
    },
    "Line B": {
        "key": "lineB_downsampled50",
        "input": TRAINING_DIR / "velodyne_downsampled_50_val",
        "output": RESULT_ROOT / "lineB_downsampled50_up_x4",
    },
}
RAW_SOURCES = {
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
FIXED_FRAMES = ["000001", "000093", "000242", "003219", "006833"]
RANDOM_SEED = 20260629


def read_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"{path} is not Nx4 float32")
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_frames() -> List[str]:
    return [x.strip() for x in VAL_SPLIT.read_text(encoding="utf-8").splitlines() if x.strip()]


def sample_frames() -> List[str]:
    frames = split_frames()
    rng = random.Random(RANDOM_SEED)
    remaining = [f for f in frames if f not in FIXED_FRAMES]
    random20 = sorted(rng.sample(remaining, 20))
    return FIXED_FRAMES + random20


def parse_log(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def output_path(line: str, method: str, frame: str) -> Path:
    return LINES[line]["output"] / method / f"{frame}.bin"


def input_path(line: str, frame: str) -> Path:
    return LINES[line]["input"] / f"{frame}.bin"


def raw_path_for(line: str, method: str, frame: str) -> str:
    if method in RAW_SOURCES:
        return str(RAW_SOURCES[method][line] / f"{frame}.bin")
    if method == "PU-EdgeFormer":
        smoke_raw = RESULT_ROOT / "method_smoke_raw" / LINES[line]["key"] / method / frame / f"{frame}_puedgeformer_raw.xyz"
        return str(smoke_raw) if smoke_raw.exists() else "not_persisted_for_full; full script samples 8192 input xyz"
    return "not_persisted; generated in adapter from input"


def point_count_from_file(path: Path, xyz: bool = False) -> int | str:
    if not path.exists():
        return "missing"
    if path.suffix == ".xyz" or xyz:
        try:
            arr = np.loadtxt(path, dtype=np.float32)
            return int(np.asarray(arr).reshape(-1, 3).shape[0])
        except Exception as exc:
            return f"read_error:{exc!r}"
    if path.stat().st_size % 16:
        return "malformed"
    return path.stat().st_size // 16


def inferred_adapter(method: str, line: str, frame: str, raw_path: str) -> str:
    if method == "EAR":
        return "EAR-style full adapter raw"
    if method == "PU-EdgeFormer":
        return "PU-EdgeFormer smoke-verified seeded patch adapter raw"
    return f"existing_{method}_raw" if Path(raw_path).exists() else f"missing historical {method} raw; input fallback adapter raw"


def inferred_normalization(method: str) -> str:
    if method == "EAR":
        return "deterministic_sample_down_to_4N_seeded"
    return "deterministic_knn_interpolation_fill_to_4N_seeded"


def source_rows(frames: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    full_script = (PROJECT_ROOT / "scripts" / "kitti_strict_x4_full_upsampling.py").read_text(encoding="utf-8")
    smoke_script = (PROJECT_ROOT / "scripts" / "kitti_strict_x4_method_smoke.py").read_text(encoding="utf-8")
    protocol_script = (PROJECT_ROOT / "scripts" / "kitti_strict_x4_protocol.py").read_text(encoding="utf-8")
    wrapper_names = [
        "run_ear_strict_x4.py",
        "run_punet_strict_x4.py",
        "run_pugcn_strict_x4.py",
        "run_pdans_strict_x4.py",
        "run_puedgeformer_strict_x4.py",
    ]
    wrapper_dirs = [PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "tools"]
    wrappers = {name: [d / name for d in wrapper_dirs if (d / name).exists()] for name in wrapper_names}
    for method in METHODS:
        method_logs = list((FULL_LOG_DIR / method).glob("*/*.log"))
        raw_adapters = sorted({parse_log(p).get("raw_adapter", "") for p in method_logs[:200] if parse_log(p).get("raw_adapter")})
        norms = sorted({parse_log(p).get("normalization", "") for p in method_logs[:200] if parse_log(p).get("normalization")})
        wrapper_name = {
            "EAR": "run_ear_strict_x4.py",
            "PU-Net": "run_punet_strict_x4.py",
            "PU-GCN": "run_pugcn_strict_x4.py",
            "PDANS": "run_pdans_strict_x4.py",
            "PU-EdgeFormer": "run_puedgeformer_strict_x4.py",
        }[method]
        if method == "EAR":
            status = "ADAPTER_FROM_INPUT_NOT_OFFICIAL_EAR"
            checkpoint = "not_required"
            method_inference = "no official EAR script call in full runner; uses ear_fast_raw"
            fallback = "same shared normalizer; raw is adapter-generated from input"
            proof = "no, final geometry is adapter geometry"
        elif method == "PU-EdgeFormer":
            status = "ADAPTER_FROM_INPUT_PATCH_NOT_FULL_METHOD"
            checkpoint = "smoke script has model-100 path; full runner does not load checkpoint"
            method_inference = "full runner does not call PU-EdgeFormer; samples 8192 input xyz"
            fallback = "same shared normalizer fills most points from 8192 sampled input xyz"
            proof = "no for full outputs; smoke only has method raw for 5 frames"
        else:
            status = "EXISTING_RAW_RENORMALIZED_NOT_STRICT_METHOD_INFERENCE"
            checkpoint = "not loaded by strict full runner"
            method_inference = "no method invocation in strict full runner; existing historical raw is read"
            fallback = "same shared normalizer fills/crops to 4N"
            proof = "only inherits historical raw if raw is valid; strict final mostly normalizer fill when raw < 4N"
        for line in LINES:
            for frame in frames[:5]:
                log = parse_log(FULL_LOG_DIR / method / LINES[line]["key"] / f"{frame}.log")
                raw_path = raw_path_for(line, method, frame)
                raw_count = log.get("raw_points", "")
                if not raw_count and raw_path and not raw_path.startswith("not_"):
                    raw_count = point_count_from_file(Path(raw_path))
                if not raw_count and method == "EAR":
                    inp_count = point_count_from_file(input_path(line, frame))
                    raw_count = int(inp_count) * 4 if isinstance(inp_count, int) else ""
                if not raw_count and method == "PU-EdgeFormer":
                    raw_count = 8192
                adapter_input = log.get("raw_adapter") or inferred_adapter(method, line, frame, raw_path)
                normalization = log.get("normalization") or inferred_normalization(method)
                rows.append(
                    {
                        "method": method,
                        "line": line,
                        "frame": frame,
                        "source_status": status,
                        "method_specific_inference_called_in_strict_full": method_inference,
                        "checkpoint_loaded_in_strict_full": checkpoint,
                        "method_specific_raw_output": "historical_existing_raw" if method in RAW_SOURCES else "adapter_raw",
                        "raw_output_path": raw_path,
                        "raw_output_points": raw_count,
                        "adapter_input": adapter_input,
                        "final_generation": normalization,
                        "fallback_to_generic_interpolation": "yes" if "fill_to_4N" in normalization or method != "EAR" else "shared normalizer/crop used",
                        "same_fallback_as_other_methods": "yes; normalize_to_strict_x4 in kitti_strict_x4_full_upsampling.py",
                        "proof_final_contains_method_geometry": proof,
                        "full_log_path": str(FULL_LOG_DIR / method / LINES[line]["key"] / f"{frame}.log"),
                        "wrapper_file_expected": wrapper_name,
                        "wrapper_file_found": ";".join(str(p) for p in wrappers[wrapper_name]) or "missing",
                        "evidence_scripts": "full_script:raw_for_method/normalize_to_strict_x4; smoke_script:run_existing_raw_method; protocol_script:METHOD_AUDIT",
                        "script_evidence_present": all(s in full_script for s in ["raw_for_method", "normalize_to_strict_x4", "PASS_WITH_ADAPTER"]) and "run_existing_raw_method" in smoke_script and "METHOD_AUDIT" in protocol_script,
                    }
                )
    return rows


def stats(points: np.ndarray) -> Dict[str, object]:
    xyz = points[:, :3]
    sample = xyz
    if len(sample) > 200000:
        idx = np.linspace(0, len(sample) - 1, 200000).astype(np.int64)
        sample = sample[idx]
    unique = len(np.unique(np.round(sample, 4), axis=0)) if len(sample) else 0
    return {
        "points": int(len(points)),
        "xyz_mean": np.mean(xyz, axis=0).round(6).tolist(),
        "xyz_std": np.std(xyz, axis=0).round(6).tolist(),
        "xyz_min": np.min(xyz, axis=0).round(6).tolist(),
        "xyz_max": np.max(xyz, axis=0).round(6).tolist(),
        "duplicate_ratio_rounded4_sample": float(1.0 - unique / len(sample)) if len(sample) else 1.0,
    }


def sample_xyz(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    xyz = points[:, :3]
    if len(xyz) <= n:
        return xyz
    rng = np.random.default_rng(seed)
    return xyz[rng.choice(len(xyz), size=n, replace=False)]


def dist_summary(d: np.ndarray, prefix: str) -> Dict[str, object]:
    if len(d) == 0:
        return {f"{prefix}_{k}": "" for k in ["mean", "median", "p95", "max"]}
    return {
        f"{prefix}_mean": float(np.mean(d)),
        f"{prefix}_median": float(np.median(d)),
        f"{prefix}_p95": float(np.percentile(d, 95)),
        f"{prefix}_max": float(np.max(d)),
    }


def nn_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cKDTree(b).query(a, k=1, workers=-1)[0]


def identity_rows(frames: Sequence[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str], List[str]]:
    file_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []
    identical: List[str] = []
    near_identical: List[str] = []
    for line in LINES:
        for frame in frames:
            input_pts = read_bin(input_path(line, frame))
            input_sample = sample_xyz(input_pts, 8192, RANDOM_SEED + int(frame))
            loaded: Dict[str, np.ndarray] = {}
            samples: Dict[str, np.ndarray] = {}
            for method in METHODS:
                path = output_path(line, method, frame)
                pts = read_bin(path)
                loaded[method] = pts
                samples[method] = sample_xyz(pts, 8192, RANDOM_SEED + int(frame) + METHODS.index(method) * 101)
                s = stats(pts)
                d_in = nn_a_to_b(samples[method], input_sample)
                nonzero = d_in[d_in > 1e-5]
                row = {
                    "line": line,
                    "frame": frame,
                    "method": method,
                    "path": str(path),
                    "sha256": sha256(path),
                    **s,
                    **dist_summary(d_in, "final_to_input_nn_sample"),
                    **dist_summary(nonzero, "generated_like_to_input_nn_sample"),
                    "sample_exact_or_near_input_ratio_1e-5": float(np.mean(d_in <= 1e-5)),
                }
                file_rows.append(row)
            for a, b in itertools.combinations(METHODS, 2):
                exact = file_rows[-len(METHODS) + METHODS.index(a)]["sha256"] == file_rows[-len(METHODS) + METHODS.index(b)]["sha256"]
                da = nn_a_to_b(samples[a], samples[b])
                db = nn_a_to_b(samples[b], samples[a])
                chamfer = float(np.mean(da) + np.mean(db)) / 2.0
                pair_key = f"{line}|{frame}|{a}|{b}"
                if exact:
                    identical.append(pair_key)
                if (not exact) and chamfer < 1e-5 and np.percentile(da, 95) < 1e-4 and np.percentile(db, 95) < 1e-4:
                    near_identical.append(pair_key)
                pair_rows.append(
                    {
                        "line": line,
                        "frame": frame,
                        "method_a": a,
                        "method_b": b,
                        "sha256_equal": exact,
                        "chamfer_like_sample": chamfer,
                        **dist_summary(da, "a_to_b_nn"),
                        **dist_summary(db, "b_to_a_nn"),
                        "near_identical_threshold_hit": pair_key in near_identical,
                    }
                )
    return file_rows, pair_rows, identical, near_identical


def write_markdown(source: List[Dict[str, object]], identity: List[Dict[str, object]], pairs: List[Dict[str, object]], identical: List[str], near: List[str], frames: Sequence[str]) -> None:
    by_method: Dict[str, List[Dict[str, object]]] = {m: [r for r in source if r["method"] == m] for m in METHODS}
    src_md = [
        "# KITTI Strict x4 Method Source Validity Audit",
        "",
        f"- Sample frames for source rows: `{', '.join(frames[:5])}`",
        "- Full runner audited: `scripts/kitti_strict_x4_full_upsampling.py`",
        "- Detector started: `NO`",
        "",
        "## Verdict",
    ]
    for method in METHODS:
        first = by_method[method][0]
        src_md.extend(
            [
                f"### {method}",
                f"- Source status: `{first['source_status']}`",
                f"- Method-specific inference: `{first['method_specific_inference_called_in_strict_full']}`",
                f"- Checkpoint loaded: `{first['checkpoint_loaded_in_strict_full']}`",
                f"- Adapter input: `{first['adapter_input']}`",
                f"- Final generation: `{first['final_generation']}`",
                f"- Generic fallback: `{first['fallback_to_generic_interpolation']}`",
                f"- Method-geometry proof: `{first['proof_final_contains_method_geometry']}`",
                f"- Wrapper found: `{first['wrapper_file_found']}`",
                "",
            ]
        )
    src_md.extend(
        [
            "## Conclusion",
            "The final strict x4 bins pass point-count audit, but they are adapter-normalized products. The strict full runner does not prove full-frame method-specific inference for any of the five methods.",
        ]
    )
    (REPORT_DIR / "kitti_strict_x4_method_source_validity_audit.md").write_text("\n".join(src_md) + "\n", encoding="utf-8")

    id_md = [
        "# KITTI Strict x4 Method Identity Audit",
        "",
        f"- Frames: `{', '.join(frames)}`",
        f"- File rows: `{len(identity)}`",
        f"- Pair rows: `{len(pairs)}`",
        f"- Identical method pairs: `{len(identical)}`",
        f"- Near-identical method pairs: `{len(near)}`",
        "",
        "## Pairwise Summary",
    ]
    if identical:
        id_md.append("- Exact identical pairs found; see CSV.")
    else:
        id_md.append("- No exact identical file checksums among sampled method pairs.")
    if near:
        id_md.append("- Near-identical geometry pairs found under strict threshold; see CSV.")
    else:
        id_md.append("- No near-identical sampled geometry pairs under strict threshold.")
    id_md.extend(
        [
            "",
            "## Caveat",
            "Distinct checksums/geometries do not prove method validity because the source audit shows shared normalization and adapter-generated or historical raw inputs.",
        ]
    )
    (REPORT_DIR / "kitti_strict_x4_method_identity_audit.md").write_text("\n".join(id_md) + "\n", encoding="utf-8")


def main() -> None:
    frames = sample_frames()
    src = source_rows(frames)
    identity, pairs, identical, near = identity_rows(frames)
    write_csv(REPORT_DIR / "kitti_strict_x4_method_source_validity_audit.csv", src)
    write_csv(REPORT_DIR / "kitti_strict_x4_method_identity_audit.csv", identity + pairs)
    write_markdown(src, identity, pairs, identical, near, frames)
    print(f"frames={','.join(frames)}")
    print(f"source_rows={len(src)}")
    print(f"identity_file_rows={len(identity)}")
    print(f"identity_pair_rows={len(pairs)}")
    print(f"identical_pairs={len(identical)}")
    print(f"near_identical_pairs={len(near)}")


if __name__ == "__main__":
    main()
