#!/usr/bin/env python3
"""ModelNet40 PU-EdgeFormer lab-params smoke (Line B + Line A A1/A2).

Protocol:
  Line B:  256 -> x4 -> exact 1024 (direct whole-object patch)
  Line A1: 1024 -> x4 -> exact 4096 (direct whole-object patch)
  Line A2: 1024 -> deterministic 4x256 chunks -> each x4 -> merge 4096

Constraints:
  - XYZ only (no KITTI intensity reattach)
  - strict deterministic crop if raw > target
  - FAIL if raw < target (no padding / jitter / interpolation / fake points)
  - no detector / classifier / KITTI AP eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG = PROJECT_ROOT / "imports" / "transfer_pu_edgeformer_to_hpc_20260716"
REPO_DIR = PROJECT_ROOT / "external" / "pu_edgeformer_ops_reuse"
CKPT_DIR = REPO_DIR / "checkpoint_model100"
WRAPPER_MANY = PKG / "wrappers" / "tf_pugcn_family_patch_infer_many.py"
TF_PYTHON = Path.home() / ".conda" / "envs" / "tf15_upsampling" / "bin" / "python"

ORIGINAL_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_original"
DOWNSAMPLED_X4_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled_x4"

OUT_LINE_A = PROJECT_ROOT / "datasets" / "modelnet40_original_x4_up" / "pu_edgeformer_labparams_smoke_20260716"
OUT_LINE_B = PROJECT_ROOT / "datasets" / "modelnet40_downsampled_x4_up" / "pu_edgeformer_labparams_smoke_20260716"

PACKAGE_PATH = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/transfer_pu_edgeformer_to_hpc_20260716.tar.gz")
PACKAGE_SHA256 = "2da8f2811b5529dc1c9de92d92372a505da5702af54429e41546a5bb31110d84"

INFER_SEED = 20260702
UP_RATIO = 4
METHOD = "PU-EdgeFormer"
MODEL = "edgetransformer"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def stable_seed(*parts: object) -> int:
    import hashlib

    token = ":".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.md5(token).hexdigest()
    return INFER_SEED + (int(digest[:8], 16) % 1_000_000)


def strict_crop_xyz(raw: np.ndarray, target: int, seed: int) -> tuple[np.ndarray, str]:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"raw must be Nx3, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("raw contains NaN/Inf")
    n = int(raw.shape[0])
    if n == target:
        return raw.copy(), "exact"
    if n > target:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=target, replace=False)
        return raw[idx].copy(), "strict_deterministic_crop"
    raise ValueError(
        f"raw shortfall: raw={n} < target={target}; padding/jitter/interpolation/fake points forbidden"
    )


def list_samples(root: Path, split: str, limit: int) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    split_dir = root / split
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for npy in sorted(cls_dir.glob("*.npy")):
            rows.append((cls_dir.name, npy.stem, npy))
            if len(rows) >= limit:
                return rows
    return rows


def write_patch_job(
    *,
    work_dir: Path,
    frame_id: str,
    patches: list[np.ndarray],
    patch_input: int,
) -> dict:
    patch_dir = work_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    meta_patches = []
    for i, patch in enumerate(patches):
        arr = np.asarray(patch, dtype=np.float32)
        if arr.shape != (patch_input, 3):
            raise ValueError(f"patch {i} shape {arr.shape} != ({patch_input}, 3)")
        if not np.isfinite(arr).all():
            raise ValueError(f"patch {i} NaN/Inf")
        fname = f"patch_{i:06d}.npy"
        np.save(patch_dir / fname, arr)
        meta_patches.append({"file": fname, "index": i})
    meta = {
        "frame_id": frame_id,
        "output_patch_dir": str(patch_dir),
        "patch_input_points": patch_input,
        "patches": meta_patches,
        "no_coordinate_modification": True,
        "dataset": "ModelNet40",
        "xyz_only": True,
    }
    meta_path = work_dir / "patch_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    raw_patch_dir = work_dir / "raw_patches"
    merged_raw = work_dir / "merged_raw.npy"
    prov = work_dir / "merged_raw_provenance.json"
    return {
        "frame_id": frame_id,
        "patch_metadata_json": str(meta_path),
        "raw_patch_output_dir": str(raw_patch_dir),
        "merged_raw_output": str(merged_raw),
        "merged_raw_provenance_json": str(prov),
    }


def chunk_1024_to_4x256(points: np.ndarray) -> list[np.ndarray]:
    """Deterministic contiguous index split: [0:256),[256:512),[512:768),[768:1024)."""
    if points.shape != (1024, 3):
        raise ValueError(f"A2 expects (1024,3), got {points.shape}")
    return [points[i * 256 : (i + 1) * 256].copy() for i in range(4)]


def run_infer_many(
    *,
    jobs: list[dict],
    jobs_json: Path,
    line: str,
    patch_input: int,
    patch_output: int,
    log_path: Path,
    gpu_memory_fraction: float,
) -> dict:
    jobs_json.parent.mkdir(parents=True, exist_ok=True)
    jobs_json.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    cmd = [
        str(TF_PYTHON),
        str(WRAPPER_MANY),
        "--repo_dir",
        str(REPO_DIR),
        "--jobs_json",
        str(jobs_json),
        "--method",
        METHOD,
        "--model",
        MODEL,
        "--checkpoint_dir",
        str(CKPT_DIR),
        "--line",
        line,
        "--patch_input_points",
        str(patch_input),
        "--patch_output_points",
        str(patch_output),
        "--up_ratio",
        str(UP_RATIO),
        "--seed",
        str(INFER_SEED),
        "--gpu_memory_fraction",
        str(gpu_memory_fraction),
    ]
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "CMD:\n"
        + " ".join(cmd)
        + "\n\nSTDOUT:\n"
        + proc.stdout
        + "\n\nSTDERR:\n"
        + proc.stderr
        + f"\n\nreturncode={proc.returncode}\nelapsed_sec={elapsed:.3f}\n",
        encoding="utf-8",
    )
    summary = {
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    # Parse last JSON status line if present
    for line_txt in reversed(proc.stdout.splitlines()):
        line_txt = line_txt.strip()
        if line_txt.startswith("{") and "status" in line_txt:
            try:
                summary["wrapper_summary"] = json.loads(line_txt)
                break
            except json.JSONDecodeError:
                pass
    return summary


def finalize_sample(
    *,
    job: dict,
    input_path: Path,
    out_root: Path,
    split: str,
    class_name: str,
    shape_id: str,
    mode: str,
    target: int,
    expected_raw: int,
) -> dict:
    work_merged = Path(job["merged_raw_output"])
    work_prov = Path(job["merged_raw_provenance_json"])
    # Mode-specific finals under outputs/<mode>/{train,test}/... to avoid A1/A2 collisions.
    # Also mirror into top-level train/test only for Line B (single mode).
    final_path = out_root / "outputs" / mode / split / class_name / f"{shape_id}.npy"
    if mode.startswith("lineB_"):
        final_path = out_root / split / class_name / f"{shape_id}.npy"
    method_raw_path = out_root / "method_raw" / mode / split / class_name / f"{shape_id}.npy"
    audit_path = out_root / "audits" / mode / split / class_name / f"{shape_id}.json"

    record = {
        "mode": mode,
        "split": split,
        "class_name": class_name,
        "shape_id": shape_id,
        "input_path": str(input_path),
        "target_points": target,
        "expected_raw_points": expected_raw,
        "status": "FAIL",
        "method_inference_called": False,
        "checkpoint_loaded": False,
        "raw_points": None,
        "final_points": None,
        "dtype": None,
        "finite": None,
        "crop_action": None,
        "raw_shortfall": False,
        "padding_used": False,
        "jitter_used": False,
        "interpolation_used": False,
        "fake_points_used": False,
        "error": "",
        "runtime_infer_sec": None,
        "final_output": str(final_path),
        "method_raw_output": str(method_raw_path),
        "merged_raw_provenance": str(work_prov),
    }
    try:
        if not work_prov.is_file():
            raise FileNotFoundError(f"missing provenance: {work_prov}")
        prov = json.loads(work_prov.read_text(encoding="utf-8"))
        record["method_inference_called"] = bool(prov.get("method_inference_called"))
        record["checkpoint_loaded"] = bool(prov.get("checkpoint_loaded"))
        record["runtime_infer_sec"] = prov.get("runtime_method_inference_sec")
        if prov.get("status") != "PASS":
            raise RuntimeError(f"wrapper provenance status={prov.get('status')} notes={prov.get('notes')}")
        if not record["method_inference_called"]:
            raise RuntimeError("method_inference_called is not true")
        if not record["checkpoint_loaded"]:
            raise RuntimeError("checkpoint_loaded is not true")
        if not work_merged.is_file():
            raise FileNotFoundError(f"missing merged raw: {work_merged}")
        raw = np.load(work_merged).astype(np.float32)
        record["raw_points"] = int(raw.shape[0])
        record["dtype"] = str(raw.dtype)
        record["finite"] = bool(np.isfinite(raw).all())
        if not record["finite"]:
            raise ValueError("merged raw NaN/Inf")
        if raw.shape[0] < target:
            record["raw_shortfall"] = True
            raise ValueError(f"raw shortfall raw={raw.shape[0]} target={target}")
        crop_seed = stable_seed(mode, split, class_name, shape_id, "strict_crop")
        final, action = strict_crop_xyz(raw, target, crop_seed)
        record["crop_action"] = action
        record["final_points"] = int(final.shape[0])
        if final.shape != (target, 3):
            raise ValueError(f"final shape {final.shape} != ({target}, 3)")
        # Save outputs
        method_raw_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(method_raw_path, raw)
        np.save(final_path, final)
        audit = {
            **record,
            "wrapper_provenance": prov,
            "constraints": {
                "no_padding": True,
                "no_jitter": True,
                "no_interpolation": True,
                "no_fake_points": True,
                "xyz_only": True,
                "no_kitti_intensity_reattach": True,
            },
            "detector_eval_started": False,
            "pointnet_classifier_started": False,
            "kitti_ap_eval_started": False,
            "timestamp_utc": utc_now(),
        }
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        record["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        record["error"] = str(exc)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def prepare_mode_jobs(
    *,
    mode: str,
    input_root: Path,
    out_root: Path,
    max_per_split: int,
    input_points: int,
    patch_input: int,
    chunked: bool,
) -> tuple[list[dict], list[dict]]:
    """Returns (jobs_for_wrapper, sample_meta_list)."""
    jobs: list[dict] = []
    metas: list[dict] = []
    work_root = out_root / "method_raw" / mode / "_work"
    for split in ("train", "test"):
        samples = list_samples(input_root, split, max_per_split)
        for class_name, shape_id, path in samples:
            pts = np.load(path).astype(np.float32)
            if pts.shape != (input_points, 3):
                raise ValueError(f"{path} shape {pts.shape} != ({input_points}, 3)")
            if not np.isfinite(pts).all():
                raise ValueError(f"{path} NaN/Inf")
            frame_id = f"{split}__{class_name}__{shape_id}"
            work_dir = work_root / split / class_name / shape_id
            if chunked:
                patches = chunk_1024_to_4x256(pts)
            else:
                patches = [pts.copy()]
            job = write_patch_job(
                work_dir=work_dir,
                frame_id=frame_id,
                patches=patches,
                patch_input=patch_input,
            )
            jobs.append(job)
            metas.append(
                {
                    "job": job,
                    "input_path": path,
                    "split": split,
                    "class_name": class_name,
                    "shape_id": shape_id,
                    "mode": mode,
                    "patch_count": len(patches),
                }
            )
    return jobs, metas


def run_mode(
    *,
    mode: str,
    input_root: Path,
    out_root: Path,
    max_per_split: int,
    input_points: int,
    target_points: int,
    patch_input: int,
    patch_output: int,
    chunked: bool,
    gpu_memory_fraction: float,
) -> dict:
    t0 = time.perf_counter()
    jobs, metas = prepare_mode_jobs(
        mode=mode,
        input_root=input_root,
        out_root=out_root,
        max_per_split=max_per_split,
        input_points=input_points,
        patch_input=patch_input,
        chunked=chunked,
    )
    jobs_json = out_root / "manifests" / f"{mode}_jobs.json"
    log_path = out_root / "logs" / f"{mode}_infer.log"
    infer = run_infer_many(
        jobs=jobs,
        jobs_json=jobs_json,
        line=mode,
        patch_input=patch_input,
        patch_output=patch_output,
        log_path=log_path,
        gpu_memory_fraction=gpu_memory_fraction,
    )
    expected_raw = patch_output * (4 if chunked else 1)
    records = []
    for meta in metas:
        rec = finalize_sample(
            job=meta["job"],
            input_path=meta["input_path"],
            out_root=out_root,
            split=meta["split"],
            class_name=meta["class_name"],
            shape_id=meta["shape_id"],
            mode=mode,
            target=target_points,
            expected_raw=expected_raw,
        )
        records.append(rec)

    manifest_csv = out_root / "manifests" / f"{mode}_smoke_manifest.csv"
    fields = [
        "mode",
        "split",
        "class_name",
        "shape_id",
        "status",
        "raw_points",
        "final_points",
        "crop_action",
        "method_inference_called",
        "checkpoint_loaded",
        "finite",
        "raw_shortfall",
        "runtime_infer_sec",
        "error",
        "input_path",
        "final_output",
        "method_raw_output",
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    n_pass = sum(1 for r in records if r["status"] == "PASS")
    n_fail = len(records) - n_pass
    any_nan = any(r.get("finite") is False for r in records)
    any_short = any(bool(r.get("raw_shortfall")) for r in records)
    real_calls = all(bool(r.get("method_inference_called")) for r in records) and n_pass == len(records)
    # real_calls for PASS samples only if some failed at wrapper
    real_calls = all(bool(r.get("method_inference_called")) for r in records if r["status"] == "PASS")
    if n_pass == 0:
        real_calls = False

    summary = {
        "mode": mode,
        "status": "PASS" if n_fail == 0 and infer.get("returncode") == 0 else "FAIL",
        "n_samples": len(records),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "any_nan_inf": any_nan,
        "any_raw_shortfall": any_short,
        "padding_used": False,
        "jitter_used": False,
        "interpolation_used": False,
        "fake_points_used": False,
        "method_inference_called_on_pass": real_calls,
        "patch_input_points": patch_input,
        "patch_output_points": patch_output,
        "target_points": target_points,
        "chunked": chunked,
        "infer": infer,
        "elapsed_total_sec": time.perf_counter() - t0,
        "manifest_csv": str(manifest_csv),
        "records": records,
    }
    (out_root / "audits" / f"{mode}_summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def write_report(summaries: dict[str, dict], env_info: dict) -> Path:
    report = PROJECT_ROOT / "reports" / "pu_edgeformer_modelnet40_labparams_smoke_20260716.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    b = summaries["lineB_direct_256_to_1024"]
    a1 = summaries["lineA_A1_direct_1024_to_4096"]
    a2 = summaries["lineA_A2_4x256_chunks"]

    # Recommendation rule
    if a1["status"] == "PASS" and not a1["any_nan_inf"] and not a1["any_raw_shortfall"]:
        rec_mode = "A1_direct_1024_to_4096"
        rec_reason = "A1 direct whole-object inference PASS and finite; prefers whole-object context."
    else:
        rec_mode = "A2_deterministic_4x256_chunks"
        rec_reason = (
            f"A1 status={a1['status']} nan_inf={a1['any_nan_inf']} shortfall={a1['any_raw_shortfall']}; "
            f"fallback to A2 (status={a2['status']})."
        )

    lines = [
        "# PU-EdgeFormer ModelNet40 Lab-Params Smoke Report",
        "",
        f"- Generated: `{utc_now()}`",
        f"- project path: `{PROJECT_ROOT}`",
        f"- package path: `{PACKAGE_PATH}`",
        f"- package SHA256: `{PACKAGE_SHA256}`",
        f"- checkpoint used: `{CKPT_DIR}` (model-100; HPC-relative checkpoint pointer)",
        f"- config used: `{CKPT_DIR / 'args.txt'}` + `{PKG / 'config' / 'configs.py'}`",
        f"- wrapper used: `{WRAPPER_MANY}`",
        f"- repo_dir (code_snapshot + PU-GCN tf_ops reuse): `{REPO_DIR}`",
        f"- environment used: `{env_info.get('env_name')}` / `{TF_PYTHON}`",
        f"- python: `{env_info.get('python')}`",
        f"- tensorflow: `{env_info.get('tensorflow')}`",
        f"- gpu: `{env_info.get('gpu')}`",
        f"- inference seed: `{INFER_SEED}`",
        "",
        "## Dataset audit (read-only)",
        "",
        f"- original 1024 train: `{ORIGINAL_ROOT / 'train'}` count=9843",
        f"- original 1024 test: `{ORIGINAL_ROOT / 'test'}` count=2468",
        f"- downsampled-x4 256 train: `{DOWNSAMPLED_X4_ROOT / 'train'}` count=9843",
        f"- downsampled-x4 256 test: `{DOWNSAMPLED_X4_ROOT / 'test'}` count=2468",
        "",
        "## Smoke outputs (new dirs, no overwrite of old results)",
        "",
        f"- Line A: `{OUT_LINE_A}`",
        f"- Line B: `{OUT_LINE_B}`",
        "",
        "## Line B smoke (direct_256_to_1024)",
        "",
        f"- status: **{b['status']}**",
        f"- samples: {b['n_pass']}/{b['n_samples']} PASS",
        f"- any NaN/Inf: {b['any_nan_inf']}",
        f"- any raw shortage: {b['any_raw_shortfall']}",
        f"- padding/jitter/interpolation/fake points: NO",
        f"- runtime total sec: {b['elapsed_total_sec']:.3f}",
        f"- manifest: `{b['manifest_csv']}`",
        "",
        "## Line A A1 smoke (direct_1024_to_4096)",
        "",
        f"- status: **{a1['status']}**",
        f"- samples: {a1['n_pass']}/{a1['n_samples']} PASS",
        f"- any NaN/Inf: {a1['any_nan_inf']}",
        f"- any raw shortage: {a1['any_raw_shortfall']}",
        f"- padding/jitter/interpolation/fake points: NO",
        f"- runtime total sec: {a1['elapsed_total_sec']:.3f}",
        f"- manifest: `{a1['manifest_csv']}`",
        "",
        "## Line A A2 smoke (deterministic_4x256_chunks)",
        "",
        f"- status: **{a2['status']}**",
        f"- samples: {a2['n_pass']}/{a2['n_samples']} PASS",
        f"- any NaN/Inf: {a2['any_nan_inf']}",
        f"- any raw shortage: {a2['any_raw_shortfall']}",
        f"- padding/jitter/interpolation/fake points: NO",
        f"- runtime total sec: {a2['elapsed_total_sec']:.3f}",
        f"- manifest: `{a2['manifest_csv']}`",
        "",
        "## Recommendation",
        "",
        f"- recommended Line A mode for full run: **{rec_mode}**",
        f"- reason: {rec_reason}",
        "",
        "## Safety flags",
        "",
        "- DETECTOR_EVAL_STARTED=NO",
        "- POINTNET_CLASSIFIER_STARTED=NO",
        "- KITTI_AP_EVAL_STARTED=NO",
        "- full run executed: NO (smoke only)",
        "",
        "## Per-sample notes",
        "",
    ]
    for name, summary in summaries.items():
        lines.append(f"### {name}")
        lines.append("")
        for r in summary["records"]:
            lines.append(
                f"- `{r['split']}/{r['class_name']}/{r['shape_id']}`: {r['status']} "
                f"raw={r['raw_points']} final={r['final_points']} crop={r['crop_action']} "
                f"infer={r['method_inference_called']} err={r['error']!r}"
            )
        lines.append("")

    # Estimated full-run commands (not executed)
    lines.extend(
        [
            "## Estimated full-run commands (NOT executed)",
            "",
            "```bash",
            f"# After smoke PASS confirmation — Line B full",
            f"sbatch jobs/pu_edgeformer/run_lineB_pu_edgeformer_full.sbatch  # to be created after confirmation",
            f"# Line A full using recommended mode: {rec_mode}",
            f"sbatch jobs/pu_edgeformer/run_lineA_pu_edgeformer_full.sbatch  # to be created after confirmation",
            "```",
            "",
            "Suggested CLI once full runner is approved:",
            "",
            "```bash",
            "source scripts/activate_upsampling_env.sh pugcn",
            "python scripts/run_pu_edgeformer_modelnet40_labparams_full.py \\",
            "  --line lineB --mode direct_256_to_1024",
            "python scripts/run_pu_edgeformer_modelnet40_labparams_full.py \\",
            f"  --line lineA --mode {rec_mode}",
            "```",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")

    # Also dump machine-readable summary
    summary_json = PROJECT_ROOT / "reports" / "pu_edgeformer_modelnet40_labparams_smoke_20260716.json"
    payload = {
        "lineB": {k: v for k, v in b.items() if k != "records"},
        "lineA_A1": {k: v for k, v in a1.items() if k != "records"},
        "lineA_A2": {k: v for k, v in a2.items() if k != "records"},
        "recommended_lineA_mode": rec_mode,
        "recommended_reason": rec_reason,
        "DETECTOR_EVAL_STARTED": "NO",
        "POINTNET_CLASSIFIER_STARTED": "NO",
        "KITTI_AP_EVAL_STARTED": "NO",
        "env_info": env_info,
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report


def collect_env_info() -> dict:
    info = {
        "env_name": "tf15_upsampling",
        "python": "unknown",
        "tensorflow": "unknown",
        "gpu": "unknown",
    }
    try:
        out = subprocess.check_output(
            [
                str(TF_PYTHON),
                "-c",
                "import sys,tensorflow as tf; print(sys.version.replace('\\n',' ')); print(tf.__version__); "
                "print(tf.test.is_gpu_available())",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        if len(out) >= 3:
            info["python"], info["tensorflow"], info["gpu"] = out[0], out[1], f"tf.test.is_gpu_available={out[2]}"
    except Exception as exc:  # noqa: BLE001
        info["gpu"] = f"env_check_failed: {exc}"
    try:
        smi = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT)
        info["nvidia_smi_L"] = smi.strip()
    except Exception as exc:  # noqa: BLE001
        info["nvidia_smi_L"] = f"unavailable: {exc}"
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-split", type=int, default=5)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--modes", nargs="+", default=["lineB", "A1", "A2"])
    args = parser.parse_args()

    if not TF_PYTHON.is_file():
        print(f"FATAL: TF python missing: {TF_PYTHON}", file=sys.stderr)
        return 2
    if not WRAPPER_MANY.is_file():
        print(f"FATAL: wrapper missing: {WRAPPER_MANY}", file=sys.stderr)
        return 2
    if not (CKPT_DIR / "checkpoint").is_file():
        print(f"FATAL: checkpoint missing under {CKPT_DIR}", file=sys.stderr)
        return 2

    env_info = collect_env_info()
    summaries: dict[str, dict] = {}

    if "lineB" in args.modes:
        print("=== Line B smoke: direct 256 -> 1024 ===", flush=True)
        summaries["lineB_direct_256_to_1024"] = run_mode(
            mode="lineB_direct_256_to_1024",
            input_root=DOWNSAMPLED_X4_ROOT,
            out_root=OUT_LINE_B,
            max_per_split=args.max_per_split,
            input_points=256,
            target_points=1024,
            patch_input=256,
            patch_output=1024,
            chunked=False,
            gpu_memory_fraction=args.gpu_memory_fraction,
        )
        print(json.dumps({k: summaries["lineB_direct_256_to_1024"][k] for k in ("status", "n_pass", "n_fail")}, sort_keys=True), flush=True)

    if "A1" in args.modes:
        print("=== Line A A1 smoke: direct 1024 -> 4096 ===", flush=True)
        summaries["lineA_A1_direct_1024_to_4096"] = run_mode(
            mode="lineA_A1_direct_1024_to_4096",
            input_root=ORIGINAL_ROOT,
            out_root=OUT_LINE_A,
            max_per_split=args.max_per_split,
            input_points=1024,
            target_points=4096,
            patch_input=1024,
            patch_output=4096,
            chunked=False,
            gpu_memory_fraction=args.gpu_memory_fraction,
        )
        print(json.dumps({k: summaries["lineA_A1_direct_1024_to_4096"][k] for k in ("status", "n_pass", "n_fail")}, sort_keys=True), flush=True)

    if "A2" in args.modes:
        print("=== Line A A2 smoke: 4x256 chunks -> 4096 ===", flush=True)
        summaries["lineA_A2_4x256_chunks"] = run_mode(
            mode="lineA_A2_4x256_chunks",
            input_root=ORIGINAL_ROOT,
            out_root=OUT_LINE_A,
            max_per_split=args.max_per_split,
            input_points=1024,
            target_points=4096,
            patch_input=256,
            patch_output=1024,
            chunked=True,
            gpu_memory_fraction=args.gpu_memory_fraction,
        )
        print(json.dumps({k: summaries["lineA_A2_4x256_chunks"][k] for k in ("status", "n_pass", "n_fail")}, sort_keys=True), flush=True)

    # Fill missing modes with FAIL placeholders if partial run
    for key in ("lineB_direct_256_to_1024", "lineA_A1_direct_1024_to_4096", "lineA_A2_4x256_chunks"):
        if key not in summaries:
            summaries[key] = {
                "mode": key,
                "status": "SKIPPED",
                "n_samples": 0,
                "n_pass": 0,
                "n_fail": 0,
                "any_nan_inf": False,
                "any_raw_shortfall": False,
                "elapsed_total_sec": 0.0,
                "manifest_csv": "",
                "records": [],
            }

    report = write_report(summaries, env_info)
    print(f"REPORT: {report}", flush=True)
    print(
        json.dumps(
            {
                "LineB": summaries["lineB_direct_256_to_1024"]["status"],
                "LineA_A1": summaries["lineA_A1_direct_1024_to_4096"]["status"],
                "LineA_A2": summaries["lineA_A2_4x256_chunks"]["status"],
                "DETECTOR_EVAL_STARTED": "NO",
                "POINTNET_CLASSIFIER_STARTED": "NO",
                "KITTI_AP_EVAL_STARTED": "NO",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    ok = all(summaries[k]["status"] == "PASS" for k in summaries)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
