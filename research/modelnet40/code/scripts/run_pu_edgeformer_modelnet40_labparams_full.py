#!/usr/bin/env python3
"""PU-EdgeFormer ModelNet40 lab-params FULL upsampling (no classifier/detector/AP).

Modes:
  lineB / direct_256_to_1024      : 256 -> 1024
  lineA / A1_direct_1024_to_4096 : 1024 -> 4096 (recommended)
  lineA / A2_4x256_chunks        : 1024 -> 4x(256->1024) -> 4096 (fallback)

Constraints: XYZ-only, strict crop if raw>target, FAIL if raw<target,
no padding/jitter/interpolation/fake points.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
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

OUT_LINE_A = PROJECT_ROOT / "datasets" / "modelnet40_original_x4_up" / "pu_edgeformer_labparams_20260716"
OUT_LINE_B = PROJECT_ROOT / "datasets" / "modelnet40_downsampled_x4_up" / "pu_edgeformer_labparams_20260716"

INFER_SEED = 20260702
UP_RATIO = 4
METHOD = "PU-EdgeFormer"
MODEL = "edgetransformer"

MODE_SPECS = {
    "direct_256_to_1024": {
        "line_key": "lineB",
        "input_root": DOWNSAMPLED_X4_ROOT,
        "out_root": OUT_LINE_B,
        "input_points": 256,
        "target_points": 1024,
        "patch_input": 256,
        "patch_output": 1024,
        "chunked": False,
    },
    "A1_direct_1024_to_4096": {
        "line_key": "lineA",
        "input_root": ORIGINAL_ROOT,
        "out_root": OUT_LINE_A,
        "input_points": 1024,
        "target_points": 4096,
        "patch_input": 1024,
        "patch_output": 4096,
        "chunked": False,
    },
    "A2_4x256_chunks": {
        "line_key": "lineA",
        "input_root": ORIGINAL_ROOT,
        "out_root": OUT_LINE_A,
        "input_points": 1024,
        "target_points": 4096,
        "patch_input": 256,
        "patch_output": 1024,
        "chunked": True,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def stable_seed(*parts: object) -> int:
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


def is_valid_final(path: Path, target: int) -> bool:
    if not path.is_file():
        return False
    try:
        arr = np.load(path)
    except Exception:  # noqa: BLE001
        return False
    return (
        isinstance(arr, np.ndarray)
        and arr.shape == (target, 3)
        and arr.dtype == np.float32
        and bool(np.isfinite(arr).all())
    )


def list_all_samples(root: Path) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "test"):
        split_dir = root / split
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for npy in sorted(cls_dir.glob("*.npy")):
                rows.append(
                    {
                        "split": split,
                        "class_name": cls_dir.name,
                        "shape_id": npy.stem,
                        "input_path": npy,
                    }
                )
    rows.sort(key=lambda r: (r["split"], r["class_name"], r["shape_id"]))
    return rows


def select_chunk(rows: list[dict], chunk_id: int, num_chunks: int) -> list[dict]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError(f"chunk_id={chunk_id} out of range for num_chunks={num_chunks}")
    chunk_size = (len(rows) + num_chunks - 1) // num_chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, len(rows))
    return rows[start:end]


def chunk_1024_to_4x256(points: np.ndarray) -> list[np.ndarray]:
    if points.shape != (1024, 3):
        raise ValueError(f"A2 expects (1024,3), got {points.shape}")
    return [points[i * 256 : (i + 1) * 256].copy() for i in range(4)]


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
    return {
        "frame_id": frame_id,
        "patch_metadata_json": str(meta_path),
        "raw_patch_output_dir": str(work_dir / "raw_patches"),
        "merged_raw_output": str(work_dir / "merged_raw.npy"),
        "merged_raw_provenance_json": str(work_dir / "merged_raw_provenance.json"),
    }


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
    # Keep logs manageable: store tails + returncode; full stdout only if fail.
    body = (
        f"CMD: {' '.join(cmd)}\n"
        f"returncode={proc.returncode}\nelapsed_sec={elapsed:.3f}\njobs={len(jobs)}\n"
        f"\nSTDOUT_TAIL:\n{proc.stdout[-8000:]}\n"
        f"\nSTDERR_TAIL:\n{proc.stderr[-8000:]}\n"
    )
    if proc.returncode != 0:
        body += f"\nSTDOUT_FULL:\n{proc.stdout}\n\nSTDERR_FULL:\n{proc.stderr}\n"
    log_path.write_text(body, encoding="utf-8")
    summary = {"returncode": proc.returncode, "elapsed_sec": elapsed, "log_path": str(log_path)}
    for line_txt in reversed(proc.stdout.splitlines()):
        line_txt = line_txt.strip()
        if line_txt.startswith("{") and "status" in line_txt:
            try:
                summary["wrapper_summary"] = json.loads(line_txt)
                break
            except json.JSONDecodeError:
                pass
    return summary


def finalize_one(
    *,
    job: dict,
    sample: dict,
    out_root: Path,
    mode: str,
    target: int,
    keep_method_raw: bool,
    cleanup_work: bool,
) -> dict:
    split = sample["split"]
    class_name = sample["class_name"]
    shape_id = sample["shape_id"]
    final_path = out_root / split / class_name / f"{shape_id}.npy"
    method_raw_path = out_root / "method_raw" / mode / split / class_name / f"{shape_id}.npy"
    work_merged = Path(job["merged_raw_output"])
    work_prov = Path(job["merged_raw_provenance_json"])
    work_dir = work_merged.parent

    record = {
        "mode": mode,
        "split": split,
        "class_name": class_name,
        "shape_id": shape_id,
        "input_path": str(sample["input_path"]),
        "final_output": str(final_path),
        "status": "FAIL",
        "raw_points": "",
        "final_points": "",
        "crop_action": "",
        "method_inference_called": False,
        "checkpoint_loaded": False,
        "raw_shortfall": False,
        "finite": "",
        "error": "",
    }
    try:
        if not work_prov.is_file():
            raise FileNotFoundError(f"missing provenance: {work_prov}")
        prov = json.loads(work_prov.read_text(encoding="utf-8"))
        record["method_inference_called"] = bool(prov.get("method_inference_called"))
        record["checkpoint_loaded"] = bool(prov.get("checkpoint_loaded"))
        if prov.get("status") != "PASS":
            raise RuntimeError(f"wrapper status={prov.get('status')} notes={prov.get('notes')}")
        if not record["method_inference_called"] or not record["checkpoint_loaded"]:
            raise RuntimeError("inference/checkpoint flags not true")
        raw = np.load(work_merged).astype(np.float32)
        record["raw_points"] = int(raw.shape[0])
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
        final_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(final_path, final.astype(np.float32))
        if keep_method_raw:
            method_raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(method_raw_path, raw)
        record["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        record["error"] = str(exc)
    finally:
        if cleanup_work and work_dir.is_dir():
            shutil.rmtree(work_dir, ignore_errors=True)
    return record


def append_manifest(path: Path, records: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", required=True, choices=("lineA", "lineB"))
    parser.add_argument(
        "--mode",
        required=True,
        choices=tuple(MODE_SPECS.keys()),
    )
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=200, help="samples per TF restore batch")
    parser.add_argument("--keep-method-raw", action="store_true")
    parser.add_argument("--keep-work", action="store_true", help="keep per-sample work dirs")
    args = parser.parse_args()

    spec = MODE_SPECS[args.mode]
    if spec["line_key"] != args.line:
        print(f"ERROR: mode {args.mode} belongs to {spec['line_key']}, got --line {args.line}", file=sys.stderr)
        return 2

    if not TF_PYTHON.is_file():
        print(f"FATAL: missing TF python {TF_PYTHON}", file=sys.stderr)
        return 2
    if not (CKPT_DIR / "checkpoint").is_file():
        print(f"FATAL: missing checkpoint under {CKPT_DIR}", file=sys.stderr)
        return 2

    out_root: Path = spec["out_root"]
    for sub in ("train", "test", "logs", "manifests", "audits", "method_raw"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    all_rows = list_all_samples(spec["input_root"])
    rows = select_chunk(all_rows, args.chunk_id, args.num_chunks)
    print(
        json.dumps(
            {
                "line": args.line,
                "mode": args.mode,
                "chunk_id": args.chunk_id,
                "num_chunks": args.num_chunks,
                "total_dataset": len(all_rows),
                "chunk_rows": len(rows),
                "out_root": str(out_root),
                "started": utc_now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

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
        "raw_shortfall",
        "finite",
        "error",
        "input_path",
        "final_output",
    ]
    manifest = out_root / "manifests" / f"{args.mode}_chunk{args.chunk_id:02d}_of_{args.num_chunks:02d}.csv"
    failed_csv = out_root / "logs" / f"{args.mode}_chunk{args.chunk_id:02d}_failed.csv"

    skipped = 0
    pending: list[dict] = []
    for sample in rows:
        final_path = out_root / sample["split"] / sample["class_name"] / f"{sample['shape_id']}.npy"
        if args.skip_existing and is_valid_final(final_path, spec["target_points"]):
            skipped += 1
            continue
        pending.append(sample)

    print(f"pending={len(pending)} skipped_existing={skipped}", flush=True)

    n_pass = 0
    n_fail = 0
    t0 = time.perf_counter()
    batch_size = max(1, int(args.batch_size))

    for batch_idx in range(0, len(pending), batch_size):
        batch = pending[batch_idx : batch_idx + batch_size]
        jobs: list[dict] = []
        metas: list[dict] = []
        work_root = out_root / "method_raw" / args.mode / "_work" / f"chunk{args.chunk_id:02d}" / f"batch{batch_idx:05d}"
        for sample in batch:
            pts = np.load(sample["input_path"]).astype(np.float32)
            if pts.shape != (spec["input_points"], 3):
                raise ValueError(f"{sample['input_path']} shape {pts.shape}")
            if not np.isfinite(pts).all():
                raise ValueError(f"{sample['input_path']} NaN/Inf")
            frame_id = f"{sample['split']}__{sample['class_name']}__{sample['shape_id']}"
            work_dir = work_root / sample["split"] / sample["class_name"] / sample["shape_id"]
            patches = chunk_1024_to_4x256(pts) if spec["chunked"] else [pts.copy()]
            job = write_patch_job(
                work_dir=work_dir,
                frame_id=frame_id,
                patches=patches,
                patch_input=spec["patch_input"],
            )
            jobs.append(job)
            metas.append({"job": job, "sample": sample})

        jobs_json = out_root / "manifests" / f"{args.mode}_chunk{args.chunk_id:02d}_batch{batch_idx:05d}_jobs.json"
        log_path = out_root / "logs" / f"{args.mode}_chunk{args.chunk_id:02d}_batch{batch_idx:05d}_infer.log"
        infer = run_infer_many(
            jobs=jobs,
            jobs_json=jobs_json,
            line=f"{args.line}_{args.mode}",
            patch_input=spec["patch_input"],
            patch_output=spec["patch_output"],
            log_path=log_path,
            gpu_memory_fraction=args.gpu_memory_fraction,
        )
        print(
            json.dumps(
                {
                    "batch_idx": batch_idx,
                    "batch_n": len(batch),
                    "infer_returncode": infer.get("returncode"),
                    "infer_elapsed_sec": infer.get("elapsed_sec"),
                    "wrapper_summary": infer.get("wrapper_summary"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        records = []
        for meta in metas:
            rec = finalize_one(
                job=meta["job"],
                sample=meta["sample"],
                out_root=out_root,
                mode=args.mode,
                target=spec["target_points"],
                keep_method_raw=args.keep_method_raw,
                cleanup_work=not args.keep_work,
            )
            records.append(rec)
            if rec["status"] == "PASS":
                n_pass += 1
            else:
                n_fail += 1
        append_manifest(manifest, records, fields)
        fail_recs = [r for r in records if r["status"] != "PASS"]
        if fail_recs:
            append_manifest(failed_csv, fail_recs, fields)
        # Drop batch work root if empty-ish
        if not args.keep_work and work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)
        # Remove large jobs_json after success to save inode/space
        if infer.get("returncode") == 0 and jobs_json.is_file():
            try:
                jobs_json.unlink()
            except FileNotFoundError:
                pass

    summary = {
        "line": args.line,
        "mode": args.mode,
        "chunk_id": args.chunk_id,
        "num_chunks": args.num_chunks,
        "chunk_rows": len(rows),
        "skipped_existing": skipped,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "pending": len(pending),
        "elapsed_sec": time.perf_counter() - t0,
        "out_root": str(out_root),
        "manifest": str(manifest),
        "finished": utc_now(),
        "DETECTOR_EVAL_STARTED": "NO",
        "POINTNET_CLASSIFIER_STARTED": "NO",
        "KITTI_AP_EVAL_STARTED": "NO",
        "status": "PASS" if n_fail == 0 else "FAIL",
    }
    summary_path = out_root / "audits" / f"{args.mode}_chunk{args.chunk_id:02d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
