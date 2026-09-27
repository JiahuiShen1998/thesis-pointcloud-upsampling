#!/usr/bin/env python3
"""Run PDANS on KITTI velodyne frames and write detector-facing .bin files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDANS_ROOT = PROJECT_ROOT / "external" / "PDANS"
RESULT_ROOT = PROJECT_ROOT / "results" / "pdans_main_kitti_pipeline"
VAL_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
TRAINING_DIR = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
ORIGINAL_INPUT = TRAINING_DIR / "velodyne_original_val"
DOWNSAMPLED_INPUT = TRAINING_DIR / "velodyne_downsampled_50_val"
PUGAN_CKPT = PDANS_ROOT / "checkpoints" / "PUGAN_PDANS.pkl"
PU1K_CKPT = PDANS_ROOT / "checkpoints" / "PU1K_PDANS.pkl"
PUGAN_CONFIG = PDANS_ROOT / "pointnet2" / "exp_configs" / "PUGAN.json"
PU1K_CONFIG = PDANS_ROOT / "pointnet2" / "exp_configs" / "PU1K.json"

METRIC_RANGE = {
    "x_min": 0.0,
    "x_max": 80.0,
    "y_min": -40.0,
    "y_max": 40.0,
    "z_min": -3.5,
    "z_max": 2.0,
}
RPN_NUM_POINTS = 16384
FAR_CAP_FOR_DEFAULT = 16000
SEED_BASE = 20260612


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit", help="Audit KITTI inputs and PDANS code/checkpoints.")
    audit.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    audit.add_argument("--val-split", type=Path, default=VAL_SPLIT)
    audit.add_argument("--original-input", type=Path, default=ORIGINAL_INPUT)
    audit.add_argument("--downsampled-input", type=Path, default=DOWNSAMPLED_INPUT)

    infer = sub.add_parser("infer", help="Run PDANS inference for a frame split.")
    infer.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    infer.add_argument("--input-folder", type=Path, required=True)
    infer.add_argument("--output-folder", type=Path, required=True)
    infer.add_argument("--line-name", type=str, required=True)
    infer.add_argument("--frame-list", type=Path, default=VAL_SPLIT)
    infer.add_argument("--frame-id", action="append", default=None)
    infer.add_argument("--limit", type=int, default=0)
    infer.add_argument("--start-index", type=int, default=0)
    infer.add_argument("--checkpoint", type=Path, default=PUGAN_CKPT)
    infer.add_argument("--config", type=Path, default=PUGAN_CONFIG)
    infer.add_argument("--num-patches", type=int, default=2)
    infer.add_argument("--patch-size", type=int, default=1024)
    infer.add_argument("--batch-size", type=int, default=2)
    infer.add_argument("--upsample-ratio", type=int, default=4)
    infer.add_argument("--step", type=int, default=30)
    infer.add_argument("--gamma", type=float, default=0.5)
    infer.add_argument("--cap-points", type=int, default=100000)
    infer.add_argument("--far-cap", type=int, default=FAR_CAP_FOR_DEFAULT)
    infer.add_argument("--include-original", dest="include_original", action="store_true", default=True)
    infer.add_argument("--no-include-original", dest="include_original", action="store_false")
    infer.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    infer.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    infer.add_argument("--device", type=str, default="cuda")
    infer.add_argument("--seed", type=int, default=SEED_BASE)
    infer.add_argument("--summary-name", type=str, default=None)
    return parser.parse_args()


def read_frame_ids(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_frame_ids(args: argparse.Namespace) -> List[str]:
    frame_ids = list(args.frame_id) if args.frame_id else read_frame_ids(args.frame_list)
    frame_ids = frame_ids[args.start_index :]
    if args.limit:
        frame_ids = frame_ids[: args.limit]
    return frame_ids


def read_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path} does not reshape to N x 4")
    return raw.reshape(-1, 4)


def write_bin(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(points, dtype=np.float32).tofile(path)


def finite_metric_mask(points: np.ndarray) -> np.ndarray:
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


def detector_masks(points: np.ndarray) -> Dict[str, np.ndarray]:
    valid = finite_metric_mask(points)
    depth = points[:, 0]
    near = valid & (depth < 40.0)
    far = valid & (depth >= 40.0)
    return {"valid": valid, "near": near, "far": far}


def choose(rng: np.random.Generator, idxs: np.ndarray, n: int) -> np.ndarray:
    idxs = np.asarray(idxs, dtype=np.int64)
    if n <= 0 or idxs.size == 0:
        return np.empty((0,), dtype=np.int64)
    if idxs.size <= n:
        return idxs.copy()
    return rng.choice(idxs, size=n, replace=False).astype(np.int64)


def cap_detector_points(points: np.ndarray, cap_points: int, far_cap: int, rng: np.random.Generator) -> np.ndarray:
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.shape[0] <= cap_points:
        return finite.astype(np.float32, copy=True)
    masks = detector_masks(finite)
    far_idxs = np.where(masks["far"])[0]
    near_idxs = np.where(masks["near"])[0]
    other_idxs = np.where(~masks["far"] & ~masks["near"])[0]

    selected: List[np.ndarray] = []
    far_take = choose(rng, far_idxs, min(far_cap, far_idxs.size))
    selected.append(far_take)
    near_min = max(0, RPN_NUM_POINTS - far_take.size)
    near_take = choose(rng, near_idxs, near_min)
    selected.append(near_take)
    selected_set = set(int(x) for x in np.concatenate(selected) if selected)

    remaining = cap_points - len(selected_set)
    for group in (other_idxs, near_idxs, far_idxs):
        if remaining <= 0:
            break
        rest = np.array([int(x) for x in group if int(x) not in selected_set], dtype=np.int64)
        take = choose(rng, rest, remaining)
        selected_set.update(int(x) for x in take)
        remaining = cap_points - len(selected_set)
    keep = np.array(sorted(selected_set), dtype=np.int64)
    if keep.size > cap_points:
        keep = choose(rng, keep, cap_points)
    rng.shuffle(keep)
    return finite[keep].astype(np.float32, copy=True)


def patch_indices(points: np.ndarray, frame_id: str, num_patches: int, patch_size: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed + int(frame_id))
    candidate = np.where(finite_metric_mask(points))[0]
    if candidate.size < max(32, patch_size // 8):
        candidate = np.where(np.isfinite(points).all(axis=1))[0]
    if candidate.size == 0:
        raise ValueError(f"{frame_id}: no finite points available")
    centers = rng.choice(candidate, size=num_patches, replace=candidate.size < num_patches)
    xyz = points[:, :3]
    patches: List[np.ndarray] = []
    for center in centers:
        diff = xyz - xyz[center]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        nearest = np.argpartition(dist2, min(patch_size, len(dist2) - 1))[:patch_size]
        if nearest.size < patch_size:
            extra = rng.choice(nearest, size=patch_size - nearest.size, replace=True)
            nearest = np.concatenate([nearest, extra])
        patches.append(nearest.astype(np.int64))
    return patches


def normalize_patch(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return (centered / scale).astype(np.float32), center.reshape(3).astype(np.float32), scale


def nearest_intensity(source_xyzi: np.ndarray, generated_xyz: np.ndarray, chunk: int = 4096) -> np.ndarray:
    src_xyz = source_xyzi[:, :3].astype(np.float32)
    src_i = source_xyzi[:, 3].astype(np.float32)
    out = np.empty((generated_xyz.shape[0],), dtype=np.float32)
    for start in range(0, generated_xyz.shape[0], chunk):
        part = generated_xyz[start : start + chunk].astype(np.float32)
        dist2 = ((part[:, None, :] - src_xyz[None, :, :]) ** 2).sum(axis=2)
        out[start : start + chunk] = src_i[np.argmin(dist2, axis=1)]
    return out


def install_pytorch3d_knn_shim() -> None:
    if "pytorch3d.ops.knn" in sys.modules:
        return
    knn_mod = types.ModuleType("pytorch3d.ops.knn")

    def knn_points(x: torch.Tensor, y: torch.Tensor, K: int = 1, return_nn: bool = False):
        dist = torch.cdist(x, y)
        vals, idx = torch.topk(dist, k=K, dim=2, largest=False, sorted=True)
        if not return_nn:
            return vals, idx, None
        bsz, npts, kval = idx.shape
        channels = y.shape[2]
        yy = y[:, None, :, :].expand(-1, npts, -1, -1)
        nn = torch.gather(yy, 2, idx[..., None].expand(bsz, npts, kval, channels))
        return vals, idx, nn

    def knn_gather(y: torch.Tensor, idx: torch.Tensor):
        bsz, npts, kval = idx.shape
        channels = y.shape[2]
        yy = y[:, None, :, :].expand(-1, npts, -1, -1)
        return torch.gather(yy, 2, idx[..., None].expand(bsz, npts, kval, channels))

    knn_mod.knn_points = knn_points
    knn_mod.knn_gather = knn_gather
    ops_mod = types.ModuleType("pytorch3d.ops")
    ops_mod.knn = knn_mod
    pt3d_mod = types.ModuleType("pytorch3d")
    pt3d_mod.ops = ops_mod
    sys.modules["pytorch3d"] = pt3d_mod
    sys.modules["pytorch3d.ops"] = ops_mod
    sys.modules["pytorch3d.ops.knn"] = knn_mod


def load_pdans(config_path: Path, checkpoint_path: Path, device: torch.device):
    install_pytorch3d_knn_shim()
    sys.path[:0] = [
        str(PDANS_ROOT),
        str(PDANS_ROOT / "pointnet2"),
        str(PDANS_ROOT / "pointnet2_ops_lib"),
    ]
    from pointnet2.json_reader import restore_string_to_list_in_a_dict
    from pointnet2.models.pointnet2_with_pcld_condition import PointNet2CloudCondition
    from pointnet2.util import calc_diffusion_hyperparams, sampling_ddim

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = restore_string_to_list_in_a_dict(config)
    net = PointNet2CloudCondition(config["pointnet_config"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"], strict=False)
    net.eval()
    diffusion = calc_diffusion_hyperparams(**config["diffusion_config"])
    for key, value in diffusion.items():
        if key != "T":
            diffusion[key] = value.to(device)
    return net, diffusion, config, sampling_ddim, checkpoint


def run_pdans_batches(
    net,
    diffusion,
    sampling_ddim,
    patches: List[np.ndarray],
    points: np.ndarray,
    batch_size: int,
    upsample_ratio: int,
    step: int,
    gamma: float,
    device: torch.device,
) -> np.ndarray:
    generated: List[np.ndarray] = []
    label = torch.full((batch_size,), fill_value=upsample_ratio - 1, dtype=torch.long, device=device)
    for start in range(0, len(patches), batch_size):
        batch_patches = patches[start : start + batch_size]
        normed = []
        centers = []
        scales = []
        for idxs in batch_patches:
            patch_xyz = points[idxs, :3]
            patch_norm, center, scale = normalize_patch(patch_xyz)
            normed.append(patch_norm)
            centers.append(center)
            scales.append(scale)
        condition = torch.from_numpy(np.stack(normed, axis=0)).to(device=device, dtype=torch.float32)
        cur_label = label[: condition.shape[0]]
        net.reset_cond_features()
        with torch.no_grad():
            dense, _, _ = sampling_ddim(
                net=net,
                size=(condition.shape[0], condition.shape[1] * upsample_ratio, 3),
                diffusion_hyperparams=diffusion,
                label=cur_label,
                condition=condition,
                R=upsample_ratio,
                gamma=gamma,
                step=step,
            )
        dense_np = dense.detach().cpu().numpy()
        for i in range(dense_np.shape[0]):
            generated.append((dense_np[i] * scales[i] + centers[i]).astype(np.float32))
        torch.cuda.empty_cache()
    return np.concatenate(generated, axis=0) if generated else np.zeros((0, 3), dtype=np.float32)


def stats_row(points: np.ndarray) -> Dict[str, object]:
    masks = detector_masks(points)
    xyz = points[:, :3]
    return {
        "points": int(points.shape[0]),
        "shape": f"{points.shape[0]}x{points.shape[1]}",
        "dtype": "float32",
        "finite": bool(np.isfinite(points).all()),
        "nan_points": int(np.isnan(points).any(axis=1).sum()),
        "inf_points": int(np.isinf(points).any(axis=1).sum()),
        "x_min": float(xyz[:, 0].min()) if len(points) else "",
        "x_max": float(xyz[:, 0].max()) if len(points) else "",
        "y_min": float(xyz[:, 1].min()) if len(points) else "",
        "y_max": float(xyz[:, 1].max()) if len(points) else "",
        "z_min": float(xyz[:, 2].min()) if len(points) else "",
        "z_max": float(xyz[:, 2].max()) if len(points) else "",
        "intensity_min": float(points[:, 3].min()) if len(points) else "",
        "intensity_max": float(points[:, 3].max()) if len(points) else "",
        "detector_valid_points": int(masks["valid"].sum()),
        "detector_near_points": int(masks["near"].sum()),
        "detector_far_points": int(masks["far"].sum()),
    }


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


def input_audit_rows(frame_ids: Sequence[str], folders: Sequence[Tuple[str, Path]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sample_ids = ["000001", "003219", "006833", "007458"]
    for name, folder in folders:
        files = sorted(folder.glob("*.bin"))
        ids = {path.stem for path in files}
        missing_val = [fid for fid in frame_ids if fid not in ids]
        extra = sorted(ids - set(frame_ids))[:10]
        row = {
            "name": name,
            "path": str(folder),
            "bin_files": len(files),
            "val_frames": len(frame_ids),
            "missing_val_count": len(missing_val),
            "missing_val_first10": ",".join(missing_val[:10]) if missing_val else "none",
            "extra_first10": ",".join(extra) if extra else "none",
        }
        rows.append(row)
        for sample_id in sample_ids:
            path = folder / f"{sample_id}.bin"
            if not path.exists():
                rows.append({"name": f"{name}:{sample_id}", "path": str(path), "status": "missing"})
                continue
            pts = read_bin(path)
            sample = {"name": f"{name}:{sample_id}", "path": str(path), "status": "ok"}
            sample.update(stats_row(pts))
            rows.append(sample)
    return rows


def command_audit(args: argparse.Namespace) -> None:
    args.result_root.mkdir(parents=True, exist_ok=True)
    frame_ids = read_frame_ids(args.val_split)
    rows = input_audit_rows(
        frame_ids,
        [("original", args.original_input), ("downsampled50", args.downsampled_input)],
    )
    write_csv(args.result_root / "input_availability_audit.csv", rows)
    md = [
        "# PDANS KITTI Input Availability Audit",
        "",
        f"- Validation split: `{args.val_split}` ({len(frame_ids)} frame ids)",
        f"- Original input: `{args.original_input}`",
        f"- Downsampled 50% input: `{args.downsampled_input}`",
        f"- Original `.bin` count: `{len(list(args.original_input.glob('*.bin')))}`",
        f"- Downsampled `.bin` count: `{len(list(args.downsampled_input.glob('*.bin')))}`",
        "- KITTI binary format check: files reshape to `N x 4` float32 as `x,y,z,intensity`.",
        "",
        "See `input_availability_audit.csv` for sample point counts and ranges.",
    ]
    (args.result_root / "input_availability_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    code_md = [
        "# PDANS Code Audit",
        "",
        f"- PDANS root: `{PDANS_ROOT}`",
        f"- PUGAN checkpoint: `{PUGAN_CKPT}`",
        f"- PU1K checkpoint: `{PU1K_CKPT}`",
        f"- PUGAN config: `{PUGAN_CONFIG}`",
        f"- PU1K config: `{PU1K_CONFIG}`",
        "- Official `example_eval.py` reads `.xyz` with Open3D, uses xyz only, and saves generated `.xyz`.",
        "- Official saving path applies `pc_normalize()` before writing, so this KITTI adapter bypasses official output saving and restores normalized patch outputs to LiDAR coordinates.",
        "- Model input is patch/object-style xyz, not KITTI `x,y,z,intensity`; intensity is restored by nearest-neighbor interpolation from the source KITTI frame.",
        "- Default upsampling ratio is 4x. The detector-facing output includes original input points plus generated PDANS points, then applies a deterministic cap for PointRCNN compatibility.",
        "- `pytorch3d` is not installed in the tested `ear` env; this adapter installs an in-process KNN shim using `torch.cdist` for the small patch sizes used here.",
    ]
    (args.result_root / "pdans_code_audit.md").write_text("\n".join(code_md) + "\n", encoding="utf-8")


def command_infer(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    frame_ids = select_frame_ids(args)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    log_dir = args.result_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_name = args.summary_name or f"{args.line_name}_pdans_inference_summary.csv"
    failure_path = args.result_root / f"{args.line_name}_pdans_failures.csv"

    net, diffusion, config, sampling_ddim, checkpoint = load_pdans(args.config, args.checkpoint, device)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    started = time.time()
    for ordinal, frame_id in enumerate(frame_ids, start=1):
        input_path = args.input_folder / f"{frame_id}.bin"
        output_path = args.output_folder / f"{frame_id}.bin"
        row: Dict[str, object] = {
            "line_name": args.line_name,
            "frame_id": frame_id,
            "ordinal": ordinal,
            "total_frames": len(frame_ids),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status": "pending",
        }
        frame_start = time.time()
        try:
            if args.skip_existing and output_path.exists():
                out = read_bin(output_path)
                row.update(stats_row(out))
                row["status"] = "skipped_existing"
                row["runtime_sec"] = 0.0
                rows.append(row)
                continue
            src = read_bin(input_path)
            patches = patch_indices(src, frame_id, args.num_patches, args.patch_size, args.seed)
            generated_xyz = run_pdans_batches(
                net=net,
                diffusion=diffusion,
                sampling_ddim=sampling_ddim,
                patches=patches,
                points=src,
                batch_size=args.batch_size,
                upsample_ratio=args.upsample_ratio,
                step=args.step,
                gamma=args.gamma,
                device=device,
            )
            generated_i = nearest_intensity(src, generated_xyz)
            generated = np.column_stack([generated_xyz, generated_i]).astype(np.float32)
            combined = np.concatenate([src, generated], axis=0) if args.include_original else generated
            rng = np.random.default_rng(args.seed + int(frame_id) + args.cap_points)
            capped = cap_detector_points(combined, args.cap_points, args.far_cap, rng)
            write_bin(output_path, capped)
            row.update(stats_row(capped))
            row.update(
                {
                    "status": "success",
                    "input_points": int(src.shape[0]),
                    "generated_points": int(generated.shape[0]),
                    "combined_before_cap": int(combined.shape[0]),
                    "num_patches": args.num_patches,
                    "patch_size": args.patch_size,
                    "upsample_ratio": args.upsample_ratio,
                    "step": args.step,
                    "cap_points": args.cap_points,
                    "runtime_sec": round(time.time() - frame_start, 3),
                }
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            row["runtime_sec"] = round(time.time() - frame_start, 3)
            failures.append(dict(row))
        rows.append(row)
        write_csv(args.result_root / summary_name, rows)
        if failures:
            write_csv(failure_path, failures)
        print(f"[{args.line_name}] {ordinal}/{len(frame_ids)} {frame_id}: {row['status']} ({row['runtime_sec']}s)", flush=True)

    write_csv(args.result_root / summary_name, rows)
    if failures:
        write_csv(failure_path, failures)
    md = [
        f"# PDANS Inference Summary: {args.line_name}",
        "",
        f"- Input folder: `{args.input_folder}`",
        f"- Output folder: `{args.output_folder}`",
        f"- Frames requested: `{len(frame_ids)}`",
        f"- Successful/skipped outputs: `{sum(1 for r in rows if r.get('status') in {'success', 'skipped_existing'})}`",
        f"- Failures: `{len(failures)}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Checkpoint epoch/iter: `{checkpoint.get('epoch')}` / `{checkpoint.get('iter')}`",
        f"- Config dataset: `{config.get('train_config', {}).get('dataset')}`",
        f"- Runtime seconds: `{round(time.time() - started, 3)}`",
        "- Intensity policy: nearest-neighbor interpolation from the source KITTI frame.",
        "- Output policy: include original points plus PDANS generated points, finite filter, deterministic 100k cap, and far-depth guard for PointRCNN default sampling.",
    ]
    (args.result_root / f"{args.line_name}_pdans_inference_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.cmd == "audit":
        command_audit(args)
    elif args.cmd == "infer":
        command_infer(args)
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
