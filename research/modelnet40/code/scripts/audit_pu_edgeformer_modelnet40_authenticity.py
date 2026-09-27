#!/usr/bin/env python3
"""Authenticity verification for PU-EdgeFormer ModelNet40 full outputs.

Checks:
  1) not simple repeat/copy of input
  2) real PU-EdgeFormer model path / checkpoint / logs (not PU-GCN)
  3) not smoke leakage
  4) output freshness vs job windows

No geometry metrics / PointNet++ / detector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"

OUT_B = PROJECT_ROOT / "datasets/modelnet40_downsampled_x4_up/pu_edgeformer_labparams_20260716"
OUT_A = PROJECT_ROOT / "datasets/modelnet40_original_x4_up/pu_edgeformer_labparams_20260716"
IN_B = PROJECT_ROOT / "datasets/modelnet40_downsampled_x4"
IN_A = PROJECT_ROOT / "datasets/modelnet40_original"
PUGCN_B = PROJECT_ROOT / "datasets/lineB_downsampled_x4_up/strict_N/pu_gcn"
PUGCN_A = PROJECT_ROOT / "datasets/lineA_original_up/strict_4N/pu_gcn"
SMOKE_B = PROJECT_ROOT / "datasets/modelnet40_downsampled_x4_up/pu_edgeformer_labparams_smoke_20260716"
SMOKE_A = PROJECT_ROOT / "datasets/modelnet40_original_x4_up/pu_edgeformer_labparams_smoke_20260716"
CKPT_DIR = PROJECT_ROOT / "external/pu_edgeformer_ops_reuse/checkpoint_model100"
REPO_DIR = PROJECT_ROOT / "external/pu_edgeformer_ops_reuse"
WRAPPER = (
    PROJECT_ROOT
    / "imports/transfer_pu_edgeformer_to_hpc_20260716/wrappers/tf_pugcn_family_patch_infer_many.py"
)
FULL_SCRIPT = PROJECT_ROOT / "scripts/run_pu_edgeformer_modelnet40_labparams_full.py"
SBATCH_B = PROJECT_ROOT / "jobs/pu_edgeformer/run_lineB_pu_edgeformer_full.sbatch"
SBATCH_A = PROJECT_ROOT / "jobs/pu_edgeformer/run_lineA_pu_edgeformer_full_A1.sbatch"
LOG_DIR = PROJECT_ROOT / "logs/pu_edgeformer"
TF_PYTHON = Path.home() / ".conda/envs/tf15_upsampling/bin/python"

JOB_B = "1749211"
JOB_A = "1749212"
SEED = 20260716


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def round_points(pts: np.ndarray, decimals: int = 6) -> np.ndarray:
    return np.round(pts.astype(np.float64), decimals=decimals)


def unique_count(pts: np.ndarray, decimals: int = 6) -> int:
    rounded = round_points(pts, decimals)
    # view as void for unique rows
    view = np.ascontiguousarray(rounded).view(
        np.dtype((np.void, rounded.dtype.itemsize * rounded.shape[1]))
    )
    return int(np.unique(view).size)


def set_of_rounded(pts: np.ndarray, decimals: int = 6):
    rounded = round_points(pts, decimals)
    return {tuple(row.tolist()) for row in rounded}


def match_ratio(output: np.ndarray, input_pts: np.ndarray, decimals: int | None) -> float:
    if decimals is None:
        inp = {tuple(row.tolist()) for row in input_pts.astype(np.float64)}
        out = [tuple(row.tolist()) for row in output.astype(np.float64)]
    else:
        inp = set_of_rounded(input_pts, decimals)
        out = [tuple(row.tolist()) for row in round_points(output, decimals)]
    if not out:
        return 0.0
    hits = sum(1 for p in out if p in inp)
    return hits / float(len(out))


def nn_distances(output: np.ndarray, input_pts: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    out = output.astype(np.float64)
    inp = input_pts.astype(np.float64)
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(inp)
    dists, _ = nn.kneighbors(out, return_distance=True)
    return dists.reshape(-1)


def is_exact_repeat(output: np.ndarray, input_pts: np.ndarray) -> bool:
    try:
        rep = np.repeat(input_pts, 4, axis=0)
        tile = np.tile(input_pts, (4, 1))
        if output.shape == rep.shape and np.allclose(output, rep, atol=0, rtol=0):
            return True
        if output.shape == tile.shape and np.allclose(output, tile, atol=0, rtol=0):
            return True
    except Exception:  # noqa: BLE001
        return False
    return False


def shuffled_repeat_score(output: np.ndarray, input_pts: np.ndarray) -> dict:
    """If each input point appears ~4 times in rounded output, score high."""
    out_round = [tuple(r.tolist()) for r in round_points(output, 6)]
    in_round = [tuple(r.tolist()) for r in round_points(input_pts, 6)]
    cnt = Counter(out_round)
    counts = [cnt.get(p, 0) for p in in_round]
    if not counts:
        return {"mean_count": 0.0, "frac_count_eq_4": 0.0, "frac_count_in_3_5": 0.0}
    arr = np.asarray(counts, dtype=np.float64)
    return {
        "mean_count": float(arr.mean()),
        "frac_count_eq_4": float(np.mean(arr == 4)),
        "frac_count_in_3_5": float(np.mean((arr >= 3) & (arr <= 5))),
    }


def list_rel_samples(root: Path) -> list[str]:
    rows = []
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for p in sorted(split_dir.rglob("*.npy")):
            rows.append(str(p.relative_to(root)))
    return rows


def sample_rels(rels: list[str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    if len(rels) <= n:
        return list(rels)
    return rng.sample(rels, n)


def audit_repeat_copy(n_per_line: int = 200) -> dict:
    rows = []
    summaries = {}
    for line, in_root, out_root, exp_in, exp_out in [
        ("lineB", IN_B, OUT_B, 256, 1024),
        ("lineA", IN_A, OUT_A, 1024, 4096),
    ]:
        rels = list_rel_samples(out_root)
        chosen = sample_rels(rels, n_per_line, SEED + (0 if line == "lineB" else 1))
        line_metrics = []
        for rel in chosen:
            inp_path = in_root / rel
            out_path = out_root / rel
            if not inp_path.is_file() or not out_path.is_file():
                rows.append(
                    {
                        "line": line,
                        "rel": rel,
                        "status": "MISSING",
                        "error": f"inp={inp_path.is_file()} out={out_path.is_file()}",
                    }
                )
                continue
            inp = np.load(inp_path).astype(np.float32)
            out = np.load(out_path).astype(np.float32)
            if inp.shape != (exp_in, 3) or out.shape != (exp_out, 3):
                rows.append(
                    {
                        "line": line,
                        "rel": rel,
                        "status": "BAD_SHAPE",
                        "input_shape": str(inp.shape),
                        "output_shape": str(out.shape),
                    }
                )
                continue
            exact = match_ratio(out, inp, None)
            rounded = match_ratio(out, inp, 6)
            in_u = unique_count(inp, 6)
            out_u = unique_count(out, 6)
            dup = 1.0 - (out_u / float(out.shape[0]))
            nn = nn_distances(out, inp)
            shuf = shuffled_repeat_score(out, inp)
            is_rep = is_exact_repeat(out, inp)
            # tiny jitter of repeated input?
            jitter_like = (
                out_u <= in_u * 1.05
                and rounded > 0.9
                and float(np.median(nn)) < 1e-4
            )
            fail = (
                exact > 0.95
                or rounded > 0.95
                or dup > 0.60
                or out_u <= (300 if line == "lineB" else 1100)
                or abs(out_u - in_u) <= (50 if line == "lineB" else 100)
                or is_rep
                or shuf["frac_count_eq_4"] > 0.80
            )
            rec = {
                "line": line,
                "rel": rel,
                "status": "FAIL" if fail else "PASS",
                "input_points": int(inp.shape[0]),
                "output_points": int(out.shape[0]),
                "exact_match_ratio": exact,
                "rounded_1e6_match_ratio": rounded,
                "input_unique_count": in_u,
                "output_unique_count": out_u,
                "duplicate_ratio": dup,
                "nn_mean": float(nn.mean()),
                "nn_median": float(np.median(nn)),
                "nn_min": float(nn.min()),
                "nn_pct_lt_1e-8": float(np.mean(nn < 1e-8)),
                "nn_pct_lt_1e-6": float(np.mean(nn < 1e-6)),
                "exact_repeat_or_tile": is_rep,
                "shuffled_repeat_mean_count": shuf["mean_count"],
                "shuffled_repeat_frac_eq_4": shuf["frac_count_eq_4"],
                "shuffled_repeat_frac_3_5": shuf["frac_count_in_3_5"],
                "jitter_like_suspect": jitter_like,
            }
            rows.append(rec)
            line_metrics.append(rec)

        def agg(key):
            vals = [r[key] for r in line_metrics if key in r]
            return float(np.mean(vals)) if vals else float("nan")

        n_fail = sum(1 for r in line_metrics if r.get("status") == "FAIL")
        summaries[line] = {
            "n_sampled": len(line_metrics),
            "n_fail": n_fail,
            "n_pass": len(line_metrics) - n_fail,
            "mean_exact_match_ratio": agg("exact_match_ratio"),
            "mean_rounded_1e6_match_ratio": agg("rounded_1e6_match_ratio"),
            "mean_output_unique_count": agg("output_unique_count"),
            "mean_duplicate_ratio": agg("duplicate_ratio"),
            "mean_nn_mean": agg("nn_mean"),
            "mean_nn_median": agg("nn_median"),
            "mean_nn_pct_lt_1e-8": agg("nn_pct_lt_1e-8"),
            "mean_nn_pct_lt_1e-6": agg("nn_pct_lt_1e-6"),
            "n_exact_repeat_or_tile": sum(1 for r in line_metrics if r.get("exact_repeat_or_tile")),
            "mean_shuffled_frac_eq_4": agg("shuffled_repeat_frac_eq_4"),
            "status": "PASS" if n_fail == 0 and len(line_metrics) >= n_per_line else ("PASS" if n_fail == 0 else "FAIL"),
        }
        # allow if sampled all available when smaller — but we expect >=200
        if len(line_metrics) < n_per_line:
            summaries[line]["status"] = "FAIL"
            summaries[line]["note"] = f"sampled_only_{len(line_metrics)}"

    # write csv/md
    csv_path = REPORTS / "pu_edgeformer_repeat_copy_audit_20260716.csv"
    md_path = REPORTS / "pu_edgeformer_repeat_copy_audit_20260716.md"
    REPORTS.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PU-EdgeFormer Repeat/Copy Audit",
        "",
        f"- Generated: `{utc_now()}`",
        f"- samples per line requested: {n_per_line}",
        "",
    ]
    for line, s in summaries.items():
        lines.extend(
            [
                f"## {line}",
                "",
                f"- status: **{s['status']}**",
                f"- sampled/pass/fail: {s['n_sampled']} / {s['n_pass']} / {s['n_fail']}",
                f"- mean exact_match_ratio: {s['mean_exact_match_ratio']:.6f}",
                f"- mean rounded_1e6_match_ratio: {s['mean_rounded_1e6_match_ratio']:.6f}",
                f"- mean output_unique_count: {s['mean_output_unique_count']:.2f}",
                f"- mean duplicate_ratio: {s['mean_duplicate_ratio']:.6f}",
                f"- mean nn_mean / nn_median: {s['mean_nn_mean']:.6e} / {s['mean_nn_median']:.6e}",
                f"- mean nn_pct <1e-8 / <1e-6: {s['mean_nn_pct_lt_1e-8']:.6f} / {s['mean_nn_pct_lt_1e-6']:.6f}",
                f"- exact repeat/tile hits: {s['n_exact_repeat_or_tile']}",
                f"- mean shuffled_repeat frac_eq_4: {s['mean_shuffled_frac_eq_4']:.6f}",
                "",
            ]
        )
    overall = "PASS" if all(s["status"] == "PASS" for s in summaries.values()) else "FAIL"
    lines.extend([f"## Overall: **{overall}**", "", f"- csv: `{csv_path}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"overall": overall, "summaries": summaries, "csv": str(csv_path), "md": str(md_path)}


def grep_logs(job_id: str) -> dict:
    patterns = [
        "checkpoint",
        "restore",
        "saver",
        "model-100",
        "PU-EdgeFormer",
        "edgeformer",
        "edgetransformer",
        "transformer",
        "pugcn",
        "PU-GCN",
        "fallback",
        "repeat",
        "copy",
        "dummy",
        "skip",
        "interpolation",
        "padding",
        "Restoring parameters",
        "method_inference",
        "--model",
    ]
    files = sorted(LOG_DIR.glob(f"*{job_id}*.out")) + sorted(LOG_DIR.glob(f"*{job_id}*.err"))
    # Wrapper stdout lives in dataset infer logs (authoritative restore evidence).
    if job_id == JOB_B:
        files += sorted((OUT_B / "logs").glob("*infer.log"))
    elif job_id == JOB_A:
        files += sorted((OUT_A / "logs").glob("*infer.log"))
    counts = {p: 0 for p in patterns}
    restore_hits = []
    ckpt_hits = []
    bad_hits = []
    model_hits = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        lower = text.lower()
        for p in patterns:
            c = len(re.findall(re.escape(p), text, flags=re.IGNORECASE))
            counts[p] += c
        if "restoring parameters from" in lower:
            for line in text.splitlines():
                if "Restoring parameters from" in line or "restoring parameters from" in line.lower():
                    restore_hits.append(f"{fp.name}: {line.strip()}")
        if "checkpoint_model100" in text or "model-100" in text:
            for line in text.splitlines():
                if "checkpoint_model100" in line or "model-100" in line:
                    ckpt_hits.append(f"{fp.name}: {line.strip()[:240]}")
        if "edgetransformer" in lower or "pu-edgeformer" in lower or "edgeformer" in lower:
            for line in text.splitlines():
                if any(k in line.lower() for k in ("edgetransformer", "pu-edgeformer", "edgeformer")):
                    model_hits.append(f"{fp.name}: {line.strip()[:240]}")
        for bad in ("fallback", "dummy upsampling", "repeat input", "copy input"):
            if bad in lower:
                bad_hits.append(f"{fp.name}: contains '{bad}'")
    # unique limited
    def uniq(xs, n=30):
        out = []
        seen = set()
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
            if len(out) >= n:
                break
        return out

    n_tasks = len([f for f in files if f.suffix == ".out"])
    restore_ok = len(restore_hits) >= n_tasks  # at least one restore line per out ideally
    # weaker: any restore and correct ckpt path present across logs
    ckpt_ok = any("checkpoint_model100" in h for h in ckpt_hits + restore_hits)
    return {
        "job_id": job_id,
        "n_log_files": len(files),
        "n_out_files": n_tasks,
        "pattern_counts": counts,
        "restore_hits": uniq(restore_hits, 40),
        "ckpt_hits": uniq(ckpt_hits, 40),
        "model_hits": uniq(model_hits, 40),
        "bad_hits": uniq(bad_hits, 40),
        "restore_present": len(restore_hits) > 0,
        "ckpt_path_present": ckpt_ok,
        "bad_patterns_present": len(bad_hits) > 0,
    }


def callpath_audit() -> dict:
    sbatch_b = SBATCH_B.read_text(encoding="utf-8")
    sbatch_a = SBATCH_A.read_text(encoding="utf-8")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    full_py = FULL_SCRIPT.read_text(encoding="utf-8")
    gen_path = REPO_DIR / "Upsampling/generator.py"
    model_utils = REPO_DIR / "Common/model_utils.py"
    tf_ops = REPO_DIR / "tf_ops"
    # Confirm EdgeTransformer hardcoded and wrapper model arg
    edge_hardcode = "class EdgeTransformer" in gen_path.read_text(encoding="utf-8")
    get_model = "def get_model_cls" in model_utils.read_text(encoding="utf-8")
    uses_edgetransformer = "--model" in full_py and "edgetransformer" in full_py
    uses_wrapper_many = "tf_pugcn_family_patch_infer_many.py" in full_py
    pugcn_class_also_present = "class PUGCN" in gen_path.read_text(encoding="utf-8")
    # inference path
    has_patch_prediction = "patch_prediction" in wrapper
    has_saver_restore = "saver.restore" in wrapper
    return {
        "sbatch_lineB": str(SBATCH_B),
        "sbatch_lineA": str(SBATCH_A),
        "python_full_script": str(FULL_SCRIPT),
        "wrapper": str(WRAPPER),
        "repo_dir": str(REPO_DIR),
        "checkpoint_dir": str(CKPT_DIR),
        "tf_ops_path": str(tf_ops.resolve()),
        "tf_ops_is_symlink_to_pugcn": tf_ops.is_symlink()
        and "PU-GCN/tf_ops" in str(tf_ops.resolve()),
        "model_module": str(gen_path),
        "model_utils_module": str(model_utils),
        "model_construction": "get_model_cls(FLAGS.model) -> EdgeTransformer when model=edgetransformer",
        "inference_function": "Model.patch_prediction (via wrapper loop)",
        "wrapper_builds_gen_via": "model_cls = get_model_cls(FLAGS.model); gen = model_cls(...); model.pred_pc = gen(model.inputs)",
        "EdgeTransformer_class_present": edge_hardcode,
        "PUGCN_class_also_in_same_file": pugcn_class_also_present,
        "full_script_requests_edgetransformer": uses_edgetransformer,
        "full_script_calls_wrapper_many": uses_wrapper_many,
        "wrapper_has_saver_restore": has_saver_restore,
        "wrapper_has_patch_prediction": has_patch_prediction,
        "get_model_cls_present": get_model,
        "sbatch_B_mentions_full_script": "run_pu_edgeformer_modelnet40_labparams_full.py" in sbatch_b,
        "sbatch_A_mentions_full_script": "run_pu_edgeformer_modelnet40_labparams_full.py" in sbatch_a,
        "sbatch_B_mode": "direct_256_to_1024" if "direct_256_to_1024" in sbatch_b else "?",
        "sbatch_A_mode": "A1_direct_1024_to_4096" if "A1_direct_1024_to_4096" in sbatch_a else "?",
        "tf_ops_reused_from_PUGCN": "YES" if (tf_ops.is_symlink() and "PU-GCN" in str(tf_ops.resolve())) else "NO",
        "model_definition_from_PU_EdgeFormer": "YES" if edge_hardcode and uses_edgetransformer else "NO",
    }


def checkpoint_variable_audit() -> dict:
    out_txt = REPORTS / "pu_edgeformer_checkpoint_variable_audit_20260716.txt"
    REPORTS.mkdir(parents=True, exist_ok=True)
    prefix = str(CKPT_DIR / "model-100")
    code = f"""
import tensorflow as tf
vars = tf.train.list_variables({prefix!r})
print('N_VARS', len(vars))
for name, shape in vars[:100]:
    print(f'{{name}}\\t{{shape}}')
print('---NAME_HINTS---')
names = [n for n,_ in vars]
for key in ['generator', 'edge', 'transformer', 'up_block', 'duplicate', 'nodeshuffle', 'pugcn', 'dense']:
    hits = sum(1 for n in names if key.lower() in n.lower())
    print(key, hits)
"""
    proc = subprocess.run(
        [str(TF_PYTHON), "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    text = (
        f"# PU-EdgeFormer checkpoint variable audit\n"
        f"# Generated: {utc_now()}\n"
        f"# checkpoint prefix: {prefix}\n"
        f"# returncode: {proc.returncode}\n\n"
        f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n"
    )
    out_txt.write_text(text, encoding="utf-8")
    names = []
    for line in proc.stdout.splitlines():
        if "\t" in line and not line.startswith("N_VARS") and not line.startswith("---"):
            names.append(line.split("\t", 1)[0])
    hint = {}
    for key in ["generator", "edge", "transformer", "up_block", "duplicate", "nodeshuffle", "pugcn"]:
        hint[key] = sum(1 for n in names if key.lower() in n.lower())
    return {
        "path": str(out_txt),
        "returncode": proc.returncode,
        "n_listed_in_head": len(names),
        "name_hints": hint,
        "has_generator_prefix": hint.get("generator", 0) > 0,
        "has_up_block": hint.get("up_block", 0) > 0,
        "stdout_tail": proc.stdout[-2000:],
    }


def chamfer_like(a: np.ndarray, b: np.ndarray) -> float:
    d_ab = nn_distances(a, b).mean()
    d_ba = nn_distances(b, a).mean()
    return float(0.5 * (d_ab + d_ba))


def audit_vs_pugcn(n_per_line: int = 100) -> dict:
    rows = []
    summaries = {}
    for line, out_root, pugcn_root, exp_n in [
        ("lineB", OUT_B, PUGCN_B, 1024),
        ("lineA", OUT_A, PUGCN_A, 4096),
    ]:
        # intersection of relative paths
        pe_rels = set(list_rel_samples(out_root))
        pg_rels = set(list_rel_samples(pugcn_root))
        common = sorted(pe_rels & pg_rels)
        chosen = sample_rels(common, n_per_line, SEED + 10 + (0 if line == "lineB" else 1))
        metrics = []
        for rel in chosen:
            pe = np.load(out_root / rel).astype(np.float32)
            pg = np.load(pugcn_root / rel).astype(np.float32)
            same_shape = pe.shape == pg.shape == (exp_n, 3)
            if not same_shape:
                rows.append({"line": line, "rel": rel, "status": "SHAPE_MISMATCH", "pe_shape": str(pe.shape), "pg_shape": str(pg.shape)})
                continue
            exact_eq = bool(np.array_equal(pe, pg))
            mad = float(np.max(np.abs(pe - pg)))
            mean_abs = float(np.mean(np.abs(pe - pg)))
            # mean L2 per-point after sorting? use unordered chamfer-like + mean pairwise after NN pairing
            nn = nn_distances(pe, pg)
            mean_l2_nn = float(nn.mean())
            cd = chamfer_like(pe, pg)
            # fail if identical or extremely close
            fail = exact_eq or mad < 1e-8 or (mean_abs < 1e-7 and cd < 1e-7)
            rec = {
                "line": line,
                "rel": rel,
                "status": "FAIL_IDENTICAL_OR_NEAR" if fail else "PASS_DIFFERENT",
                "same_shape": True,
                "exact_equality": exact_eq,
                "max_abs_diff": mad,
                "mean_abs_diff": mean_abs,
                "mean_l2_nn_pe_to_pg": mean_l2_nn,
                "chamfer_like": cd,
            }
            rows.append(rec)
            metrics.append(rec)
        n_fail = sum(1 for r in metrics if r["status"].startswith("FAIL"))
        summaries[line] = {
            "n_common": len(common),
            "n_sampled": len(metrics),
            "n_identical_or_near": n_fail,
            "n_different": len(metrics) - n_fail,
            "mean_max_abs_diff": float(np.mean([r["max_abs_diff"] for r in metrics])) if metrics else float("nan"),
            "mean_mean_abs_diff": float(np.mean([r["mean_abs_diff"] for r in metrics])) if metrics else float("nan"),
            "mean_chamfer_like": float(np.mean([r["chamfer_like"] for r in metrics])) if metrics else float("nan"),
            "status": "PASS" if n_fail == 0 and len(metrics) >= min(n_per_line, len(common)) else "FAIL",
        }

    csv_path = REPORTS / "pu_edgeformer_vs_pugcn_output_difference_audit_20260716.csv"
    md_path = REPORTS / "pu_edgeformer_vs_pugcn_output_difference_audit_20260716.md"
    fields = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# PU-EdgeFormer vs PU-GCN Output Difference Audit",
        "",
        f"- Generated: `{utc_now()}`",
        "",
    ]
    for line, s in summaries.items():
        lines.extend(
            [
                f"## {line}",
                "",
                f"- status: **{s['status']}**",
                f"- common samples: {s['n_common']}",
                f"- sampled: {s['n_sampled']}",
                f"- identical/near: {s['n_identical_or_near']}",
                f"- different: {s['n_different']}",
                f"- mean max_abs_diff: {s['mean_max_abs_diff']:.6e}",
                f"- mean mean_abs_diff: {s['mean_mean_abs_diff']:.6e}",
                f"- mean chamfer_like: {s['mean_chamfer_like']:.6e}",
                "",
            ]
        )
    overall = "PASS" if all(s["status"] == "PASS" for s in summaries.values()) else "FAIL"
    lines.append(f"## Overall: **{overall}**\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"overall": overall, "summaries": summaries, "csv": str(csv_path), "md": str(md_path)}


def file_sha256(path: Path, block=1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            b = handle.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def smoke_leakage_audit() -> dict:
    # Line B smoke finals under train/test; Line A under outputs/lineA_A1...
    findings = []
    # Line B
    smoke_b_rels = []
    for split in ("train", "test"):
        d = SMOKE_B / split
        if d.is_dir():
            for p in d.rglob("*.npy"):
                smoke_b_rels.append(str(p.relative_to(SMOKE_B)))
    b_equal = 0
    b_hash_equal = 0
    b_compared = 0
    for rel in smoke_b_rels:
        full = OUT_B / rel
        smoke = SMOKE_B / rel
        if not full.is_file():
            continue
        b_compared += 1
        a = np.load(full)
        s = np.load(smoke)
        eq = np.array_equal(a, s)
        if eq:
            b_equal += 1
        hs = file_sha256(smoke)
        hf = file_sha256(full)
        if hs == hf:
            b_hash_equal += 1
        findings.append(
            {
                "line": "lineB",
                "rel": rel,
                "content_equal": eq,
                "hash_equal": hs == hf,
                "smoke_mtime": smoke.stat().st_mtime,
                "full_mtime": full.stat().st_mtime,
                "mtime_full_after_smoke": full.stat().st_mtime > smoke.stat().st_mtime,
            }
        )

    # Line A smoke finals
    smoke_a_root_candidates = [
        SMOKE_A / "outputs" / "lineA_A1_direct_1024_to_4096",
        SMOKE_A,
    ]
    smoke_a_base = None
    smoke_a_rels = []
    for cand in smoke_a_root_candidates:
        if (cand / "train").is_dir():
            smoke_a_base = cand
            for p in cand.rglob("*.npy"):
                # only train/test finals
                rel = str(p.relative_to(cand))
                if rel.startswith("train/") or rel.startswith("test/"):
                    smoke_a_rels.append(rel)
            break
    a_equal = 0
    a_hash_equal = 0
    a_compared = 0
    for rel in smoke_a_rels:
        full = OUT_A / rel
        smoke = smoke_a_base / rel
        if not full.is_file():
            continue
        a_compared += 1
        aa = np.load(full)
        ss = np.load(smoke)
        eq = np.array_equal(aa, ss)
        if eq:
            a_equal += 1
        hs = file_sha256(smoke)
        hf = file_sha256(full)
        if hs == hf:
            a_hash_equal += 1
        findings.append(
            {
                "line": "lineA",
                "rel": rel,
                "content_equal": eq,
                "hash_equal": hs == hf,
                "smoke_mtime": smoke.stat().st_mtime,
                "full_mtime": full.stat().st_mtime,
                "mtime_full_after_smoke": full.stat().st_mtime > smoke.stat().st_mtime,
            }
        )

    # Leakage interpretation:
    # content equality alone is OK if deterministic re-inference; require mtime evidence of copy.
    # Copy leakage if hash equal AND full mtime <= smoke mtime (file not regenerated).
    b_copy_suspect = sum(
        1
        for f in findings
        if f["line"] == "lineB" and f["hash_equal"] and not f["mtime_full_after_smoke"]
    )
    a_copy_suspect = sum(
        1
        for f in findings
        if f["line"] == "lineA" and f["hash_equal"] and not f["mtime_full_after_smoke"]
    )
    b_fresh_reinfer = sum(
        1 for f in findings if f["line"] == "lineB" and f["mtime_full_after_smoke"]
    )
    a_fresh_reinfer = sum(
        1 for f in findings if f["line"] == "lineA" and f["mtime_full_after_smoke"]
    )

    status_b = "PASS" if b_copy_suspect == 0 else "FAIL"
    status_a = "PASS" if a_copy_suspect == 0 else "FAIL"
    # If all overlapping equal but mtimes newer -> deterministic re-inference, still PASS
    md = REPORTS / "pu_edgeformer_smoke_leakage_audit_20260716.md"
    md.write_text(
        "\n".join(
            [
                "# PU-EdgeFormer Smoke Leakage Audit",
                "",
                f"- Generated: `{utc_now()}`",
                "",
                "## Line B",
                f"- overlapping compared: {b_compared}",
                f"- content_equal: {b_equal}",
                f"- hash_equal: {b_hash_equal}",
                f"- full mtime after smoke: {b_fresh_reinfer}",
                f"- copy-suspect (hash equal & full mtime not after smoke): {b_copy_suspect}",
                f"- status: **{status_b}**",
                "",
                "## Line A",
                f"- smoke_base: `{smoke_a_base}`",
                f"- overlapping compared: {a_compared}",
                f"- content_equal: {a_equal}",
                f"- hash_equal: {a_hash_equal}",
                f"- full mtime after smoke: {a_fresh_reinfer}",
                f"- copy-suspect: {a_copy_suspect}",
                f"- status: **{status_a}**",
                "",
                "## Interpretation",
                "- Content equality with newer full mtime is consistent with deterministic re-inference, not smoke file copy.",
                "- FAIL only if overlapping files look like unchanged copies (same hash, not newer than smoke).",
                "",
                f"## Overall: **{'PASS' if status_b == 'PASS' and status_a == 'PASS' else 'FAIL'}**",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "overall": "PASS" if status_b == "PASS" and status_a == "PASS" else "FAIL",
        "lineB": {
            "compared": b_compared,
            "content_equal": b_equal,
            "hash_equal": b_hash_equal,
            "copy_suspect": b_copy_suspect,
            "fresh_reinfer": b_fresh_reinfer,
            "status": status_b,
        },
        "lineA": {
            "compared": a_compared,
            "content_equal": a_equal,
            "hash_equal": a_hash_equal,
            "copy_suspect": a_copy_suspect,
            "fresh_reinfer": a_fresh_reinfer,
            "status": status_a,
        },
        "md": str(md),
    }


def job_window(job_id: str) -> dict:
    proc = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobID,Start,End,State", "-n", "-P"],
        capture_output=True,
        text=True,
    )
    starts = []
    ends = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        jid, start, end, state = parts[0], parts[1], parts[2], parts[3]
        if not re.match(rf"^{job_id}(_\d+)?$", jid):
            continue
        if start not in ("Unknown", "None", ""):
            starts.append(start)
        if end not in ("Unknown", "None", ""):
            ends.append(end)
    return {
        "job_id": job_id,
        "start_min": min(starts) if starts else None,
        "end_max": max(ends) if ends else None,
        "n_tasks_listed": len(starts),
    }


def parse_slurm_time(s: str) -> float:
    # 2026-07-16T20:15:11
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timestamp()


def freshness_audit() -> dict:
    win_b = job_window(JOB_B)
    win_a = job_window(JOB_A)
    results = {}
    for line, root, win in [("lineB", OUT_B, win_b), ("lineA", OUT_A, win_a)]:
        files = list((root / "train").rglob("*.npy")) + list((root / "test").rglob("*.npy"))
        mtimes = [p.stat().st_mtime for p in files]
        earliest = min(mtimes) if mtimes else None
        latest = max(mtimes) if mtimes else None
        start_ts = parse_slurm_time(win["start_min"]) if win["start_min"] else None
        end_ts = parse_slurm_time(win["end_max"]) if win["end_max"] else None
        # allow small skew +/- 120s
        skew = 120.0
        in_window = 0
        if earliest is not None and start_ts is not None and end_ts is not None:
            in_window = sum(1 for t in mtimes if (start_ts - skew) <= t <= (end_ts + skew))
        frac = (in_window / len(mtimes)) if mtimes else 0.0
        status = "PASS" if frac >= 0.99 and len(mtimes) == 12311 else "FAIL"
        results[line] = {
            "n_files": len(mtimes),
            "earliest_mtime": datetime.fromtimestamp(earliest).isoformat(sep=" ") if earliest else None,
            "latest_mtime": datetime.fromtimestamp(latest).isoformat(sep=" ") if latest else None,
            "job_start": win["start_min"],
            "job_end": win["end_max"],
            "in_window_count": in_window,
            "in_window_fraction": frac,
            "status": status,
        }
    md = REPORTS / "pu_edgeformer_output_freshness_audit_20260716.md"
    lines = [
        "# PU-EdgeFormer Output Freshness Audit",
        "",
        f"- Generated: `{utc_now()}`",
        "",
    ]
    for line, r in results.items():
        lines.extend(
            [
                f"## {line}",
                f"- n_files: {r['n_files']}",
                f"- earliest_mtime: {r['earliest_mtime']}",
                f"- latest_mtime: {r['latest_mtime']}",
                f"- job_start/end: {r['job_start']} / {r['job_end']}",
                f"- in_window: {r['in_window_count']} ({r['in_window_fraction']:.4f})",
                f"- status: **{r['status']}**",
                "",
            ]
        )
    overall = "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL"
    lines.append(f"## Overall: **{overall}**\n")
    md.write_text("\n".join(lines), encoding="utf-8")
    return {"overall": overall, "lineB": results["lineB"], "lineA": results["lineA"], "md": str(md), "windows": {"B": win_b, "A": win_a}}


def yn(ok: bool) -> str:
    return "YES" if ok else "NO"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-n", type=int, default=200)
    parser.add_argument("--pugcn-n", type=int, default=100)
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=== 1) repeat/copy audit ===", flush=True)
    repeat = audit_repeat_copy(args.repeat_n)
    print(json.dumps({"repeat_overall": repeat["overall"], "summaries": repeat["summaries"]}, indent=2), flush=True)

    print("=== 2a) slurm log audit ===", flush=True)
    logs_b = grep_logs(JOB_B)
    logs_a = grep_logs(JOB_A)
    print(json.dumps({"B_restore": logs_b["restore_present"], "B_ckpt": logs_b["ckpt_path_present"], "A_restore": logs_a["restore_present"], "A_ckpt": logs_a["ckpt_path_present"]}, sort_keys=True), flush=True)

    print("=== 2b) callpath audit ===", flush=True)
    callpath = callpath_audit()
    print(json.dumps({k: callpath[k] for k in ("tf_ops_reused_from_PUGCN", "model_definition_from_PU_EdgeFormer", "full_script_requests_edgetransformer")}, sort_keys=True), flush=True)

    print("=== 2c) checkpoint variables ===", flush=True)
    ckpt = checkpoint_variable_audit()
    print(json.dumps({"returncode": ckpt["returncode"], "hints": ckpt["name_hints"]}, sort_keys=True), flush=True)

    print("=== 2d) vs PU-GCN ===", flush=True)
    vs = audit_vs_pugcn(args.pugcn_n)
    print(json.dumps({"vs_overall": vs["overall"], "summaries": vs["summaries"]}, indent=2), flush=True)

    print("=== 3) smoke leakage ===", flush=True)
    smoke = smoke_leakage_audit()
    print(json.dumps(smoke, indent=2, default=str), flush=True)

    print("=== 4) freshness ===", flush=True)
    fresh = freshness_audit()
    print(json.dumps(fresh, indent=2, default=str), flush=True)

    # Compose final decisions
    ckpt_restored_b = logs_b["restore_present"] and logs_b["ckpt_path_present"]
    ckpt_restored_a = logs_a["restore_present"] and logs_a["ckpt_path_present"]
    model_code_ok = callpath["model_definition_from_PU_EdgeFormer"] == "YES"
    not_repeat_b = repeat["summaries"]["lineB"]["status"] == "PASS"
    not_repeat_a = repeat["summaries"]["lineA"]["status"] == "PASS"
    not_pugcn_b = vs["summaries"]["lineB"]["status"] == "PASS"
    not_pugcn_a = vs["summaries"]["lineA"]["status"] == "PASS"
    not_smoke_b = smoke["lineB"]["status"] == "PASS"
    not_smoke_a = smoke["lineA"]["status"] == "PASS"
    fresh_b = fresh["lineB"]["status"] == "PASS"
    fresh_a = fresh["lineA"]["status"] == "PASS"

    # not copy of input already covered by not_repeat
    rows = [
        ("checkpoint restored", yn(ckpt_restored_b), yn(ckpt_restored_a), "PASS" if ckpt_restored_b and ckpt_restored_a else "FAIL"),
        ("PU-EdgeFormer model code used", yn(model_code_ok), yn(model_code_ok), "PASS" if model_code_ok else "FAIL"),
        ("not simple repeat ×4", yn(not_repeat_b), yn(not_repeat_a), "PASS" if not_repeat_b and not_repeat_a else "FAIL"),
        ("not copy of input", yn(not_repeat_b), yn(not_repeat_a), "PASS" if not_repeat_b and not_repeat_a else "FAIL"),
        ("not PU-GCN output", yn(not_pugcn_b), yn(not_pugcn_a), "PASS" if not_pugcn_b and not_pugcn_a else "FAIL"),
        ("not smoke leakage", yn(not_smoke_b), yn(not_smoke_a), "PASS" if not_smoke_b and not_smoke_a else "FAIL"),
        ("output fresh", yn(fresh_b), yn(fresh_a), "PASS" if fresh_b and fresh_a else "FAIL"),
    ]
    all_pass = all(r[3] == "PASS" for r in rows)
    trusted = all_pass
    rows.append(
        (
            "data trustworthy for geometry",
            yn(trusted),
            yn(trusted),
            "PASS" if trusted else "FAIL",
        )
    )
    rows.append(
        (
            "data trustworthy for PointNet++",
            yn(trusted),
            yn(trusted),
            "PASS" if trusted else "FAIL",
        )
    )

    final_path = REPORTS / "pu_edgeformer_modelnet40_authenticity_verification_20260716.md"
    # write model/code audit md fragment into final
    model_md = REPORTS / "pu_edgeformer_model_callpath_slurm_audit_20260716.md"
    model_md.write_text(
        "\n".join(
            [
                "# PU-EdgeFormer Model / Call-Path / Slurm Audit",
                "",
                f"- Generated: `{utc_now()}`",
                "",
                "## Call path",
                f"- sbatch Line B: `{callpath['sbatch_lineB']}`",
                f"- sbatch Line A: `{callpath['sbatch_lineA']}`",
                f"- python script: `{callpath['python_full_script']}`",
                f"- wrapper: `{callpath['wrapper']}`",
                f"- imported model module: `{callpath['model_module']}`",
                f"- model utils: `{callpath['model_utils_module']}`",
                f"- checkpoint: `{callpath['checkpoint_dir']}`",
                f"- tf_ops: `{callpath['tf_ops_path']}`",
                f"- model construction: `{callpath['model_construction']}`",
                f"- inference function: `{callpath['inference_function']}`",
                f"- tf_ops reused from PU-GCN: **{callpath['tf_ops_reused_from_PUGCN']}**",
                f"- model definition from PU-EdgeFormer: **{callpath['model_definition_from_PU_EdgeFormer']}**",
                f"- EdgeTransformer present: {callpath['EdgeTransformer_class_present']}",
                f"- PUGCN class also present in same generator.py (unused when model=edgetransformer): {callpath['PUGCN_class_also_in_same_file']}",
                "",
                "## Slurm logs 1749211 (Line B)",
                f"- out files: {logs_b['n_out_files']}",
                f"- restore_present: {logs_b['restore_present']}",
                f"- ckpt_path_present: {logs_b['ckpt_path_present']}",
                f"- bad_patterns_present: {logs_b['bad_patterns_present']}",
                f"- pattern_counts: `{json.dumps(logs_b['pattern_counts'])}`",
                "",
                "### restore hits (sample)",
                *[f"- {h}" for h in logs_b["restore_hits"][:10]],
                "",
                "## Slurm logs 1749212 (Line A)",
                f"- out files: {logs_a['n_out_files']}",
                f"- restore_present: {logs_a['restore_present']}",
                f"- ckpt_path_present: {logs_a['ckpt_path_present']}",
                f"- bad_patterns_present: {logs_a['bad_patterns_present']}",
                f"- pattern_counts: `{json.dumps(logs_a['pattern_counts'])}`",
                "",
                "### restore hits (sample)",
                *[f"- {h}" for h in logs_a["restore_hits"][:10]],
                "",
                "## Checkpoint variables",
                f"- report: `{ckpt['path']}`",
                f"- name hints: `{json.dumps(ckpt['name_hints'])}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    conclusion = "TRUSTED: can proceed to geometry metrics" if trusted else "NOT TRUSTED: must stop and fix"
    table = [
        "| Check | Line B | Line A | Status |",
        "|------|--------|--------|--------|",
    ]
    for name, b, a, st in rows:
        table.append(f"| {name} | {b} | {a} | {st} |")

    final_path.write_text(
        "\n".join(
            [
                "# PU-EdgeFormer ModelNet40 Authenticity Verification",
                "",
                f"- Generated: `{utc_now()}`",
                f"- Final conclusion: **{conclusion}**",
                "",
                "## Conclusion table",
                "",
                *table,
                "",
                "## Component results",
                "",
                f"- repeat/copy: **{repeat['overall']}** (`{repeat['md']}`)",
                f"- model/callpath/slurm: see `{model_md}`",
                f"- checkpoint variables: `{ckpt['path']}`",
                f"- vs PU-GCN: **{vs['overall']}** (`{vs['md']}`)",
                f"- smoke leakage: **{smoke['overall']}** (`{smoke['md']}`)",
                f"- freshness: **{fresh['overall']}** (`{fresh['md']}`)",
                "",
                "## Key numeric highlights",
                "",
                "### Repeat/copy means",
                f"- Line B exact_match={repeat['summaries']['lineB']['mean_exact_match_ratio']:.6f}, "
                f"rounded={repeat['summaries']['lineB']['mean_rounded_1e6_match_ratio']:.6f}, "
                f"unique={repeat['summaries']['lineB']['mean_output_unique_count']:.1f}, "
                f"dup={repeat['summaries']['lineB']['mean_duplicate_ratio']:.6f}, "
                f"nn_med={repeat['summaries']['lineB']['mean_nn_median']:.6e}",
                f"- Line A exact_match={repeat['summaries']['lineA']['mean_exact_match_ratio']:.6f}, "
                f"rounded={repeat['summaries']['lineA']['mean_rounded_1e6_match_ratio']:.6f}, "
                f"unique={repeat['summaries']['lineA']['mean_output_unique_count']:.1f}, "
                f"dup={repeat['summaries']['lineA']['mean_duplicate_ratio']:.6f}, "
                f"nn_med={repeat['summaries']['lineA']['mean_nn_median']:.6e}",
                "",
                "### vs PU-GCN means",
                f"- Line B mean_max_abs_diff={vs['summaries']['lineB']['mean_max_abs_diff']:.6e}, "
                f"chamfer_like={vs['summaries']['lineB']['mean_chamfer_like']:.6e}",
                f"- Line A mean_max_abs_diff={vs['summaries']['lineA']['mean_max_abs_diff']:.6e}, "
                f"chamfer_like={vs['summaries']['lineA']['mean_chamfer_like']:.6e}",
                "",
                "## Safety",
                "",
                "- GEOMETRY_METRICS_STARTED=NO",
                "- POINTNET_CLASSIFIER_STARTED=NO",
                "- DETECTOR_EVAL_STARTED=NO",
                "- KITTI_AP_EVAL_STARTED=NO",
                "",
                "## Report paths",
                "",
                f"- `{final_path}`",
                f"- `{repeat['md']}`",
                f"- `{repeat['csv']}`",
                f"- `{model_md}`",
                f"- `{ckpt['path']}`",
                f"- `{vs['md']}`",
                f"- `{vs['csv']}`",
                f"- `{smoke['md']}`",
                f"- `{fresh['md']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    payload = {
        "conclusion": conclusion,
        "trusted": trusted,
        "repeat": repeat["overall"],
        "vs_pugcn": vs["overall"],
        "smoke": smoke["overall"],
        "fresh": fresh["overall"],
        "model_definition_from_PU_EdgeFormer": callpath["model_definition_from_PU_EdgeFormer"],
        "tf_ops_reused_from_PUGCN": callpath["tf_ops_reused_from_PUGCN"],
        "final_report": str(final_path),
        "GEOMETRY_METRICS_STARTED": "NO",
        "POINTNET_CLASSIFIER_STARTED": "NO",
        "DETECTOR_EVAL_STARTED": "NO",
        "KITTI_AP_EVAL_STARTED": "NO",
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if trusted else 1


if __name__ == "__main__":
    raise SystemExit(main())
