#!/usr/bin/env python3
"""Sanity-check upsampling outputs vs inputs (detect obvious duplicate/resample artifacts)."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LINE_CONFIG = {
    "A": {
        "input_root": PROJECT_ROOT / "datasets" / "modelnet40_original",
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_original_up" / "ear_x4",
        "expected_input": 1024,
        "expected_output": 4096,
    },
    "B": {
        "input_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50",
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear_x4",
        "expected_input": 512,
        "expected_output": 2048,
    },
}


def line_config_for_method(method: str, line: str) -> dict:
    subdir = {"EAR": "ear_x4", "PDANS": "pdans_x4"}.get(
        method.upper(), f"{method.lower()}_x4"
    )
    if line == "A":
        return {
            "input_root": PROJECT_ROOT / "datasets" / "modelnet40_original",
            "output_root": PROJECT_ROOT / "datasets" / "modelnet40_original_up" / subdir,
            "expected_input": 1024,
            "expected_output": 4096,
        }
    return {
        "input_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50",
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / subdir,
        "expected_input": 512,
        "expected_output": 2048,
    }


def list_pairs(input_root: Path, output_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for split in ("train", "test"):
        split_out = output_root / split
        if not split_out.is_dir():
            continue
        for npy_path in sorted(split_out.rglob("*.npy")):
            rel = npy_path.relative_to(split_out)
            inp = input_root / split / rel
            if inp.is_file():
                pairs.append((inp, npy_path))
    return pairs


def exact_duplicate_ratio(points: np.ndarray, decimals: int = 6) -> float:
    if points.shape[0] == 0:
        return 0.0
    rounded = np.round(points, decimals=decimals)
    _, counts = np.unique(rounded, axis=0, return_counts=True)
    dup = int((counts - 1).clip(min=0).sum())
    return dup / points.shape[0]


def count_input_points_in_output(input_pts: np.ndarray, output_pts: np.ndarray, atol: float = 1e-6) -> int:
    """Count output points that exactly match some input point (within atol)."""
    if input_pts.shape[0] == 0 or output_pts.shape[0] == 0:
        return 0
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(input_pts)
    dist, _ = nbrs.kneighbors(output_pts)
    return int((dist[:, 0] <= atol).sum())


def nn_distance_stats(input_pts: np.ndarray, output_pts: np.ndarray) -> dict[str, float]:
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(input_pts)
    dist, _ = nbrs.kneighbors(output_pts)
    d = dist[:, 0]
    return {
        "nn_mean": float(np.mean(d)),
        "nn_median": float(np.median(d)),
        "nn_p95": float(np.percentile(d, 95)),
    }


def analyze_pair(inp: Path, out: Path, cfg: dict) -> dict:
    input_pts = np.load(inp).astype(np.float64)
    output_pts = np.load(out).astype(np.float64)
    n_in = input_pts.shape[0]
    n_out = output_pts.shape[0]
    ratio = n_out / n_in if n_in else float("nan")
    dup_ratio = exact_duplicate_ratio(output_pts)
    unique_out = int(np.unique(np.round(output_pts, 6), axis=0).shape[0])
    input_exact_in_output = count_input_points_in_output(input_pts, output_pts)
    nn = nn_distance_stats(input_pts, output_pts)
    return {
        "input_path": str(inp),
        "output_path": str(out),
        "input_points": n_in,
        "output_points": n_out,
        "output_input_ratio": round(ratio, 4),
        "exact_duplicate_output_ratio": round(dup_ratio, 6),
        "unique_output_points": unique_out,
        "output_points_matching_input_exact": input_exact_in_output,
        "output_points_matching_input_exact_ratio": round(input_exact_in_output / n_out, 6) if n_out else 0.0,
        "nn_mean_output_to_input": round(nn["nn_mean"], 8),
        "nn_median_output_to_input": round(nn["nn_median"], 8),
        "nn_p95_output_to_input": round(nn["nn_p95"], 8),
        "expected_input": cfg["expected_input"],
        "expected_output": cfg["expected_output"],
        "count_ok": n_in == cfg["expected_input"] and n_out == cfg["expected_output"],
        "heavy_duplicate_flag": dup_ratio > 0.5,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md(rows: list[dict], path: Path, method: str, line: str, n_sampled: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dup_ratios = [r["exact_duplicate_output_ratio"] for r in rows]
    input_match = [r["output_points_matching_input_exact_ratio"] for r in rows]
    nn_mean = [r["nn_mean_output_to_input"] for r in rows]
    heavy = sum(1 for r in rows if r["heavy_duplicate_flag"])
    lines = [
        f"# {method} ×4 Output Similarity Sanity — Line {line}",
        "",
        f"Samples analyzed: **{len(rows)}** (requested ≥ {n_sampled})",
        "",
        "## Summary",
        "",
        f"- Mean exact-duplicate output ratio: **{np.mean(dup_ratios):.4f}**",
        f"- Mean output→input exact-match ratio: **{np.mean(input_match):.4f}**",
        f"- Mean NN distance (output→input): **{np.mean(nn_mean):.6f}**",
        f"- Samples with >50% exact duplicate outputs: **{heavy}**",
        "",
        "## Interpretation",
        "",
        "High exact-duplicate ratio or near-zero NN distance for most new points would suggest",
        "trivial resampling (repeat/random duplicate) rather than geometric upsampling.",
        "This check alone cannot prove a specific upsampling method was used.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit upsampling output vs input similarity.")
    parser.add_argument("--method", default="EAR")
    parser.add_argument("--line", choices=("A", "B", "both"), default="both")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ear_x4_output_similarity_sanity.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "reports" / "ear_x4_output_similarity_sanity.md",
    )
    args = parser.parse_args()

    lines = ["A", "B"] if args.line == "both" else [args.line]
    all_rows: list[dict] = []
    md_sections: list[str] = [
        f"# {args.method} ×4 Output Similarity Sanity Check",
        "",
        f"Method: **{args.method}** | Seed: {args.seed} | Samples per line: ≥{args.n_samples}",
        "",
        "> This is a data-layer sanity check only; it cannot prove method provenance by itself.",
        "",
    ]

    for line in lines:
        cfg = line_config_for_method(args.method, line)
        pairs = list_pairs(cfg["input_root"], cfg["output_root"])
        if len(pairs) < args.n_samples:
            print(f"WARNING: Line {line} has only {len(pairs)} pairs", file=sys.stderr)
        rng = random.Random(args.seed + (0 if line == "A" else 1))
        sampled = rng.sample(pairs, min(args.n_samples, len(pairs)))
        rows = []
        for inp, out in sampled:
            row = analyze_pair(inp, out, cfg)
            row["line"] = line
            row["method"] = args.method
            rows.append(row)
        all_rows.extend(rows)

        dup = [r["exact_duplicate_output_ratio"] for r in rows]
        match = [r["output_points_matching_input_exact_ratio"] for r in rows]
        nn_m = [r["nn_mean_output_to_input"] for r in rows]
        md_sections.extend([
            f"## Line {line}",
            "",
            f"- Pairs sampled: {len(rows)}",
            f"- Mean duplicate ratio: {np.mean(dup):.4f}",
            f"- Mean input-exact-in-output ratio: {np.mean(match):.4f}",
            f"- Mean NN distance: {np.mean(nn_m):.6f}",
            f"- Heavy duplicate (>50%): {sum(1 for r in rows if r['heavy_duplicate_flag'])}",
            "",
        ])

    write_csv(all_rows, args.output_csv)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md_sections), encoding="utf-8")
    print(f"Wrote {len(all_rows)} rows to {args.output_csv}")
    print(f"Wrote summary to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
