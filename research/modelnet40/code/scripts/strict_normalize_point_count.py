#!/usr/bin/env python3
"""Strict normalize point clouds to exact target count (deterministic)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import GLOBAL_SEED, reports_dir, resample_to_target, stable_seed


def normalize_file(src: Path, dst: Path, target_n: int, seed: int) -> dict:
    points = np.load(src).astype(np.float32)
    raw_n = points.shape[0]
    out = resample_to_target(points, target_n, seed)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, out)
    action = "unchanged" if raw_n == target_n else ("truncate" if raw_n > target_n else "pad")
    return {
        "source": str(src),
        "output": str(dst),
        "raw_points": raw_n,
        "target_points": target_n,
        "final_points": int(out.shape[0]),
        "action": action,
        "status": "ok" if out.shape[0] == target_n else "fail",
    }


def normalize_tree(input_root: Path, output_root: Path, target_n: int, audit_name: str) -> list[dict]:
    rows: list[dict] = []
    for src in sorted(input_root.rglob("*.npy")):
        rel = src.relative_to(input_root)
        dst = output_root / rel
        parts = rel.parts
        seed_parts = list(parts[:-1]) + [parts[-1].replace(".npy", "")]
        seed = stable_seed(GLOBAL_SEED, "strict_norm", audit_name, *seed_parts)
        rows.append(normalize_file(src, dst, target_n, seed))
    return rows


def write_audit(rows: list[dict], audit_name: str) -> tuple[Path, Path]:
    rep = reports_dir()
    rep.mkdir(parents=True, exist_ok=True)
    csv_path = rep / f"{audit_name}.csv"
    md_path = rep / f"{audit_name}.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    actions = {}
    for r in rows:
        actions[r["action"]] = actions.get(r["action"], 0) + 1
    bad = [r for r in rows if r["status"] != "ok"]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)

    md_path.write_text(
        "\n".join([
            f"# Point Count Audit — {audit_name}",
            "",
            f"- Generated at: {ts}",
            f"- Total files: {len(rows)}",
            f"- Target points: {rows[0]['target_points'] if rows else 'N/A'}",
            f"- Actions: {actions}",
            f"- Failures: {len(bad)}",
        ]),
        encoding="utf-8",
    )
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-n", type=int, required=True)
    parser.add_argument("--audit-name", type=str, required=True)
    args = parser.parse_args()

    rows = normalize_tree(args.input_root, args.output_root, args.target_n, args.audit_name)
    csv_path, md_path = write_audit(rows, args.audit_name)
    fails = sum(1 for r in rows if r["status"] != "ok")
    print(f"Normalized {len(rows)} files -> {args.output_root}, failures={fails}")
    print(f"Audit: {csv_path}, {md_path}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
