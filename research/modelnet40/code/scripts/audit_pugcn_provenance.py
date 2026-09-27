#!/usr/bin/env python3
"""Provenance audit for PU-GCN two-line protocol (spot-check samples)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    GLOBAL_SEED,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    lineA_paths,
    lineB_paths,
    reports_dir,
)

EAR_STRICT = PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "strict_N" / "ear"
PDANS_STRICT = PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "strict_N" / "pdans"
PUNET_STRICT = PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "strict_N" / "pu_net"
LINEA_STRICT = lineA_paths("pu_gcn")["strict_4N"]


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_samples(input_root: Path, n_pick: int, seed: int) -> list[tuple[str, str, str]]:
    keys: list[tuple[str, str, str]] = []
    for split in ("train", "test"):
        mp = input_root / "metadata" / f"{split}_manifest.csv"
        with open(mp, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                keys.append((split, row["class_name"], row["shape_id"]))
    rng = random.Random(seed)
    rng.shuffle(keys)
    return keys[:n_pick]


def audit_sample(line: str, split: str, cls: str, sid: str, n: int) -> dict:
    if line == "lineA":
        inp = ORIGINAL_ROOT / split / cls / f"{sid}.npy"
        raw = lineA_paths("pu_gcn")["raw"] / split / cls / f"{sid}.npy"
        strict = lineA_paths("pu_gcn")["strict_4N"] / split / cls / f"{sid}.npy"
        exp_in, exp_out = n, n * 4
        log_evidence = f"run_pugcn_modelnet40_x4.py --line lineA; input={ORIGINAL_ROOT}"
    else:
        inp = DOWNSAMPLED_X4_ROOT / split / cls / f"{sid}.npy"
        raw = lineB_paths("pu_gcn")["raw"] / split / cls / f"{sid}.npy"
        strict = lineB_paths("pu_gcn")["strict_N"] / split / cls / f"{sid}.npy"
        exp_in, exp_out = n // 4, n
        log_evidence = f"run_pugcn_modelnet40_x4.py --line lineB; input={DOWNSAMPLED_X4_ROOT}"

    rec = {
        "line": line,
        "sample": f"{split}/{cls}/{sid}",
        "input_path": str(inp),
        "input_shape": "",
        "input_hash": "",
        "raw_path": str(raw),
        "raw_shape": "",
        "raw_hash": "",
        "strict_path": str(strict),
        "strict_shape": "",
        "strict_hash": "",
        "inference_log_evidence": log_evidence,
        "status": "FAIL",
        "note": "",
    }
    issues: list[str] = []

    try:
        if not inp.is_file():
            raise FileNotFoundError(f"missing input {inp}")
        inp_arr = np.load(inp)
        rec["input_shape"] = str(tuple(inp_arr.shape))
        rec["input_hash"] = file_md5(inp)
        if inp_arr.shape != (exp_in, 3):
            issues.append(f"wrong input shape {inp_arr.shape}")

        if not strict.is_file():
            raise FileNotFoundError(f"missing strict {strict}")
        strict_arr = np.load(strict)
        rec["strict_shape"] = str(tuple(strict_arr.shape))
        rec["strict_hash"] = file_md5(strict)
        if strict_arr.shape != (exp_out, 3):
            issues.append(f"wrong strict shape {strict_arr.shape}")

        if raw.is_file():
            raw_arr = np.load(raw)
            rec["raw_shape"] = str(tuple(raw_arr.shape))
            rec["raw_hash"] = file_md5(raw)
        else:
            issues.append("missing raw output")

        # Contamination checks
        if line == "lineB":
            for other_name, other_root in (
                ("EAR", EAR_STRICT),
                ("PDANS", PDANS_STRICT),
                ("PU-Net", PUNET_STRICT),
            ):
                other_p = other_root / split / cls / f"{sid}.npy"
                if other_p.is_file() and strict.is_file():
                    if file_md5(other_p) == rec["strict_hash"]:
                        issues.append(f"strict hash matches {other_name} output — contamination")
            linea_p = LINEA_STRICT / split / cls / f"{sid}.npy"
            if linea_p.is_file() and strict.is_file():
                la = np.load(linea_p)
                if la.shape[0] == exp_out * 4:
                    pass  # different point count
                elif file_md5(linea_p) == rec["strict_hash"]:
                    issues.append("strict hash matches Line A output — contamination")

        legacy_b = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "pugcn_x4" / split / cls / f"{sid}.npy"
        if legacy_b.is_file() and strict.is_file() and file_md5(legacy_b) == rec["strict_hash"]:
            issues.append("matches old protocol 512->2048 output")

        if issues:
            rec["note"] = "; ".join(issues)
        else:
            rec["status"] = "PASS"
            rec["note"] = (
                "no evidence of EAR/PDANS/PU-Net reuse; "
                "no Line A output for Line B; no old protocol reuse"
            )
    except Exception as exc:  # noqa: BLE001
        rec["note"] = str(exc)

    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-line", type=int, default=20)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    n = detect_original_point_count()
    rows: list[dict] = []
    for line, root in (("lineA", ORIGINAL_ROOT), ("lineB", DOWNSAMPLED_X4_ROOT)):
        for split, cls, sid in pick_samples(root, args.n_per_line, stable_seed(args.seed, line)):
            rows.append(audit_sample(line, split, cls, sid, n))

    rep = reports_dir()
    csv_path = rep / "modelnet40_pugcn_provenance_audit.csv"
    md_path = rep / "modelnet40_pugcn_provenance_audit.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for r in rows if r["status"] == "PASS")
    overall = "PASS" if passed == len(rows) and rows else "FAIL"

    md_lines = [
        "# PU-GCN Provenance Audit",
        "",
        f"- Generated at: {ts}",
        f"- Spot checks: {len(rows)} ({args.n_per_line} per line)",
        f"- Overall: **{overall}** ({passed}/{len(rows)} PASS)",
        "",
        "**Contamination policy:**",
        "- No evidence of using EAR / PDANS / PU-Net outputs",
        "- No evidence of using Line A output for Line B",
        "- No evidence of old protocol (512→2048) reuse",
        "",
        "| line | sample | input_shape | strict_shape | status | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['line']} | {r['sample']} | {r['input_shape']} | {r['strict_shape']} | "
            f"{r['status']} | {r['note'][:80]} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"overall={overall} pass={passed}/{len(rows)}")
    print(f"Wrote {csv_path}")
    return 0 if overall == "PASS" else 1


def stable_seed(global_seed: int, *parts: str) -> int:
    token = ":".join([str(global_seed), *parts]).encode("utf-8")
    digest = hashlib.md5(token).hexdigest()
    return global_seed + (int(digest[:8], 16) % 1_000_000)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
