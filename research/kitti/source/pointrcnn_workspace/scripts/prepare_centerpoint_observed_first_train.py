#!/usr/bin/env python3
"""Build deterministic observed-first CenterPoint training inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


BASE_SEED = 20260718


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if not values.size or values.size % 4:
        raise ValueError(f"invalid KITTI XYZI file: {path}")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"NaN/Inf in {path}")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--observed-dir", type=Path, required=True)
    parser.add_argument("--predicted-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-token", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    frames = [
        line.strip()
        for line in args.split_file.read_text().splitlines()
        if line.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, frame in enumerate(frames, start=1):
        observed = read_bin(args.observed_dir / f"{frame}.bin")
        predicted = read_bin(args.predicted_dir / f"{frame}.bin")
        expected = 4 * observed.shape[0]
        if predicted.shape[0] != expected:
            raise ValueError(
                f"{frame}: predicted={predicted.shape[0]} expected={expected}"
            )
        output = args.output_dir / f"{frame}.bin"
        seed = stable_seed(BASE_SEED, "e1", args.reference_token, frame)
        rng = np.random.default_rng(seed)
        selected = rng.choice(expected, size=3 * observed.shape[0], replace=False)
        if output.is_file() and output.stat().st_size == expected * 16:
            status = "PASS_REUSED"
        else:
            final = np.concatenate((observed, predicted[selected]), axis=0).astype(
                np.float32, copy=False
            )
            final.tofile(output)
            status = "PASS"
        stored = read_bin(output)
        observed_prefix_exact = np.array_equal(stored[:len(observed)], observed)
        generated_suffix_exact = np.array_equal(stored[len(observed):], predicted[selected])
        if not observed_prefix_exact or not generated_suffix_exact:
            raise ValueError(f"{frame}: stored observed-first input fails exact sampling audit")
        rows.append(
            {
                "frame_id": frame,
                "reference_token": args.reference_token,
                "seed": seed,
                "observed_points": int(observed.shape[0]),
                "predicted_points": int(predicted.shape[0]),
                "selected_generated_points": int(selected.size),
                "selected_indices_sha256": hashlib.sha256(
                    selected.astype("<i8").tobytes()
                ).hexdigest(),
                "observed_prefix_exact": observed_prefix_exact,
                "generated_suffix_exact": generated_suffix_exact,
                "output_float32_sha256": hashlib.sha256(stored.tobytes()).hexdigest(),
                "output_points": expected,
                "output": str(output.resolve()),
                "status": status,
            }
        )
        if index == 1 or index % 128 == 0 or index == len(frames):
            print(f"CENTERPOINT_PREP {index}/{len(frames)} {frame}", flush=True)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CENTERPOINT_OBSERVED_FIRST_PASS={len(rows)}")


if __name__ == "__main__":
    main()
