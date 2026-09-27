#!/usr/bin/env python3
"""Audit that every output point cloud has an exact byte-count ratio."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratio", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    frame_ids = [line.strip() for line in args.split_file.read_text().splitlines() if line.strip()]
    failures = []
    passed = 0

    for frame_id in frame_ids:
        input_path = args.input_dir / f"{frame_id}.bin"
        output_path = args.output_dir / f"{frame_id}.bin"
        if not input_path.is_file() or not output_path.is_file():
            failures.append({"frame_id": frame_id, "reason": "missing_file"})
            continue
        expected = input_path.stat().st_size * args.ratio
        actual = output_path.stat().st_size
        if actual != expected:
            failures.append(
                {"frame_id": frame_id, "reason": "byte_count", "expected": expected, "actual": actual}
            )
            continue
        passed += 1

    report = {
        "status": "PASS" if passed == len(frame_ids) else "FAIL",
        "frames": len(frame_ids),
        f"strict_{args.ratio}n_pass": passed,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
