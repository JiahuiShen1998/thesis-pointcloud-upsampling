#!/usr/bin/env python3
"""Report whether the current PointRCNN worktree touches any core detector files."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DANGEROUS_PREFIXES = (
    "lib/",
    "pointnet2_lib/",
    "cfgs/",
    "tools/eval_rcnn.py",
    "tools/train_rcnn.py",
    "tools/train_rpn.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether core PointRCNN files were modified")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the integrity report.",
    )
    parser.add_argument(
        "--fail-on-dangerous",
        action="store_true",
        help="Exit with code 2 if a dangerous file is detected.",
    )
    return parser.parse_args()


def run_git(args: Sequence[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout.strip()


def classify_files(paths: Iterable[str]) -> tuple[list[str], list[str]]:
    dangerous: List[str] = []
    safe: List[str] = []
    for path in paths:
        if any(path == prefix or path.startswith(prefix) for prefix in DANGEROUS_PREFIXES):
            dangerous.append(path)
        else:
            safe.append(path)
    return dangerous, safe


def build_report() -> tuple[str, bool]:
    name_only = [line.strip() for line in run_git(["diff", "--name-only"]).splitlines() if line.strip()]
    status = run_git(["status", "--short"])
    commit = run_git(["rev-parse", "--short", "HEAD"])
    dangerous, safe = classify_files(name_only)

    lines = [
        "PointRCNN integrity check",
        f"repo_root={REPO_ROOT}",
        f"git_commit={commit}",
        "",
        "git diff --name-only:",
    ]
    if name_only:
        lines.extend(f"  {path}" for path in name_only)
    else:
        lines.append("  (clean)")

    lines.extend(["", "git status --short:"])
    if status:
        lines.extend(f"  {line}" for line in status.splitlines())
    else:
        lines.append("  (clean)")

    lines.extend(["", "dangerous_files:"])
    if dangerous:
        lines.extend(f"  WARNING {path}" for path in dangerous)
    else:
        lines.append("  none")

    lines.extend(["", "safe_changes:"])
    if safe:
        lines.extend(f"  {path}" for path in safe)
    else:
        lines.append("  none")

    ok = len(dangerous) == 0
    if ok:
        lines.append("")
        lines.append("Result: no core detector files were modified by the density-baseline tooling.")
    else:
        lines.append("")
        lines.append("Result: core detector files are present in the current diff and should be reviewed.")
    return "\n".join(lines) + "\n", ok


def main() -> int:
    args = parse_args()
    report, ok = build_report()
    print(report, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    if args.fail_on_dangerous and not ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
