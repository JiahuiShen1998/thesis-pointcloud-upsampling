#!/usr/bin/env python3
"""Combine an adapted RPN checkpoint with the official PointRCNN RCNN weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--rpn", type=Path, required=True)
    parser.add_argument("--rcnn", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official = torch.load(str(args.official), map_location="cpu")
    adapted_rpn = torch.load(str(args.rpn), map_location="cpu")
    official_state = dict(official["model_state"])
    rpn_state = adapted_rpn["model_state"]
    unknown = sorted(set(rpn_state) - set(official_state))
    if unknown:
        raise RuntimeError(f"adapted RPN has keys absent from official model: {unknown[:10]}")
    official_state.update(rpn_state)
    rcnn_state = None
    if args.rcnn is not None:
        adapted_rcnn = torch.load(str(args.rcnn), map_location="cpu")
        rcnn_state = adapted_rcnn["model_state"]
        unknown = sorted(set(rcnn_state) - set(official_state))
        if unknown:
            raise RuntimeError(f"adapted RCNN has keys absent from official model: {unknown[:10]}")
        official_state.update(rcnn_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": official_state}, str(args.output))
    print(f"COMBINED_KEYS={len(official_state)}")
    print(f"ADAPTED_RPN_KEYS={len(rpn_state)}")
    print(f"ADAPTED_RCNN_KEYS={len(rcnn_state) if rcnn_state is not None else 0}")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
