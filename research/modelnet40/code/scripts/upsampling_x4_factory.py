#!/usr/bin/env python3
"""Load upsampler for a main-protocol method."""

from __future__ import annotations

from typing import Any

METHODS = ("pdans", "punet", "pugcn")


def normalize_method(method: str) -> str:
    return method.strip().lower().rstrip("}")


def load_upsampler(method: str, **kwargs: Any):
    method = normalize_method(method)
    if method == "pdans":
        from pdans_modelnet40_utils import load_pdans_upsampler

        return load_pdans_upsampler(**kwargs)
    if method == "punet":
        from punet_modelnet40_utils import load_punet_upsampler

        return load_punet_upsampler(**kwargs)
    if method == "pugcn":
        from pugcn_modelnet40_utils import load_pugcn_upsampler

        return load_pugcn_upsampler(**kwargs)
    raise ValueError(f"Unknown method: {method}. Expected one of {METHODS}")
