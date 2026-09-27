#!/usr/bin/env python3
"""Bootstrap the official PU-Net entrypoint with GPU-friendly TensorFlow config.

This wrapper leaves the official PU-Net sources untouched while intercepting
the TensorFlow session config construction at runtime. The upstream prediction
path hard-codes `tf.ConfigProto(device_count={'GPU': 0})`, which disables GPU
usage even when CUDA is visible. The smoke test uses this bootstrap only for
verification; it does not change the repository code.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _patch_tensorflow_configproto() -> None:
    import tensorflow as tf  # Imported here so the module object can be patched.

    original_config_proto = tf.ConfigProto

    def patched_config_proto(*args, **kwargs):
        device_count = kwargs.get("device_count")
        if isinstance(device_count, dict) and device_count.get("GPU") == 0:
            device_count = dict(device_count)
            device_count.pop("GPU", None)
            if device_count:
                kwargs["device_count"] = device_count
            else:
                kwargs.pop("device_count", None)
        return original_config_proto(*args, **kwargs)

    tf.ConfigProto = patched_config_proto

    print(
        "[punet-gpu-bootstrap] tensorflow_version=%s gpu_available=%s cuda_visible_devices=%s"
        % (
            getattr(tf, "__version__", "unknown"),
            tf.test.is_gpu_available(cuda_only=True),
            os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        ),
        flush=True,
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: punet_tf_gpu_bootstrap.py <official-punet-main.py> [args...]"
        )

    script = Path(sys.argv[1]).resolve()
    if not script.exists():
        raise FileNotFoundError(str(script))

    os.environ.pop("PU_NET_FORCE_CPU", None)
    os.chdir(str(script.parent))
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    _patch_tensorflow_configproto()

    # Preserve the original CLI for the official entrypoint.
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
