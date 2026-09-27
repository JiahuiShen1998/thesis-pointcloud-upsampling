#!/usr/bin/env python3
"""Resume the observed-first matrix while protecting it from shared-GPU races."""

from __future__ import annotations

import subprocess
import os
import signal
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results/pugcn_full_retrain_20260824/pointrcnn_observed_first_run"
STATUS_LOG = RUN_ROOT / "gpu_wait.log"
MATRIX = ROOT / "scripts/run_pointrcnn_observed_first_matrix_20260826.sh"
POLL_SECONDS = 30
RUN_POLL_SECONDS = 10
STABLE_IDLE_SECONDS = 300
MAX_WAIT_SECONDS = 48 * 60 * 60
MAX_RETRIES = 3


def log(message: str) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with STATUS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def compute_processes() -> list[tuple[int, str]] | None:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as error:
        log(f"nvidia-smi unavailable: {error}")
        return None
    processes: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line.split(",", 1)[0].strip())
        except ValueError:
            continue
        processes.append((pid, line))
    return processes


def pid_is_ours(pid: int) -> bool:
    try:
        return (Path("/proc") / str(pid)).stat().st_uid == os.getuid()
    except OSError:
        return False


def external_prelaunch_candidates() -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,uid=,command="],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ["PROCESS_QUERY_FAILED"]
    candidates: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            uid = int(parts[1])
        except ValueError:
            continue
        command = parts[2]
        if uid != os.getuid() and "lossy_attribute.train" in command:
            candidates.append(line.strip())
    return candidates


def terminate_matrix(process: subprocess.Popen[object]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> int:
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    retries = 0
    while time.monotonic() < deadline and retries < MAX_RETRIES:
        idle_since: float | None = None
        previous: tuple[str, ...] | None = None
        log("waiting for a stable idle GPU before resuming PointRCNN")
        while time.monotonic() < deadline:
            processes = compute_processes()
            candidates = external_prelaunch_candidates()
            descriptions = (
                [item[1] for item in processes] if processes is not None else ["GPU_QUERY_FAILED"]
            ) + candidates
            snapshot = tuple(descriptions)
            if snapshot != previous:
                log("blockers: " + ("; ".join(snapshot) if snapshot else "NONE"))
                previous = snapshot
            if processes == [] and candidates == []:
                if idle_since is None:
                    idle_since = time.monotonic()
                    log(f"GPU idle; starting {STABLE_IDLE_SECONDS}s stability window")
                elif time.monotonic() - idle_since >= STABLE_IDLE_SECONDS:
                    log("stable idle window passed; resuming matrix")
                    break
            else:
                idle_since = None
            time.sleep(POLL_SECONDS)
        else:
            log("timed out waiting for the shared GPU")
            return 1

        matrix_log = RUN_ROOT / "matrix.log"
        interrupted = False
        with matrix_log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                ["bash", str(MATRIX)],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log(f"matrix started pid={process.pid}")
            while process.poll() is None:
                time.sleep(RUN_POLL_SECONDS)
                gpu_processes = compute_processes()
                if gpu_processes is None:
                    continue
                external = [item for item in gpu_processes if not pid_is_ours(item[0])]
                if external:
                    log(
                        "external GPU process appeared; interrupting matrix: "
                        + "; ".join(item[1] for item in external)
                    )
                    terminate_matrix(process)
                    interrupted = True
                    break
            code = process.returncode
        retries += 1
        log(f"matrix exit_code={code} attempt={retries}/{MAX_RETRIES}")
        if code == 0:
            return 0
        if interrupted:
            log("matrix returned to the GPU queue without discarding completed stages")
        else:
            log("matrix failed without a detected external GPU process; retrying after idle")
        time.sleep(POLL_SECONDS)

    log("retry or wait limit reached")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
