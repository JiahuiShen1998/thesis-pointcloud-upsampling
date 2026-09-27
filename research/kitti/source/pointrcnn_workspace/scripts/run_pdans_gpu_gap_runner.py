#!/usr/bin/env python3
"""Run PDANS missing-frame resume only when the shared GPU has a stable gap."""

from __future__ import annotations

import argparse
import getpass
import os
import pwd
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "kitti_unified_x4_current_methods_no_detector"
AUDIT_DIR = RESULT_ROOT / "audits"
LOG_PATH = AUDIT_DIR / "pdans_gpu_gap_runner.log"
EAR_PY = Path("/home/ra87racy/miniconda3/envs/ear/bin/python")
LINE_A = "line_a_original_x4_up"
LINE_B = "line_b_downsampled_x4_up"
TOTAL_FRAMES = 3769
PDANS_SCRIPT_NAMES = (
    "run_pdans_resume_chunks.py",
    "run_pdans_after_line_a_supervisor.py",
    "pdans_patch_infer.py",
)


@dataclass
class GpuInfo:
    index: str
    used_mb: int
    total_mb: int
    util_pct: int

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb


@dataclass
class GpuProcess:
    pid: int
    used_mb: int | None
    user: str
    cmd: str


@dataclass
class PsProcess:
    pid: int
    user: str
    etimes: int
    pcpu: float
    cmd: str


class RunnerState:
    def __init__(self) -> None:
        self.stable_since: float | None = None
        self.last_count_change = time.time()
        self.last_counts: tuple[int, int] | None = None
        self.low_cpu_hits: dict[int, int] = {}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return (exc.output or "").strip()
    except FileNotFoundError:
        return ""


def cmd_preview(pid: int, fallback: str = "") -> str:
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = path.read_bytes().replace(b"\x00", b" ").strip()
        text = raw.decode("utf-8", errors="replace")
    except OSError:
        text = fallback
    return " ".join(text.split())[:220]


def pid_user(pid: int) -> str:
    try:
        return pwd.getpwuid((Path("/proc") / str(pid)).stat().st_uid).pw_name
    except OSError:
        return "UNKNOWN"


def query_gpus() -> list[GpuInfo]:
    output = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[GpuInfo] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(GpuInfo(parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
        except ValueError:
            continue
    return gpus


def query_gpu_processes() -> list[GpuProcess]:
    output = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    procs: list[GpuProcess] = []
    for line in output.splitlines():
        if not line or "No running" in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            used_mb: int | None = int(parts[1])
        except ValueError:
            used_mb = None
        procs.append(GpuProcess(pid=pid, used_mb=used_mb, user=pid_user(pid), cmd=cmd_preview(pid)))
    return procs


def read_ps_processes() -> list[PsProcess]:
    output = run_text(["ps", "-eo", "pid=,user=,etimes=,pcpu=,command="])
    rows: list[PsProcess] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        try:
            rows.append(
                PsProcess(
                    pid=int(parts[0]),
                    user=parts[1],
                    etimes=int(float(parts[2])),
                    pcpu=float(parts[3]),
                    cmd=parts[4],
                )
            )
        except ValueError:
            continue
    return rows


def own_pdans_processes(user: str) -> list[PsProcess]:
    current_pid = os.getpid()
    matches: list[PsProcess] = []
    for proc in read_ps_processes():
        if proc.pid == current_pid or proc.user != user:
            continue
        if any(name in proc.cmd for name in PDANS_SCRIPT_NAMES):
            matches.append(proc)
    return matches


def count_final_bin(line: str) -> int:
    path = RESULT_ROOT / line / "pdans" / "final_bin"
    return len(list(path.glob("*.bin"))) if path.exists() else 0


def log_snapshot(line_a_count: int, line_b_count: int, gpus: list[GpuInfo], gpu_procs: list[GpuProcess], pdans_procs: list[PsProcess]) -> None:
    gpu_summary = "; ".join(
        f"gpu={gpu.index} used={gpu.used_mb}MiB free={gpu.free_mb}MiB total={gpu.total_mb}MiB util={gpu.util_pct}%"
        for gpu in gpus
    ) or "gpu=NONE"
    log(
        f"counts line_a={line_a_count}/{TOTAL_FRAMES} line_b={line_b_count}/{TOTAL_FRAMES} "
        f"{gpu_summary} pdans_running={'YES' if pdans_procs else 'NO'} DETECTOR_EVAL_STARTED=NO"
    )
    if not gpu_procs:
        log("compute_processes=NONE")
    for proc in gpu_procs:
        mem = "UNKNOWN" if proc.used_mb is None else f"{proc.used_mb}MiB"
        log(f"compute_process pid={proc.pid} user={proc.user} used_memory={mem} cmd={proc.cmd}")
    for proc in pdans_procs:
        log(f"own_pdans_process pid={proc.pid} etimes={proc.etimes}s pcpu={proc.pcpu:.1f} cmd={cmd_preview(proc.pid, proc.cmd)}")


def external_gpu_processes(user: str, gpu_procs: list[GpuProcess]) -> list[GpuProcess]:
    return [proc for proc in gpu_procs if proc.user not in {user, "UNKNOWN"}]


def gpu_memory_ok(gpus: list[GpuInfo], max_used_mb: int, min_free_mb: int) -> bool:
    return bool(gpus) and all(gpu.used_mb < max_used_mb or gpu.free_mb > min_free_mb for gpu in gpus)


def target_command(line_a_count: int, line_b_count: int, chunk_size: int, timeout: int) -> tuple[str, list[str]] | None:
    if line_a_count < TOTAL_FRAMES:
        return (
            "line_a_resume",
            [
                str(EAR_PY),
                str(PROJECT_ROOT / "scripts" / "run_pdans_resume_chunks.py"),
                "--line",
                LINE_A,
                "--chunk-size",
                str(chunk_size),
                "--timeout",
                str(timeout),
            ],
        )
    if line_b_count < TOTAL_FRAMES:
        return (
            "line_b_supervisor",
            [
                str(EAR_PY),
                str(PROJECT_ROOT / "scripts" / "run_pdans_after_line_a_supervisor.py"),
                "--chunk-size",
                str(chunk_size),
                "--timeout",
                str(timeout),
            ],
        )
    return None


def start_command(kind: str, cmd: list[str], dry_run: bool) -> None:
    log(f"start_requested kind={kind} cmd={' '.join(cmd)} dry_run={dry_run}")
    if dry_run:
        log(f"dry_run would_start={kind}")
        return
    env = os.environ.copy()
    ear_bin = str(EAR_PY.parent)
    env["PATH"] = ear_bin + (":" + env["PATH"] if env.get("PATH") else "")
    log_file = AUDIT_DIR / ("pdans_line_a_resume_nohup.log" if kind == "line_a_resume" else "pdans_supervisor_nohup.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"started kind={kind} pid={proc.pid} log={log_file}")


def update_count_progress(state: RunnerState, line_a_count: int, line_b_count: int) -> None:
    counts = (line_a_count, line_b_count)
    if state.last_counts is None or counts != state.last_counts:
        state.last_counts = counts
        state.last_count_change = time.time()


def blocked_by_external_gpu(
    state: RunnerState,
    user: str,
    gpu_procs: list[GpuProcess],
    pdans_procs: list[PsProcess],
    low_cpu_threshold: float,
    low_cpu_checks: int,
) -> bool:
    external = external_gpu_processes(user, gpu_procs)
    compute_pids = {proc.pid for proc in gpu_procs}
    infer_procs = [proc for proc in pdans_procs if "pdans_patch_infer.py" in proc.cmd]
    if not external or not infer_procs:
        return False
    no_count_progress = time.time() - state.last_count_change >= 1800
    if not no_count_progress:
        return False
    for proc in infer_procs:
        if proc.pcpu <= low_cpu_threshold:
            state.low_cpu_hits[proc.pid] = state.low_cpu_hits.get(proc.pid, 0) + 1
        else:
            state.low_cpu_hits[proc.pid] = 0
        if proc.etimes >= 1800 and proc.pid not in compute_pids and state.low_cpu_hits.get(proc.pid, 0) >= low_cpu_checks:
            log(
                f"PDANS blocked by external GPU process infer_pid={proc.pid} "
                f"elapsed={proc.etimes}s low_cpu_hits={state.low_cpu_hits[proc.pid]}"
            )
            return True
    return False


def kill_own_pdans(pdans_procs: list[PsProcess], dry_run: bool) -> None:
    if not pdans_procs:
        return
    for proc in pdans_procs:
        log(f"kill_requested own_pid={proc.pid} cmd={cmd_preview(proc.pid, proc.cmd)} dry_run={dry_run}")
    if dry_run:
        log("dry_run would_kill_own_stuck_pdans=YES")
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for proc in pdans_procs:
            try:
                os.kill(proc.pid, sig)
                log(f"killed own_pid={proc.pid} signal={sig.name}")
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                log(f"kill_failed own_pid={proc.pid} signal={sig.name} error={exc}")
        if sig == signal.SIGTERM:
            time.sleep(10)


def one_iteration(args: argparse.Namespace, state: RunnerState) -> bool:
    user = getpass.getuser()
    line_a_count = count_final_bin(LINE_A)
    line_b_count = count_final_bin(LINE_B)
    update_count_progress(state, line_a_count, line_b_count)
    gpus = query_gpus()
    gpu_procs = query_gpu_processes()
    pdans_procs = own_pdans_processes(user)
    log_snapshot(line_a_count, line_b_count, gpus, gpu_procs, pdans_procs)

    target = target_command(line_a_count, line_b_count, args.chunk_size, args.timeout)
    if target is None:
        log("PDANS completed")
        return True

    external = external_gpu_processes(user, gpu_procs)
    for proc in external:
        mem = "UNKNOWN" if proc.used_mb is None else f"{proc.used_mb}MiB"
        log(f"external GPU process found PID={proc.pid} user={proc.user} used_memory={mem} command_preview={proc.cmd}")

    if blocked_by_external_gpu(state, user, gpu_procs, pdans_procs, args.low_cpu_threshold, args.low_cpu_checks):
        kill_own_pdans(pdans_procs, args.dry_run)
        state.stable_since = None
        return False

    if pdans_procs:
        log("own_pdans_already_running=YES action=wait")
        return False

    memory_ok = gpu_memory_ok(gpus, args.max_used_mb, args.min_free_mb)
    if external or not memory_ok:
        state.stable_since = None
        log(f"gpu_gap_ready=NO external_processes={len(external)} memory_ok={memory_ok}")
        return False

    if state.stable_since is None:
        state.stable_since = time.time()
        log(f"gpu_gap_candidate=YES stable_elapsed=0s required={args.stable_seconds}s")
        return False

    stable_elapsed = time.time() - state.stable_since
    if stable_elapsed < args.stable_seconds:
        log(f"gpu_gap_candidate=YES stable_elapsed={int(stable_elapsed)}s required={args.stable_seconds}s")
        return False

    kind, cmd = target
    start_command(kind, cmd, args.dry_run)
    state.stable_since = None
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-interval", type=int, default=60)
    parser.add_argument("--stable-seconds", type=int, default=300)
    parser.add_argument("--max-used-mb", type=int, default=1500)
    parser.add_argument("--min-free-mb", type=int, default=6500)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--low-cpu-threshold", type=float, default=1.0)
    parser.add_argument("--low-cpu-checks", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    state = RunnerState()
    log(
        "gpu_gap_runner_start "
        f"dry_run={args.dry_run} once={args.once} check_interval={args.check_interval}s "
        f"stable_seconds={args.stable_seconds}s max_used_mb={args.max_used_mb} min_free_mb={args.min_free_mb} "
        "DETECTOR_EVAL_STARTED=NO"
    )
    while True:
        completed = one_iteration(args, state)
        if completed or args.once:
            return 0
        time.sleep(args.check_interval)


if __name__ == "__main__":
    raise SystemExit(main())
