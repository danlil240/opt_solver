"""Run Linux-only benchmark solvers (SPRAL SSIDS) from Windows through WSL2.

SSIDS does not build natively on Windows. solver/benchmarks/wsl/ssids_replay.c
is a small C driver with bench_replay's command line and JSONL record schema;
it lives inside WSL (build recipe: tools/bench/wsl_ssids.md). tools/bench/run.py
dispatches the solver ids in WSL_SOLVERS here when given `--runner wsl`.

Standalone use (a thin wrapper over run.py that defaults to the ssids solver):

    python tools/bench/run_wsl.py --manifest benchmarks/manifests/smoke.json

Environment:
    SMF_WSL_DISTRO          WSL distribution (default Ubuntu)
    SMF_WSL_SSIDS_DRIVER    Linux path of the driver (default ~/ssids_bench/ssids_replay)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import List, Optional, Tuple

WSL_SOLVERS = {"ssids"}
DISTRO = os.environ.get("SMF_WSL_DISTRO", "Ubuntu")
# bench_replay options whose value is a filesystem path.
_PATH_OPTIONS = {"--manifest", "--mtx", "--seq-values", "--JsonlOut"}
# SSIDS requires OMP_CANCELLATION and OMP_PROC_BIND (it aborts or warns without).
_SSIDS_ENV = ("OMP_CANCELLATION=TRUE", "OMP_PROC_BIND=TRUE")

_DRIVE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$")


def to_wsl_path(path) -> str:
    """Windows path (absolute or relative to the cwd) to its /mnt/<drive>/ form."""
    s = str(path)
    if s.startswith("/") and not _DRIVE.match(s):
        return s
    m = _DRIVE.match(s)
    if not m:
        s = str(pathlib.Path(s).resolve())
        m = _DRIVE.match(s)
        if not m:
            return s.replace("\\", "/")
    return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}"


def wsl_home(distro: str = DISTRO) -> Optional[str]:
    try:
        out = subprocess.run(["wsl.exe", "-d", distro, "--exec", "printenv", "HOME"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    home = out.stdout.strip().replace("\x00", "")
    return home if out.returncode == 0 and home else None


def driver_path(distro: str = DISTRO) -> str:
    explicit = os.environ.get("SMF_WSL_SSIDS_DRIVER")
    if explicit:
        return explicit
    return f"{wsl_home(distro) or '/root'}/ssids_bench/ssids_replay"


def wsl_command(args: List[str], threads: int, driver: Optional[str] = None,
                distro: str = DISTRO) -> List[str]:
    """wsl.exe command running the driver with bench_replay-style `args`
    (the arguments after the binary), Windows paths translated, thread env set."""
    translated: List[str] = []
    prev = None
    for a in args:
        translated.append(to_wsl_path(a) if prev in _PATH_OPTIONS else a)
        prev = a
    env = [f"OMP_NUM_THREADS={threads}", f"OPENBLAS_NUM_THREADS={threads}",
           f"MKL_NUM_THREADS={threads}", *_SSIDS_ENV]
    return ["wsl.exe", "-d", distro, "--exec", "/usr/bin/env", *env,
            driver or driver_path(distro), *translated]


def export_builtin_case(binary: pathlib.Path, case_id: str, cache_dir: pathlib.Path) -> Tuple[pathlib.Path, str]:
    """Write a built-in (generated) case as Matrix Market with `bench_replay
    --export-mtx`; returns (path, "spd"|"indefinite"). Cached per case id."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{case_id}.mtx"
    mtype_file = cache_dir / f"{case_id}.mtype"
    if not (path.exists() and mtype_file.exists()):
        out = subprocess.run([str(binary), "--case", case_id, "--export-mtx", str(path)],
                             capture_output=True, text=True, timeout=120)
        m = re.search(r"mtype=(\w+)", out.stdout)
        if out.returncode != 0 or not m:
            raise RuntimeError(f"cannot export built-in case {case_id}: {out.stderr.strip() or out.stdout.strip()}")
        mtype_file.write_text(m.group(1), encoding="utf-8")
    return path, mtype_file.read_text(encoding="utf-8").strip()


def unavailable_record(case: dict, solver_id: str, mode: str, track: str, trial_id: int,
                       warmup: bool, threads: int, reason: str) -> dict:
    matrix_id = case["matrix_id"]
    return {
        "schema_version": 1,
        "matrix_id": matrix_id,
        "sequence_id": case.get("sequence_id", matrix_id),
        "family_id": case.get("family_id", case.get("sequence_id", matrix_id)),
        "solver_id": solver_id, "mode": mode, "track": track, "trial_id": trial_id,
        "warmup": warmup, "threads": threads, "outcome": "Unavailable", "reason": reason,
        "wall_ms": 0.0, "peak_rss_bytes": -1,
        "validation": {"finite": False, "eta_inf": None, "omega": None, "relative_residual": None},
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the WSL-hosted solvers (ssids) over a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None, help="default: build/wsl_results/<manifest stem>")
    parser.add_argument("--solvers", default="ssids")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", default="1")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--binary", default=None)
    args = parser.parse_args(argv)

    import run  # sibling module (tools/bench is sys.path[0] when run as a script)

    out_dir = args.output or str(pathlib.Path(args.build_dir) / "wsl_results" / pathlib.Path(args.manifest).stem)
    cmd = ["--manifest", args.manifest, "--output", out_dir, "--runner", "wsl",
           "--solvers", args.solvers, "--trials", str(args.trials), "--warmups", str(args.warmups),
           "--threads", args.threads, "--timeout", str(args.timeout), "--build-dir", args.build_dir]
    if args.trials < 10:
        cmd.append("--allow-short-trials")
    if args.cases:
        cmd += ["--cases", args.cases]
    if args.binary:
        cmd += ["--binary", args.binary]
    rc = run.main(cmd)

    records = pathlib.Path(out_dir) / "records.jsonl"
    bad = False
    if records.exists():
        print(f"{'case':36} {'solver':8} {'outcome':12} {'factor_ms':>10} {'eta_inf':>10}")
        for line in records.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("warmup"):
                continue
            v = r.get("validation") or {}
            eta = v.get("eta_inf")
            print(f"{r['matrix_id'][:36]:36} {r['solver_id']:8} {r['outcome']:12} "
                  f"{float(r.get('factor_ms') or 0):10.3f} {('%.1e' % eta) if eta is not None else '-':>10}")
            bad |= r["outcome"] != "Success"
    return 1 if (rc or bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
