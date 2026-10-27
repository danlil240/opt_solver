"""Benchmark orchestration and environment inventory for SMF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import run_wsl


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_environment(build_dir: pathlib.Path) -> Dict[str, Any]:
    env_info: Dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count() or 1,
        "blas_threads_env": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
            if k in os.environ
        },
        "cmake_cache": {},
    }
    cache_file = build_dir / "CMakeCache.txt"
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8", errors="replace").splitlines():
            for key in (
                "CMAKE_CXX_COMPILER:",
                "CMAKE_BUILD_TYPE:",
                "SMF_USE_MKL:",
                "SMF_HAS_OPENBLAS:",
                "SMF_BLAS_REFERENCE:",
                "SMF_HAS_AMD:",
                "SMF_HAS_METIS:",
                "SMF_HAS_CHOLMOD:",
                "SMF_HAS_MA27:",
                "SMF_HAS_MUMPS:",
            ):
                if line.startswith(key):
                    k, _, v = line.partition("=")
                    env_info["cmake_cache"][k.split(":")[0]] = v
    return env_info


def find_replay_binary(build_dir: pathlib.Path, explicit_binary: Optional[str]) -> Optional[pathlib.Path]:
    if explicit_binary:
        p = pathlib.Path(explicit_binary)
        return p if p.exists() else None
    candidates = [
        build_dir / "benchmarks" / "bench_replay",
        build_dir / "benchmarks" / "Release" / "bench_replay.exe",
        build_dir / "benchmarks" / "bench_replay.exe",
        build_dir / "benchmarks" / "bench_kkt_fixture_compare",
        build_dir / "benchmarks" / "Release" / "bench_kkt_fixture_compare.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def replay_override_args(case: dict) -> List[str]:
    """Per-case bench_replay overrides. The `ipopt` track only implies u = 1e-8;
    a case may additionally pin the ordering (front-tier manifest uses
    "ordering": "matching" on the ipopt track)."""
    args: List[str] = []
    if case.get("ordering"):
        args += ["--ordering", str(case["ordering"])]
    return args


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run SMF benchmark manifest in isolated subprocesses.")
    parser.add_argument("--manifest", required=True, help="Path to benchmark manifest JSON")
    parser.add_argument("--output", required=True, help="Output directory for JSONL and environment inventory")
    parser.add_argument("--trials", type=int, default=10, help="Number of measured trials (minimum 10 for qualification)")
    parser.add_argument("--warmups", type=int, default=2, help="Number of warmup trials")
    parser.add_argument("--require-baselines", action="store_true", help="Fail closed if required baselines are unavailable")
    parser.add_argument("--allow-short-trials", action="store_true", help="Allow <10 trials for smoke/unit testing")
    parser.add_argument("--build-dir", default="build", help="CMake build directory")
    parser.add_argument("--binary", default=None, help="Explicit benchmark runner binary")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for solver order")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-attempt subprocess timeout in seconds")
    parser.add_argument("--threads", default="1",
                        help="Comma-separated thread counts to sweep (e.g. 1,8,16); applied to smf, BLAS and OpenMP")
    parser.add_argument("--serial-solvers", default="ma27",
                        help="Comma-separated solvers that only run at 1 thread (no parallelism to measure)")
    parser.add_argument("--solvers", default=None, help="Comma-separated override of the manifest's solver list")
    parser.add_argument("--cases", default=None, help="Comma-separated matrix_id filter")
    parser.add_argument("--runner", choices=("native", "wsl"), default="native",
                        help="'wsl' runs the Linux-only solvers (ssids) through WSL2 via run_wsl.py; "
                             "the others stay native")
    args = parser.parse_args(argv)

    if args.trials < 10 and not args.allow_short_trials:
        print("error: qualification requires at least 10 measured trials (--trials >= 10)", file=sys.stderr)
        return 2

    manifest_path = pathlib.Path(args.manifest)
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"error: invalid manifest JSON: {exc}", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir = pathlib.Path(args.build_dir)

    env_info = inventory_environment(build_dir)
    (out_dir / "environment.json").write_text(json.dumps(env_info, indent=2) + "\n", encoding="utf-8")

    records_path = out_dir / "records.jsonl"
    if records_path.exists():
        records_path.unlink()

    cases = manifest.get("cases", [])
    if not cases:
        print("error: manifest contains no cases", file=sys.stderr)
        return 2

    solvers = args.solvers.split(",") if args.solvers else manifest.get("solvers", ["smf"])
    thread_list = [int(t) for t in args.threads.split(",") if t]
    serial_solvers = {x for x in args.serial_solvers.split(",") if x}
    case_filter = {x for x in args.cases.split(",") if x} if args.cases else None
    required_baselines = manifest.get("required_baselines", []) if args.require_baselines else []
    binary = find_replay_binary(build_dir, args.binary)

    rng = random.Random(args.seed)
    any_failure = False

    for case in cases:
        matrix_id = case["matrix_id"]
        if case_filter is not None and matrix_id not in case_filter:
            continue
        # (solver, threads) pairs that timed out or are unavailable on this
        # case: later attempts would only repeat that, so they are skipped.
        dead: set = set()
        sequence_id = case.get("sequence_id", matrix_id)
        family_id = case.get("family_id", sequence_id)
        mode = case.get("mode", "cold")
        track = case.get("track", "recommended")
        total_attempts = args.warmups + args.trials

        for threads, attempt_idx in [(t, a) for t in thread_list for a in range(total_attempts)]:
            is_warmup = attempt_idx < args.warmups
            trial_id = attempt_idx if is_warmup else (attempt_idx - args.warmups)
            ordered_solvers = list(solvers)
            rng.shuffle(ordered_solvers)

            for solver_id in ordered_solvers:
                if threads > 1 and solver_id in serial_solvers:
                    continue
                if (solver_id, threads) in dead:
                    continue
                use_wsl = args.runner == "wsl" and solver_id in run_wsl.WSL_SOLVERS
                if use_wsl or (binary is not None and binary.name.startswith("bench_replay")):
                    cmd = [
                        str(binary) if binary is not None else "bench_replay",
                        "--manifest",
                        str(manifest_path),
                        "--case",
                        matrix_id,
                        "--sequence-id",
                        sequence_id,
                        "--family-id",
                        family_id,
                        "--solver",
                        solver_id,
                        "--mode",
                        mode,
                        "--track",
                        track,
                        "--trial",
                        str(trial_id),
                        "--threads",
                        str(threads),
                        "--JsonlOut",
                        str(records_path),
                    ]
                    exported_mtype = None
                    if case.get("mtx_path"):
                        cmd.extend(["--mtx", str(case["mtx_path"])])
                    elif use_wsl and binary is not None:
                        # Built-in generated case: the Linux driver needs a file.
                        try:
                            mtx, exported_mtype = run_wsl.export_builtin_case(
                                binary, matrix_id, build_dir / "wsl_cases")
                            cmd.extend(["--mtx", str(mtx)])
                        except Exception as exc:  # noqa: BLE001
                            print(f"warning: {exc}", file=sys.stderr)
                    if case.get("matrix_type"):
                        cmd.extend(["--mtype", str(case["matrix_type"])])
                    elif exported_mtype:
                        cmd.extend(["--mtype", exported_mtype])
                    if case.get("seq_values_path"):
                        cmd.extend(["--seq-values", str(case["seq_values_path"])])
                    cmd.extend(replay_override_args(case))
                    if is_warmup:
                        cmd.append("--warmup")
                    t0 = time.perf_counter()
                    try:
                        env = dict(os.environ)
                        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                            env[var] = str(threads)
                        lines_before = records_path.read_text(encoding="utf-8").count("\n") if records_path.exists() else 0
                        if use_wsl:
                            cmd = run_wsl.wsl_command(cmd[1:], threads, timeout_s=max(1.0, args.timeout - 3.0))
                        proc = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=args.timeout,
                            env=env,
                        )
                        new_lines = []
                        if records_path.exists():
                            new_lines = records_path.read_text(encoding="utf-8").splitlines()[lines_before:]
                            if any('"outcome":"Unavailable"' in ln for ln in new_lines):
                                dead.add((solver_id, threads))
                        if use_wsl and not new_lines:
                            # No record: environment broken (dead) or a per-case driver failure.
                            outcome, why, env_dead = run_wsl.classify_no_record(
                                proc.returncode, proc.stderr or proc.stdout)
                            rec = run_wsl.unavailable_record(
                                case, solver_id, mode, track, trial_id, is_warmup, threads, why, outcome)
                            with records_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(rec) + "\n")
                            if env_dead:
                                dead.add((solver_id, threads))
                        if proc.returncode != 0 and not is_warmup:
                            if solver_id == "smf" or solver_id in required_baselines:
                                any_failure = True
                    except OSError as exc:
                        if not use_wsl:
                            raise
                        rec = run_wsl.unavailable_record(
                            case, solver_id, mode, track, trial_id, is_warmup, threads,
                            f"wsl.exe could not be started: {exc}")
                        with records_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                        dead.add((solver_id, threads))
                    except subprocess.TimeoutExpired:
                        if use_wsl:
                            run_wsl.kill_stray_drivers()
                        rec = {
                            "schema_version": 1,
                            "matrix_id": matrix_id,
                            "sequence_id": sequence_id,
                            "family_id": family_id,
                            "solver_id": solver_id,
                            "mode": mode,
                            "track": track,
                            "trial_id": trial_id,
                            "warmup": is_warmup,
                            "threads": threads,
                            "outcome": "Timeout",
                            "reason": f"subprocess timed out after {args.timeout}s",
                            "wall_ms": (time.perf_counter() - t0) * 1000.0,
                            "peak_rss_bytes": -1,
                            "validation": {
                                "finite": False,
                                "eta_inf": None,
                                "omega": None,
                                "relative_residual": None,
                            },
                        }
                        with records_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                        if not use_wsl:
                            dead.add((solver_id, threads))
                        if not is_warmup and (solver_id == "smf" or solver_id in required_baselines):
                            any_failure = True
                else:
                    outcome = "Unavailable" if solver_id != "smf" else "Failed"
                    reason = "bench_replay binary not built or solver unavailable"
                    rec = {
                        "schema_version": 1,
                        "matrix_id": matrix_id,
                        "sequence_id": sequence_id,
                        "family_id": family_id,
                        "solver_id": solver_id,
                        "mode": mode,
                        "track": track,
                        "trial_id": trial_id,
                        "warmup": is_warmup,
                        "outcome": outcome,
                        "reason": reason,
                        "wall_ms": 0.0,
                        "peak_rss_bytes": -1,
                        "validation": {
                            "finite": False,
                            "eta_inf": None,
                            "omega": None,
                            "relative_residual": None,
                        },
                    }
                    with records_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    if not is_warmup and (solver_id == "smf" or solver_id in required_baselines):
                        any_failure = True

    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
