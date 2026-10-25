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

    solvers = manifest.get("solvers", ["smf"])
    required_baselines = manifest.get("required_baselines", []) if args.require_baselines else []
    binary = find_replay_binary(build_dir, args.binary)

    rng = random.Random(args.seed)
    any_failure = False

    for case in cases:
        matrix_id = case["matrix_id"]
        sequence_id = case.get("sequence_id", matrix_id)
        family_id = case.get("family_id", sequence_id)
        mode = case.get("mode", "cold")
        track = case.get("track", "recommended")
        total_attempts = args.warmups + args.trials

        for attempt_idx in range(total_attempts):
            is_warmup = attempt_idx < args.warmups
            trial_id = attempt_idx if is_warmup else (attempt_idx - args.warmups)
            ordered_solvers = list(solvers)
            rng.shuffle(ordered_solvers)

            for solver_id in ordered_solvers:
                if binary is not None and binary.name.startswith("bench_replay"):
                    cmd = [
                        str(binary),
                        "--manifest",
                        str(manifest_path),
                        "--case",
                        matrix_id,
                        "--solver",
                        solver_id,
                        "--mode",
                        mode,
                        "--track",
                        track,
                        "--trial",
                        str(trial_id),
                        "--JsonlOut",
                        str(records_path),
                    ]
                    if is_warmup:
                        cmd.append("--warmup")
                    t0 = time.perf_counter()
                    try:
                        proc = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=args.timeout,
                        )
                        if proc.returncode != 0 and not is_warmup:
                            if solver_id == "smf" or solver_id in required_baselines:
                                any_failure = True
                    except subprocess.TimeoutExpired:
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
