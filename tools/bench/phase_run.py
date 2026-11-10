#!/usr/bin/env python3
"""Phase-traced smf runs of a manifest's cases (ipopt track, every stored factorization).

  python tools/bench/phase_run.py <bench_replay.exe> <manifest> <out dir> [--trials 3] [--threads 1,8] [--cases a,b]

Per case and thread count: 1 warmup run, then N trials with --print-steps. Each run's stderr (the library's
SMF_BENCH_PHASE_TRACE marks when that variable is set, and bench_replay's per-step lines) goes to
<out>/phase_<case>_T<threads>_<trial>.txt; tools/bench/phase_split.py tabulates them. Records go to
<out>/records.jsonl. One process at a time; on the 12900K run it under start /affinity FFFF /belownormal
(~/.claude/smf-remote/rpy.sh does).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time


def replay_cmd(binary: str, manifest: str, case: dict, threads: int, trial: int, records: pathlib.Path,
               warm: bool) -> list[str]:
    cmd = [binary, "--manifest", manifest, "--case", case["matrix_id"],
           "--sequence-id", case.get("sequence_id", case["matrix_id"]),
           "--family-id", case.get("family_id", case["matrix_id"]), "--solver", "smf",
           "--mode", case.get("mode", "cold"), "--track", case.get("track", "recommended"),
           "--trial", str(trial), "--threads", str(threads), "--JsonlOut", str(records),
           "--seq-steps", "all", "--print-steps"]
    for key, flag in (("mtx_path", "--mtx"), ("matrix_type", "--mtype"),
                      ("seq_values_path", "--seq-values"), ("ordering", "--ordering")):
        if case.get(key):
            cmd += [flag, case[key]]
    if warm:
        cmd.append("--warmup")
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary")
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--cases", default="")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))["cases"]
    if a.cases:
        want = set(a.cases.split(","))
        cases = [c for c in cases if c["matrix_id"] in want]
    log = open(out / "run.log", "a", encoding="utf-8")
    for case in cases:
        mid = case["matrix_id"]
        for t in [int(x) for x in a.threads.split(",")]:
            env = dict(os.environ)
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[v] = str(t)
            for trial in ["w"] + list(range(a.trials)):
                warm = trial == "w"
                records = out / ("warmup.jsonl" if warm else "records.jsonl")
                cmd = replay_cmd(a.binary, a.manifest, case, t, 0 if warm else trial, records, warm)
                t0 = time.perf_counter()
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
                (out / f"phase_{mid}_T{t}_{trial}.txt").write_text(p.stderr + p.stdout, encoding="utf-8")
                log.write(f"{mid} T{t} {trial} rc={p.returncode} {time.perf_counter() - t0:.1f}s\n")
                log.flush()
    log.write("ALL_DONE\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
