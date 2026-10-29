#!/usr/bin/env python3
"""Accuracy gate on the recorded IPOPT KKT sequences: failing steps per model.

Replays every stored factorization (bench_replay --seq-steps all, ipopt track,
u = 1e-8) of each ipopt_hard.json model once per variant and counts the steps
that fail the accuracy gate of score.py (eta_inf <= 1e-10, omega <= 1e-8 after
the replay's refinement), plus the inertia of the last step. The gate passes
when no variant fails a step its baseline (the first variant) passes.

Usage (from the repo root, where the fixtures live):
  python tools/bench/replay_gate.py --binary build/benchmarks/Release/bench_replay.exe \
      --variant smf=SMF_BENCH_AMALGAMATION=smf --variant cost=SMF_BENCH_AMALGAMATION=cost \
      [--manifest benchmarks/manifests/ipopt_hard.json] [--models a,b] [--output out.md]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import score  # noqa: E402
from ab_run import parse_variant  # noqa: E402


def replay(binary: str, case: dict, env: dict[str, str], timeout: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "r.jsonl"
        cmd = [binary, "--solver", "smf", "--mode", "warm_sequence", "--track", "ipopt",
               "--mtype", case.get("matrix_type", "indefinite"), "--mtx", case["mtx_path"],
               "--seq-values", case["seq_values_path"], "--seq-steps", "all",
               "--case", case["matrix_id"], "--JsonlOut", str(out)]
        priority = 0x00004000 if os.name == "nt" else 0  # BELOW_NORMAL_PRIORITY_CLASS
        try:
            subprocess.run(cmd, env={**os.environ, **env}, timeout=timeout, creationflags=priority,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return {"outcome": "Timeout"}
        lines = out.read_text(encoding="utf-8").splitlines() if out.exists() else []
    if not lines:
        return {"outcome": "NoRecord"}
    rec = json.loads(lines[-1])
    steps = rec.get("per_update_validation") or []
    fails = [i for i, v in enumerate(steps) if not score._validation_ok(v)]
    return {
        "outcome": rec.get("outcome"),
        "steps": len(steps),
        "fails": fails,
        "inertia": score.inertia_tuple(rec),
        "factor_ms": float(rec.get("factor_ms") or 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--variant", action="append", required=True,
                    help="NAME=VAR=value[,VAR=value...]; the first is the baseline")
    ap.add_argument("--manifest", default="benchmarks/manifests/ipopt_hard.json")
    ap.add_argument("--models", default="")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    variants = [parse_variant(v) for v in args.variant]
    models = {m for m in args.models.split(",") if m}
    cases = [c for c in json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))["cases"]
             if c.get("seq_values_path") and (not models or c["matrix_id"] in models)]

    names = [n for n, _ in variants]
    lines = ["| model | " + " | ".join(f"{n} fail/steps | {n} inertia | {n} factor ms" for n in names) + " |",
             "|---|" + "---|---|---|" * len(names)]
    ok = True
    for c in cases:
        res = {n: replay(args.binary, c, env, args.timeout) for n, env in variants}
        base = res[names[0]]
        cells = []
        for n in names:
            r = res[n]
            if "steps" not in r:
                cells.append(f"{r['outcome']} | - | -")
                if n != names[0] and "steps" in base:
                    ok = False
                continue
            cells.append(f"{len(r['fails'])}/{r['steps']} | {r['inertia']} | {r['factor_ms']:.0f}")
            if n != names[0] and "steps" in base and set(r["fails"]) - set(base["fails"]):
                ok = False
        lines.append(f"| {c['matrix_id']} | " + " | ".join(cells) + " |")
        print(lines[-1], flush=True)
    lines.append(f"\ngate: {'PASS' if ok else 'FAIL'} (no step failing that the baseline passes)")
    text = "\n".join(lines) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(text, encoding="utf-8")
    print(lines[-1])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
