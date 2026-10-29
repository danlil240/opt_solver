#!/usr/bin/env python3
"""Interleaved smf-vs-smf A/B on one binary: variants differ only by environment.

For each case of the manifest (optionally one split / class / id list), every
round runs each variant once through tools/bench/run.py (same protocol as a
normal benchmark run: fresh process, in-process warmup, 1 measured trial),
alternating the variant order from case to case and round to round so that
machine drift hits both sides alike. Records go to <output>/<variant>/
records.jsonl with trial_id = round, ready for ab_compare.py.

Variants are NAME=VAR=value[,VAR=value...], e.g.
  python tools/bench/ab_run.py --manifest benchmarks/manifests/fronttier.json \
      --split tuning --rounds 3 --output benchmarks/results/amalg_ab \
      --binary build/benchmarks/Release/bench_replay.exe \
      --variant smf=SMF_BENCH_AMALGAMATION=smf \
      --variant cost=SMF_BENCH_AMALGAMATION=cost

Runs below normal priority on Windows (the machine stays usable) and with 1
thread unless --threads says otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent


def parse_variant(spec: str) -> tuple[str, dict[str, str]]:
    name, _, rest = spec.partition("=")
    env: dict[str, str] = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        env[k] = v
    if not name:
        raise SystemExit(f"bad --variant {spec!r}")
    return name, env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variant", action="append", required=True,
                    help="NAME=VAR=value[,VAR=value...] (two or more)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--split", default="all", choices=["tuning", "heldout", "all"])
    ap.add_argument("--classes", default="", help="comma-separated class filter")
    ap.add_argument("--cases", default="", help="comma-separated matrix_id filter")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args(argv)

    variants = [parse_variant(v) for v in args.variant]
    if len(variants) < 2:
        raise SystemExit("need at least two --variant")
    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    classes = {c for c in args.classes.split(",") if c}
    ids = [c for c in args.cases.split(",") if c]
    seen: list[str] = []
    for c in manifest["cases"]:
        if args.split != "all" and c.get("split") != args.split:
            continue
        if classes and c.get("class") not in classes:
            continue
        if ids and c["matrix_id"] not in ids:
            continue
        if c["matrix_id"] not in seen:
            seen.append(c["matrix_id"])

    out = pathlib.Path(args.output)
    for name, _ in variants:
        (out / name).mkdir(parents=True, exist_ok=True)
    priority = 0x00004000 if os.name == "nt" else 0  # BELOW_NORMAL_PRIORITY_CLASS

    for ci, mid in enumerate(seen):
        for r in range(args.rounds):
            order = variants if (ci + r) % 2 == 0 else list(reversed(variants))
            for name, venv in order:
                with tempfile.TemporaryDirectory() as tmp:
                    cmd = [sys.executable, str(HERE / "run.py"), "--manifest", args.manifest,
                           "--output", tmp, "--trials", "1", "--warmups", "0",
                           "--allow-short-trials", "--cases", mid, "--solvers", "smf",
                           "--threads", args.threads, "--build-dir", args.build_dir,
                           "--timeout", str(args.timeout)]
                    if args.binary:
                        cmd += ["--binary", args.binary]
                    env = dict(os.environ)
                    env.update(venv)
                    subprocess.run(cmd, env=env, creationflags=priority,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    rec_file = pathlib.Path(tmp) / "records.jsonl"
                    lines = rec_file.read_text(encoding="utf-8").splitlines() if rec_file.exists() else []
                with (out / name / "records.jsonl").open("a", encoding="utf-8") as f:
                    for ln in lines:
                        if not ln.strip():
                            continue
                        rec = json.loads(ln)
                        rec["trial_id"] = r
                        rec["ab_variant"] = name
                        f.write(json.dumps(rec) + "\n")
        print(f"[{ci + 1}/{len(seen)}] {mid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
