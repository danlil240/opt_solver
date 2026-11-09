#!/usr/bin/env python3
"""Projected factorization time of a whole IPOPT run, smf against a rival, and the 8T-engine stage gates.

A case's projection is analyse + factor + max(0, N - batch_count) x refactor (medians over the non-warmup
records of one case, thread count and solver), with N the case's IPOPT factorization count from the manifest
(`factorizations`). factor_ms covers the replayed batch (`batch_count` factorizations), so the rest of the run
is priced at the median refactorization. Same formula as the 2026-11-08 probe (PR #64).

  python tools/bench/project_run.py --smf <records dir|file> --rivals <records dir|file> \\
      --manifest benchmarks/manifests/probe_refactor8t.json --threads 8 [--gate stage1|stage2]

--gate exits 1 when a case of the stage's set has smf / rival >= 1.00, has no ratio, or has an smf record whose
outcome is not Success. Spec: docs/superpowers/specs/2026-11-09-refactor-8t-engine-design.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys
from typing import Iterable

STAGE1_WINS = frozenset({
    "pglib_opf_case1354_pegase", "pglib_opf_case2000_goc__sad", "portfolio_20k",
    "obstacle_surface_200", "pde_control_120", "pde_control_250",
})
STAGE2_WINS = STAGE1_WINS | {"acopf_grid_80", "acopf_grid_160"}
GATES = {"stage1": STAGE1_WINS, "stage2": STAGE2_WINS}


def _median(rs: list[dict], key: str) -> float:
    xs = [r[key] for r in rs if isinstance(r.get(key), (int, float)) and r[key] >= 0]
    return statistics.median(xs) if xs else float("nan")


def _groups(records: Iterable[dict]) -> dict[tuple[str, int, str], list[dict]]:
    g: dict[tuple[str, int, str], list[dict]] = collections.defaultdict(list)
    for r in records:
        if not r.get("warmup"):
            g[(r["matrix_id"], int(r["threads"]), r["solver_id"])].append(r)
    return g


def project(records: list[dict], factorizations: dict[str, int]) -> dict[tuple[str, int, str], float]:
    out = {}
    for key, rs in _groups(records).items():
        n = factorizations.get(key[0])
        if n is None:
            continue
        batch = _median(rs, "batch_count")
        out[key] = (_median(rs, "analyse_ms") + _median(rs, "factor_ms")
                    + max(0.0, n - batch) * _median(rs, "refactor_ms"))
    return out


def ratios(proj: dict[tuple[str, int, str], float], threads: int,
           rival: str = "mkl_pardiso") -> dict[str, float]:
    out = {}
    for (case, t, solver), ms in proj.items():
        if t == threads and solver == "smf" and (case, t, rival) in proj:
            out[case] = ms / proj[(case, t, rival)]
    return out


def gate_failures(r: dict[str, float], wins: set[str] | frozenset[str], records: list[dict],
                  threads: int | None = None) -> list[str]:
    bad = {x["matrix_id"] for x in records
           if x.get("solver_id") == "smf" and not x.get("warmup") and x.get("outcome") != "Success"
           and (threads is None or int(x["threads"]) == threads)}
    return sorted(c for c in wins if c not in r or not r[c] < 1.0 or c in bad)


def load(path: str) -> list[dict]:
    p = pathlib.Path(path)
    files = sorted(p.rglob("records.jsonl")) if p.is_dir() else [p]
    out = []
    for f in files:
        out += [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smf", required=True, help="smf records (directory or records.jsonl)")
    ap.add_argument("--rivals", required=True, help="rival records (directory or records.jsonl)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rival", default="mkl_pardiso")
    ap.add_argument("--gate", choices=sorted(GATES), default=None)
    a = ap.parse_args(argv)
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))
    nf = {c["matrix_id"]: int(c["factorizations"]) for c in man["cases"] if "factorizations" in c}
    smf = [r for r in load(a.smf) if r.get("solver_id") == "smf"]
    riv = [r for r in load(a.rivals) if r.get("solver_id") == a.rival]
    recs = smf + riv
    proj = project(recs, nf)
    rat = ratios(proj, a.threads, a.rival)
    g = _groups(recs)
    print(f"{'case':30s} {'N':>4s} {'refac ms':>9s} {'smf ms':>9s} {a.rival + ' ms':>16s} {'ratio':>6s} "
          f"{'RSS MB':>7s} outcome")
    for case in sorted(nf, key=lambda c: nf[c]):
        rs = g.get((case, a.threads, "smf"), [])
        sm = proj.get((case, a.threads, "smf"), float("nan"))
        rv = proj.get((case, a.threads, a.rival), float("nan"))
        rss = _median(rs, "peak_rss_bytes") / 1e6
        oc = ",".join(sorted({r.get("outcome", "?") for r in rs})) or "-"
        print(f"{case:30s} {nf[case]:4d} {_median(rs, 'refactor_ms'):9.2f} {sm:9.1f} {rv:16.1f} "
              f"{rat.get(case, float('nan')):6.3f} {rss:7.1f} {oc}")
    if a.gate:
        fails = gate_failures(rat, GATES[a.gate], smf, a.threads)
        print(f"gate {a.gate}: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
