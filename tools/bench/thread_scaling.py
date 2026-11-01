#!/usr/bin/env python3
"""Thread-scaling report for the smf records of a benchmark run.

Per (case, track) it lists the median factorization time (refactor_ms, else
first_factor_ms) and solve + refine time at every thread count found, each
with its ratio to the 1-thread value, the best thread count (smallest
factor + solve) and the flags
  FAC>1T  factor ratio > 1.05 at some T > 1 (1-thread factor >= --min-ms),
  SOL>1T  solve ratio > 1.05 at some T > 1 (1-thread solve >= --min-ms),
  16>8    16-thread factor > 1.05 x 8-thread factor (8-thread factor >= --min-ms).
The report ends with a count of each flag. Aggregation is score.aggregate;
only smf records count, and records are matched to manifest cases by
(matrix_id, track) as in score.py.

Usage:
  python tools/bench/thread_scaling.py --records DIR [DIR...] \
      --manifest benchmarks/manifests/fronttier.json [--min-ms 1.0] --output out.md
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import score  # noqa: E402

RATIO_TOL = 1.05
FLAGS = ("FAC>1T", "SOL>1T", "16>8")


def _factor_ms(agg: dict[str, Any]) -> float | None:
    ph = agg["phases"]
    return ph["refactor_ms"] if ph["refactor_ms"] is not None else ph["first_factor_ms"]


def _series(by_t: dict[int, dict[str, Any]], key) -> dict[int, float]:
    """Thread count -> value for the aggregates that are usable and have one."""
    out = {}
    for t, agg in sorted(by_t.items()):
        v = key(agg)
        if agg["status"] == "ok" and v is not None:
            out[t] = v
    return out


def _flag_rising(series: dict[int, float], min_ms: float) -> bool:
    base = series.get(1)
    if base is None or base < min_ms:
        return False
    return any(v > RATIO_TOL * base for t, v in series.items() if t > 1)


def _fmt_series(series: dict[int, float], digits: int = 1) -> str:
    return " ".join(f"{t}:{v:.{digits}f}" for t, v in series.items()) or "-"


def _fmt_ratios(series: dict[int, float]) -> str:
    base = series.get(1)
    if base is None or base <= 0:
        return "-"
    return " ".join(f"{t}:{v / base:.2f}" for t, v in series.items())


def analyse(records: list[dict[str, Any]], manifest: dict[str, Any],
            min_ms: float = 1.0, subject: str = "smf") -> list[dict[str, Any]]:
    cases = {(c["matrix_id"], c.get("track", "recommended")): c for c in manifest.get("cases", [])}
    grouped: dict[tuple, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        if r.get("warmup") or r.get("solver_id") != subject:
            continue
        key = (r["matrix_id"], r.get("track", "recommended"))
        if key not in cases:
            continue
        grouped[(key, int(r.get("threads", 1) or 1))].append(r)

    per_case: dict[tuple, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    for (key, threads), recs in grouped.items():
        per_case[key][threads] = score.aggregate(recs)

    rows = []
    for key in sorted(per_case):
        by_t = per_case[key]
        fac = _series(by_t, _factor_ms)
        sol = _series(by_t, lambda a: a["phases"]["solve_refine_ms"])
        both = [t for t in fac if t in sol]
        best = min(both, key=lambda t: fac[t] + sol[t]) if both else None
        flags = []
        if _flag_rising(fac, min_ms):
            flags.append("FAC>1T")
        if _flag_rising(sol, min_ms):
            flags.append("SOL>1T")
        f8, f16 = fac.get(8), fac.get(16)
        if f8 is not None and f16 is not None and f8 >= min_ms and f16 > RATIO_TOL * f8:
            flags.append("16>8")
        rows.append({"case": key[0], "track": key[1], "class": cases[key].get("class", ""),
                     "factor": fac, "solve": sol, "best": best, "flags": flags})
    return rows


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# smf thread scaling",
        "",
        "Times in ms (`T:value`), ratios to the 1-thread value; factor = refactor "
        "(else first factor), solve = solve + refine; best = fewest factor + solve ms.",
        "",
        "| case | track | class | factor ms | factor / 1T | solve ms | solve / 1T | best T | flags |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['track']} | {r['class']} | {_fmt_series(r['factor'])} | "
            f"{_fmt_ratios(r['factor'])} | {_fmt_series(r['solve'])} | {_fmt_ratios(r['solve'])} | "
            f"{r['best'] if r['best'] is not None else '-'} | {' '.join(r['flags'])} |")
    counts = {f: sum(f in r["flags"] for r in rows) for f in FLAGS}
    lines += ["", ", ".join(f"{f}: {counts[f]}" for f in FLAGS), ""]
    return "\n".join(lines)


def report(records: list[dict[str, Any]], manifest: dict[str, Any],
           min_ms: float = 1.0, subject: str = "smf") -> str:
    return render(analyse(records, manifest, min_ms, subject))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", nargs="+", required=True,
                    help="records.jsonl files or directories (searched recursively)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--min-ms", type=float, default=1.0,
                    help="ignore flags on cases whose baseline value is below this (default %(default)g)")
    ap.add_argument("--subject", default="smf")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    records: list[dict[str, Any]] = []
    for p in args.records:
        records.extend(score.load_records(pathlib.Path(p)))
    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    md = report(records, manifest, args.min_ms, args.subject)
    print(md)
    if args.output:
        pathlib.Path(args.output).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
