#!/usr/bin/env python3
"""Markdown report for a thread-sweep benchmark run (tools/bench/run.py --threads).

For each case: median wall time per solver at each thread count, the
fastest baseline, smf's ratio to it, and smf's own 1 -> max-thread speedup.
Then per-thread geomeans and an outcome summary (timeouts, inaccurate,
unavailable) so failures are as visible as times.

Usage:
  python tools/bench/scaling_table.py --input <records.jsonl> [--manifest m.json] [--output out.md]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics

QUAL_ETA = 1e-10


def outcome_of(r: dict) -> str:
    out = r.get("outcome", "Failed")
    if out == "Success":
        eta = (r.get("validation") or {}).get("eta_inf")
        if eta is None or not math.isfinite(eta) or eta > QUAL_ETA:
            return "Inaccurate"
    return out


def fmt(ms: float | None) -> str:
    if ms is None:
        return "-"
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.1f}" if ms >= 10 else f"{ms:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if not r.get("warmup")]
    meta = {}
    if args.manifest:
        for c in json.load(open(args.manifest, encoding="utf-8"))["cases"]:
            meta[c["matrix_id"]] = c
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["matrix_id"], r["solver_id"], int(r.get("threads", 1)))].append(r)

    cases = sorted({k[0] for k in by}, key=lambda c: (meta.get(c, {}).get("n", 0), c))
    solvers = sorted({k[1] for k in by}, key=lambda s: (s != "smf", s))
    threads = sorted({k[2] for k in by})

    def cell(case, solver, t):
        rs = by.get((case, solver, t), [])
        if not rs:
            return None, None
        outs = [outcome_of(r) for r in rs]
        ok = [r["wall_ms"] for r, o in zip(rs, outs) if o == "Success"]
        if len(ok) == len(rs):
            return statistics.median(ok), "Success"
        bad = collections.Counter(o for o in outs if o != "Success").most_common(1)[0][0]
        return None, bad

    lines = ["# Large-scale benchmark", ""]
    present = [s for s in solvers if any(cell(c, s, t)[1] not in (None, "Unavailable") for c in cases for t in threads)]
    absent = [s for s in solvers if s not in present]
    if absent:
        reasons = {}
        for r in recs:
            if r["solver_id"] in absent and r.get("reason"):
                reasons.setdefault(r["solver_id"], r["reason"])
        lines.append("Not measured: " + "; ".join(f"`{s}` ({reasons.get(s, 'unavailable')})" for s in absent))
        lines.append("")

    geo = {t: [] for t in threads}
    wins = {t: 0 for t in threads}
    cmp_cnt = {t: 0 for t in threads}
    for t in threads:
        lines += [f"## {t} thread{'s' if t > 1 else ''} — median wall ms (analyse + factor + solve)", ""]
        hdr = "| case | n | type | " + " | ".join(present) + " | best baseline | smf / best |"
        lines += [hdr, "|---|---:|---|" + "---:|" * len(present) + "---|---:|"]
        for c in cases:
            m = meta.get(c, {})
            vals = {}
            row = [c, str(m.get("n", "")), m.get("matrix_type", "")]
            for s in present:
                ms, oc = cell(c, s, t)
                if ms is None and oc is None and t > 1:
                    ms, oc = cell(c, s, 1)  # serial-only solver: reuse 1-thread time
                    if ms is not None:
                        vals[s] = ms
                        row.append(fmt(ms) + "¹")
                        continue
                vals[s] = ms
                row.append(fmt(ms) if ms is not None else ("-" if oc is None else oc.upper()[:7]))
            base = {s: v for s, v in vals.items() if s != "smf" and v is not None}
            best = min(base, key=base.get) if base else None
            smf = vals.get("smf")
            if best and smf:
                ratio = base[best] / smf
                geo[t].append(math.log(ratio))
                cmp_cnt[t] += 1
                wins[t] += ratio >= 1.0
                row += [best, f"{ratio:.2f}x"]
            else:
                row += [best or "-", "-"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("¹ serial solver, 1-thread time.  smf / best > 1 means smf is faster.")
        lines.append("")

    lines += ["## Summary", "", "| threads | cases compared | smf faster | geomean best-baseline / smf |", "|---:|---:|---:|---:|"]
    for t in threads:
        g = math.exp(sum(geo[t]) / len(geo[t])) if geo[t] else float("nan")
        lines.append(f"| {t} | {cmp_cnt[t]} | {wins[t]} | {g:.2f}x |")
    lines.append("")

    if len(threads) > 1:
        tmax = threads[-1]
        lines += [f"## Thread scaling (1 -> {tmax} threads, speedup of median wall time)", "",
                  "| case | " + " | ".join(present) + " |", "|---|" + "---:|" * len(present)]
        for c in cases:
            row = [c]
            for s in present:
                a, _ = cell(c, s, 1)
                b, _ = cell(c, s, tmax)
                row.append(f"{a / b:.1f}x" if a and b else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines += ["## Outcomes (measured trials)", "", "| solver | threads | " + " | ".join(
        ["Success", "Inaccurate", "Timeout", "Failed", "Singular", "Unavailable"]) + " |",
        "|---|---:|" + "---:|" * 6]
    cnt = collections.Counter((r["solver_id"], int(r.get("threads", 1)), outcome_of(r)) for r in recs)
    for s in solvers:
        for t in threads:
            row = [cnt[(s, t, o)] for o in ("Success", "Inaccurate", "Timeout", "Failed", "Singular", "Unavailable")]
            if sum(row):
                lines.append(f"| {s} | {t} | " + " | ".join(str(x) for x in row) + " |")
    text = "\n".join(lines) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
