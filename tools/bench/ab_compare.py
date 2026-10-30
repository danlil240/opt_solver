#!/usr/bin/env python3
"""Compare two smf record sets (A = baseline, B = candidate) case by case.

Per (case, track, threads): median over trials of wall / analyse / factor /
solve time, B/A ratios, delays, factor entries and flops, and the accuracy
gate of score.py on both sides. Summary: shifted geometric mean of B/A (shift
10 ms, as score.py) per class and split, over the case-tracks both sides
solved accurately; cases accurate on one side only are listed.

Usage:
  python tools/bench/ab_compare.py --a results/ab/smf --b results/ab/cost \
      --manifest benchmarks/manifests/fronttier.json [--split tuning] \
      [--metric wall|factor] [--output out.md]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import score  # noqa: E402

CLASS_ORDER = score.CLASS_ORDER


def collect(path: pathlib.Path) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    by: dict[tuple[str, str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in score.load_records(path):
        if r.get("solver_id") != "smf" or r.get("warmup"):
            continue
        by[(r["matrix_id"], r.get("track", "recommended"), int(r.get("threads") or 1))].append(r)
    return by


def _pos(x: Any) -> bool:
    return isinstance(x, (int, float)) and x > 0


def summarize(recs: list[dict[str, Any]]) -> dict[str, Any]:
    med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
    ok = all(score.is_accurate(r) for r in recs)
    return {
        "ok": ok,
        "trials": len(recs),
        "wall": med([score.wall_ms(r) for r in recs]),
        "analyse": med([float(r.get("analyse_ms") or 0.0) for r in recs]),
        "factor": med([float(r.get("factor_ms") or 0.0) for r in recs]),
        "solve": med([float(r.get("solve_ms") or 0.0) + float(r.get("refine_ms") or 0.0)
                      for r in recs]),
        "rss": med([float(r["peak_rss_bytes"]) for r in recs
                    if _pos(r.get("peak_rss_bytes"))]),
        "delays": recs[-1].get("delayed_pivots"),
        "entries": recs[-1].get("factor_entries"),
        "flops": recs[-1].get("flops"),
        "inertia": score.inertia_tuple(recs[-1]),
        "omega": max((float((r.get("validation") or {}).get("omega") or 0.0) for r in recs),
                     default=None),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="baseline records (file or directory)")
    ap.add_argument("--b", required=True, help="candidate records (file or directory)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="all", choices=["tuning", "heldout", "all"])
    ap.add_argument("--metric", default="wall", choices=["wall", "factor", "analyse", "solve"])
    ap.add_argument("--shift-ms", type=float, default=score.DEFAULT_SHIFT_MS)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    meta = {(c["matrix_id"], c.get("track", "recommended")): c for c in manifest["cases"]}
    A, B = collect(pathlib.Path(args.a)), collect(pathlib.Path(args.b))

    rows = []
    for key in sorted(set(A) & set(B), key=lambda k: (k[2], k[0], k[1])):
        c = meta.get(key[:2])
        if c is None or (args.split != "all" and c.get("split") != args.split):
            continue
        rows.append((key, c, summarize(A[key]), summarize(B[key])))

    lines: list[str] = []
    m = args.metric
    lines.append(f"A = {args.a}, B = {args.b}; metric {m}, median over trials, "
                 f"shifted geomean shift {args.shift_ms:g} ms; split {args.split}\n")
    lines.append(f"| case | track | thr | class | split | A {m} ms | B {m} ms | B/A | "
                 "A fac | B fac | B/A fac | delays A>B | entries B/A | flops B/A | rss B/A | ok A/B |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    groups: dict[tuple[int, str, str], list[tuple[float, float]]] = collections.defaultdict(list)
    one_sided = []
    for key, c, a, b in rows:
        ratio = (b[m] / a[m]) if a[m] and b[m] else math.nan
        fr = (b["factor"] / a["factor"]) if a["factor"] and b["factor"] else math.nan
        er = (b["entries"] / a["entries"]) if a["entries"] and b["entries"] else math.nan
        flr = (b["flops"] / a["flops"]) if a["flops"] and b["flops"] else math.nan
        rr = (b["rss"] / a["rss"]) if a["rss"] and b["rss"] else math.nan
        lines.append(
            f"| {key[0]} | {key[1][:5]} | {key[2]} | {c.get('class')} | {c.get('split')} | "
            f"{a[m]:.2f} | {b[m]:.2f} | {ratio:.3f} | {a['factor']:.2f} | {b['factor']:.2f} | "
            f"{fr:.3f} | {a['delays']}>{b['delays']} | {er:.3f} | {flr:.3f} | {rr:.3f} | "
            f"{'y' if a['ok'] else 'N'}/{'y' if b['ok'] else 'N'} |")
        if a["ok"] and b["ok"]:
            for g in ((key[2], c.get("class"), c.get("split")), (key[2], "all", c.get("split"))):
                groups[g].append((b[m], a[m]))
        elif a["ok"] != b["ok"]:
            one_sided.append((key, a["ok"], b["ok"], a["omega"], b["omega"]))
        if a["ok"] and b["ok"] and a["inertia"] and b["inertia"] and a["inertia"] != b["inertia"]:
            one_sided.append((key, "inertia", a["inertia"], b["inertia"], None))

    lines.append("\n**Shifted geomean B/A** (< 1: B faster)\n")
    lines.append("| threads | class | split | cases | B/A |")
    lines.append("|---|---|---|---|---|")
    for g in sorted(groups, key=lambda g: (g[0], (CLASS_ORDER + ["all"]).index(g[1])
                                           if g[1] in CLASS_ORDER + ["all"] else 99, g[2])):
        sg = score.shifted_geomean(groups[g], args.shift_ms)
        lines.append(f"| {g[0]} | {g[1]} | {g[2]} | {len(groups[g])} | {sg:.3f} |")
    if one_sided:
        lines.append("\n**Accuracy / inertia differences**\n")
        for item in one_sided:
            lines.append(f"- {item}")
    text = "\n".join(lines) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
