"""Render a per-case x per-solver median-time comparison table from a
tools/bench records.jsonl, as Markdown. Also emits per-solver solve rates
and per-case speedup-vs-best summary.

Usage:
    python tools/bench/compare_table.py --input benchmarks/results/optim20 \
        [--manifest benchmarks/manifests/optim20.json] [--output out.md]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from typing import Any

SOLVER_ORDER = ["smf", "ma27", "ma57", "ma86", "ma87", "ma97", "ssids",
                "mumps", "pardiso", "mkl_pardiso", "panua_pardiso", "cholmod"]

FAILURE_CODES = {
    "Failed": "FAIL",
    "Timeout": "TIME",
    "OutOfMemory": "OOM",
    "Inaccurate": "INAC",
    "Singular": "SING",
    "Unavailable": "  -  ",
    "InvalidInput": "INVL",
}


def qualified_outcome(rec: dict[str, Any]) -> str:
    raw = str(rec.get("outcome", "Failed"))
    if raw != "Success":
        return raw
    v = rec.get("validation") or {}
    eta, om = v.get("eta_inf"), v.get("omega")
    if not v.get("finite") or eta is None or om is None:
        return "Inaccurate"
    if not (math.isfinite(float(eta)) and math.isfinite(float(om))):
        return "Inaccurate"
    if float(eta) > 1e-10 or float(om) > 1e-8:
        return "Inaccurate"
    for key in ("per_rhs_validation", "per_update_validation"):
        sub = rec.get(key)
        if isinstance(sub, list):
            for sv in sub:
                if not isinstance(sv, dict) or not sv.get("finite"):
                    return "Inaccurate"
                e2, o2 = sv.get("eta_inf"), sv.get("omega")
                if e2 is None or o2 is None:
                    return "Inaccurate"
                if float(e2) > 1e-10 or float(o2) > 1e-8:
                    return "Inaccurate"
    return "Success"


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    out = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    if ms >= 10:
        return f"{ms:.1f}"
    return f"{ms:.2f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    records = load_records(pathlib.Path(args.input))
    measured = [r for r in records if not r.get("warmup", False)]
    if not measured:
        print("error: no measured records", file=sys.stderr)
        return 2

    case_meta: dict[str, dict[str, Any]] = {}
    solvers: list[str] = []
    by_cs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in measured:
        mid, sid = str(r["matrix_id"]), str(r["solver_id"])
        case_meta.setdefault(mid, {"n": r.get("n"), "nnz": r.get("nnz"),
                                   "family_id": r.get("family_id")})
        if sid not in solvers:
            solvers.append(sid)
        by_cs.setdefault((mid, sid), []).append(r)

    solvers.sort(key=lambda s: (SOLVER_ORDER.index(s) if s in SOLVER_ORDER else 99, s))
    case_order = sorted(case_meta, key=lambda m: (case_meta[m]["n"] or 0, m))

    manifest_types: dict[str, str] = {}
    if args.manifest:
        man = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
        for c in man.get("cases", []):
            manifest_types[c["matrix_id"]] = c.get("matrix_type", "?")

    # cell: (mid, sid) -> (outcome, median wall ms)
    cell: dict[tuple[str, str], tuple[str, float | None]] = {}
    for (mid, sid), recs in by_cs.items():
        outs = [qualified_outcome(r) for r in recs]
        if all(o == "Success" for o in outs):
            times = [float(r["wall_ms"]) for r in recs if float(r.get("wall_ms", 0)) > 0]
            cell[(mid, sid)] = ("Success", statistics.median(times) if times else None)
        else:
            worst = next((o for o in outs if o != "Success"), outs[0])
            cell[(mid, sid)] = (worst, None)

    # per-case best + summary stats
    solves = {s: 0 for s in solvers}
    smf_wins = base_wins = ties = compared = 0
    fam_speedups: dict[str, list[float]] = {}
    rows: list[str] = []
    hdr = "| # | case | n | nnz | type | " + " | ".join(solvers) + " | best | smf/best_base |"
    sep = "|--:|---|---:|---:|---|" + "---:|" * (len(solvers) + 2)
    rows.append(hdr)
    rows.append(sep)

    for i, mid in enumerate(case_order, 1):
        meta = case_meta[mid]
        mtype = manifest_types.get(mid, "?")
        cells_txt: list[str] = []
        best_ms: float | None = None
        best_sid: str | None = None
        best_base_ms: float | None = None
        best_base_sid: str | None = None
        smf_ms: float | None = None
        for sid in solvers:
            outcome, ms = cell.get((mid, sid), ("Missing", None))
            if outcome == "Success" and ms is not None:
                cells_txt.append(fmt_ms(ms))
                solves[sid] += 1
                if best_ms is None or ms < best_ms:
                    best_ms, best_sid = ms, sid
                if sid == "smf":
                    smf_ms = ms
                elif best_base_ms is None or ms < best_base_ms:
                    best_base_ms, best_base_sid = ms, sid
            else:
                cells_txt.append(FAILURE_CODES.get(outcome, outcome[:5]))
        speed_txt = "–"
        if smf_ms is not None and best_base_ms is not None:
            sp = best_base_ms / smf_ms
            speed_txt = f"{sp:.2f}x"
            compared += 1
            fam = str(meta["family_id"] or mid)
            fam_speedups.setdefault(fam, []).append(sp)
            if sp >= 1.05:
                smf_wins += 1
            elif sp <= 1 / 1.05:
                base_wins += 1
            else:
                ties += 1
        elif smf_ms is None:
            speed_txt = "smf " + FAILURE_CODES.get(
                cell.get((mid, "smf"), ("Missing", None))[0], "?")
        rows.append(
            f"| {i} | {mid} | {meta['n']} | {meta['nnz']} | {mtype} | "
            + " | ".join(cells_txt)
            + f" | {best_sid or '–'} | {speed_txt} |")

    n_cases = len(case_order)
    lines: list[str] = []
    lines.append("## Per-case median wall time (analyse+factor+solve, ms unless noted)\n")
    lines.extend(rows)
    lines.append("")
    lines.append("## Solve rate\n")
    lines.append("| solver | solved / " + str(n_cases) + " |")
    lines.append("|---|---:|")
    for sid in solvers:
        lines.append(f"| {sid} | {solves[sid]}/{n_cases} |")
    lines.append("")
    lines.append("## smf vs fastest baseline per case\n")
    lines.append(f"- compared cases (smf + >=1 baseline solved): {compared}/{n_cases}")
    lines.append(f"- smf wins (>=1.05x): {smf_wins}; ties: {ties}; baseline wins: {base_wins}")
    if fam_speedups:
        fam_gms = []
        for fam, sps in sorted(fam_speedups.items()):
            gm = math.exp(sum(math.log(s) for s in sps) / len(sps))
            fam_gms.append(gm)
            lines.append(f"- geomean smf/best in `{fam}`: {gm:.3f}x ({len(sps)} cases)")
        overall = math.exp(sum(math.log(g) for g in fam_gms) / len(fam_gms))
        lines.append(f"- **overall geomean (equal family weights): {overall:.3f}x**")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.output:
        pathlib.Path(args.output).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
