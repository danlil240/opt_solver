#!/usr/bin/env python3
"""Score benchmark records against the front-tier manifest.

Per (class, track, thread count) it reports, for a chosen split:
  * shifted geometric mean of subject-time / best-accurate-competitor-time
    (shift 10 ms; the subject itself and any record failing the accuracy gate
    are never the "best competitor"). The competitor set is fixed per class and
    does not depend on the thread count (ruling R43):
      indefinite (small_kkt, large_kkt, hard_indef): ma57, ma97, ssids,
        mkl_pardiso, mumps
      spd: cholmod, mkl_pardiso, ma57, ma97, ssids, mumps
    Other solvers (ma27, cudss, ...) are reference columns only. A set member
    with no record in a thread-count table (e.g. a serial-only solver in the
    8-thread table) is reported as missing for that table,
  * solved counts, accuracy failures and hard failures per solver,
  * implausible records: factor throughput above --max-gflops-per-thread x
    threads (a mis-threaded run); excluded like failures and listed,
  * backward error after refinement (worst eta_inf / omega),
  * median time per phase: analyse, first factorization, warm refactorization,
    solve + refine (a library's internal refinement is part of its solve time),
  * an inertia pass column on the `ipopt` track,
  * a performance-profile CSV (one row per tau, one column per solver).

1-thread and tuned-thread results are separate tables (one per thread count).
Records are matched to manifest cases by (matrix_id, track), so the same matrix
may appear on both tracks.

Usage:
  python tools/bench/score.py --records benchmarks/results/x/records.jsonl \
      --manifest benchmarks/manifests/fronttier.json --split tuning \
      [--class hard_indef] [--shift-ms 10] [--profile-csv outdir] [--output out.md]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import statistics
import sys
from typing import Any, Iterable

ETA_GATE = 1e-10
OMEGA_GATE = 1e-8
DEFAULT_SHIFT_MS = 10.0
# Throughput gate: a record whose factorization ran faster than this many
# GFLOP/s per thread (its own flops over its first-factorization time) cannot
# be a real measurement at its thread count. It is reported as implausible:
# never solved, never the best competitor, listed per table. 165 = 1.2 x the
# 136 GF/s of single-thread OpenBLAS DGEMM (AVX-512) on the benchmark host,
# Ryzen 9 9950X3D, measured 2026-10-29; no sparse factorization beats dense
# GEMM. It caught SSIDS "1-thread" records that ran on 16 cores (up to
# 1300 GF/s). Records without positive finite flops are not judged.
DEFAULT_MAX_GFLOPS_PER_THREAD = 165.0
DEFAULT_TAUS = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
CLASS_ORDER = ["small_kkt", "large_kkt", "hard_indef", "spd"]
INDEF_COMPETITORS = ("ma57", "ma97", "ssids", "mkl_pardiso", "mumps")
SPD_COMPETITORS = ("cholmod", "mkl_pardiso", "ma57", "ma97", "ssids", "mumps")


def competitor_set(cls: str) -> tuple[str, ...]:
    """Solvers eligible as "best competitor" for a class (R43). Fixed per class,
    independent of thread count; ma27 / cudss are reference columns only."""
    return SPD_COMPETITORS if cls == "spd" else INDEF_COMPETITORS
TRACK_ORDER = ["recommended", "ipopt"]


# --------------------------------------------------------------------------
# record helpers
# --------------------------------------------------------------------------

def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _validation_ok(v: Any) -> bool:
    """Accuracy gate: eta_inf <= 1e-10 and, when present, omega <= 1e-8."""
    if not isinstance(v, dict) or v.get("finite") is False:
        return False
    eta, omega = v.get("eta_inf"), v.get("omega")
    if not _finite(eta) or eta > ETA_GATE:
        return False
    if omega is not None and (not _finite(omega) or omega > OMEGA_GATE):
        return False
    return True


def is_accurate(rec: dict[str, Any]) -> bool:
    if rec.get("outcome") != "Success":
        return False
    if not _validation_ok(rec.get("validation")):
        return False
    sub = rec.get("per_update_validation")
    if isinstance(sub, list) and not all(_validation_ok(v) for v in sub):
        return False
    return True


def factor_gflops(rec: dict[str, Any]) -> float | None:
    """GFLOP/s of flops over the record's whole factor time; None without
    positive finite flops or a factor time. flops is one factorization's count,
    so for a warm sequence (factor_ms sums batch_count steps) this is a lower
    bound: a step whose flops do not match the timed step (a fast first
    factorization, delay-heavy later ones) can never trip the gate."""
    flops = rec.get("flops")
    t = rec.get("factor_ms")
    if not _finite(flops) or flops <= 0 or not _finite(t) or t <= 0:
        return None
    return flops / t / 1e6


def is_implausible(rec: dict[str, Any],
                   max_gflops_per_thread: float = DEFAULT_MAX_GFLOPS_PER_THREAD) -> bool:
    """Factor throughput above max_gflops_per_thread x threads (<= 0: off)."""
    if max_gflops_per_thread <= 0:
        return False
    g = factor_gflops(rec)
    threads = max(1, int(rec.get("threads", 1) or 1))
    return g is not None and g > max_gflops_per_thread * threads


def record_status(rec: dict[str, Any],
                  max_gflops_per_thread: float = DEFAULT_MAX_GFLOPS_PER_THREAD) -> str:
    outcome = rec.get("outcome")
    if outcome == "Unavailable":
        return "unavailable"
    if outcome in ("Success", "Inaccurate") and is_implausible(rec, max_gflops_per_thread):
        return "implausible"
    if outcome == "Success":
        return "ok" if is_accurate(rec) else "inaccurate"
    if outcome == "Inaccurate":
        return "inaccurate"
    return "failed"


def wall_ms(rec: dict[str, Any]) -> float:
    w = float(rec.get("wall_ms") or 0.0)
    if w > 0:
        return w
    return sum(float(rec.get(k) or 0.0)
               for k in ("analyse_ms", "factor_ms", "solve_ms", "refine_ms"))


def phases(rec: dict[str, Any]) -> dict[str, float | None]:
    """analyse / first factorization / warm refactorization / solve + refine.

    Records may carry explicit `first_factor_ms` / `refactor_ms`. Otherwise a
    cold record's factor_ms is the first factorization, and a warm_sequence
    record's factor_ms (total over batch_count replayed steps) is averaged into
    the warm refactorization time.
    """
    factor = float(rec.get("factor_ms") or 0.0)
    warm = rec.get("mode") == "warm_sequence"
    first = rec.get("first_factor_ms")
    refac = rec.get("refactor_ms")
    if first is None and not warm:
        first = factor
    if refac is None and warm:
        steps = int(rec.get("batch_count") or 0)
        refac = factor / steps if steps > 0 else None
    return {
        "analyse_ms": float(rec.get("analyse_ms") or 0.0),
        "first_factor_ms": None if first is None else float(first),
        "refactor_ms": None if refac is None else float(refac),
        "solve_refine_ms": float(rec.get("solve_ms") or 0.0)
        + float(rec.get("refine_ms") or 0.0),
    }


def inertia_tuple(rec: dict[str, Any]) -> tuple[int, int, int] | None:
    i = rec.get("inertia")
    if not isinstance(i, dict):
        return None
    try:
        return int(i["pos"]), int(i["neg"]), int(i["zero"])
    except (KeyError, TypeError, ValueError):
        return None


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    out: list[dict[str, Any]] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _median(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.median(vals) if vals else None


def aggregate(recs: list[dict[str, Any]],
              max_gflops_per_thread: float = DEFAULT_MAX_GFLOPS_PER_THREAD) -> dict[str, Any]:
    """Collapse the measured trials of one (case, track, threads, solver)."""
    statuses = [record_status(r, max_gflops_per_thread) for r in recs]
    gflops = [g for g in (factor_gflops(r) for r in recs) if g is not None]
    if all(s == "unavailable" for s in statuses):
        status = "unavailable"
    elif any(s == "implausible" for s in statuses):
        status = "implausible"
    elif all(s == "ok" for s in statuses):
        status = "ok"
    elif any(s == "failed" for s in statuses):
        status = "failed"
    elif any(s == "inaccurate" for s in statuses):
        status = "inaccurate"
    else:
        status = "failed"
    ph = [phases(r) for r in recs]
    return {
        "status": status,
        "wall_ms": _median(wall_ms(r) for r in recs),
        "phases": {k: _median(p[k] for p in ph) for k in ph[0]},
        "eta": max((float(r["validation"]["eta_inf"]) for r in recs
                    if r.get("outcome") in ("Success", "Inaccurate")
                    and _finite((r.get("validation") or {}).get("eta_inf"))), default=None),
        "omega": max((float(r["validation"]["omega"]) for r in recs
                      if r.get("outcome") in ("Success", "Inaccurate")
                      and _finite((r.get("validation") or {}).get("omega"))), default=None),
        "inertia": inertia_tuple(recs[-1]),
        "n": recs[-1].get("n"),
        "gflops": max(gflops, default=None),
    }


def shifted_geomean(pairs: list[tuple[float, float]], shift_ms: float) -> float | None:
    """exp(mean(log((t + s) / (best + s)))) over (t, best) pairs."""
    if not pairs:
        return None
    return math.exp(sum(math.log((t + shift_ms) / (b + shift_ms)) for t, b in pairs)
                    / len(pairs))


def performance_profile(times: dict[str, dict[str, float | None]],
                        taus: list[float]) -> list[dict[str, float]]:
    """Dolan-More profile. times[solver][case] is an accurate time or None.

    The denominator is every case seen (including ones no solver solved, which
    therefore cap all curves below 1), as in Dolan & More."""
    cases = sorted({c for per in times.values() for c in per})
    best = {}
    for c in cases:
        vals = [per[c] for per in times.values() if per.get(c) is not None]
        if vals:
            best[c] = min(vals)
    rows = []
    for tau in taus:
        row: dict[str, float] = {"tau": tau}
        for solver, per in times.items():
            hit = sum(1 for c in cases
                      if per.get(c) is not None and c in best
                      and per[c] <= tau * best[c] * (1 + 1e-12))
            row[solver] = hit / len(cases) if cases else 0.0
        rows.append(row)
    return rows


def write_profile_csv(rows: list[dict[str, float]], solvers: list[str], fh) -> None:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["tau"] + solvers)
    for r in rows:
        w.writerow([f"{r['tau']:g}"] + [f"{r[s]:.6g}" for s in solvers])


def _inertia_reference(per_solver: dict[str, dict[str, Any]], subject: str):
    """Modal inertia of the accurate non-subject solvers; None if none/tied."""
    votes = collections.Counter()
    for sid, a in per_solver.items():
        if sid == subject or a["status"] != "ok":
            continue
        t = a["inertia"]
        if t is None or t[2] != 0:
            continue
        if a["n"] is not None and t[0] + t[1] != int(a["n"]):
            continue
        votes[t] += 1
    if not votes:
        return None
    ranked = votes.most_common(2)
    if len(ranked) == 2 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score(records: list[dict[str, Any]], manifest: dict[str, Any], split: str = "all",
          shift_ms: float = DEFAULT_SHIFT_MS, subject: str = "smf",
          classes: set[str] | None = None,
          max_gflops_per_thread: float = DEFAULT_MAX_GFLOPS_PER_THREAD) -> dict[str, Any]:
    cases: dict[tuple[str, str], dict[str, Any]] = {}
    for c in manifest.get("cases", []):
        if split != "all" and c.get("split") != split:
            continue
        if classes and c.get("class") not in classes:
            continue
        cases[(c["matrix_id"], c.get("track", "recommended"))] = c

    grouped: dict[tuple, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        if r.get("warmup"):
            continue
        key = (r["matrix_id"], r.get("track", "recommended"))
        if key not in cases:
            continue
        grouped[(key, int(r.get("threads", 1) or 1), str(r["solver_id"]))].append(r)

    # (class, track, threads) -> case -> solver -> aggregate
    tables: dict[tuple, dict[str, dict[str, dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    for (key, threads, sid), recs in grouped.items():
        cls = cases[key].get("class", "unclassified")
        tables[(cls, key[1], threads)][key[0]][sid] = aggregate(recs, max_gflops_per_thread)

    result: dict[str, Any] = {}
    for (cls, track, threads), per_case in tables.items():
        solver_ids = sorted({s for per in per_case.values() for s in per},
                            key=lambda s: (s != subject, s))
        eligible = [s for s in competitor_set(cls) if s != subject]
        missing_competitors = [s for s in eligible if s not in solver_ids]
        stats = {s: {"total": 0, "solved": 0, "inaccurate": 0, "failed": 0,
                     "implausible": 0,
                     "unavailable": 0, "inertia_pass": 0, "inertia_checked": 0,
                     "eta_max": None, "omega_max": None,
                     "wall_median_ms": None,
                     "phases": {}} for s in solver_ids}
        walls: dict[str, list[float]] = collections.defaultdict(list)
        phase_vals: dict[str, dict[str, list]] = collections.defaultdict(
            lambda: collections.defaultdict(list))
        pairs: list[tuple[float, float]] = []
        best_counts: collections.Counter = collections.Counter()
        unsolved_subject = 0
        no_competitor = 0
        report_only: dict[str, dict[str, dict[str, Any]]] = {}
        implausible: list[dict[str, Any]] = []
        times: dict[str, dict[str, float | None]] = {s: {} for s in solver_ids}

        for mid, per in per_case.items():
            if cases[(mid, track)].get("report_only"):
                report_only[mid] = per
                continue
            ref = _inertia_reference(per, subject) if track == "ipopt" else None
            for sid in solver_ids:
                a = per.get(sid)
                if a is None:
                    continue
                st = stats[sid]
                if a["status"] == "unavailable":
                    st["unavailable"] += 1
                    continue
                st["total"] += 1
                if a["status"] == "ok":
                    st["solved"] += 1
                    times[sid][mid] = a["wall_ms"]
                    walls[sid].append(a["wall_ms"])
                    for k, v in a["phases"].items():
                        phase_vals[sid][k].append(v)
                else:
                    if a["status"] == "implausible":
                        st["implausible"] += 1
                        implausible.append({"case": mid, "solver": sid, "gflops": a["gflops"],
                                            "wall_ms": a["wall_ms"]})
                    else:
                        st["inaccurate" if a["status"] == "inaccurate" else "failed"] += 1
                    times[sid][mid] = None
                for key_, val in (("eta_max", a["eta"]), ("omega_max", a["omega"])):
                    if val is not None:
                        st[key_] = val if st[key_] is None else max(st[key_], val)
                if (track == "ipopt" and ref is not None and a["status"] in ("ok", "inaccurate")
                        and a["inertia"] is not None):
                    st["inertia_checked"] += 1
                    st["inertia_pass"] += int(a["inertia"] == ref)

            subj = per.get(subject)
            comp = [(a["wall_ms"], sid) for sid, a in per.items()
                    if sid in eligible and a["status"] == "ok" and a["wall_ms"] is not None]
            if subj is None or subj["status"] == "unavailable":
                continue
            if subj["status"] != "ok":
                unsolved_subject += 1
                continue
            if comp:
                best_t, best_s = min(comp)
                pairs.append((subj["wall_ms"], best_t))
                best_counts[best_s] += 1
            else:
                no_competitor += 1

        for sid in solver_ids:
            stats[sid]["wall_median_ms"] = _median(walls[sid])
            stats[sid]["phases"] = {k: _median(v) for k, v in phase_vals[sid].items()}
        result.setdefault(cls, {}).setdefault(track, {})[threads] = {
            "solvers": stats,
            "ratio": {
                "subject": subject,
                "n": len(pairs),
                "geomean": shifted_geomean(pairs, shift_ms),
                "shift_ms": shift_ms,
                "best_solver_counts": dict(best_counts),
                "subject_unsolved": unsolved_subject,
                "no_accurate_competitor": no_competitor,
                "competitor_set": eligible,
                "missing_competitors": missing_competitors,
            },
            "report_only": report_only,
            "implausible": sorted(implausible, key=lambda i: (i["case"], i["solver"])),
            "max_gflops_per_thread": max_gflops_per_thread,
            "times": times,
            "n_cases": len(per_case) - len(report_only),
        }
    return result


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        return f"{x:.1e}"
    return f"{x:.{digits}g}"


def render(result: dict[str, Any], split: str) -> str:
    out = [f"# Front-tier score (split: {split})", ""]
    for cls in sorted(result, key=lambda c: (CLASS_ORDER.index(c) if c in CLASS_ORDER else 99, c)):
        for track in sorted(result[cls], key=lambda t: (TRACK_ORDER.index(t)
                                                        if t in TRACK_ORDER else 99, t)):
            for threads in sorted(result[cls][track]):
                g = result[cls][track][threads]
                ratio = g["ratio"]
                out.append(f"## {cls} / {track} / {threads} thread(s) "
                           f"({g['n_cases']} cases)")
                out.append("")
                gm = ratio["geomean"]
                out.append(
                    f"{ratio['subject']} vs best accurate competitor, shifted geomean "
                    f"(shift {ratio['shift_ms']:g} ms): "
                    f"**{'n/a' if gm is None else f'{gm:.3f}'}** over {ratio['n']} cases; "
                    f"{ratio['subject']} unsolved on {ratio['subject_unsolved']}; "
                    f"dropped for no accurate competitor: {ratio['no_accurate_competitor']}; "
                    f"best competitor counts: {ratio['best_solver_counts'] or '-'}")
                out.append("")
                missing = ratio.get("missing_competitors") or []
                out.append(
                    f"Competitor set ({cls}): {', '.join(ratio.get('competitor_set', [])) or '-'}; "
                    f"missing in this table: {', '.join(missing) if missing else 'none'}"
                    + (" (ratio is against the competitors present only)" if missing else ""))
                out.append("")
                ip = track == "ipopt"
                out.append("| solver | solved/total | inaccurate | failed | unavail | "
                           "max eta_inf | max omega |" + (" inertia ok |" if ip else ""))
                out.append("|---|---:|---:|---:|---:|---:|---:|" + ("---:|" if ip else ""))
                for sid, s in g["solvers"].items():
                    row = (f"| {sid} | {s['solved']}/{s['total']} | {s['inaccurate']} | "
                           f"{s['failed']} | {s['unavailable']} | {_fmt(s['eta_max'])} | "
                           f"{_fmt(s['omega_max'])} |")
                    if ip:
                        row += (f" {s['inertia_pass']}/{s['inertia_checked']} "
                                f"{'PASS' if s['inertia_pass'] == s['inertia_checked'] else 'FAIL'} |")
                    out.append(row)
                out.append("")
                out.append("Median ms over accurately solved cases:")
                out.append("")
                out.append("| solver | analyse | first factor | warm refactor | "
                           "solve+refine | wall |" + (" inertia |" if ip else ""))
                out.append("|---|---:|---:|---:|---:|---:|" + ("---:|" if ip else ""))
                for sid, s in g["solvers"].items():
                    p = s["phases"]
                    row = (f"| {sid} | {_fmt(p.get('analyse_ms'))} | "
                           f"{_fmt(p.get('first_factor_ms'))} | {_fmt(p.get('refactor_ms'))} | "
                           f"{_fmt(p.get('solve_refine_ms'))} | {_fmt(s['wall_median_ms'])} |")
                    if ip:
                        if s["inertia_checked"]:
                            row += (" pass |" if s["inertia_pass"] == s["inertia_checked"]
                                    else f" FAIL ({s['inertia_checked'] - s['inertia_pass']}) |")
                        else:
                            row += " n/a |"
                    out.append(row)
                out.append("")
                if g["report_only"]:
                    out.append("Report-only (singular; excluded from gates, ratio and profile):")
                    out.append("")
                    out.append("| case | solver | status | wall ms | eta_inf | omega | inertia |")
                    out.append("|---|---|---|---:|---:|---:|---|")
                    for mid, per in sorted(g["report_only"].items()):
                        for sid, a in sorted(per.items()):
                            out.append(f"| {mid} | {sid} | {a['status']} | {_fmt(a['wall_ms'])} | "
                                       f"{_fmt(a['eta'])} | {_fmt(a['omega'])} | {a['inertia']} |")
                    out.append("")
                if g["implausible"]:
                    cap = g["max_gflops_per_thread"]
                    out.append(f"Implausible (factor faster than {cap:g} GF/s x {threads} "
                               f"thread(s); not solved, not a competitor, counted in total only):")
                    out.append("")
                    out.append("| case | solver | factor GF/s | wall ms |")
                    out.append("|---|---|---:|---:|")
                    for i in g["implausible"]:
                        out.append(f"| {i['case']} | {i['solver']} | {_fmt(i['gflops'])} | "
                                   f"{_fmt(i['wall_ms'])} |")
                    out.append("")
    return "\n".join(out) + "\n"


def write_profiles(result: dict[str, Any], split: str, outdir: pathlib.Path,
                   taus: list[float]) -> list[pathlib.Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for cls, by_track in result.items():
        for track, by_threads in by_track.items():
            for threads, g in by_threads.items():
                solvers = list(g["times"])
                rows = performance_profile(g["times"], taus)
                path = outdir / f"profile_{split}_{cls}_{track}_t{threads}.csv"
                with path.open("w", encoding="utf-8", newline="") as fh:
                    write_profile_csv(rows, solvers, fh)
                written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True, help="records.jsonl file or directory")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="all", choices=["tuning", "heldout", "all"])
    ap.add_argument("--class", dest="classes", default="",
                    help="comma-separated class filter")
    ap.add_argument("--subject", default="smf")
    ap.add_argument("--shift-ms", type=float, default=DEFAULT_SHIFT_MS)
    ap.add_argument("--max-gflops-per-thread", type=float, default=DEFAULT_MAX_GFLOPS_PER_THREAD,
                    help="factor throughput above this x threads marks a record implausible "
                         "(0 = off; default %(default)g, the benchmark host's DGEMM peak x 1.2)")
    ap.add_argument("--profile-csv", default=None,
                    help="directory receiving one profile CSV per class/track/thread count")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    records = load_records(pathlib.Path(args.records))
    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    classes = {c for c in args.classes.split(",") if c} or None
    result = score(records, manifest, args.split, args.shift_ms, args.subject, classes,
                   args.max_gflops_per_thread)
    if not result:
        print("error: no records matched the manifest/split", file=sys.stderr)
        return 2
    n_bad = sum(len(g["implausible"]) for t in result.values() for th in t.values()
                for g in th.values())
    if n_bad:
        print(f"warning: {n_bad} implausible (case, solver, threads) result(s) excluded; "
              f"see the 'Implausible' sections", file=sys.stderr)
    md = render(result, args.split)
    print(md)
    if args.output:
        pathlib.Path(args.output).write_text(md, encoding="utf-8")
    if args.profile_csv:
        for p in write_profiles(result, args.split, pathlib.Path(args.profile_csv), DEFAULT_TAUS):
            print(f"wrote {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
