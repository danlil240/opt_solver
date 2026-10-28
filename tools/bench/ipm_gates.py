#!/usr/bin/env python3
"""Gates for the smf IPM-mode work: checks that a change left IPOPT's run alone.

identical_iterations(log_a, log_b) compares only IPOPT's iteration table
(lines like "  12  1.0e+00 ...", "  13r 1.0e+00 ..." or "2999r-5.1e+03 ...");
everything else in an
ipopt.log, such as the plugin's "[smf_ma97] ... t=..." trace and timing
lines, is ignored.

gate_abort(...) is the Feature 1 (early abort) gate: identical IPOPT runs
with and without the abort, and the IPOPT factorization time it saves.

gate_outcome(...) is the outcome gate of Features 2 and 3 (re-analysis,
static mode), which may change IPOPT's path: iterations within +-5%
(geometric mean), no new failures, wall time not worse (geometric mean over
the models whose base run takes at least 1 s).

Usage (from the repo root):
  <smfbench>/python.exe tools/bench/ipm_gates.py identical <log_a> <log_b> [...]
    compares pairs (log_a, log_b), (log_c, log_d), ...; exit 1 if any differ.
  <smfbench>/python.exe tools/bench/ipm_gates.py abort --base X.json --feat Y.json
    [--base-solver smf] [--feat-solver smf_abort]
    e2e results (ipopt_kkt.py e2e) of both solver ids, possibly one file;
    prints the per-model table and the overall pass; exit 1 on FAIL.
  <smfbench>/python.exe tools/bench/ipm_gates.py outcome --base X.json --feat Y.json [--models a,b]
    [--base-solver smf] [--feat-solver smf_reanalyse]
    the outcome gate; prints the per-model table and the overall pass; exit 1 on FAIL.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
WORK = REPO / "benchmarks" / "results" / "ipopt_work"

# IPOPT prints "%4d%c%14.7e ...": "r" marks restoration, and a 14-character
# negative objective leaves no blank after it ("2999r-5.0728013e+03 ..."),
# so "-" also ends the iteration number.
ITERATION_LINE = re.compile(r"^\s*\d+r?[\s-]")


def iteration_lines(log: str) -> list[str]:
    """IPOPT iteration-table lines of a log, in order, trailing blanks stripped."""
    return [line.rstrip() for line in log.splitlines() if ITERATION_LINE.match(line)]


def identical_iterations(log_a: str, log_b: str) -> bool:
    """True when both logs have the same IPOPT iteration table."""
    return iteration_lines(log_a) == iteration_lines(log_b)


def _first_difference(a: list[str], b: list[str]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"line {i}:\n  a: {x}\n  b: {y}"
    return f"lengths differ: {len(a)} vs {len(b)}"


# ------------------------------------------------------------ abort gate --
# Plugin trace line (SMF_PLUGIN_STATS>=2), one per factorization:
#   [smf_ma97] factor <k>: corr=<0|1> ... neg=<q> ... t=<s> ... abort=<0|1> flops=<f> skipped=<f>
TRACE_LINE = re.compile(r"^\[smf_ma97\] factor (\d+): (.*)$")
_TIME_FIELD = re.compile(r"\st=\S+")
COMPLETED_LINE = re.compile(r"^\[smf_ma97\] completed aborted factor")


def trace_entries(log: str) -> list[dict]:
    """Per-factorization trace of a log, in order: {"k", "fields", "key"}.

    `fields` maps each key=value token; `key` is the line minus its wall time
    (t=...), which is what two runs of the same factorization must share.
    """
    out = []
    for line in log.splitlines():
        m = TRACE_LINE.match(line.strip())
        if not m:
            continue
        rest = m.group(2)
        fields = dict(tok.split("=", 1) for tok in rest.split() if "=" in tok)
        out.append({"k": int(m.group(1)), "fields": fields, "key": _TIME_FIELD.sub("", " " + rest).strip()})
    return out


def compare_traces(base_log: str, feat_log: str) -> dict:
    """Match the factorizations of a run without the abort (base) and with it.

    IPOPT requests the same factorizations in both runs when the abort is
    transparent, so the traces align index by index:
      - a factorization the feature did not abort must match the base line
        field by field (t= excepted);
      - an aborted one (abort=1) shows a partial inertia, so only its corr
        flag is compared; its partial negative count must not exceed the base
        count, and base must show IPOPT rejected it (the next factorization is
        an inertia correction, corr=1). An aborted factorization IPOPT then
        solved with would have been completed by the session; it counts as a
        mismatch because it means the expectation was wrong.
    """
    base, feat = trace_entries(base_log), trace_entries(feat_log)
    res = {"match": True, "factorizations": len(feat), "aborts": 0, "completed": 0,
           "flops_avoided": 0.0, "base_flops": 0.0, "detail": ""}
    res["completed"] = sum(1 for line in feat_log.splitlines() if COMPLETED_LINE.match(line.strip()))
    res["base_flops"] = sum(float(e["fields"].get("flops", 0.0)) for e in base)

    def fail(msg: str) -> dict:
        if res["match"]:
            res["detail"] = msg
        res["match"] = False
        return res

    if len(base) != len(feat):
        fail(f"factorization count differs: {len(base)} vs {len(feat)}")
    for i, (b, f) in enumerate(zip(base, feat)):
        if f["fields"].get("abort") == "1":
            res["aborts"] += 1
            res["flops_avoided"] += float(f["fields"].get("skipped", 0.0))
            accepted_in_base = i + 1 >= len(base) or base[i + 1]["fields"].get("corr") != "1"
            if b["fields"].get("corr") != f["fields"].get("corr"):
                fail(f"factor {f['k']}: corr differs")
            elif int(f["fields"]["neg"]) > int(b["fields"]["neg"]):
                fail(f"factor {f['k']}: aborted neg {f['fields']['neg']} > base neg {b['fields']['neg']}")
            elif accepted_in_base:
                fail(f"factor {f['k']}: aborted, but IPOPT accepted it in base")
        elif b["key"] != f["key"]:
            fail(f"factor {f['k']}:\n  base: {b['key']}\n  feat: {f['key']}")
    if res["completed"]:
        fail(f"{res['completed']} aborted factorizations were completed for a solve")
    return res


def gate_abort(base: list[dict], feat: list[dict], logs_base: dict, logs_feat: dict,
               tuning: set[str] | None = None, min_saving: float = 0.05) -> dict:
    """Feature 1 gate: the early abort leaves IPOPT's run unchanged and saves time.

    base / feat: e2e result rows (ipopt_kkt.py) without / with the abort, one
    per model; logs_base / logs_feat: model -> ipopt.log text. Per model:
    `identical` (same IPOPT iteration table and EXIT status, and the
    factorization traces match, see compare_traces) and `fact_saving`
    = 1 - feat / base IPOPT LinearSystemFactorization time. `pass`: every
    model identical, and fact_saving >= min_saving on at least one identical
    tuning model (`tuning`: model names; None = every model) with inertia
    corrections in base.
    """
    base_by = {r["model"]: r for r in base}
    feat_by = {r["model"]: r for r in feat}
    models = [m for m in base_by if m in feat_by]
    per_model = {}
    for m in models:
        b, f = base_by[m], feat_by[m]
        lb, lf = logs_base.get(m, ""), logs_feat.get(m, "")
        iters_same = bool(lb) and identical_iterations(lb, lf) and b.get("status") == f.get("status")
        tr = compare_traces(lb, lf)
        tb, tf = b.get("linear_system_factorization_s"), f.get("linear_system_factorization_s")
        saving = (1.0 - tf / tb) if tb and tf is not None else None
        per_model[m] = {
            "identical": iters_same and tr["match"],
            "iterations_identical": iters_same,
            "trace_match": tr["match"],
            "trace_detail": tr["detail"],
            "iterations": f.get("iterations"),
            "factorizations": tr["factorizations"],
            "corrections": b.get("inertia_corrections"),
            "aborts": tr["aborts"],
            "flops_avoided": tr["flops_avoided"],
            "flops_avoided_frac": tr["flops_avoided"] / tr["base_flops"] if tr["base_flops"] else None,
            "fact_base_s": tb,
            "fact_feat_s": tf,
            "fact_saving": saving,
            "tuning": tuning is None or m in tuning,
        }
    all_identical = bool(per_model) and all(r["identical"] for r in per_model.values())
    # A saving counts only where the run was unchanged: a diverged run's
    # time measures a different IPOPT path.
    saves = any(r["identical"] and r["tuning"] and (r["corrections"] or 0) > 0
                and r["fact_saving"] is not None and r["fact_saving"] >= min_saving
                for r in per_model.values())
    return {"models": per_model, "all_identical": all_identical, "saving_met": saves,
            "pass": all_identical and saves}


# ---------------------------------------------------------- outcome gate --
SUCCESS_STATUSES = ("Optimal Solution Found", "Solved To Acceptable Level")


def succeeded(status: str | None) -> bool:
    return bool(status) and status.startswith(SUCCESS_STATUSES)


def _geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def _ratio(a, b):
    return a / b if a is not None and b else None


def gate_outcome(base: list[dict], feat: list[dict], models=None, wall_floor: float = 1.0,
                 iter_band: tuple[float, float] = (0.95, 1.05)) -> dict:
    """Outcome gate (Features 2 and 3): the feature may change IPOPT's path, not its outcome.

    base / feat: e2e result rows (ipopt_kkt.py) without / with the feature;
    models: names to judge (None = every base model). Per model: `iter_ratio`
    and `wall_ratio` (feat / base iterations and IPOPT wall time),
    `failed_new` (base succeeded, feat did not or has no row) and
    `in_wall_mean` (base wall >= wall_floor s; shorter runs are noise).
    `pass`: geometric-mean iteration ratio (every model) within iter_band, no
    failed_new, and geometric-mean wall ratio (models in the wall mean) <= 1
    (vacuous when no model is long enough).
    """
    base_by = {r["model"]: r for r in base}
    feat_by = {r["model"]: r for r in feat}
    names = [m for m in (models if models is not None else base_by) if m in base_by]
    per_model = {}
    for m in names:
        b, f = base_by[m], feat_by.get(m)
        if f is None:
            per_model[m] = {"iter_ratio": None, "wall_ratio": None, "fact_ratio": None,
                            "failed_new": succeeded(b.get("status")), "in_wall_mean": False,
                            "base": b, "feat": None}
            continue
        bw = b.get("ipopt_wall_s")
        per_model[m] = {
            "iter_ratio": _ratio(f.get("iterations"), b.get("iterations")),
            "wall_ratio": _ratio(f.get("ipopt_wall_s"), bw),
            "fact_ratio": _ratio(f.get("linear_system_factorization_s"), b.get("linear_system_factorization_s")),
            "failed_new": succeeded(b.get("status")) and not succeeded(f.get("status")),
            "in_wall_mean": bw is not None and bw >= wall_floor,
            "base": b,
            "feat": f,
        }
    iter_mean = _geomean([r["iter_ratio"] for r in per_model.values()])
    wall_mean = _geomean([r["wall_ratio"] for r in per_model.values() if r["in_wall_mean"]])
    iter_ok = iter_mean is not None and iter_band[0] <= iter_mean <= iter_band[1]
    no_new_failure = not any(r["failed_new"] for r in per_model.values())
    wall_ok = wall_mean is None or wall_mean <= 1.0
    return {"models": per_model, "iter_geomean": iter_mean, "wall_geomean": wall_mean,
            "iter_ok": iter_ok, "no_new_failure": no_new_failure, "wall_ok": wall_ok,
            "pass": bool(per_model) and iter_ok and no_new_failure and wall_ok}


def objective_differs(a, b, rel: float = 1e-6) -> bool:
    """True when two final objectives differ by more than `rel` relative
    (to max(|a|, |b|, 1e-3)): the runs converged to different points, e.g.
    another local optimum. Missing objectives are not flagged."""
    if a is None or b is None:
        return False
    return abs(a - b) > rel * max(abs(a), abs(b), 1e-3)


# Timings that merge_repeats replaces by their median over the repetitions.
REPEAT_TIMINGS = ("ipopt_wall_s", "linear_system_factorization_s", "linear_system_backsolve_s")


def _median(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else 0.5 * (xs[k - 1] + xs[k])


def merge_repeats(runs: list[list[dict]]) -> list[dict]:
    """One row per (model, solver) from repeated e2e runs (lists of rows).

    The first run's row is kept, with each REPEAT_TIMINGS field replaced by
    its median over the runs (the per-run values kept as <field>_runs);
    `repeats` = the number of runs of that pair and `repeats_consistent` =
    every run had the same status, iterations and objective."""
    groups: dict[tuple, list[dict]] = {}
    for rows in runs:
        for r in rows:
            groups.setdefault((r["model"], r["solver"]), []).append(r)
    merged = []
    for rows in groups.values():
        m = dict(rows[0])
        for f in REPEAT_TIMINGS:
            m[f + "_runs"] = [r.get(f) for r in rows]
            m[f] = _median(m[f + "_runs"])
        m["repeats"] = len(rows)
        m["repeats_consistent"] = all(
            (r.get("status"), r.get("iterations"), r.get("objective")) ==
            (rows[0].get("status"), rows[0].get("iterations"), rows[0].get("objective")) for r in rows)
        merged.append(m)
    return merged


def _per_fact(total, n):
    return total / n if total is not None and n else None


def outcome_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ipm_gates.py outcome", description=gate_outcome.__doc__)
    ap.add_argument("--base", required=True, help="e2e JSON with the base solver's rows")
    ap.add_argument("--feat", required=True, help="e2e JSON with the feature solver's rows (may be --base)")
    ap.add_argument("--base-solver", default="smf")
    ap.add_argument("--feat-solver", default="smf_reanalyse")
    ap.add_argument("--models", default="", help="comma-separated model names ('' = every base model)")
    ap.add_argument("--wall-floor", type=float, default=1.0)
    args = ap.parse_args(argv)
    base = _load_rows(pathlib.Path(args.base), args.base_solver)
    feat = _load_rows(pathlib.Path(args.feat), args.feat_solver)
    res = gate_outcome(base, feat, args.models.split(",") if args.models else None, args.wall_floor)

    def fmt(x, spec):
        return format(x, spec) if x is not None else "-"

    print(f"{'model':28s} {'status base -> feat':40s} {'iters':>11s} {'iter_r':>6s} {'wall_s':>17s} "
          f"{'wall_r':>6s} {'fact_s':>17s} {'d/f base':>9s} {'d/f feat':>9s} {'d/f re':>8s} {'t/a/r':>7s}")
    for m, r in res["models"].items():
        b, f = r["base"], r["feat"] or {}
        st = f"{(b.get('status') or '-')[:18]} -> {(f.get('status') or '-')[:18]}"
        tar = (f"{f.get('reanalyses_tried')}/{f.get('reanalyses_adopted')}/{f.get('reanalyses_reverted')}"
               if f.get("reanalyses_tried") is not None else "-")
        print(f"{m:28s} {st:40s} {fmt(b.get('iterations'), 'd'):>5s}/{fmt(f.get('iterations'), 'd'):>5s} "
              f"{fmt(r['iter_ratio'], '.3f'):>6s} "
              f"{fmt(b.get('ipopt_wall_s'), '.2f'):>8s}/{fmt(f.get('ipopt_wall_s'), '.2f'):>8s} "
              f"{fmt(r['wall_ratio'], '.3f'):>6s}{'' if r['in_wall_mean'] else '*'}"
              f"{fmt(b.get('linear_system_factorization_s'), '.2f'):>8s}/"
              f"{fmt(f.get('linear_system_factorization_s'), '.2f'):>8s} "
              f"{fmt(_per_fact(b.get('delays_total'), b.get('factorizations')), '.0f'):>9s} "
              f"{fmt(_per_fact(f.get('delays_total'), f.get('factorizations')), '.0f'):>9s} "
              f"{fmt(_per_fact(f.get('delays_reanalysed'), f.get('factorizations_reanalysed')), '.0f'):>8s} "
              f"{tar:>7s}" + ("  NEW FAILURE" if r["failed_new"] else ""))
    print("(* = base wall < floor, not in the wall mean; d/f = delays per factorization; "
          "d/f re = on re-analysed orders; t/a/r = re-analyses tried/adopted/reverted)")
    print(f"iteration geomean {fmt(res['iter_geomean'], '.4f')} (ok: {res['iter_ok']}); "
          f"no new failure: {res['no_new_failure']}; "
          f"wall geomean {fmt(res['wall_geomean'], '.4f')} (ok: {res['wall_ok']})")
    print("PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


def _load_rows(path: pathlib.Path, solver: str) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in rows if r.get("solver") == solver]


def _log_of(row: dict) -> str:
    p = pathlib.Path(row["log"]) if row.get("log") else \
        WORK / "e2e" / f"{row['model']}_{row['solver']}_t{row.get('threads', 1)}" / "ipopt.log"
    if not p.is_absolute():
        p = REPO / p
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def abort_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ipm_gates.py abort", description=gate_abort.__doc__)
    ap.add_argument("--base", required=True, help="e2e JSON with the base solver's rows")
    ap.add_argument("--feat", required=True, help="e2e JSON with the feature solver's rows (may be --base)")
    ap.add_argument("--base-solver", default="smf")
    ap.add_argument("--feat-solver", default="smf_abort")
    ap.add_argument("--manifest", default=str(REPO / "benchmarks" / "manifests" / "ipopt_hard.json"),
                    help="its split=tuning cases are the tuning models ('' = every model)")
    ap.add_argument("--min-saving", type=float, default=0.05)
    args = ap.parse_args(argv)
    base = _load_rows(pathlib.Path(args.base), args.base_solver)
    feat = _load_rows(pathlib.Path(args.feat), args.feat_solver)
    tuning = None
    if args.manifest and pathlib.Path(args.manifest).exists():
        cases = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))["cases"]
        tuning = {c["matrix_id"] for c in cases if c.get("split") == "tuning"}
    res = gate_abort(base, feat, {r["model"]: _log_of(r) for r in base},
                     {r["model"]: _log_of(r) for r in feat}, tuning, args.min_saving)
    print(f"{'model':28s} {'identical':9s} {'iters':>5s} {'fact':>5s} {'corr':>5s} {'aborts':>6s} "
          f"{'flops_avoided':>13s} {'base_s':>8s} {'feat_s':>8s} {'saving':>7s}")
    for m, r in res["models"].items():
        frac = f"{100 * r['flops_avoided_frac']:.1f}%" if r["flops_avoided_frac"] is not None else "-"
        sav = f"{100 * r['fact_saving']:+.1f}%" if r["fact_saving"] is not None else "-"
        print(f"{m:28s} {'yes' if r['identical'] else 'NO':9s} {r['iterations'] or 0:5d} "
              f"{r['factorizations']:5d} {r['corrections'] or 0:5d} {r['aborts']:6d} {frac:>13s} "
              f"{r['fact_base_s'] or 0:8.3f} {r['fact_feat_s'] or 0:8.3f} {sav:>7s}")
        if not r["identical"]:
            print(f"    iterations identical: {r['iterations_identical']}; trace: {r['trace_detail']}")
    print(f"all identical: {res['all_identical']}; saving >= {100 * args.min_saving:.0f}% on a tuning model "
          f"with corrections: {res['saving_met']}")
    print("PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


def main(argv: list[str]) -> int:
    if argv and argv[0] == "abort":
        return abort_main(argv[1:])
    if argv and argv[0] == "outcome":
        return outcome_main(argv[1:])
    if len(argv) < 3 or argv[0] != "identical" or len(argv[1:]) % 2:
        print(__doc__)
        return 2
    paths = [pathlib.Path(p) for p in argv[1:]]
    ok = True
    for pa, pb in zip(paths[0::2], paths[1::2]):
        la = pa.read_text(encoding="utf-8", errors="replace")
        lb = pb.read_text(encoding="utf-8", errors="replace")
        same = identical_iterations(la, lb)
        n = len(iteration_lines(la))
        print(f"{'IDENTICAL' if same else 'DIFFERENT'}  {n:5d} iteration lines  {pa}  vs  {pb}")
        if not same:
            print(_first_difference(iteration_lines(la), iteration_lines(lb)))
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
