#!/usr/bin/env python3
"""Gates for the smf IPM-mode work: checks that a change left IPOPT's run alone.

identical_iterations(log_a, log_b) compares only IPOPT's iteration table
(lines like "  12  1.0e+00 ...", "  13r 1.0e+00 ..." or "2999r-5.1e+03 ...");
everything else in an
ipopt.log, such as the plugin's "[smf_ma97] ... t=..." trace and timing
lines, is ignored.

gate_abort(...) is the Feature 1 (early abort) gate: identical IPOPT runs
with and without the abort, and the IPOPT factorization time it saves.

Usage (from the repo root):
  <smfbench>/python.exe tools/bench/ipm_gates.py identical <log_a> <log_b> [...]
    compares pairs (log_a, log_b), (log_c, log_d), ...; exit 1 if any differ.
  <smfbench>/python.exe tools/bench/ipm_gates.py abort --base X.json --feat Y.json
    [--base-solver smf] [--feat-solver smf_abort]
    e2e results (ipopt_kkt.py e2e) of both solver ids, possibly one file;
    prints the per-model table and the overall pass; exit 1 on FAIL.
"""

from __future__ import annotations

import argparse
import json
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
