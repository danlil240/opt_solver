#!/usr/bin/env python3
"""Offline fit of smf's thread-count cost constants (factor and solve).

smf picks the factorization's thread count with plan_threads and the solve's
with plan_solve_threads (solver/src/parallel/thread_plan.{hpp,cpp}). With
SMF_BENCH_PLAN_TRACE=1 the library prints, when it computes a plan, the terms
those models cost per candidate thread count (factor_plan_trace /
solve_plan_trace):

  plan-factor T=.. ok=.. sub_total=.. max_subtree=.. top_total=.. cp_top=..
              top_fronts=.. split_top_fronts=.. split_top_work=.. total=..
  plan-factor chosen=..
  plan-solve T=.. sub_total=.. max_subtree=.. top_total=.. top_cp=.. n_top=..
             top_rows=..
  plan-solve chosen=..

The terms do not depend on the cost constants, so this script re-evaluates
both models in Python for every point of a constant grid, picks the thread
count per case-track, and scores it against measured ("truth") times at fixed
thread counts. The re-implementation mirrors the C++ exactly:
  * candidate_threads: 1, 2, 4, 8, ... below max_threads, and max_threads;
  * least_cost_threads: in ascending T, the first candidate whose cost is
    within 2% of the least (cost <= least * 1.02); cost < 0 is no candidate;
  * factor_plan_cost: cost(1) = total; T > 1 with ok = 0 is no candidate; else
      max(max_subtree, sub_total / T) + top
      + T * per_thread + top_fronts * per_top_front
      + split_top_fronts * T * per_split_front_thread,
      top = max(cp_top, (top_total - split_top_work) / T
                        + split_top_work / min(T, split_cap)),
    split_cap below 1 counting as 1 (the split fronts' work stops speeding
    up past split_cap threads; cp_top keeps dividing it by T, as traced);
  * solve_plan_cost: cost(1) = sub_total + top_total; T > 1:
      max(max_subtree, sub_total / T) + top_cp + T * per_thread
      + n_top * per_top_step + T * top_rows * per_top_row.
    A solve with no parallel layout (no plan-solve block, or only T = 1
    traced) runs serially: T = 1.
The trace prints %.6g, so a cost within ~1e-6 of the 2% tie boundary can
resolve differently from the library.

Traces are collected with SMF_BENCH_AUTO_THREADS=0 at 16 threads (ruling
R18): the factorization is partitioned for all 16 threads, so the solve's
terms are those of the 16-thread solve layout whatever the factor constants;
with auto_threads off the library evaluates both plans for the trace only.
Evaluation at max_threads 8 uses the candidates <= 8 of the 16-thread trace:
exact for the factor (each candidate's partition depends on T alone), an
approximation for the solve (an 8-thread factorization leaves a different
layout).

Loss of a grid point: sum over case-tracks and over max_threads in (16, 8) of
log(measured(T_chosen) / min over T <= max_threads of measured(T)), measured =
median factor ms (refactor, else first factor) or solve + refine ms of the
truth records (score.aggregate, as thread_scaling.py). Case-tracks whose
1-thread time is below --min-ms, or with no measurement at some candidate T
(1, 2, 4, 8, 16), or whose trace lacks the terms of a candidate T, are skipped
at every grid point (the last counted as skipped_trace). A1 proxy: case-tracks
whose chosen T (max 16) is > 1.05 x slower than 1 thread; A2 proxy: > 1.05 x
slower than 8 threads (8-thread time >= --min-ms); real-A2: the measured time
at the max-16 pick > 1.05 x the measured time at the max-8 pick (that time >=
--min-ms), the `16>8` flag of thread_scaling.py evaluated on truth.

Usage:
  # traces: bench_replay --print-steps per case-track (SMF_BENCH_AUTO_THREADS=0,
  # 16 threads), killed after step 0
  python tools/bench/fit_thread_costs.py --collect-traces \\
      --binary build/benchmarks/Release/bench_replay.exe \\
      --manifest benchmarks/manifests/fronttier.json --traces DIR [--cases a,b]
  # fit
  python tools/bench/fit_thread_costs.py --truth DIR [DIR ...] --traces DIR \\
      --manifest benchmarks/manifests/fronttier.json --output fit.md
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import threading
from typing import Any, NamedTuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import score  # noqa: E402
import thread_scaling  # noqa: E402

FIT_CLASSES = ("large_kkt", "hard_indef", "spd")
DEFAULT_REPO_ROOT = "C:/Users/daniel/projects/opt_solver"
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
TIE = 1.02
RATIO_TOL = thread_scaling.RATIO_TOL
MAX_THREADS_EVAL = (16, 8)


class FactorConsts(NamedTuple):
    per_thread: float
    per_top_front: float
    per_split_front_thread: float
    split_cap: int


class SolveConsts(NamedTuple):
    per_thread: float
    per_top_step: float
    per_top_row: float


FACTOR_GRID = [FactorConsts(*p) for p in itertools.product(
    (2e5, 5e5, 7e5, 1e6), (2e3, 5e3, 1e4), (0.0, 1e5, 2e5, 3e5, 5e5, 1e6), (4, 6, 8, 12, 16))]
SOLVE_GRID = [SolveConsts(*p) for p in itertools.product(
    (5e3, 1e4, 2e4, 5e4, 1e5), (0.0, 50.0, 100.0, 200.0, 500.0, 1000.0), (0.25, 0.5, 1.0, 2.0))]
FACTOR_DEFAULT = FactorConsts(7e5, 5e3, 3e5, 8)   # ThreadCosts{}
SOLVE_DEFAULT = SolveConsts(2e4, 100.0, 1.0)      # SolveCosts{}

# --------------------------------------------------------------------------
# trace parsing
# --------------------------------------------------------------------------

_KV = re.compile(r"(\w+)=(\S+)")
_INT_FIELDS = {"T", "ok", "top_fronts", "split_top_fronts", "n_top", "top_rows"}


def _fields(line: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in _KV.findall(line):
        out[k] = int(v) if k in _INT_FIELDS else float(v)
    return out


def parse_trace(text: str) -> dict[str, Any]:
    """The first plan-factor and first plan-solve block of a trace:
    {"factor": {T: terms}, "factor_chosen": T | None, "solve": {T: terms},
    "solve_chosen": T | None}. A block ends at its "chosen=" line."""
    tr: dict[str, Any] = {"factor": {}, "factor_chosen": None, "solve": {}, "solve_chosen": None}
    done = {"factor": False, "solve": False}
    for raw in text.splitlines():
        line = raw.strip()
        for kind in ("factor", "solve"):
            prefix = f"plan-{kind} "
            if not line.startswith(prefix) or done[kind]:
                continue
            f = _fields(line[len(prefix):])
            if "chosen" in f:
                tr[f"{kind}_chosen"] = int(f["chosen"])
                done[kind] = True
            elif "T" in f:
                if kind == "factor":
                    f["ok"] = bool(f.get("ok", 1))
                tr[kind][f["T"]] = f
    return tr


# --------------------------------------------------------------------------
# the models (mirror of thread_plan.cpp)
# --------------------------------------------------------------------------

def candidate_threads(max_threads: int) -> list[int]:
    """thread_plan.cpp candidate_threads."""
    out = [1]
    if max_threads < 2:
        return out
    t = 2
    while t < max_threads:
        out.append(t)
        t *= 2
    out.append(max_threads)
    return out


def least_cost_threads(costs: list[tuple[int, float]]) -> int:
    """thread_plan.cpp least_cost_threads: (T, cost) ascending in T, the
    first T = 1; cost < 0 is no candidate; the first within 2% of the least."""
    least = costs[0][1]
    for _, c in costs:
        if c >= 0.0:
            least = min(least, c)
    for t, c in costs:
        if c >= 0.0 and c <= least * TIE:
            return t
    return costs[0][0]


def factor_cost(k: dict[str, Any], c: FactorConsts) -> float:
    """thread_plan.cpp factor_plan_cost."""
    T = k["T"]
    if T <= 1:
        return k["total"]
    if not k["ok"]:
        return -1.0
    cap = max(1, int(c.split_cap))
    top = max(k["cp_top"], (k["top_total"] - k["split_top_work"]) / T
              + k["split_top_work"] / min(T, cap))
    return (max(k["max_subtree"], k["sub_total"] / T) + top
            + T * c.per_thread + float(k["top_fronts"]) * c.per_top_front
            + float(k["split_top_fronts"]) * T * c.per_split_front_thread)


def _candidates(terms: dict[int, dict[str, Any]], max_threads: int) -> list[int]:
    cands = candidate_threads(max_threads)
    missing = [t for t in cands if t not in terms]
    if missing:
        raise ValueError(f"trace has no terms at T = {missing} (traced: {sorted(terms)})")
    return cands


def trace_covers(terms: dict[int, dict[str, Any]], max_threads: int) -> bool:
    """Whether the traced terms can be evaluated at max_threads: a block of
    only T = 1 (or none) is the serial answer; else every candidate T needs
    its terms. evaluate skips (and counts) the case-tracks where not."""
    return set(terms) <= {1} or all(t in terms for t in candidate_threads(max_threads))


def pick_factor_threads(terms: dict[int, dict[str, Any]], max_threads: int,
                        c: FactorConsts) -> int:
    """plan_threads on traced terms. Only T = 1 traced (a tree below
    kTreeParallelMinWork): 1."""
    if set(terms) <= {1}:
        return 1
    return least_cost_threads([(t, factor_cost(terms[t], c))
                               for t in _candidates(terms, max_threads)])


def solve_cost(k: dict[str, Any], T: int, c: SolveConsts) -> float:
    """thread_plan.cpp solve_plan_cost."""
    if T <= 1:
        return k["sub_total"] + k["top_total"]
    n_top = float(k["n_top"])
    rows = float(k["top_rows"])
    return (max(k["max_subtree"], k["sub_total"] / T) + k["top_cp"] + T * c.per_thread
            + n_top * c.per_top_step + T * rows * c.per_top_row)


def pick_solve_threads(terms: dict[int, dict[str, Any]], max_threads: int,
                       c: SolveConsts) -> int:
    """plan_solve_threads on traced terms (the same at every T). No plan or
    only T = 1 traced (no parallel solve layout): 1."""
    if set(terms) <= {1}:
        return 1
    k = terms[1] if 1 in terms else next(iter(terms.values()))
    return least_cost_threads([(t, solve_cost(k, t, c))
                               for t in _candidates(terms, max_threads)])


# --------------------------------------------------------------------------
# truth, loss, grid
# --------------------------------------------------------------------------

def measured_series(records: list[dict[str, Any]], manifest: dict[str, Any],
                    subject: str = "smf") -> dict[tuple[str, str], dict[str, dict[int, float]]]:
    """(matrix_id, track) -> {"factor": {T: ms}, "solve": {T: ms}}: medians of
    the subject's accurate records, as thread_scaling.analyse."""
    cases = {(c["matrix_id"], c.get("track", "recommended")) for c in manifest.get("cases", [])}
    grouped: dict[tuple, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        if r.get("warmup") or r.get("solver_id") != subject:
            continue
        key = (r["matrix_id"], r.get("track", "recommended"))
        if key in cases:
            grouped[(key, int(r.get("threads", 1) or 1))].append(r)
    by_case: dict[tuple, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    for (key, threads), recs in grouped.items():
        by_case[key][threads] = score.aggregate(recs)
    return {key: {"factor": thread_scaling._series(by_t, thread_scaling._factor_ms),
                  "solve": thread_scaling._series(by_t, lambda a: a["phases"]["solve_refine_ms"])}
            for key, by_t in by_case.items()}


def case_loss(chosen: int, series: dict[int, float], max_threads: int) -> float | None:
    """log(measured(chosen) / best measured at T <= max_threads); None when
    either is not measured."""
    within = [v for t, v in series.items() if t <= max_threads]
    if chosen not in series or not within:
        return None
    return math.log(series[chosen] / min(within))


def _pick(kind: str, trace: dict[str, Any], max_threads: int, c) -> int:
    if kind == "factor":
        return pick_factor_threads(trace["factor"], max_threads, c)
    return pick_solve_threads(trace["solve"], max_threads, c)


def evaluate(cases: list[dict[str, Any]], kind: str, c, min_ms: float = 1.0,
             max_threads: tuple[int, ...] = MAX_THREADS_EVAL) -> dict[str, Any]:
    """Loss, A1 / A2 proxies and per-case rows of one constant set. A
    case-track counts only with a measurement at every candidate T (so the
    same case-tracks count at every grid point) and a 1-thread time >= min_ms."""
    loss = 0.0
    by_max = {m: 0.0 for m in max_threads}
    a1 = a2 = a2_real = used = skipped_fast = skipped_incomplete = skipped_trace = 0
    need = {t for m in max_threads for t in candidate_threads(m)}
    rows = []
    for cs in cases:
        series = cs[kind]
        base = series.get(1)
        if base is None or base < min_ms:
            skipped_fast += 1
            continue
        if not need <= set(series):
            skipped_incomplete += 1
            continue
        if not all(trace_covers(cs["trace"][kind], m) for m in max_threads):
            skipped_trace += 1
            continue
        row = {"case": cs["key"][0], "track": cs["key"][1], "class": cs.get("class", ""),
               "traced": cs["trace"][f"{kind}_chosen"], "chosen": {}, "loss": {}}
        for m in max_threads:
            t = _pick(kind, cs["trace"], m, c)
            row["chosen"][m] = t
            row["loss"][m] = case_loss(t, series, m)
        used += 1
        for m, lo in row["loss"].items():
            loss += lo
            by_max[m] += lo
        top = max_threads[0]
        t = row["chosen"][top]
        within = {tt: v for tt, v in series.items() if tt <= top}
        best = min(within, key=lambda tt: (within[tt], tt))
        row.update({"ms": series[t], "best": best, "best_ms": series[best],
                    "ratio": series[t] / series[best], "series": series})
        if series[t] > RATIO_TOL * base:
            a1 += 1
            row["a1"] = True
        s8 = series.get(8)
        if s8 is not None and s8 >= min_ms and series[t] > RATIO_TOL * s8:
            a2 += 1
            row["a2"] = True
        if len(max_threads) > 1:
            t_low = row["chosen"][max_threads[1]]
            if series[t_low] >= min_ms and series[t] > RATIO_TOL * series[t_low]:
                a2_real += 1
                row["a2_real"] = True
        rows.append(row)
    return {"loss": loss, "loss_by_max": by_max, "a1": a1, "a2": a2, "a2_real": a2_real,
            "used": used, "skipped": skipped_fast + skipped_incomplete + skipped_trace,
            "skipped_fast": skipped_fast, "skipped_incomplete": skipped_incomplete,
            "skipped_trace": skipped_trace, "rows": rows}


def grid_search(cases: list[dict[str, Any]], kind: str, grid, min_ms: float = 1.0,
                max_threads: tuple[int, ...] = MAX_THREADS_EVAL) -> list[tuple[Any, dict]]:
    """[(constants, evaluate(...))] over the grid, least loss first (grid order
    on ties)."""
    out = [(c, evaluate(cases, kind, c, min_ms, max_threads)) for c in grid]
    out.sort(key=lambda x: x[1]["loss"])
    return out


# --------------------------------------------------------------------------
# inputs and report
# --------------------------------------------------------------------------

def trace_path(traces: pathlib.Path, matrix_id: str, track: str) -> pathlib.Path:
    return traces / f"{matrix_id}__{track}.txt"


def fit_case_keys(manifest: dict[str, Any], only: set[str] | None = None) -> list[dict[str, Any]]:
    out = []
    for c in manifest.get("cases", []):
        if c.get("class") not in FIT_CLASSES:
            continue
        if only and c["matrix_id"] not in only:
            continue
        out.append(c)
    return out


def build_cases(records: list[dict[str, Any]], manifest: dict[str, Any],
                traces: pathlib.Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Case-tracks of FIT_CLASSES with a trace and truth records; and the keys
    missing a trace / truth."""
    truth = measured_series(records, manifest)
    cases, no_trace, no_truth = [], [], []
    for c in fit_case_keys(manifest):
        key = (c["matrix_id"], c.get("track", "recommended"))
        name = f"{key[0]} ({key[1]})"
        p = trace_path(traces, *key)
        if not p.is_file():
            no_trace.append(name)
            continue
        if key not in truth:
            no_truth.append(name)
            continue
        cases.append({"key": key, "class": c.get("class", ""),
                      "trace": parse_trace(p.read_text(encoding="utf-8", errors="replace")),
                      **truth[key]})
    return cases, no_trace, no_truth


def _g(x: float) -> str:
    return f"{x:g}"


def _rank_table(ranked: list[tuple[Any, dict]], default, default_eval: dict, n: int,
                max_threads: tuple[int, ...]) -> list[str]:
    names = list(type(default)._fields)
    head = ["rank"] + names + ["loss"] + [f"loss@{m}" for m in max_threads] + [
        "A1 proxy", "A2 proxy", "real-A2", "cases"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def row(label, c, ev):
        return ("| " + " | ".join([label] + [_g(v) for v in c] + [f"{ev['loss']:.4f}"]
                                  + [f"{ev['loss_by_max'][m]:.4f}" for m in max_threads]
                                  + [str(ev["a1"]), str(ev["a2"]), str(ev["a2_real"]),
                                     str(ev["used"])]) + " |")

    for i, (c, ev) in enumerate(ranked[:n], 1):
        lines.append(row(str(i), c, ev))
    lines.append(row("default", default, default_eval))
    if ranked:
        best = ranked[0][1]["loss"]
        ties = sum(1 for _, ev in ranked if ev["loss"] <= best * 1.01 + 1e-12)
        need = sorted({t for m in max_threads for t in candidate_threads(m)})
        lines += ["", f"{ties} of {len(ranked)} grid points within 1% of the best loss. "
                  f"Case-tracks: {default_eval['used']} counted, {default_eval['skipped_fast']} "
                  f"below min-ms at 1 thread, {default_eval['skipped_incomplete']} without a "
                  f"measurement at every candidate T ({', '.join(map(str, need))}), "
                  f"{default_eval['skipped_trace']} whose trace lacks a candidate T."]
    return lines


def _case_table(ev: dict[str, Any], max_threads: tuple[int, ...]) -> list[str]:
    hi, lo = max_threads[0], max_threads[-1]
    lines = [f"| case | track | class | traced T | chosen T (max {hi}) | ms at chosen | best T | "
             f"best ms | ratio | chosen T (max {lo}) | measured ms by T | flags |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ev["rows"], key=lambda r: (-r["ratio"], r["case"], r["track"])):
        flags = " ".join(f for f, k in (("A1", "a1"), ("A2", "a2"), ("real-A2", "a2_real"))
                         if r.get(k))
        series = " ".join(f"{t}:{v:.1f}" for t, v in sorted(r["series"].items()))
        traced = "-" if r["traced"] is None else str(r["traced"])
        lines.append(f"| {r['case']} | {r['track']} | {r['class']} | {traced} | "
                     f"{r['chosen'][hi]} | {r['ms']:.2f} | {r['best']} | {r['best_ms']:.2f} | "
                     f"{r['ratio']:.3f} | {r['chosen'][lo]} | {series} | {flags} |")
    return lines


def render(cases: list[dict[str, Any]], no_trace: list[str], no_truth: list[str],
           factor_ranked, solve_ranked, factor_default, solve_default, min_ms: float,
           top_n: int = 5, max_threads: tuple[int, ...] = MAX_THREADS_EVAL) -> str:
    lines = [
        "# Thread-cost fit",
        "",
        f"{len(cases)} case-tracks with a plan trace and truth records "
        f"(classes {', '.join(FIT_CLASSES)}); per model, case-tracks with a 1-thread time "
        f"below {min_ms:g} ms or without a measurement at every candidate T are skipped. Loss = sum over case-tracks and max_threads "
        f"{' and '.join(map(str, max_threads))} of log(measured(chosen T) / best "
        "measured T <= max_threads). A1 / A2 proxies at max "
        f"{max_threads[0]}: chosen T > {RATIO_TOL} x the 1-thread / 8-thread time; "
        f"real-A2: the measured time at the max-{max_threads[0]} pick > {RATIO_TOL} x that at "
        f"the max-{max_threads[-1]} pick. "
        "`default` is the C++ defaults (Control::thread_cost_*, solve_cost_*).",
        "",
        f"## Factor: {top_n} best",
        "",
    ]
    lines += _rank_table(factor_ranked, FACTOR_DEFAULT, factor_default, top_n, max_threads)
    lines += ["", f"## Solve: {top_n} best", ""]
    lines += _rank_table(solve_ranked, SOLVE_DEFAULT, solve_default, top_n, max_threads)
    if factor_ranked:
        best_c, best_ev = factor_ranked[0]
        lines += ["", f"## Factor per case at the best point {tuple(best_c)}", ""]
        lines += _case_table(best_ev, max_threads)
    if solve_ranked:
        best_c, best_ev = solve_ranked[0]
        lines += ["", f"## Solve per case at the best point {tuple(best_c)}", ""]
        lines += _case_table(best_ev, max_threads)
    if no_trace or no_truth:
        lines += ["", "## Missing inputs", ""]
        if no_trace:
            lines.append(f"No trace ({len(no_trace)}): " + ", ".join(no_trace))
        if no_truth:
            lines.append(f"No truth records ({len(no_truth)}): " + ", ".join(no_truth))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# trace collection
# --------------------------------------------------------------------------

def replay_command(binary: pathlib.Path, manifest_path: pathlib.Path, case: dict[str, Any],
                   threads: int) -> list[str]:
    """bench_replay's command line as run.py builds it, with --print-steps."""
    matrix_id = case["matrix_id"]
    seq = case.get("sequence_id", matrix_id)
    cmd = [str(binary), "--manifest", str(manifest_path), "--case", matrix_id,
           "--sequence-id", seq, "--family-id", case.get("family_id", seq),
           "--solver", "smf", "--mode", case.get("mode", "cold"),
           "--track", case.get("track", "recommended"), "--trial", "0",
           "--threads", str(threads), "--print-steps"]
    if case.get("mtx_path"):
        cmd += ["--mtx", str(case["mtx_path"])]
    if case.get("matrix_type"):
        cmd += ["--mtype", str(case["matrix_type"])]
    if case.get("seq_values_path"):
        cmd += ["--seq-values", str(case["seq_values_path"])]
    if case.get("ordering"):  # run.replay_override_args
        cmd += ["--ordering", str(case["ordering"])]
    return cmd


def trace_env(base: dict[str, str], threads: int) -> dict[str, str]:
    """The replay's environment: the plan trace on, auto_threads off (R18:
    the factorization and the solve layout at all `threads`; the library
    still evaluates the plans for the trace), default cost constants, and
    the thread caps run.py exports."""
    env = dict(base)
    env["SMF_BENCH_PLAN_TRACE"] = "1"
    env["SMF_BENCH_AUTO_THREADS"] = "0"
    for var in ("SMF_BENCH_THREAD_COSTS", "SMF_BENCH_SOLVE_COSTS"):
        env.pop(var, None)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = str(threads)
    return env


def collect_one(cmd: list[str], cwd: pathlib.Path, threads: int, timeout_s: float) -> tuple[str, str]:
    """Runs one replay with the plan trace on; returns (stderr up to and with
    the first step line, how it ended)."""
    env = trace_env(dict(os.environ), threads)
    flags = BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", creationflags=flags)
    fired = threading.Event()

    def on_timeout() -> None:
        fired.set()
        proc.kill()

    timer = threading.Timer(timeout_s, on_timeout)
    timer.start()
    lines: list[str] = []
    how = "exited"
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            lines.append(line)
            if line.startswith("step "):
                how = "killed after step 0"
                break
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        timer.cancel()
    if fired.is_set() and how == "exited":
        how = f"timeout ({timeout_s:g} s)"
    elif how == "exited":
        how = f"exited {proc.returncode}"
    return "".join(lines), how


def collect_traces(manifest_path: pathlib.Path, binary: pathlib.Path, traces: pathlib.Path,
                   repo_root: pathlib.Path, threads: int, only: set[str] | None,
                   timeout_s: float) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    traces.mkdir(parents=True, exist_ok=True)
    cases = fit_case_keys(manifest, only)
    if not cases:
        print("no case-tracks selected", file=sys.stderr)
        return 1
    for c in cases:
        track = c.get("track", "recommended")
        cmd = replay_command(binary, manifest_path, c, threads)
        text, how = collect_one(cmd, repo_root, threads, timeout_s)
        trace_path(traces, c["matrix_id"], track).write_text(text, encoding="utf-8")
        tr = parse_trace(text)
        print(f"{c['matrix_id']} ({track}): {how}; plan-factor T={sorted(tr['factor'])} "
              f"chosen={tr['factor_chosen']}; plan-solve chosen={tr['solve_chosen']}", flush=True)
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--traces", required=True,
                    help="directory of <matrix_id>__<track>.txt plan traces")
    ap.add_argument("--truth", nargs="+", help="run.py record files or directories")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-threads", type=int, default=16,
                    help="thread count of the traces; the fit also evaluates at 8")
    ap.add_argument("--min-ms", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--collect-traces", action="store_true",
                    help="run bench_replay per case-track and write the traces; the replays "
                         "run with SMF_BENCH_AUTO_THREADS=0 and SMF_BENCH_PLAN_TRACE=1 at "
                         "--max-threads threads (R18), cost env overrides removed")
    ap.add_argument("--binary", default=None, help="bench_replay executable (--collect-traces)")
    ap.add_argument("--repo-root", default=DEFAULT_REPO_ROOT,
                    help="working directory of the replays (manifest paths are relative to it)")
    ap.add_argument("--cases", default=None, help="comma-separated matrix_id filter (collect)")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per replay (collect)")
    args = ap.parse_args(argv)

    manifest_path = pathlib.Path(args.manifest)
    if not manifest_path.exists() and not manifest_path.is_absolute():
        manifest_path = pathlib.Path(args.repo_root) / manifest_path  # relative to the repo
    traces = pathlib.Path(args.traces)

    if args.collect_traces:
        if not args.binary:
            ap.error("--collect-traces needs --binary")
        only = {s for s in args.cases.split(",") if s} if args.cases else None
        return collect_traces(manifest_path.resolve(), pathlib.Path(args.binary).resolve(),
                              traces.resolve(), pathlib.Path(args.repo_root), args.max_threads,
                              only, args.timeout)

    if not args.truth:
        ap.error("--truth is required unless --collect-traces")
    records: list[dict[str, Any]] = []
    for p in args.truth:
        records.extend(score.load_records(pathlib.Path(p)))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases, no_trace, no_truth = build_cases(records, manifest, traces)
    mt = (args.max_threads, 8) if args.max_threads > 8 else (args.max_threads,)
    factor_ranked = grid_search(cases, "factor", FACTOR_GRID, args.min_ms, mt)
    solve_ranked = grid_search(cases, "solve", SOLVE_GRID, args.min_ms, mt)
    md = render(cases, no_trace, no_truth, factor_ranked, solve_ranked,
                evaluate(cases, "factor", FACTOR_DEFAULT, args.min_ms, mt),
                evaluate(cases, "solve", SOLVE_DEFAULT, args.min_ms, mt), args.min_ms, args.top, mt)
    print(md)
    if args.output:
        pathlib.Path(args.output).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
