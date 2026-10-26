#!/usr/bin/env python3
"""Capture real IPOPT KKT sequences and time IPOPT end-to-end per linear solver.

Builds scalable NLP test problems with Pyomo (run with the `smfbench` conda
env's python, which has pyomo + ipopt), writes each as an AMPL .nl file, then:

  capture : runs IPOPT with linear_solver=ma97 + hsllib=<smf_ma97 plugin>
            and SMF_KKT_DUMP_DIR set, so the plugin records the KKT pattern
            (kkt_s0.mtx) and every factorization's values (kkt_s0.vals) that
            IPOPT actually requested. The capture is copied to
            benchmarks/fixtures/ipopt_kkt/<model>/ and a manifest
            (benchmarks/manifests/ipopt_kkt.json, warm_sequence mode) is written.
  e2e     : runs IPOPT on each model with every requested linear solver
            (mumps = bundled, ma27 = CoinHSL archive via hsllib, smf = plugin)
            and writes iterations / timings to a JSON report.

Usage (from the repo root):
  <smfbench>/python.exe tools/bench/ipopt_kkt.py capture [--models a,b] [--scale 1]
  <smfbench>/python.exe tools/bench/ipopt_kkt.py e2e --solvers mumps,ma27,smf --threads 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import time

import hard_models
from plugin_trace import parse_trace

REPO = pathlib.Path(__file__).resolve().parents[2]
BUILD_REL = REPO / "build" / "Release"
COINHSL_BIN = pathlib.Path(
    os.environ.get(
        "SMF_COINHSL_BIN",
        r"C:\Users\daniel\Downloads\CoinHSL-archive.v2024.5.15.x86_64-w64-mingw32-libgfortran5"
        r"\CoinHSL-archive.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin",
    )
)
IPOPT_EXE = pathlib.Path(sys.executable).parent / "Library" / "bin" / "ipopt.exe"
WORK = REPO / "benchmarks" / "results" / "ipopt_work"


# ----------------------------------------------------------------- models --
def pde_control(N: int):
    """Semilinear elliptic optimal control on an N x N grid with control
    bounds: min 1/2|y - yd|^2 + a/2 |u|^2  s.t.  -lap(y) + y^3 = u."""
    import pyomo.environ as pe
    h = 1.0 / (N + 1)
    m = pe.ConcreteModel()
    I = range(N)
    m.y = pe.Var(I, I, initialize=0.0)
    m.u = pe.Var(I, I, bounds=(-5.0, 15.0), initialize=0.0)
    yd = {(i, j): 3.0 * math.sin(2 * math.pi * (i + 1) * h) * math.sin(math.pi * (j + 1) * h)
          for i in I for j in I}
    a = 1e-3
    m.obj = pe.Objective(expr=0.5 * h * h * sum((m.y[i, j] - yd[i, j]) ** 2 + a * m.u[i, j] ** 2
                                               for i in I for j in I))

    def yv(i, j):
        return m.y[i, j] if 0 <= i < N and 0 <= j < N else 0.0

    m.pde = pe.Constraint(I, I, rule=lambda m, i, j: (
        4 * m.y[i, j] - yv(i - 1, j) - yv(i + 1, j) - yv(i, j - 1) - yv(i, j + 1)) / (h * h)
        + m.y[i, j] ** 3 - m.u[i, j] == 0)
    return m


def robot_arm(N: int):
    """COPS 'robot': minimum-time point-to-point motion of a robot arm with
    6 states and 3 bounded controls, trapezoidal collocation on N intervals."""
    import pyomo.environ as pe
    L = 5.0
    m = pe.ConcreteModel()
    K = range(N + 1)
    m.tf = pe.Var(bounds=(0.1, 20.0), initialize=9.0)
    m.rho = pe.Var(K, bounds=(0, L), initialize=4.5)
    m.th = pe.Var(K, bounds=(-math.pi, math.pi), initialize=lambda m, k: 2 * math.pi / 3 * (k / N) ** 2)
    m.ph = pe.Var(K, bounds=(0, math.pi), initialize=math.pi / 4)
    m.rd = pe.Var(K, initialize=0.0)
    m.td = pe.Var(K, initialize=lambda m, k: 4 * math.pi / 3 * k / N)
    m.pd = pe.Var(K, initialize=0.0)
    m.ur = pe.Var(K, bounds=(-1, 1), initialize=0.0)
    m.ut = pe.Var(K, bounds=(-1, 1), initialize=0.0)
    m.up = pe.Var(K, bounds=(-1, 1), initialize=0.0)
    m.obj = pe.Objective(expr=m.tf)

    def I_t(k):
        return ((L - m.rho[k]) ** 3 + m.rho[k] ** 3) / 3 * pe.sin(m.ph[k]) ** 2

    def I_p(k):
        return ((L - m.rho[k]) ** 3 + m.rho[k] ** 3) / 3

    h = lambda: m.tf / N
    S = range(N)
    m.c1 = pe.Constraint(S, rule=lambda m, k: m.rho[k + 1] == m.rho[k] + h() / 2 * (m.rd[k] + m.rd[k + 1]))
    m.c2 = pe.Constraint(S, rule=lambda m, k: m.th[k + 1] == m.th[k] + h() / 2 * (m.td[k] + m.td[k + 1]))
    m.c3 = pe.Constraint(S, rule=lambda m, k: m.ph[k + 1] == m.ph[k] + h() / 2 * (m.pd[k] + m.pd[k + 1]))
    m.c4 = pe.Constraint(S, rule=lambda m, k: L * (m.rd[k + 1] - m.rd[k]) == h() / 2 * (m.ur[k] + m.ur[k + 1]))
    m.c5 = pe.Constraint(S, rule=lambda m, k: I_t(k) * (m.td[k + 1] - m.td[k]) == h() / 2 * (m.ut[k] + m.ut[k + 1]))
    m.c6 = pe.Constraint(S, rule=lambda m, k: I_p(k) * (m.pd[k + 1] - m.pd[k]) == h() / 2 * (m.up[k] + m.up[k + 1]))
    for v, a, b in ((m.rho, 4.5, 4.5), (m.th, 0.0, 2 * math.pi / 3), (m.ph, math.pi / 4, math.pi / 4)):
        v[0].fix(a)
        v[N].fix(b)
    for v in (m.rd, m.td, m.pd):
        v[0].fix(0.0)
        v[N].fix(0.0)
    return m


def obstacle_surface(N: int):
    """Minimal surface over an obstacle (COPS-style), bound constrained."""
    import pyomo.environ as pe
    h = 1.0 / (N + 1)
    m = pe.ConcreteModel()
    I = range(N)

    def lb(i, j):
        x, y = (i + 1) * h, (j + 1) * h
        return 1.0 if abs(x - 0.5) < 0.25 and abs(y - 0.5) < 0.25 else -1e20

    m.v = pe.Var(I, I, initialize=lambda m, i, j: max(0.0, lb(i, j)),
                 bounds=lambda m, i, j: (None if lb(i, j) < -1e19 else lb(i, j), None))

    def vv(i, j):
        if 0 <= i < N and 0 <= j < N:
            return m.v[i, j]
        return 1.0 - 4 * ((i + 1) * h - 0.5) ** 2 if j < 0 or j >= N else 0.0

    m.obj = pe.Objective(expr=h * h * sum(
        pe.sqrt(1 + ((vv(i + 1, j) - vv(i, j)) / h) ** 2 + ((vv(i, j + 1) - vv(i, j)) / h) ** 2)
        for i in range(-1, N) for j in range(-1, N)))
    return m


def acopf_grid(N: int, seed: int = 7):
    """Synthetic AC optimal power flow (polar form) on an N x N meshed grid of
    buses with random line impedances, loads and generators."""
    import pyomo.environ as pe
    rng = random.Random(seed)
    buses = [(i, j) for i in range(N) for j in range(N)]
    idx = {b: k for k, b in enumerate(buses)}
    lines = []
    for (i, j) in buses:
        for (di, dj) in ((1, 0), (0, 1)):
            if i + di < N and j + dj < N:
                r, x = rng.uniform(0.005, 0.02), rng.uniform(0.02, 0.1)
                z2 = r * r + x * x
                lines.append((idx[(i, j)], idx[(i + di, j + dj)], r / z2, -x / z2))
    B = len(buses)
    gens = sorted(rng.sample(range(B), max(2, B // 6)))
    load_p = [rng.uniform(0.1, 0.6) for _ in range(B)]
    load_q = [0.3 * p for p in load_p]
    m = pe.ConcreteModel()
    m.vm = pe.Var(range(B), bounds=(0.9, 1.1), initialize=1.0)
    m.va = pe.Var(range(B), initialize=0.0)
    m.pg = pe.Var(gens, bounds=(0.0, 10.0), initialize=1.0)
    m.qg = pe.Var(gens, bounds=(-5.0, 5.0), initialize=0.0)
    m.va[0].fix(0.0)
    cost = {g: (rng.uniform(0.5, 2.0), rng.uniform(5, 20)) for g in gens}
    m.obj = pe.Objective(expr=sum(c2 * m.pg[g] ** 2 + c1 * m.pg[g] for g, (c2, c1) in cost.items()))
    adj = {k: [] for k in range(B)}
    for a, b, g, bb in lines:
        adj[a].append((b, g, bb))
        adj[b].append((a, g, bb))
    gset = set(gens)

    def p_bal(m, k):
        flow = sum(m.vm[k] * m.vm[o] * (g * pe.cos(m.va[k] - m.va[o]) + bb * pe.sin(m.va[k] - m.va[o]))
                   - g * m.vm[k] ** 2 for o, g, bb in adj[k])
        return (m.pg[k] if k in gset else 0.0) - load_p[k] == -flow

    def q_bal(m, k):
        flow = sum(m.vm[k] * m.vm[o] * (g * pe.sin(m.va[k] - m.va[o]) - bb * pe.cos(m.va[k] - m.va[o]))
                   + bb * m.vm[k] ** 2 for o, g, bb in adj[k])
        return (m.qg[k] if k in gset else 0.0) - load_q[k] == -flow

    m.pb = pe.Constraint(range(B), rule=p_bal)
    m.qb = pe.Constraint(range(B), rule=q_bal)
    return m


def portfolio_qp(n: int, k: int = 40, seed: int = 3):
    """Factor-model Markowitz QP: dense factor rows couple every asset."""
    import pyomo.environ as pe
    rng = random.Random(seed)
    m = pe.ConcreteModel()
    m.x = pe.Var(range(n), bounds=(0.0, 0.05), initialize=1.0 / n)
    m.f = pe.Var(range(k), initialize=0.0)
    d = [rng.uniform(0.01, 0.1) for _ in range(n)]
    mu = [rng.uniform(0.0, 0.2) for _ in range(n)]
    F = [[rng.gauss(0, 0.3) if rng.random() < 0.3 else 0.0 for _ in range(n)] for _ in range(k)]
    m.obj = pe.Objective(expr=sum(d[i] * m.x[i] ** 2 for i in range(n)) + sum(m.f[j] ** 2 for j in range(k))
                         - 0.5 * sum(mu[i] * m.x[i] for i in range(n)))
    m.fac = pe.Constraint(range(k), rule=lambda m, j: m.f[j] == sum(F[j][i] * m.x[i] for i in range(n) if F[j][i] != 0.0))
    m.bud = pe.Constraint(expr=sum(m.x[i] for i in range(n)) == 1.0)
    return m


# name -> (builder, base size). --scale multiplies the base size.
MODELS = {
    "pde_control_120": (pde_control, 120),
    "pde_control_250": (pde_control, 250),
    "robot_arm_8k": (robot_arm, 8000),
    "obstacle_surface_200": (obstacle_surface, 200),
    "acopf_grid_80": (acopf_grid, 80),
    "acopf_grid_160": (acopf_grid, 160),
    "portfolio_20k": (portfolio_qp, 20000),
}
# Models run by capture/e2e when --models is empty. Hard models stay out of it.
DEFAULT_MODELS = list(MODELS)
# Hard nonconvex corpus (PGLib-OPF via Egret, ...): arg may be a case-name string.
# Addressable by name via --models only.
MODELS.update(hard_models.MODELS)


HARD_MAX_ITER = 3000  # hard models need far more iterations than the original 7 (glider ~800)


def selected_models(models_arg: str, suite: str = "") -> list[str]:
    """Model names for capture/e2e: --suite hard (every hard model), the --models
    list, or DEFAULT_MODELS when both are empty."""
    if suite == "hard":
        return list(hard_models.MODELS)
    if suite:
        raise SystemExit(f"unknown suite {suite!r} (only 'hard')")
    return models_arg.split(",") if models_arg else list(DEFAULT_MODELS)


def max_iter_for(name: str) -> int:
    return HARD_MAX_ITER if name in hard_models.MODELS else 200


# ------------------------------------------------------------------ ipopt --
def ipopt_env(threads: int) -> dict:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(BUILD_REL), str(COINHSL_BIN), str(IPOPT_EXE.parent), env.get("PATH", "")])
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = str(threads)
    return env


def solver_options(solver: str) -> list[str]:
    if solver in ("smf", "smf_abort", "smf_reanalyse", "smf_fallback",
                  "smf_fallback_zero"):
        return ["linear_solver ma97", f"hsllib {BUILD_REL / 'smf_ma97.dll'}"]
    if solver == "smf_noscale":
        # IPOPT's MA97 interface defaults to MC64 scaling, which the plugin
        # maps to smf's matching scaling on every factorization.
        return ["linear_solver ma97", f"hsllib {BUILD_REL / 'smf_ma97.dll'}", "ma97_scaling none"]
    if solver == "ma27":
        return ["linear_solver ma27", f"hsllib {COINHSL_BIN / 'libcoinhsl.dll'}"]
    if solver == "mumps":
        return ["linear_solver mumps"]
    if solver in ("ma57", "ma97", "ma86"):
        lib = os.environ.get("SMF_HSL_LIBRARY")
        if not lib:
            raise SystemExit(f"{solver}: set SMF_HSL_LIBRARY to a licensed HSL library")
        return [f"linear_solver {solver}", f"hsllib {lib}"]
    raise SystemExit(f"unknown solver {solver}")


def solver_env(solver: str) -> dict:
    """Extra environment variables per solver id (smf* always traces factorizations).

    smf_abort = smf with the IPM early abort on wrong inertia (SMF_IPM_ABORT=1).
    smf_reanalyse = smf with re-analysis from the realized pivot order (SMF_IPM_REANALYSE=1).
    smf_fallback = smf with fallback static pivoting, static pivots counted by their sign
    (SMF_IPM_STATIC=fallback).
    smf_fallback_zero = smf_fallback with static pivots counted as zero eigenvalues
    (SMF_IPM_STATIC_ZERO=1).
    """
    env = {}
    if solver.startswith("smf"):
        env["SMF_PLUGIN_STATS"] = "2"
    if solver == "smf_abort":
        env["SMF_IPM_ABORT"] = "1"
    if solver == "smf_reanalyse":
        env["SMF_IPM_REANALYSE"] = "1"
    if solver in ("smf_fallback", "smf_fallback_zero"):
        env["SMF_IPM_STATIC"] = "fallback"
    if solver == "smf_fallback_zero":
        env["SMF_IPM_STATIC_ZERO"] = "1"
    return env


def run_ipopt(nl: pathlib.Path, solver: str, rundir: pathlib.Path, threads: int,
              extra_env: dict | None = None, max_iter: int = 200, timeout: float = 900.0) -> dict:
    rundir.mkdir(parents=True, exist_ok=True)
    opts = solver_options(solver) + [f"max_iter {max_iter}", "print_level 5", "print_timing_statistics yes"]
    (rundir / "ipopt.opt").write_text("\n".join(opts) + "\n", encoding="utf-8")
    env = ipopt_env(threads)
    env.update(solver_env(solver))
    if extra_env:
        env.update(extra_env)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([str(IPOPT_EXE), str(nl)], cwd=rundir, env=env, capture_output=True, text=True,
                              timeout=timeout)
        out, err = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = f"\nTIMEOUT after {timeout:.0f}s\n"
        out += "\nEXIT: Timeout (killed by ipopt_kkt.py)\n"
    wall = time.perf_counter() - t0
    (rundir / "ipopt.log").write_text(out + err, encoding="utf-8")

    def grab(pattern, cast=float):
        mm = re.search(pattern, out)
        return cast(mm.group(1)) if mm else None

    trace = parse_trace(out + err) if solver.startswith("smf") else {
        k: None for k in ("factorizations", "inertia_corrections", "delays_total", "static_total", "factor_s",
                          "reanalyses_tried", "reanalyses_adopted", "reanalyses_reverted", "reanalyse_s",
                          "ipm_sessions", "ipm_factorizations", "ipm_accepted", "ipm_aborted", "flops_avoided",
                          "factorizations_reanalysed", "delays_reanalysed")}
    return {
        **trace,
        "solver": solver,
        "threads": threads,
        "status": (re.search(r"EXIT: (.*)", out).group(1).strip() if "EXIT:" in out else "no EXIT line"),
        "iterations": grab(r"Number of Iterations\.+:\s+(\d+)", int),
        "objective": grab(r"Objective\.+:\s+\S+\s+(\S+)"),
        "ipopt_wall_s": wall,
        "linear_system_factorization_s": grab(r"LinearSystemFactorization\.+:\s+(\S+)"),
        "linear_system_backsolve_s": grab(r"LinearSystemBackSolve\.+:\s+(\S+)"),
        "linear_system_symbolic_s": grab(r"LinearSystemSymbolicFactorization\.+:\s+(\S+)"),
        "overall_algorithm_s": grab(r"OverallAlgorithm\.+:\s+(\S+)"),
        "n_vars": grab(r"Total number of variables\.+:\s+(\d+)", int),
        "n_eq": grab(r"Total number of equality constraints\.+:\s+(\d+)", int),
    }


def manifest_fields(status: str, iterations, dumped: int, kept: int, traced_total=None) -> dict:
    """Run-dependent manifest fields of a captured case (pure; no IPOPT needed).

    `factorizations` is the true number of factorizations IPOPT requested in the
    capture run (the plugin trace count); `dumped` is what SMF_KKT_DUMP_MAX let the
    plugin write and `kept` what is stored in the .vals file. When the trace is
    unavailable the dumped count is used and `factorizations_source` says so.
    """
    total = traced_total if traced_total else dumped
    truncated = kept < total
    optimal = status.startswith("Optimal Solution Found") or status.startswith("Solved To Acceptable")
    note = f" (stored {kept} of {total} factorizations)" if truncated else f" ({total} factorizations)"
    return {
        "source": f"IPOPT {status} in {iterations} iterations{note}",
        "factorizations": total,
        "factorizations_source": "plugin_trace" if traced_total else "dump_count",
        "dumped_factorizations": dumped,
        "kept_factorizations": kept,
        "truncated": truncated,
        "ipopt_status": status,
        "ipopt_iterations": iterations,
        "expected_outcome": "Success" if optimal else "MaxIterations" if "Maximum Number of Iterations" in status
        else "Failure",
    }


def build_nl(name: str, scale: float) -> pathlib.Path:
    builder, size = MODELS[name]
    nl = WORK / "nl" / f"{name}.nl"
    if nl.exists():
        return nl
    nl.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model = builder(size if isinstance(size, str) else max(4, int(size * scale)))
    model.write(str(nl), format="nl")
    print(f"  built {name} in {time.perf_counter() - t0:.1f}s", flush=True)
    return nl


def capture(args) -> int:
    names = selected_models(args.models, args.suite)
    hard = [n in hard_models.MODELS for n in names]
    if any(hard) and not all(hard):
        raise SystemExit("capture: do not mix hard and original models in one run")
    is_hard = bool(names) and all(hard)
    suite_id = "ipopt_hard" if is_hard else "ipopt_kkt"
    fixtures = REPO / "benchmarks" / "fixtures" / suite_id
    manifest_out = pathlib.Path(args.manifest_out) if args.manifest_out else \
        REPO / "benchmarks" / "manifests" / f"{suite_id}.json"
    cases = []
    for name in names:
        print(f"{name}:", flush=True)
        nl = build_nl(name, args.scale)
        dump = WORK / "dump" / name
        if dump.exists():
            shutil.rmtree(dump)
        dump.mkdir(parents=True)
        res = run_ipopt(nl, "smf", WORK / "capture" / name, 1,
                        {"SMF_KKT_DUMP_DIR": str(dump), "SMF_KKT_DUMP_MAX": "1000"},
                        max_iter=max_iter_for(name), timeout=args.timeout)
        print(f"  {res['status']}  iters={res['iterations']}", flush=True)
        mtx, vals = dump / "kkt_s0.mtx", dump / "kkt_s0.vals"
        if not mtx.exists():
            print("  no KKT captured", flush=True)
            continue
        dest = fixtures / name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mtx, dest / f"{name}.mtx")
        with mtx.open() as f:
            line = f.readline()
            while line.startswith("%"):
                line = f.readline()
            n, _, nnz = (int(t) for t in line.split())
        nfact = vals.stat().st_size // (8 * nnz)
        # --keep 5 (default): the harness replays 5 refactorizations; keep 5
        # evenly spaced over the run (first to last IPOPT iteration).
        # --keep all: every factorization the plugin dumped (the full sequence).
        if args.keep == "all":
            keep = list(range(nfact))
        else:
            keep = sorted({round(i * (nfact - 1) / 4) for i in range(5)}) if nfact > 5 else list(range(nfact))
        with vals.open("rb") as src, (dest / f"{name}.mtx.vals").open("wb") as dst:
            for i in keep:
                src.seek(i * 8 * nnz)
                dst.write(src.read(8 * nnz))
        shutil.rmtree(dump)
        cases.append({
            "matrix_id": name,
            "sequence_id": name,
            "family_id": suite_id,
            "mtx_path": (dest / f"{name}.mtx").relative_to(REPO).as_posix(),
            "seq_values_path": (dest / f"{name}.mtx.vals").relative_to(REPO).as_posix(),
            "n": n,
            "nnz_lower": nnz,
            **manifest_fields(res["status"], res["iterations"], nfact, len(keep), res.get("factorizations")),
            "matrix_type": "indefinite",
            "split": hard_models.split_of(name) if is_hard else "test",
            "mode": "warm_sequence",
            "track": "ipopt",
        })
        print(f"  KKT n={n} nnz={nnz} dumped={nfact} kept={len(keep)} traced={res.get('factorizations')}", flush=True)
    cases.sort(key=lambda c: c["n"])
    manifest = {
        "schema_version": 1,
        "suite": suite_id,
        "solvers": ["smf", "ma27", "mumps", "mkl_pardiso", "ma57", "ma97", "ssids"],
        "required_baselines": [],
        "cases": cases,
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_out} ({len(cases)} cases)")
    return 0


def e2e(args) -> int:
    names = selected_models(args.models, args.suite)
    solvers = args.solvers.split(",")
    threads = [int(t) for t in args.threads.split(",")]
    results = []
    for name in names:
        nl = build_nl(name, args.scale)
        for t in threads:
            for s in solvers:
                rundir = WORK / "e2e" / f"{name}_{s}_t{t}"
                res = run_ipopt(nl, s, rundir, t, timeout=args.timeout, max_iter=max_iter_for(name))
                res["model"] = name
                res["log"] = (rundir / "ipopt.log").relative_to(REPO).as_posix()
                results.append(res)
                print(f"{name:22s} t={t:2d} {s:9s} {res['status'][:28]:28s} iters={res['iterations']} "
                      f"wall={res['ipopt_wall_s']:.2f}s fact={res['linear_system_factorization_s']} "
                      f"solve={res['linear_system_backsolve_s']}", flush=True)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["capture", "e2e"])
    ap.add_argument("--models", default="")
    ap.add_argument("--suite", default="", help="'hard' selects every hard_models.MODELS name")
    ap.add_argument("--keep", choices=["5", "all"], default="5",
                    help="capture: 5 evenly spaced factorizations (default) or the full sequence")
    ap.add_argument("--manifest-out", default="",
                    help="capture: manifest path (default manifests/ipopt_kkt.json; ipopt_hard.json for hard models)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--solvers", default="mumps,ma27,smf")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--timeout", type=float, default=900.0, help="per IPOPT run, seconds")
    ap.add_argument("--output", default=str(REPO / "benchmarks" / "results" / "ipopt_e2e.json"))
    args = ap.parse_args()
    return capture(args) if args.command == "capture" else e2e(args)


if __name__ == "__main__":
    sys.exit(main())
