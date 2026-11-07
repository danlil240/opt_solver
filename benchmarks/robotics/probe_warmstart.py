"""Optimizer-headroom probe: where IPOPT + smf spends a solve, how many
factorizations it needs per iteration, and what a warm-started MPC replan
costs. Measurement only: nothing in the solver changes.

  python probe_warmstart.py suite --out <file.jsonl> --cells p_quad_forest:100,p_arm7:200 \
      --seeds 0,1,2 --solvers smf [--no-replan] [--env ROBO_STAGE_JIT=1]
  python probe_warmstart.py cell <problem module> <N> <seed> <solver> [--no-replan]

Each cell runs in its own process (runner_env(1): one thread,
SMF_SLACK_ELIM=auto; SMF_PLUGIN_STATS=2 for smf) and does three solves of
the same NLP (one CasADi build, a fresh nlpsol per solve):

  base         the problem as built, from its own initial guess (cold)
  replan_cold  the initial state perturbed, from the problem's initial guess
  replan_warm  the same perturbed problem, from base's primal and dual
               solution (warm_start_init_point, mu_init 1e-4, bound
               push/frac 1e-6)

The perturbation is an MPC replan's state update: every component of the
initial state (the variables the problem pins with lbx == ubx at node 0)
moves by u * 1.5% of that component's range over base's trajectory, u
uniform in [-1, 1] from a generator seeded by the seed. car_track is a
periodic flying lap with no initial state: it gets the base solve only.

Per solve: wall time, CasADi's per-function evaluation times and call
counts, IPOPT's timing statistics (symbolic, numeric factorization, back
solves), the iteration-table counts (regularized iterations, restoration,
line-search trials), and for smf the plugin's per-factorization trace,
split by solve with markers on stderr: factorizations, inertia-correction
refactorizations (corr=1), aborted factorizations and their summed time.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
BELOW_NORMAL = 0x00004000
PERTURB = 0.015
FUNCS = ("f", "g", "grad_f", "jac_g", "hess_l")

# Node stride of the decision vector and the per-node state size, to find a
# state component's trajectory: (stride, nx, state blocks per node and their
# offsets within the node). Checked against the pinned initial-state count.
LAYOUT = {
    "quad_forest": (16, 12, (0,)),
    "drone_swap": (48, 12, (0, 16, 32)),
    "truck_trailer": (8, 6, (0,)),
    "arm7": (21, 14, (0,)),
}


def marker(name: str) -> None:
    # os.write: unbuffered, so it lands between the plugin's own stderr lines.
    os.write(2, f"@@PHASE {name}\n".encode())


def solve_once(prob, solver: str, lbx, ubx, x0, extra: dict | None = None, lam_x0=None, lam_g0=None) -> tuple[dict, dict]:
    import casadi as ca
    import numpy as np
    import common
    import diag_one
    fd, out_file = tempfile.mkstemp(prefix="ipopt_probe_", suffix=".txt")
    os.close(fd)
    opts = common.nlp_options(solver, out_file, extra)
    t_b = time.perf_counter()
    S = ca.nlpsol("S", "ipopt", {"x": prob.x, "f": prob.f, "g": prob.g}, opts)
    build_s = time.perf_counter() - t_b
    args = dict(x0=x0, lbx=lbx, ubx=ubx, lbg=prob.lbg, ubg=prob.ubg)
    if lam_x0 is not None:
        args.update(lam_x0=lam_x0, lam_g0=lam_g0)
    t0 = time.perf_counter()
    r = S(**args)
    wall = time.perf_counter() - t0
    st = S.stats()
    del S
    import gc
    gc.collect()
    text = pathlib.Path(out_file).read_text(errors="replace")
    os.remove(out_file)
    info = common.parse_ipopt_output(text)
    xs = np.array(r["x"]).ravel()
    gval = np.array(r["g"]).ravel()
    viol = float(max(0.0, np.max(np.r_[prob.lbg - gval, gval - prob.ubg, lbx - xs, xs - ubx, 0.0])))
    rec = {"status": st.get("return_status"), "success": bool(st.get("success")), "iter": st.get("iter_count"),
           "obj": float(r["f"]), "max_viol": viol, "wall_s": wall, "build_s": build_s,
           **{f"t_{k}": st.get(f"t_wall_nlp_{k}") or 0.0 for k in FUNCS},
           **{f"n_{k}": st.get(f"n_call_nlp_{k}") or 0 for k in FUNCS},
           **{f"ipopt_{k}": v for k, v in info.items()}, **diag_one.iteration_stats(text)}
    sol = {"x": xs, "lam_x": np.array(r["lam_x"]).ravel(), "lam_g": np.array(r["lam_g"]).ravel()}
    return rec, sol


def perturbed_bounds(prob, base_x, seed: int):
    """lbx/ubx with the initial state moved; None when the problem has no
    pinned initial state this probe knows how to find."""
    import numpy as np
    if prob.name not in LAYOUT:
        return None, None
    stride, nx, blocks = LAYOUT[prob.name]
    N = prob.meta["N"]
    fixed = np.nonzero(prob.lbx == prob.ubx)[0]
    init = fixed[fixed < len(prob.lbx) // 2]
    assert len(init) == nx * len(blocks), (prob.name, len(init))
    rng = np.random.default_rng(1000 + seed)
    lbx, ubx = prob.lbx.copy(), prob.ubx.copy()
    deltas = []
    for i in init:
        traj = base_x[i + stride * np.arange(N)]
        d = PERTURB * float(np.ptp(traj)) * rng.uniform(-1.0, 1.0)
        lbx[i] += d
        ubx[i] += d
        deltas.append(d)
    return (lbx, ubx), deltas


WARM = {"ipopt.warm_start_init_point": "yes", "ipopt.mu_init": 1e-4,
        "ipopt.warm_start_bound_push": 1e-6, "ipopt.warm_start_bound_frac": 1e-6,
        "ipopt.warm_start_slack_bound_push": 1e-6, "ipopt.warm_start_slack_bound_frac": 1e-6,
        "ipopt.warm_start_mult_bound_push": 1e-6}


def run_cell(mod: str, N: int, seed: int, solver: str, replan: bool) -> None:
    import common
    common.ensure_runner_env(1)
    t_b = time.perf_counter()
    prob = importlib.import_module(mod).build(N, seed)
    problem_build_s = time.perf_counter() - t_b
    head = {"problem": prob.name, "N": N, "seed": seed, "solver": solver,
            "stage_jit": common.stage_jit(), "slack_elim": os.environ.get("SMF_SLACK_ELIM"),
            "problem_build_s": problem_build_s}
    marker("base")
    rec, sol = solve_once(prob, solver, prob.lbx, prob.ubx, prob.x0)
    rec["check"] = prob.check(sol["x"])
    print("PROBE " + json.dumps({**head, "phase": "base", **rec}, default=float), flush=True)
    if not replan:
        return
    bounds, deltas = perturbed_bounds(prob, sol["x"], seed)
    if bounds is None:
        print("PROBE " + json.dumps({**head, "phase": "replan_skipped",
                                     "reason": "no pinned initial state (periodic lap)"}), flush=True)
        return
    lbx, ubx = bounds
    marker("replan_cold")
    rc, sc = solve_once(prob, solver, lbx, ubx, prob.x0)
    rc["check"] = prob.check(sc["x"])
    print("PROBE " + json.dumps({**head, "phase": "replan_cold", "perturb": deltas, **rc}, default=float), flush=True)
    marker("replan_warm")
    rw, sw = solve_once(prob, solver, lbx, ubx, sol["x"], WARM, sol["lam_x"], sol["lam_g"])
    rw["check"] = prob.check(sw["x"])
    rw["obj_vs_cold"] = rw["obj"] - rc["obj"]
    print("PROBE " + json.dumps({**head, "phase": "replan_warm", **rw}, default=float), flush=True)


FACTOR = re.compile(r"\[smf_ma97\] factor (\d+): corr=(\d+) .*? t=([\d.eE+-]+) .*?abort=(\d+)")


def plugin_by_phase(stderr: str) -> dict:
    out: dict[str, dict] = {}
    phase = None
    for line in stderr.splitlines():
        if line.startswith("@@PHASE "):
            phase = line[8:].strip()
            out[phase] = {"plugin_factors": 0, "plugin_corr": 0, "plugin_aborts": 0, "plugin_factor_s": 0.0}
            continue
        m = FACTOR.search(line)
        if m and phase:
            p = out[phase]
            p["plugin_factors"] += 1
            p["plugin_corr"] += int(m.group(2))
            p["plugin_aborts"] += int(m.group(4))
            p["plugin_factor_s"] += float(m.group(3))
    return out


def suite(a) -> None:
    sys.path.insert(0, str(HERE))
    import common
    flags = BELOW_NORMAL if os.name == "nt" else 0
    extra_env = dict(kv.split("=", 1) for kv in a.env)
    with open(a.out, "a", encoding="utf-8") as fh:
        for cell in a.cells.split(","):
            mod, n = cell.split(":")
            for seed in a.seeds.split(","):
                for sv in a.solvers.split(","):
                    env = common.runner_env(1)
                    env.update(extra_env)
                    if sv == "smf":
                        env["SMF_PLUGIN_STATS"] = "2"
                    cmd = [sys.executable, str(HERE / "probe_warmstart.py"), "cell", mod, n, seed, sv]
                    if a.no_replan:
                        cmd.append("--no-replan")
                    t0 = time.perf_counter()
                    try:
                        cp = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True,
                                            timeout=a.timeout, creationflags=flags)
                        stdout, stderr = cp.stdout, cp.stderr
                    except subprocess.TimeoutExpired as e:
                        stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                        stderr = "timeout"
                    plug = plugin_by_phase(stderr) if sv == "smf" else {}
                    recs = [json.loads(l[6:]) for l in stdout.splitlines() if l.startswith("PROBE ")]
                    if not recs:
                        recs = [{"cell": cell, "seed": int(seed), "solver": sv, "phase": "harness_error",
                                 "stderr": stderr[-2000:]}]
                    for r in recs:
                        r.update(plug.get(r.get("phase"), {}))
                        r.update({f"env_{k}": v for k, v in extra_env.items()})
                        fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    for r in recs:
                        print({k: r.get(k) for k in ("problem", "N", "seed", "solver", "phase", "status", "iter",
                                                     "wall_s", "plugin_factors", "plugin_corr")},
                              f"{time.perf_counter() - t0:.0f}s", flush=True)


def split(r: dict) -> dict:
    """A solve's time split and its ideal-optimizer time.

    eval: CasADi's function evaluations (all five functions);
    lin: IPOPT's linear-solver time (symbolic + numeric factorization + back solves);
    other: everything else in the wall time (IPOPT's own work and the interface).
    ideal: eval + symbolic + back solves + numeric factorization scaled to one
    factorization per iteration (count ratio; an inertia-correction refactor
    is priced like a first one, so it slightly overstates the saving), with
    `other` removed. None for a solver without a factorization count."""
    ev = sum(r.get(f"t_{k}") or 0.0 for k in FUNCS)
    sym, fac, bs = (r.get(f"ipopt_{k}") or 0.0 for k in ("symbolic", "factor", "backsolve"))
    lin = sym + fac + bs
    wall = r["wall_s"]
    out = {"eval": ev, "lin": lin, "other": max(0.0, wall - ev - lin), "wall": wall,
           "iters": r.get("iter_lines") or (r.get("iter") or 0) + 1}
    nf = r.get("plugin_factors")
    if nf:
        out["fpi"] = nf / out["iters"]
        out["ideal"] = ev + sym + bs + fac * min(1.0, out["iters"] / nf)
    return out


def median(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def summary(paths: list[str]) -> None:
    recs = [json.loads(l) for p in paths for l in open(p, encoding="utf-8") if l.strip()]
    key = lambda r: (r["problem"], r["N"], r.get("env_ROBO_STAGE_JIT", "0"), r["solver"])
    cells: dict = {}
    for r in recs:
        if "problem" not in r:
            continue
        cells.setdefault(key(r), {}).setdefault(r["seed"], {})[r["phase"]] = r
    print("| problem | N | solver | seeds | eval | lin | other | f/it | reg it | corr | iters base | replan iters cold / warm | replan wall cold / warm (s) | one-shot bound | replan bound vs cold / vs warm |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|")
    for (prob, N, jit, sv), seeds in sorted(cells.items(), key=lambda kv: (kv[0][3] != "smf", kv[0][2], kv[0][0], kv[0][1])):
        rows = []
        for s, ph in seeds.items():
            b = ph.get("base")
            if not b or not b.get("wall_s"):
                continue
            sb = split(b)
            row = {"eval": sb["eval"] / sb["wall"], "lin": sb["lin"] / sb["wall"], "other": sb["other"] / sb["wall"],
                   "fpi": sb.get("fpi"), "reg": b.get("regularized_iters"), "corr": b.get("plugin_corr"),
                   "it": sb["iters"], "b1": sb["wall"] / sb["ideal"] if "ideal" in sb else None}
            c, w = ph.get("replan_cold"), ph.get("replan_warm")
            if c and w:
                sc, sw = split(c), split(w)
                use = sw if (w.get("success") and sw["iters"] <= sc["iters"]) else sc
                row.update(ic=sc["iters"], iw=sw["iters"], wc=sc["wall"], ww=sw["wall"], wsucc=w.get("success"))
                if "ideal" in use:
                    row.update(rc=sc["wall"] / use["ideal"], rw=sw["wall"] / use["ideal"])
            rows.append(row)
        if not rows:
            continue
        m = {k: median([r.get(k) for r in rows]) for k in ("eval", "lin", "other", "fpi", "reg", "corr", "it", "b1",
                                                             "ic", "iw", "wc", "ww", "rc", "rw")}
        f = lambda v, fmt: "-" if v is None else fmt.format(v)
        name = prob + (" (stage JIT)" if jit == "1" else "")
        fails = sum(1 for r in rows if r.get("wsucc") is False)
        print(f"| {name} | {N} | {sv} | {len(rows)} | {f(m['eval'], '{:.0%}')} | {f(m['lin'], '{:.0%}')} | "
              f"{f(m['other'], '{:.0%}')} | {f(m['fpi'], '{:.2f}')} | {f(m['reg'], '{:.0f}')} | {f(m['corr'], '{:.0f}')} | "
              f"{f(m['it'], '{:.0f}')} | {f(m['ic'], '{:.0f}')} / {f(m['iw'], '{:.0f}')}{' (warm failed x%d)' % fails if fails else ''} | "
              f"{f(m['wc'], '{:.3f}')} / {f(m['ww'], '{:.3f}')} | {f(m['b1'], '{:.2f}x')} | "
              f"{f(m['rc'], '{:.1f}x')} / {f(m['rw'], '{:.2f}x')} |")
    # Per-iteration costs and the function-evaluation split of the base solve.
    print()
    print("| problem | N | solver | ms/iter wall | eval | lin | other | bound: no overhead only | bound: 1 factor/iter only | f | g | grad_f | jac_g | hess_l |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (prob, N, jit, sv), seeds in sorted(cells.items(), key=lambda kv: (kv[0][3] != "smf", kv[0][2], kv[0][0], kv[0][1])):
        rows = []
        for ph in seeds.values():
            b = ph.get("base")
            if not b or not b.get("wall_s"):
                continue
            sb = split(b)
            it = sb["iters"]
            ev = sb["eval"] or 1.0
            rows.append({"w": 1e3 * sb["wall"] / it, "e": 1e3 * sb["eval"] / it, "l": 1e3 * sb["lin"] / it,
                         "o": 1e3 * sb["other"] / it,
                         "bo": sb["wall"] / (sb["eval"] + sb["lin"]),
                         "bf": sb["wall"] / (sb["ideal"] + sb["other"]) if "ideal" in sb else None,
                         **{k: (b.get(f"t_{k}") or 0.0) / ev for k in FUNCS}})
        if not rows:
            continue
        m = {k: median([r.get(k) for r in rows]) for k in rows[0]}
        f = lambda v, fmt: "-" if v is None else fmt.format(v)
        name = prob + (" (stage JIT)" if jit == "1" else "")
        print(f"| {name} | {N} | {sv} | {m['w']:.2f} | {m['e']:.2f} | {m['l']:.2f} | {m['o']:.2f} | {m['bo']:.2f}x | "
              f"{f(m['bf'], '{:.2f}x')} | " + " | ".join(f"{m[k]:.0%}" for k in FUNCS) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("mod"); c.add_argument("N", type=int); c.add_argument("seed", type=int); c.add_argument("solver")
    c.add_argument("--no-replan", action="store_true")
    s = sub.add_parser("suite")
    s.add_argument("--out", required=True); s.add_argument("--cells", required=True)
    s.add_argument("--seeds", default="0"); s.add_argument("--solvers", default="smf")
    s.add_argument("--no-replan", action="store_true")
    s.add_argument("--env", action="append", default=[], help="VAR=value for every cell (e.g. ROBO_STAGE_JIT=1)")
    s.add_argument("--timeout", type=int, default=3600)
    m = sub.add_parser("summary", help="markdown table (medians over seeds) from suite records")
    m.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "summary":
        summary(a.files)
    elif a.cmd == "cell":
        run_cell(a.mod, a.N, a.seed, a.solver, not a.no_replan)
    else:
        suite(a)


if __name__ == "__main__":
    main()
