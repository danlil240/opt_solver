"""Diagnose one solve: IPOPT's regularisation and restoration counts from its
iteration log, its timing statistics, and (smf) the plugin's call counts.

  python diag_one.py <problem module> <N> <seed> <solver> [threads]

Prints one JSON line prefixed DIAG. For smf, run it with SMF_PLUGIN_STATS=1:
the plugin's totals go to stderr at exit, and diag_suite.py merges them.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import tempfile
import time

import casadi as ca
import numpy as np

import common


def iteration_stats(text: str) -> dict:
    """Counts over IPOPT's iteration table: iterations with a Hessian
    regularisation (lg(rg) not '-'), restoration iterations ('r' suffix),
    the largest lg(rg), and line-search trial counts."""
    reg = rest = n = 0
    max_rg = None
    ls_total = 0
    for line in text.splitlines():
        # iter, objective, inf_pr, inf_du, lg(mu), ||d||, lg(rg), alpha_du, alpha_pr, ls
        m = re.match(r"^\s*(\d+)(r?)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*$", line)
        if not m:
            continue
        n += 1
        if m.group(2) == "r":
            rest += 1
        rg = m.group(8)
        if rg != "-":
            reg += 1
            try:
                max_rg = max(float(rg), max_rg if max_rg is not None else -99.0)
            except ValueError:
                pass
        ls_total += int(m.group(11))
    return {"iter_lines": n, "regularized_iters": reg, "restoration_iters": rest, "max_lg_rg": max_rg,
            "ls_trials": ls_total}


def main() -> None:
    mod, N, seed, solver = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    common.ensure_thread_env(threads)
    prob = importlib.import_module(mod).build(N, seed)
    fd, out_file = tempfile.mkstemp(prefix="ipopt_diag_", suffix=".txt")
    os.close(fd)
    opts = common.ipopt_options(solver, out_file)
    opts["ipopt.print_level"] = 0
    if os.environ.get("DIAG_DETAILED"):
        opts["ipopt.file_print_level"] = 6  # prints the regularisation (delta_w, delta_c) per iteration
    S = ca.nlpsol("S", "ipopt", {"x": prob.x, "f": prob.f, "g": prob.g}, opts)
    t0 = time.perf_counter()
    r = S(x0=prob.x0, lbx=prob.lbx, ubx=prob.ubx, lbg=prob.lbg, ubg=prob.ubg)
    wall = time.perf_counter() - t0
    st = S.stats()
    del S
    import gc
    gc.collect()
    text = open(out_file, errors="replace").read()
    os.remove(out_file)
    rec = {"problem": prob.name, "N": N, "seed": seed, "solver": solver, "threads": threads,
           "status": st.get("return_status"), "iter": st.get("iter_count"), "obj": float(r["f"]),
           "wall_s": wall, **{f"ipopt_{k}": v for k, v in common.parse_ipopt_output(text).items()},
           **iteration_stats(text),
           "delta_c_pos_iters": sum(1 for v in re.findall(r"delta_c=([0-9.eE+-]+)", text) if float(v) > 0),
           "trial_factorizations": sum(int(v) for v in re.findall(r"Number of trial factorizations performed: (\d+)", text))}
    print("DIAG " + json.dumps(rec, default=float), flush=True)


if __name__ == "__main__":
    main()
