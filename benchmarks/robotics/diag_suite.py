"""Diagnostic runs (diag_one.py) over chosen cells; for smf, the plugin's
per-factorization trace (SMF_PLUGIN_STATS=2) is aggregated into the record:
factorizations, static pivots, delays, inertia-unknown (flag 7) reports,
aborted factorizations, zero pivots, and the solve-call counts.

  python diag_suite.py --out diag.jsonl --cells p_truck_trailer:100,p_drone_swap:100 --solvers ma27,ma57,smf
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

import common

HERE = pathlib.Path(__file__).resolve().parent
BELOW_NORMAL = 0x00004000


def plugin_summary(stderr: str) -> dict:
    fac = re.findall(r"\[smf_ma97\] factor (\d+): corr=(\d+) delays=(\d+) static=(\d+) pos=(\d+) neg=(\d+) zero=(\d+).*?abort=(\d+)", stderr)
    out = {"plugin_factors": len(fac),
           "plugin_static_total": sum(int(f[3]) for f in fac),
           "plugin_factors_with_static": sum(1 for f in fac if int(f[3]) > 0),
           "plugin_delays_total": sum(int(f[2]) for f in fac),
           "plugin_zero_factors": sum(1 for f in fac if int(f[6]) > 0),
           "plugin_aborts": sum(1 for f in fac if int(f[7]) > 0),
           "plugin_corrections": sum(1 for f in fac if int(f[1]) > 0),
           "plugin_flag7": stderr.count("flag 7")}
    m = re.search(r"factors=(\d+) \(([\d.]+) s\) solves=(\d+) rhs=(\d+) \(([\d.]+) s", stderr)
    if m:
        out.update(plugin_factor_s=float(m.group(2)), plugin_solves=int(m.group(3)), plugin_rhs=int(m.group(4)),
                   plugin_solve_s=float(m.group(5)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--solvers", default="ma27,ma57,ma97,smf")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    flags = BELOW_NORMAL if os.name == "nt" else 0  # creationflags exist only on Windows
    with open(a.out, "a", encoding="utf-8") as fh:
        for cell in a.cells.split(","):
            mod, n = cell.split(":")
            for sv in a.solvers.split(","):
                env = common.thread_env(a.threads)  # set before the child loads anything
                env["DIAG_DETAILED"] = "1"
                if sv == "smf":
                    env["SMF_PLUGIN_STATS"] = "2"
                cp = subprocess.run([sys.executable, str(HERE / "diag_one.py"), mod, n, a.seed, sv, str(a.threads)], cwd=HERE, env=env,
                                    capture_output=True, text=True, timeout=3600, creationflags=flags)
                line = next((l for l in cp.stdout.splitlines() if l.startswith("DIAG ")), None)
                rec = json.loads(line[5:]) if line else {"cell": cell, "solver": sv, "status": "harness_error",
                                                         "stderr": cp.stderr[-1500:]}
                if sv == "smf":
                    rec.update(plugin_summary(cp.stderr))
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                keys = ["problem", "N", "solver", "status", "iter", "obj", "regularized_iters", "max_lg_rg",
                        "delta_c_pos_iters", "trial_factorizations", "restoration_iters", "ipopt_factor", "ipopt_backsolve",
                        "plugin_factors", "plugin_static_total", "plugin_flag7", "plugin_aborts", "plugin_zero_factors",
                        "plugin_solves", "plugin_solve_s"]
                print({k: rec.get(k) for k in keys if k in rec}, flush=True)


if __name__ == "__main__":
    main()
