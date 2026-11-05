"""Run the robotics trajectory-optimization suite: every (problem, N, seed)
with every linear solver, one solve per process, one at a time, at
below-normal priority.

  python run_suite.py --out results.jsonl [--problems p_quad_forest,...]
                      [--sizes 50,100,200,400] [--seeds 0] [--solvers ...]
                      [--threads 1] [--reps 1]

Each line of the output is one run_one.py record (see common.solve). The
records carry IPOPT's timing statistics, CasADi's function-evaluation split,
the KKT size and the problem's own feasibility check.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PROBLEMS = ["p_quad_forest", "p_drone_swap", "p_car_track", "p_truck_trailer", "p_arm7"]
BELOW_NORMAL = 0x00004000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--problems", default=",".join(PROBLEMS))
    ap.add_argument("--sizes", default="50,100,200,400")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--solvers", default="mumps,ma27,ma57,ma97,smf")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800.0)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    flags = BELOW_NORMAL if os.name == "nt" else 0
    jobs = [(p, int(n), int(s), sv, int(t), r)
            for p in a.problems.split(",") for n in a.sizes.split(",") for s in a.seeds.split(",")
            for r in range(a.reps) for t in a.threads.split(",") for sv in a.solvers.split(",")
            if not (sv != "smf" and int(t) > 1 and sv not in ("ma97", "mumps"))]
    with out.open("a", encoding="utf-8") as fh:
        for i, (p, n, s, sv, t, r) in enumerate(jobs):
            t0 = time.perf_counter()
            # The thread counts go into the child's environment block: the
            # mingw-built HSL (OpenMP) and MUMPS (OpenBLAS) read it through
            # their own C runtime, which never sees os.environ changes made
            # inside the Python process.
            env = os.environ.copy()
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "SMF_NUM_THREADS"):
                env[k] = str(t)
            if sv == "smf":
                env["SMF_PLUGIN_STATS"] = "1"
            try:
                cp = subprocess.run([sys.executable, str(HERE / "run_one.py"), p, str(n), str(s), sv, str(t)],
                                    cwd=HERE, capture_output=True, text=True, timeout=a.timeout,
                                    creationflags=flags, env=env)
                line = next((ln for ln in cp.stdout.splitlines() if ln.startswith("RESULT ")), None)
                rec = json.loads(line[7:]) if line else {
                    "problem": p, "N": n, "seed": s, "solver": sv, "threads": t,
                    "status": "harness_error", "stderr": cp.stderr[-2000:]}
            except subprocess.TimeoutExpired:
                rec = {"problem": p, "N": n, "seed": s, "solver": sv, "threads": t, "status": "timeout"}
            rec["rep"] = r
            if sv == "smf" and "cp" in locals():
                m = re.search(r"factors=(\d+) \(([\d.]+) s\) solves=(\d+) rhs=(\d+) \(([\d.]+) s", cp.stderr)
                if m:
                    rec.update(plugin_factors=int(m.group(1)), plugin_factor_s=float(m.group(2)),
                               plugin_solves=int(m.group(3)), plugin_rhs=int(m.group(4)),
                               plugin_solve_s=float(m.group(5)))
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            print(f"[{i + 1}/{len(jobs)}] {p} N={n} seed={s} {sv} {t}T: {rec.get('status')} "
                  f"iter={rec.get('iter')} wall={rec.get('wall_s')} ({time.perf_counter() - t0:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
