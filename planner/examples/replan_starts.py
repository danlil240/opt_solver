"""Which starts fix the slow truck replans? Measurement only.

  python replan_starts.py --solver smf --N 100 --seed-tasks 60 --chains 3 --replans 8

Same setup as truck_replan.py (seeded library, chains of replans after
driving 10% of the plan with a small disturbance), but every replan runs all
of these starts to the end, one per worker (8 workers), and records each:

  hint         previous plan shifted in time, duals, mu_init 1e-4 (the planner's default)
  hint_mu2     same, mu_init 1e-2, bound pushes 1e-3
  hint_mu1     same, mu_init 1e-1, bound pushes 1e-2
  hint_adapt   same as hint with mu_strategy adaptive
  hint_primal  shifted primal only, IPOPT defaults
  lib1, lib2   nearest stored solutions with duals, mu_init 1e-4
  cold         the hand-made guess
The chain continues from the fastest successful start. Every successful
solve is stored, as the planner does.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import specs  # noqa: E402
from smfplan import Planner  # noqa: E402

VARIANTS = {
    "hint": None,
    "hint_mu2": {"ipopt.mu_init": 1e-2, "ipopt.warm_start_bound_push": 1e-3, "ipopt.warm_start_bound_frac": 1e-3,
                 "ipopt.warm_start_slack_bound_push": 1e-3, "ipopt.warm_start_slack_bound_frac": 1e-3,
                 "ipopt.warm_start_mult_bound_push": 1e-3},
    "hint_adapt": {"ipopt.mu_strategy": "adaptive"},
    "hint_mu1": {"ipopt.mu_init": 1e-1, "ipopt.warm_start_bound_push": 1e-2, "ipopt.warm_start_bound_frac": 1e-2,
                 "ipopt.warm_start_slack_bound_push": 1e-2, "ipopt.warm_start_slack_bound_frac": 1e-2,
                 "ipopt.warm_start_mult_bound_push": 1e-2},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="smf")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--seed-tasks", type=int, default=60)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--replans", type=int, default=8)
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--out", default="replan_starts.jsonl")
    a = ap.parse_args()
    opts, env = specs.solver_setup(a.solver)
    rng = np.random.default_rng(a.rng)
    out = open(a.out, "w", encoding="utf-8")
    names = ["hint", "hint_mu2", "hint_mu1", "hint_adapt", "hint_primal", "lib1", "lib2", "cold"]
    t0 = time.perf_counter()
    with Planner(specs.truck_factory(a.N), workers=len(names), k=2, solver_options=opts, env=env) as pl:
        pl.seed([pl.spec.sample(rng) for _ in range(a.seed_tasks)])
        print(f"library {len(pl.library)} after {time.perf_counter() - t0:.0f} s", flush=True)
        stride, j = 8, a.N // 10
        rows = []
        for ch in range(a.chains):
            r = pl.plan(pl.spec.sample(rng))
            prev = r.as_hint()
            for k in range(a.replans):
                o = 1 + j * stride
                state = np.array(prev["x"][o:o + 6])
                state[:2] += rng.uniform(-0.3, 0.3, 2)
                state[2:4] += rng.uniform(-0.03, 0.03, 2)
                state[4] = np.clip(state[4], -3.0, 5.0)
                state[5] = np.clip(state[5], -0.6, 0.6)
                lbx, ubx = pl.spec.bounds(state)
                h = pl.spec.shift_hint(prev, j / a.N)
                starts = []
                for nm in ("hint", "hint_mu2", "hint_mu1", "hint_adapt"):
                    s = {"x0": np.clip(h["x"], lbx, ubx), "lam_x0": h["lam_x"], "lam_g0": h["lam_g"], "tag": nm}
                    if VARIANTS[nm]:
                        s["opts"] = VARIANTS[nm]
                    starts.append(s)
                starts.append({"x0": np.clip(h["x"], lbx, ubx), "tag": "hint_primal"})
                _, idx = pl.library.nearest(pl.spec.features(state), 2)
                for rank, jj in enumerate(idx, 1):
                    e = pl.library.entry(int(jj))
                    starts.append({"x0": np.clip(e["x"], lbx, ubx), "lam_x0": e["lam_x"], "lam_g0": e["lam_g"],
                                   "tag": f"lib{rank}"})
                starts.append({"x0": pl.spec.cold_guess(state), "tag": "cold"})
                _, res, _ = pl.pool.race(lbx, ubx, None, starts, stop_losers=False)
                by = {x["tag"]: x for x in res}
                rec = {"chain": ch, "step": k, "state": state.tolist(),
                       **{f"{n}_{f}": by[n][f] for n in names if n in by for f in ("iter", "wall_s", "success", "obj")}}
                rows.append(rec)
                out.write(json.dumps(rec, default=float) + "\n")
                out.flush()
                ok = [x for x in res if x["success"]]
                if not ok:
                    print(f"{ch}.{k}: every start failed; chain stops", flush=True)
                    break
                best = min(ok, key=lambda x: x["wall_s"])
                for x in ok:
                    pl.library.add(pl.spec.features(state), x["x"], x["lam_x"], x["lam_g"], x["obj"])
                prev = {"x": best["x"], "lam_x": best["lam_x"], "lam_g": best["lam_g"]}
                print(f"{ch}.{k} " + "  ".join(f"{n} {by[n]['iter']:4d}{'' if by[n]['success'] else '!'}"
                                                for n in names if n in by), flush=True)
    out.close()
    print()
    print("| start | solved | median iters | p90 iters | max iters | median s | wins (fastest) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    wins = {n: 0 for n in names}
    for r in rows:
        ok = [n for n in names if r.get(f"{n}_success")]
        if ok:
            wins[min(ok, key=lambda n: r[f"{n}_wall_s"])] += 1
    for n in names:
        it = [r[f"{n}_iter"] for r in rows if r.get(f"{n}_success")]
        ws = [r[f"{n}_wall_s"] for r in rows if r.get(f"{n}_success")]
        if not it:
            continue
        print(f"| {n} | {len(it)}/{len(rows)} | {np.median(it):.0f} | {np.percentile(it, 90):.0f} | {max(it)} | "
              f"{np.median(ws):.2f} | {wins[n]} |")
    # what a 3-wide race of each trio would give (min wall over the trio)
    print()
    for trio in (("hint", "lib1", "lib2"), ("hint", "hint_mu2", "lib1"), ("hint", "hint_mu1", "lib1"),
                 ("hint", "hint_mu2", "hint_mu1"), ("hint", "hint_primal", "lib1"), ("hint", "hint_adapt", "lib1")):
        w = [min(r[f"{n}_wall_s"] for n in trio if r.get(f"{n}_success")) for r in rows
             if any(r.get(f"{n}_success") for n in trio)]
        it = [min(r[f"{n}_iter"] for n in trio if r.get(f"{n}_success")) for r in rows
              if any(r.get(f"{n}_success") for n in trio)]
        print(f"race {'+'.join(trio)}: solved {len(w)}/{len(rows)}, median {np.median(w):.2f} s, "
              f"p90 {np.percentile(w, 90):.2f} s, max {max(w):.2f} s, iters median {np.median(it):.0f} max {max(it)}")
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
