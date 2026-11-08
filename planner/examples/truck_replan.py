"""Truck replanning demo for the planner.

  python truck_replan.py --solver smf --N 100 --seed-tasks 60 --new-tasks 20 --chains 3 --replans 8

1. Seed: cold-solve `seed-tasks` random start poses (the truck at rest
   somewhere in the yard) in parallel and store them.
2. New tasks: plan `new-tasks` unseen start poses with the planner (race of
   the 3 nearest stored solutions), and cold-solve each for comparison.
3. Replanning: for `chains` new start poses, plan, drive 10% of the plan
   (take the state at node N/10, with a small disturbance), and replan from
   there, `replans` times; every replan is also cold-solved for comparison.
   The previous plan races as one of the starts (--hint 0 turns that off).
The planner learns from every solve (the library grows as it plans).
Prints a summary and writes every record to --out (JSON lines).
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


def cold_batch(pl: Planner, tasks: list) -> list[dict]:
    jobs = []
    for t in tasks:
        lbx, ubx = pl.spec.bounds(t)
        jobs.append((lbx, ubx, pl.spec.params(t), {"x0": pl.spec.cold_guess(t), "tag": "cold"}))
    return pl.pool.map(jobs)


def summarize(name: str, recs: list[dict]) -> None:
    c = np.array([r["cold_s"] for r in recs if r["cold_ok"] and r["plan_ok"]])
    p = np.array([r["plan_s"] for r in recs if r["cold_ok"] and r["plan_ok"]])
    nf_c = sum(not r["cold_ok"] for r in recs)
    nf_p = sum(not r["plan_ok"] for r in recs)
    if not len(c):
        print(f"{name}: nothing to compare")
        return
    print(f"{name}: {len(recs)} tasks, failures cold {nf_c} / planner {nf_p}")
    print(f"  cold    median {np.median(c):7.2f} s  p90 {np.percentile(c, 90):7.2f} s  max {c.max():7.2f} s")
    print(f"  planner median {np.median(p):7.2f} s  p90 {np.percentile(p, 90):7.2f} s  max {p.max():7.2f} s")
    print(f"  total speedup {c.sum() / p.sum():.2f}x, median per-task speedup {np.median(c / p):.1f}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="smf")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed-tasks", type=int, default=60)
    ap.add_argument("--new-tasks", type=int, default=20)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--replans", type=int, default=8)
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--hint", type=int, default=1, help="1: the previous plan races alongside the library")
    ap.add_argument("--out", default="truck_replan.jsonl")
    a = ap.parse_args()
    opts, env = specs.solver_setup(a.solver)
    rng = np.random.default_rng(a.rng)
    out = open(a.out, "w", encoding="utf-8")

    def log(rec):
        out.write(json.dumps(rec, default=float) + "\n")
        out.flush()

    t0 = time.perf_counter()
    with Planner(specs.truck_factory(a.N), workers=a.workers, k=3, solver_options=opts, env=env) as pl:
        print(f"pool ready: build {max(pl.pool.build_s):.1f} s per worker, {time.perf_counter() - t0:.1f} s total",
              flush=True)
        stride = 8
        seed_tasks = [pl.spec.sample(rng) for _ in range(a.seed_tasks)]
        t1 = time.perf_counter()
        res = pl.seed(seed_tasks)
        ok = sum(r["success"] for r in res)
        print(f"seed: {ok}/{len(res)} cold solves stored in {time.perf_counter() - t1:.0f} s "
              f"(median {np.median([r['wall_s'] for r in res]):.1f} s per solve)", flush=True)
        for t, r in zip(seed_tasks, res):
            log({"phase": "seed", "task": t.tolist(), "ok": r["success"], "wall_s": r["wall_s"], "iter": r["iter"]})

        # New tasks.
        new = [pl.spec.sample(rng) for _ in range(a.new_tasks)]
        cold = cold_batch(pl, new)
        recs = []
        for t, c in zip(new, cold):
            r = pl.plan(t)
            rec = {"phase": "new", "task": t.tolist(), "cold_ok": c["success"], "cold_s": c["wall_s"],
                   "cold_iter": c["iter"], "cold_obj": c["obj"], "plan_ok": r.success, "plan_s": r.wall_s,
                   "plan_iter": r.iters, "plan_obj": r.obj, "source": r.source, "starts": r.starts,
                   "library": len(pl.library)}
            recs.append(rec)
            log(rec)
            print(f"new  {r.source:5s} plan {r.wall_s:6.2f} s ({r.iters:4d} it)  cold {c['wall_s']:6.2f} s "
                  f"({c['iter']:4d} it)  obj {r.obj:8.3f} vs {c['obj']:8.3f}", flush=True)
        summarize("new tasks", recs)

        # Replanning chains.
        reps = []
        for ch in range(a.chains):
            state = pl.spec.sample(rng)
            r = pl.plan(state)
            for k in range(a.replans):
                if not r.success:
                    break
                j = a.N // 10
                o = 1 + j * stride
                state = np.array(r.x[o:o + 6])
                state[:2] += rng.uniform(-0.3, 0.3, 2)
                state[2:4] += rng.uniform(-0.03, 0.03, 2)
                state[4] = np.clip(state[4], -3.0, 5.0)
                state[5] = np.clip(state[5], -0.6, 0.6)
                r = pl.plan(state, hint=r.as_hint() if a.hint else None)
                rec = {"phase": "replan", "chain": ch, "step": k, "task": state.tolist(), "plan_ok": r.success,
                       "plan_s": r.wall_s, "plan_iter": r.iters, "plan_obj": r.obj, "source": r.source,
                       "starts": r.starts}
                reps.append(rec)
                print(f"replan {ch}.{k} {r.source:5s} {r.wall_s:6.2f} s ({r.iters:4d} it)", flush=True)
        cold = cold_batch(pl, [np.array(r["task"]) for r in reps])
        for rec, c in zip(reps, cold):
            rec.update(cold_ok=c["success"], cold_s=c["wall_s"], cold_iter=c["iter"], cold_obj=c["obj"])
            log(rec)
        summarize("replans", reps)
    out.close()
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
