"""The planner: library lookup + parallel race + learning from every solve."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from .library import SolutionLibrary
from .pool import SolverPool
from .problem import build_spec


@dataclasses.dataclass
class PlanResult:
    success: bool
    x: np.ndarray | None
    obj: float
    wall_s: float          # time to the winning solve (what a caller waits)
    iters: int
    source: str            # "cold" or "lib<k>" (k-th nearest stored solution)
    status: str
    starts: list           # (tag, status, iters, wall_s) of every start in the race
    lam_x: np.ndarray | None = None
    lam_g: np.ndarray | None = None

    def as_hint(self) -> dict:
        return {"x": self.x, "lam_x": self.lam_x, "lam_g": self.lam_g}


class Planner:
    """Plans tasks of one spec. `k` nearest stored solutions are raced on
    `workers` processes; spare workers (and an empty library) get the cold
    guess. Every successful plan is stored, so the planner gets faster as it
    sees more tasks. `min_dist` (feature units) keeps the library from filling
    with near-duplicates: a new solution that close to a stored one replaces
    it only if its objective is lower."""

    def __init__(self, factory: tuple, workers: int = 3, k: int = 3, solver_options: dict | None = None,
                 env: dict | None = None, library: SolutionLibrary | None = None, min_dist: float = 0.0,
                 cold_in_race: bool = False):
        self.spec = build_spec(factory)
        self.pool = SolverPool(factory, workers, solver_options, env)
        self.library = library if library is not None else SolutionLibrary()
        self.k = min(k, workers)
        self.min_dist = min_dist
        self.cold_in_race = cold_in_race

    def starts_for(self, task, hint: dict | None = None) -> tuple[list[dict], Any, Any, Any]:
        lbx, ubx = self.spec.bounds(task)
        p = self.spec.params(task)
        feat = self.spec.features(task)
        starts = []
        if hint is not None:
            starts.append({"x0": np.clip(hint["x"], lbx, ubx), "lam_x0": hint["lam_x"], "lam_g0": hint["lam_g"],
                           "tag": "hint"})
        _, idx = self.library.nearest(feat, self.k - len(starts))
        for rank, j in enumerate(idx, 1):
            e = self.library.entry(int(j))
            starts.append({"x0": np.clip(e["x"], lbx, ubx), "lam_x0": e["lam_x"], "lam_g0": e["lam_g"],
                           "tag": f"lib{rank}"})
        if not starts or (self.cold_in_race and len(starts) < self.pool.workers):
            starts.append({"x0": self.spec.cold_guess(task), "tag": "cold"})
        return starts, lbx, ubx, p

    def plan(self, task, learn: bool = True, hint: dict | None = None) -> PlanResult:
        """Plan one task. `hint` (x, lam_x, lam_g; e.g. the previous plan's
        PlanResult.as_hint() in a replanning loop) takes one of the race's
        slots ahead of the library."""
        starts, lbx, ubx, p = self.starts_for(task, hint)
        win, results, wall = self.pool.race(lbx, ubx, p, starts)
        log = [(r.get("tag"), r["status"], r["iter"], r["wall_s"]) for r in results]
        if win is None:
            # every start failed: fall back to a cold solve when it wasn't in the race
            if not any(s["tag"] == "cold" for s in starts):
                win2, results2, wall2 = self.pool.race(lbx, ubx, p, [{"x0": self.spec.cold_guess(task), "tag": "cold"}])
                log += [(r.get("tag"), r["status"], r["iter"], r["wall_s"]) for r in results2]
                wall += wall2
                win = win2
            if win is None:
                best = min(results, key=lambda r: r["obj"])
                return PlanResult(False, best.get("x"), best["obj"], wall, best["iter"], best.get("tag"),
                                  best["status"], log)
        if learn:
            self.library.add(self.spec.features(task), win["x"], win["lam_x"], win["lam_g"], win["obj"],
                             self.min_dist)
        return PlanResult(True, win["x"], win["obj"], wall, win["iter"], win.get("tag"), win["status"], log,
                          win["lam_x"], win["lam_g"])

    def seed(self, tasks: list) -> list[dict]:
        """Cold-solve tasks in parallel over the workers and store the
        successes (to fill an empty library)."""
        jobs = []
        for t in tasks:
            lbx, ubx = self.spec.bounds(t)
            jobs.append((lbx, ubx, self.spec.params(t), {"x0": self.spec.cold_guess(t), "tag": "cold"}))
        res = self.pool.map(jobs)
        for t, r in zip(tasks, res):
            if r["success"]:
                self.library.add(self.spec.features(t), r["x"], r["lam_x"], r["lam_g"], r["obj"], self.min_dist)
        return res

    def close(self) -> None:
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
