"""Planner specs for the robotics suite's task families, adapted from the
warm-start probe (benchmarks/robotics/probe_learned_ws.py):

  truck     tractor-trailer backing into a bay, task = start pose (x, y, heading)
  quad      quadrotor across a fixed forest, task = start and goal (y, z)
  quadmov   same forest plus 4 movable obstacles, task = start/goal + 4 (x, y)
  quadrand  a fresh 24-tree forest per task

Factory for the planner: specs.factory("truck", 100).
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROBOTICS = HERE.parents[1] / "benchmarks" / "robotics"


def factory(name: str, N: int) -> tuple:
    return ("specs", "probe_spec", (name, N), {}, [str(HERE), str(ROBOTICS)])


class ProbeSpec:
    def __init__(self, name: str, N: int):
        if str(ROBOTICS) not in sys.path:
            sys.path.insert(0, str(ROBOTICS))
        import probe_learned_ws as P
        self.T = P.PROBLEMS[name](N)
        self._features = P.features(self.T)
        prob = self.T.prob
        self.nlp = {"x": prob.x, "f": prob.f, "g": prob.g}
        if getattr(self.T, "p", None) is not None:
            self.nlp["p"] = self.T.p
        self.lbg, self.ubg = prob.lbg, prob.ubg

    def sample(self, rng):
        return self.T.sample(rng)

    def bounds(self, task):
        return self.T.bounds(task)

    def params(self, task):
        return self.T.pvec(task) if "p" in self.nlp else None

    def cold_guess(self, task):
        return self.T.cold_guess(task)

    def features(self, task):
        return self._features(task)


def probe_spec(name: str, N: int) -> ProbeSpec:
    return ProbeSpec(name, N)


def solver_setup(solver: str) -> tuple[dict, dict]:
    """IPOPT options and child environment for a linear solver, as the
    robotics suite runs it (one thread per process; smf with SMF_SLACK_ELIM=auto)."""
    if str(ROBOTICS) not in sys.path:
        sys.path.insert(0, str(ROBOTICS))
    import common
    opts = common.ipopt_options(solver, "", {})
    for k in ("ipopt.output_file", "ipopt.file_print_level", "ipopt.print_timing_statistics"):
        opts.pop(k, None)
    opts["ipopt.print_level"] = 0
    env = {k: v for k, v in common.runner_env(1).items() if k in (*common.THREAD_VARS, "SMF_SLACK_ELIM")}
    return opts, env


class TruckReplanSpec(ProbeSpec):
    """The truck with a full initial state, for replanning while moving:
    task = (x, y, th0, th1, v, delta). A library task from the probe's
    sampler (x, y, heading) is the same truck at rest, aligned."""

    def __init__(self, N: int):
        super().__init__("truck", N)
        import numpy as np
        self._np = np

    @staticmethod
    def at_rest(p):
        import numpy as np
        return np.array([p[0], p[1], p[2], p[2], 0.0, 0.0])

    def sample(self, rng):
        return self.at_rest(self.T.sample(rng))

    def bounds(self, task):
        lbx, ubx = self.T.prob.lbx.copy(), self.T.prob.ubx.copy()
        lbx[1:7] = ubx[1:7] = task
        return lbx, ubx

    def cold_guess(self, task):
        x0 = self.T.cold_guess(task[[0, 1, 2]])
        x0[1:7] = task
        return x0

    def features(self, task):
        np = self._np
        f = 2 * (task[:3] - self.T.lo) / (self.T.hi - self.T.lo) - 1
        return np.r_[f, (task[3] - task[2]) / 0.5, task[4] / 2.0, task[5] / 0.3]


def truck_replan_spec(N: int) -> TruckReplanSpec:
    return TruckReplanSpec(N)


def truck_factory(N: int) -> tuple:
    return ("specs", "truck_replan_spec", (N,), {}, [str(HERE), str(ROBOTICS)])
