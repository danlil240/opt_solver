"""A pool of warm IPOPT processes that race several starts of one task.

Each worker builds the spec's NLP once (and its CasADi functions, compiled or
not) and keeps two IPOPT solvers alive: one with default options for cold
starts, one with warm-start options for starts that carry duals. A race
sends one start to each worker; the first success wins and the shared race
flag makes every other worker stop at its next iteration (IPOPT's
iteration callback, "User_Requested_Stop").
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from multiprocessing.connection import wait
from typing import Any

import numpy as np

from .problem import build_spec

WARM_OPTIONS = {
    "ipopt.warm_start_init_point": "yes", "ipopt.mu_init": 1e-4,
    "ipopt.warm_start_bound_push": 1e-6, "ipopt.warm_start_bound_frac": 1e-6,
    "ipopt.warm_start_slack_bound_push": 1e-6, "ipopt.warm_start_slack_bound_frac": 1e-6,
    "ipopt.warm_start_mult_bound_push": 1e-6,
}


def _stop_callback(nx: int, ng: int, np_: int, flag, race_box: list):
    import casadi as ca

    class StopWhenRaceOver(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.construct("smfplan_stop", {})

        def get_n_in(self):
            return ca.nlpsol_n_out()

        def get_n_out(self):
            return 1

        def get_name_in(self, i):
            return ca.nlpsol_out(i)

        def get_name_out(self, i):
            return "ret"

        def get_sparsity_in(self, i):
            n = ca.nlpsol_out(i)
            if n == "f":
                return ca.Sparsity.scalar()
            if n in ("x", "lam_x"):
                return ca.Sparsity.dense(nx)
            if n in ("g", "lam_g"):
                return ca.Sparsity.dense(ng)
            if n == "lam_p":
                return ca.Sparsity.dense(np_)
            return ca.Sparsity(0, 0)

        def eval(self, arg):
            return [1 if flag.value == race_box[0] else 0]

    return StopWhenRaceOver()


def _worker(factory: tuple, base_options: dict, conn, flag) -> None:
    import casadi as ca
    t0 = time.perf_counter()
    spec = build_spec(factory)
    nlp = spec.nlp
    nx, ng = nlp["x"].shape[0], nlp["g"].shape[0]
    np_ = nlp["p"].shape[0] if "p" in nlp else 0
    race_box = [-1]
    cb = _stop_callback(nx, ng, np_, flag, race_box)
    solvers = {}

    def solver(warm: bool):
        if warm not in solvers:
            opts = {**base_options, **(WARM_OPTIONS if warm else {}), "iteration_callback": cb}
            solvers[warm] = ca.nlpsol("smfplan", "ipopt", nlp, opts)
        return solvers[warm]

    solver(False)
    solver(True)
    conn.send({"ready": True, "build_s": time.perf_counter() - t0})
    while True:
        msg = conn.recv()
        if msg is None:
            break
        race_id, lbx, ubx, p, start = msg
        race_box[0] = race_id
        warm = "lam_g0" in start
        args = dict(x0=start["x0"], lbx=lbx, ubx=ubx, lbg=spec.lbg, ubg=spec.ubg)
        if p is not None:
            args["p"] = p
        if warm:
            args.update(lam_x0=start["lam_x0"], lam_g0=start["lam_g0"])
        S = solver(warm)
        t1 = time.perf_counter()
        try:
            r = S(**args)
            st = S.stats()
            out = {"success": bool(st.get("success")), "status": st.get("return_status"),
                   "iter": int(st.get("iter_count") or 0), "obj": float(r["f"]),
                   "x": np.array(r["x"]).ravel(), "lam_x": np.array(r["lam_x"]).ravel(),
                   "lam_g": np.array(r["lam_g"]).ravel()}
        except Exception as e:  # noqa: BLE001 - report, keep the worker alive
            out = {"success": False, "status": f"error: {e}", "iter": 0, "obj": np.inf}
        out["wall_s"] = time.perf_counter() - t1
        out["tag"] = start.get("tag")
        conn.send(out)


class SolverPool:
    """`workers` processes, each holding the spec's NLP and two IPOPT
    solvers. `env` is applied to the children's environment (thread counts,
    SMF_* variables) before they start."""

    def __init__(self, factory: tuple, workers: int = 3, solver_options: dict | None = None,
                 env: dict | None = None):
        self.ctx = mp.get_context("spawn")
        self.flag = self.ctx.Value("q", -1)
        self._race = 0
        base = {"print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes"}
        base.update(solver_options or {})
        saved = {k: os.environ.get(k) for k in (env or {})}
        os.environ.update(env or {})
        try:
            self.conns, self.procs = [], []
            for _ in range(workers):
                a, b = self.ctx.Pipe()
                pr = self.ctx.Process(target=_worker, args=(factory, base, b, self.flag), daemon=True)
                pr.start()
                self.conns.append(a)
                self.procs.append(pr)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.build_s = [c.recv()["build_s"] for c in self.conns]

    @property
    def workers(self) -> int:
        return len(self.conns)

    def race(self, lbx, ubx, p, starts: list[dict]) -> tuple[dict | None, list[dict], float]:
        """Solve one task from each start in parallel. Returns the winner
        (first success; None when none succeeded), every worker's result, and
        the wall time to the winner (or to the last result)."""
        assert 0 < len(starts) <= self.workers, "one start per worker"
        self._race += 1
        rid = self._race
        t0 = time.perf_counter()
        busy = {}
        for c, s in zip(self.conns, starts):
            c.send((rid, lbx, ubx, p, s))
            busy[c] = s
        results, winner, t_win = [], None, None
        while busy:
            for c in wait(list(busy)):
                r = c.recv()
                busy.pop(c)
                results.append(r)
                if winner is None and r["success"]:
                    winner, t_win = r, time.perf_counter() - t0
                    self.flag.value = rid
        return winner, results, t_win if t_win is not None else time.perf_counter() - t0

    def map(self, jobs: list[tuple]) -> list[dict]:
        """Independent solves (lbx, ubx, p, start) spread over the workers,
        no racing; results in job order."""
        self._race += 1
        rid = self._race
        out: list = [None] * len(jobs)
        todo = list(enumerate(jobs))
        busy = {}
        idle = list(self.conns)
        while todo or busy:
            while todo and idle:
                c = idle.pop()
                i, (lbx, ubx, p, s) = todo.pop(0)
                c.send((rid, lbx, ubx, p, s))
                busy[c] = i
            for c in wait(list(busy)):
                out[busy.pop(c)] = c.recv()
                idle.append(c)
        return out

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for pr in self.procs:
            pr.join(timeout=10)
            if pr.is_alive():
                pr.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
