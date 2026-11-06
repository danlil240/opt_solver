"""Stage-ordered variants of the robotics problems, for FATROP (and the same
NLPs for IPOPT, so the comparison is like for like).

FATROP needs the variables stage by stage, [x_k, u_k], the dynamics as
x_{k+1} = F(x_k, u_k), and each stage's other constraints on that stage's
variables only. A free final time T becomes a state with T_{k+1} = T_k (the
standard reformulation); the collocation car keeps its intra-interval
collocation states as part of the stage's "controls".

  python stagewise.py <problem> <N> <solver> [seed]   solver: fatrop | mumps | ma27 | ma57 | ma97 | smf
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time

import casadi as ca
import numpy as np

import common
import p_car_track as C
import p_drone_swap as DS
import p_quad_forest as QF
import p_truck_trailer as TT
import quadrotor as Q
from common import Builder, rk4


class Staged(Builder):
    """Builder that also records the stage dimensions (nx, nu, ng)."""

    def __init__(self, N):
        super().__init__()
        self.nx, self.nu, self.ng = [0] * (N + 1), [0] * (N + 1), [0] * (N + 1)

    def x(self, k, name, size, lb, ub, g):
        self.nx[k] += size
        return self.var(name, size, lb, ub, g)

    def u(self, k, name, size, lb, ub, g):
        self.nu[k] += size
        return self.var(name, size, lb, ub, g)

    def path(self, k, expr, lb, ub):
        self.ng[k] += ca.vec(expr).shape[0]
        self.ineq(expr, lb, ub)

    def path_eq(self, k, expr):
        self.ng[k] += ca.vec(expr).shape[0]
        self.eq(expr)


def quad_forest(N, seed):
    trees = QF.forest(seed)
    start, goal = np.array([0, 0, 1.5]), np.array([QF.LENGTH, 0, 1.5])
    xs, us, hs = ca.SX.sym("x", Q.NX), ca.SX.sym("u", Q.NU), ca.SX.sym("h")
    F = ca.Function("F", [xs, us, hs], [rk4(Q.dynamics, xs, us, hs)])
    b = Staged(N)
    X, U, T = [], [], []
    for k in range(N + 1):
        s = k / N
        g = np.zeros(Q.NX)
        g[:3] = start + s * (goal - start)
        g[3] = (goal - start)[0] / 12.0
        lb, ub = Q.X_LB.copy(), Q.X_UB.copy()
        if k == 0:
            lb = ub = np.r_[start, np.zeros(9)]
        if k == N:
            lb = np.r_[goal, -0.5 * np.ones(3), -0.2, -0.2, -np.pi, -1, -1, -1]
            ub = np.r_[goal, 0.5 * np.ones(3), 0.2, 0.2, np.pi, 1, 1, 1]
        X.append(b.x(k, f"x_{k}", Q.NX, lb, ub, g))
        T.append(b.x(k, f"T_{k}", 1, 2.0, 60.0, 12.0))
        if k < N:
            U.append(b.u(k, f"u_{k}", Q.NU, 0.0, Q.U_MAX, Q.HOVER))
        if k > 0:
            for tx, ty, tr in trees:
                b.path(k, (X[k][0] - tx) ** 2 + (X[k][1] - ty) ** 2, (tr + QF.MARGIN) ** 2, ca.inf)
    f = T[0]
    gs = []
    for k in range(N):
        h = T[k] / N
        f += 0.05 * ca.sumsqr(U[k] - Q.HOVER) * h
        gs.append(ca.vertcat(X[k + 1] - F(X[k], U[k], h), T[k + 1] - T[k]))
    return b, f, gs


def drone_swap(N, seed):
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(DS.N_DRONES) / DS.N_DRONES + rng.uniform(-0.2, 0.2)
    z0 = 1.5 + rng.uniform(-0.3, 0.3, DS.N_DRONES)
    starts = np.c_[DS.RADIUS * np.cos(ang), DS.RADIUS * np.sin(ang), z0]
    goals = np.c_[-starts[:, 0], -starts[:, 1], z0[::-1]]
    h = DS.HORIZON / N
    xs, us = ca.SX.sym("x", Q.NX), ca.SX.sym("u", Q.NU)
    F = ca.Function("F", [xs, us], [rk4(Q.dynamics, xs, us, h)])
    b = Staged(N)
    X = [[None] * (N + 1) for _ in range(DS.N_DRONES)]
    U = [[None] * N for _ in range(DS.N_DRONES)]
    # P[d][k] = U[d][k-1], carried as a state (P_{k+1} = U_k), so the control
    # smoothing term (U_k - U_{k-1}) is a function of stage k's variables only,
    # as FATROP's stage-wise cost requires. P_0 is fixed (no k = 0 term uses it).
    P = [[None] * (N + 1) for _ in range(DS.N_DRONES)]
    for k in range(N + 1):
        s = k / N
        for d in range(DS.N_DRONES):
            a = ang[d] + np.pi * s
            g = np.zeros(Q.NX)
            g[:3] = [DS.RADIUS * np.cos(a), DS.RADIUS * np.sin(a), z0[d] + s * (z0[::-1][d] - z0[d])]
            lb, ub = Q.X_LB.copy(), Q.X_UB.copy()
            if k == 0:
                lb = ub = np.r_[starts[d], np.zeros(9)]
            if k == N:
                lb = np.r_[goals[d], -0.2 * np.ones(3), -0.1, -0.1, -np.pi, -0.5, -0.5, -0.5]
                ub = np.r_[goals[d], 0.2 * np.ones(3), 0.1, 0.1, np.pi, 0.5, 0.5, 0.5]
            X[d][k] = b.x(k, f"x{d}_{k}", Q.NX, lb, ub, g)
        for d in range(DS.N_DRONES):
            if k == 0:
                P[d][k] = b.x(k, f"p{d}_{k}", Q.NU, Q.HOVER, Q.HOVER, Q.HOVER)
            else:
                P[d][k] = b.x(k, f"p{d}_{k}", Q.NU, -ca.inf, ca.inf, Q.HOVER)
        for d in range(DS.N_DRONES):
            if k < N:
                U[d][k] = b.u(k, f"u{d}_{k}", Q.NU, 0.0, Q.U_MAX, Q.HOVER)
        if 0 < k < N:
            for d in range(DS.N_DRONES):
                b.path(k, X[d][k][0] ** 2 + X[d][k][1] ** 2, (DS.PILLAR + 0.25) ** 2, ca.inf)
                for e in range(d + 1, DS.N_DRONES):
                    b.path(k, ca.sumsqr(X[d][k][:3] - X[e][k][:3]), DS.D_MIN ** 2, ca.inf)
    f = 0
    gs = []
    for k in range(N):
        gs.append(ca.vertcat(*[X[d][k + 1] - F(X[d][k], U[d][k]) for d in range(DS.N_DRONES)],
                             *[P[d][k + 1] - U[d][k] for d in range(DS.N_DRONES)]))
        for d in range(DS.N_DRONES):
            f += h * ca.sumsqr(U[d][k] - Q.HOVER)
            if k > 0:
                f += 0.01 * ca.sumsqr(U[d][k] - P[d][k]) / h
    return b, f, gs


def car_track(N, seed):
    kappa = C.curvature_fn(seed)
    obs = C.obstacles(seed)
    tau = np.array([0.0] + ca.collocation_points(C.DEGREE, "legendre"))
    Cm = np.zeros((C.DEGREE + 1, C.DEGREE + 1))
    Dend = np.zeros(C.DEGREE + 1)
    Bq = np.zeros(C.DEGREE + 1)
    for j in range(C.DEGREE + 1):
        p = np.poly1d([1.0])
        for r in range(C.DEGREE + 1):
            if r != j:
                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])
        Dend[j] = p(1.0)
        dp = np.polyder(p)
        for r in range(C.DEGREE + 1):
            Cm[j, r] = dp(tau[r])
        Bq[j] = np.polyint(p)(1.0)
    ds = C.LAP / N
    xs, us, ks = ca.SX.sym("x", C.NX), ca.SX.sym("u", C.NU), ca.SX.sym("k")
    fdx, sdot = C.time_dynamics(xs, us, ks)
    f_sp = ca.Function("f_sp", [xs, us, ks], [fdx / sdot, 1 / sdot])
    lbx = np.array([-(C.HALF_WIDTH - C.CAR_HALF), -1.2, 0.3, -2.0, -8.0, -0.35, -0.1])
    ubx = np.array([C.HALF_WIDTH - C.CAR_HALF, 1.2, 3.5, 2.0, 8.0, 0.35, 1.0])
    x0 = np.array([0, 0, 1.0, 0, 0, 0, 0.3])
    b = Staged(N)
    X, gs = [], []
    lap = 0
    reg = 0
    for k in range(N + 1):
        # A fixed start state (the stagewise form cannot close the lap).
        X.append(b.x(k, f"x_{k}", C.NX, x0 if k == 0 else lbx, x0 if k == 0 else ubx, x0))
        if k == N:
            break
        Uk = b.u(k, f"u_{k}", C.NU, [-5.0, -10.0], [5.0, 10.0], 0.0)
        Xc = [b.u(k, f"xc_{k}_{j}", C.NX, lbx, ubx, x0) for j in range(C.DEGREE)]
        reg += ca.sumsqr(Uk)
        xend = Dend[0] * X[k]
        stage_eq = []
        for j in range(1, C.DEGREE + 1):
            xp = Cm[0, j] * X[k]
            for r in range(C.DEGREE):
                xp += Cm[r + 1, j] * Xc[r]
            sj = (k + tau[j]) * ds
            fj, qj = f_sp(Xc[j - 1], Uk, kappa(sj))
            stage_eq.append(ds * fj - xp)
            xend += Dend[j] * Xc[j - 1]
            lap += Bq[j] * qj * ds
            for so, sl, no, nw in obs:
                if abs(sj - so) <= sl:
                    b.path(k, (Xc[j - 1][0] - no) ** 2, (nw + C.CAR_HALF) ** 2, ca.inf)
        b.path_eq(k, ca.vertcat(*stage_eq))
        X_end = xend
        gs.append(X_end)  # placeholder, dynamics rewritten below
    # Dynamics in FATROP form: x_{k+1} = (collocation end point of interval k).
    gs = [None] * N
    return b, lap + 1e-4 * reg, gs, X, Dend


def truck_trailer(N, seed):
    obs = TT.obstacle_set(seed)
    rng = np.random.default_rng(500 + seed)
    start = np.array([-22.0 + rng.uniform(-2, 2), 1.5, 0.0, 0.0, 0.0, 0.0])
    xs, us, hs = ca.SX.sym("x", TT.NX), ca.SX.sym("u", TT.NU), ca.SX.sym("h")
    F = ca.Function("F", [xs, us, hs], [rk4(TT.dynamics, xs, us, hs)])
    b = Staged(N)
    lbx = np.array([-60, -30, -4 * np.pi, -4 * np.pi, -3.0, -0.6])
    ubx = np.array([60, 25, 4 * np.pi, 4 * np.pi, 5.0, 0.6])
    X, U, T = [], [], []
    for k in range(N + 1):
        s = k / N
        if s < 0.5:
            g = [start[0] + (18.0 - start[0]) * 2 * s, 1.5, 0, 0, 1.0, 0]
        else:
            t = (s - 0.5) * 2
            g = [18.0 * (1 - t) + TT.BAY_X * t, 1.5 + 6.0 * np.sin(np.pi * t) + (TT.BAY_Y + 6.0) * t,
                 t * np.pi / 2, t * np.pi / 2, -1.0, 0]
        lb, ub = lbx.copy(), ubx.copy()
        if k == 0:
            lb = ub = start
        X.append(b.x(k, f"x_{k}", TT.NX, lb, ub, g))
        T.append(b.x(k, f"T_{k}", 1, 5.0, 200.0, 50.0))
        if k < N:
            U.append(b.u(k, f"u_{k}", TT.NU, [-1.5, -0.4], [1.5, 0.4], 0.0))
        if k > 0:
            b.path(k, X[k][2] - X[k][3], -np.pi / 3, np.pi / 3)
            for cx, cy, cr in TT.body_circles(X[k]):
                for ox, oy, orad in obs:
                    b.path(k, (cx - ox) ** 2 + (cy - oy) ** 2, (cr + orad) ** 2, ca.inf)
        if k == N:
            xe = X[N]
            tx = xe[0] - TT.L1 * ca.cos(xe[3])
            ty = xe[1] - TT.L1 * ca.sin(xe[3])
            b.path_eq(k, ca.vertcat(tx - TT.BAY_X, ty - (TT.BAY_Y - 2.0)))
            b.path_eq(k, ca.vertcat(xe[3] - np.pi / 2, xe[2] - np.pi / 2, xe[4], xe[5]))
    f = T[0]
    gs = []
    for k in range(N):
        h = T[k] / N
        f += 0.1 * h * ca.sumsqr(U[k])
        gs.append(ca.vertcat(X[k + 1] - F(X[k], U[k], h), T[k + 1] - T[k]))
    return b, f, gs


def assemble(b: Staged, f, gs, N):
    """Interleaves the dynamics with the path constraints in FATROP's order:
    for stage k, the gap-closing constraints x_{k+1} - F(x_k, u_k), then the
    path constraints of stage k (Builder order)."""
    # Builder recorded path constraints in stage order already; split them by ng.
    path = list(zip(b.g, b.lbg, b.ubg))
    rows = []
    lbg, ubg = [], []
    pi = 0
    for k in range(N + 1):
        if k < N:
            rows.append(gs[k])
            lbg.append(np.zeros(gs[k].shape[0]))
            ubg.append(np.zeros(gs[k].shape[0]))
        need = b.ng[k]
        got = 0
        while got < need:
            e, lo, hi = path[pi]
            rows.append(e)
            lbg.append(lo)
            ubg.append(hi)
            got += e.shape[0]
            pi += 1
    assert pi == len(path)
    return (ca.vertcat(*b.w), f, ca.vertcat(*rows), np.concatenate(b.lbw), np.concatenate(b.ubw),
            np.concatenate(lbg), np.concatenate(ubg), np.concatenate(b.w0))


def build(problem, N, seed=0):
    if problem == "car_track":
        b, f, _, X, Dend = car_track(N, seed)
        # The collocation end point is the last path_eq row block? No: rebuild
        # the end points from the variables directly.
        gs = []
        nper = C.NU + C.DEGREE * C.NX
        for k in range(N):
            xc = [b.w[1 + k * (C.DEGREE + 2) + 1 + j] for j in range(C.DEGREE)]
            xend = Dend[0] * X[k]
            for j in range(1, C.DEGREE + 1):
                xend += Dend[j] * xc[j - 1]
            gs.append(X[k + 1] - xend)
        return b, f, gs
    return {"quad_forest": quad_forest, "drone_swap": drone_swap, "truck_trailer": truck_trailer}[problem](N, seed)


def main():
    common.ensure_runner_env(1)  # its records are 1-thread runs
    problem, N, solver = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    b, f, gs = build(problem, N, seed)
    x, f, g, lbx, ubx, lbg, ubg, x0 = assemble(b, f, gs, N)
    fd, out_file = tempfile.mkstemp(prefix="stagewise_", suffix=".txt")
    os.close(fd)
    if solver == "fatrop":
        opts = {"structure_detection": "manual", "nx": b.nx, "nu": b.nu, "ng": b.ng, "N": N,
                "print_time": False, "fatrop": {"print_level": 0, "tol": 1e-8, "max_iter": 3000}}
    else:
        opts = common.nlp_options(solver, out_file)
    S = ca.nlpsol("S", solver if solver == "fatrop" else "ipopt", {"x": x, "f": f, "g": g}, opts)
    t0 = time.perf_counter()
    r = S(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    wall = time.perf_counter() - t0
    st = S.stats()
    del S
    import gc
    gc.collect()
    rec = {"problem": problem, "N": N, "seed": seed, "solver": solver, "status": st.get("return_status"),
           "success": bool(st.get("success")), "iter": st.get("iter_count"), "obj": float(r["f"]), "wall_s": wall,
           "n_var": int(x.shape[0]), "n_con": int(g.shape[0]),
           "nx_stage": b.nx[1], "nu_stage": b.nu[1], "ng_stage": b.ng[1],
           "t_nlp": sum(v for k, v in st.items() if k.startswith("t_wall_nlp")),
           "stats_keys": [k for k in st if k.startswith("t_wall")]}
    if solver != "fatrop":
        try:
            text = open(out_file, errors="replace").read()
            info = common.parse_ipopt_output(text)
            rec["lin_s"] = sum((info.get(k) or 0) for k in ("symbolic", "factor", "backsolve"))
        except OSError:
            pass
    else:
        rec.update({k: v for k, v in st.items() if isinstance(v, (int, float)) and k.startswith("t_")})
    try:
        os.remove(out_file)
    except OSError:
        pass
    print("STAGEWISE " + json.dumps(rec, default=float))


if __name__ == "__main__":
    main()
