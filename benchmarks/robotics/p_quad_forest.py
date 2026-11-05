"""(a) Quadrotor, minimum time plus energy, through a forest of vertical
cylinders.

Multiple shooting, one RK4 step per interval, free final time. The forest is
drawn from `seed`: trees in x in [3, L - 3], |y| <= 4, none within 1.5 m of
the straight start-goal line's endpoints. Every node keeps clear of every
tree by a margin (the drone's radius).
"""

from __future__ import annotations

import casadi as ca
import numpy as np

import quadrotor as Q
from common import Builder, Problem, rk4

LENGTH = 30.0
MARGIN = 0.35
N_TREES = 24


def forest(seed: int):
    rng = np.random.default_rng(seed)
    trees = []
    while len(trees) < N_TREES:
        c = np.array([rng.uniform(3, LENGTH - 3), rng.uniform(-4, 4)])
        r = rng.uniform(0.25, 0.6)
        if all(np.hypot(*(c - t[:2])) > r + t[2] + 2 * MARGIN + 0.4 for t in trees):
            trees.append(np.array([c[0], c[1], r]))
    return np.array(trees)


def build(N: int, seed: int = 0) -> Problem:
    trees = forest(seed)
    start = np.array([0, 0, 1.5])
    goal = np.array([LENGTH, 0, 1.5])
    b = Builder()
    T = b.var("T", 1, 2.0, 60.0, 12.0)
    h = T / N
    xs, us = ca.SX.sym("xs", Q.NX), ca.SX.sym("us", Q.NU)
    hs = ca.SX.sym("hs")
    F = ca.Function("F", [xs, us, hs], [rk4(Q.dynamics, xs, us, hs)])
    X, U = [], []
    energy = 0
    for k in range(N + 1):
        s = k / N
        guess = np.zeros(Q.NX)
        guess[:3] = start + s * (goal - start)
        guess[3] = (goal - start)[0] / 12.0
        lb, ub = Q.X_LB.copy(), Q.X_UB.copy()
        if k == 0:
            lb = ub = np.r_[start, np.zeros(9)]
        if k == N:
            lb = np.r_[goal, -0.5 * np.ones(3), -0.2, -0.2, -np.pi, -1, -1, -1]
            ub = np.r_[goal, 0.5 * np.ones(3), 0.2, 0.2, np.pi, 1, 1, 1]
        X.append(b.var(f"x_{k}", Q.NX, lb, ub, guess))
        if k < N:
            U.append(b.var(f"u_{k}", Q.NU, 0.0, Q.U_MAX, Q.HOVER))
    for k in range(N):
        b.eq(F(X[k], U[k], h) - X[k + 1])
        energy += ca.sumsqr(U[k] - Q.HOVER) * h
    for k in range(1, N):
        for tx, ty, tr in trees:
            b.ineq((X[k][0] - tx) ** 2 + (X[k][1] - ty) ** 2, (tr + MARGIN) ** 2, ca.inf)
    f = T + 0.05 * energy

    nx_tot = Q.NX + Q.NU

    def check(w: np.ndarray) -> dict:
        Tsol = w[0]
        P = np.array([w[1 + k * nx_tot: 1 + k * nx_tot + 3] for k in range(N)] + [w[1 + N * nx_tot: 1 + N * nx_tot + 3]])
        # Clearance at the nodes and at interval midpoints (straight-line interpolation).
        pts = np.vstack([P, 0.5 * (P[1:] + P[:-1])])
        d = np.min(np.hypot(pts[:, None, 0] - trees[None, :, 0], pts[:, None, 1] - trees[None, :, 1]) - trees[None, :, 2])
        return {"T": float(Tsol), "min_clearance": float(d), "goal_err": float(np.linalg.norm(P[-1] - goal))}

    return b.problem("quad_forest", f, check, {"N": N, "seed": seed, "family": "quadrotor"})
