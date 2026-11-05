"""(b) Three quadrotors swap places around a central pillar, with pairwise
collision avoidance.

Each drone is the full 12-state model of quadrotor.py. The drones start on a
circle of radius 3 m at 120 deg spacing and fly to the antipodal point of
the circle (drone i to the start of drone (i + 1) mod 3, rotated by 180 deg);
the straight paths all cross the centre, where a pillar of radius 0.6 m
stands. Fixed horizon T (seed jitters the start heights), minimum energy
plus a small time-regularised smoothness term. Every node keeps every pair
of drones at least D_MIN apart and every drone outside the pillar.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

import quadrotor as Q
from common import Builder, Problem, rk4

N_DRONES = 3
RADIUS = 3.0
PILLAR = 0.6
D_MIN = 0.8
HORIZON = 6.0


def build(N: int, seed: int = 0) -> Problem:
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(N_DRONES) / N_DRONES + rng.uniform(-0.2, 0.2)
    z0 = 1.5 + rng.uniform(-0.3, 0.3, N_DRONES)
    starts = np.c_[RADIUS * np.cos(ang), RADIUS * np.sin(ang), z0]
    goals = np.c_[-starts[:, 0], -starts[:, 1], z0[::-1]]
    h = HORIZON / N
    xs, us = ca.SX.sym("xs", Q.NX), ca.SX.sym("us", Q.NU)
    F = ca.Function("F", [xs, us], [rk4(Q.dynamics, xs, us, h)])
    b = Builder()
    X = [[None] * (N + 1) for _ in range(N_DRONES)]
    U = [[None] * N for _ in range(N_DRONES)]
    f = 0
    for k in range(N + 1):
        s = k / N
        for d in range(N_DRONES):
            # Initial guess: an arc around the pillar (counter-clockwise half turn).
            a = ang[d] + np.pi * s
            guess = np.zeros(Q.NX)
            guess[:3] = [RADIUS * np.cos(a), RADIUS * np.sin(a), z0[d] + s * (z0[::-1][d] - z0[d])]
            lb, ub = Q.X_LB.copy(), Q.X_UB.copy()
            if k == 0:
                lb = ub = np.r_[starts[d], np.zeros(9)]
            if k == N:
                lb = np.r_[goals[d], -0.2 * np.ones(3), -0.1, -0.1, -np.pi, -0.5, -0.5, -0.5]
                ub = np.r_[goals[d], 0.2 * np.ones(3), 0.1, 0.1, np.pi, 0.5, 0.5, 0.5]
            X[d][k] = b.var(f"x{d}_{k}", Q.NX, lb, ub, guess)
            if k < N:
                U[d][k] = b.var(f"u{d}_{k}", Q.NU, 0.0, Q.U_MAX, Q.HOVER)
    for d in range(N_DRONES):
        for k in range(N):
            b.eq(F(X[d][k], U[d][k]) - X[d][k + 1])
            f += h * ca.sumsqr(U[d][k] - Q.HOVER)
            if k > 0:
                f += 0.01 * ca.sumsqr(U[d][k] - U[d][k - 1]) / h
    for k in range(1, N):
        for d in range(N_DRONES):
            b.ineq(X[d][k][0] ** 2 + X[d][k][1] ** 2, (PILLAR + 0.25) ** 2, ca.inf)
            for e in range(d + 1, N_DRONES):
                b.ineq(ca.sumsqr(X[d][k][:3] - X[e][k][:3]), D_MIN ** 2, ca.inf)

    per_node = N_DRONES * (Q.NX + Q.NU)

    def check(w: np.ndarray) -> dict:
        P = np.zeros((N + 1, N_DRONES, 3))
        for k in range(N + 1):
            for d in range(N_DRONES):
                off = k * per_node + d * (Q.NX + (Q.NU if k < N else 0))
                P[k, d] = w[off: off + 3]
        dmin = min(np.min(np.linalg.norm(P[:, d] - P[:, e], axis=1))
                   for d in range(N_DRONES) for e in range(d + 1, N_DRONES))
        pil = float(np.min(np.hypot(P[..., 0], P[..., 1])))
        return {"min_pair_dist": float(dmin), "min_pillar_dist": pil,
                "goal_err": float(np.max(np.linalg.norm(P[-1] - goals, axis=1)))}

    return b.problem("drone_swap", f, check, {"N": N, "seed": seed, "family": "multi-quadrotor"})
