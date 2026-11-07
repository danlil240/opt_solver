"""(e) 7-DoF arm (Franka Panda kinematics), point-to-point motion around a
spherical obstacle with torque and joint limits.

Kinematics: the Panda's modified DH parameters. Dynamics: each link is a
point mass at its frame origin (masses close to the Panda's link masses);
the gravity torque is the gradient of the potential energy, the inertia is
a configuration-dependent diagonal (rotor inertia plus the mass of the links
beyond times the squared distance from the joint axis, a cheap stand-in for
the full M(q)), and viscous joint friction. States q, qd (14); inputs the
joint torques (7). Free final time, minimum time plus torque effort.
Collision: points along the links (frame origins and midpoints) keep out of
the obstacle sphere and above the table. Multiple shooting, RK4.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from common import Builder, Problem, rk4, stage_function, stage_jit

# Panda modified DH: a_{i-1}, d_i, alpha_{i-1}.
DH = [(0, 0.333, 0), (0, 0, -np.pi / 2), (0, 0.316, np.pi / 2), (0.0825, 0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0, 0, np.pi / 2), (0.088, 0, np.pi / 2)]
FLANGE_D = 0.107 + 0.1034
MASS = np.array([4.97, 0.65, 3.23, 3.59, 1.23, 1.67, 0.74 + 0.73])
Q_LIM = np.array([2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973])
Q_LO = -Q_LIM.copy()
Q_LO[3], Q_LIM[3] = -3.0718, -0.0698
Q_LO[5], Q_LIM[5] = -0.0175, 3.7525
QD_LIM = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
TAU_LIM = np.array([87, 87, 87, 87, 12, 12, 12])
ROTOR = np.array([0.6, 0.6, 0.5, 0.5, 0.15, 0.15, 0.1])
FRICTION = 0.5
NQ = 7
OBST = np.array([0.45, 0.0, 0.45])


def tf(a, d, alpha, q):
    ca_, sa = np.cos(alpha), np.sin(alpha)
    cq, sq = ca.cos(q), ca.sin(q)
    return ca.vertcat(
        ca.horzcat(cq, -sq, 0, a),
        ca.horzcat(sq * ca_, cq * ca_, -sa, -sa * d),
        ca.horzcat(sq * sa, cq * sa, ca_, ca_ * d),
        ca.horzcat(0, 0, 0, 1))


def frames(q):
    """World-frame origins of joints 1..7 and the flange, and the z axes."""
    T = ca.SX.eye(4)
    origins, axes = [], []
    for i, (a, d, al) in enumerate(DH):
        T = T @ tf(a, d, al, q[i])
        origins.append(T[:3, 3])
        axes.append(T[:3, 2])
    Tf = T @ ca.vertcat(ca.horzcat(1, 0, 0, 0), ca.horzcat(0, 1, 0, 0),
                        ca.horzcat(0, 0, 1, FLANGE_D), ca.horzcat(0, 0, 0, 1))
    origins.append(Tf[:3, 3])
    return origins, axes


def make_dynamics():
    q, qd, tau = ca.SX.sym("q", NQ), ca.SX.sym("qd", NQ), ca.SX.sym("tau", NQ)
    origins, axes = frames(q)
    V = sum(MASS[i] * 9.81 * origins[i + 1][2] for i in range(NQ))
    grav = ca.gradient(V, q)
    # Diagonal inertia: rotor plus point masses beyond joint i, at their squared
    # distance from joint i's axis.
    diag = []
    for i in range(NQ):
        o, z = origins[i], axes[i]
        m = ROTOR[i]
        for j in range(i, NQ):
            r = origins[j + 1] - o
            perp = r - ca.dot(r, z) * z
            m += MASS[j] * ca.sumsqr(perp)
        diag.append(m)
    qdd = (tau - grav - FRICTION * qd) / ca.vertcat(*diag)
    f = ca.Function("arm", [ca.vertcat(q, qd), tau], [ca.vertcat(qd, qdd)])
    pts = []
    for i in range(1, len(origins)):
        pts.append(origins[i])
        pts.append(0.5 * (origins[i] + origins[i - 1]))
    P = stage_function("pts", [q], [ca.horzcat(*pts)])
    return f, P


def build(N: int, seed: int = 0) -> Problem:
    rng = np.random.default_rng(seed)
    fdyn, P = make_dynamics()
    q_start = np.array([-1.2, 0.3, 0.0, -1.8, 0.0, 2.0, 0.8]) + rng.uniform(-0.1, 0.1, NQ)
    q_goal = np.array([1.2, 0.3, 0.0, -1.8, 0.0, 2.0, 0.8]) + rng.uniform(-0.1, 0.1, NQ)
    obst_r = 0.18 + rng.uniform(0, 0.04)
    xs, us, hs = ca.SX.sym("x", 2 * NQ), ca.SX.sym("u", NQ), ca.SX.sym("h")
    F = stage_function("F", [xs, us, hs], [rk4(lambda x, u: fdyn(x, u), xs, us, hs)])
    b = Builder(mx=stage_jit())
    T = b.var("T", 1, 0.5, 10.0, 3.0)
    h = T / N
    X, U = [], []
    n_pts = P.size_out(0)[1]
    for k in range(N + 1):
        s = k / N
        g = np.r_[q_start + s * (q_goal - q_start), np.zeros(NQ)]
        lb, ub = np.r_[Q_LO, -QD_LIM], np.r_[Q_LIM, QD_LIM]
        if k == 0:
            lb = ub = np.r_[q_start, np.zeros(NQ)]
        if k == N:
            lb = ub = np.r_[q_goal, np.zeros(NQ)]
        X.append(b.var(f"x_{k}", 2 * NQ, lb, ub, g))
        if k < N:
            U.append(b.var(f"u_{k}", NQ, -TAU_LIM, TAU_LIM, 0.0))
    f = T
    for k in range(N):
        b.eq(F(X[k], U[k], h) - X[k + 1])
        f += 1e-3 * h * ca.sumsqr(U[k] / TAU_LIM * 10)
    for k in range(1, N):
        pts = P(X[k][:NQ])
        for j in range(2, n_pts):  # the base points cannot move
            b.ineq(ca.sumsqr(pts[:, j] - OBST), (obst_r + 0.06) ** 2, ca.inf)
            b.ineq(pts[2, j], 0.05, ca.inf)
    per = 3 * NQ

    def check(w: np.ndarray) -> dict:
        Qs = [w[1 + k * per: 1 + k * per + NQ] for k in range(N + 1)]
        dmin, zmin = np.inf, np.inf
        for q in Qs:
            pts = np.array(P(q))
            dmin = min(dmin, float(np.min(np.linalg.norm(pts[:, 2:].T - OBST, axis=1))) - obst_r)
            zmin = min(zmin, float(np.min(pts[2, 2:])))
        return {"T": float(w[0]), "obstacle_clearance": dmin, "min_height": zmin}

    return b.problem("arm7", f, check, {"N": N, "seed": seed, "family": "manipulator"})
