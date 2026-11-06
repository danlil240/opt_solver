"""(d) Tractor-trailer reversing into a loading bay between parked vehicles.

Kinematic single-track tractor with one on-axle hitched trailer:
  x' = v cos th0, y' = v sin th0, th0' = v tan(delta) / L0,
  th1' = v sin(th0 - th1) / L1,
with speed v and steering delta as states, acceleration and steering rate as
inputs. Free final time, minimum time plus effort. The vehicle is covered by
circles (tractor and trailer), every circle keeps clear of the obstacle
circles (parked trucks along the yard and the walls of the bay), and the
hitch angle stays below 60 deg (no jackknife). The trailer has to end inside
the bay, aligned with it, at rest. Multiple shooting, RK4.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from common import Builder, Problem, rk4

L0, L1 = 3.5, 8.0           # tractor wheelbase, trailer length (hitch to axle)
TRAC_R, TRAIL_R = 1.3, 1.3  # covering circle radii
NX, NU = 6, 2
BAY_X, BAY_Y, BAY_W = 0.0, -14.0, 4.0


def dynamics(x, u):
    _, _, th0, th1, v, delta = [x[i] for i in range(NX)]
    return ca.vertcat(v * ca.cos(th0), v * ca.sin(th0), v * ca.tan(delta) / L0,
                      v * ca.sin(th0 - th1) / L1, u[0], u[1])


def body_circles(x):
    """Centres of the covering circles: two on the tractor (rear axle at
    (x, y), hitch there), three along the trailer."""
    px, py, th0, th1 = x[0], x[1], x[2], x[3]
    c = []
    for t in (0.5, 2.5):
        c.append((px + t * ca.cos(th0), py + t * ca.sin(th0), TRAC_R))
    for t in (1.0, 4.0, 7.0):
        c.append((px - t * ca.cos(th1), py - t * ca.sin(th1), TRAIL_R))
    return c


def obstacle_set(seed: int):
    rng = np.random.default_rng(seed)
    obs = []
    # Bay walls: rows of circles at x = BAY_X +- (BAY_W / 2 + 1), y from -8 to -24.
    for y in np.arange(-6.0, -24.0, -1.5):
        obs.append((BAY_X - BAY_W / 2 - 1.0, y, 1.0))
        obs.append((BAY_X + BAY_W / 2 + 1.0, y, 1.0))
    # Parked trucks along the far side of the yard (y = +18), seeded gaps.
    for xc in np.arange(-30.0, 31.0, 6.0) + rng.uniform(-1.0, 1.0, 11):
        obs.append((float(xc), 18.0 + rng.uniform(-0.5, 0.5), 1.8))
    return obs


def build(N: int, seed: int = 0) -> Problem:
    obs = obstacle_set(seed)
    rng = np.random.default_rng(500 + seed)
    start = np.array([-22.0 + rng.uniform(-2, 2), 1.5, 0.0, 0.0, 0.0, 0.0])
    xs, us, hs = ca.SX.sym("x", NX), ca.SX.sym("u", NU), ca.SX.sym("h")
    F = ca.Function("F", [xs, us, hs], [rk4(dynamics, xs, us, hs)])
    b = Builder()
    T = b.var("T", 1, 5.0, 200.0, 50.0)
    h = T / N
    lbx = np.array([-60, -30, -4 * np.pi, -4 * np.pi, -3.0, -0.6])
    ubx = np.array([60, 25, 4 * np.pi, 4 * np.pi, 5.0, 0.6])
    X, U = [], []
    for k in range(N + 1):
        s = k / N
        # Guess: drive forward past the bay, then reverse in (a crude two-leg path).
        if s < 0.5:
            g = [start[0] + (18.0 - start[0]) * 2 * s, 1.5, 0, 0, 1.0, 0]
        else:
            t = (s - 0.5) * 2
            g = [18.0 * (1 - t) + BAY_X * t, 1.5 + 6.0 * np.sin(np.pi * t) + (BAY_Y + 6.0) * t,
                 t * np.pi / 2, t * np.pi / 2, -1.0, 0]
        lb, ub = lbx.copy(), ubx.copy()
        if k == 0:
            lb = ub = start
        X.append(b.var(f"x_{k}", NX, lb, ub, g))
        if k < N:
            U.append(b.var(f"u_{k}", NU, [-1.5, -0.4], [1.5, 0.4], 0.0))
    f = T
    for k in range(N):
        b.eq(F(X[k], U[k], h) - X[k + 1])
        f += 0.1 * h * ca.sumsqr(U[k])
    for k in range(1, N + 1):
        b.ineq(X[k][2] - X[k][3], -np.pi / 3, np.pi / 3)
        for cx, cy, cr in body_circles(X[k]):
            for ox, oy, orad in obs:
                # Only pairs that can meet: skip far obstacles cheaply? No: keep all,
                # the KKT should carry the full collision structure.
                b.ineq((cx - ox) ** 2 + (cy - oy) ** 2, (cr + orad) ** 2, ca.inf)
    # Terminal: trailer axle in the bay, trailer and tractor pointing up (+y),
    # at rest (the vehicle backs in, so the hitch is above the trailer axle).
    xe = X[N]
    tx = xe[0] - L1 * ca.cos(xe[3])
    ty = xe[1] - L1 * ca.sin(xe[3])
    b.eq(ca.vertcat(tx - BAY_X, ty - (BAY_Y - 2.0)))
    b.eq(ca.vertcat(xe[3] - np.pi / 2, xe[2] - np.pi / 2, xe[4], xe[5]))
    per = NX + NU

    def check(w: np.ndarray) -> dict:
        Xs = np.array([w[1 + k * per: 1 + k * per + NX] for k in range(N)] + [w[1 + N * per: 1 + N * per + NX]])
        worst = np.inf
        for x in Xs[1:]:
            for cx, cy, cr in body_circles(x):
                for ox, oy, orad in obs:
                    worst = min(worst, float(np.hypot(float(cx) - ox, float(cy) - oy) - cr - orad))
        return {"T": float(w[0]), "min_clearance": worst,
                "max_hitch_deg": float(np.degrees(np.max(np.abs(Xs[:, 2] - Xs[:, 3]))))}

    return b.problem("truck_trailer", f, check, {"N": N, "seed": seed, "family": "car-trailer"})
