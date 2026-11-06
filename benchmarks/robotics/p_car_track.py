"""(c) Race car, minimum lap time on a winding track with parked obstacles.

Dynamic bicycle model with simplified Pacejka (magic formula) lateral tire
forces and a motor/drag longitudinal force: the 1:43 RC car of Liniger,
Domahidi and Morari, "Optimization-based autonomous racing of 1:43 scale RC
cars" (2015). The track is described in Frenet coordinates along its
centreline (arc length s, curvature kappa(s), a seeded sum of sinusoids), and
the problem is posed in the spatial domain: the independent variable is s,
so the lap time is the integral of ds / s_dot.

States: lateral offset n, heading error mu, vx, vy, yaw rate r, steering
delta, throttle D. Inputs: steering rate and throttle rate. Direct
collocation, Legendre points of degree 3, N intervals over the lap. Track
bounds |n| <= half width minus the car's half width; obstacles are boxes in
(s, n) that the car must pass on either side (a nonconvex constraint
(n - n_o)^2 >= w_o^2 at the collocation nodes inside the box's s range).
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from common import Builder, Problem

# Liniger et al. 2015, table I.
M, IZ, LF, LR = 0.041, 27.8e-6, 0.029, 0.033
CM1, CM2, CR0, CR2 = 0.287, 0.0545, 0.0518, 0.00035
BR, CR, DR = 3.3852, 1.2691, 0.1737
BF, CF, DF = 2.579, 1.2, 0.192
HALF_WIDTH = 0.18
CAR_HALF = 0.03
LAP = 12.0
NX, NU = 7, 2
DEGREE = 3


def curvature_fn(seed: int):
    rng = np.random.default_rng(seed)
    k = np.arange(1, 5)
    amp = rng.uniform(0.6, 1.4, 4) / k
    ph = rng.uniform(0, 2 * np.pi, 4)
    s = ca.SX.sym("s")
    kap = sum(amp[i] * ca.sin(2 * np.pi * k[i] * s / LAP + ph[i]) for i in range(4))
    return ca.Function("kappa", [s], [kap])


def time_dynamics(x, u, kap):
    n, mu, vx, vy, r, delta, D = [x[i] for i in range(NX)]
    alpha_f = -ca.atan2(LF * r + vy, vx) + delta
    alpha_r = ca.atan2(LR * r - vy, vx)
    Ffy = DF * ca.sin(CF * ca.atan(BF * alpha_f))
    Fry = DR * ca.sin(CR * ca.atan(BR * alpha_r))
    Frx = (CM1 - CM2 * vx) * D - CR0 - CR2 * vx ** 2
    sdot = (vx * ca.cos(mu) - vy * ca.sin(mu)) / (1 - n * kap)
    return ca.vertcat(
        vx * ca.sin(mu) + vy * ca.cos(mu),
        r - kap * sdot,
        (Frx - Ffy * ca.sin(delta) + M * vy * r) / M,
        (Fry + Ffy * ca.cos(delta) - M * vx * r) / M,
        (Ffy * LF * ca.cos(delta) - Fry * LR) / IZ,
        u[0],
        u[1]), sdot


def obstacles(seed: int):
    rng = np.random.default_rng(1000 + seed)
    obs = []
    for s0 in np.linspace(1.5, LAP - 1.5, 4) + rng.uniform(-0.4, 0.4, 4):
        side = rng.choice([-1, 1])
        obs.append((s0, 0.12, side * 0.08, 0.045))  # s centre, s half length, n centre, n half width
    return obs


def build(N: int, seed: int = 0) -> Problem:
    kappa = curvature_fn(seed)
    obs = obstacles(seed)
    tau = np.array([0.0] + ca.collocation_points(DEGREE, "legendre"))
    C = np.zeros((DEGREE + 1, DEGREE + 1))
    Dend = np.zeros(DEGREE + 1)
    Bq = np.zeros(DEGREE + 1)
    for j in range(DEGREE + 1):
        p = np.poly1d([1.0])
        for r in range(DEGREE + 1):
            if r != j:
                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])
        Dend[j] = p(1.0)
        dp = np.polyder(p)
        for r in range(DEGREE + 1):
            C[j, r] = dp(tau[r])
        Bq[j] = np.polyint(p)(1.0)
    ds = LAP / N
    xs, us, ks = ca.SX.sym("x", NX), ca.SX.sym("u", NU), ca.SX.sym("k")
    fdx, sdot = time_dynamics(xs, us, ks)
    # Spatial dynamics: dx/ds = (dx/dt) / s_dot; cost integrand dt/ds = 1 / s_dot.
    f_sp = ca.Function("f_sp", [xs, us, ks], [fdx / sdot, 1 / sdot])
    lbx = np.array([-(HALF_WIDTH - CAR_HALF), -1.2, 0.3, -2.0, -8.0, -0.35, -0.1])
    ubx = np.array([HALF_WIDTH - CAR_HALF, 1.2, 3.5, 2.0, 8.0, 0.35, 1.0])
    b = Builder()
    x0 = np.array([0, 0, 1.0, 0, 0, 0, 0.3])
    Xk = b.var("x_0", NX, lbx, ubx, x0)
    lap_time = 0
    node_s = []
    for k in range(N):
        Uk = b.var(f"u_{k}", NU, [-5.0, -10.0], [5.0, 10.0], 0.0)
        Xc = [b.var(f"xc_{k}_{j}", NX, lbx, ubx, x0) for j in range(DEGREE)]
        Xend = Dend[0] * Xk
        for j in range(1, DEGREE + 1):
            xp = C[0, j] * Xk
            for r in range(DEGREE):
                xp += C[r + 1, j] * Xc[r]
            sj = (k + tau[j]) * ds
            fj, qj = f_sp(Xc[j - 1], Uk, kappa(sj))
            b.eq(ds * fj - xp)
            Xend += Dend[j] * Xc[j - 1]
            lap_time += Bq[j] * qj * ds
            for so, sl, no, nw in obs:
                if abs(sj - so) <= sl:
                    b.ineq((Xc[j - 1][0] - no) ** 2, (nw + CAR_HALF) ** 2, ca.inf)
            node_s.append(sj)
        Xk = b.var(f"x_{k + 1}", NX, lbx, ubx, x0)
        b.eq(Xend - Xk)
    # Periodic lap (flying lap): the end state equals the start state.
    first = b.w[0]
    b.eq(Xk - first)
    f = lap_time + 1e-4 * sum(ca.sumsqr(w) for w in b.w[1::DEGREE + 2])
    per = NX + NU + DEGREE * NX

    def check(w: np.ndarray) -> dict:
        n_vals = []
        for k in range(N):
            base = NX + k * per + NU
            for j in range(DEGREE):
                n_vals.append(w[base + j * NX])
        n_vals = np.array(n_vals)
        node_s_arr = np.array(node_s)
        worst = np.inf
        for so, sl, no, nw in obs:
            m = np.abs(node_s_arr - so) <= sl
            if m.any():
                worst = min(worst, float(np.min(np.abs(n_vals[m] - no)) - nw - CAR_HALF))
        return {"max_abs_n": float(np.max(np.abs(n_vals))), "obstacle_margin": worst}

    return b.problem("car_track", f, check, {"N": N, "seed": seed, "family": "car"})
