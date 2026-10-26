"""Hard nonconvex NLP corpus for IPOPT (IPM-mode work).

MODELS maps name -> (builder, arg) in the same shape as ipopt_kkt.MODELS;
ipopt_kkt.build_nl calls builder(arg), scaling arg only when it is numeric.

PGLib-OPF AC-OPF cases (v23.07) are built with Egret's polar-power-voltage
formulation (egret.models.acopf.create_psv_acopf_model) from the MATPOWER .m
files fetched by tools/bench/fetch_pglib.py into benchmarks/fixtures/pglib/.

Dependency: gridx-egret 0.6.2 (pip install gridx-egret into the smfbench env).

COPS 3.0 problems (cops_*) are plain Pyomo; their size arg is an int, so
ipopt_kkt --scale applies to them.
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
PGLIB_DIR = REPO / "benchmarks" / "fixtures" / "pglib"


def acopf_pglib(case_name: str):
    """Egret PSV AC-OPF Pyomo model for a PGLib-OPF case (e.g. pglib_opf_case118_ieee)."""
    from egret.data.model_data import ModelData
    from egret.models.acopf import create_psv_acopf_model

    path = PGLIB_DIR / f"{case_name}.m"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run tools/bench/fetch_pglib.py")
    model_data = ModelData.read(str(path))
    model, _md = create_psv_acopf_model(model_data)
    return model


PGLIB_CASES = [
    "pglib_opf_case118_ieee",
    "pglib_opf_case300_ieee",
    "pglib_opf_case500_goc__api",
    "pglib_opf_case1354_pegase",
    "pglib_opf_case2000_goc__sad",
    "pglib_opf_case2869_pegase",
    "pglib_opf_case4661_sdet",
    "pglib_opf_case9241_pegase",
]

MODELS = {name: (acopf_pglib, name) for name in PGLIB_CASES}


# ------------------------------------------------------------------- COPS --
# COPS 3.0 problems (Dolan, More & Munson, "Benchmarking Optimization Software
# with COPS 3.0", ANL/MCS-TM-273, 2004): formulations, discretizations and
# starting points as in the report, cross-checked against the JuMP ports in
# COPSBenchmark.jl. AMPL defined variables (collocation values, drag, lift, ...)
# are Pyomo Expressions, so they are substituted as in the AMPL models.
# Objective senses follow the report (rocket, glider, polygon maximize).

_GAUSS3 = (0.11270166537926, 0.5, 0.88729833462074)  # roots of the degree-3 Legendre polynomial on [0,1]


def cops_chain(nh: int):
    """Hanging chain (COPS 3.0 #4): minimize the potential energy x2(1) of a
    chain of length L between x(0)=a and x(1)=b; trapezoidal rule, nh intervals."""
    import pyomo.environ as pe
    L, a, b = 4.0, 1.0, 3.0
    tm = 0.25 if b > a else 0.75
    h = 1.0 / nh
    m = pe.ConcreteModel()
    K = range(nh + 1)
    t = [k / nh for k in K]
    x1_0 = [2 * abs(b - a) * tk * (tk - 2 * tm) + a for tk in t]
    u_0 = [4 * abs(b - a) * (tk - tm) for tk in t]
    m.u = pe.Var(K, initialize=lambda m, k: u_0[k])
    m.x1 = pe.Var(K, initialize=lambda m, k: x1_0[k])
    m.x2 = pe.Var(K, initialize=lambda m, k: x1_0[k] * u_0[k])
    m.x3 = pe.Var(K, initialize=lambda m, k: u_0[k])
    m.obj = pe.Objective(expr=m.x2[nh])
    J = range(nh)
    m.dx1 = pe.Constraint(J, rule=lambda m, j: m.x1[j + 1] - m.x1[j] == 0.5 * h * (m.u[j] + m.u[j + 1]))
    m.dx2 = pe.Constraint(J, rule=lambda m, j: m.x2[j + 1] - m.x2[j] == 0.5 * h * (
        m.x1[j] * pe.sqrt(1 + m.u[j] ** 2) + m.x1[j + 1] * pe.sqrt(1 + m.u[j + 1] ** 2)))
    m.dx3 = pe.Constraint(J, rule=lambda m, j: m.x3[j + 1] - m.x3[j] == 0.5 * h * (
        pe.sqrt(1 + m.u[j] ** 2) + pe.sqrt(1 + m.u[j + 1] ** 2)))
    m.x1_a = pe.Constraint(expr=m.x1[0] == a)
    m.x1_b = pe.Constraint(expr=m.x1[nh] == b)
    m.x2_0 = pe.Constraint(expr=m.x2[0] == 0.0)
    m.x3_0 = pe.Constraint(expr=m.x3[0] == 0.0)
    m.x3_L = pe.Constraint(expr=m.x3[nh] == L)
    return m


def cops_catmix(nh: int):
    """Catalyst mixing (COPS 3.0 #14): minimize -1 + x1(1) + x2(1); 3-stage
    collocation on nh intervals, control 0 <= u <= 1."""
    import pyomo.environ as pe
    from math import factorial
    nc, ne = 3, 2
    h = 1.0 / nh
    rho = _GAUSS3
    bc = (1.0, 0.0)
    m = pe.ConcreteModel()
    I, C, S = range(nh), range(nc), range(ne)
    m.u = pe.Var(I, C, bounds=(0.0, 1.0), initialize=0.0)
    m.v = pe.Var(I, S, initialize=lambda m, i, s: bc[s])
    m.w = pe.Var(I, C, S, initialize=0.0)
    # Collocation approximation p(t) = v_i + sum_k w_ik (t - t_i)^k / (k! h^(k-1))
    m.pp = pe.Expression(I, C, S, rule=lambda m, i, j, s: m.v[i, s] + h * sum(
        m.w[i, k, s] * rho[j] ** (k + 1) / factorial(k + 1) for k in C))
    m.Dpp = pe.Expression(I, C, S, rule=lambda m, i, j, s: sum(
        m.w[i, k, s] * rho[j] ** k / factorial(k) for k in C))
    m.ppf = pe.Expression(S, rule=lambda m, s: m.v[nh - 1, s] + h * sum(
        m.w[nh - 1, k, s] / factorial(k + 1) for k in C))
    m.obj = pe.Objective(expr=-1.0 + m.ppf[0] + m.ppf[1])
    m.bc = pe.Constraint(S, rule=lambda m, s: m.v[0, s] == bc[s])
    m.cont = pe.Constraint(range(nh - 1), S, rule=lambda m, i, s: m.v[i, s] + h * sum(
        m.w[i, k, s] / factorial(k + 1) for k in C) == m.v[i + 1, s])
    m.de1 = pe.Constraint(I, C, rule=lambda m, i, j: m.Dpp[i, j, 0] == m.u[i, j] * (
        10.0 * m.pp[i, j, 1] - m.pp[i, j, 0]))
    m.de2 = pe.Constraint(I, C, rule=lambda m, i, j: m.Dpp[i, j, 1] == m.u[i, j] * (
        m.pp[i, j, 0] - 10.0 * m.pp[i, j, 1]) - (1 - m.u[i, j]) * m.pp[i, j, 1])
    return m


def cops_rocket(nh: int):
    """Goddard rocket (COPS 3.0 #10): maximize final altitude h(tf) with thrust
    0 <= T <= Tmax; trapezoidal rule, nh intervals, free final time."""
    import pyomo.environ as pe
    h0, v0, m0, g0 = 1.0, 0.0, 1.0, 1.0
    Tc, hc, vc, mc = 3.5, 500.0, 620.0, 0.6
    c = 0.5 * (g0 * h0) ** 0.5
    mf = mc * m0
    Dc = 0.5 * vc * (m0 / g0)
    Tmax = Tc * m0 * g0
    m = pe.ConcreteModel()
    K = range(nh + 1)
    m.h = pe.Var(K, bounds=(h0, None), initialize=1.0)
    m.v = pe.Var(K, bounds=(0.0, None), initialize=lambda m, k: (k / nh) * (1.0 - k / nh))
    m.m = pe.Var(K, bounds=(mf, m0), initialize=lambda m, k: (mf - m0) * (k / nh) + m0)
    m.T = pe.Var(K, bounds=(0.0, Tmax), initialize=Tmax / 2.0)
    m.step = pe.Var(bounds=(0.0, None), initialize=1.0 / nh)  # tf / nh, tf = 1 initially
    m.D = pe.Expression(K, rule=lambda m, k: Dc * m.v[k] ** 2 * pe.exp(-hc * (m.h[k] - h0) / h0))
    m.g = pe.Expression(K, rule=lambda m, k: g0 * (h0 / m.h[k]) ** 2)
    m.dv = pe.Expression(K, rule=lambda m, k: (m.T[k] - m.D[k] - m.m[k] * m.g[k]) / m.m[k])
    m.obj = pe.Objective(expr=m.h[nh], sense=pe.maximize)
    J = range(1, nh + 1)
    m.ode_h = pe.Constraint(J, rule=lambda m, k: m.h[k] == m.h[k - 1] + 0.5 * m.step * (m.v[k] + m.v[k - 1]))
    m.ode_v = pe.Constraint(J, rule=lambda m, k: m.v[k] == m.v[k - 1] + 0.5 * m.step * (m.dv[k] + m.dv[k - 1]))
    m.ode_m = pe.Constraint(J, rule=lambda m, k: m.m[k] == m.m[k - 1] - 0.5 * m.step * (m.T[k] + m.T[k - 1]) / c)
    m.h_ic = pe.Constraint(expr=m.h[0] == h0)
    m.v_ic = pe.Constraint(expr=m.v[0] == v0)
    m.m_ic = pe.Constraint(expr=m.m[0] == m0)
    m.m_fc = pe.Constraint(expr=m.m[nh] == mf)
    return m


def cops_steering(nh: int):
    """Particle steering (COPS 3.0 #9): minimize the time tf to reach height 5
    and velocity (45, 0) under thrust a=100 at angle |u| <= pi/2; trapezoidal rule."""
    import pyomo.environ as pe
    from math import pi
    a = 100.0
    m = pe.ConcreteModel()
    K = range(nh + 1)
    start = {1: lambda k: 0.0, 2: lambda k: 5.0 * k / nh, 3: lambda k: 45.0 * k / nh, 4: lambda k: 0.0}
    m.u = pe.Var(K, bounds=(-pi / 2, pi / 2), initialize=0.0)
    m.y = pe.Var(K, range(1, 5), initialize=lambda m, k, j: start[j](k))
    m.tf = pe.Var(bounds=(0.0, None), initialize=1.0)
    h = m.tf / nh
    m.obj = pe.Objective(expr=m.tf)
    J = range(nh)
    m.ode1 = pe.Constraint(J, rule=lambda m, k: m.y[k + 1, 1] == m.y[k, 1] + 0.5 * h * (m.y[k, 3] + m.y[k + 1, 3]))
    m.ode2 = pe.Constraint(J, rule=lambda m, k: m.y[k + 1, 2] == m.y[k, 2] + 0.5 * h * (m.y[k, 4] + m.y[k + 1, 4]))
    m.ode3 = pe.Constraint(J, rule=lambda m, k: m.y[k + 1, 3] == m.y[k, 3] + 0.5 * h * (
        a * pe.cos(m.u[k]) + a * pe.cos(m.u[k + 1])))
    m.ode4 = pe.Constraint(J, rule=lambda m, k: m.y[k + 1, 4] == m.y[k, 4] + 0.5 * h * (
        a * pe.sin(m.u[k]) + a * pe.sin(m.u[k + 1])))
    m.ic = pe.Constraint(range(1, 5), rule=lambda m, j: m.y[0, j] == 0.0)
    yf = {2: 5.0, 3: 45.0, 4: 0.0}
    m.fc = pe.Constraint(list(yf), rule=lambda m, j: m.y[nh, j] == yf[j])
    return m


def cops_glider(nh: int):
    """Hang glider (COPS 3.0 #11): maximize the final range x(tf) through a
    thermal updraft, lift coefficient 0 <= cL <= 1.4; trapezoidal rule, free tf."""
    import pyomo.environ as pe
    x_0, y_0, y_f = 0.0, 1000.0, 900.0
    vx_0 = vx_f = 13.23
    vy_0 = vy_f = -1.288
    u_c, r_0, mass, g = 2.5, 100.0, 100.0, 9.81
    c0, c1, S, rho = 0.034, 0.069662, 14.0, 1.13
    cL_max = 1.4
    m = pe.ConcreteModel()
    K = range(nh + 1)
    m.t_f = pe.Var(bounds=(0.0, None), initialize=1.0)
    m.x = pe.Var(K, bounds=(0.0, None), initialize=lambda m, k: x_0 + vx_0 * k / nh)
    m.y = pe.Var(K, initialize=lambda m, k: y_0 + (k / nh) * (y_f - y_0))
    m.vx = pe.Var(K, bounds=(0.0, None), initialize=vx_0)
    m.vy = pe.Var(K, initialize=vy_0)
    m.cL = pe.Var(K, bounds=(0.0, cL_max), initialize=cL_max / 2.0)
    m.r = pe.Expression(K, rule=lambda m, k: (m.x[k] / r_0 - 2.5) ** 2)
    m.upd = pe.Expression(K, rule=lambda m, k: u_c * (1 - m.r[k]) * pe.exp(-m.r[k]))
    m.w = pe.Expression(K, rule=lambda m, k: m.vy[k] - m.upd[k])
    m.v = pe.Expression(K, rule=lambda m, k: pe.sqrt(m.vx[k] ** 2 + m.w[k] ** 2))
    m.D = pe.Expression(K, rule=lambda m, k: 0.5 * (c0 + c1 * m.cL[k] ** 2) * rho * S * m.v[k] ** 2)
    m.L = pe.Expression(K, rule=lambda m, k: 0.5 * m.cL[k] * rho * S * m.v[k] ** 2)
    m.ax = pe.Expression(K, rule=lambda m, k: (-m.L[k] * (m.w[k] / m.v[k]) - m.D[k] * (m.vx[k] / m.v[k])) / mass)
    m.ay = pe.Expression(K, rule=lambda m, k: (m.L[k] * (m.vx[k] / m.v[k]) - m.D[k] * (m.w[k] / m.v[k])) / mass - g)
    m.obj = pe.Objective(expr=m.x[nh], sense=pe.maximize)
    step = m.t_f / nh
    J = range(1, nh + 1)
    m.ode_x = pe.Constraint(J, rule=lambda m, k: m.x[k] == m.x[k - 1] + 0.5 * step * (m.vx[k] + m.vx[k - 1]))
    m.ode_y = pe.Constraint(J, rule=lambda m, k: m.y[k] == m.y[k - 1] + 0.5 * step * (m.vy[k] + m.vy[k - 1]))
    m.ode_vx = pe.Constraint(J, rule=lambda m, k: m.vx[k] == m.vx[k - 1] + 0.5 * step * (m.ax[k] + m.ax[k - 1]))
    m.ode_vy = pe.Constraint(J, rule=lambda m, k: m.vy[k] == m.vy[k - 1] + 0.5 * step * (m.ay[k] + m.ay[k - 1]))
    m.x_ic = pe.Constraint(expr=m.x[0] == x_0)
    m.y_ic = pe.Constraint(expr=m.y[0] == y_0)
    m.y_fc = pe.Constraint(expr=m.y[nh] == y_f)
    m.vx_ic = pe.Constraint(expr=m.vx[0] == vx_0)
    m.vx_fc = pe.Constraint(expr=m.vx[nh] == vx_f)
    m.vy_ic = pe.Constraint(expr=m.vy[0] == vy_0)
    m.vy_fc = pe.Constraint(expr=m.vy[nh] == vy_f)
    return m


def cops_elec(np: int):
    """Electrons on a sphere (COPS 3.0 #2, Thomson problem): minimize the
    Coulomb potential of np unit charges on the unit sphere. Many local minima.
    Start: seeded quasi-uniform random points on the sphere."""
    import random
    from math import cos, pi, sin

    import pyomo.environ as pe
    rng = random.Random(2713)
    theta = [2 * pi * rng.random() for _ in range(np)]
    phi = [pi * rng.random() for _ in range(np)]
    m = pe.ConcreteModel()
    I = range(np)
    m.x = pe.Var(I, initialize=lambda m, i: cos(theta[i]) * sin(phi[i]))
    m.y = pe.Var(I, initialize=lambda m, i: sin(theta[i]) * sin(phi[i]))
    m.z = pe.Var(I, initialize=lambda m, i: cos(phi[i]))
    m.obj = pe.Objective(expr=sum(
        1.0 / pe.sqrt((m.x[i] - m.x[j]) ** 2 + (m.y[i] - m.y[j]) ** 2 + (m.z[i] - m.z[j]) ** 2)
        for i in I for j in range(i + 1, np)))
    m.ball = pe.Constraint(I, rule=lambda m, i: m.x[i] ** 2 + m.y[i] ** 2 + m.z[i] ** 2 == 1.0)
    return m


def cops_polygon(nv: int):
    """Largest small polygon (COPS 3.0 #1): maximize the area of an nv-gon of
    diameter <= 1 in polar coordinates, last vertex fixed at the origin
    (r_nv = 0, theta_nv = pi). O(nv^2) nonconvex constraints, many local maxima."""
    import pyomo.environ as pe
    from math import pi
    m = pe.ConcreteModel()
    I = range(1, nv + 1)
    # Start (report: "a polygon with almost equal sides"), as in pgon.mod of
    # the AMPL model library the report's implementation follows (Gay, netlib).
    m.r = pe.Var(I, bounds=(0.0, 1.0), initialize=lambda m, i: 4 * i * (nv + 1 - i) / (nv + 1) ** 2)
    m.theta = pe.Var(I, bounds=(0.0, pi), initialize=lambda m, i: pi * i / nv)
    m.obj = pe.Objective(expr=0.5 * sum(
        m.r[i] * m.r[i + 1] * pe.sin(m.theta[i + 1] - m.theta[i]) for i in range(1, nv)), sense=pe.maximize)
    m.fix_theta = pe.Constraint(expr=m.theta[nv] == pi)
    m.fix_r = pe.Constraint(expr=m.r[nv] == 0.0)
    m.order = pe.Constraint(range(1, nv), rule=lambda m, i: m.theta[i] <= m.theta[i + 1])
    m.diameter = pe.Constraint(
        [(i, j) for i in range(1, nv) for j in range(i + 1, nv + 1)],
        rule=lambda m, i, j: m.r[i] ** 2 + m.r[j] ** 2 - 2 * m.r[i] * m.r[j] * pe.cos(m.theta[i] - m.theta[j]) <= 1.0)
    return m


def cops_pinene(nh: int):
    """Isomerization of alpha-pinene (COPS 3.0 #5): fit the 5 rate constants of
    a linear kinetic model to 8 measurements; 3-stage collocation on nh intervals."""
    import pyomo.environ as pe
    from math import factorial, floor
    nc, ne, npar, nm = 3, 5, 5, 8
    rho = _GAUSS3
    bc = (100.0, 0.0, 0.0, 0.0, 0.0)
    tau = (1230.0, 3060.0, 4920.0, 7800.0, 10680.0, 15030.0, 22620.0, 36420.0)
    z = ((88.35, 7.3, 2.3, 0.4, 1.75),
         (76.4, 15.6, 4.5, 0.7, 2.8),
         (65.1, 23.1, 5.3, 1.1, 5.8),
         (50.4, 32.9, 6.0, 1.5, 9.3),
         (37.5, 42.7, 6.0, 1.9, 12.0),
         (25.9, 49.1, 5.9, 2.2, 17.0),
         (14.0, 57.4, 5.1, 2.6, 21.0),
         (4.5, 63.1, 3.8, 2.9, 25.7))
    tf = tau[-1]
    h = tf / nh
    t = [i * h for i in range(nh + 1)]  # mesh t_0..t_nh (0-based intervals)
    # itau[j]: 0-based interval containing tau[j] (largest i with t_i <= tau_j, capped at nh-1)
    itau = [min(nh - 1, int(floor(tau[j] / h))) for j in range(nm)]
    # Start: piecewise-constant interpolation of the data.
    v0 = [[0.0] * ne for _ in range(nh)]
    for i in range(nh):
        if i <= itau[0]:
            v0[i] = list(bc)
        else:
            j = next((j for j in range(1, nm) if itau[j - 1] < i <= itau[j]), nm - 1)
            v0[i] = list(z[j])
    m = pe.ConcreteModel()
    I, C, S = range(nh), range(nc), range(ne)
    m.theta = pe.Var(range(npar), bounds=(0.0, None), initialize=0.0)
    m.v = pe.Var(I, S, initialize=lambda m, i, s: v0[i][s])
    m.w = pe.Var(I, C, S, initialize=0.0)
    m.uc = pe.Expression(I, C, S, rule=lambda m, i, j, s: m.v[i, s] + h * sum(
        m.w[i, k, s] * rho[j] ** (k + 1) / factorial(k + 1) for k in C))
    m.Duc = pe.Expression(I, C, S, rule=lambda m, i, j, s: sum(
        m.w[i, k, s] * rho[j] ** k / factorial(k) for k in C))
    err = [[m.v[itau[j], s] + sum(m.w[itau[j], k, s] * (tau[j] - t[itau[j]]) ** (k + 1)
                                  / (factorial(k + 1) * h ** k) for k in C) - z[j][s]
            for s in S] for j in range(nm)]
    m.obj = pe.Objective(expr=sum(err[j][s] ** 2 for j in range(nm) for s in S))
    m.bc = pe.Constraint(S, rule=lambda m, s: m.v[0, s] == bc[s])
    m.cont = pe.Constraint(range(nh - 1), S, rule=lambda m, i, s: m.v[i, s] + h * sum(
        m.w[i, k, s] / factorial(k + 1) for k in C) == m.v[i + 1, s])
    th, uc, Duc = m.theta, m.uc, m.Duc
    m.ode1 = pe.Constraint(I, C, rule=lambda m, i, j: Duc[i, j, 0] == -(th[0] + th[1]) * uc[i, j, 0])
    m.ode2 = pe.Constraint(I, C, rule=lambda m, i, j: Duc[i, j, 1] == th[0] * uc[i, j, 0])
    m.ode3 = pe.Constraint(I, C, rule=lambda m, i, j: Duc[i, j, 2] == th[1] * uc[i, j, 0]
                           - (th[2] + th[3]) * uc[i, j, 2] + th[4] * uc[i, j, 4])
    m.ode4 = pe.Constraint(I, C, rule=lambda m, i, j: Duc[i, j, 3] == th[2] * uc[i, j, 2])
    m.ode5 = pe.Constraint(I, C, rule=lambda m, i, j: Duc[i, j, 4] == th[3] * uc[i, j, 2] - th[4] * uc[i, j, 4])
    return m


COPS = {
    "cops_glider_400": (cops_glider, 400),
    "cops_rocket_800": (cops_rocket, 800),
    "cops_catmix_800": (cops_catmix, 800),
    "cops_chain_800": (cops_chain, 800),
    "cops_pinene_400": (cops_pinene, 400),
    "cops_steering_800": (cops_steering, 800),
    "cops_elec_200": (cops_elec, 200),
    "cops_polygon_200": (cops_polygon, 200),
}
MODELS.update(COPS)


# Models reserved for held-out evaluation; every other hard model is "tuning".
HELDOUT = frozenset({
    "pglib_opf_case300_ieee",
    "pglib_opf_case1354_pegase",
    "pglib_opf_case2000_goc__sad",
    "pglib_opf_case9241_pegase",
    "cops_rocket_800",
    "cops_polygon_200",
})


def split_of(name: str) -> str:
    return "heldout" if name in HELDOUT else "tuning"
