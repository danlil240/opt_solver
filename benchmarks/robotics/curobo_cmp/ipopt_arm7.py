"""IPOPT (+ smf) on the arm7 scenes, for the cuRobo comparison.

  python ipopt_arm7.py --form dyn --N 100 --seeds 0-9 --solver smf
  python ipopt_arm7.py --form kin --N 32  --seeds 0-9 --solver smf

form dyn: benchmarks/robotics/p_arm7 unchanged (torque inputs, RK4 multiple
          shooting, free final time). With ROBO_STAGE_JIT=1 the per-stage
          functions are compiled once and reused for every seed.
form kin: a formulation sized like cuRobo's (16 B-spline knots there): the joint positions are a
          piecewise-cubic (jerk input, triple integrator) with N intervals,
          the same limits cuRobo's franka.yml uses (velocity, acceleration
          15 rad/s^2, jerk 500 rad/s^3), the arm7 torque limits through the
          arm7 point-mass model at the knots, and the arm7 collision
          constraints at the knots. Minimum time.

Every solution is checked densely (10 samples per interval for kin) with
arm7_model.check, the same check the cuRobo trajectories get. One JSON record
per seed on stdout (prefixed RESULT) and appended to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("ROBO_DIR", str(HERE.parent)))
sys.path.insert(0, str(HERE))

import casadi as ca  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
import p_arm7  # noqa: E402
import arm7_model as am  # noqa: E402

ACC_LIM, JERK_LIM = 15.0, 500.0  # cuRobo franka.yml
NQ = 7

_cache: dict = {}
_orig_stage_function = common.stage_function


def _cached_stage_function(name, inputs, outputs):
    # The per-stage functions do not depend on the seed: compile them once.
    if name not in _cache:
        _cache[name] = _orig_stage_function(name, inputs, outputs)
    return _cache[name]


common.stage_function = _cached_stage_function
p_arm7.stage_function = _cached_stage_function


def inverse_dynamics():
    q, qd, qdd = ca.SX.sym("q", NQ), ca.SX.sym("qd", NQ), ca.SX.sym("qdd", NQ)
    origins, axes = p_arm7.frames(q)
    V = sum(p_arm7.MASS[i] * 9.81 * origins[i + 1][2] for i in range(NQ))
    grav = ca.gradient(V, q)
    diag = []
    for i in range(NQ):
        o, z = origins[i], axes[i]
        m = p_arm7.ROTOR[i]
        for j in range(i, NQ):
            r = origins[j + 1] - o
            perp = r - ca.dot(r, z) * z
            m += p_arm7.MASS[j] * ca.sumsqr(perp)
        diag.append(m)
    tau = ca.vertcat(*diag) * qdd + grav + p_arm7.FRICTION * qd
    return ca.Function("invdyn", [q, qd, qdd], [tau])


def sphere_centres():
    """CasADi function q -> centres (3 x n) of cuRobo's moving collision spheres."""
    q = ca.SX.sym("q", NQ)
    T = ca.SX.eye(4)
    frames = [T]
    for i, (a, d, al) in enumerate(p_arm7.DH):
        T = T @ p_arm7.tf(a, d, al, q[i])
        frames.append(T)
    sph = [s for s in am.load_spheres() if s["frame"] > 0]
    cols = [frames[s["frame"]][:3, :3] @ ca.DM(s["offset"]) + frames[s["frame"]][:3, 3] for s in sph]
    return ca.Function("spheres", [q], [ca.horzcat(*cols)]), np.array([s["radius"] for s in sph])


def build_kin(N: int, seed: int, geom: str = "arm7"):
    qs, qg, r = am.scene(seed)
    _, P = p_arm7.make_dynamics()
    ID = inverse_dynamics()
    if geom == "curobo":
        SPH, RAD = sphere_centres()
    b = common.Builder()
    T = b.var("T", 1, 0.3, 10.0, 2.0)
    h = T / N
    X, U = [], []
    for k in range(N + 1):
        s = k / N
        g = np.r_[qs + s * (qg - qs), np.zeros(2 * NQ)]
        lb = np.r_[am.Q_LO, -am.QD_LIM, -ACC_LIM * np.ones(NQ)]
        ub = np.r_[am.Q_LIM, am.QD_LIM, ACC_LIM * np.ones(NQ)]
        if k == 0:
            lb = ub = np.r_[qs, np.zeros(2 * NQ)]
        if k == N:
            lb = ub = np.r_[qg, np.zeros(2 * NQ)]
        X.append(b.var(f"x_{k}", 3 * NQ, lb, ub, g))
        if k < N:
            U.append(b.var(f"j_{k}", NQ, -JERK_LIM, JERK_LIM, 0.0))
    f = T
    n_pts = P.size_out(0)[1]
    for k in range(N):
        q, qd, qdd = X[k][:NQ], X[k][NQ:2 * NQ], X[k][2 * NQ:]
        j = U[k]
        nxt = ca.vertcat(q + h * qd + h ** 2 / 2 * qdd + h ** 3 / 6 * j,
                         qd + h * qdd + h ** 2 / 2 * j, qdd + h * j)
        b.eq(nxt - X[k + 1])
        f += 1e-7 * h * ca.sumsqr(j)
    for k in range(N + 1):
        q, qd, qdd = X[k][:NQ], X[k][NQ:2 * NQ], X[k][2 * NQ:]
        b.ineq(ID(q, qd, qdd), -am.TAU_LIM, am.TAU_LIM)
        if 0 < k < N and geom == "arm7":
            pts = P(q)
            for jj in range(2, n_pts):
                b.ineq(ca.sumsqr(pts[:, jj] - p_arm7.OBST), (r + 0.06) ** 2, ca.inf)
                b.ineq(pts[2, jj], 0.05, ca.inf)
        elif 0 < k < N:
            C = SPH(q)
            for jj in range(C.shape[1]):
                b.ineq(ca.sumsqr(C[:, jj] - p_arm7.OBST), (r + RAD[jj]) ** 2, ca.inf)
                b.ineq(C[2, jj] - RAD[jj], am.TABLE_TOP, ca.inf)
    per = 3 * NQ + NQ

    def dense(w):
        Tv = float(w[0])
        hv = Tv / N
        Q, QD, QDD = [], [], []
        for k in range(N):
            x = w[1 + k * per: 1 + k * per + 3 * NQ]
            jv = w[1 + k * per + 3 * NQ: 1 + (k + 1) * per]
            q, qd, qdd = x[:NQ], x[NQ:2 * NQ], x[2 * NQ:]
            for t in np.linspace(0, hv, 10, endpoint=False):
                Q.append(q + t * qd + t ** 2 / 2 * qdd + t ** 3 / 6 * jv)
                QD.append(qd + t * qdd + t ** 2 / 2 * jv)
                QDD.append(qdd + t * jv)
        xN = w[1 + N * per: 1 + N * per + 3 * NQ]
        Q.append(xN[:NQ]); QD.append(xN[NQ:2 * NQ]); QDD.append(xN[2 * NQ:])
        return Tv, np.array(Q), np.array(QD), np.array(QDD)

    prob = b.problem("arm7_kin", f, lambda w: {}, {"N": N, "seed": seed, "geom": geom})
    return prob, dense, r


def build_dyn(N: int, seed: int):
    prob = p_arm7.build(N, seed)
    _, _, r = am.scene(seed)
    per = 3 * NQ
    fdyn, _ = p_arm7.make_dynamics()

    def dense(w):
        # knots only: the torques are piecewise constant inputs; qdd from the model.
        Tv = float(w[0])
        Q, QD, QDD = [], [], []
        for k in range(N + 1):
            x = w[1 + k * per: 1 + k * per + 2 * NQ]
            u = w[1 + k * per + 2 * NQ: 1 + (k + 1) * per] if k < N else np.zeros(NQ)
            Q.append(x[:NQ]); QD.append(x[NQ:])
            QDD.append(np.array(fdyn(x, u)).ravel()[NQ:])
        return Tv, np.array(Q), np.array(QD), np.array(QDD)

    return prob, dense, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=["dyn", "kin"], default="kin")
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--solver", default="smf")
    ap.add_argument("--out", default=str(HERE / "ipopt_arm7.jsonl"))
    ap.add_argument("--save", default="")
    ap.add_argument("--geom", choices=["arm7", "curobo"], default="arm7",
                    help="collision model for --form kin: arm7 link points + 6 cm, or cuRobo's spheres")
    a = ap.parse_args()
    common.ensure_runner_env(1)
    lo, hi = (int(v) for v in a.seeds.split("-"))
    for seed in range(lo, hi + 1):
        t0 = time.perf_counter()
        prob, dense, r = (build_kin(a.N, seed, a.geom) if a.form == "kin" else build_dyn(a.N, seed))
        build_s = time.perf_counter() - t0
        got = {}
        own_check = prob.check
        prob.check = lambda w: (got.__setitem__("w", np.array(w, dtype=float)), own_check(w))[1]
        rec = common.solve(prob, a.solver, 1)
        rec.update({"form": a.form, "build_s_total": build_s, "stage_jit": common.stage_jit()})
        if "w" in got:
            w = got["w"]
            Tv, Q, QD, QDD = dense(w)
            rec["duration_s"] = Tv
            rec["path_len_rad"] = float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1)))
            rec.update({"arm7_" + k: v for k, v in am.check(Q, QD, QDD, r).items()})
            rec.update({"curobo_" + k: v for k, v in am.sphere_check(Q, r).items()})
            if a.save:
                np.save(f"{a.save}_{a.form}_{a.geom}_N{a.N}_s{seed}.npy", Q)
        line = json.dumps(rec, default=float)
        print("RESULT " + line, flush=True)
        with open(a.out, "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
