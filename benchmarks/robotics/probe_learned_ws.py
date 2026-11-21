"""Learned warm-start probe: can a guess predicted from the task (start and
goal) cut IPOPT's iterations on a new instance, and does a small neural net
beat simply reusing the nearest stored solution? Measurement only.

  python probe_learned_ws.py run --problem truck --N 100 --ntrain 240 --ntest 60 \
      --workers 8 --solver smf --out results/learned_ws_truck
  python probe_learned_ws.py summary results/learned_ws_truck

The environment (obstacles) is fixed; each instance changes only the task:
  truck  start pose of the tractor-trailer in the yard (x, y, heading),
         at rest, aligned; it still has to back into the same bay.
  quad   start and goal (y, z) of the quadrotor across the same forest.
The task enters only through bounds (the pinned start and goal states), so
each worker builds the NLP once and reuses it for every solve.

Phases:
  cold   every instance (train and test) from a hand-made guess built the
         same way as the problem's own (truck: drive past the bay then
         reverse in; quad: straight line), with IPOPT's default options.
  train  kNN library = the successful train solutions (primal and duals);
         MLP = 2 hidden layers of 256 tanh units, numpy, Adam, task -> primal.
  warm   every test instance from each method's guess:
           knn_primal  nearest train solution's primal, default options
           knn_warm    nearest train solution's primal and duals,
                       warm_start_init_point with mu_init 1e-4
           mlp_primal  MLP primal, default options
           mlp_warm    MLP primal with the nearest solution's duals, warm options
Workers run as separate processes in runner_env(1) (one thread each,
SMF_SLACK_ELIM=auto), at below-normal priority on Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BELOW_NORMAL = 0x00004000
METHODS = ("knn_primal", "knn_warm", "mlp_primal", "mlp_warm")
WARM = {"ipopt.warm_start_init_point": "yes", "ipopt.mu_init": 1e-4,
        "ipopt.warm_start_bound_push": 1e-6, "ipopt.warm_start_bound_frac": 1e-6,
        "ipopt.warm_start_slack_bound_push": 1e-6, "ipopt.warm_start_slack_bound_frac": 1e-6,
        "ipopt.warm_start_mult_bound_push": 1e-6}


# ---------------------------------------------------------------- problems

class Truck:
    name = "truck_trailer"
    lo = np.array([-28.0, -1.0, -np.pi / 3])
    hi = np.array([22.0, 8.0, np.pi / 3])

    def __init__(self, N: int):
        import p_truck_trailer as P
        self.P, self.N = P, N
        self.prob = P.build(N, 0)
        self.obs = np.array(P.obstacle_set(0))
        self.stride = P.NX + P.NU

    def sample(self, rng) -> np.ndarray:
        while True:
            p = rng.uniform(self.lo, self.hi)
            if self._clear(p) > 0.5:
                return p

    def _clear(self, p) -> float:
        x = np.array([p[0], p[1], p[2], p[2], 0, 0])
        worst = np.inf
        for cx, cy, cr in self.P.body_circles(x):
            d = np.hypot(float(cx) - self.obs[:, 0], float(cy) - self.obs[:, 1]) - cr - self.obs[:, 2]
            worst = min(worst, float(d.min()))
        return worst

    def bounds(self, p):
        lbx, ubx = self.prob.lbx.copy(), self.prob.ubx.copy()
        s = np.array([p[0], p[1], p[2], p[2], 0.0, 0.0])
        lbx[1:7] = ubx[1:7] = s
        return lbx, ubx

    def cold_guess(self, p) -> np.ndarray:
        """The problem's own two-leg guess, from this start."""
        x0 = self.prob.x0.copy()
        N, P = self.N, self.P
        for k in range(N + 1):
            s = k / N
            if s < 0.5:
                t = 2 * s
                g = [p[0] + (18.0 - p[0]) * t, p[1] + (1.5 - p[1]) * t, p[2] * (1 - t), p[2] * (1 - t), 1.0, 0]
            else:
                t = (s - 0.5) * 2
                g = [18.0 * (1 - t) + P.BAY_X * t, 1.5 + 6.0 * np.sin(np.pi * t) + (P.BAY_Y + 6.0) * t,
                     t * np.pi / 2, t * np.pi / 2, -1.0, 0]
            o = 1 + k * self.stride
            x0[o:o + P.NX] = g
        x0[1:7] = [p[0], p[1], p[2], p[2], 0, 0]
        return x0


class Quad:
    name = "quad_forest"
    lo = np.array([-3.5, 0.8, -3.5, 0.8])
    hi = np.array([3.5, 3.0, 3.5, 3.0])

    def __init__(self, N: int):
        import p_quad_forest as P
        import quadrotor as Q
        self.P, self.Q, self.N = P, Q, N
        self.prob = P.build(N, 0)
        self.stride = Q.NX + Q.NU

    def sample(self, rng) -> np.ndarray:
        return rng.uniform(self.lo, self.hi)

    def bounds(self, p):
        lbx, ubx = self.prob.lbx.copy(), self.prob.ubx.copy()
        lbx[1:13] = ubx[1:13] = np.r_[0.0, p[0], p[1], np.zeros(9)]
        o = 1 + self.N * self.stride
        lbx[o:o + 3] = ubx[o:o + 3] = [self.P.LENGTH, p[2], p[3]]
        return lbx, ubx

    def cold_guess(self, p) -> np.ndarray:
        x0 = self.prob.x0.copy()
        a, b = np.array([0.0, p[0], p[1]]), np.array([self.P.LENGTH, p[2], p[3]])
        for k in range(self.N + 1):
            o = 1 + k * self.stride
            x0[o:o + 3] = a + (k / self.N) * (b - a)
        return x0


class QuadObs(Quad):
    """Quadrotor across a forest whose obstacles are NLP parameters, so one
    build serves every task. Task = start/goal (y, z) then obstacle (x, y)
    pairs, sorted by x so similar layouts have similar vectors.
      quadmov  the fixed seed-0 forest plus 4 movable obstacles (r 0.5)
      quadrand a fresh 24-tree layout every task (radii of the seed-0 forest)
    The lookup and the MLP see start/goal plus, for quadrand, a blurred
    occupancy grid of the field (a raw position list is order-dependent)."""

    n_fixed, n_mov = 24, 4

    def __init__(self, N: int):
        import casadi as ca
        import p_quad_forest as P
        import quadrotor as Q
        from common import Builder, rk4
        self.P, self.Q, self.N = P, Q, N
        self.stride = Q.NX + Q.NU
        base = P.forest(0)
        self.fixed = base if self.n_fixed else np.zeros((0, 3))
        self.radii = self.obstacle_radii()
        n_obs = len(self.radii)
        self.p = ca.SX.sym("obs", 2 * n_obs)
        self.lo = np.r_[Quad.lo, np.tile([3.0, -4.0], self.n_mov)]
        self.hi = np.r_[Quad.hi, np.tile([P.LENGTH - 3.0, 4.0], self.n_mov)]
        start, goal = np.array([0, 0, 1.5]), np.array([P.LENGTH, 0, 1.5])
        b = Builder()
        T = b.var("T", 1, 2.0, 60.0, 12.0)
        h = T / N
        xs, us, hs = ca.SX.sym("xs", Q.NX), ca.SX.sym("us", Q.NU), ca.SX.sym("hs")
        F = ca.Function("F", [xs, us, hs], [rk4(Q.dynamics, xs, us, hs)])
        X, U, energy = [], [], 0
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
            for j in range(n_obs):
                b.ineq((X[k][0] - self.p[2 * j]) ** 2 + (X[k][1] - self.p[2 * j + 1]) ** 2,
                       (self.radii[j] + P.MARGIN) ** 2, ca.inf)
        self.prob = b.problem(self.name, T + 0.05 * energy, lambda w: {}, {"N": N})

    def obstacle_radii(self) -> np.ndarray:
        return np.r_[self.fixed[:, 2], 0.5 * np.ones(self.n_mov)]

    def sample(self, rng) -> np.ndarray:
        mov = np.array(sorted(rng.uniform([3.0, -4.0], [self.P.LENGTH - 3.0, 4.0], (self.n_mov, 2)).tolist()))
        return np.r_[rng.uniform(Quad.lo, Quad.hi), mov.ravel()]

    def pvec(self, p) -> np.ndarray:
        return np.r_[self.fixed[:, :2].ravel(), p[4:]]


class QuadMov(QuadObs):
    name = "quad_mov4"


class QuadRand(QuadObs):
    name = "quad_rand"
    n_fixed, n_mov = 0, 24

    def obstacle_radii(self) -> np.ndarray:
        return self.P.forest(0)[:, 2].copy()

    def sample(self, rng) -> np.ndarray:
        r = self.radii
        while True:
            trees = []
            ok = True
            for j in range(len(r)):
                for _ in range(200):
                    c = rng.uniform([3.0, -4.0], [self.P.LENGTH - 3.0, 4.0])
                    if all(np.hypot(*(c - t[:2])) > r[j] + t[2] + 2 * self.P.MARGIN + 0.4 for t in trees):
                        trees.append(np.r_[c, r[j]])
                        break
                else:
                    ok = False
                    break
            if ok:
                break
        xy = np.array([t[:2] for t in trees])  # index j keeps radius j
        return np.r_[rng.uniform(Quad.lo, Quad.hi), xy.ravel()]

    def features(self, p) -> np.ndarray:
        sg = 2 * (p[:4] - Quad.lo) / (Quad.hi - Quad.lo) - 1
        gx, gy = np.meshgrid(np.linspace(0, self.P.LENGTH, 31), np.linspace(-4.5, 4.5, 10))
        xy = p[4:].reshape(-1, 2)
        occ = np.zeros_like(gx)
        for (x, y), r in zip(xy, self.radii):
            occ += np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * (r + 0.5) ** 2))
        return np.r_[3 * sg, np.minimum(occ, 1.0).ravel()]


def features(T):
    if hasattr(T, "features"):
        return T.features
    return lambda p: 2 * (p - T.lo) / (T.hi - T.lo) - 1


PROBLEMS = {"truck": Truck, "quad": Quad, "quadmov": QuadMov, "quadrand": QuadRand}


# ---------------------------------------------------------------- worker

def worker(a) -> None:
    sys.path.insert(0, str(HERE))
    import casadi as ca
    import common
    common.ensure_runner_env(1)
    T = PROBLEMS[a.problem](a.N)
    prob = T.prob
    d = pathlib.Path(a.dir)
    params = np.load(d / "params.npy")
    jobs = json.loads(pathlib.Path(a.jobs).read_text())
    fd, out_file = tempfile.mkstemp(prefix="ipopt_lws_", suffix=".txt")
    os.close(fd)
    nlp = {"x": prob.x, "f": prob.f, "g": prob.g}
    if getattr(T, "p", None) is not None:
        nlp["p"] = T.p
    quiet = {"ipopt.file_print_level": 0, "ipopt.print_timing_statistics": "no"}
    solvers = {}

    def get(warm: bool):
        if warm not in solvers:
            solvers[warm] = ca.nlpsol("S", "ipopt", nlp,
                                      common.nlp_options(a.solver, out_file, {**quiet, **(WARM if warm else {})}))
        return solvers[warm]

    for i, method in jobs:
        p = params[i]
        lbx, ubx = T.bounds(p)
        args = dict(lbx=lbx, ubx=ubx, lbg=prob.lbg, ubg=prob.ubg)
        if "p" in nlp:
            args["p"] = T.pvec(p)
        warm = method.endswith("_warm")
        if method == "cold":
            args["x0"] = T.cold_guess(p)
        else:
            gz = np.load(d / f"guess_{method}_{i}.npz")
            args["x0"] = gz["x"]
            if warm:
                args.update(lam_x0=gz["lam_x"], lam_g0=gz["lam_g"])
        S = get(warm)
        t0 = time.perf_counter()
        r = S(**args)
        wall = time.perf_counter() - t0
        st = S.stats()
        xs = np.array(r["x"]).ravel()
        gv = np.array(r["g"]).ravel()
        viol = float(max(0.0, np.max(np.r_[prob.lbg - gv, gv - prob.ubg, lbx - xs, xs - ubx, 0.0])))
        rec = {"id": int(i), "method": method, "status": st.get("return_status"), "success": bool(st.get("success")),
               "iter": int(st.get("iter_count") or 0), "wall_s": wall, "obj": float(r["f"]), "max_viol": viol}
        np.savez(d / f"sol_{method}_{i}.npz", x=xs, lam_x=np.array(r["lam_x"]).ravel(),
                 lam_g=np.array(r["lam_g"]).ravel())
        with open(d / f"rec_{os.getpid()}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    os.remove(out_file)


def run_jobs(a, d: pathlib.Path, jobs: list, tag: str) -> None:
    sys.path.insert(0, str(HERE))
    import common
    env = common.runner_env(1)
    flags = BELOW_NORMAL if os.name == "nt" else 0
    chunks = [jobs[k::a.workers] for k in range(a.workers)]
    procs = []
    for k, ch in enumerate(chunks):
        if not ch:
            continue
        jf = d / f"jobs_{tag}_{k}.json"
        jf.write_text(json.dumps(ch))
        cmd = [sys.executable, str(HERE / "probe_learned_ws.py"), "worker", "--problem", a.problem,
               "--N", str(a.N), "--solver", a.solver, "--dir", str(d), "--jobs", str(jf)]
        log = open(d / f"log_{tag}_{k}.txt", "w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=flags), log))
    for p, log in procs:
        p.wait()
        log.close()


def records(d: pathlib.Path) -> list[dict]:
    return [json.loads(l) for f in sorted(d.glob("rec_*.jsonl")) for l in open(f, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------- learning

class MLP:
    """task -> standardized primal, tanh, two hidden layers, Adam, full batch."""

    def __init__(self, din: int, dout: int, h: int = 256, seed: int = 0):
        r = np.random.default_rng(seed)
        self.W = [r.normal(0, 1 / np.sqrt(din), (din, h)), r.normal(0, 1 / np.sqrt(h), (h, h)),
                  r.normal(0, 1 / np.sqrt(h), (h, dout))]
        self.b = [np.zeros(h), np.zeros(h), np.zeros(dout)]

    def forward(self, X):
        h1 = np.tanh(X @ self.W[0] + self.b[0])
        h2 = np.tanh(h1 @ self.W[1] + self.b[1])
        return h1, h2, h2 @ self.W[2] + self.b[2]

    def fit(self, X, Y, epochs=4000, lr=2e-3, wd=1e-5):
        ps = self.W + self.b
        m = [np.zeros_like(p) for p in ps]
        v = [np.zeros_like(p) for p in ps]
        n = len(X)
        for t in range(1, epochs + 1):
            h1, h2, out = self.forward(X)
            e = (out - Y) / n
            gW2 = h2.T @ e; gb2 = e.sum(0)
            d2 = (e @ self.W[2].T) * (1 - h2 ** 2)
            gW1 = h1.T @ d2; gb1 = d2.sum(0)
            d1 = (d2 @ self.W[1].T) * (1 - h1 ** 2)
            gW0 = X.T @ d1; gb0 = d1.sum(0)
            gs = [gW0 + wd * self.W[0], gW1 + wd * self.W[1], gW2 + wd * self.W[2], gb0, gb1, gb2]
            for j, (p, g) in enumerate(zip(ps, gs)):
                m[j] = 0.9 * m[j] + 0.1 * g
                v[j] = 0.999 * v[j] + 0.001 * g * g
                p -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
        return float(np.mean((self.forward(X)[2] - Y) ** 2))


def train_and_guess(a, d: pathlib.Path, T, params, train, test) -> dict:
    recs = {r["id"]: r for r in records(d) if r["method"] == "cold"}
    ok = [i for i in train if recs.get(i, {}).get("success")]
    norm = features(T)
    Xtr = np.array([norm(p) for p in params[ok]])
    sols = {i: np.load(d / f"sol_cold_{i}.npz") for i in ok}
    Ytr = np.array([sols[i]["x"] for i in ok])
    mu, sd = Ytr.mean(0), Ytr.std(0)
    sd[sd < 1e-9] = 1.0
    t0 = time.perf_counter()
    net = MLP(Xtr.shape[1], Ytr.shape[1])
    mse = net.fit(Xtr, (Ytr - mu) / sd, epochs=a.epochs)
    train_s = time.perf_counter() - t0
    info = {"train_ok": len(ok), "train_total": len(train), "mlp_train_mse_std": mse, "mlp_train_s": train_s}
    t_inf = []
    for i in test:
        q = norm(params[i])[None, :]
        t1 = time.perf_counter()
        xm = net.forward(q)[2][0] * sd + mu
        t_inf.append(time.perf_counter() - t1)
        j = ok[int(np.argmin(np.sum((Xtr - q) ** 2, axis=1)))]
        nb = sols[j]
        lbx, ubx = T.bounds(params[i])
        for meth, x in (("knn", nb["x"]), ("mlp", xm)):
            x = np.clip(x, lbx, ubx)  # pins the new start/goal; IPOPT pushes off bounds itself
            for kind in ("primal", "warm"):
                np.savez(d / f"guess_{meth}_{kind}_{i}.npz", x=x, lam_x=nb["lam_x"], lam_g=nb["lam_g"])
    info["mlp_infer_ms_median"] = 1e3 * float(np.median(t_inf))
    # How far the held-out guesses are from the cold solution (when it exists), in sd units.
    errs = {"knn": [], "mlp": []}
    for i in test:
        f = d / f"sol_cold_{i}.npz"
        if f.exists() and recs.get(i, {}).get("success"):
            xc = np.load(f)["x"]
            for meth in errs:
                xg = np.load(d / f"guess_{meth}_primal_{i}.npz")["x"]
                errs[meth].append(float(np.sqrt(np.mean(((xg - xc) / sd) ** 2))))
    info.update({f"{k}_guess_rms_sd": float(np.median(v)) if v else None for k, v in errs.items()})
    return info


def neighbors(a) -> None:
    """Warm starts from the 2nd..Kth nearest library solutions on an existing
    run's held-out tasks (methods knn2_warm, ...; knn_warm is the 1st), then
    the parallel nearest-K table."""
    d = pathlib.Path(a.dir)
    meta = json.loads((d / "meta.json").read_text())
    key = meta["problem"]
    T = PROBLEMS[key](meta["N"])
    params = np.load(d / "params.npy")
    train = range(meta["ntrain"])
    test = list(range(meta["ntrain"], meta["ntrain"] + meta["ntest"]))
    cold = {r["id"]: r for r in records(d) if r["method"] == "cold"}
    ok = [i for i in train if cold.get(i, {}).get("success")]
    norm = features(T)
    Xtr = np.array([norm(p) for p in params[ok]])
    done = {(r["id"], r["method"]) for r in records(d)}
    jobs = []
    for i in test:
        order = np.argsort(np.sum((Xtr - norm(params[i])) ** 2, axis=1))
        lbx, ubx = T.bounds(params[i])
        for k in range(2, a.k + 1):
            m = f"knn{k}_warm"
            nb = np.load(d / f"sol_cold_{ok[int(order[k - 1])]}.npz")
            np.savez(d / f"guess_{m}_{i}.npz", x=np.clip(nb["x"], lbx, ubx), lam_x=nb["lam_x"], lam_g=nb["lam_g"])
            if (i, m) not in done:
                jobs.append((i, m))
    run_args = argparse.Namespace(problem=key, N=meta["N"], solver=meta["solver"], workers=a.workers)
    run_jobs(run_args, d, jobs, "nbr")
    parallel_table(d, test, a.k)


def parallel_table(d: pathlib.Path, test: list, K: int) -> None:
    """race: K solves start together, the first success wins (wall = its wall).
    best: all K finish, the lowest objective is kept (wall = slowest)."""
    R = {(r["id"], r["method"]): r for r in records(d)}
    names = ["knn_warm"] + [f"knn{k}_warm" for k in range(2, K + 1)]
    rows = {}
    for kk in range(1, K + 1):
        race_w, best_w, race_it, fails, ro_race, ro_best, cw = [], [], [], 0, [], [], []
        for i in test:
            c = R.get((i, "cold"))
            rs = [R.get((i, m)) for m in names[:kk]]
            good = [r for r in rs if r and r["success"]]
            if not good:
                fails += 1
                continue
            first = min(good, key=lambda r: r["wall_s"])
            best = min(good, key=lambda r: r["obj"])
            race_w.append(first["wall_s"]); race_it.append(first["iter"])
            best_w.append(max(r["wall_s"] for r in rs if r))
            if c and c["success"]:
                cw.append(c["wall_s"])
                ro_race.append((first["obj"] - c["obj"]) / abs(c["obj"]))
                ro_best.append((best["obj"] - c["obj"]) / abs(c["obj"]))
            else:
                cw.append(np.nan)
        cw = np.array(cw)
        m = ~np.isnan(cw)
        rows[kk] = dict(fails=fails, race_med=np.median(race_w), race_it=np.median(race_it),
                        race_tot=np.nansum(cw) / np.sum(np.array(race_w)[m]), race_p90=np.percentile(race_w, 90),
                        race_max=max(race_w), best_tot=np.nansum(cw) / np.sum(np.array(best_w)[m]),
                        best_p90=np.percentile(best_w, 90),
                        race_worse=sum(x > 0.01 for x in ro_race), race_better=sum(x < -0.01 for x in ro_race),
                        best_worse=sum(x > 0.01 for x in ro_best), best_better=sum(x < -0.01 for x in ro_best))
    c = [R[(i, "cold")]["wall_s"] for i in test if (i, "cold") in R and R[(i, "cold")]["success"]]
    print(f"cold: median {np.median(c):.2f} s, p90 {np.percentile(c, 90):.2f} s, max {max(c):.2f} s")
    print("| K | unsolved | race median iters | race median s | race p90 s | race max s | race total speedup | "
          "race obj worse/better >1% | best-of-K p90 s | best-of-K total speedup | best obj worse/better >1% |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|")
    for kk, r in rows.items():
        print(f"| {kk} | {r['fails']} | {r['race_it']:.0f} | {r['race_med']:.2f} | {r['race_p90']:.2f} | {r['race_max']:.2f} | "
              f"{r['race_tot']:.2f}x | {r['race_worse']}/{r['race_better']} | {r['best_p90']:.2f} | {r['best_tot']:.2f}x | "
              f"{r['best_worse']}/{r['best_better']} |")


# ---------------------------------------------------------------- driver

def run(a) -> None:
    sys.path.insert(0, str(HERE))
    d = pathlib.Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    T = PROBLEMS[a.problem](a.N)
    rng = np.random.default_rng(a.seed)
    n = a.ntrain + a.ntest
    params = np.array([T.sample(rng) for _ in range(n)])
    np.save(d / "params.npy", params)
    train, test = list(range(a.ntrain)), list(range(a.ntrain, n))
    meta = {"problem": a.problem, "N": a.N, "solver": a.solver, "ntrain": a.ntrain, "ntest": a.ntest,
            "workers": a.workers, "seed": a.seed, "nx": int(T.prob.x.shape[0]), "ng": int(T.prob.g.shape[0])}
    t0 = time.perf_counter()
    run_jobs(a, d, [(i, "cold") for i in range(n)], "cold")
    meta["cold_phase_s"] = time.perf_counter() - t0
    meta.update(train_and_guess(a, d, T, params, train, test))
    t0 = time.perf_counter()
    run_jobs(a, d, [(i, m) for i in test for m in METHODS], "warm")
    meta["warm_phase_s"] = time.perf_counter() - t0
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    summary_dir(d)


def summary_dir(d: pathlib.Path) -> None:
    d = pathlib.Path(d)
    meta = json.loads((d / "meta.json").read_text())
    recs = records(d)
    test = set(range(meta["ntrain"], meta["ntrain"] + meta["ntest"]))
    by = {}
    for r in recs:
        if r["id"] in test:
            by.setdefault(r["method"], {})[r["id"]] = r
    cold = by.get("cold", {})
    print(f"## {meta['problem']} N={meta['N']} ({meta['solver']}), {meta['ntest']} held-out tasks, "
          f"library {meta['train_ok']}/{meta['train_total']} cold train solves succeeded")
    print()
    print("| start | solved | median iters | mean iters | median wall s | iters vs cold (median ratio) | wall vs cold (median ratio) | obj vs cold (median) | worse local opt >1% |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in ("cold",) + METHODS:
        rs = by.get(m, {})
        if not rs:
            continue
        ok = [r for r in rs.values() if r["success"]]
        it = [r["iter"] for r in ok]
        wl = [r["wall_s"] for r in ok]
        both = [(r, cold[i]) for i, r in rs.items() if r["success"] and cold.get(i, {}).get("success")]
        ri = [r["iter"] / max(1, c["iter"]) for r, c in both]
        rw = [r["wall_s"] / c["wall_s"] for r, c in both]
        ro = [(r["obj"] - c["obj"]) / abs(c["obj"]) for r, c in both]
        worse = sum(1 for x in ro if x > 0.01)
        f = lambda v, fmt: "-" if not v else fmt.format(np.median(v))
        print(f"| {m} | {len(ok)}/{len(rs)} | {f(it, '{:.0f}')} | {np.mean(it) if it else 0:.0f} | {f(wl, '{:.2f}')} | "
              f"{f(ri, '{:.3f}')} | {f(rw, '{:.3f}')} | {f(ro, '{:+.2%}')} | {worse} |")
    print()
    print(f"MLP: train MSE {meta['mlp_train_mse_std']:.3g} (standardized), trained in {meta['mlp_train_s']:.1f} s, "
          f"inference {meta['mlp_infer_ms_median']:.3f} ms. Guess error vs the cold solution (RMS, sd units): "
          f"kNN {meta['knn_guess_rms_sd']}, MLP {meta['mlp_guess_rms_sd']}. "
          f"Phases: cold {meta['cold_phase_s']:.0f} s, warm {meta['warm_phase_s']:.0f} s on {meta['workers']} workers.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    r.add_argument("--N", type=int, default=100)
    r.add_argument("--ntrain", type=int, default=240)
    r.add_argument("--ntest", type=int, default=60)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--solver", default="smf")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--epochs", type=int, default=4000)
    r.add_argument("--out", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--problem", required=True); w.add_argument("--N", type=int, required=True)
    w.add_argument("--solver", required=True); w.add_argument("--dir", required=True); w.add_argument("--jobs", required=True)
    s = sub.add_parser("summary")
    s.add_argument("dir")
    nb = sub.add_parser("neighbors", help="parallel nearest-K warm starts on an existing run")
    nb.add_argument("dir"); nb.add_argument("--k", type=int, default=3); nb.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "neighbors":
        neighbors(a)
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "worker":
        worker(a)
    else:
        summary_dir(pathlib.Path(a.dir))


if __name__ == "__main__":
    main()
