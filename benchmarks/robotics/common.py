"""Shared harness for the robotics trajectory-optimization suite.

Every problem module exposes ``build(N, seed) -> Problem``: a CasADi NLP
(exact Hessian through CasADi's AD), its bounds, an initial guess and a
``check(sol) -> dict`` that verifies feasibility and the task (obstacles
cleared, goal reached). ``solve()`` runs IPOPT on it with one linear solver:

  mumps        IPOPT's bundled MUMPS
  ma27/ma57/ma97  CoinHSL through ``hsllib`` (SMF_HSL_LIBRARY, or the copy in
               build-deps)
  smf          the smf MA97-compatible plugin (``linear_solver ma97`` with
               ``hsllib`` = smf_ma97.dll; SMF_PLUGIN_DLL overrides the path),
               with its opt-in slack-pair elimination on (SMF_SLACK_ELIM=on
               unless the environment sets it: off for the plain plugin)

and returns wall time, iterations, objective, CasADi's function-evaluation
split and IPOPT's own timing statistics (``print_timing_statistics``), parsed
from IPOPT's output file: symbolic factorization, numerical factorization and
back solves are the linear solver's share.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import tempfile
import time
from typing import Any, Callable

import casadi as ca
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_HSL = (REPO / "build-deps" / "coinhsl-2024.5.15" /
               "CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5" / "bin" / "libhsl.dll")
SOLVERS = ("mumps", "ma27", "ma57", "ma97", "smf")


@dataclasses.dataclass
class Problem:
    name: str
    x: Any                 # decision vector (SX/MX column)
    f: Any                 # objective
    g: Any                 # constraints
    lbx: np.ndarray
    ubx: np.ndarray
    lbg: np.ndarray
    ubg: np.ndarray
    x0: np.ndarray
    check: Callable[[np.ndarray], dict]
    meta: dict = dataclasses.field(default_factory=dict)


def hsl_library() -> str:
    p = os.environ.get("SMF_HSL_LIBRARY")
    if p:
        return p
    # build-deps is not in git: in a worktree, fall back to the build's copy.
    for cand in (DEFAULT_HSL, REPO / "build-board" / "Release" / "libhsl.dll", REPO / "build" / "Release" / "libhsl.dll"):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError("libhsl.dll not found; set SMF_HSL_LIBRARY")


def plugin_library() -> str:
    p = os.environ.get("SMF_PLUGIN_DLL")
    if p:
        return p
    for cand in (REPO / "build-board" / "Release" / "smf_ma97.dll", REPO / "build" / "Release" / "smf_ma97.dll"):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError("smf_ma97.dll not found; set SMF_PLUGIN_DLL")


def jit_options() -> dict:
    """ROBO_JIT=1: CasADi's code generation for the NLP functions, compiled
    with `zig cc`, as robotics deployments do. Off by default: compiling the
    whole NLP's derivatives takes minutes to over 15 minutes at N >= 100
    (the arm). IPOPT's timing statistics split the linear solver's time out
    either way; only the function-evaluation share depends on it."""
    if os.environ.get("ROBO_JIT", "0") != "1":
        return {}
    jo = {"compiler": "zig cc", "compiler_setup": "-c", "compiler_output_flag": "-o ",
          "compiler_flags": ["-O1"], "linker": "zig cc", "linker_setup": "-shared",
          "linker_output_flag": "-o ", "verbose": False}
    return {"jit": True, "compiler": "shell", "jit_options": jo, "jit_cleanup": True}


def ipopt_options(solver: str, out_file: str, extra: dict | None = None) -> dict:
    o = {
        "ipopt.print_level": 5,
        "ipopt.file_print_level": 5,
        "ipopt.output_file": out_file,
        "ipopt.print_timing_statistics": "yes",
        "ipopt.tol": 1e-8,
        "ipopt.max_iter": 3000,
        "ipopt.max_cpu_time": 600.0,
        "print_time": False,
    }
    if solver == "mumps":
        o["ipopt.linear_solver"] = "mumps"
    elif solver in ("ma27", "ma57", "ma97"):
        o["ipopt.linear_solver"] = solver
        o["ipopt.hsllib"] = hsl_library()
    elif solver == "smf":
        o["ipopt.linear_solver"] = "ma97"
        o["ipopt.hsllib"] = plugin_library()
    else:
        raise ValueError(f"unknown solver {solver}")
    o.update(extra or {})
    return o


_TIMING = {
    "symbolic": "LinearSystemSymbolicFactorization",
    "factor": "LinearSystemFactorization",
    "backsolve": "LinearSystemBackSolve",
    "linsys_total": "PDSystemSolverTotal",
    "ipopt_total": "OverallAlgorithm",
    "func_evals": "Function Evaluations",
}


def parse_ipopt_output(text: str) -> dict:
    """IPOPT's timing statistics (the wall-clock column) and problem
    dimensions."""
    out: dict[str, Any] = {}
    for key, label in _TIMING.items():
        m = re.search(re.escape(label) + r"\.*:\s*[\d.]+\s*\(sys:\s*[\d.]+\s*wall:\s*([\d.]+)\)", text)
        out[key] = float(m.group(1)) if m else None
    for key, pat in (("n_var", r"Total number of variables\.*:\s*(\d+)"),
                     ("n_eq", r"Total number of equality constraints\.*:\s*(\d+)"),
                     ("n_ineq", r"Total number of inequality constraints\.*:\s*(\d+)"),
                     ("nnz_jac_eq", r"Number of nonzeros in equality constraint Jacobian\.*:\s*(\d+)"),
                     ("nnz_jac_ineq", r"Number of nonzeros in inequality constraint Jacobian\.*:\s*(\d+)"),
                     ("nnz_hess", r"Number of nonzeros in Lagrangian Hessian\.*:\s*(\d+)")):
        m = re.search(pat, text)
        out[key] = int(m.group(1)) if m else None
    m = re.search(r"EXIT:\s*(.+)", text)
    out["exit"] = m.group(1).strip() if m else None
    return out


def kkt_size(info: dict) -> tuple[int | None, int | None]:
    """IPOPT's augmented system order: x, the inequality slacks s, and the
    multipliers of the equalities and inequalities (n_x + n_eq + 2 n_ineq),
    and its lower-triangle nonzeros (Hessian, both Jacobians, the slacks'
    identity coupling and the diagonal)."""
    try:
        n = info["n_var"] + info["n_eq"] + 2 * info["n_ineq"]
        nnz = info["nnz_hess"] + info["nnz_jac_eq"] + info["nnz_jac_ineq"] + info["n_ineq"] + n
        return n, nnz
    except TypeError:
        return None, None


def solve(prob: Problem, solver: str, threads: int = 1, extra: dict | None = None,
          warm: bool = False) -> dict:
    """Solves prob once with `solver`; returns a result record."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "SMF_NUM_THREADS"):
        os.environ[k] = str(threads)
    if solver == "smf":
        # The plugin's slack-pair elimination pays on these planners (many
        # inequality slacks, most inactive); SMF_SLACK_ELIM=off measures without.
        os.environ.setdefault("SMF_SLACK_ELIM", "on")
    fd, out_file = tempfile.mkstemp(prefix=f"ipopt_{prob.name}_{solver}_", suffix=".txt")
    os.close(fd)
    opts = ipopt_options(solver, out_file, extra)
    opts["ipopt.print_level"] = 0
    opts.update(jit_options())
    nlp = {"x": prob.x, "f": prob.f, "g": prob.g}
    cwd = os.getcwd()
    jit_dir = tempfile.mkdtemp(prefix="casadi_jit_")
    os.chdir(jit_dir)  # the shell compiler writes its sources and objects here
    t_build = time.perf_counter()
    try:
        S = ca.nlpsol("S", "ipopt", nlp, opts)
    finally:
        os.chdir(cwd)
    t_build = time.perf_counter() - t_build
    t0 = time.perf_counter()
    r = S(x0=prob.x0, lbx=prob.lbx, ubx=prob.ubx, lbg=prob.lbg, ubg=prob.ubg)
    wall = time.perf_counter() - t0
    st = S.stats()
    del S  # IPOPT keeps its output file open until the solver is freed
    import gc
    gc.collect()
    try:
        text = pathlib.Path(out_file).read_text(errors="replace")
    finally:
        try:
            os.remove(out_file)
        except OSError:
            pass
    info = parse_ipopt_output(text)
    xs = np.array(r["x"]).ravel()
    gval = np.array(r["g"]).ravel()
    viol = float(max(0.0, np.max(np.r_[prob.lbg - gval, gval - prob.ubg, prob.lbx - xs, xs - prob.ubx, 0.0])))
    n, nnz = kkt_size(info)
    rec = {
        "problem": prob.name, "solver": solver, "threads": threads, **prob.meta,
        "status": st.get("return_status"), "success": bool(st.get("success")),
        "iter": st.get("iter_count"), "obj": float(r["f"]), "max_viol": viol,
        "wall_s": wall, "build_s": t_build, "jit": bool(opts.get("jit")),
        "t_nlp_f": st.get("t_wall_nlp_f"), "t_nlp_g": st.get("t_wall_nlp_g"),
        "t_nlp_grad_f": st.get("t_wall_nlp_grad_f"), "t_nlp_jac_g": st.get("t_wall_nlp_jac_g"),
        "t_nlp_hess_l": st.get("t_wall_nlp_hess_l"),
        "kkt_n": n, "kkt_nnz": nnz, **{f"ipopt_{k}": v for k, v in info.items()},
    }
    rec["check"] = prob.check(xs)
    return rec


# ------------------------------------------------------------ utilities --
def rk4(f: Callable, x, u, h, *p):
    k1 = f(x, u, *p)
    k2 = f(x + h / 2 * k1, u, *p)
    k3 = f(x + h / 2 * k2, u, *p)
    k4 = f(x + h * k3, u, *p)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Builder:
    """Collects decision variables (with bounds and guesses) and constraints."""

    def __init__(self, mx: bool = False):
        # mx: MX decision variables, so the NLP calls the per-interval
        # functions (SX) instead of inlining them: compact generated code for
        # models whose interval function is large (the arm).
        self.sym = ca.MX.sym if mx else ca.SX.sym
        self.w, self.lbw, self.ubw, self.w0 = [], [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.offsets: dict[str, list[int]] = {}
        self._n = 0

    def var(self, name: str, size: int, lb, ub, guess):
        v = self.sym(name, size)
        self.w.append(v)
        self.lbw.append(np.broadcast_to(np.asarray(lb, float), (size,)))
        self.ubw.append(np.broadcast_to(np.asarray(ub, float), (size,)))
        self.w0.append(np.broadcast_to(np.asarray(guess, float), (size,)))
        self.offsets.setdefault(name.split("_")[0], []).append(self._n)
        self._n += size
        return v

    def eq(self, expr):
        expr = ca.vec(expr)
        self.g.append(expr)
        self.lbg.append(np.zeros(expr.shape[0]))
        self.ubg.append(np.zeros(expr.shape[0]))

    def ineq(self, expr, lb, ub):
        expr = ca.vec(expr)
        k = expr.shape[0]
        self.g.append(expr)
        self.lbg.append(np.broadcast_to(np.asarray(lb, float), (k,)))
        self.ubg.append(np.broadcast_to(np.asarray(ub, float), (k,)))

    def problem(self, name: str, f, check, meta=None) -> Problem:
        return Problem(name=name, x=ca.vertcat(*self.w), f=f, g=ca.vertcat(*self.g),
                       lbx=np.concatenate(self.lbw), ubx=np.concatenate(self.ubw),
                       lbg=np.concatenate(self.lbg), ubg=np.concatenate(self.ubg),
                       x0=np.concatenate(self.w0), check=check, meta=meta or {})
