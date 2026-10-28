"""Generate the 'optim20' benchmark suite: 20 optimization-derived symmetric
matrices (KKT/saddle-point indefinite + SPD Hessian/normal-equation types)
with varied sizes and complexity, as Matrix Market fixtures + manifest.

Usage:
    python tools/bench/gen_optim20.py

Writes:
    benchmarks/fixtures/optim20/<id>.mtx / .rhs
    benchmarks/manifests/optim20.json
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX_DIR = ROOT / "benchmarks" / "fixtures" / "optim20"
MANIFEST = ROOT / "benchmarks" / "manifests" / "optim20.json"

SOLVERS = ["smf", "ma27", "mumps", "mkl_pardiso", "cholmod"]


class MatBuilder:
    """Accumulates lower-triangle COO entries (row >= col); duplicates summed."""

    def __init__(self, n: int):
        self.n = n
        self.e: dict[tuple[int, int], float] = {}

    def add(self, r: int, c: int, v: float) -> None:
        if v == 0.0:
            return
        if r < c:
            r, c = c, r
        key = (r, c)
        self.e[key] = self.e.get(key, 0.0) + v

    def nnz(self) -> int:
        return len(self.e)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(mid: str, mb: MatBuilder, comment: str) -> dict:
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    mtx = FIX_DIR / f"{mid}.mtx"
    rhs = FIX_DIR / f"{mid}.rhs"
    entries = sorted(mb.e.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    with mtx.open("w", encoding="utf-8", newline="\n") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"% {comment} n={mb.n} nnz={mb.nnz()}\n")
        f.write(f"{mb.n} {mb.n} {mb.nnz()}\n")
        for (r, c), v in entries:
            f.write(f"{r + 1} {c + 1} {v:.17g}\n")
    with rhs.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{mb.n} 1\n")
        for _ in range(mb.n):
            f.write("1\n")
    return {
        "mtx_path": str(mtx.relative_to(ROOT)).replace("\\", "/"),
        "matrix_sha256": sha256_file(mtx),
        "rhs_path": str(rhs.relative_to(ROOT)).replace("\\", "/"),
        "rhs_sha256": sha256_file(rhs),
    }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def kkt_stage(horizon: int, n_prim: int, n_dual: int, variant: int) -> MatBuilder:
    """Stage-coupled OCP/MPC KKT saddle matrix (mirrors make_kkt_saddle_stage)."""
    stage_sz = n_prim + n_dual
    n = horizon * stage_sz
    mb = MatBuilder(n)
    base_h = 4.0 + 0.25 * (variant % 5)
    reg_d = -1e-2 - 2e-3 * (variant % 4)
    for t in range(horizon):
        base = t * stage_sz
        for p in range(n_prim):
            col = base + p
            mb.add(col, col, base_h + 0.1 * ((p + t) % 4))
            if p + 1 < n_prim:
                mb.add(col + 1, col, -0.35)
            d0 = base + n_prim + (p % n_dual)
            mb.add(d0, col, 1.0 + 0.05 * p)
            if n_dual > 1:
                d1 = base + n_prim + ((p + 1) % n_dual)
                if d1 != d0:
                    mb.add(d1, col, -0.4)
        for d in range(n_dual):
            col = base + n_prim + d
            mb.add(col, col, reg_d)
            if t + 1 < horizon:
                mb.add((t + 1) * stage_sz + (d % n_prim), col, -1.0)
    return mb


def saddle(h_blocks: list[tuple[int, int, float]],
           couplings: list[tuple[int, int, float]],
           dual_base: int, m: int, dual_diag) -> MatBuilder:
    """Generic [H A^T; A D] saddle matrix.

    h_blocks: (row, col, val) entries for the primal block (0-based, lower tri).
    couplings: (dual_row_k, primal_col, val) — stored at (dual_base+k, col).
    dual_diag: scalar or callable(k) -> dual diagonal value.
    """
    n = dual_base + m
    mb = MatBuilder(n)
    for r, c, v in h_blocks:
        mb.add(r, c, v)
    for k, c, v in couplings:
        mb.add(dual_base + k, c, v)
    for k in range(m):
        d = dual_diag(k) if callable(dual_diag) else dual_diag
        if d != 0.0:
            mb.add(dual_base + k, dual_base + k, d)
    return mb


def banded_spd(n: int, bw: int) -> MatBuilder:
    mb = MatBuilder(n)
    for j in range(n):
        mb.add(j, j, float(2 * bw + 4))
        for d in range(1, bw + 1):
            if j + d < n:
                mb.add(j + d, j, -1.0 / d)
    return mb


def poisson2d(grid: int, diag: float = 4.01) -> MatBuilder:
    n = grid * grid
    mb = MatBuilder(n)
    for r in range(grid):
        for c in range(grid):
            j = r * grid + c
            mb.add(j, j, diag)
            if c + 1 < grid:
                mb.add(j + 1, j, -1.0)
            if r + 1 < grid:
                mb.add(j + grid, j, -1.0)
    return mb


def poisson3d(g: int) -> MatBuilder:
    n = g * g * g
    mb = MatBuilder(n)
    L = g * g
    for z in range(g):
        for y in range(g):
            for x in range(g):
                j = z * L + y * g + x
                mb.add(j, j, 6.01)
                if x + 1 < g:
                    mb.add(j + 1, j, -1.0)
                if y + 1 < g:
                    mb.add(j + g, j, -1.0)
                if z + 1 < g:
                    mb.add(j + L, j, -1.0)
    return mb


def shifted(mb: MatBuilder, sigma: float) -> MatBuilder:
    for j in range(mb.n):
        mb.add(j, j, -sigma)  # accumulates onto existing diagonal
    return mb


def random_indef(n: int, per_col: int, seed: int) -> MatBuilder:
    rng = random.Random(seed)
    mb = MatBuilder(n)
    for j in range(n):
        sign = 1.0 if rng.random() < 0.55 else -1.0
        mb.add(j, j, sign * (0.4 + 0.6 * rng.random()))
        for _ in range(per_col):
            r = rng.randrange(j + 1, n) if j + 1 < n else None
            if r is not None:
                mb.add(r, j, rng.gauss(0.0, 1.0))
    return mb


def normal_eq(m: int, n: int, per_col: int, delta: float, seed: int) -> MatBuilder:
    """G = A^T A + delta*I for sparse random A (m x n)."""
    rng = random.Random(seed)
    mb = MatBuilder(n)
    cols: list[list[tuple[int, float]]] = []
    for j in range(n):
        rows = rng.sample(range(m), k=min(per_col, m))
        cols.append([(r, rng.gauss(0.0, 1.0)) for r in rows])
    for j in range(n):
        for cj in range(j + 1):
            s = 0.0
            dj = dict(cols[j])
            for r, v in cols[cj]:
                if r in dj:
                    s += v * dj[r]
            if s != 0.0:
                mb.add(j, cj, s)
        mb.add(j, j, delta)
    return mb


def graded_barrier(n: int, kappa: float, seed: int) -> MatBuilder:
    """H = D^1/2 (I - 0.4 tridiag offdiag) D^1/2, D log-spaced 1..kappa -> SPD."""
    rng = random.Random(seed)
    mb = MatBuilder(n)
    d = [math.exp(math.log(kappa) * rng.random()) for _ in range(n)]
    sq = [math.sqrt(v) for v in d]
    for j in range(n):
        mb.add(j, j, d[j])
        if j + 1 < n:
            mb.add(j + 1, j, -0.4 * sq[j] * sq[j + 1])
    return mb


def grid_incidence_kkt(g: int) -> MatBuilder:
    """Network-flow QP KKT: min 1/2 x'Hx s.t. Bx=d on a g x g grid graph.

    Primal = directed edge flows; dual = node balances (last node grounded to
    keep B full row rank). H diagonal with graded edge costs.
    """
    nodes = g * g
    edges: list[tuple[int, int]] = []
    for y in range(g):
        for x in range(g):
            u = y * g + x
            if x + 1 < g:
                edges.append((u, u + 1))
            if y + 1 < g:
                edges.append((u, u + g))
    ne = len(edges)
    m = nodes - 1  # ground last node
    mb = MatBuilder(ne + m)
    for e, (u, v) in enumerate(edges):
        mb.add(e, e, 1.0 + 0.5 * ((u + v) % 7) / 7.0)
        if u < m:
            mb.add(ne + u, e, -1.0)
        if v < m:
            mb.add(ne + v, e, 1.0)
    return mb


def kkt_pde_opt(ny: int, nu: int, m: int, bw: int = 28) -> MatBuilder:
    """PDE-constrained optimization KKT: [My 0 Jy^T; 0 Mu Ju^T; Jy Ju -eI].

    y-block: banded SPD (state mass/stiffness). u-block: tridiagonal SPD.
    Constraint row k couples to y_{k..k+2} and u_{k mod nu}.
    """
    n = ny + nu + m
    mb = MatBuilder(n)
    for j in range(ny):
        mb.add(j, j, float(2 * bw + 4))
        for d in range(1, bw + 1):
            if j + d < ny:
                mb.add(j + d, j, -1.0 / d)
    for j in range(nu):
        c = ny + j
        mb.add(c, c, 3.0)
        if j + 1 < nu:
            mb.add(c + 1, c, -1.0)
    z0 = ny + nu
    for k in range(m):
        for t in range(3):
            mb.add(z0 + k, (k + t) % ny, 1.0 + 0.2 * t)
        mb.add(z0 + k, ny + (k % nu), -0.8)
        mb.add(z0 + k, z0 + k, -1e-3)
    return mb


# ---------------------------------------------------------------------------
# Case table: (matrix_id, matrix_type, family_id, mode, builder, comment)
# ---------------------------------------------------------------------------

def cases() -> list[tuple[str, str, str, str, MatBuilder, str]]:
    out: list[tuple[str, str, str, str, MatBuilder, str]] = []

    def add(mid, mtype, fam, mode, mb, comment):
        out.append((mid, mtype, fam, mode, mb, comment))

    # --- KKT / OCP (indefinite, warm_sequence refactors) ---------------------
    add("kkt_lqr_h20", "indefinite", "kkt_ocp", "warm_sequence",
        kkt_stage(20, 6, 3, 0), "smf optim20 LQR OCP KKT h=20")
    add("kkt_lqr_h200", "indefinite", "kkt_ocp", "warm_sequence",
        kkt_stage(200, 6, 3, 0), "smf optim20 LQR OCP KKT h=200")
    add("kkt_nmpc_h30", "indefinite", "kkt_ocp", "warm_sequence",
        kkt_stage(30, 12, 6, 1), "smf optim20 NMPC KKT h=30 wide stage")
    add("kkt_contact_h40", "indefinite", "kkt_ocp", "warm_sequence",
        kkt_stage(40, 9, 6, 4), "smf optim20 contact-dynamics KKT h=40")

    # --- KKT / QP forms (indefinite) ---------------------------------------
    eq_h = banded_spd(900, 12)
    rng = random.Random(101)
    eq_cpl = [(k, rng.randrange(0, 900), 0.5 + rng.random())
              for k in range(150) for _ in range(3)]
    eq_cpl += [(k, (k * 7) % 900, 1.5) for k in range(150)]  # keep rows distinct
    eq_sad = MatBuilder(1050)
    for (r, c), v in eq_h.e.items():
        eq_sad.add(r, c, v)
    for k, c, v in eq_cpl:
        eq_sad.add(900 + k, c, v)
    add("kkt_eq_qp", "indefinite", "kkt_qp", "warm_sequence",
        eq_sad, "smf optim20 equality-QP KKT exact zero block")

    ineq = saddle(
        h_blocks=[(r, c, v) for (r, c), v in banded_spd(400, 8).e.items()],
        couplings=[(k, (k * 2 + t) % 400, 1.0 + 0.3 * t)
                   for k in range(200) for t in range(2)],
        dual_base=400, m=200,
        dual_diag=lambda k: -1e-1 * math.pow(1e-2, k / 199.0))
    add("kkt_ineq_slack", "indefinite", "kkt_qp", "warm_sequence",
        ineq, "smf optim20 inequality-QP KKT graded slack dual")

    aug = saddle(
        h_blocks=[(r, c, v) for (r, c), v in poisson2d(26).e.items()],
        couplings=[(k, (k * 5 + t) % 676, 0.8 + 0.1 * t)
                   for k in range(124) for t in range(3)],
        dual_base=676, m=124, dual_diag=-1e-5)
    add("kkt_auglag", "indefinite", "kkt_qp", "warm_sequence",
        aug, "smf optim20 augmented-Lagrangian KKT rho=1e5")

    ipm = saddle(
        h_blocks=[(r, c, v) for (r, c), v in banded_spd(500, 10).e.items()] +
                 [(j, j, math.exp(-14.0 * j / 499.0)) for j in range(500)],
        couplings=[(k, (k * 3 + t) % 500, 1.0 - 0.2 * t)
                   for k in range(100) for t in range(3)],
        dual_base=500, m=100,
        dual_diag=lambda k: -math.exp(-13.0 - 8.0 * k / 99.0))
    add("kkt_ipm_late", "indefinite", "kkt_qp", "warm_sequence",
        ipm, "smf optim20 interior-point KKT late barrier weights")

    add("kkt_network_flow", "indefinite", "kkt_qp", "warm_sequence",
        grid_incidence_kkt(36), "smf optim20 network-flow QP KKT 36x36")
    add("kkt_pde_opt", "indefinite", "kkt_pde", "warm_sequence",
        kkt_pde_opt(1400, 700, 900), "smf optim20 PDE-constrained opt KKT")

    # --- Indefinite PDE / random (cold) -------------------------------------
    add("indef_helmholtz_50", "indefinite", "indef_pde", "cold",
        shifted(poisson2d(50), 2.6), "smf optim20 shifted Poisson (Helmholtz-like)")
    add("indef_rand_1500", "indefinite", "indef_rand", "cold",
        random_indef(1500, 4, seed=7), "smf optim20 random indefinite pivoting test")

    # --- SPD (cold) ----------------------------------------------------------
    add("spd_normal_300", "spd", "spd_normal_eq", "cold",
        normal_eq(600, 300, 6, 0.05, seed=11), "smf optim20 normal equations small")
    add("spd_normal_1200", "spd", "spd_normal_eq", "cold",
        normal_eq(1800, 1200, 5, 0.05, seed=12), "smf optim20 normal equations medium")
    add("spd_barrier_800", "spd", "spd_hessian", "cold",
        graded_barrier(800, 1e7, seed=13), "smf optim20 barrier Hessian graded kappa~1e7")
    add("spd_elasticnet_1500", "spd", "spd_normal_eq", "cold",
        normal_eq(2000, 1500, 4, 1e-2, seed=14), "smf optim20 elastic-net Gram + ridge")
    add("spd_mrf_2500", "spd", "spd_pde", "cold",
        shifted(poisson2d(50, diag=4.0), -0.05), "smf optim20 MRF precision (Laplacian+shift)")
    add("spd_poisson2d_60", "spd", "spd_pde", "cold",
        poisson2d(60), "smf optim20 Poisson2D 60x60")
    add("spd_poisson3d_15", "spd", "spd_pde", "cold",
        poisson3d(15), "smf optim20 Poisson3D 15^3")
    add("spd_banded_qp_4000", "spd", "spd_hessian", "cold",
        banded_spd(4000, 16), "smf optim20 separable-QP banded Hessian")

    return out


def main() -> int:
    manifest_cases = []
    for mid, mtype, fam, mode, mb, comment in cases():
        meta = write_fixture(mid, mb, comment)
        manifest_cases.append({
            "matrix_id": mid,
            "sequence_id": f"{mid}_seq",
            "family_id": fam,
            **meta,
            "matrix_type": mtype,
            "split": "test",
            "mode": mode,
            "track": "recommended",
            "expected_outcome": "Success",
        })
        print(f"{mid:22s} {mtype:10s} n={mb.n:5d} nnz={mb.nnz():6d}")

    manifest = {
        "schema_version": 1,
        "suite": "optim20",
        "solvers": SOLVERS,
        "required_baselines": ["ma27", "mumps", "mkl_pardiso", "cholmod"],
        "cases": manifest_cases,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {MANIFEST.relative_to(ROOT)} with {len(manifest_cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
