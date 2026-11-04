// test_posdef_front_kernel.cpp — Control::posdef_front_kernel = Lapack
// (posdef_factor_front_lapack: dpotrf + dtrsm + dsyrk on SPD fronts)
// against the Ldlt kernel (ldlt_factor_front with prm.posdef, D folded into
// L as multifrontal_serial stores it).

#include "smf/blas_thread_guard.hpp"
#include "smf/ipm_session.hpp"
#include "smf/ldlt_kernel.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace smf;

namespace {

// ---------------------------------------------------------------------------
// Dense fronts
// ---------------------------------------------------------------------------

// Random SPD f x f (lower triangle, column-major, ld f): B B^T / f + I with
// B uniform in [-1, 1]; the upper triangle holds garbage, as a reused front
// buffer does.
std::vector<double> random_spd_front(int f, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const std::size_t sf = static_cast<std::size_t>(f);
    std::vector<double> B(sf * sf);
    for (double &v : B) v = u(rng);
    std::vector<double> F(sf * sf, 0.0);
    for (int j = 0; j < f; ++j)
        for (int i = 0; i < f; ++i) {
            if (i < j) {
                F[j * sf + i] = 1.0e300;  // never read
                continue;
            }
            double s = 0.0;
            for (int k = 0; k < f; ++k) s += B[k * sf + i] * B[k * sf + j];
            F[j * sf + i] = s / f + (i == j ? 1.0 : 0.0);
        }
    return F;
}

struct KernelRun {
    LdltKernelResult kr;
    std::vector<double> F;
    std::vector<int8_t> tags;
};

LdltKernelParams posdef_params(int blocked_min_front) {
    LdltKernelParams prm;
    prm.posdef = true;
    prm.eff_small = 0.0;  // factor_posdef's
    prm.static_pivot = 0.0;
    prm.nb = 16;          // Control::indef_panel_size
    prm.blocked_min_front = blocked_min_front;
    return prm;
}

// The Ldlt kernel, then D folded into L (multifrontal_serial's SPD store).
KernelRun run_ldlt(std::vector<double> F, int f, int p, int blocked_min_front) {
    const LdltKernelParams prm = posdef_params(blocked_min_front);
    std::vector<Int> rows(static_cast<std::size_t>(f));
    std::iota(rows.begin(), rows.end(), Int{0});
    std::vector<int8_t> tags(static_cast<std::size_t>(f), 0);
    std::vector<double> work(ldlt_work_size(f, prm));
    KernelRun r;
    r.kr = ldlt_factor_front(F.data(), f, p, rows.data(), prm, tags.data(), work.data());
    for (int t = 0; t < f; ++t) EXPECT_EQ(rows[static_cast<std::size_t>(t)], t);  // SPD: no swaps
    const std::size_t sf = static_cast<std::size_t>(f);
    for (int c = 0; c < r.kr.num_accepted; ++c) {
        double *Lc = F.data() + c * sf;
        const double sd = std::sqrt(Lc[c]);
        Lc[c] = sd;
        for (int i = c + 1; i < f; ++i) Lc[i] *= sd;
    }
    r.F = std::move(F);
    r.tags = std::move(tags);
    return r;
}

KernelRun run_lapack(std::vector<double> F, int f, int p) {
    KernelRun r;
    r.tags.assign(static_cast<std::size_t>(f), 0);
    r.kr = posdef_factor_front_lapack(F.data(), f, p, r.tags.data());
    r.F = std::move(F);
    return r;
}

// Largest |a - b| over L (columns [0, p), rows c..f) and over the lower
// triangle of the Schur complement F[p:f, p:f], relative to the largest
// entry of the original front.
struct FrontDiff {
    double l = 0.0;
    double s = 0.0;
};

FrontDiff front_diff(const KernelRun &a, const KernelRun &b, int f, int p, double scale) {
    const std::size_t sf = static_cast<std::size_t>(f);
    FrontDiff d;
    for (int c = 0; c < f; ++c)
        for (int i = c; i < f; ++i) {
            const double e = std::abs(a.F[c * sf + i] - b.F[c * sf + i]) / scale;
            if (c < p) d.l = std::max(d.l, e);
            else d.s = std::max(d.s, e);
        }
    return d;
}

double max_abs(const std::vector<double> &F, int f) {
    const std::size_t sf = static_cast<std::size_t>(f);
    double m = 0.0;
    for (int c = 0; c < f; ++c)
        for (int i = c; i < f; ++i) m = std::max(m, std::abs(F[c * sf + i]));
    return m;
}

// The pivot (Schur complement diagonal) column j of the SPD front would get:
// a(j, j) - sum_k l(j, k)^2 over the Cholesky factor of the leading block.
double pivot_at(std::vector<double> F, int f, int j) {
    if (j == 0) return F[0];
    KernelRun r = run_lapack(std::move(F), f, j);
    EXPECT_EQ(r.kr.num_accepted, j);
    return r.F[static_cast<std::size_t>(j) * f + j];  // updated by dsyrk
}

} // namespace

// ---------------------------------------------------------------------------
// Kernel: L, Schur complement, tags, inertia and flops match the Ldlt kernel
// ---------------------------------------------------------------------------

TEST(PosdefFrontKernel, MatchesLdltOnRandomSpdFronts) {
    // (f, p): tiny, p < f, p = f, panel-boundary and blocked sizes.
    const int shapes[][2] = {{1, 1},   {2, 1},    {3, 3},    {5, 2},    {12, 12},  {13, 7},
                             {17, 16}, {33, 1},   {40, 40},  {40, 10},  {95, 95},  {96, 96},
                             {97, 33}, {130, 50}, {200, 200}, {257, 64}, {300, 299}, {400, 17}};
    unsigned seed = 1;
    for (const auto &sh : shapes) {
        const int f = sh[0];
        const int p = sh[1];
        SCOPED_TRACE(std::string("f=") + std::to_string(f) + std::string(" p=") + std::to_string(p));
        const std::vector<double> A = random_spd_front(f, seed++);
        const double scale = max_abs(A, f);
        const KernelRun lp = run_lapack(A, f, p);
        ASSERT_EQ(lp.kr.num_accepted, p);
        // Both Ldlt paths: as dispatched (unblocked below 96 rows) and blocked.
        for (const int bmin : {96, 1}) {
            SCOPED_TRACE(std::string("blocked_min_front=") + std::to_string(bmin));
            const KernelRun ld = run_ldlt(A, f, p, bmin);
            ASSERT_EQ(ld.kr.num_accepted, p);
            EXPECT_EQ(lp.kr.inertia.positive, ld.kr.inertia.positive);
            EXPECT_EQ(lp.kr.inertia.negative, 0);
            EXPECT_EQ(lp.kr.inertia.zero, 0);
            EXPECT_EQ(ld.kr.inertia.negative, 0);
            EXPECT_EQ(ld.kr.inertia.zero, 0);
            EXPECT_EQ(lp.kr.static_pivots, 0);
            EXPECT_EQ(lp.kr.flops, ld.kr.flops);
            for (int c = 0; c < p; ++c) EXPECT_EQ(lp.tags[static_cast<std::size_t>(c)], 1);
            const FrontDiff d = front_diff(lp, ld, f, p, scale);
            // Rounding only: well conditioned (eigenvalues in [1, ~4]).
            EXPECT_LT(d.l, 1e-13);
            EXPECT_LT(d.s, 1e-13);
        }
        EXPECT_EQ(lp.kr.inertia.positive, p);
    }
}

// ---------------------------------------------------------------------------
// Kernel: a front not positive definite fails at the same column
// ---------------------------------------------------------------------------

TEST(PosdefFrontKernel, NotPositiveDefiniteSameColumn) {
    struct Case {
        int f, p, j;
        int kind;  // 0: pivot -1, 1: row/column j exactly zero, 2: NaN diagonal, 3: +inf diagonal
    };
    const Case cases[] = {
        {10, 10, 0, 0},   {10, 10, 7, 0},   {40, 25, 24, 0},  {120, 120, 0, 0},
        {120, 120, 57, 0}, {200, 150, 149, 0}, {300, 300, 133, 0}, {300, 300, 133, 1},
        {150, 100, 3, 2}, {150, 100, 77, 2}, {150, 100, 0, 3}, {150, 100, 64, 3},
        {12, 12, 11, 3},
    };
    unsigned seed = 100;
    for (const Case &c : cases) {
        SCOPED_TRACE(std::string("f=") + std::to_string(c.f) + std::string(" p=") + std::to_string(c.p) + std::string(" j=") + std::to_string(c.j) + std::string(" kind=") + std::to_string(c.kind));
        std::vector<double> A = random_spd_front(c.f, seed++);
        double &ajj = A[static_cast<std::size_t>(c.j) * c.f + c.j];
        if (c.kind == 0) {
            // Lower a(j, j) so that column j's pivot is -1.
            ajj -= pivot_at(A, c.f, c.j) + 1.0;
        } else if (c.kind == 1) {
            // Row and column j zero: its pivot is exactly 0 in either kernel.
            const std::size_t sf = static_cast<std::size_t>(c.f);
            for (int k = 0; k < c.j; ++k) A[k * sf + c.j] = 0.0;
            for (int i = c.j; i < c.f; ++i) A[c.j * sf + i] = 0.0;
        } else if (c.kind == 2) {
            ajj = std::numeric_limits<double>::quiet_NaN();
        } else {
            ajj = std::numeric_limits<double>::infinity();
        }
        const KernelRun ld = run_ldlt(A, c.f, c.p, 1);
        const KernelRun ldu = run_ldlt(A, c.f, c.p, 1 << 20);  // unblocked
        const KernelRun lp = run_lapack(A, c.f, c.p);
        EXPECT_EQ(ld.kr.num_accepted, c.j);
        EXPECT_EQ(ldu.kr.num_accepted, c.j);
        EXPECT_EQ(lp.kr.num_accepted, c.j);
        EXPECT_EQ(lp.kr.inertia.positive, ld.kr.inertia.positive);
        EXPECT_EQ(lp.kr.inertia.negative, 0);
        EXPECT_EQ(lp.kr.inertia.zero, 0);
        EXPECT_EQ(lp.kr.flops, ld.kr.flops);
    }
}

// ---------------------------------------------------------------------------
// End to end
// ---------------------------------------------------------------------------

namespace {

// 7-point Laplacian on an m^3 grid minus `shift` on the diagonal, lower CSC.
CscLower grid_3d(int m, double shift) {
    const int n = m * m * m;
    CscLower A;
    A.n = n;
    A.col_ptr.push_back(0);
    for (int z = 0; z < m; ++z)
        for (int y = 0; y < m; ++y)
            for (int x = 0; x < m; ++x) {
                const int j = (z * m + y) * m + x;
                A.row_idx.push_back(j);
                A.values.push_back(6.0 - shift);
                if (x + 1 < m) { A.row_idx.push_back(j + 1); A.values.push_back(-1.0); }
                if (y + 1 < m) { A.row_idx.push_back(j + m); A.values.push_back(-1.0); }
                if (z + 1 < m) { A.row_idx.push_back(j + m * m); A.values.push_back(-1.0); }
                A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
            }
    return A;
}

// Random sparse SPD: a random symmetric pattern (about `deg` entries per
// column) made diagonally dominant, with unequal row scales.
CscLower random_sparse_spd(int n, int deg, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_int_distribution<int> pick(0, n - 1);
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n));
    std::vector<double> rowsum(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j)
        for (int t = 0; t < deg; ++t) {
            const int i = pick(rng);
            if (i <= j) continue;
            const double v = u(rng);
            cols[static_cast<std::size_t>(j)].push_back({i, v});
            rowsum[static_cast<std::size_t>(i)] += std::abs(v);
            rowsum[static_cast<std::size_t>(j)] += std::abs(v);
        }
    CscLower A;
    A.n = n;
    A.col_ptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        auto &c = cols[static_cast<std::size_t>(j)];
        std::sort(c.begin(), c.end());
        A.row_idx.push_back(j);
        A.values.push_back(rowsum[static_cast<std::size_t>(j)] + 0.5 + (j % 7));
        int last = -1;
        for (const auto &[i, v] : c) {
            if (i == last) {
                A.values.back() += v;
                continue;
            }
            A.row_idx.push_back(i);
            A.values.push_back(v);
            last = i;
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

double residual(const CscLower &A, const std::vector<double> &x, const std::vector<double> &b) {
    const std::size_t n = static_cast<std::size_t>(A.n);
    std::vector<double> r(b);
    double anorm = 0.0;
    for (std::size_t j = 0; j < n; ++j)
        for (Int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k) {
            const std::size_t i = static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(k)]);
            const double v = A.values[static_cast<std::size_t>(k)];
            anorm = std::max(anorm, std::abs(v));
            r[i] -= v * x[j];
            if (i != j) r[j] -= v * x[i];
        }
    double rn = 0.0, xn = 0.0, bn = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        rn = std::max(rn, std::abs(r[i]));
        xn = std::max(xn, std::abs(x[i]));
        bn = std::max(bn, std::abs(b[i]));
    }
    return rn / (anorm * xn + bn);
}

struct E2E {
    FactorStatus status = FactorStatus::Success;
    ErrorCode info_status = ErrorCode::Success;
    std::vector<double> factor;
    std::vector<double> x;
    double resid = 0.0;
    int lapack_fronts = 0;
    int max_front = 0;
    int pos = 0, neg = 0, zero = 0;
};

E2E run_e2e(const CscLower &A, MatrixType type, PosdefFrontKernel kernel, int threads) {
    Control ctrl;
    ctrl.matrix_type = type;
    ctrl.num_threads = threads;
    ctrl.auto_threads = false;  // small test trees
    ctrl.posdef_front_kernel = kernel;
    Solver solver;
    Info info;
    E2E r;
    auto ak = solver.analyse(A, ctrl, info);
    EXPECT_NE(ak.get(), nullptr);
    if (!ak) return r;
    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;
    r.status = solver.factor(*ak, A, ctrl, info, fk, nws);
    r.info_status = info.status;
    r.lapack_fronts = info.posdef_lapack_fronts;
    r.max_front = info.max_front_size;
    r.pos = info.num_positive;
    r.neg = info.num_negative;
    r.zero = info.num_zero;
    if (r.status != FactorStatus::Success) return r;
    r.factor = fk.factor_values;
    const int n = static_cast<int>(A.n);
    std::vector<double> b(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
    r.x = b;
    EXPECT_EQ(solver.solve(fk, ctrl, info, r.x.data(), n, 1, sws), 0);
    r.resid = residual(A, r.x, b);
    return r;
}

} // namespace

TEST(PosdefFrontKernel, SolveAccuracyAndFactorMatch) {
    struct M {
        const char *name;
        CscLower A;
    };
    std::vector<M> mats;
    mats.push_back({"poisson_3d_22", grid_3d(22, 0.0)});
    mats.push_back({"random_spd_3000", random_sparse_spd(3000, 6, 7)});
    for (const M &m : mats) {
        for (const int threads : {1, 8}) {
            SCOPED_TRACE(std::string(m.name) + std::string(" threads=") + std::to_string(threads));
            const E2E ld = run_e2e(m.A, MatrixType::RealSymmetricPositiveDefinite,
                                   PosdefFrontKernel::Ldlt, threads);
            const E2E lp = run_e2e(m.A, MatrixType::RealSymmetricPositiveDefinite,
                                   PosdefFrontKernel::Lapack, threads);
            ASSERT_EQ(ld.status, FactorStatus::Success);
            ASSERT_EQ(lp.status, FactorStatus::Success);
            EXPECT_EQ(ld.lapack_fronts, 0);
            EXPECT_GT(lp.lapack_fronts, 0) << "max front " << lp.max_front;
            EXPECT_LT(ld.resid, 1e-14);
            EXPECT_LT(lp.resid, 1e-14);
            std::printf("[SolveAccuracy] %s %dT: %d lapack fronts, max front %d, resid ldlt %.2e "
                        "lapack %.2e\n",
                        m.name, threads, lp.lapack_fronts, lp.max_front, ld.resid, lp.resid);
            ASSERT_EQ(ld.factor.size(), lp.factor.size());
            double xd = 0.0;
            for (std::size_t i = 0; i < ld.x.size(); ++i)
                xd = std::max(xd, std::abs(ld.x[i] - lp.x[i]) / std::max(1.0, std::abs(ld.x[i])));
            EXPECT_LT(xd, 1e-11);
            // One thread: the same factor layout (the threaded engine's
            // depends on which thread took which task). L agrees to rounding.
            if (threads != 1) continue;
            double fmax = 0.0, dmax = 0.0;
            for (std::size_t i = 0; i < ld.factor.size(); ++i) {
                fmax = std::max(fmax, std::abs(ld.factor[i]));
                dmax = std::max(dmax, std::abs(ld.factor[i] - lp.factor[i]));
            }
            EXPECT_LT(dmax, 1e-12 * fmax);
        }
    }
}

// The SPD factor's inertia is (n, 0, 0) (IpmSession reports it for a
// successful SPD factor), and its solves agree with the Ldlt kernel's.
TEST(PosdefFrontKernel, IpmSessionInertia) {
    const CscLower A = grid_3d(18, 0.0);
    for (const PosdefFrontKernel k : {PosdefFrontKernel::Ldlt, PosdefFrontKernel::Lapack}) {
        SCOPED_TRACE(std::to_string(static_cast<int>(k)));
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
        ctrl.posdef_front_kernel = k;
        IpmControl ipm;
        ipm.learn_inertia = false;
        ipm.abort_on_wrong_inertia = false;
        ipm.reanalyse = false;
        ipm.static_mode = StaticMode::Off;
        IpmSession s;
        ASSERT_EQ(s.analyse(A, {}, ctrl, ipm), ErrorCode::Success);
        ASSERT_EQ(s.factor(A.values), IpmFactorStatus::Ok);
        const InertiaCounts in = s.inertia();
        EXPECT_EQ(in.positive, static_cast<int>(A.n));
        EXPECT_EQ(in.negative, 0);
        EXPECT_EQ(in.zero, 0);
        if (k == PosdefFrontKernel::Lapack) EXPECT_GT(s.info().posdef_lapack_fronts, 0);
    }
}

// A matrix that is not positive definite, failing in a large front: both
// kernels report NotPositiveDefinite the same way, and the caller's fallback
// (factor it as indefinite) gives a correct factorization, whatever the knob.
TEST(PosdefFrontKernel, NotPositiveDefiniteAndIndefiniteFallback) {
    // 7-point Laplacian on 20^3: eigenvalues 6 - 2 sum cos(k pi / 21), the
    // smallest ~0.067; a shift of 0.5 leaves a few dozen negative ones, so
    // the leaves factor and a large separator front fails.
    const CscLower A = grid_3d(20, 0.5);
    for (const int threads : {1, 8}) {
        SCOPED_TRACE(std::string("threads=") + std::to_string(threads));
        const E2E ld = run_e2e(A, MatrixType::RealSymmetricPositiveDefinite,
                               PosdefFrontKernel::Ldlt, threads);
        const E2E lp = run_e2e(A, MatrixType::RealSymmetricPositiveDefinite,
                               PosdefFrontKernel::Lapack, threads);
        EXPECT_EQ(ld.status, FactorStatus::NotPositiveDefinite);
        EXPECT_EQ(lp.status, FactorStatus::NotPositiveDefinite);
        EXPECT_EQ(ld.info_status, ErrorCode::NotPositiveDefinite);
        EXPECT_EQ(lp.info_status, ld.info_status);
        if (threads == 1) EXPECT_GT(lp.lapack_fronts, 0) << "no Lapack-path front ran";

        // Fallback: the indefinite path, untouched by the SPD knob.
        const E2E il = run_e2e(A, MatrixType::RealSymmetricIndefinite, PosdefFrontKernel::Ldlt,
                               threads);
        const E2E ip = run_e2e(A, MatrixType::RealSymmetricIndefinite, PosdefFrontKernel::Lapack,
                               threads);
        ASSERT_EQ(il.status, FactorStatus::Success);
        ASSERT_EQ(ip.status, FactorStatus::Success);
        EXPECT_EQ(il.lapack_fronts, 0);
        EXPECT_EQ(ip.lapack_fronts, 0);
        EXPECT_LT(il.resid, 1e-12);
        EXPECT_EQ(il.pos + il.neg + il.zero, static_cast<int>(A.n));
        EXPECT_GT(il.neg, 0);
        EXPECT_EQ(il.zero, 0);
        EXPECT_EQ(ip.pos, il.pos);
        EXPECT_EQ(ip.neg, il.neg);
        if (threads == 1) {
            // One thread: the indefinite factor is the same bit for bit.
            EXPECT_TRUE(il.factor == ip.factor);
            EXPECT_TRUE(il.x == ip.x);
        }
    }
}
