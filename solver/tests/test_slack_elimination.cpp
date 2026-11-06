// Control::slack_elimination (slack_elimination.hpp): IPOPT-like KKTs with
// inequality slacks, factored through the slack plan (On) against the full
// matrix (Off): the same inertia and the same solutions; every SolveJob and
// multiple right-hand sides; zero slack diagonals; forced fallbacks, which
// must be bitwise the full path; Auto below its threshold, bitwise Off;
// deterministic mode across thread counts; the IpmSession path.
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/ipm_session.hpp"
#include "smf/ipopt_plugin_control.hpp"
#include "smf/slack_elimination.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <span>
#include <vector>

using namespace smf;

namespace {

// The augmented system of min f(x) s.t. c(x) = 0, d(x) - s = 0, as IPOPT
// builds it:
//   [ W + Sx     .     Jc^T   Jd^T ]  x
//   [   .     Ss+dw     .     -I   ]  s
//   [  Jc        .     -dc     .   ]  y_c
//   [  Jd       -I      .     -dc  ]  y_d
// W: a random sparse symmetric matrix (banded plus random couplings), Jd
// rows of 1..4 entries, Jc rows of 2..5. `zero_slack`: the share of slack
// diagonals set to exactly 0 (a variable in one constraint only); `dc`: the
// multipliers' diagonal (0: no stored diagonal for y_d).
struct Kkt {
    CscLower A;
    int nx = 0, ns = 0, nc = 0;
};

Kkt make_kkt(unsigned seed, int nx, int nc, int nd, double zero_slack = 0.0, double dc = 1e-4,
             double sigma_spread = 1.0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::uniform_int_distribution<int> var(0, nx - 1);
    const int n = nx + nd + nc + nd;
    std::map<std::pair<int, int>, double> e;  // (col, row), row >= col
    const auto add = [&](int i, int j, double v) {
        if (i < j) std::swap(i, j);
        e[{j, i}] += v;
    };
    for (int i = 0; i < nx; ++i) {
        add(i, i, 4.0 + U(rng));
        if (i + 1 < nx) add(i + 1, i, U(rng));
        if (i % 3 == 0) add(var(rng), i, 0.5 * U(rng));
    }
    const int s0 = nx, yc0 = nx + nd, yd0 = nx + nd + nc;
    std::uniform_real_distribution<double> E(-sigma_spread, sigma_spread);
    for (int k = 0; k < nd; ++k) {
        const int s = s0 + k, y = yd0 + k;
        const bool z = (k * 7919) % 1000 < static_cast<int>(1000 * zero_slack);
        add(s, s, z ? 0.0 : std::pow(10.0, E(rng)));
        add(y, s, -1.0);
        if (dc > 0.0) add(y, y, -dc);
        const int kk = 1 + static_cast<int>(rng() % 4);
        for (int t = 0; t < kk; ++t) add(y, var(rng), U(rng));
    }
    for (int k = 0; k < nc; ++k) {
        const int y = yc0 + k;
        add(y, y, -dc > 0 ? -dc : -1e-8);
        const int kk = 2 + static_cast<int>(rng() % 4);
        for (int t = 0; t < kk; ++t) add(y, var(rng), U(rng));
    }
    Kkt out;
    out.nx = nx;
    out.ns = nd;
    out.nc = nc;
    CscLower &A = out.A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto &[key, v] : e) {
        A.row_idx.push_back(key.second);
        A.values.push_back(v);
        ++A.col_ptr[static_cast<std::size_t>(key.first) + 1];
    }
    for (int j = 0; j < n; ++j) A.col_ptr[static_cast<std::size_t>(j) + 1] += A.col_ptr[static_cast<std::size_t>(j)];
    return out;
}

Control base(SlackElimination mode) {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.pivot_u = 1e-8;
    c.slack_elimination = mode;
    return c;
}

struct Run {
    FactorStatus status = FactorStatus::Success;
    Info info;
    std::vector<double> x;  // solution of A x = b, b_i = 1 + i % 7
    std::vector<double> parts;  // Forward, then DiagOnly, then Backward of b
    bool slack = false;
};

Run run(const CscLower &A, const Control &c, int nrhs = 1) {
    Solver s;
    Info ia;
    auto ak = s.analyse(A, c, ia);
    EXPECT_NE(ak.get(), nullptr);
    Run r;
    if (!ak) return r;
    FactorKeep fk;
    NumericWorkspace ws;
    r.status = s.factor(*ak, std::span<const double>(A.values), c, r.info, fk, ws);
    r.slack = fk.slack != nullptr;
    const int n = static_cast<int>(A.n);
    r.x.resize(static_cast<std::size_t>(n) * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) r.x[static_cast<std::size_t>(j) * n + i] = 1.0 + (i + 3 * j) % 7;
    r.parts = r.x;
    Info si;
    EXPECT_EQ(s.solve(fk, c, si, r.x.data(), n, nrhs), 0);
    EXPECT_EQ(s.solve(fk, c, si, r.parts.data(), n, nrhs, SolveJob::Forward), 0);
    EXPECT_EQ(s.solve(fk, c, si, r.parts.data(), n, nrhs, SolveJob::DiagOnly), 0);
    EXPECT_EQ(s.solve(fk, c, si, r.parts.data(), n, nrhs, SolveJob::Backward), 0);
    return r;
}

double rel_diff(const std::vector<double> &a, const std::vector<double> &b) {
    double d = 0, m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::abs(a[i] - b[i]));
        m = std::max(m, std::abs(b[i]));
    }
    return d / std::max(m, 1e-300);
}

// Normwise backward error of x for A x = b (A lower CSC, symmetric).
double backward_error(const CscLower &A, const std::vector<double> &x, const std::vector<double> &b) {
    const int n = static_cast<int>(A.n);
    std::vector<double> r(b), rowsum(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const int i = A.row_idx[static_cast<std::size_t>(p)];
            const double v = A.values[static_cast<std::size_t>(p)];
            r[static_cast<std::size_t>(i)] -= v * x[static_cast<std::size_t>(j)];
            rowsum[static_cast<std::size_t>(i)] += std::abs(v);
            if (i != j) {
                r[static_cast<std::size_t>(j)] -= v * x[static_cast<std::size_t>(i)];
                rowsum[static_cast<std::size_t>(j)] += std::abs(v);
            }
        }
    double rn = 0, an = 0, xn = 0, bn = 0;
    for (int i = 0; i < n; ++i) {
        rn = std::max(rn, std::abs(r[static_cast<std::size_t>(i)]));
        an = std::max(an, rowsum[static_cast<std::size_t>(i)]);
        xn = std::max(xn, std::abs(x[static_cast<std::size_t>(i)]));
        bn = std::max(bn, std::abs(b[static_cast<std::size_t>(i)]));
    }
    return rn / (an * xn + bn);
}

// Column j of an n x nrhs block, and run()'s right-hand side j.
std::vector<double> col(const std::vector<double> &X, int n, int j) {
    return std::vector<double>(X.begin() + static_cast<std::ptrdiff_t>(j) * n,
                               X.begin() + static_cast<std::ptrdiff_t>(j + 1) * n);
}
std::vector<double> rhs_col(int n, int j) {
    std::vector<double> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = 1.0 + (i + 3 * j) % 7;
    return b;
}

std::vector<double> rhs_of(int n) {
    std::vector<double> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = 1.0 + i % 7;
    return b;
}

bool same_bits(const std::vector<double> &a, const std::vector<double> &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}


// Componentwise backward error max_i |b - A x|_i / (|A| |x| + |b|)_i (rows
// with a zero denominator: their residual must be 0).
double omega(const CscLower &A, const std::vector<double> &x, const std::vector<double> &b) {
    const int n = static_cast<int>(A.n);
    std::vector<double> r(b), d(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) d[static_cast<std::size_t>(i)] = std::abs(b[static_cast<std::size_t>(i)]);
    for (int j = 0; j < n; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const int i = A.row_idx[static_cast<std::size_t>(p)];
            const double v = A.values[static_cast<std::size_t>(p)];
            r[static_cast<std::size_t>(i)] -= v * x[static_cast<std::size_t>(j)];
            d[static_cast<std::size_t>(i)] += std::abs(v * x[static_cast<std::size_t>(j)]);
            if (i != j) {
                r[static_cast<std::size_t>(j)] -= v * x[static_cast<std::size_t>(i)];
                d[static_cast<std::size_t>(j)] += std::abs(v * x[static_cast<std::size_t>(i)]);
            }
        }
    double w = 0.0;
    for (int i = 0; i < n; ++i) {
        const double ri = std::abs(r[static_cast<std::size_t>(i)]), di = d[static_cast<std::size_t>(i)];
        w = std::max(w, di > 0.0 ? ri / di : (ri > 0.0 ? 1.0 : 0.0));
    }
    return w;
}

// The solve of A x = 1 (all ones) through a fresh analysis and factor.
struct OnesRun {
    std::vector<double> x;
    bool slack = false;
    Info info;
};
OnesRun solve_ones(const CscLower &A, const Control &c, double rhs = 1.0) {
    Solver s;
    Info ia;
    auto ak = s.analyse(A, c, ia);
    OnesRun r;
    if (!ak) return r;
    FactorKeep fk;
    NumericWorkspace ws;
    s.factor(*ak, std::span<const double>(A.values), c, r.info, fk, ws);
    r.slack = fk.slack != nullptr;
    r.x.assign(static_cast<std::size_t>(A.n), rhs);
    Info si;
    s.solve(fk, c, si, r.x.data(), static_cast<int>(A.n), 1);
    r.info.slack_refine_steps = si.slack_refine_steps;
    r.info.slack_solve_fallbacks = si.slack_solve_fallbacks;
    return r;
}

// A with the independent block [3 0.7; 0.7 2.3] in front (rows 0, 1).
CscLower prepend_block(const CscLower &A) {
    CscLower B;
    B.n = A.n + 2;
    B.col_ptr = {0, 2, 3};
    B.row_idx = {0, 1, 1};
    B.values = {3.0, 0.7, 2.3};
    for (Int j = 0; j < A.n; ++j) {
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            B.row_idx.push_back(A.row_idx[static_cast<std::size_t>(p)] + 2);
            B.values.push_back(A.values[static_cast<std::size_t>(p)]);
        }
        B.col_ptr.push_back(static_cast<Int>(B.row_idx.size()));
    }
    return B;
}

// Codex's repro: 100 states (diagonal 4, neighbours -0.5), each coupled by 1
// to a multiplier whose slack block is [1e14 -1; -1 0] (an active
// constraint at a huge barrier term).
CscLower huge_slack_kkt() {
    const int m = 100, n = 3 * m;
    std::map<std::pair<int, int>, double> e;
    for (int i = 0; i < m; ++i) {
        e[{i, i}] = 4.0;
        if (i + 1 < m) e[{i, i + 1}] = -0.5;
        e[{i, 2 * m + i}] = 1.0;            // state - multiplier
        e[{m + i, m + i}] = 1e14;           // slack
        e[{m + i, 2 * m + i}] = -1.0;       // slack - multiplier
    }
    CscLower A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto &[key, v] : e) {
        A.row_idx.push_back(key.second);
        A.values.push_back(v);
        ++A.col_ptr[static_cast<std::size_t>(key.first) + 1];
    }
    for (int j = 0; j < n; ++j) A.col_ptr[static_cast<std::size_t>(j) + 1] += A.col_ptr[static_cast<std::size_t>(j)];
    return A;
}

} // namespace

TEST(SlackElimination, MatchesFullPathOnRandomKkts) {
    int fired = 0;
    for (unsigned seed = 1; seed <= 12; ++seed) {
        SCOPED_TRACE(seed);
        const Kkt k = make_kkt(seed, 60 + 10 * static_cast<int>(seed), 10, 120 + 15 * static_cast<int>(seed),
                               seed % 3 == 0 ? 0.2 : 0.0, seed % 4 == 0 ? 0.0 : 1e-8);
        const Run on = run(k.A, base(SlackElimination::On), 3);
        const Run off = run(k.A, base(SlackElimination::Off), 3);
        ASSERT_EQ(on.status, off.status);
        EXPECT_FALSE(off.slack);
        fired += on.slack;
        EXPECT_EQ(on.info.num_positive, off.info.num_positive);
        EXPECT_EQ(on.info.num_negative, off.info.num_negative);
        EXPECT_EQ(on.info.num_zero, off.info.num_zero);
        // Saddle-point systems: judged by backward error (the solutions
        // themselves differ at 1e-9 between the full path's own solve
        // variants). Forward, DiagOnly and Backward compose to a solve too.
        const int n = static_cast<int>(k.A.n);
        for (int j = 0; j < 3; ++j) {
            const auto b = rhs_col(n, j);
            const double ef = backward_error(k.A, col(off.x, n, j), b);
            EXPECT_LT(backward_error(k.A, col(on.x, n, j), b), 10.0 * ef + 1e-15);
            EXPECT_LT(backward_error(k.A, col(on.parts, n, j), b), 10.0 * ef + 1e-15);
        }
        if (on.slack) EXPECT_GT(on.info.slack_pairs, 0);
    }
    EXPECT_GE(fired, 10) << "the plan should run on almost every case";
}

TEST(SlackElimination, ZeroSlackDiagonalsAreExact) {
    // Every slack diagonal 0 (variables in one constraint only): the 2x2
    // block pivots stay exact where 1x1 pivots on the slacks would divide by 0.
    const Kkt k = make_kkt(77, 80, 8, 200, 1.0);
    const Run on = run(k.A, base(SlackElimination::On));
    const Run off = run(k.A, base(SlackElimination::Off));
    EXPECT_TRUE(on.slack);
    EXPECT_EQ(on.info.num_negative, off.info.num_negative);
    EXPECT_EQ(on.info.num_positive, off.info.num_positive);
    const auto b = rhs_of(static_cast<int>(k.A.n));
    EXPECT_LT(backward_error(k.A, on.x, b), 10.0 * backward_error(k.A, off.x, b) + 1e-15);
}

TEST(SlackElimination, ForcedFallbackIsTheFullPath) {
    const Kkt k = make_kkt(5, 100, 10, 250);
    Control c = base(SlackElimination::On);
    c.slack_update_limit = 1e-30;  // every pair trips the guard
    Solver s;
    Info ia;
    auto ak = s.analyse(k.A, c, ia);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_NE(ak->slack, nullptr);
    const Run off = run(k.A, base(SlackElimination::Off));
    std::vector<double> x_off(static_cast<std::size_t>(k.A.n));
    {
        Solver so;
        Info oi;
        auto ok = so.analyse(k.A, base(SlackElimination::Off), oi);
        ASSERT_NE(ok.get(), nullptr);
        FactorKeep fk;
        NumericWorkspace ws;
        ASSERT_EQ(so.factor(*ok, std::span<const double>(k.A.values), base(SlackElimination::Off), oi, fk, ws),
                  off.status);
        for (int i = 0; i < k.A.n; ++i) x_off[static_cast<std::size_t>(i)] = 1.0 + i % 7;
        ASSERT_EQ(so.solve(fk, base(SlackElimination::Off), oi, x_off.data(), static_cast<int>(k.A.n), 1), 0);
    }
    for (int rep = 0; rep < kSlackMaxFallbacks + 2; ++rep) {
        FactorKeep fk;
        NumericWorkspace ws;
        Info fi;
        ASSERT_EQ(s.factor(*ak, std::span<const double>(k.A.values), c, fi, fk, ws), off.status);
        EXPECT_EQ(fk.slack, nullptr);
        // Fallbacks are counted until the plan is given up on.
        EXPECT_EQ(fi.slack_fallbacks, rep < kSlackMaxFallbacks ? 1 : 0);
        std::vector<double> x(static_cast<std::size_t>(k.A.n));
        for (int i = 0; i < k.A.n; ++i) x[static_cast<std::size_t>(i)] = 1.0 + i % 7;
        Info si;
        ASSERT_EQ(s.solve(fk, c, si, x.data(), static_cast<int>(k.A.n), 1), 0);
        // The full matrix's own analysis: bitwise the plain full path.
        EXPECT_TRUE(same_bits(x, x_off)) << "rep " << rep;
        EXPECT_EQ(fi.num_negative, off.info.num_negative);
    }
}

TEST(SlackElimination, AutoBelowThresholdIsOff) {
    // Few pairs (about 6% of the rows): Auto keeps the full path, bitwise.
    const Kkt k = make_kkt(9, 400, 40, 15);
    const Run a = run(k.A, base(SlackElimination::Auto));
    const Run off = run(k.A, base(SlackElimination::Off));
    EXPECT_FALSE(a.slack);
    EXPECT_EQ(a.info.slack_pairs, 0);
    EXPECT_TRUE(same_bits(a.x, off.x));
    EXPECT_TRUE(same_bits(a.parts, off.parts));
}

TEST(SlackElimination, IllConditionedBackwardError) {
    // Slacks over 12 decades (active and inactive constraints) and a tiny
    // delta_c: the condensed elimination's backward error stays that of
    // the full path's order.
    for (unsigned seed = 40; seed < 44; ++seed) {
        SCOPED_TRACE(seed);
        const Kkt k = make_kkt(seed, 200, 20, 500, 0.1, 1e-10, 6.0);
        const Run on = run(k.A, base(SlackElimination::On));
        const Run off = run(k.A, base(SlackElimination::Off));
        EXPECT_TRUE(on.slack);
        EXPECT_EQ(on.info.num_negative, off.info.num_negative);
        const auto b = rhs_of(static_cast<int>(k.A.n));
        const double eo = backward_error(k.A, on.x, b), ef = backward_error(k.A, off.x, b);
        std::printf("[SlackElimination] seed %u: backward error %.2e (full path %.2e)\n", seed, eo, ef);
        EXPECT_LT(eo, 100.0 * ef + 1e-14);
    }
}

TEST(SlackElimination, AboveMaxPivotUIsUnchanged) {
    // At pivot_u > kSlackMaxPivotU (the u = 0.01 front tier) no plan is
    // built: analysis, factor and solve are bitwise those with Off.
    const Kkt k = make_kkt(17, 200, 20, 400);
    Control a = base(SlackElimination::Auto), off = base(SlackElimination::Off);
    a.pivot_u = off.pivot_u = 0.01;
    Solver s;
    Info ia, io;
    auto ka = s.analyse(k.A, a, ia);
    auto ko = s.analyse(k.A, off, io);
    ASSERT_TRUE(ka && ko);
    EXPECT_EQ(ka->slack, nullptr);
    EXPECT_EQ(ka->perm, ko->perm);
    const Run ra = run(k.A, a), ro = run(k.A, off);
    EXPECT_FALSE(ra.slack);
    EXPECT_TRUE(same_bits(ra.x, ro.x));
    EXPECT_TRUE(same_bits(ra.parts, ro.parts));
    EXPECT_EQ(ra.info.num_negative, ro.info.num_negative);
}

TEST(SlackElimination, StridedRightHandSides) {
    const Kkt k = make_kkt(13, 120, 12, 300);
    const Control c = base(SlackElimination::On);
    Solver s;
    Info ia;
    auto ak = s.analyse(k.A, c, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    NumericWorkspace ws;
    Info fi;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(k.A.values), c, fi, fk, ws), FactorStatus::Success);
    ASSERT_NE(fk.slack, nullptr);
    const int n = static_cast<int>(k.A.n), ldb = n + 5, nrhs = 3;
    std::vector<double> B(static_cast<std::size_t>(ldb) * nrhs, -7.0), ref;
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) B[static_cast<std::size_t>(j) * ldb + i] = 1.0 + (i * (j + 1)) % 5;
    ref = B;
    SolveWorkspace sws;
    Info si;
    ASSERT_EQ(s.solve(fk, c, si, std::span<double>(B), nrhs, ldb, sws), 0);
    for (int j = 0; j < nrhs; ++j) {
        std::vector<double> x(ref.begin() + j * ldb, ref.begin() + j * ldb + n);
        ASSERT_EQ(s.solve(fk, c, si, x.data(), n, 1), 0);
        std::vector<double> bj(B.begin() + j * ldb, B.begin() + j * ldb + n);
        const std::vector<double> rj(ref.begin() + j * ldb, ref.begin() + j * ldb + n);
        EXPECT_LT(backward_error(k.A, bj, rj), 10.0 * backward_error(k.A, x, rj) + 1e-15) << "column " << j;
        for (int i = n; i < ldb; ++i) EXPECT_EQ(B[static_cast<std::size_t>(j) * ldb + i], -7.0);
    }
}

TEST(SlackElimination, DeterministicAcrossThreads) {
    const Kkt k = make_kkt(21, 300, 30, 700);
    std::vector<double> x1;
    for (int t : {1, 2, 8}) {
        Control c = base(SlackElimination::On);
        c.deterministic = true;
        c.num_threads = t;
        const Run r = run(k.A, c);
        EXPECT_TRUE(r.slack);
        if (x1.empty()) x1 = r.x;
        else EXPECT_TRUE(same_bits(r.x, x1)) << t << " threads";
    }
}

TEST(SlackElimination, IpmSessionPath) {
    const Kkt k = make_kkt(31, 150, 15, 350);
    const Run off = run(k.A, base(SlackElimination::Off));
    IpmSession sess;
    IpmControl ipm;
    ASSERT_EQ(sess.analyse(k.A, std::span<const double>(k.A.values), base(SlackElimination::Auto), ipm),
              ErrorCode::Success);
    for (int rep = 0; rep < 3; ++rep) {
        ASSERT_EQ(sess.factor(std::span<const double>(k.A.values)), IpmFactorStatus::Ok);
        EXPECT_GT(sess.info().slack_pairs, 0);
        EXPECT_EQ(sess.inertia().negative, off.info.num_negative);
        std::vector<double> x(static_cast<std::size_t>(k.A.n));
        for (int i = 0; i < k.A.n; ++i) x[static_cast<std::size_t>(i)] = 1.0 + i % 7;
        ASSERT_EQ(sess.solve(x.data(), static_cast<int>(k.A.n), 1), 0);
        EXPECT_LT(backward_error(k.A, x, rhs_of(static_cast<int>(k.A.n))),
                  10.0 * backward_error(k.A, off.x, rhs_of(static_cast<int>(k.A.n))) + 1e-15);
    }
}

TEST(SlackElimination, DefaultIsOff) {
    const Kkt k = make_kkt(5, 100, 10, 250);
    Control c = base(SlackElimination::Off);
    c.slack_elimination = Control{}.slack_elimination;
    EXPECT_EQ(c.slack_elimination, SlackElimination::Off);
    Solver s;
    Info ia;
    auto ak = s.analyse(k.A, c, ia);
    ASSERT_NE(ak.get(), nullptr);
    EXPECT_EQ(ak->slack, nullptr);
    EXPECT_EQ(ia.slack_pairs, 0);
}

TEST(SlackElimination, PluginEnvironmentKnob) {
    EXPECT_EQ(plugin_slack_elimination_resolve(nullptr).mode, SlackElimination::Off);
    EXPECT_EQ(plugin_slack_elimination_resolve("").mode, SlackElimination::Off);
    EXPECT_EQ(plugin_slack_elimination_resolve("off").mode, SlackElimination::Off);
    EXPECT_EQ(plugin_slack_elimination_resolve("on").mode, SlackElimination::On);
    EXPECT_EQ(plugin_slack_elimination_resolve("auto").mode, SlackElimination::Auto);
    const PluginSlackElimination bad = plugin_slack_elimination_resolve("yes");
    EXPECT_FALSE(bad.valid);
    EXPECT_EQ(bad.mode, SlackElimination::Off);
}

// Codex P2 (PR #57): pairs with a huge slack diagonal pass the update guard
// but the substituted solution loses componentwise accuracy; the solve must
// still meet the full path's.
TEST(SlackElimination, HugeSlackDiagonalIsAccurate) {
    const CscLower A = huge_slack_kkt();
    const std::vector<double> ones(static_cast<std::size_t>(A.n), 1.0);
    const OnesRun off = solve_ones(A, base(SlackElimination::Off));
    const OnesRun on = solve_ones(A, base(SlackElimination::On));
    const double woff = omega(A, off.x, ones), won = omega(A, on.x, ones);
    std::printf("[ huge     ] omega on %.2e (slack %d) off %.2e\n", won, on.slack ? 1 : 0, woff);
    EXPECT_LT(won, 1e-12);
    EXPECT_EQ(on.info.num_negative, off.info.num_negative);
}

// Codex P2 (PR #57): a KKT scaled by 1e-161 makes the pair determinants
// subnormal; the pairs must be inverted from a normalized block.
TEST(SlackElimination, TinyScaledKktIsAccurate) {
    Kkt k = make_kkt(5, 100, 10, 250, 0.0, 1e-4);
    for (double &v : k.A.values) v *= 1e-161;
    const std::vector<double> ones(static_cast<std::size_t>(k.A.n), 1.0);
    const OnesRun off = solve_ones(k.A, base(SlackElimination::Off));
    // The pair inverses alone (no solve check), then with the check: as
    // accurate as the full path (itself ~4e-9 here), not 1e-3.
    Control nc = base(SlackElimination::On);
    nc.slack_omega_target = 0.0;
    const OnesRun raw = solve_ones(k.A, nc);
    const OnesRun on = solve_ones(k.A, base(SlackElimination::On));
    const double woff = omega(k.A, off.x, ones), won = omega(k.A, on.x, ones), wraw = omega(k.A, raw.x, ones);
    std::printf("[ tiny     ] omega on %.2e unchecked %.2e (slack %d) off %.2e\n", won, wraw, on.slack ? 1 : 0, woff);
    EXPECT_TRUE(raw.slack);
    EXPECT_LT(wraw, 10.0 * woff + 1e-12);
    EXPECT_LT(won, 10.0 * woff + 1e-12);
    EXPECT_EQ(on.info.num_negative, off.info.num_negative);
}

// Codex P2 (PR #57, re-review): the solve check's backward error must not
// overflow or underflow with the right-hand side's scale (1e200, 1e-200).
TEST(SlackElimination, SolveCheckIsScaleInvariant) {
    const CscLower A = prepend_block(huge_slack_kkt());
    for (const double scale : {1e200, 1e-200}) {
        const std::vector<double> b(static_cast<std::size_t>(A.n), scale);
        const OnesRun off = solve_ones(A, base(SlackElimination::Off), scale);
        const OnesRun on = solve_ones(A, base(SlackElimination::On), scale);
        const double woff = omega(A, off.x, b), won = omega(A, on.x, b);
        std::printf("[ rhs %.0e ] omega on %.2e (slack %d, refine %d, fallback %d) off %.2e\n", scale, won,
                    on.slack ? 1 : 0, on.info.slack_refine_steps, on.info.slack_solve_fallbacks, woff);
        EXPECT_TRUE(on.slack);
        EXPECT_LT(won, 1e-12) << "scale " << scale;
    }
}

// Codex P2 (PR #57, re-review): a slack factor's solve rejects a null
// right-hand side and a non-positive nrhs before touching them.
TEST(SlackElimination, SolveRejectsBadArguments) {
    const CscLower A = huge_slack_kkt();
    const Control c = base(SlackElimination::On);
    Solver s;
    Info ia;
    auto ak = s.analyse(A, c, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    NumericWorkspace ws;
    Info fi;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), c, fi, fk, ws), FactorStatus::Success);
    ASSERT_NE(fk.slack, nullptr);
    const int n = static_cast<int>(A.n);
    Info si;
    EXPECT_EQ(s.solve(fk, c, si, nullptr, n, 1), static_cast<int>(ErrorCode::IllegalValue));
    EXPECT_EQ(si.status, ErrorCode::IllegalValue);
    std::vector<double> x(static_cast<std::size_t>(n), 1.0);
    Info sj;
    EXPECT_EQ(s.solve(fk, c, sj, x.data(), n, -1), static_cast<int>(ErrorCode::IllegalValue));
    EXPECT_EQ(s.solve(fk, c, sj, x.data(), n, 0), static_cast<int>(ErrorCode::IllegalValue));
}

// Codex P2 (PR #57, re-review): the solve-time fallback factors the full
// matrix in the cleaned layout, so input with duplicates and unsorted rows
// (factored through either overload) falls back to an accurate solve.
TEST(SlackElimination, SolveFallbackOnUncleanInput) {
    const CscLower clean = huge_slack_kkt();
    // The same matrix with every state diagonal split into two duplicate
    // entries and each column's rows reversed.
    CscLower A;
    A.n = clean.n;
    A.col_ptr.push_back(0);
    for (Int j = 0; j < clean.n; ++j) {
        std::vector<std::pair<Int, double>> col;
        for (Int p = clean.col_ptr[static_cast<std::size_t>(j)]; p < clean.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int i = clean.row_idx[static_cast<std::size_t>(p)];
            const double v = clean.values[static_cast<std::size_t>(p)];
            if (i == j && v == 4.0) {
                col.push_back({i, 1.5});
                col.push_back({i, 2.5});
            } else {
                col.push_back({i, v});
            }
        }
        std::reverse(col.begin(), col.end());
        for (const auto &[i, v] : col) {
            A.row_idx.push_back(i);
            A.values.push_back(v);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    const std::vector<double> ones(static_cast<std::size_t>(clean.n), 1.0);
    Control c = base(SlackElimination::On);
    c.slack_omega_target = 1e-300;  // refinement can never reach it: always the fallback
    for (int overload = 0; overload < 2; ++overload) {
        Solver s;
        Info ia;
        auto ak = s.analyse(A, c, ia);
        ASSERT_NE(ak.get(), nullptr);
        ASSERT_NE(ak->slack, nullptr);
        FactorKeep fk;
        NumericWorkspace ws;
        Info fi;
        if (overload == 0) {
            ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), c, fi, fk, ws), FactorStatus::Success);
        } else {
            // The CscLower overload with values in the cleaned pattern.
            CscLower U = ak->cleaned;
            ASSERT_EQ(s.factor(*ak, U, c, fi, fk, ws), FactorStatus::Success);
        }
        ASSERT_NE(fk.slack, nullptr);
        std::vector<double> x(ones);
        Info si;
        ASSERT_EQ(s.solve(fk, c, si, x.data(), static_cast<int>(clean.n), 1), 0);
        EXPECT_EQ(si.slack_solve_fallbacks, 1) << "overload " << overload;
        const double w = omega(clean, x, ones);
        std::printf("[ unclean  ] overload %d: omega %.2e (fallback %d)\n", overload, w, si.slack_solve_fallbacks);
        EXPECT_LT(w, 1e-12) << "overload " << overload;
    }
}

// Codex P2 (PR #57, re-review): a right-hand side of mixed magnitude (1e20 on
// an independent block, 1 elsewhere) must not let the large rows hide an
// inaccurate solution in the others: the check is per row (Oettli-Prager).
TEST(SlackElimination, SolveCheckMixedMagnitudeRhs) {
    const CscLower A = prepend_block(huge_slack_kkt());
    std::vector<double> b(static_cast<std::size_t>(A.n), 1.0);
    b[0] = b[1] = 1e20;
    for (const bool on : {false, true}) {
        const Control c = base(on ? SlackElimination::On : SlackElimination::Off);
        Solver s;
        Info ia;
        auto ak = s.analyse(A, c, ia);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        NumericWorkspace ws;
        Info fi;
        ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), c, fi, fk, ws), FactorStatus::Success);
        EXPECT_EQ(fk.slack != nullptr, on);
        std::vector<double> x(b);
        Info si;
        ASSERT_EQ(s.solve(fk, c, si, x.data(), static_cast<int>(A.n), 1), 0);
        const double w = omega(A, x, b);
        std::printf("[ mixed    ] %s: omega %.2e (refine %d, fallback %d)\n", on ? "on" : "off", w,
                    si.slack_refine_steps, si.slack_solve_fallbacks);
        EXPECT_LT(w, 1e-12) << (on ? "on" : "off");
    }
}

// Codex P2 (PR #57, re-review): accumulated pair updates can make the reduced
// matrix numerically singular although the full matrix factors cleanly (two
// states, 128 pairs [1e15 -1; -1 0] each coupled by 1 to both states). The
// factorization must then take the full matrix: inertia (130, 128, 0).
TEST(SlackElimination, SingularReducedMatrixTakesTheFullMatrix) {
    const int m = 128, n = 2 + 2 * m;
    std::map<std::pair<int, int>, double> e;  // (col, row), row >= col
    e[{0, 0}] = 4.0;
    e[{1, 1}] = 4.0;
    for (int k = 0; k < m; ++k) {
        const int s = 2 + k, y = 2 + m + k;
        e[{s, s}] = 1e15;
        e[{s, y}] = -1.0;
        e[{0, y}] = 1.0;
        e[{1, y}] = 1.0;
    }
    CscLower A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto &[key, v] : e) {
        A.row_idx.push_back(key.second);
        A.values.push_back(v);
        ++A.col_ptr[static_cast<std::size_t>(key.first) + 1];
    }
    for (int j = 0; j < n; ++j) A.col_ptr[static_cast<std::size_t>(j) + 1] += A.col_ptr[static_cast<std::size_t>(j)];
    for (const bool on : {false, true}) {
        const Control c = base(on ? SlackElimination::On : SlackElimination::Off);
        Solver s;
        Info ia;
        auto ak = s.analyse(A, c, ia);
        ASSERT_NE(ak.get(), nullptr);
        if (on) ASSERT_NE(ak->slack, nullptr);
        FactorKeep fk;
        NumericWorkspace ws;
        Info fi;
        const FactorStatus st = s.factor(*ak, std::span<const double>(A.values), c, fi, fk, ws);
        std::printf("[ singular ] %s: status %d inertia (%d, %d, %d) slack %d fallbacks %d\n", on ? "on" : "off",
                    static_cast<int>(st), fi.num_positive, fi.num_negative, fi.num_zero, fk.slack ? 1 : 0,
                    fi.slack_fallbacks);
        EXPECT_EQ(st, FactorStatus::Success) << (on ? "on" : "off");
        EXPECT_EQ(fi.num_positive, 130);
        EXPECT_EQ(fi.num_negative, 128);
        EXPECT_EQ(fi.num_zero, 0);
        if (on) EXPECT_EQ(fi.slack_fallbacks, 1);
        std::vector<double> x(static_cast<std::size_t>(n), 1.0);
        const std::vector<double> b(x);
        Info si;
        ASSERT_EQ(s.solve(fk, c, si, x.data(), n, 1), 0);
        EXPECT_LT(omega(A, x, b), 1e-12);
    }
}
