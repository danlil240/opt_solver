/// @file test_lean_engine.cpp
/// Randomized end-to-end checks of the serial multifrontal engine
/// (multifrontal_factor_serial) and the relaxed-amalgamation analysis.
///
///   - LeanEngine.KktInertiaAndResidual : saddle-point matrices with a zero
///     (2,2) block — the delayed-pivot / 2x2-pivot path. Inertia must be
///     exactly (n_primal, n_dual, 0) and the backward error tiny.
///   - LeanEngine.RandomIndefinite      : random-sign diagonals, threshold
///     pivoting across several pivot_u / nemin settings.
///   - LeanEngine.RandomSpd              : diagonally dominant SPD through the
///     posdef (Cholesky-layout) path.
///   - LeanEngine.RefactorReusesWorkspace : value-only refactorizations on a
///     warmed NumericWorkspace stay accurate and allocate nothing new.

#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <utility>
#include <vector>

using namespace smf;

namespace {

/// Lower-CSC from (row, col) -> value with row >= col.
CscLower to_lower_csc(int n, const std::map<std::pair<int, int>, double> &entries) {
    CscLower A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto &e : entries) ++A.col_ptr[static_cast<std::size_t>(e.first.second) + 1];
    for (int j = 0; j < n; ++j) A.col_ptr[static_cast<std::size_t>(j) + 1] += A.col_ptr[static_cast<std::size_t>(j)];
    A.row_idx.resize(entries.size());
    A.values.resize(entries.size());
    std::vector<Int> head(A.col_ptr.begin(), A.col_ptr.end() - 1);
    // std::map orders by (row, col); emit column-major with sorted rows.
    std::vector<std::pair<std::pair<int, int>, double>> v(entries.begin(), entries.end());
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
        return a.first.second != b.first.second ? a.first.second < b.first.second
                                                : a.first.first < b.first.first;
    });
    for (const auto &e : v) {
        const Int slot = head[static_cast<std::size_t>(e.first.second)]++;
        A.row_idx[static_cast<std::size_t>(slot)] = e.first.first;
        A.values[static_cast<std::size_t>(slot)] = e.second;
    }
    return A;
}

/// y = A x for a symmetric matrix stored as lower CSC.
std::vector<double> sym_matvec(const CscLower &A, const std::vector<double> &x) {
    std::vector<double> y(static_cast<std::size_t>(A.n), 0.0);
    for (Int j = 0; j < A.n; ++j) {
        for (Int k = A.col_ptr[static_cast<std::size_t>(j)]; k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            const Int i = A.row_idx[static_cast<std::size_t>(k)];
            const double a = A.values[static_cast<std::size_t>(k)];
            y[static_cast<std::size_t>(i)] += a * x[static_cast<std::size_t>(j)];
            if (i != j) y[static_cast<std::size_t>(j)] += a * x[static_cast<std::size_t>(i)];
        }
    }
    return y;
}

/// Normwise backward error ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf).
double backward_error(const CscLower &A, const std::vector<double> &x, const std::vector<double> &b) {
    const auto Ax = sym_matvec(A, x);
    std::vector<double> rowsum(static_cast<std::size_t>(A.n), 0.0);
    for (Int j = 0; j < A.n; ++j) {
        for (Int k = A.col_ptr[static_cast<std::size_t>(j)]; k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            const Int i = A.row_idx[static_cast<std::size_t>(k)];
            const double a = std::abs(A.values[static_cast<std::size_t>(k)]);
            rowsum[static_cast<std::size_t>(i)] += a;
            if (i != j) rowsum[static_cast<std::size_t>(j)] += a;
        }
    }
    double r = 0.0, an = 0.0, xn = 0.0, bn = 0.0;
    for (Int i = 0; i < A.n; ++i) {
        const auto si = static_cast<std::size_t>(i);
        r = std::max(r, std::abs(b[si] - Ax[si]));
        an = std::max(an, rowsum[si]);
        xn = std::max(xn, std::abs(x[si]));
        bn = std::max(bn, std::abs(b[si]));
    }
    return r / (an * xn + bn);
}

/// [H A^T; A 0] with H diagonally dominant SPD (n1 x n1) and A (m x n1)
/// having a dense "identity-like" part so it has full row rank.
CscLower make_kkt(int n1, int m, double density, std::mt19937 &rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> p01(0.0, 1.0);
    std::map<std::pair<int, int>, double> e;
    std::vector<double> rowabs(static_cast<std::size_t>(n1), 0.0);
    for (int j = 0; j < n1; ++j)
        for (int i = j + 1; i < n1; ++i)
            if (p01(rng) < density) {
                const double v = u(rng);
                e[{i, j}] = v;
                rowabs[static_cast<std::size_t>(i)] += std::abs(v);
                rowabs[static_cast<std::size_t>(j)] += std::abs(v);
            }
    for (int j = 0; j < n1; ++j) e[{j, j}] = 1.0 + rowabs[static_cast<std::size_t>(j)] + p01(rng);
    for (int r = 0; r < m; ++r) {
        const int row = n1 + r;
        e[{row, r % n1}] = 2.0 + p01(rng); // full row rank anchor
        for (int j = 0; j < n1; ++j)
            if (j != r % n1 && p01(rng) < density) e[{row, j}] = u(rng);
        // Explicit zero diagonal: the pattern a KKT assembler hands over.
        e[{row, row}] = 0.0;
    }
    return to_lower_csc(n1 + m, e);
}

CscLower make_random_sym(int n, double density, bool spd, std::mt19937 &rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> p01(0.0, 1.0);
    std::map<std::pair<int, int>, double> e;
    std::vector<double> rowabs(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j)
        for (int i = j + 1; i < n; ++i)
            if (p01(rng) < density) {
                const double v = u(rng);
                e[{i, j}] = v;
                rowabs[static_cast<std::size_t>(i)] += std::abs(v);
                rowabs[static_cast<std::size_t>(j)] += std::abs(v);
            }
    for (int j = 0; j < n; ++j) {
        double d = spd ? 0.5 + rowabs[static_cast<std::size_t>(j)] : u(rng) * 0.5;
        if (!spd && p01(rng) < 0.2) d = 0.0; // forces delayed / 2x2 pivots
        e[{j, j}] = d;
    }
    return to_lower_csc(n, e);
}

struct RunResult {
    FactorStatus status;
    Info info;
    double berr = 0.0;
};

RunResult factor_and_solve(const CscLower &A, const Control &ctrl, std::mt19937 &rng) {
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    EXPECT_NE(ak, nullptr);
    FactorKeep fk;
    NumericWorkspace ws;
    RunResult rr;
    rr.status = solver.factor(*ak, A, ctrl, rr.info, fk, ws);
    if (rr.status != FactorStatus::Success) return rr;
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> b(static_cast<std::size_t>(A.n));
    for (auto &v : b) v = u(rng);
    std::vector<double> x = b;
    Info sinfo;
    EXPECT_EQ(solver.solve(fk, ctrl, sinfo, x.data(), static_cast<int>(A.n)), 0);
    rr.berr = backward_error(A, x, b);
    return rr;
}

} // namespace

TEST(LeanEngine, KktInertiaAndResidual) {
    std::mt19937 rng(12345);
    const int shapes[][2] = {{4, 2}, {10, 5}, {30, 12}, {60, 40}, {150, 60}, {300, 200}};
    for (const auto &sh : shapes) {
        for (int nemin : {1, 8, 32}) {
            const CscLower A = make_kkt(sh[0], sh[1], 0.05, rng);
            Control ctrl;
            ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
            ctrl.nemin = nemin;
            const RunResult rr = factor_and_solve(A, ctrl, rng);
            ASSERT_EQ(rr.status, FactorStatus::Success) << "n1=" << sh[0] << " m=" << sh[1] << " nemin=" << nemin;
            EXPECT_EQ(rr.info.num_positive, sh[0]);
            EXPECT_EQ(rr.info.num_negative, sh[1]);
            EXPECT_EQ(rr.info.num_zero, 0);
            EXPECT_LT(rr.berr, 1e-13) << "n1=" << sh[0] << " m=" << sh[1] << " nemin=" << nemin;
        }
    }
}

TEST(LeanEngine, RandomIndefinite) {
    std::mt19937 rng(777);
    for (int n : {3, 17, 64, 200, 450}) {
        for (double pu : {0.01, 0.1, 0.5}) {
            for (int nemin : {1, 8}) {
                const CscLower A = make_random_sym(n, 6.0 / n, false, rng);
                Control ctrl;
                ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
                ctrl.pivot_u = pu;
                ctrl.nemin = nemin;
                const RunResult rr = factor_and_solve(A, ctrl, rng);
                if (rr.status != FactorStatus::Success) continue; // genuinely singular draw
                EXPECT_EQ(rr.info.num_positive + rr.info.num_negative, n);
                EXPECT_LT(rr.berr, 1e-11) << "n=" << n << " u=" << pu << " nemin=" << nemin;
            }
        }
    }
}

TEST(LeanEngine, RandomSpd) {
    std::mt19937 rng(4242);
    for (int n : {1, 9, 50, 300, 800}) {
        for (int nemin : {1, 8, 32}) {
            const CscLower A = make_random_sym(n, std::min(1.0, 8.0 / n), true, rng);
            Control ctrl;
            ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
            ctrl.nemin = nemin;
            const RunResult rr = factor_and_solve(A, ctrl, rng);
            ASSERT_EQ(rr.status, FactorStatus::Success) << "n=" << n;
            EXPECT_LT(rr.berr, 1e-14) << "n=" << n << " nemin=" << nemin;
        }
    }
}

TEST(LeanEngine, RefactorReusesWorkspace) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> scale(0.8, 1.2);
    const CscLower A0 = make_kkt(120, 50, 0.04, rng);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    Solver solver;
    Info info;
    auto ak = solver.analyse(A0, ctrl, info);
    ASSERT_NE(ak, nullptr);
    FactorKeep fk;
    NumericWorkspace ws;
    ASSERT_EQ(solver.factor(*ak, A0, ctrl, info, fk, ws), FactorStatus::Success);
    for (int it = 0; it < 5; ++it) {
        CscLower A = A0;
        for (Int j = 0; j < A.n; ++j) {
            const Int k = A.col_ptr[static_cast<std::size_t>(j)];
            if (A.row_idx[static_cast<std::size_t>(k)] == j && A.values[static_cast<std::size_t>(k)] != 0.0)
                A.values[static_cast<std::size_t>(k)] *= scale(rng);
        }
        Info finfo;
        ASSERT_EQ(solver.factor(*ak, A, ctrl, finfo, fk, ws), FactorStatus::Success);
        EXPECT_EQ(ws.front_heap_allocations_last_call, 0) << "iter " << it;
        std::vector<double> b(static_cast<std::size_t>(A.n), 1.0), x = b;
        Info sinfo;
        ASSERT_EQ(solver.solve(fk, ctrl, sinfo, x.data(), static_cast<int>(A.n)), 0);
        EXPECT_LT(backward_error(A, x, b), 1e-13) << "iter " << it;
        EXPECT_EQ(finfo.num_negative, 50);
    }
}
