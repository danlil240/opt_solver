/// test_cholmod_compare.cpp
///
/// Compares smf solver results against CHOLMOD on SPD matrices.
/// Gated by SMF_BUILD_CHOLMOD_COMPARE cmake option.
///
/// Test cases:
///   1. 5×5 tridiagonal SPD
///   2. 50×50 random SPD
///   3. 100×100 Poisson2D SPD
///
/// For each, asserts:
///   ‖x_smf − x_cholmod‖_∞ / ‖x_cholmod‖_∞ < 1e-10

#include "smf/csc_matrix.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <cholmod.h>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace smf;

// ---------------------------------------------------------------------------
// CHOLMOD fixture: manages Common lifetime
// ---------------------------------------------------------------------------

class CholmodFixture : public ::testing::Test {
protected:
    cholmod_common cc{};

    void SetUp() override {
        cholmod_start(&cc);
    }

    void TearDown() override {
        cholmod_finish(&cc);
    }

    /// Build a CHOLMOD sparse matrix from a lower-CSC smf matrix.
    /// stype = -1 → lower triangle stored
    cholmod_sparse *make_cholmod_sparse(const CscLower &A) {
        int n   = A.n;
        int nnz = static_cast<int>(A.row_idx.size());

        cholmod_sparse *C = cholmod_allocate_sparse(
            static_cast<size_t>(n), static_cast<size_t>(n),
            static_cast<size_t>(nnz),
            1 /*sorted*/, 1 /*packed*/,
            -1 /*stype: lower triangle*/,
            CHOLMOD_REAL, &cc);
        EXPECT_NE(C, nullptr);
        if (!C) return nullptr;

        // Copy col_ptr, row_idx, values
        auto *cp = static_cast<int *>(C->p);
        auto *ri = static_cast<int *>(C->i);
        auto *xv = static_cast<double *>(C->x);

        for (int j = 0; j <= n; ++j)   cp[j] = A.col_ptr[j];
        for (int k = 0; k < nnz; ++k) {
            ri[k] = A.row_idx[k];
            xv[k] = A.values[k];
        }

        return C;
    }

    /// Solve A x = rhs using CHOLMOD; returns x (size n).
    std::vector<double> cholmod_solve_vec(const CscLower &A,
                                          const std::vector<double> &rhs) {
        int n = A.n;
        cholmod_sparse *C = make_cholmod_sparse(A);
        EXPECT_NE(C, nullptr);

        cholmod_factor *L = cholmod_analyze(C, &cc);
        EXPECT_NE(L, nullptr);

        int ok = cholmod_factorize(C, L, &cc);
        EXPECT_NE(ok, 0);

        // Build dense RHS
        cholmod_dense *B = cholmod_zeros(static_cast<size_t>(n), 1,
                                         CHOLMOD_REAL, &cc);
        EXPECT_NE(B, nullptr);
        std::memcpy(B->x, rhs.data(), static_cast<size_t>(n) * sizeof(double));

        cholmod_dense *X = cholmod_solve(CHOLMOD_A, L, B, &cc);
        EXPECT_NE(X, nullptr);

        std::vector<double> result(n);
        std::memcpy(result.data(), X->x, static_cast<size_t>(n) * sizeof(double));

        // Cleanup
        cholmod_free_dense(&X, &cc);
        cholmod_free_dense(&B, &cc);
        cholmod_free_factor(&L, &cc);
        cholmod_free_sparse(&C, &cc);

        return result;
    }
};

// ---------------------------------------------------------------------------
// Helper: solve with smf, return x (size n)
// ---------------------------------------------------------------------------

static std::vector<double> smf_solve_vec(const CscLower &A,
                                          const std::vector<double> &rhs) {
    int n = A.n;
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    Info info;

    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    if (!ak) return {};

    FactorKeep fk;
    FactorStatus fs = solver.factor(*ak, ctrl, info, fk);
    if (fs != FactorStatus::Success) return {};

    std::vector<double> x(rhs.begin(), rhs.end());
    int ret = solver.solve(fk, ctrl, info, x.data(), n, 1);
    if (ret != 0) return {};
    return x;
}

// ---------------------------------------------------------------------------
// Helper: inf-norm relative error
// ---------------------------------------------------------------------------

static double rel_inf_error(const std::vector<double> &a,
                             const std::vector<double> &b) {
    EXPECT_EQ(a.size(), b.size());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, std::abs(a[i] - b[i]));
        den = std::max(den, std::abs(b[i]));
    }
    if (den == 0.0) return (num == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
    return num / den;
}

// ---------------------------------------------------------------------------
// Test 1: 5×5 tridiagonal SPD  (diag=4, subdiag=-1)
// ---------------------------------------------------------------------------

TEST_F(CholmodFixture, Tridiag5x5) {
    const int n = 5;
    // Build lower CSC for 5×5 tridiagonal: diag=4, subdiag=-1
    // col 0: rows {0,1}; col 1: rows {1,2}; ... col 4: rows {4}
    CscLower A;
    A.n = n;
    A.col_ptr.resize(n + 1);
    int nnz = 0;
    for (int j = 0; j < n; ++j) {
        A.col_ptr[j] = nnz;
        A.row_idx.push_back(j);      // diagonal
        A.values.push_back(4.0);
        nnz++;
        if (j + 1 < n) {
            A.row_idx.push_back(j + 1); // subdiagonal
            A.values.push_back(-1.0);
            nnz++;
        }
    }
    A.col_ptr[n] = nnz;

    std::vector<double> rhs(n);
    for (int i = 0; i < n; ++i) rhs[i] = static_cast<double>(i + 1);

    auto x_smf    = smf_solve_vec(A, rhs);
    auto x_cholmd = cholmod_solve_vec(A, rhs);

    ASSERT_EQ(x_smf.size(), static_cast<std::size_t>(n));
    ASSERT_EQ(x_cholmd.size(), static_cast<std::size_t>(n));

    double err = rel_inf_error(x_smf, x_cholmd);
    EXPECT_LT(err, 1e-10)
        << "Tridiag5x5: relative inf-norm error = " << err;
}

// ---------------------------------------------------------------------------
// Test 2: 50×50 random SPD
// ---------------------------------------------------------------------------

static CscLower make_random_spd(int n, unsigned seed = 42) {
    // A = L * L^T + n*I  where L is random lower-triangular
    // We store only the lower triangle as lower-CSC.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    // Dense lower triangle storage (column-major)
    std::vector<double> dense(static_cast<std::size_t>(n * n), 0.0);
    auto at = [&](int r, int c) -> double & {
        return dense[static_cast<std::size_t>(c * n + r)];
    };

    // Build L (random lower triangular with unit diagonal)
    std::vector<double> L(static_cast<std::size_t>(n * n), 0.0);
    auto L_at = [&](int r, int c) -> double & {
        return L[static_cast<std::size_t>(c * n + r)];
    };
    for (int j = 0; j < n; ++j) {
        L_at(j, j) = 1.0;
        for (int i = j + 1; i < n; ++i)
            L_at(i, j) = dist(rng);
    }

    // Compute A = L*L^T + n*I (lower triangle only)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += L_at(i, k) * L_at(j, k);
            at(i, j) = s;
        }
        at(i, i) += static_cast<double>(n); // diagonal dominance
    }

    // Convert to lower-CSC (only store lower triangle, i >= j)
    CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n + 1));
    int nnz = 0;
    for (int j = 0; j < n; ++j) {
        A.col_ptr[j] = nnz;
        for (int i = j; i < n; ++i) {
            double v = at(i, j);
            if (v != 0.0) {
                A.row_idx.push_back(i);
                A.values.push_back(v);
                nnz++;
            }
        }
    }
    A.col_ptr[n] = nnz;
    return A;
}

TEST_F(CholmodFixture, RandomSPD50x50) {
    const int n = 50;
    CscLower A = make_random_spd(n, 12345);

    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> rhs(static_cast<std::size_t>(n));
    for (auto &v : rhs) v = dist(rng);

    auto x_smf    = smf_solve_vec(A, rhs);
    auto x_cholmd = cholmod_solve_vec(A, rhs);

    ASSERT_EQ(x_smf.size(), static_cast<std::size_t>(n));
    ASSERT_EQ(x_cholmd.size(), static_cast<std::size_t>(n));

    double err = rel_inf_error(x_smf, x_cholmd);
    EXPECT_LT(err, 1e-10)
        << "RandomSPD50x50: relative inf-norm error = " << err;
}

// ---------------------------------------------------------------------------
// Test 3: 100×100 large random SPD (different from 50×50 test)
// ---------------------------------------------------------------------------

static CscLower make_random_spd_100(unsigned seed = 777) {
    return make_random_spd(100, seed);
}

TEST_F(CholmodFixture, RandomSPD100x100) {
    const int n = 100;
    CscLower A = make_random_spd_100(777);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> rhs(static_cast<std::size_t>(n));
    for (auto &v : rhs) v = dist(rng);

    auto x_smf    = smf_solve_vec(A, rhs);
    auto x_cholmd = cholmod_solve_vec(A, rhs);

    ASSERT_EQ(x_smf.size(), static_cast<std::size_t>(n));
    ASSERT_EQ(x_cholmd.size(), static_cast<std::size_t>(n));

    double err = rel_inf_error(x_smf, x_cholmd);
    EXPECT_LT(err, 1e-10)
        << "RandomSPD100x100: relative inf-norm error = " << err;
}
