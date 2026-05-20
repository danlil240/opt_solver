/// @file test_indef_larger.cpp
/// Regression tests for the indefinite (LDLᵀ) factorization accuracy bug
/// that caused large residuals for coupled matrices larger than 2×2.
///
/// Bug root cause (fixed in BugFix.IndefFactor mission):
///   1. sym_swap_front did not swap extension rows.
///   2. apply_pivot_1x1 / apply_pivot_2x2 did not update extension rows.
///   3. Contribution-block computation re-applied the D-inverse that was
///      already embedded in the (now-fixed) extension-row L entries.
///
/// These tests exercise:
///   - IndefLarger.Tridiag_4x4 : 4×4 tridiagonal indefinite
///   - IndefLarger.Tridiag_8x8 : 8×8 tridiagonal indefinite
///   - IndefLarger.Poisson2D_3x3_Indef : 9×9 2D Poisson via indef path (SPD
///     matrix, all pivots accepted, inertia = (9,0,0))

#include "smf/solver.hpp"
#include "smf/supernode.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace smf;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a lower-CSC tridiagonal matrix of order n.
///   diagonal[i]   = diag[i]
///   sub-diagonal  = offdiag (constant)
static CscLower make_tridiag(int n, const std::vector<double> &diag, double offdiag)
{
    CscLower A;
    A.n = n;
    A.col_ptr.reserve(static_cast<std::size_t>(n + 1));
    A.row_idx.reserve(static_cast<std::size_t>(2 * n - 1));
    A.values.reserve(static_cast<std::size_t>(2 * n - 1));

    int nnz = 0;
    for (int j = 0; j < n; ++j)
    {
        A.col_ptr.push_back(nnz);
        // diagonal
        A.row_idx.push_back(j);
        A.values.push_back(diag[static_cast<std::size_t>(j)]);
        ++nnz;
        // sub-diagonal
        if (j + 1 < n)
        {
            A.row_idx.push_back(j + 1);
            A.values.push_back(offdiag);
            ++nnz;
        }
    }
    A.col_ptr.push_back(nnz);
    return A;
}

/// 2D Poisson (5-point stencil) lower-CSC of order N×N.
static CscLower make_poisson2d(int N)
{
    const int n = N * N;
    CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n + 1), 0);
    A.row_idx.clear();
    A.values.clear();

    for (int k = 0; k < n; ++k)
    {
        int cnt = 1;
        int row = k / N, col = k % N;
        if (col < N - 1)
            ++cnt;
        if (row < N - 1)
            ++cnt;
        A.col_ptr[static_cast<std::size_t>(k + 1)] = cnt;
    }
    for (int k = 0; k < n; ++k)
        A.col_ptr[static_cast<std::size_t>(k + 1)] += A.col_ptr[static_cast<std::size_t>(k)];

    const int nnz = A.col_ptr[static_cast<std::size_t>(n)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    for (int k = 0; k < n; ++k)
    {
        int row = k / N, col = k % N;
        int p = A.col_ptr[static_cast<std::size_t>(k)];
        A.row_idx[static_cast<std::size_t>(p)] = k;
        A.values[static_cast<std::size_t>(p)] = 4.0;
        ++p;
        if (col < N - 1)
        {
            A.row_idx[static_cast<std::size_t>(p)] = k + 1;
            A.values[static_cast<std::size_t>(p)] = -1.0;
            ++p;
        }
        if (row < N - 1)
        {
            A.row_idx[static_cast<std::size_t>(p)] = k + N;
            A.values[static_cast<std::size_t>(p)] = -1.0;
            ++p;
        }
    }
    return A;
}

/// Full symmetric matvec  y = A_full * x
static std::vector<double> sym_matvec(const CscLower &A, const std::vector<double> &x)
{
    const int n = A.n;
    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j)
    {
        for (int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j + 1)]; ++p)
        {
            const int i = A.row_idx[static_cast<std::size_t>(p)];
            const double v = A.values[static_cast<std::size_t>(p)];
            y[static_cast<std::size_t>(i)] += v * x[static_cast<std::size_t>(j)];
            if (i != j)
                y[static_cast<std::size_t>(j)] += v * x[static_cast<std::size_t>(i)];
        }
    }
    return y;
}

/// Solve A·x = b with the indefinite factorization, return relative residual.
/// Returns 1e30 on any failure.
static double solve_indef(const CscLower &A, const std::vector<double> &b, smf::Info* info_out = nullptr)
{
    const int n = A.n;
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    Info info;

    Solver solver;
    auto ak_ptr = solver.analyse(A, ctrl, info);
    if (!ak_ptr)
        return 1e30;

    FactorKeep fk;
    const FactorStatus fs = solver.factor(*ak_ptr, ctrl, info, fk);
    if (fs != FactorStatus::Success)
        return 1e30;

    std::vector<double> x = b;
    const int rc = solver.solve(fk, ctrl, info, x.data(), n);
    if (rc != 0)
        return 1e30;

    if (info_out)
        *info_out = info;

    // residual r = A*x - b
    const std::vector<double> Ax = sym_matvec(A, x);
    double r2 = 0.0, b2 = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double r = Ax[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)];
        r2 += r * r;
        b2 += b[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
    }
    return std::sqrt(r2) / (std::sqrt(b2) > 0.0 ? std::sqrt(b2) : 1.0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(IndefLarger, NonAdjacentSupernodesAreNotAmalgamated)
{
    std::vector<Supernode> supernodes(4);
    for (int i = 0; i < 4; ++i)
    {
        supernodes[static_cast<std::size_t>(i)].col_start = i;
        supernodes[static_cast<std::size_t>(i)].col_end = i + 1;
        supernodes[static_cast<std::size_t>(i)].parent = -1;
    }

    supernodes[0].parent = 3;
    supernodes[3].children.push_back(0);

    const std::vector<Supernode> merged = amalgamate_supernodes(supernodes, 8);

    std::vector<int> covered(4, 0);
    int width_sum = 0;
    for (const Supernode &sn : merged)
    {
        width_sum += sn.width();
        for (Int col = sn.col_start; col < sn.col_end; ++col)
            ++covered[static_cast<std::size_t>(col)];
    }

    EXPECT_EQ(width_sum, 4);
    for (int count : covered)
        EXPECT_EQ(count, 1);
}

/// 4×4 indefinite tridiagonal: diag = [2,2,2,-2], sub-diag = -1.
///
///   A = [[ 2,-1, 0, 0],
///        [-1, 2,-1, 0],
///        [ 0,-1, 2,-1],
///        [ 0, 0,-1,-2]]
///
/// x_true = [1,1,1,1] → b = A*x_true = [1, 0, 0, -3].
TEST(IndefLarger, Tridiag_4x4)
{
    const std::vector<double> diag = {2.0, 2.0, 2.0, -2.0};
    CscLower A = make_tridiag(4, diag, -1.0);

    // b = A * x_true where x_true = ones
    const std::vector<double> x_true(4, 1.0);
    const std::vector<double> b = sym_matvec(A, x_true);

    Info info;
    const double rel = solve_indef(A, b, &info);
    EXPECT_LT(rel, 1e-9) << "Tridiag_4x4 relative residual=" << rel;

    // Inertia: sum must equal n=4; must have at least 1 negative eigenvalue.
    EXPECT_EQ(info.num_positive + info.num_negative + info.num_zero, 4);
    EXPECT_GE(info.num_negative, 1);
    EXPECT_GE(info.num_positive, 1);
}

/// 8×8 indefinite tridiagonal: diag = [2,2,2,2,2,2,2,-2], sub-diag = -1.
TEST(IndefLarger, Tridiag_8x8)
{
    std::vector<double> diag(8, 2.0);
    diag[7] = -2.0;
    CscLower A = make_tridiag(8, diag, -1.0);

    const std::vector<double> x_true(8, 1.0);
    const std::vector<double> b = sym_matvec(A, x_true);

    Info info;
    const double rel = solve_indef(A, b, &info);
    EXPECT_LT(rel, 1e-9) << "Tridiag_8x8 relative residual=" << rel;

    EXPECT_EQ(info.num_positive + info.num_negative + info.num_zero, 8);
    EXPECT_GE(info.num_negative, 1);
    EXPECT_GE(info.num_positive, 1);
}

/// 9×9 2D Poisson (3×3 grid) solved via the indefinite code path.
///
/// The matrix IS SPD, so the indefinite factorization should accept all pivots
/// and return inertia = (9, 0, 0).  This directly tests the bug case mentioned
/// in the mission (large residuals for 3×3 grids under RealSymmetricIndefinite).
TEST(IndefLarger, Poisson2D_3x3_Indef)
{
    const int N = 3;
    CscLower A = make_poisson2d(N);

    const std::vector<double> x_true(9, 1.0);
    const std::vector<double> b = sym_matvec(A, x_true);

    Info info;
    const double rel = solve_indef(A, b, &info);
    EXPECT_LT(rel, 1e-9) << "Poisson2D 3x3 (indef path) relative residual=" << rel;

    // SPD matrix → inertia should be (9, 0, 0)
    EXPECT_EQ(info.num_positive, 9);
    EXPECT_EQ(info.num_negative, 0);
    EXPECT_EQ(info.num_zero, 0);
}

/// 16×16 2D Poisson (4×4 grid) solved via the indefinite code path.
/// Further exercises multi-level supernodal contribution blocks.
TEST(IndefLarger, Poisson2D_4x4_Indef)
{
    const int N = 4;
    CscLower A = make_poisson2d(N);
    const int n = N * N;

    const std::vector<double> x_true(static_cast<std::size_t>(n), 1.0);
    const std::vector<double> b = sym_matvec(A, x_true);

    Info info;
    const double rel = solve_indef(A, b, &info);
    EXPECT_LT(rel, 1e-9) << "Poisson2D 4x4 (indef path) relative residual=" << rel;

    EXPECT_EQ(info.num_positive, n);
    EXPECT_EQ(info.num_negative, 0);
    EXPECT_EQ(info.num_zero, 0);
}
