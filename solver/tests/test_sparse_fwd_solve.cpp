// test_sparse_fwd_solve.cpp — Tests for sparse forward solve (M9.S3).
//
// Tests verify that solve_sparse_forward() produces bit-identical results
// to the dense Solver::solve(SolveJob::Forward, ...) path, while only
// processing the minimal set of supernodes (the "reach" of the RHS
// non-zero pattern).

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/info.hpp"
#include "smf/solve_forward.hpp"
#include "smf/solve_sparse_fwd.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

using namespace smf;

// ===========================================================================
// Matrix helpers
// ===========================================================================

/// Build an n×n tridiagonal SPD lower-CSC matrix: diag=4, sub-diag=-1.
static CscLower make_tridiag(int n) {
    CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n + 1));
    // Each interior column has 2 entries (diagonal + subdiagonal); last col 1.
    A.col_ptr[0] = 0;
    for (int j = 0; j < n; ++j)
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            A.col_ptr[static_cast<std::size_t>(j)] + (j < n - 1 ? 2 : 1);

    const int nnz = A.col_ptr[static_cast<std::size_t>(n)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    int idx = 0;
    for (int j = 0; j < n; ++j) {
        A.row_idx[static_cast<std::size_t>(idx)] = j;
        A.values[static_cast<std::size_t>(idx)] = 4.0;
        ++idx;
        if (j < n - 1) {
            A.row_idx[static_cast<std::size_t>(idx)] = j + 1;
            A.values[static_cast<std::size_t>(idx)] = -1.0;
            ++idx;
        }
    }
    return A;
}

/// Build a block-diagonal n×n lower-CSC matrix.
/// n must be divisible by block_size.  Each block is an SPD tridiagonal:
///   diag = 6.0, sub-diag = -1.0.
static CscLower make_block_diag(int n, int block_size) {
    CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n + 1));
    A.col_ptr[0] = 0;

    // Count entries per column (same as tridiag within each block).
    for (int j = 0; j < n; ++j) {
        const int pos_in_block = j % block_size;
        const int entries = (pos_in_block < block_size - 1) ? 2 : 1;
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            A.col_ptr[static_cast<std::size_t>(j)] + entries;
    }

    const int nnz = A.col_ptr[static_cast<std::size_t>(n)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    int idx = 0;
    for (int j = 0; j < n; ++j) {
        const int pos_in_block = j % block_size;
        A.row_idx[static_cast<std::size_t>(idx)] = j;          // diagonal
        A.values[static_cast<std::size_t>(idx)] = 6.0;
        ++idx;
        if (pos_in_block < block_size - 1) {
            A.row_idx[static_cast<std::size_t>(idx)] = j + 1;  // sub-diagonal
            A.values[static_cast<std::size_t>(idx)] = -1.0;
            ++idx;
        }
    }
    return A;
}

// ===========================================================================
// Helper: run the dense forward solve via Solver::solve(SolveJob::Forward)
// and return the resulting vector.
// ===========================================================================
static std::vector<double>
dense_fwd(const FactorKeep &fkeep, const std::vector<double> &b) {
    const int n = static_cast<int>(b.size());
    std::vector<double> x(b);
    // Use the low-level solve_forward directly (same as SolveJob::Forward).
    solve_forward(fkeep, x.data(), n, 1);
    return x;
}

// ===========================================================================
// Utility: infinity norm of a vector.
// ===========================================================================
static double inf_norm(const std::vector<double> &v) {
    double m = 0.0;
    for (double d : v)
        m = std::max(m, std::abs(d));
    return m;
}

// ===========================================================================
// Fixture: build analysis + factor for a given matrix.
// ===========================================================================
struct SpdSetup {
    std::unique_ptr<AnalysisKeep> ak;
    FactorKeep                    fk;
    int                           n = 0;

    bool build(const CscLower &A) {
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
        Info info;
        Solver s;
        ak = s.analyse(A, ctrl, info);
        if (!ak) return false;
        n = ak->n;
        if (s.factor(*ak, ctrl, info, fk) != FactorStatus::Success)
            return false;
        return true;
    }
};

// ===========================================================================
// Test 1 — DenseRHSMatchesDensePath
//
// 10×10 tridiagonal SPD, dense RHS (all-ones).
// solve_sparse_forward must give x_sparse == x_dense (inf-norm diff < 1e-14
// * ‖x_dense‖_∞).
// ===========================================================================
TEST(SparseFwdSolve, DenseRHSMatchesDensePath) {
    SpdSetup setup;
    ASSERT_TRUE(setup.build(make_tridiag(10)));
    const int n = setup.n;

    std::vector<double> b(static_cast<std::size_t>(n), 1.0);

    // Dense path.
    const std::vector<double> x_dense = dense_fwd(setup.fk, b);

    // Sparse path.
    std::vector<int>    reach;
    std::vector<double> x_sparse;
    solve_sparse_forward(*setup.ak, setup.fk, b, reach, x_sparse);

    ASSERT_EQ(static_cast<int>(x_sparse.size()), n);

    const double norm_dense = inf_norm(x_dense);
    const double tol = 1e-14 * std::max(norm_dense, 1.0);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x_sparse[static_cast<std::size_t>(i)],
                    x_dense[static_cast<std::size_t>(i)], tol)
            << "mismatch at index " << i;
    }
}

// ===========================================================================
// Test 2 — SparseRHSMatchesDensePath
//
// 10×10 tridiagonal SPD, sparse RHS = e_3 (unit vector at position 3).
// Both paths must give identical results.
// ===========================================================================
TEST(SparseFwdSolve, SparseRHSMatchesDensePath) {
    SpdSetup setup;
    ASSERT_TRUE(setup.build(make_tridiag(10)));
    const int n = setup.n;

    std::vector<double> b(static_cast<std::size_t>(n), 0.0);
    b[3] = 1.0;  // e_3

    const std::vector<double> x_dense = dense_fwd(setup.fk, b);

    std::vector<int>    reach;
    std::vector<double> x_sparse;
    solve_sparse_forward(*setup.ak, setup.fk, b, reach, x_sparse);

    ASSERT_EQ(static_cast<int>(x_sparse.size()), n);

    const double norm_dense = inf_norm(x_dense);
    const double tol = 1e-14 * std::max(norm_dense, 1.0);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x_sparse[static_cast<std::size_t>(i)],
                    x_dense[static_cast<std::size_t>(i)], tol)
            << "mismatch at index " << i;
    }
}

// ===========================================================================
// Test 3 — ReachSizeSmall
//
// 100×100 BLOCK-DIAGONAL matrix (10 blocks of 10×10 tridiagonal SPD),
// RHS = e_0 (only first entry nonzero).
//
// The block-diagonal structure ensures that the reach is confined to the
// supernodes belonging to block 0.  With ≤10 columns per block and the
// assembler merging columns into supernodes, at most ~block_size supernodes
// are needed — well below n/10 = 10.
//
// Correctness: x_sparse == x_dense.
// ===========================================================================
TEST(SparseFwdSolve, ReachSizeSmall) {
    const int n = 100;
    const int block_size = 10;

    SpdSetup setup;
    ASSERT_TRUE(setup.build(make_block_diag(n, block_size)));

    std::vector<double> b(static_cast<std::size_t>(n), 0.0);
    b[0] = 1.0;  // e_0

    const std::vector<double> x_dense = dense_fwd(setup.fk, b);

    std::vector<int>    reach;
    std::vector<double> x_sparse;
    solve_sparse_forward(*setup.ak, setup.fk, b, reach, x_sparse);

    // Correctness.
    ASSERT_EQ(static_cast<int>(x_sparse.size()), n);
    const double norm_dense = inf_norm(x_dense);
    const double tol = 1e-14 * std::max(norm_dense, 1.0);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x_sparse[static_cast<std::size_t>(i)],
                    x_dense[static_cast<std::size_t>(i)], tol)
            << "mismatch at index " << i;
    }

    // Sparsity: only the supernodes in block 0 should be processed.
    // block 0 has block_size = 10 columns so at most block_size supernodes.
    EXPECT_LE(static_cast<int>(reach.size()), n / 10)
        << "reach.size()=" << reach.size()
        << " expected <= " << n / 10
        << " for block-diagonal matrix with sparse RHS e_0";
}

// ===========================================================================
// Test 4 — ZeroRHS
//
// Any matrix, RHS = all-zeros → x_out = all-zeros, reach empty.
// ===========================================================================
TEST(SparseFwdSolve, ZeroRHS) {
    SpdSetup setup;
    ASSERT_TRUE(setup.build(make_tridiag(10)));
    const int n = setup.n;

    const std::vector<double> b(static_cast<std::size_t>(n), 0.0);

    std::vector<int>    reach;
    std::vector<double> x_out;
    solve_sparse_forward(*setup.ak, setup.fk, b, reach, x_out);

    ASSERT_EQ(static_cast<int>(x_out.size()), n);
    EXPECT_TRUE(reach.empty()) << "reach should be empty for zero RHS";
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(x_out[static_cast<std::size_t>(i)], 0.0)
            << "x_out[" << i << "] should be 0 for zero RHS";
}

// ===========================================================================
// Test 5 — AllNonzeroRHS
//
// 10×10 tridiagonal SPD, RHS = all-ones.
// Reach covers all supernodes; result matches dense forward solve.
// ===========================================================================
TEST(SparseFwdSolve, AllNonzeroRHS) {
    SpdSetup setup;
    ASSERT_TRUE(setup.build(make_tridiag(10)));
    const int n = setup.n;
    const int ns = static_cast<int>(setup.ak->supernodes.size());

    std::vector<double> b(static_cast<std::size_t>(n), 1.0);

    const std::vector<double> x_dense = dense_fwd(setup.fk, b);

    std::vector<int>    reach;
    std::vector<double> x_sparse;
    solve_sparse_forward(*setup.ak, setup.fk, b, reach, x_sparse);

    // All supernodes in reach.
    EXPECT_EQ(static_cast<int>(reach.size()), ns)
        << "all " << ns << " supernodes should be in reach for all-nonzero RHS";

    // Correctness.
    const double norm_dense = inf_norm(x_dense);
    const double tol = 1e-14 * std::max(norm_dense, 1.0);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x_sparse[static_cast<std::size_t>(i)],
                    x_dense[static_cast<std::size_t>(i)], tol)
            << "mismatch at index " << i;
    }
}
