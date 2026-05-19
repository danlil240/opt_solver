/// test_singular.cpp  — M5.S2 acceptance tests for singular / near-singular
/// handling in factor_indef.
///
/// Tests:
///   Singular.ZeroRow            – matrix with a structurally zero row/column
///   Singular.RankDeficientKKT   – deliberately rank-deficient KKT block
///   Singular.NearZeroPivot      – diagonal pivot below ctrl.small_pivot
///   Singular.ContinueOnSingular – factorisation completes and returns Singular
///                                  when ctrl.continue_on_singular = true

#include "smf/control.hpp"
#include "smf/factor_indef.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace smf;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a CscLower from explicit triplet lists (lower-triangle only).
/// Entries must be given in CSC order (col-major, rows increasing within col).
static CscLower make_csc(int n,
                          const std::vector<int> &col_ptr,
                          const std::vector<int> &row_idx,
                          const std::vector<double> &values) {
    CscLower A;
    A.n       = n;
    A.col_ptr = col_ptr;
    A.row_idx = row_idx;
    A.values  = values;
    return A;
}

// ---------------------------------------------------------------------------
// Singular.ZeroRow
//
//  A = [ 2   0   0 ]
//      [ 0   0   0 ]   ← row / col 1 is entirely zero
//      [ 0   0   3 ]
//
//  Lower CSC:
//    col 0: (0,2)
//    col 1: (empty)
//    col 2: (2,3)
// ---------------------------------------------------------------------------
TEST(Singular, ZeroRow) {
    CscLower A = make_csc(3,
        /*col_ptr*/ {0, 1, 1, 2},
        /*row_idx*/ {0,       2},
        /*values */ {2.0,     3.0});

    Control ctrl;
    ctrl.matrix_type       = MatrixType::RealSymmetricIndefinite;
    ctrl.small_pivot       = 1e-20;
    ctrl.continue_on_singular = false;

    Info     info;
    Solver   solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr) << "analyse failed";

    FactorKeep fk;
    FactorStatus st = solver.factor(*ak, ctrl, info, fk);

    EXPECT_EQ(st, FactorStatus::Singular)      << "expected Singular for zero row";
    EXPECT_GE(info.num_zero, 1)                << "expected at least one zero pivot";
    EXPECT_LT(info.numerical_rank, ak->n)      << "numerical_rank must be < n";
}

// ---------------------------------------------------------------------------
// Singular.RankDeficientKKT
//
//  Saddle-point / KKT matrix (n=4, m=2):
//
//    K = [ H   B^T ]   H = diag(1, 1)  (2×2)
//        [ B    0  ]   B = [[1,0],[0,0]] (2×2, rank 1)
//
//  Full 4×4 symmetric:
//    [ 1  0  1  0 ]
//    [ 0  1  0  0 ]
//    [ 1  0  0  0 ]
//    [ 0  0  0  0 ]   ← row/col 3 is zero → rank deficiency
//
//  Lower CSC (rows >= col):
//    col 0: (0,1), (2,1)
//    col 1: (1,1)
//    col 2: (2,0)
//    col 3: (empty)
// ---------------------------------------------------------------------------
TEST(Singular, RankDeficientKKT) {
    CscLower A = make_csc(4,
        /*col_ptr*/ {0, 2, 3, 4, 4},
        /*row_idx*/ {0, 2, 1, 2},
        /*values */ {1.0, 1.0, 1.0, 0.0});
    // Note: value 0.0 at (2,2) — the (2,2) diagonal of this block is zero.
    // Row/col 3 entirely missing → treated as all-zero.

    Control ctrl;
    ctrl.matrix_type          = MatrixType::RealSymmetricIndefinite;
    ctrl.small_pivot          = 1e-20;
    ctrl.continue_on_singular = false;

    Info   info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    FactorStatus st = solver.factor(*ak, ctrl, info, fk);

    EXPECT_EQ(st, FactorStatus::Singular);
    EXPECT_LT(info.numerical_rank, ak->n);
}

// ---------------------------------------------------------------------------
// Singular.NearZeroPivot
//
//  2×2 matrix whose only acceptable 1×1 diagonal is way below small_pivot:
//    A = [ ε   0 ]    ε = small_pivot / 10  (definitely zero)
//        [ 0   1 ]
//
//  Since ε < small_pivot, choose_pivot returns Reject for col 0.
//  Col 1 is processed normally.
// ---------------------------------------------------------------------------
TEST(Singular, NearZeroPivot) {
    Control ctrl;
    ctrl.matrix_type          = MatrixType::RealSymmetricIndefinite;
    ctrl.small_pivot          = 1e-10;   // use a generous threshold
    ctrl.continue_on_singular = false;

    const double eps = ctrl.small_pivot / 10.0;  // definitely below threshold

    //  lower CSC:  col 0: (0, eps)   col 1: (1, 1.0)
    CscLower A = make_csc(2,
        /*col_ptr*/ {0, 1, 2},
        /*row_idx*/ {0, 1},
        /*values */ {eps, 1.0});

    Info   info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    FactorStatus st = solver.factor(*ak, ctrl, info, fk);

    EXPECT_EQ(st, FactorStatus::Singular)  << "near-zero pivot must be detected";
    EXPECT_GE(info.num_zero, 1)            << "zero pivot count must be >= 1";
    EXPECT_LT(info.numerical_rank, ak->n)  << "numerical_rank < n";
}

// ---------------------------------------------------------------------------
// Singular.ContinueOnSingular
//
//  Same zero-row matrix as ZeroRow, but with continue_on_singular = true.
//  Expected: status == Singular, factorisation completes (fkeep is populated),
//  and a (degraded) solve can be attempted without crashing.
// ---------------------------------------------------------------------------
TEST(Singular, ContinueOnSingular) {
    //  3×3 with a zero row/col at index 1
    CscLower A = make_csc(3,
        /*col_ptr*/ {0, 1, 1, 2},
        /*row_idx*/ {0,       2},
        /*values */ {2.0,     3.0});

    Control ctrl;
    ctrl.matrix_type          = MatrixType::RealSymmetricIndefinite;
    ctrl.small_pivot          = 1e-20;
    ctrl.continue_on_singular = true;   // ← key flag

    Info   info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    FactorStatus st = solver.factor(*ak, ctrl, info, fk);

    // Status must still be Singular
    EXPECT_EQ(st, FactorStatus::Singular)
        << "continue_on_singular does not suppress Singular status";

    // num_zero and numerical_rank must be consistent
    EXPECT_GE(info.num_zero,       1)     << "at least one zero pivot";
    EXPECT_LT(info.numerical_rank, ak->n) << "numerical_rank < n";

    // Factor storage must be non-empty (factorisation did complete)
    EXPECT_FALSE(fk.factor_values.empty())
        << "factor storage must be populated even for singular matrix";

    // Attempt a solve — should not crash (result may be garbage for the
    // singular variable, but the call must return without error)
    double rhs[3] = {1.0, 0.0, 1.0};
    int solve_status = solver.solve(fk, ctrl, info, rhs, 3);
    EXPECT_EQ(solve_status, 0) << "solve must not return an error code";
}
