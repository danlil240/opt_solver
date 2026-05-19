#include "smf/solver.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace smf;

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

/// 3×3 SPD lower-CSC:
///   A = [ 4  2  0 ]
///       [ 2  5  1 ]
///       [ 0  1  6 ]
/// Eigenvalues all positive (det = 92 > 0, all leading minors > 0).
static CscLower make_spd3() {
  CscLower A;
  A.n = 3;
  A.col_ptr = {0, 2, 4, 5};
  A.row_idx = {0, 1, 1, 2, 2};
  A.values = {4.0, 2.0, 5.0, 1.0, 6.0};
  return A;
}

/// 2×2 indefinite lower-CSC:
///   A = [ 2  1 ]
///       [ 1 -1 ]
static CscLower make_indef2() {
  CscLower A;
  A.n = 2;
  A.col_ptr = {0, 2, 3};
  A.row_idx = {0, 1, 1};
  A.values = {2.0, 1.0, -1.0};
  return A;
}

// ---------------------------------------------------------------------------
// Test 1: SPD 3×3 — factor_solve vs separate analyse/factor/solve
// ---------------------------------------------------------------------------
TEST(FactorSolve, SPD3x3_vs_separate) {
  CscLower A = make_spd3();
  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;

  const double b0[3] = {1.0, 2.0, 3.0};

  // --- separate path ---
  double b_sep[3] = {b0[0], b0[1], b0[2]};
  {
    Info info;
    Solver s;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    ASSERT_EQ(s.factor(*ak, ctrl, info, fk), FactorStatus::Success);
    ASSERT_EQ(s.solve(fk, ctrl, info, b_sep, 3), 0);
  }

  // --- combined factor_solve path ---
  double b_comb[3] = {b0[0], b0[1], b0[2]};
  {
    Info info;
    Solver s;
    auto fk = s.factor_solve(A, ctrl, info, b_comb, 3);
    ASSERT_NE(fk.get(), nullptr);
  }

  for (int i = 0; i < 3; ++i)
    EXPECT_NEAR(b_comb[i], b_sep[i], 1e-12) << "index i=" << i;
}

// ---------------------------------------------------------------------------
// Test 2: Indefinite 2×2 — factor_solve vs separate analyse/factor/solve
// ---------------------------------------------------------------------------
TEST(FactorSolve, Indef2x2_vs_separate) {
  CscLower A = make_indef2();
  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;

  const double b0[2] = {3.0, 0.0};

  // --- separate path ---
  double b_sep[2] = {b0[0], b0[1]};
  {
    Info info;
    Solver s;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    ASSERT_EQ(s.factor(*ak, ctrl, info, fk), FactorStatus::Success);
    ASSERT_EQ(s.solve(fk, ctrl, info, b_sep, 2), 0);
  }

  // --- combined factor_solve path ---
  double b_comb[2] = {b0[0], b0[1]};
  {
    Info info;
    Solver s;
    auto fk = s.factor_solve(A, ctrl, info, b_comb, 2);
    ASSERT_NE(fk.get(), nullptr);
  }

  for (int i = 0; i < 2; ++i)
    EXPECT_NEAR(b_comb[i], b_sep[i], 1e-12) << "index i=" << i;
}
