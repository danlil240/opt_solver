#include "smf/factor_posdef.hpp"
#include "smf/solve_backward.hpp"
#include "smf/solve_diag.hpp"
#include "smf/solve_forward.hpp"
#include "smf/solver.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace smf;

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

/// Diagonal 5×5 SPD lower-CSC: A = diag(1,2,3,4,5)
static CscLower make_diag5() {
  CscLower A;
  A.n = 5;
  A.col_ptr = {0, 1, 2, 3, 4, 5};
  A.row_idx = {0, 1, 2, 3, 4};
  A.values = {1.0, 2.0, 3.0, 4.0, 5.0};
  return A;
}

/// Tridiagonal 4×4 SPD lower-CSC: diag=4, subdiag=-1.
/// Lower triangle: col j stores (j,j) and (j+1,j) entries.
static CscLower make_tridiag4() {
  CscLower A;
  A.n = 4;
  A.col_ptr = {0, 2, 4, 6, 7};
  A.row_idx = {0, 1, 1, 2, 2, 3, 3};
  A.values = {4.0, -1.0, 4.0, -1.0, 4.0, -1.0, 4.0};
  return A;
}

/// Indefinite 2×2 lower-CSC: A = [[2,1],[1,-1]]
static CscLower make_indef2() {
  CscLower A;
  A.n = 2;
  A.col_ptr = {0, 2, 3};
  A.row_idx = {0, 1, 1};
  A.values = {2.0, 1.0, -1.0};
  return A;
}

// Full symmetric matrix-vector multiply for the 4×4 tridiagonal:
//   Ax[0] = 4*x[0] - x[1]
//   Ax[1] = -x[0] + 4*x[1] - x[2]
//   Ax[2] = -x[1] + 4*x[2] - x[3]
//   Ax[3] = -x[2] + 4*x[3]
static void matvec_tridiag4(const double *x, double *Ax) {
  Ax[0] = 4.0 * x[0] - x[1];
  Ax[1] = -x[0] + 4.0 * x[1] - x[2];
  Ax[2] = -x[1] + 4.0 * x[2] - x[3];
  Ax[3] = -x[2] + 4.0 * x[3];
}

// ---------------------------------------------------------------------------
// Test 1: SPD 5×5 diagonal — direct calls to factor_posdef / solve_forward /
//         solve_backward.  Expected x = {1,1,1,1,1}.
// ---------------------------------------------------------------------------
TEST(SolveEndToEnd, SPD5x5DiagonalDirect) {
  CscLower A = make_diag5();
  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
  Info info;

  Solver s;
  auto ak_ptr = s.analyse(A, ctrl, info);
  ASSERT_NE(ak_ptr.get(), nullptr);
  ASSERT_EQ(info.status, ErrorCode::Success);

  FactorKeep fk;
  const FactorStatus fs = factor_posdef(*ak_ptr, ctrl, info, fk);
  ASSERT_EQ(fs, FactorStatus::Success);

  double b[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  solve_forward(fk, b, 5);
  solve_diag(fk, b, 5); // no-op for SPD
  solve_backward(fk, b, 5);

  for (int i = 0; i < 5; ++i)
    EXPECT_NEAR(b[i], 1.0, 1e-10) << "index i=" << i;
}

// ---------------------------------------------------------------------------
// Test 2: SPD 4×4 tridiagonal — full pipeline via Solver.
//         Verify ‖Ax − b‖ < 1e-10.
// ---------------------------------------------------------------------------
TEST(SolveEndToEnd, SPD4x4TridiagonalPipeline) {
  CscLower A = make_tridiag4();
  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
  Info info;

  Solver s;
  auto ak_ptr = s.analyse(A, ctrl, info);
  ASSERT_NE(ak_ptr.get(), nullptr);

  FactorKeep fk;
  const FactorStatus fs = s.factor(*ak_ptr, ctrl, info, fk);
  ASSERT_EQ(fs, FactorStatus::Success);

  const double b0[4] = {1.0, 0.0, 0.0, 1.0};
  double b[4] = {b0[0], b0[1], b0[2], b0[3]};
  const int rc = s.solve(fk, ctrl, info, b, 4);
  ASSERT_EQ(rc, 0);

  double Ax[4];
  matvec_tridiag4(b, Ax);

  double norm2 = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double r = Ax[i] - b0[i];
    norm2 += r * r;
  }
  EXPECT_LT(std::sqrt(norm2), 1e-10);
}

// ---------------------------------------------------------------------------
// Test 3: Indefinite 2×2 — full pipeline via Solver.
//         A = [[2,1],[1,-1]], b = {3,0}.  det(A) = -3.
//         A⁻¹ = (1/-3)[[-1,-1],[-1,2]]  →  x = {1,1}.
//         Verify ‖Ax − b‖ < 1e-6.
// ---------------------------------------------------------------------------
TEST(SolveEndToEnd, Indef2x2Pipeline) {
  CscLower A = make_indef2();
  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
  Info info;

  Solver s;
  auto ak_ptr = s.analyse(A, ctrl, info);
  ASSERT_NE(ak_ptr.get(), nullptr);

  FactorKeep fk;
  const FactorStatus fs = s.factor(*ak_ptr, ctrl, info, fk);
  ASSERT_EQ(fs, FactorStatus::Success);

  const double b0[2] = {3.0, 0.0};
  double b[2] = {b0[0], b0[1]};
  const int rc = s.solve(fk, ctrl, info, b, 2);
  ASSERT_EQ(rc, 0);

  // Full symmetric matrix-vector multiply: [[2,1],[1,-1]] * x
  const double Ax0 = 2.0 * b[0] + 1.0 * b[1];
  const double Ax1 = 1.0 * b[0] - 1.0 * b[1];

  const double r0 = Ax0 - b0[0];
  const double r1 = Ax1 - b0[1];
  EXPECT_LT(std::sqrt(r0 * r0 + r1 * r1), 1e-6);
}

// ---------------------------------------------------------------------------
// Test 4: Relative residual check for SPD 4×4.
//         ‖Ax − b‖ / ‖b‖ < 1e-10.
// ---------------------------------------------------------------------------
TEST(SolveEndToEnd, SPD4x4ResidualCheck) {
  CscLower A = make_tridiag4();
  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
  Info info;

  Solver s;
  auto ak_ptr = s.analyse(A, ctrl, info);
  ASSERT_NE(ak_ptr.get(), nullptr);

  FactorKeep fk;
  const FactorStatus fs = s.factor(*ak_ptr, ctrl, info, fk);
  ASSERT_EQ(fs, FactorStatus::Success);

  const double b0[4] = {1.0, 0.0, 0.0, 1.0};
  double b[4] = {b0[0], b0[1], b0[2], b0[3]};
  const int rc = s.solve(fk, ctrl, info, b, 4);
  ASSERT_EQ(rc, 0);

  double Ax[4];
  matvec_tridiag4(b, Ax);

  double res_norm2 = 0.0;
  double b_norm2 = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double r = Ax[i] - b0[i];
    res_norm2 += r * r;
    b_norm2 += b0[i] * b0[i];
  }
  EXPECT_LT(std::sqrt(res_norm2) / std::sqrt(b_norm2), 1e-10);
}
