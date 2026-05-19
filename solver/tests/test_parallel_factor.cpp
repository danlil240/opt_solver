// test_parallel_factor.cpp — M6.A1 acceptance test
//
// Verifies that factor_posdef with num_threads > 1 produces the same
// numerical result as the serial path for a block-diagonal SPD matrix
// (4 independent 20×20 blocks → 4 separate roots in the assembly tree).
//
// Timing speedup is printed as informational output; no timing assertion
// is made because 20×20 blocks are too small to reliably show speedup in CI.

#include "smf/factor_posdef.hpp"
#include "smf/solver.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

using namespace smf;

// ---------------------------------------------------------------------------
// Matrix builder helpers
// ---------------------------------------------------------------------------

/// Build a block-diagonal SPD lower-CSC matrix with @p num_blocks blocks of
/// size @p block_size each.  Each block is a tridiagonal matrix with diagonal
/// 4 and sub-diagonal -1 (symmetric positive definite by diagonal dominance).
static CscLower make_block_diagonal_spd(int block_size = 20,
                                        int num_blocks = 4) {
  const int n = block_size * num_blocks;
  CscLower A;
  A.n = n;
  A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);

  // Count nnz per column in lower triangle (diagonal + sub-diagonal).
  for (int b = 0; b < num_blocks; ++b) {
    for (int lj = 0; lj < block_size; ++lj) {
      const int gj = b * block_size + lj;
      int cnt = 1;                     // diagonal
      if (lj + 1 < block_size) ++cnt; // sub-diagonal
      A.col_ptr[static_cast<std::size_t>(gj) + 1] = cnt;
    }
  }
  // Prefix sum
  for (int j = 0; j < n; ++j)
    A.col_ptr[static_cast<std::size_t>(j) + 1] +=
        A.col_ptr[static_cast<std::size_t>(j)];

  const int nnz = A.col_ptr[static_cast<std::size_t>(n)];
  A.row_idx.resize(static_cast<std::size_t>(nnz));
  A.values.resize(static_cast<std::size_t>(nnz));

  // Fill entries.
  for (int b = 0; b < num_blocks; ++b) {
    for (int lj = 0; lj < block_size; ++lj) {
      const int gj = b * block_size + lj;
      int pos = A.col_ptr[static_cast<std::size_t>(gj)];
      // Diagonal
      A.row_idx[static_cast<std::size_t>(pos)] = gj;
      A.values[static_cast<std::size_t>(pos)] = 4.0;
      ++pos;
      // Sub-diagonal
      if (lj + 1 < block_size) {
        A.row_idx[static_cast<std::size_t>(pos)] = gj + 1;
        A.values[static_cast<std::size_t>(pos)] = -1.0;
        ++pos;
      }
    }
  }

  return A;
}

// ---------------------------------------------------------------------------
// Residual helper
// ---------------------------------------------------------------------------

/// Compute ||A*x - b||_inf / max(1, ||b||_inf) after a solve.
/// A is given in lower-CSC (symmetric); b is the original RHS; x is the
/// solution (stored in b_in_x).
static double compute_residual(const CscLower &A, const std::vector<double> &b,
                                const std::vector<double> &x) {
  const int n = A.n;
  std::vector<double> r(static_cast<std::size_t>(n), 0.0);

  // Multiply: r = A * x  (using both L and L^T since A is symmetric lower)
  for (int j = 0; j < n; ++j) {
    for (int k = A.col_ptr[static_cast<std::size_t>(j)];
         k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
      const int i = A.row_idx[static_cast<std::size_t>(k)];
      const double v = A.values[static_cast<std::size_t>(k)];
      r[static_cast<std::size_t>(i)] +=
          v * x[static_cast<std::size_t>(j)]; // lower: A[i][j] * x[j]
      if (i != j) {
        r[static_cast<std::size_t>(j)] +=
            v * x[static_cast<std::size_t>(i)]; // upper: A[j][i] * x[i]
      }
    }
  }

  // Compute ||r - b||_inf
  double norm_r = 0.0;
  double norm_b = 0.0;
  for (int i = 0; i < n; ++i) {
    r[static_cast<std::size_t>(i)] -= b[static_cast<std::size_t>(i)];
    norm_r = std::max(norm_r, std::abs(r[static_cast<std::size_t>(i)]));
    norm_b = std::max(norm_b, std::abs(b[static_cast<std::size_t>(i)]));
  }

  return norm_r / std::max(1.0, norm_b);
}

// ---------------------------------------------------------------------------
// Test: correctness of parallel factorization
// ---------------------------------------------------------------------------

TEST(ParallelFactor, BlockDiagonalSPD_CorrectResult) {
  const int block_size = 20;
  const int num_blocks = 4;
  const int n = block_size * num_blocks;

  CscLower A = make_block_diagonal_spd(block_size, num_blocks);

  // Build a non-trivial RHS: b[i] = 1.0 + 0.1*(i % 7)
  std::vector<double> b_serial(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    b_serial[static_cast<std::size_t>(i)] =
        1.0 + 0.1 * static_cast<double>(i % 7);
  std::vector<double> b_parallel = b_serial;
  const std::vector<double> b_orig = b_serial; // keep original

  Solver solver;

  // ---- Serial run --------------------------------------------------------
  Info info_s;
  Control ctrl_s;
  ctrl_s.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
  ctrl_s.num_threads = 1;

  auto t0s = std::chrono::steady_clock::now();
  auto ak = solver.analyse(A, ctrl_s, info_s);
  ASSERT_NE(ak.get(), nullptr) << "Serial analyse failed";

  FactorKeep fk_serial;
  const FactorStatus fs = solver.factor(*ak, ctrl_s, info_s, fk_serial);
  ASSERT_EQ(fs, FactorStatus::Success) << "Serial factor failed";

  auto t1s = std::chrono::steady_clock::now();
  const double t_serial =
      std::chrono::duration<double>(t1s - t0s).count();

  // Solve with serial result
  int ret_s =
      solver.solve(fk_serial, ctrl_s, info_s, b_serial.data(), n);
  ASSERT_EQ(ret_s, 0) << "Serial solve failed";

  const double res_serial = compute_residual(A, b_orig, b_serial);
  EXPECT_LT(res_serial, 1e-9)
      << "Serial residual too large: " << res_serial;

  // ---- Parallel run (num_threads=4) --------------------------------------
  Info info_p;
  Control ctrl_p;
  ctrl_p.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
  ctrl_p.num_threads = 4;

  auto t0p = std::chrono::steady_clock::now();
  // Reuse same analysis (independent of num_threads)
  FactorKeep fk_parallel;
  const FactorStatus fp = solver.factor(*ak, ctrl_p, info_p, fk_parallel);
  ASSERT_EQ(fp, FactorStatus::Success) << "Parallel factor failed";

  auto t1p = std::chrono::steady_clock::now();
  const double t_parallel =
      std::chrono::duration<double>(t1p - t0p).count();

  // Solve with parallel result
  int ret_p =
      solver.solve(fk_parallel, ctrl_p, info_p, b_parallel.data(), n);
  ASSERT_EQ(ret_p, 0) << "Parallel solve failed";

  const double res_parallel = compute_residual(A, b_orig, b_parallel);
  EXPECT_LT(res_parallel, 1e-9)
      << "Parallel residual too large: " << res_parallel;

  // ---- Compare serial vs parallel solutions ------------------------------
  double max_diff = 0.0;
  for (int i = 0; i < n; ++i) {
    max_diff = std::max(
        max_diff, std::abs(b_serial[static_cast<std::size_t>(i)] -
                           b_parallel[static_cast<std::size_t>(i)]));
  }
  EXPECT_LT(max_diff, 1e-10)
      << "Serial/parallel solutions differ: max |diff| = " << max_diff;

  // ---- Timing (informational, no assertion) ------------------------------
  const double speedup = (t_parallel > 0.0) ? t_serial / t_parallel : 0.0;
  std::printf(
      "[ParallelFactor] serial=%.4fs  parallel(4T)=%.4fs  speedup=%.2fx\n",
      t_serial, t_parallel, speedup);
  std::printf("[ParallelFactor] serial residual=%.2e  parallel residual=%.2e\n",
              res_serial, res_parallel);
}

// ---------------------------------------------------------------------------
// Test: parallel factor returns correct result on larger block-diagonal
// ---------------------------------------------------------------------------

TEST(ParallelFactor, BlockDiagonalSPD_MultipleBlocks) {
  // 8 independent blocks of 15×15 = 120×120 total
  const int block_size = 15;
  const int num_blocks = 8;
  const int n = block_size * num_blocks;

  CscLower A = make_block_diagonal_spd(block_size, num_blocks);

  std::vector<double> rhs(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    rhs[static_cast<std::size_t>(i)] = static_cast<double>(i + 1);
  const std::vector<double> rhs_orig = rhs;

  Solver solver;
  Info info;
  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
  ctrl.num_threads = 4;

  auto ak = solver.analyse(A, ctrl, info);
  ASSERT_NE(ak.get(), nullptr);

  FactorKeep fk;
  ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
  ASSERT_EQ(solver.solve(fk, ctrl, info, rhs.data(), n), 0);

  const double res = compute_residual(A, rhs_orig, rhs);
  EXPECT_LT(res, 1e-9) << "Residual too large: " << res;
}

// ---------------------------------------------------------------------------
// Test: num_threads=1 (serial path) still works via factor_posdef
// ---------------------------------------------------------------------------

TEST(ParallelFactor, SerialPathUnchanged) {
  const int block_size = 10;
  const int num_blocks = 3;
  const int n = block_size * num_blocks;

  CscLower A = make_block_diagonal_spd(block_size, num_blocks);

  std::vector<double> rhs(static_cast<std::size_t>(n), 1.0);
  const std::vector<double> rhs_orig = rhs;

  Solver solver;
  Info info;
  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
  ctrl.num_threads = 1; // explicitly serial

  auto ak = solver.analyse(A, ctrl, info);
  ASSERT_NE(ak.get(), nullptr);

  FactorKeep fk;
  ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
  ASSERT_EQ(solver.solve(fk, ctrl, info, rhs.data(), n), 0);

  const double res = compute_residual(A, rhs_orig, rhs);
  EXPECT_LT(res, 1e-9) << "Serial residual too large: " << res;
}
