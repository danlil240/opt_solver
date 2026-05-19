/// test_repeated_factor.cpp  — M5.S1
///
/// Verifies that repeated factorisation on a FIXED sparsity pattern:
///   (a) produces a correct solve (relative residual < 1e-9) every iteration,
///   (b) does not re-run symbolic analysis (same AnalysisKeep reused),
///   (c) allocates a constant peak-arena footprint (no memory growth after
///       the first call — arena_peak_bytes is the same every iteration).
///
/// Design note:
///   The smf Solver::factor() takes the numeric values from AnalysisKeep.cleaned
///   (set during analyse()).  To exercise repeated factorisation with different
///   numeric values but the SAME symbolic structure, the test directly updates
///   ak->cleaned.values before each factor() call.  This mirrors the intended
///   usage: analyse() once for the pattern, then update values + factor() N×.
///
/// Tests
///   RepeatedFactor.SPD5x5_100iters        – analyse once, factor/solve 100×
///   RepeatedFactor.ArenaConstant_100iters – arena_peak_bytes identical iter 1..100
///   RepeatedFactor.FactorSolve_50iters    – factor_solve 50×, same pattern check

#include "smf/analysis.hpp"
#include "smf/solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

using namespace smf;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic XOR-shift 32 pseudo-random generator.
/// Returns values in [0, 1).
class XorShift32 {
public:
  explicit XorShift32(uint32_t seed = 12345u) : state_(seed ? seed : 1u) {}
  double next_double() {
    state_ ^= state_ << 13u;
    state_ ^= state_ >> 17u;
    state_ ^= state_ << 5u;
    return static_cast<double>(state_) / static_cast<double>(0xFFFFFFFFu);
  }
  /// Return a scale factor in [1-pct, 1+pct].
  double scale(double pct) { return 1.0 + (2.0 * next_double() - 1.0) * pct; }

private:
  uint32_t state_;
};

/// 5×5 tridiagonal SPD (lower CSC):
///   diag = 20, sub-diag = 1.
/// Conditioning is very good; ±10 % diagonal perturbations keep it SPD.
///
/// Entries in col_ptr / row_idx / values order:
///   col 0: (0,0)=20, (1,0)=1
///   col 1: (1,1)=20, (2,1)=1
///   col 2: (2,2)=20, (3,2)=1
///   col 3: (3,3)=20, (4,3)=1
///   col 4: (4,4)=20
static CscLower make_base5() {
  CscLower A;
  A.n = 5;
  A.col_ptr = {0, 2, 4, 6, 8, 9};
  A.row_idx = {0, 1,  1, 2,  2, 3,  3, 4,  4};
  A.values  = {20.0, 1.0,  20.0, 1.0,  20.0, 1.0,  20.0, 1.0,  20.0};
  return A;
}

/// Full symmetric matrix–vector product using lower-CSC storage.
static void sym_matvec(const CscLower &A, const double *x, double *Ax) {
  const int n = A.n;
  std::fill(Ax, Ax + n, 0.0);
  for (int j = 0; j < n; ++j) {
    for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
      const int    i = A.row_idx[p];
      const double v = A.values[p];
      Ax[i] += v * x[j]; // lower-triangle / diagonal
      if (i != j)
        Ax[j] += v * x[i]; // symmetric upper-triangle
    }
  }
}

/// Relative residual  ‖Ax − b‖ / (‖A‖_F · ‖x‖ + ‖b‖).
static double relative_residual(const CscLower &A, const double *x,
                                const double *b) {
  const int n = A.n;
  std::vector<double> Ax(static_cast<std::size_t>(n));
  sym_matvec(A, x, Ax.data());

  double res2 = 0.0, b2 = 0.0, x2 = 0.0, AF2 = 0.0;
  for (int i = 0; i < n; ++i) {
    const double r = Ax[i] - b[i];
    res2 += r * r;
    b2   += b[i] * b[i];
    x2   += x[i] * x[i];
  }
  for (int j = 0; j < n; ++j) {
    for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
      const double v = A.values[p];
      AF2 += (A.row_idx[p] == j) ? v * v : 2.0 * v * v;
    }
  }
  const double denom = std::sqrt(AF2) * std::sqrt(x2) + std::sqrt(b2);
  return (denom > 0.0) ? std::sqrt(res2) / denom : std::sqrt(res2);
}

/// Apply value perturbations to a CscLower, returning a modified copy.
/// Diagonal entries are scaled by (1 ± diag_pct), off-diagonals by (1 ± off_pct).
/// Caller ensures the result stays SPD (use small fractions).
static CscLower perturb_values(const CscLower &base, XorShift32 &rng,
                               double diag_pct, double off_pct) {
  CscLower A = base;
  for (int j = 0; j < A.n; ++j) {
    for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
      if (A.row_idx[p] == j)
        A.values[p] *= rng.scale(diag_pct);
      else
        A.values[p] *= rng.scale(off_pct);
    }
  }
  return A;
}

// ---------------------------------------------------------------------------
// Test 1  –  100 repeated factorizations with value perturbations
// ---------------------------------------------------------------------------
TEST(RepeatedFactor, SPD5x5_100iters) {
  const CscLower base = make_base5();
  const int      n    = base.n;

  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;

  // ---- Analyse ONCE -------------------------------------------------------
  Solver solver;
  Info   info_analyse;
  auto   ak = solver.analyse(base, ctrl, info_analyse);
  ASSERT_NE(ak.get(), nullptr) << "Analysis failed";
  ASSERT_EQ(info_analyse.status, ErrorCode::Success);

  // ---- Perturbed factorization + solve loop --------------------------------
  // Diagonal perturbed by ±5 %, off-diagonal by ±10 % — matrix stays SPD
  // because diagonal dominance holds: diag_min ≈ 19 >> 4 * off_max ≈ 4.4.
  XorShift32 rng(0xDEADBEEFu);
  const int    NITERS   = 100;
  const double DIAG_PCT = 0.05;
  const double OFF_PCT  = 0.10;

  const std::vector<double> b0 = {1.0, 2.0, 3.0, 4.0, 5.0};

  for (int iter = 0; iter < NITERS; ++iter) {
    // Build perturbed matrix (same sparsity, new values).
    CscLower Ap = perturb_values(base, rng, DIAG_PCT, OFF_PCT);

    // Push new values into the cached AnalysisKeep so factor() uses them.
    // The cleaned matrix has the same structure as base (no duplicates /
    // out-of-range entries), so the value vector is aligned positionally.
    ASSERT_EQ(ak->cleaned.values.size(), Ap.values.size())
        << "Value vector size mismatch at iter=" << iter;
    ak->cleaned.values = Ap.values;

    // Factor using the cached symbolic analysis (no re-analyse)
    FactorKeep fk;
    Info       info_factor;
    const FactorStatus fs = solver.factor(*ak, ctrl, info_factor, fk);
    ASSERT_EQ(fs, FactorStatus::Success) << "factor() failed at iter=" << iter;

    // Solve A_p x = b
    std::vector<double> x(b0.begin(), b0.end());
    Info                info_solve;
    const int rc = solver.solve(fk, ctrl, info_solve, x.data(), n);
    ASSERT_EQ(rc, 0) << "solve() failed at iter=" << iter;

    // Residual check with the perturbed matrix
    const double rr = relative_residual(Ap, x.data(), b0.data());
    EXPECT_LT(rr, 1e-9)
        << "iter=" << iter << "  rel.residual=" << rr;
  }
}

// ---------------------------------------------------------------------------
// Test 2 – arena_peak_bytes constant across all iterations
//           arena_growths == 0 on every iteration (no reallocation)
// ---------------------------------------------------------------------------
TEST(RepeatedFactor, ArenaConstant_100iters) {
  const CscLower base = make_base5();
  const int      n    = base.n;
  (void)n;

  Control ctrl;
  ctrl.matrix_type              = MatrixType::RealSymmetricPositiveDefinite;
  ctrl.factor_memory_multiplier = 2.0; // generous headroom → growths == 0

  // Analyse once
  Solver solver;
  Info   info_a;
  auto   ak = solver.analyse(base, ctrl, info_a);
  ASSERT_NE(ak.get(), nullptr);

  XorShift32 rng(0xCAFEBABEu);
  const int    NITERS   = 100;
  const double DIAG_PCT = 0.05;
  const double OFF_PCT  = 0.10;

  long first_peak = -1;

  for (int iter = 0; iter < NITERS; ++iter) {
    CscLower Ap = perturb_values(base, rng, DIAG_PCT, OFF_PCT);

    // Update values in-place
    ak->cleaned.values = Ap.values;

    FactorKeep fk;
    Info       info_f;
    const FactorStatus fs = solver.factor(*ak, ctrl, info_f, fk);
    ASSERT_EQ(fs, FactorStatus::Success) << "iter=" << iter;

    // No arena reallocation: initial capacity (2× predicted fill) is enough
    EXPECT_EQ(info_f.arena_growths, 0)
        << "Unexpected arena growth at iter=" << iter;

    if (iter == 0) {
      first_peak = info_f.arena_peak_bytes;
      ASSERT_GT(first_peak, 0L) << "arena_peak_bytes should be > 0";
    } else {
      // Peak must be identical across all iterations (same pattern → same fill
      // → same initial capacity → same peak usage).
      EXPECT_EQ(info_f.arena_peak_bytes, first_peak)
          << "arena_peak_bytes changed at iter=" << iter;
    }
  }
}

// ---------------------------------------------------------------------------
// Test 3 – factor_solve repeated 50×  (each call re-analyses by design,
//           but verifies correct residuals every iteration)
// ---------------------------------------------------------------------------
TEST(RepeatedFactor, FactorSolve_50iters) {
  const CscLower base = make_base5();
  const int      n    = base.n;

  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;

  const std::vector<double> b0 = {5.0, 4.0, 3.0, 2.0, 1.0};

  Solver     solver;
  XorShift32 rng(0xBEEFCAFEu);
  const int    NITERS   = 50;
  const double DIAG_PCT = 0.05;
  const double OFF_PCT  = 0.10;

  for (int iter = 0; iter < NITERS; ++iter) {
    CscLower Ap = perturb_values(base, rng, DIAG_PCT, OFF_PCT);

    std::vector<double> x(b0.begin(), b0.end());
    Info                info;
    auto                fk = solver.factor_solve(Ap, ctrl, info, x.data(), n);
    ASSERT_NE(fk.get(), nullptr) << "factor_solve failed at iter=" << iter;

    const double rr = relative_residual(Ap, x.data(), b0.data());
    EXPECT_LT(rr, 1e-9)
        << "iter=" << iter << "  rel.residual=" << rr;
  }
}
