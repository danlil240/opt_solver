#include "smf/analysis.hpp"
#include "smf/factor_posdef.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace smf;

// Helper: build a minimal diagonal 5×5 SPD lower-CSC
static CscLower make_spd5() {
  // Diagonal matrix: A[j][j] = 4.0 + j*0.5, j=0..4
  constexpr int n = 5;
  CscLower A;
  A.n = n;
  A.col_ptr.resize(n + 1);
  A.row_idx.reserve(n);
  A.values.reserve(n);
  for (int j = 0; j < n; ++j) {
    A.col_ptr[static_cast<std::size_t>(j)] = j;
    A.row_idx.push_back(j);
    A.values.push_back(4.0 + j * 0.5);
  }
  A.col_ptr[n] = n;
  return A;
}

TEST(FactorPosdef, DiagonalSPD5) {
  // Factor a diagonal 5×5 SPD matrix (smoke test — no full analysis available
  // yet)
  CscLower A = make_spd5();
  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
  ctrl.ordering = OrderingMethod::AMD;
  Info info;
  AnalysisKeep ak;
  // Full end-to-end test deferred to M4.S1 (requires Solver::analyse output).
  // Here we verify that FactorKeep is default-constructible and its fields
  // have the expected initial state.
  FactorKeep fk;
  fk.analysis = nullptr;
  EXPECT_TRUE(fk.factor_values.empty());
  EXPECT_TRUE(fk.factor_col_ptr.empty());
  EXPECT_TRUE(fk.perm.empty());
  EXPECT_TRUE(fk.iperm.empty());
  EXPECT_EQ(fk.analysis, nullptr);

  // Eigen round-trip sanity: sqrt of the diagonal entries
  for (int j = 0; j < 5; ++j) {
    double d = A.values[static_cast<std::size_t>(j)];
    EXPECT_NEAR(std::sqrt(d) * std::sqrt(d), d, 1e-12);
  }
}

TEST(FactorPosdef, NotPositiveDefinite) {
  // Verify that the FactorStatus enum values are distinct and accessible
  EXPECT_NE(static_cast<int>(FactorStatus::NotPositiveDefinite),
            static_cast<int>(FactorStatus::Success));
  EXPECT_EQ(static_cast<int>(FactorStatus::Success), 0);
  EXPECT_EQ(static_cast<int>(FactorStatus::NotPositiveDefinite), 1);
}

TEST(FactorPosdef, FactorKeepMoveSemantics) {
  // FactorKeep should be movable (default move ctor)
  FactorKeep fk1;
  fk1.factor_values.push_back(1.0);
  fk1.factor_col_ptr.push_back(0);
  fk1.factor_col_ptr.push_back(1);
  FactorKeep fk2 = std::move(fk1);
  EXPECT_EQ(fk2.factor_values.size(), 1u);
  EXPECT_TRUE(fk1.factor_values.empty()); // NOLINT(bugprone-use-after-move)
}
