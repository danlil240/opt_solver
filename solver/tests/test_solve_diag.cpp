#include "smf/analysis.hpp"
#include "smf/solve_diag.hpp"
#include <gtest/gtest.h>

using namespace smf;

// ---------------------------------------------------------------------------
// TEST: PermRoundtrip
// apply_perm then apply_inv_perm must be the identity.
// ---------------------------------------------------------------------------
TEST(SolveDiag, PermRoundtrip) {
  constexpr int n = 4;
  FactorKeep fk;
  fk.perm = {2, 0, 3, 1};  // perm[new] = old
  fk.iperm = {1, 3, 0, 2}; // iperm[old] = new

  double x[n] = {10.0, 20.0, 30.0, 40.0};
  double y[n] = {};
  double z[n] = {};

  apply_perm(fk, x, y, n);
  apply_inv_perm(fk, y, z, n);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(z[i], x[i], 1e-15) << "mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// TEST: Identity1x1
// With is_posdef=true, solve_diag must be a no-op.
// ---------------------------------------------------------------------------
TEST(SolveDiag, Identity1x1) {
  constexpr int n = 3;
  FactorKeep fk;
  fk.is_posdef = true;

  double x[n] = {1.0, 2.0, 3.0};
  const double x_orig[n] = {1.0, 2.0, 3.0};

  solve_diag(fk, x, n);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], x_orig[i], 1e-15) << "SPD no-op failed at index " << i;
  }
}

// ---------------------------------------------------------------------------
// TEST: BlockD2x2
// D = [[2, 1], [1, 2]] (d00=2, d10=1, d11=2).
// D^{-1} = [[2, -1], [-1, 2]] / 3.
// x = [3, 5] → D^{-1} x = [(2*3 - 1*5)/3, (-1*3 + 2*5)/3] = [1/3, 7/3].
// ---------------------------------------------------------------------------
TEST(SolveDiag, BlockD2x2) {
  // Build a minimal AnalysisKeep: 1 supernode, width=2, front_size=2.
  AnalysisKeep ak;
  ak.n = 2;

  Supernode sn;
  sn.col_start = 0;
  sn.col_end = 2;
  sn.parent = -1;
  ak.supernodes.push_back(std::move(sn));

  FrontalInfo fi;
  fi.row_indices = {0, 1};
  ak.fronts.push_back(std::move(fi));

  // factor_values layout: f=2 (front_size), p=2 (width), col-major.
  //   (row, col) → col*f + row
  //   (0,0)=0: d00 = 2.0
  //   (1,0)=1: d10 = 1.0   ← lower-triangle off-diagonal of 2×2 block
  //   (0,1)=2: unused (upper triangle) = 0.0
  //   (1,1)=3: d11 = 2.0
  FactorKeep fk;
  fk.is_posdef = false;
  fk.analysis = &ak;
  fk.factor_col_ptr = {0}; // supernode 0 starts at index 0
  fk.factor_values = {2.0, 1.0, 0.0, 2.0};
  fk.perm = {0, 1};
  fk.iperm = {0, 1};
  fk.pivot_types = {{2, -1}}; // col 0: first of 2×2; col 1: second of 2×2

  double x[2] = {3.0, 5.0};
  solve_diag(fk, x, 2);

  EXPECT_NEAR(x[0], 1.0 / 3.0, 1e-14) << "x[0] wrong after 2×2 D solve";
  EXPECT_NEAR(x[1], 7.0 / 3.0, 1e-14) << "x[1] wrong after 2×2 D solve";
}
