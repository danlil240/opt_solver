#include "smf/analysis.hpp"
#include "smf/assembly_tree.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/solve_forward.hpp"
#include "smf/supernode.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace smf;

// ---------------------------------------------------------------------------
// Helpers: build a minimal AnalysisKeep + FactorKeep for a 1×1 system.
// A = [4.0], Cholesky L = [2.0].
// ---------------------------------------------------------------------------

static void make_1x1(AnalysisKeep &ak, FactorKeep &fk) {
  ak.n = 1;
  ak.perm = {0};
  ak.iperm = {0};

  Supernode sn;
  sn.col_start = 0;
  sn.col_end = 1;
  sn.parent = -1;
  // no children
  ak.supernodes.push_back(std::move(sn));

  FrontalInfo fi;
  fi.row_indices = {0};
  ak.fronts.push_back(std::move(fi));

  // Factor: L11 = [2.0]  (sqrt(4))
  fk.factor_values = {2.0};
  fk.factor_col_ptr = {0, 1};
  fk.perm = {0};
  fk.iperm = {0};
  fk.analysis = &ak;
  fk.is_posdef = true;
}

// ---------------------------------------------------------------------------
// Helpers: build AnalysisKeep + FactorKeep for a 2×2 system.
// A = [[4,2],[2,5]], Cholesky: L11=2, L21=1, L22=2.
// Supernode tree: sn0 (col 0, f=2) → child of sn1 (col 1, f=1).
// ---------------------------------------------------------------------------

static void make_2x2(AnalysisKeep &ak, FactorKeep &fk) {
  ak.n = 2;
  ak.perm = {0, 1};
  ak.iperm = {0, 1};

  // sn0: col 0, parent=1, no children
  Supernode sn0;
  sn0.col_start = 0;
  sn0.col_end = 1;
  sn0.parent = 1;
  ak.supernodes.push_back(std::move(sn0));

  // sn1: col 1, root, child=0
  Supernode sn1;
  sn1.col_start = 1;
  sn1.col_end = 2;
  sn1.parent = -1;
  sn1.children = {0};
  ak.supernodes.push_back(std::move(sn1));

  // fi0: front rows {0,1} → f=2, p=1, q=1
  FrontalInfo fi0;
  fi0.row_indices = {0, 1};
  ak.fronts.push_back(std::move(fi0));

  // fi1: front rows {1} → f=1, p=1, q=0
  FrontalInfo fi1;
  fi1.row_indices = {1};
  ak.fronts.push_back(std::move(fi1));

  // Factor values:
  //   sn0: f×p = 2×1 col-major: [L[0,0], L[1,0]] = [2.0, 1.0]
  //   sn1: f×p = 1×1:            [L[1,1]]          = [2.0]
  fk.factor_values = {2.0, 1.0, 2.0};
  fk.factor_col_ptr = {0, 2, 3};
  fk.perm = {0, 1};
  fk.iperm = {0, 1};
  fk.analysis = &ak;
  fk.is_posdef = true;
}

// ---------------------------------------------------------------------------
// Test 1: trivial 1×1 system
//   b = [8.0] → y = L^{-1} b = [8/2] = [4.0]
// ---------------------------------------------------------------------------
TEST(SolveForward, Trivial1x1) {
  AnalysisKeep ak;
  FactorKeep fk;
  make_1x1(ak, fk);

  double x[1] = {8.0};
  solve_forward(fk, x, 1);

  EXPECT_NEAR(x[0], 4.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Test 2: 2×2 system with off-diagonal update
//   A = [[4,2],[2,5]], L = [[2,0],[1,2]], perm = identity
//   b = [6, 7] → y = L^{-1} b = [3, 2]
//   Verify: L * y = [[2,0],[1,2]] * [3,2] = [6, 3+4] = [6,7] ✓
// ---------------------------------------------------------------------------
TEST(SolveForward, TwoByTwo) {
  AnalysisKeep ak;
  FactorKeep fk;
  make_2x2(ak, fk);

  double x[2] = {6.0, 7.0};
  solve_forward(fk, x, 2);

  EXPECT_NEAR(x[0], 3.0, 1e-12);
  EXPECT_NEAR(x[1], 2.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Test 3: multi-RHS on 1×1 — two columns at once
//   b0=[8], b1=[6] → y0=[4], y1=[3]
// ---------------------------------------------------------------------------
TEST(SolveForward, MultiRHS1x1) {
  AnalysisKeep ak;
  FactorKeep fk;
  make_1x1(ak, fk);

  // Column-major: col0=[8], col1=[6] → layout [8, 6]
  double x[2] = {8.0, 6.0};
  solve_forward(fk, x, 1, 2);

  EXPECT_NEAR(x[0], 4.0, 1e-12);
  EXPECT_NEAR(x[1], 3.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Test 4: zero-size — no crash
// ---------------------------------------------------------------------------
TEST(SolveForward, ZeroSize) {
  FactorKeep fk;
  solve_forward(fk, nullptr, 0, 0);
  // no crash
}
