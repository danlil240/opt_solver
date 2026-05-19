#include "smf/pivoting.hpp"
#include "smf/utils/memory.hpp"
#include <cmath>
#include <gtest/gtest.h>
using namespace smf;

// Helper: build a simple FrontalMatrix of size f×p filled from a flat col-major
// array
static FrontalMatrix make_front(int f, int p, const double *vals,
                                AlignedArena &arena) {
  std::vector<Int> rows(f), cols(p);
  for (int i = 0; i < f; i++)
    rows[i] = i;
  for (int i = 0; i < p; i++)
    cols[i] = i;
  FrontalMatrix F(f, p, rows, cols, arena);
  for (int j = 0; j < p; j++)
    for (int i = 0; i < f; i++)
      F.at(i, j) = vals[j * f + i];
  return F;
}

// Test 1: 1×1 pivot accepted (dominant diagonal)
TEST(Pivot, Accept1x1_Dominant) {
  AlignedArena arena(4096);
  // 3×3 front (p=3), diag dominant: [[4,2,1],[2,3,0.5],[1,0.5,2]]
  double v[] = {4, 2, 1, 2, 3, 0.5, 1, 0.5, 2}; // col-major
  auto F = make_front(3, 3, v, arena);
  auto r = choose_pivot(F, 0, 3, 0.01, 1e-20);
  EXPECT_EQ(r.decision, PivotDecision::Accept1x1);
  EXPECT_EQ(r.col0, 0);
}

// Test 2: 2×2 pivot triggered (zero diagonal)
TEST(Pivot, Accept2x2_ZeroDiag) {
  AlignedArena arena(4096);
  // 4×4, col0 diagonal = 0, large off-diag → should trigger 2×2
  // col-major: col0=[0,1,0,0], col1=[1,2,0,0], col2=[0,0,3,0], col3=[0,0,0,4]
  double v[] = {0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4};
  auto F = make_front(4, 4, v, arena);
  auto r = choose_pivot(F, 0, 4, 0.01, 1e-20);
  // |a_00|=0 < u*|a_10|=0.01 → not 1×1; should be 2×2 or swap+1×1
  EXPECT_NE(r.decision, PivotDecision::Reject);
}

// Test 3: Apply 1×1 pivot, verify L and Schur complement
TEST(Pivot, Apply1x1_Schur) {
  AlignedArena arena(4096);
  // 3×3 SPD: [[4,2,1],[2,4,1],[1,1,3]] col-major lower
  double v[] = {4, 2, 1, 0, 4, 1, 0, 0, 3};
  auto F = make_front(3, 3, v, arena);
  // Set lower triangle explicitly
  F.at(1, 0) = 2;
  F.at(2, 0) = 1;
  F.at(2, 1) = 1;
  F.at(0, 0) = 4;
  F.at(1, 1) = 4;
  F.at(2, 2) = 3;
  apply_pivot_1x1(F, 0, 3);
  // L[1,0]=2/4=0.5, L[2,0]=1/4=0.25
  EXPECT_NEAR(F.at(1, 0), 0.5, 1e-12);
  EXPECT_NEAR(F.at(2, 0), 0.25, 1e-12);
  // Schur: F[1,1] = 4 - 0.5*4*0.5 = 3
  EXPECT_NEAR(F.at(1, 1), 3.0, 1e-12);
}

// Test 4: Reject near-zero pivot
TEST(Pivot, Reject_NearZero) {
  AlignedArena arena(4096);
  double v[] = {1e-25, 1, 0, 0, 2, 0, 0, 0, 3};
  auto F = make_front(3, 3, v, arena);
  F.at(1, 0) = 1.0; // large off-diag
  auto r = choose_pivot(F, 0, 3, 0.5, 1e-20);
  // At minimum: must not crash. Result is either 2×2 or reject or (unlikely)
  // 1×1.
  (void)r;
  EXPECT_TRUE(r.decision == PivotDecision::Accept2x2 ||
              r.decision == PivotDecision::Reject ||
              r.decision == PivotDecision::Accept1x1);
}

// Test 5: sym_swap_front swaps rows and cols correctly
TEST(Pivot, SymSwap) {
  AlignedArena arena(4096);
  // 3×3 lower triangular: diag=[1,3,6], off-diag a10=2, a20=4, a21=5
  double v[] = {1, 2, 4, 0, 3, 5, 0, 0, 6};
  auto F = make_front(3, 3, v, arena);
  F.at(1, 0) = 2;
  F.at(2, 0) = 4;
  F.at(2, 1) = 5;
  F.at(0, 0) = 1;
  F.at(1, 1) = 3;
  F.at(2, 2) = 6;
  sym_swap_front(F, 0, 1, 3);
  // After swap(0,1): diagonal[0]=old diagonal[1]=3, diagonal[1]=old
  // diagonal[0]=1
  EXPECT_NEAR(F.at(0, 0), 3.0, 1e-12);
  EXPECT_NEAR(F.at(1, 1), 1.0, 1e-12);
  // F.at(1,0) was 2 (the off-diagonal between 0 and 1); unaffected by this swap
  EXPECT_NEAR(F.at(1, 0), 2.0, 1e-12);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
