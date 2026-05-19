#include "smf/analysis.hpp"
#include "smf/assembly_tree.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/solve_backward.hpp"
#include "smf/supernode.hpp"
#include <cassert>
#include <cmath>

// ---------------------------------------------------------------------------
// Helpers to build minimal AnalysisKeep / FactorKeep stubs for unit tests.
// ---------------------------------------------------------------------------

static smf::AnalysisKeep make_analysis_1x1() {
  smf::AnalysisKeep ak;
  ak.n = 1;
  ak.perm = {0};
  ak.iperm = {0};

  smf::Supernode sn;
  sn.col_start = 0;
  sn.col_end = 1;
  sn.parent = -1;
  ak.supernodes.push_back(sn);

  smf::FrontalInfo fi;
  fi.row_indices = {0};
  ak.fronts.push_back(fi);

  return ak;
}

static smf::FactorKeep make_factor_1x1(const smf::AnalysisKeep &ak,
                                       double L11) {
  smf::FactorKeep fk;
  fk.perm = {0};
  fk.iperm = {0};
  fk.analysis = &ak;
  fk.factor_values = {L11};
  fk.factor_col_ptr = {0, 1};
  fk.is_posdef = true;
  return fk;
}

// ---------------------------------------------------------------------------
// Test 1: smoke — solve_backward compiles and runs without crashing on an
//         n=0, nrhs=0 call (null-op guard path).
// ---------------------------------------------------------------------------
static void test_smoke() {
  smf::AnalysisKeep ak;
  ak.n = 0;

  smf::FactorKeep fk;
  fk.analysis = &ak;
  fk.factor_col_ptr = {0}; // one entry for ns=0 supernodes

  // No RHS — should be a no-op.
  smf::solve_backward(fk, nullptr, 0, 0);
}

// ---------------------------------------------------------------------------
// Test 2: 1×1 system, identity permutation.
//   L = [3.0], solve Lᵀ z = [6.0] → z = [2.0].
// ---------------------------------------------------------------------------
static void test_1x1_basic() {
  smf::AnalysisKeep ak = make_analysis_1x1();
  smf::FactorKeep fk = make_factor_1x1(ak, 3.0);

  double x[1] = {6.0};
  smf::solve_backward(fk, x, 1);

  assert(std::abs(x[0] - 2.0) < 1e-12);
}

// ---------------------------------------------------------------------------
// Test 3: 1×1, L = [1.0] — should return x unchanged.
// ---------------------------------------------------------------------------
static void test_1x1_unit_L() {
  smf::AnalysisKeep ak = make_analysis_1x1();
  smf::FactorKeep fk = make_factor_1x1(ak, 1.0);

  double x[1] = {5.0};
  smf::solve_backward(fk, x, 1);

  assert(std::abs(x[0] - 5.0) < 1e-12);
}

// ---------------------------------------------------------------------------
// Test 4: 1×1, multi-RHS (nrhs=2).
//   L = [2.0]; RHS[0]=4.0 → 2.0, RHS[1]=8.0 → 4.0.
// ---------------------------------------------------------------------------
static void test_1x1_multi_rhs() {
  smf::AnalysisKeep ak = make_analysis_1x1();
  smf::FactorKeep fk = make_factor_1x1(ak, 2.0);

  // Column-major: x[0] = rhs0, x[1] = rhs1
  double x[2] = {4.0, 8.0};
  smf::solve_backward(fk, x, 1, 2);

  assert(std::abs(x[0] - 2.0) < 1e-12);
  assert(std::abs(x[1] - 4.0) < 1e-12);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  test_smoke();
  test_1x1_basic();
  test_1x1_unit_L();
  test_1x1_multi_rhs();
  return 0;
}
