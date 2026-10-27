#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/factor_indef.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/refinement.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace smf;

TEST(FactorIndef, HeaderCompiles) {
  FactorKeep fk;
  EXPECT_TRUE(fk.factor_values.empty());
  EXPECT_NE(FactorStatus::Singular, FactorStatus::Success);
}

TEST(FactorIndef, InertiaEnums) {
  InertiaCounts ic;
  add_inertia_1x1(1.0, 1e-15, ic);
  EXPECT_EQ(ic.positive, 1);
  add_inertia_1x1(-1.0, 1e-15, ic);
  EXPECT_EQ(ic.negative, 1);
}

namespace {

namespace fs = std::filesystem;

// Random symmetric indefinite matrix: ~deg off-diagonals per column with
// uniform values, random-sign diagonals, a fraction of them zero so that
// threshold pivoting must use 2x2 pivots or delay.
CscLower random_indefinite(Int n, int deg, double zero_diag_frac, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::uniform_int_distribution<Int> pick(0, n - 1);
  std::vector<std::set<Int>> cols(static_cast<std::size_t>(n));
  for (Int j = 0; j < n; ++j) {
    cols[static_cast<std::size_t>(j)].insert(j);
    for (int t = 0; t < deg; ++t) {
      const Int i = pick(rng);
      cols[static_cast<std::size_t>(std::min(i, j))].insert(std::max(i, j));
    }
  }
  CscLower A;
  A.n = n;
  A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  for (Int j = 0; j < n; ++j) {
    for (const Int i : cols[static_cast<std::size_t>(j)]) {
      A.row_idx.push_back(i);
      double v = uni(rng);
      if (i == j) v = (u01(rng) < zero_diag_frac) ? 0.0 : 2.0 * uni(rng);
      A.values.push_back(v);
    }
    A.col_ptr[static_cast<std::size_t>(j) + 1] = static_cast<Int>(A.row_idx.size());
  }
  return A;
}

struct EngineRun {
  bool ok = false;
  int pos = 0, neg = 0, zero = 0;
  double eta = 1.0, omega = 1.0;
  int aptp_failed = 0;
};

EngineRun run_engine(const CscLower &A, Control ctrl) {
  EngineRun r;
  Solver solver;
  Info info;
  auto ak = solver.analyse(A, ctrl, info);
  if (!ak) return r;
  FactorKeep fk;
  NumericWorkspace ws;
  const FactorStatus st = solver.factor(*ak, A, ctrl, info, fk, ws);
  if (st != FactorStatus::Success) return r;
  r.pos = info.num_positive;
  r.neg = info.num_negative;
  r.zero = info.num_zero;
  r.aptp_failed = info.aptp_failed_columns;
  std::vector<double> b(static_cast<std::size_t>(A.n));
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.37 * static_cast<double>(i)) + 0.25;
  std::vector<double> x = b;
  SolveWorkspace sws;
  if (solver.solve(fk, ctrl, info, x.data(), static_cast<int>(A.n), 1, sws) != 0) return r;
  (void)refine(A, fk, ctrl, b, x, 1, sws);
  const auto v = smf::bench::validate(A, b, x, 1);
  r.eta = v.eta_inf;
  r.omega = v.omega;
  r.ok = smf::bench::is_qualified_validation(v);
  return r;
}

void expect_engines_agree(const CscLower &A, const std::string &name) {
  Control ctrl;
  ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
  ctrl.front_kernel = FrontKernel::Tpp;
  const EngineRun tpp = run_engine(A, ctrl);
  ctrl.front_kernel = FrontKernel::Aptp;
  const EngineRun aptp = run_engine(A, ctrl);
  EXPECT_TRUE(tpp.ok) << name << " tpp eta=" << tpp.eta << " omega=" << tpp.omega;
  EXPECT_TRUE(aptp.ok) << name << " aptp eta=" << aptp.eta << " omega=" << aptp.omega;
  EXPECT_EQ(tpp.pos, aptp.pos) << name;
  EXPECT_EQ(tpp.neg, aptp.neg) << name;
  EXPECT_EQ(tpp.zero, aptp.zero) << name;
}

} // namespace

// The APTP engine (FrontKernel::Aptp, every plain front) and the TPP engine
// both pass the accuracy gate and agree on inertia, on every indefinite
// optim20 case and on random indefinite matrices with zero diagonals.
TEST(FactorIndef, AptpEngineMatchesTppAccuracy) {
  const fs::path dir = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" / "fixtures" / "optim20";
  const char *cases[] = {"kkt_lqr_h20",      "kkt_lqr_h200",   "kkt_nmpc_h30",   "kkt_contact_h40",
                         "kkt_eq_qp",        "kkt_ineq_slack", "kkt_auglag",     "kkt_ipm_late",
                         "kkt_network_flow", "kkt_pde_opt",    "indef_helmholtz_50", "indef_rand_1500"};
  int loaded = 0;
  for (const char *c : cases) {
    const fs::path mtx = dir / (std::string(c) + ".mtx");
    std::error_code ec;
    if (!fs::exists(mtx, ec)) continue;
    auto mm = read_matrix_market(mtx.string());
    ASSERT_TRUE(mm.ok) << c << ": " << mm.error_message;
    ++loaded;
    expect_engines_agree(mm.matrix, c);
  }
  if (loaded == 0) GTEST_SKIP() << "optim20 fixtures not present";
  for (unsigned s = 0; s < 5; ++s) {
    const CscLower A = random_indefinite(400 + 150 * static_cast<Int>(s), 6, 0.3, 1000 + s);
    expect_engines_agree(A, "random seed " + std::to_string(1000 + s));
  }
}
