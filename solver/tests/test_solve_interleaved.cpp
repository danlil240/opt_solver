// The interleaved full solve (solve_interleaved.cpp) and the precomputed
// D^-1 (FactorKeep::dinv) against the per-column partial jobs (Forward then
// DiagBack), which still run the original front kernels.
#include "smf/factor_posdef.hpp"
#include "smf/solve_diag.hpp"
#include "smf/solve_interleaved.hpp"
#include "smf/solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <vector>
#include <gtest/gtest.h>

using namespace smf;

namespace {

// Lower CSC from (row, col, value) triplets with row >= col; duplicates add.
CscLower from_triplets(Int n, std::vector<std::tuple<Int, Int, double>> t) {
  std::sort(t.begin(), t.end(), [](const auto &a, const auto &b) {
    return std::get<1>(a) != std::get<1>(b) ? std::get<1>(a) < std::get<1>(b)
                                            : std::get<0>(a) < std::get<0>(b);
  });
  CscLower A;
  A.n = n;
  A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  for (std::size_t k = 0; k < t.size(); ++k) {
    const auto [r, c, v] = t[k];
    if (!A.row_idx.empty() && k > 0 && std::get<0>(t[k - 1]) == r && std::get<1>(t[k - 1]) == c) {
      A.values.back() += v;
      continue;
    }
    A.row_idx.push_back(r);
    A.values.push_back(v);
    ++A.col_ptr[static_cast<std::size_t>(c) + 1];
  }
  for (Int j = 0; j < n; ++j)
    A.col_ptr[static_cast<std::size_t>(j) + 1] += A.col_ptr[static_cast<std::size_t>(j)];
  return A;
}

// A trajectory-like KKT: nx primal variables with a banded Hessian (stage
// coupling), nc constraints each touching a few nearby variables and a zero
// diagonal (2x2 pivots), plus ng global variables coupled to everything
// (ng >= 16 gives a root front large enough for the BLAS-3 path).
CscLower make_kkt(Int nx, Int nc, Int ng, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const Int n = nx + nc + ng;
  std::vector<std::tuple<Int, Int, double>> t;
  for (Int i = 0; i < nx; ++i) {
    t.emplace_back(i, i, 2.0 + u(rng));
    for (Int d = 1; d <= 2 && i + d < nx; ++d)
      t.emplace_back(i + d, i, 0.3 * u(rng));
  }
  for (Int k = 0; k < nc; ++k) {
    const Int row = nx + k;
    const Int base = (k * nx) / std::max<Int>(nc, 1);
    for (Int d = 0; d < 3 && base + d < nx; ++d)
      t.emplace_back(row, base + d, u(rng));
    t.emplace_back(row, row, 0.0);
  }
  for (Int g = 0; g < ng; ++g) {
    const Int row = nx + nc + g;
    t.emplace_back(row, row, 1.0 + u(rng));
    for (Int j = 0; j < nx; j += 7)
      t.emplace_back(row, j, 0.1 * u(rng));
    for (Int h = 0; h < g; ++h)
      t.emplace_back(row, nx + nc + h, 0.05 * u(rng));
  }
  return from_triplets(n, std::move(t));
}

CscLower make_spd(Int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<std::tuple<Int, Int, double>> t;
  for (Int i = 0; i < n; ++i) {
    t.emplace_back(i, i, 8.0);
    for (Int d : {1, 3, 11})
      if (i + d < n)
        t.emplace_back(i + d, i, u(rng));
  }
  return from_triplets(n, std::move(t));
}

// r = A x (A symmetric, lower CSC).
std::vector<double> matvec(const CscLower &A, const double *x) {
  std::vector<double> r(static_cast<std::size_t>(A.n), 0.0);
  for (Int j = 0; j < A.n; ++j)
    for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      const Int i = A.row_idx[static_cast<std::size_t>(p)];
      const double v = A.values[static_cast<std::size_t>(p)];
      r[static_cast<std::size_t>(i)] += v * x[j];
      if (i != j)
        r[static_cast<std::size_t>(j)] += v * x[i];
    }
  return r;
}

double rel_residual(const CscLower &A, const double *x, const double *b) {
  const std::vector<double> ax = matvec(A, x);
  double rn = 0.0, bn = 0.0;
  for (Int i = 0; i < A.n; ++i) {
    rn = std::max(rn, std::abs(ax[static_cast<std::size_t>(i)] - b[i]));
    bn = std::max(bn, std::abs(b[i]));
  }
  return rn / std::max(bn, 1e-300);
}

struct Factored {
  CscLower A;
  Control ctrl;
  std::unique_ptr<AnalysisKeep> ak;
  FactorKeep fk;
};

void factor(Factored &F) {
  Solver s;
  Info info;
  F.ak = s.analyse(F.A, F.ctrl, info);
  ASSERT_NE(F.ak.get(), nullptr);
  NumericWorkspace ws;
  const FactorStatus st = s.factor(*F.ak, std::span<const double>(F.A.values), F.ctrl, info, F.fk, ws);
  ASSERT_EQ(st, FactorStatus::Success);
}

// Full solves of nrhs columns (stride ldb, padding checked) against the
// per-column Forward + DiagBack jobs, and the residual.
void check_solves(const Factored &F, int nrhs, int ldb, unsigned seed) {
  const int n = static_cast<int>(F.A.n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  constexpr double kPad = 12345.0;
  std::vector<double> B(static_cast<std::size_t>(ldb) * nrhs, kPad);
  for (int r = 0; r < nrhs; ++r)
    for (int i = 0; i < n; ++i)
      B[static_cast<std::size_t>(r) * ldb + i] = (r % 3 == 2 && i % 2) ? 0.0 : u(rng);
  std::vector<double> X = B;
  Solver s;
  Info info;
  SolveWorkspace ws;
  ASSERT_EQ(s.solve(F.fk, F.ctrl, info, std::span<double>(X), nrhs, ldb, ws, SolveJob::Full), 0);
  for (int r = 0; r < nrhs; ++r) {
    const double *b = B.data() + static_cast<std::size_t>(r) * ldb;
    const double *x = X.data() + static_cast<std::size_t>(r) * ldb;
    for (int i = n; i < ldb; ++i)
      ASSERT_EQ(x[i], kPad) << "padding written, column " << r;
    EXPECT_LT(rel_residual(F.A, x, b), 1e-10) << "nrhs " << nrhs << " column " << r;
    // The original kernels, one column at a time.
    std::vector<double> y(b, b + n);
    ASSERT_EQ(s.solve(F.fk, F.ctrl, info, y.data(), n, 1, ws, SolveJob::Forward), 0);
    ASSERT_EQ(s.solve(F.fk, F.ctrl, info, y.data(), n, 1, ws, SolveJob::DiagBack), 0);
    double diff = 0.0, ynorm = 0.0;
    for (int i = 0; i < n; ++i) {
      diff = std::max(diff, std::abs(x[i] - y[static_cast<std::size_t>(i)]));
      ynorm = std::max(ynorm, std::abs(y[static_cast<std::size_t>(i)]));
    }
    EXPECT_LE(diff, 1e-11 * std::max(ynorm, 1.0)) << "nrhs " << nrhs << " column " << r;
  }
}

} // namespace

TEST(SolveInterleaved, SmallFrontKktTakesThePathAndMatchesTheFrontKernels) {
  Factored F;
  F.A = make_kkt(300, 150, 4, 7u);
  F.ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
  factor(F);
  ASSERT_TRUE(F.fk.inline_d);
  EXPECT_TRUE(F.fk.solve_small_fronts);
  EXPECT_TRUE(interleaved_solve_supported(F.fk));
  ASSERT_EQ(F.fk.dinv_offset.size(), F.fk.solve_steps.size());
  // The KKT's zero diagonals need 2x2 pivots: the 2x2 lanes are exercised.
  EXPECT_NE(std::count(F.fk.dinv_kind.begin(), F.fk.dinv_kind.end(), std::int8_t{2}), 0);
  const int n = static_cast<int>(F.A.n);
  for (int nrhs : {1, 2, 3, 4, 5, 6, 8, 9, 13})
    check_solves(F, nrhs, n, 100u + static_cast<unsigned>(nrhs));
  check_solves(F, 1, n + 7, 1u);
  check_solves(F, 6, n + 7, 2u);
}

TEST(SolveInterleaved, LargeFrontsGatherIntoBlas3) {
  Factored F;
  F.A = make_kkt(300, 150, 40, 11u);
  F.ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
  factor(F);
  EXPECT_FALSE(F.fk.solve_small_fronts);
  bool large = false;
  for (const SolveStep &st : F.fk.solve_steps)
    large = large || st.p >= kInvDiagMinPivots;
  ASSERT_TRUE(large);
  const int n = static_cast<int>(F.A.n);
  for (int nrhs : {1, 2, 6, 8, 12})
    check_solves(F, nrhs, n + 3, 200u + static_cast<unsigned>(nrhs));
}

TEST(SolveInterleaved, CachedInverseDiagFronts) {
  Factored F;
  F.A = make_kkt(300, 150, 40, 13u);
  F.ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
  F.ctrl.cache_inverse_diag = true;
  factor(F);
  const int n = static_cast<int>(F.A.n);
  for (int nrhs : {2, 6})
    check_solves(F, nrhs, n, 300u + static_cast<unsigned>(nrhs));
}

TEST(SolveInterleaved, PositiveDefinite) {
  Factored F;
  F.A = make_spd(400, 5u);
  F.ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
  factor(F);
  EXPECT_TRUE(F.fk.dinv.empty());
  const int n = static_cast<int>(F.A.n);
  for (int nrhs : {1, 3, 6, 10})
    check_solves(F, nrhs, n, 400u + static_cast<unsigned>(nrhs));
}

TEST(SolveInterleaved, RefactorRebuildsDinv) {
  Factored F;
  F.A = make_kkt(200, 100, 4, 17u);
  F.ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
  factor(F);
  // New values, same pattern: D^-1 must follow the new factor.
  for (double &v : F.A.values)
    v *= 1.5;
  Solver s;
  Info info;
  NumericWorkspace ws;
  ASSERT_EQ(s.factor(*F.ak, std::span<const double>(F.A.values), F.ctrl, info, F.fk, ws),
            FactorStatus::Success);
  check_solves(F, 1, static_cast<int>(F.A.n), 3u);
  check_solves(F, 6, static_cast<int>(F.A.n), 4u);
}
