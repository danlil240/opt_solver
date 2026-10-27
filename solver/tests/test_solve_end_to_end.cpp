#include "smf/factor_posdef.hpp"
#include "smf/solve_backward.hpp"
#include "smf/solve_diag.hpp"
#include "smf/solve_forward.hpp"
#include "smf/matrix_market.hpp"
#include "smf/solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <span>
#include <string>
#include <vector>
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

// ---------------------------------------------------------------------------
// Task 16: cached inverse diagonal blocks and the blocked multi-RHS solve.
// Fixtures: optim20 and the 7 large_kkt IPOPT replays (first factorization's
// values in the .mtx), each when present on disk.
// ---------------------------------------------------------------------------
namespace {

namespace fs = std::filesystem;

fs::path fixture_root() {
  return fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" / "fixtures";
}

struct SolveCase {
  std::string name;
  CscLower A;
  MatrixType type;
};

std::vector<SolveCase> load_cases(bool optim20, bool large_kkt) {
  std::vector<SolveCase> cases;
  std::error_code ec;
  if (optim20) {
    const fs::path dir = fixture_root() / "optim20";
    std::vector<fs::path> files;
    if (fs::exists(dir, ec))
      for (const auto &e : fs::directory_iterator(dir))
        if (e.path().extension() == ".mtx") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const fs::path &f : files) {
      auto mm = read_matrix_market(f.string());
      if (!mm.ok) continue;
      const std::string name = f.stem().string();
      cases.push_back({name, std::move(mm.matrix),
                       name.rfind("spd_", 0) == 0 ? MatrixType::RealSymmetricPositiveDefinite
                                                   : MatrixType::RealSymmetricIndefinite});
    }
  }
  if (large_kkt) {
    for (const char *name : {"portfolio_20k", "acopf_grid_80", "obstacle_surface_200",
                             "pde_control_120", "acopf_grid_160", "robot_arm_8k",
                             "pde_control_250"}) {
      const fs::path f = fixture_root() / "ipopt_kkt" / name / (std::string(name) + ".mtx");
      if (!fs::exists(f, ec)) continue;
      auto mm = read_matrix_market(f.string());
      if (!mm.ok) continue;
      cases.push_back({name, std::move(mm.matrix), MatrixType::RealSymmetricIndefinite});
    }
  }
  return cases;
}

double max_abs(const std::vector<double> &v) {
  double m = 0.0;
  for (double x : v) m = std::max(m, std::abs(x));
  return m;
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

// Two solutions of A x = b that differ only in rounding: their difference
// relative to x, and the same difference seen through A (normwise, the
// backward-error scale: ||A (xa - xb)|| / (||A|| ||xb||)). On an
// ill-conditioned matrix any change in the order of the floating-point
// operations moves x by up to cond(A) * eps, so the solution-space check
// alone cannot hold there; the residual-space one is what a backward-stable
// solve guarantees.
// within(tol, floor): the solution check, or the residual one against tol
// or, when larger, floor (a multiple of the reference solve's own backward
// error: robot_arm_8k's unrefined solve has 8e-7, and its parallel solve
// alone moves x by that much from run to run).
struct Agreement {
  double solution = 0.0;
  double residual = 0.0;
  bool within(double tol, double floor = 0.0) const {
    return solution <= tol || residual <= std::max(tol, floor);
  }
};

Agreement agreement(const CscLower &A, const double *xa, const double *xb) {
  const auto n = static_cast<std::size_t>(A.n);
  std::vector<double> d(n), Ad(n, 0.0), rowsum(n, 0.0);
  double dmax = 0.0, xmax = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    d[i] = xa[i] - xb[i];
    dmax = std::max(dmax, std::abs(d[i]));
    xmax = std::max(xmax, std::abs(xb[i]));
  }
  for (std::size_t j = 0; j < n; ++j)
    for (auto k = static_cast<std::size_t>(A.col_ptr[j]); k < static_cast<std::size_t>(A.col_ptr[j + 1]); ++k) {
      const auto i = static_cast<std::size_t>(A.row_idx[k]);
      const double a = A.values[k];
      Ad[i] += a * d[j];
      rowsum[i] += std::abs(a);
      if (i != j) {
        Ad[j] += a * d[i];
        rowsum[j] += std::abs(a);
      }
    }
  Agreement g;
  g.solution = xmax > 0.0 ? dmax / xmax : dmax;
  g.residual = max_abs(Ad) / (max_abs(rowsum) * std::max(xmax, 1e-300));
  return g;
}

// Normwise backward error of x for A x = b(column 0 of rhs_entry):
// ||b - A x|| / (||A|| ||x|| + ||b||), infinity norms.
double backward_error(const CscLower &A, const double *x, int col);

// fk without its cached inverses (the TRSV path).
FactorKeep without_inverses(const FactorKeep &fk) {
  FactorKeep f = fk;
  f.inv_diag_offset.clear();
  return f;
}

double rhs_entry(std::size_t i, int col) {
  return std::sin(0.013 * static_cast<double>(i) + 0.7 * col) + 0.5;
}

double backward_error(const CscLower &A, const double *x, int col) {
  const auto n = static_cast<std::size_t>(A.n);
  std::vector<double> r(n), rowsum(n, 0.0);
  double bmax = 0.0, xmax = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    r[i] = rhs_entry(i, col);
    bmax = std::max(bmax, std::abs(r[i]));
    xmax = std::max(xmax, std::abs(x[i]));
  }
  for (std::size_t j = 0; j < n; ++j)
    for (auto k = static_cast<std::size_t>(A.col_ptr[j]); k < static_cast<std::size_t>(A.col_ptr[j + 1]); ++k) {
      const auto i = static_cast<std::size_t>(A.row_idx[k]);
      const double a = A.values[k];
      r[i] -= a * x[j];
      rowsum[i] += std::abs(a);
      if (i != j) {
        r[j] -= a * x[i];
        rowsum[j] += std::abs(a);
      }
    }
  return max_abs(r) / (max_abs(rowsum) * xmax + bmax);
}

} // namespace

// L11^-1 cached at factor time (GEMV in the solve) against TRSV on L11:
// solutions within 1e-13 relative (in the solution, or through A on an
// ill-conditioned matrix: see Agreement). 1 thread: a factor without the
// cache against one with it (the serial factor is deterministic). 4 threads
// (a parallel solve on the large models): the same factor solved with its
// cached inverses and with them dropped, since threaded factorizations
// differ in rounding from run to run.
TEST(SolveEndToEnd, InverseDiagMatchesTrsv) {
  const std::vector<SolveCase> cases = load_cases(true, true);
  ASSERT_FALSE(cases.empty()) << "no optim20 / large_kkt fixtures";
  int cached_total = 0;
  for (const SolveCase &c : cases) {
    SCOPED_TRACE(c.name);
    const int n = static_cast<int>(c.A.n);
    const auto solve_one = [&](const Solver &s, const FactorKeep &fk, const Control &ctrl) {
      std::vector<double> x(static_cast<std::size_t>(n));
      for (std::size_t i = 0; i < x.size(); ++i) x[i] = rhs_entry(i, 0);
      SolveWorkspace sws;
      Info info;
      EXPECT_EQ(s.solve(fk, ctrl, info, x.data(), n, 1, sws), 0);
      return x;
    };
    for (const int threads : {1, 4}) {
      if (threads > 1 && c.A.n < 20000) continue;
      Control ctrl;
      ctrl.matrix_type = c.type;
      ctrl.num_threads = threads;
      Solver s;
      Info info;
      auto ak = s.analyse(c.A, ctrl, info);
      ASSERT_NE(ak.get(), nullptr);
      NumericWorkspace nws;
      FactorKeep fk;
      ctrl.cache_inverse_diag = true;
      ASSERT_EQ(s.factor(*ak, c.A, ctrl, info, fk, nws), FactorStatus::Success);
      cached_total += info.inverse_diag_fronts;
      const std::vector<double> xi = solve_one(s, fk, ctrl);
      std::vector<double> xt;
      if (threads == 1) {
        ctrl.cache_inverse_diag = false;
        FactorKeep fk0;
        ASSERT_EQ(s.factor(*ak, c.A, ctrl, info, fk0, nws), FactorStatus::Success);
        EXPECT_EQ(info.inverse_diag_fronts, 0);
        EXPECT_TRUE(fk0.inv_diag_offset.empty());
        xt = solve_one(s, fk0, ctrl);
      } else {
        xt = solve_one(s, without_inverses(fk), ctrl);
      }
      const Agreement g = agreement(c.A, xi.data(), xt.data());
      std::printf("[ inverse  ] %-22s %d thr: berr inv %.2e trsv %.2e;"
                  " solution %.2e residual %.2e\n", c.name.c_str(),
                  threads, backward_error(c.A, xi.data(), 0), backward_error(c.A, xt.data(), 0),
                  g.solution, g.residual);
      EXPECT_TRUE(g.within(1e-13, 10.0 * backward_error(c.A, xt.data(), 0))) << threads << " threads: solution " << g.solution
                                   << " residual " << g.residual;
    }
  }
  EXPECT_GT(cached_total, 0) << "no front had its L11 inverse cached";
}

// The blocked multi-RHS solve (BLAS-3 per front, leading dimension ldb > n)
// against nrhs separate single-RHS solves: within 1e-14 relative (in the
// solution, or through A on an ill-conditioned matrix: see Agreement).
TEST(SolveEndToEnd, MultiRhsMatchesLoop) {
  std::vector<SolveCase> cases = load_cases(true, false);
  {
    std::vector<SolveCase> big = load_cases(false, true);
    for (auto &c : big)
      if (c.name == "portfolio_20k" || c.name == "pde_control_120")
        cases.push_back(std::move(c));
  }
  ASSERT_FALSE(cases.empty()) << "no optim20 / large_kkt fixtures";
  for (const SolveCase &c : cases) {
    SCOPED_TRACE(c.name);
    for (const bool cache : {false, true}) {
      Control ctrl;
      ctrl.matrix_type = c.type;
      ctrl.cache_inverse_diag = cache;
      Solver s;
      Info info;
      auto ak = s.analyse(c.A, ctrl, info);
      ASSERT_NE(ak.get(), nullptr);
      FactorKeep fk;
      NumericWorkspace nws;
      SolveWorkspace sws;
      ASSERT_EQ(s.factor(*ak, c.A, ctrl, info, fk, nws), FactorStatus::Success);
      const int n = static_cast<int>(c.A.n);
      for (const int nrhs : {1, 2, 7, 32}) {
        const int ldb = n + 3;
        std::vector<double> B(static_cast<std::size_t>(ldb) * nrhs, -7.0);
        std::vector<double> loop(static_cast<std::size_t>(n) * nrhs);
        for (int r = 0; r < nrhs; ++r)
          for (int i = 0; i < n; ++i) {
            B[static_cast<std::size_t>(r) * ldb + i] = rhs_entry(static_cast<std::size_t>(i), r);
            loop[static_cast<std::size_t>(r) * n + i] = rhs_entry(static_cast<std::size_t>(i), r);
          }
        for (int r = 0; r < nrhs; ++r)
          ASSERT_EQ(s.solve(fk, ctrl, info, loop.data() + static_cast<std::size_t>(r) * n, n, 1, sws), 0);
        ASSERT_EQ(s.solve(fk, ctrl, info, std::span<double>(B), nrhs, ldb, sws), 0);
        Agreement worst;
        for (int r = 0; r < nrhs; ++r) {
          const Agreement g = agreement(c.A, B.data() + static_cast<std::size_t>(r) * ldb,
                                        loop.data() + static_cast<std::size_t>(r) * n);
          EXPECT_TRUE(g.within(1e-14, 10.0 * backward_error(c.A, loop.data() + static_cast<std::size_t>(r) * n, r))) << "nrhs " << nrhs << " cache " << cache << " column " << r
                                       << ": solution " << g.solution << " residual " << g.residual;
          worst.solution = std::max(worst.solution, g.solution);
          worst.residual = std::max(worst.residual, g.residual);
        }
        if (nrhs == 32)
          std::printf("[ multirhs ] %-22s cache %d: solution %.2e residual %.2e\n", c.name.c_str(),
                      cache ? 1 : 0, worst.solution, worst.residual);
        // The padding rows between columns are untouched.
        for (int r = 0; r < nrhs; ++r)
          for (int i = n; i < ldb; ++i)
            EXPECT_EQ(B[static_cast<std::size_t>(r) * ldb + i], -7.0);
      }
    }
  }
}
