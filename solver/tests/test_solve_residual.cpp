#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

namespace {

/// Compute ||x||_2 
double norm2(const std::vector<double> &x) {
  double sum = 0.0;
  for (double v : x) sum += v * v;
  return std::sqrt(sum);
}

/// Compute ||A||_F from lower CSC
double frobenius_norm_lower(const smf::CscLower &A) {
  double sum = 0.0;
  for (double v : A.values) sum += v * v;
  // Account for symmetric off-diagonals (each off-diag appears twice in full matrix)
  for (int j = 0; j < A.n; ++j) {
    for (int p = A.col_ptr[j]; p < A.col_ptr[j+1]; ++p) {
      if (A.row_idx[p] != j) { // off-diagonal
        sum += A.values[p] * A.values[p]; // count it twice
      }
    }
  }
  return std::sqrt(sum);
}

/// Compute residual: r = A*x - b (for symmetric A stored as lower CSC)
std::vector<double> compute_residual(const smf::CscLower &A, 
                                     const std::vector<double> &x,
                                     const std::vector<double> &b) {
  std::vector<double> r(b.size(), 0.0);
  const int n = A.n;
  
  // r = -b initially
  for (int i = 0; i < n; ++i) r[i] = -b[i];
  
  // r += A*x (symmetric: use lower triangle and transpose contribution)
  for (int j = 0; j < n; ++j) {
    for (int p = A.col_ptr[j]; p < A.col_ptr[j+1]; ++p) {
      int i = A.row_idx[p];
      double aij = A.values[p];
      r[i] += aij * x[j];  // Lower triangle
      if (i != j) {
        r[j] += aij * x[i];  // Symmetric contribution
      }
    }
  }
  return r;
}

/// Relative residual: ||A*x - b|| / (||A||*||x|| + ||b||)
double relative_residual(const smf::CscLower &A,
                         const std::vector<double> &x,
                         const std::vector<double> &b) {
  auto r = compute_residual(A, x, b);
  double norm_r = norm2(r);
  double norm_A = frobenius_norm_lower(A);
  double norm_x = norm2(x);
  double norm_b = norm2(b);
  double denom = norm_A * norm_x + norm_b;
  return (denom > 0.0) ? (norm_r / denom) : norm_r;
}

} // anonymous

TEST(SolveResidual, SmallIndefinite_SingleRHS) {
  // Small 4x4 indefinite system
  // A = [ 2   0  -1   0 ]
  //     [ 0   3   0  -1 ]
  //     [-1   0   1   0 ]  
  //     [ 0  -1   0   1 ]
  // Lower triangle in CSC:
  //   col 0: A(0,0)=2, A(2,0)=-1
  //   col 1: A(1,1)=3, A(3,1)=-1  
  //   col 2: A(2,2)=1
  //   col 3: A(3,3)=1
  smf::CscLower A;
  A.n = 4;
  A.col_ptr = {0, 2, 4, 5, 6};  // column pointers
  A.row_idx = {0, 2,  1, 3,  2,  3};  // row indices
  A.values  = {2.0, -1.0,  3.0, -1.0,  1.0,  1.0};
  
  std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> x = b; // copy for in-place solve
  
  smf::Solver solver;
  smf::Control ctrl;
  ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
  ctrl.ordering = smf::OrderingMethod::AMD;
  ctrl.print_level = 1;
  
  smf::Info info;
  auto ak = solver.analyse(A, ctrl, info);
  ASSERT_TRUE(ak) << "analyse failed: " << static_cast<int>(info.status);
  
  std::cout << "Analysis: n=" << ak->n << " supernodes=" << ak->supernodes.size() << "\n";
  
  smf::FactorKeep fkeep;
  smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fkeep);
  std::cout << "Factor status: " << static_cast<int>(fs) 
            << " inertia: +" << info.num_positive 
            << " -" << info.num_negative 
            << " 0=" << info.num_zero << "\n";
  ASSERT_TRUE(fs == smf::FactorStatus::Success || fs == smf::FactorStatus::MaxPivotDelays);
  
  fkeep.analysis = ak.get();  // Temporary link for solve
  int rc = solver.solve(fkeep, ctrl, info, x.data(), 4, 1, smf::SolveJob::Full);
  fkeep.analysis = nullptr;
  
  ASSERT_EQ(rc, 0) << "solve failed";
  
  // Check residual
  double rel_res = relative_residual(A, x, b);
  std::cout << std::scientific << std::setprecision(3);
  std::cout << "Relative residual: " << rel_res << "\n";
  std::cout << "Solution: [" << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3] << "]\n";
  std::cout << "||b|| = " << norm2(b) << ", ||x|| = " << norm2(x) << "\n";
  
  // Expect residual < 1e-10 for well-conditioned small indefinite system
  EXPECT_LT(rel_res, 1e-10) << "Poor solve accuracy for indefinite system";
}

TEST(SolveResidual, SmallIndefinite_MultiRHS) {
  // Same matrix as above
  smf::CscLower A;
  A.n = 4;
  A.col_ptr = {0, 2, 4, 5, 6};
  A.row_idx = {0, 2,  1, 3,  2,  3};
  A.values  = {2.0, -1.0,  3.0, -1.0,  1.0,  1.0};
  
  // Two RHS: b1 = [1,2,3,4], b2 = [4,3,2,1]
  std::vector<double> b = {1.0, 2.0, 3.0, 4.0,  4.0, 3.0, 2.0, 1.0};
  std::vector<double> x = b; // copy for in-place solve
  
  smf::Solver solver;
  smf::Control ctrl;
  ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
  ctrl.ordering = smf::OrderingMethod::AMD;
  
  smf::Info info;
  auto ak = solver.analyse(A, ctrl, info);
  ASSERT_TRUE(ak);
  
  smf::FactorKeep fkeep;
  smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fkeep);
  ASSERT_TRUE(fs == smf::FactorStatus::Success || fs == smf::FactorStatus::MaxPivotDelays);
  
  fkeep.analysis = ak.get();
  int rc = solver.solve(fkeep, ctrl, info, x.data(), 4, 2, smf::SolveJob::Full);
  fkeep.analysis = nullptr;
  
  ASSERT_EQ(rc, 0);
  
  // Check each RHS
  std::vector<double> x1(x.begin(), x.begin() + 4);
  std::vector<double> x2(x.begin() + 4, x.end());
  std::vector<double> b1(b.begin(), b.begin() + 4);
  std::vector<double> b2(b.begin() + 4, b.end());
  
  double rel_res1 = relative_residual(A, x1, b1);
  double rel_res2 = relative_residual(A, x2, b2);
  
  std::cout << std::scientific << std::setprecision(3);
  std::cout << "RHS 1 residual: " << rel_res1 << ", ||x1|| = " << norm2(x1) << "\n";
  std::cout << "RHS 2 residual: " << rel_res2 << ", ||x2|| = " << norm2(x2) << "\n";
  
  EXPECT_LT(rel_res1, 1e-10) << "Poor accuracy for RHS 1";
  EXPECT_LT(rel_res2, 1e-10) << "Poor accuracy for RHS 2";
}
