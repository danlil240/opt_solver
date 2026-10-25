#include "smf/solve_backward.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/solve_diag.hpp"
#include "smf/solve_forward.hpp"
#include "smf/solver.hpp"
#include <chrono>
#include <vector>

namespace smf {

FactorStatus Solver::factor(const AnalysisKeep &ak, const Control &ctrl,
                            Info &info, FactorKeep &fkeep) const {
  if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
    return factor_posdef(ak, ctrl, info, fkeep);
  }
  return factor_indef(ak, ctrl, info, fkeep);
}

int Solver::solve(const FactorKeep &fkeep, const Control &ctrl, Info &info,
                  double *b, int n, int nrhs, SolveJob job) const {
  auto t0 = std::chrono::steady_clock::now();

  const int expected_n =
      (fkeep.n > 0) ? static_cast<int>(fkeep.n)
                    : (fkeep.use_banded_spd ? fkeep.band_n
                                            : static_cast<int>(fkeep.perm.size()));

  if (n <= 0 || nrhs <= 0 || b == nullptr ||
      (expected_n > 0 && n != expected_n) ||
      static_cast<int>(fkeep.perm.size()) != n ||
      static_cast<int>(fkeep.iperm.size()) != n) {
    info.status = ErrorCode::IllegalValue;
    return static_cast<int>(ErrorCode::IllegalValue);
  }

  switch (job) {
  case SolveJob::Full:
  case SolveJob::Forward:
  case SolveJob::DiagOnly:
  case SolveJob::Backward:
  case SolveJob::DiagBack:
    break;
  default:
    info.status = ErrorCode::IllegalValue;
    return static_cast<int>(ErrorCode::IllegalValue);
  }

  if (fkeep.status != FactorStatus::Success && !ctrl.continue_on_singular) {
    info.status = (fkeep.status == FactorStatus::NotPositiveDefinite)
                      ? ErrorCode::NotPositiveDefinite
                      : ErrorCode::SingularMatrix;
    return static_cast<int>(info.status);
  }

  if (fkeep.use_banded_spd) {
    if (job != SolveJob::Full) {
      info.status = ErrorCode::FeatureNotAvailable;
      return static_cast<int>(ErrorCode::FeatureNotAvailable);
    }
    if (n != fkeep.band_n) {
      info.status = ErrorCode::IllegalValue;
      return static_cast<int>(ErrorCode::IllegalValue);
    }

    std::vector<double> rhs_perm(static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(nrhs));

    for (int rhs = 0; rhs < nrhs; ++rhs) {
      const double *src = b + static_cast<std::ptrdiff_t>(rhs) * n;
      double *dst = rhs_perm.data() + static_cast<std::ptrdiff_t>(rhs) * n;
      for (int j = 0; j < n; ++j) {
        dst[static_cast<std::size_t>(
            fkeep.iperm[static_cast<std::size_t>(j)])] = src[j];
      }
    }

    const int rc = smf_dpbtrs_lower(rhs_perm.data(), n, fkeep.band_kd, nrhs,
                                    fkeep.band_factor.data(),
                                    fkeep.band_kd + 1, n);
    if (rc != 0) {
      info.status = ErrorCode::SingularMatrix;
      return rc;
    }

    for (int rhs = 0; rhs < nrhs; ++rhs) {
      const double *src = rhs_perm.data() + static_cast<std::ptrdiff_t>(rhs) * n;
      double *dst = b + static_cast<std::ptrdiff_t>(rhs) * n;
      for (int new_i = 0; new_i < n; ++new_i) {
        dst[static_cast<std::size_t>(
            fkeep.perm[static_cast<std::size_t>(new_i)])] =
            src[new_i];
      }
    }

    auto t1 = std::chrono::steady_clock::now();
    info.solve_seconds += std::chrono::duration<double>(t1 - t0).count();
    info.status = ErrorCode::Success;
    return 0;
  }

  switch (job) {
  case SolveJob::Full:
    solve_forward(fkeep, b, n, nrhs);
    for (int j = 0; j < nrhs; ++j)
      solve_diag(fkeep, b + j * n, n);
    solve_backward(fkeep, b, n, nrhs);
    break;
  case SolveJob::Forward:
    solve_forward(fkeep, b, n, nrhs);
    break;
  case SolveJob::DiagOnly:
    for (int j = 0; j < nrhs; ++j)
      solve_diag(fkeep, b + j * n, n);
    break;
  case SolveJob::Backward:
    solve_backward(fkeep, b, n, nrhs);
    break;
  case SolveJob::DiagBack:
    for (int j = 0; j < nrhs; ++j)
      solve_diag(fkeep, b + j * n, n);
    solve_backward(fkeep, b, n, nrhs);
    break;
  }

  auto t1 = std::chrono::steady_clock::now();
  info.solve_seconds += std::chrono::duration<double>(t1 - t0).count();
  info.status = ErrorCode::Success;
  return 0;
}

} // namespace smf
