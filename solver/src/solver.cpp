#include "smf/solver.hpp"
#include "smf/factor_indef.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/refinement.hpp"
#include "smf/scaling.hpp"
#include "smf/workspace.hpp"

#include <chrono>
#include <span>
#include <vector>

namespace smf {

std::unique_ptr<FactorKeep> Solver::factor_solve(const CscLower &A,
                                                 const Control &ctrl,
                                                 Info &info, double *b, int n,
                                                 int nrhs) {
  if (n <= 0 || nrhs <= 0 || b == nullptr || n != static_cast<int>(A.n)) {
    info.status = ErrorCode::IllegalValue;
    return nullptr;
  }

  std::vector<double> b_orig;
  if (ctrl.refinement_max_iters > 0) {
    b_orig.assign(b, b + static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs));
  }

  // Step 1: symbolic analysis
  auto ak = analyse(A, ctrl, info);
  if (!ak || info.status != ErrorCode::Success)
    return nullptr;

  // Step 2: factorization (computes and stores fkeep.scale when ctrl.scaling != None)
  NumericWorkspace nws;
  FactorKeep fkeep;
  const FactorStatus fs = factor(*ak, A, ctrl, info, fkeep, nws);
  if (fs != FactorStatus::Success) {
    info.factor_status = fs;
    if (info.status == ErrorCode::Success) {
      info.status = (fs == FactorStatus::NotPositiveDefinite)
                        ? ErrorCode::NotPositiveDefinite
                        : ErrorCode::SingularMatrix;
    }
    return nullptr;
  }

  // Step 3: solve in-place (applies fkeep.scale consistently)
  SolveWorkspace sws;
  const int solve_rc = solve(fkeep, ctrl, info, b, n, nrhs, sws, SolveJob::Full);
  if (solve_rc != 0) {
    return nullptr;
  }

  // Step 4: iterative refinement on original unscaled system when enabled
  if (ctrl.refinement_max_iters > 0) {
    const auto tr0 = std::chrono::steady_clock::now();
    const std::size_t total_len =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
    RefinementResult rr = refine(
        A,
        fkeep,
        ctrl,
        std::span<const double>(b_orig.data(), total_len),
        std::span<double>(b, total_len),
        nrhs,
        sws);
    const auto tr1 = std::chrono::steady_clock::now();
    info.refinement_seconds += std::chrono::duration<double>(tr1 - tr0).count();
    info.refinement_iterations = rr.iterations;
  }

  fkeep.analysis = nullptr;
  return std::make_unique<FactorKeep>(std::move(fkeep));
}

} // namespace smf
