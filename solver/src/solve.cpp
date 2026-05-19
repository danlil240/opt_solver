#include "smf/solve_backward.hpp"
#include "smf/solve_diag.hpp"
#include "smf/solve_forward.hpp"
#include "smf/solver.hpp"
#include <chrono>

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
  (void)ctrl;
  auto t0 = std::chrono::steady_clock::now();

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
  return 0;
}

} // namespace smf
