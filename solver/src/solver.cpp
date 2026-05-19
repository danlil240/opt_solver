#include "smf/solver.hpp"
#include "smf/factor_indef.hpp"
#include "smf/factor_posdef.hpp"

namespace smf {

std::unique_ptr<FactorKeep> Solver::factor_solve(const CscLower &A,
                                                 const Control &ctrl,
                                                 Info &info, double *b, int n,
                                                 int nrhs) {
  // Step 1: symbolic analysis
  auto ak = analyse(A, ctrl, info);
  if (!ak)
    return nullptr;

  // Step 2: factorization
  FactorKeep fkeep;
  const FactorStatus fs = factor(*ak, ctrl, info, fkeep);
  if (fs != FactorStatus::Success)
    return nullptr;

  // Step 3: solve in-place (ak is still alive here)
  solve(fkeep, ctrl, info, b, n, nrhs);

  // Null out the raw analysis pointer before ak is destroyed to prevent
  // a dangling pointer in the returned FactorKeep.
  fkeep.analysis = nullptr;

  return std::make_unique<FactorKeep>(std::move(fkeep));
}

} // namespace smf
