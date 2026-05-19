#include "smf/solver.hpp"
#include "smf/factor_indef.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/scaling.hpp"

namespace smf {

std::unique_ptr<FactorKeep> Solver::factor_solve(const CscLower &A,
                                                 const Control &ctrl,
                                                 Info &info, double *b, int n,
                                                 int nrhs) {
  // Scaling: if Equilibration requested, scale A and b before factorization
  std::vector<double> scale;
  const bool do_scale = (ctrl.scaling == ScalingMethod::Equilibration);

  if (do_scale) {
    scale = compute_equilibration_scale(A, 3);
  }

  // Use a scaled copy if needed, otherwise reference original
  CscLower A_scaled;
  const CscLower *A_ptr = &A;
  if (do_scale) {
    A_scaled = A; // copy
    apply_scale_to_matrix(A_scaled, scale);
    A_ptr = &A_scaled;
    apply_scale_to_rhs(b, n, nrhs, scale);
  }

  // Step 1: symbolic analysis
  auto ak = analyse(*A_ptr, ctrl, info);
  if (!ak)
    return nullptr;

  // Step 2: factorization
  FactorKeep fkeep;
  const FactorStatus fs = factor(*ak, ctrl, info, fkeep);
  if (fs != FactorStatus::Success)
    return nullptr;

  // Step 3: solve in-place (ak is still alive here)
  solve(fkeep, ctrl, info, b, n, nrhs);

  // Unscale solution: x <- S * x  (S is symmetric, so same operation)
  if (do_scale) {
    apply_scale_to_solution(b, n, nrhs, scale);
  }

  // Null out the raw analysis pointer before ak is destroyed to prevent
  // a dangling pointer in the returned FactorKeep.
  fkeep.analysis = nullptr;

  return std::make_unique<FactorKeep>(std::move(fkeep));
}

} // namespace smf
