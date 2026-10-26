#pragma once
#include "smf/factor_posdef.hpp"
#include <span>

namespace smf {

/// Backward solve: z = P L^{-T} y (where y is in permuted ordering, z in
/// original). On entry x = y (permuted); on exit x = P L^{-T} y (original
/// ordering).
/// @param nrhs number of RHS columns
void solve_backward(const FactorKeep &fkeep, double *x, int n, int nrhs = 1);

/// num_threads > 1: see solve_forward. apply_diag also applies D^-1 front by
/// front; only valid when backward_applies_diag(fkeep). permute = false
/// leaves x in permuted order (the caller applies P).
void solve_backward(const FactorKeep &fkeep, double *x, int n, int nrhs,
                    SolveWorkspace &ws, int num_threads = 1, bool apply_diag = false,
                    bool permute = true);

/// True when solve_backward can apply D itself (indefinite factor with D
/// inline, FactorKeep::inline_d), saving the separate solve_diag pass.
inline bool backward_applies_diag(const FactorKeep &fkeep) {
  return fkeep.inline_d && !fkeep.is_posdef && !fkeep.solve_steps.empty();
}

} // namespace smf
