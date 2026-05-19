#pragma once
#include "smf/factor_posdef.hpp"
#include <span>

namespace smf {

/// Backward solve: z = P L^{-T} y (where y is in permuted ordering, z in
/// original). On entry x = y (permuted); on exit x = P L^{-T} y (original
/// ordering).
/// @param nrhs number of RHS columns
void solve_backward(const FactorKeep &fkeep, double *x, int n, int nrhs = 1);

} // namespace smf
