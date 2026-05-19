#pragma once
#include "smf/factor_posdef.hpp"
#include "smf/types.hpp"
#include <span>

namespace smf {

/// Forward solve: y = L^{-1} (P^T b)
/// Operates in-place: on entry, x holds b (in original ordering);
/// on exit, x holds L^{-1} (P^T b) (in permuted ordering).
/// For SPD (is_posdef=true): L from dpotrf, NON_UNIT dtrsv.
/// Indef unit-L path deferred to M4.G1.
/// @param nrhs  number of RHS columns (x is n × nrhs, column-major, ldx=n)
void solve_forward(const FactorKeep &fkeep, double *x, int n, int nrhs = 1);

} // namespace smf
