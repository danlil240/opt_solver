#pragma once
// solve_interleaved.hpp — serial full solve of several right-hand sides with
// the columns interleaved row by row, for factors of many tiny fronts.

#include "smf/factor_posdef.hpp"
#include "smf/workspace.hpp"

namespace smf {

/// Most columns one interleaved pass takes; more are solved in chunks.
inline constexpr int kInterleavedMaxRhs = 8;

/// True when solve_full_interleaved can take this factor: solve steps, and
/// D^-1 precomputed (FactorKeep::dinv) for an indefinite factor.
bool interleaved_solve_supported(const FactorKeep &fkeep);

/// x := S P (L D L^T)^-1 P^T S b for nrhs columns of b (stride ldb >= n),
/// serial, as solve_impl's full job does, but with the right-hand sides
/// interleaved (row i's nrhs values contiguous, padded to a vector width):
/// each factor entry is loaded once for all the columns instead of once per
/// column. Fronts of fewer than kInvDiagMinPivots pivots (without a cached
/// L11^-1) run lane kernels on the interleaved rows; larger ones gather a
/// column-major block and use BLAS-3. Requires
/// interleaved_solve_supported(fkeep).
void solve_full_interleaved(const FactorKeep &fkeep, double *b, int n, int nrhs, int ldb,
                            SolveWorkspace &ws);

} // namespace smf
