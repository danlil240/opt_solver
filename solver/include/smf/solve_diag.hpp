#pragma once
#include "smf/factor_posdef.hpp"

namespace smf {

/// Apply D^{-1} to x in-place (indefinite case only).
/// x is in permuted ordering, length n.
/// For each supernode s and each fully-summed column k:
///   1×1: x[fi.row_indices[k]] /= D[k,k]
///   2×2: solve [d00 d10; d10 d11]^{-1} * [x[r0]; x[r1]]
/// For SPD (is_posdef): this is a no-op.
void solve_diag(const FactorKeep &fkeep, double *x, int n);

/// Apply permutation P: x_out[j] = x_in[perm[j]] (perm[new]=old).
void apply_perm(const FactorKeep &fkeep, const double *x_in, double *x_out,
                int n);

/// Apply inverse permutation: x_out[j] = x_in[iperm[j]].
void apply_inv_perm(const FactorKeep &fkeep, const double *x_in, double *x_out,
                    int n);

} // namespace smf
