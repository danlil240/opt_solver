#pragma once
#include "smf/analysis.hpp"
#include "smf/factor_posdef.hpp"  // FactorKeep definition
#include <span>
#include <vector>

namespace smf {

/// Sparse forward solve: L y = b, exploiting sparsity of b.
///
/// This computes the same result as solve_forward() but skips supernodes
/// whose contribution to the solution is guaranteed to be zero (i.e., fronts
/// not in the "reach" of the non-zero entries of b in the supernode assembly
/// tree).
///
/// Algorithm:
///   1. Apply P^T permutation to b to get x (permuted ordering).
///   2. Detect seed supernodes: those containing a permuted column j where
///      x[j] != 0.
///   3. Traverse the supernode parent chain upward from each seed, collecting
///      the "reach" set (all ancestors of all seeds, including the seeds).
///   4. Process only the supernodes in the reach set, in postorder, applying
///      the same L-solve kernel as solve_forward().
///
/// Result semantics:
///   x_out is in the same state as the buffer after solve_forward() returns —
///   i.e., in permuted ordering.  The caller must apply the inverse
///   permutation (via solve_backward / P) if the original ordering is needed.
///
/// Correctness guarantee:
///   ‖solve_sparse_forward(b) − solve_forward(b)‖_∞ = 0 (exact equality in
///   exact arithmetic; floating-point result is bit-identical to the dense
///   path when non-reach supernodes truly do not contribute).
///
/// @param akeep  Symbolic analysis (elimination tree, supernode list, fronts).
/// @param fkeep  Numeric factor data (L blocks, permutation, is_posdef flag).
/// @param b      RHS vector in original ordering (length akeep.n). Read-only.
/// @param reach  Output: supernode indices (in postorder) that were processed.
///               Empty if b is zero.  Equals all supernodes if b is dense.
/// @param x_out  Output: L^{-1}(P^T b) in permuted ordering (length akeep.n).
void solve_sparse_forward(const AnalysisKeep     &akeep,
                          const FactorKeep        &fkeep,
                          std::span<const double>  b,
                          std::vector<int>        &reach,
                          std::vector<double>     &x_out);

} // namespace smf
