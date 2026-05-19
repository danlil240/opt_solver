#pragma once
#include "smf/types.hpp"
#include "smf/supernode.hpp"
#include "smf/info.hpp"
#include "smf/csc_matrix.hpp"
#include <vector>

namespace smf {

/// Per-supernode data needed for assembly and factorization.
struct FrontalInfo {
    std::vector<Int> row_indices; ///< all row indices in the frontal matrix, sorted
    Int front_size() const { return static_cast<Int>(row_indices.size()); }
};

/// Compute per-supernode frontal row indices and populate prediction stats in Info.
/// Uses the original lower-CSC sparsity pattern (no symbolic fill).
///
/// @param A         cleaned lower-CSC matrix
/// @param supernodes  supernode list (from M2.S2), already in column order
/// @param info       output: predicted_factor_entries, predicted_flops, max_front_size,
///                           max_supernode_size, max_tree_depth populated
/// @returns FrontalInfo for each supernode (same indexing as supernodes)
std::vector<FrontalInfo> build_assembly_tree(
    const CscLower& A,
    const std::vector<Supernode>& supernodes,
    Info& info);

} // namespace smf
