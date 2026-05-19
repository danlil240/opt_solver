#pragma once
#include "smf/types.hpp"
#include <vector>

namespace smf {

/// Elimination tree of the Cholesky factor.
/// Computed from the *permuted* sparsity pattern (lower triangle) using
/// Liu (1986) path-compressed algorithm, O(n·α(n)).
struct EliminationTree {
  std::vector<Int> parent;    ///< parent[j] = parent of column j; -1 for roots
  std::vector<Int> postorder; ///< columns in postorder (leaves first)
  Int max_depth = 0;          ///< longest root-to-leaf path
};

/// Build the elimination tree for the *lower* sparsity pattern of A(perm,perm).
/// @param n        matrix order
/// @param col_ptr  CSC column pointers of the LOWER triangle (permuted pattern)
/// @param row_idx  CSC row indices of the LOWER triangle (permuted pattern)
/// @returns        EliminationTree with parent, postorder, max_depth filled in
EliminationTree build_elimination_tree(Int n, const std::vector<Int> &col_ptr,
                                       const std::vector<Int> &row_idx);

} // namespace smf
