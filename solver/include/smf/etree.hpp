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

/// Same tree from the lower triangle stored by rows: row k lists the columns
/// c <= k with A(k, c) != 0 (any order; diagonal entries are ignored). Avoids
/// the internal transpose when the caller already has this structure.
EliminationTree build_elimination_tree_from_rows(Int n, const Int *row_ptr,
                                                 const Int *row_cols);

/// Postorder of a forest given by parent[] (parent[j] = -1 for roots), as
/// EliminationTree::postorder (roots and children in increasing index);
/// returns max_depth. `work` holds 4n + 1 ints.
Int etree_postorder(Int n, const Int *parent, Int *postorder, Int *work);

/// Elimination tree (parent array, new numbering) of P A P^T straight from
/// the symmetric adjacency of A (both triangles, no diagonal; see
/// lower_csc_adjacency), perm[new] = old and iperm[old] = new: no permuted
/// copy of A is formed. `ancestor` is n ints of workspace.
void etree_from_adjacency(Int n, const Int *xadj, const Int *adj, const Int *perm,
                          const Int *iperm, Int *parent, Int *ancestor);

} // namespace smf
