#pragma once
#include "smf/types.hpp"
#include "smf/etree.hpp"
#include "smf/csc_matrix.hpp"
#include <vector>

namespace smf {

/// A supernode (relaxed fundamental supernode after amalgamation).
/// Columns col_start .. col_end-1 (inclusive) form the supernode.
struct Supernode {
    Int col_start;  ///< first column in this supernode (0-based)
    Int col_end;    ///< one-past-last column (col_end - col_start = width)
    Int parent;     ///< index of parent supernode in supernode tree; -1 for roots
    std::vector<Int> children; ///< indices of child supernodes

    /// Width (number of pivot columns) of this supernode.
    Int width() const { return col_end - col_start; }
};

/// Detect fundamental supernodes from the elimination tree and sparsity pattern.
///
/// A fundamental supernode is a maximal set of consecutive columns j..k where:
///   - parent[j] = j+1, ..., parent[k-1] = k  (chain in etree)
///   - col_count[j] = col_count[j+1]+1, ..., col_count[k-1] = col_count[k]+1
///   - each of j..k-1 has exactly one child in the etree (which must be j-1..k-2)
///
/// Column j continues the previous supernode iff ALL hold:
///   (1) etree.parent[j-1] == j
///   (2) nchildren[j] == 1
///   (3) col_count[j-1] == col_count[j] + 1
///
/// @param n         matrix order
/// @param col_ptr   CSC column pointers of the lower factor pattern (or original)
/// @param row_idx   CSC row indices (structural; not used for count-based detection)
/// @param etree     precomputed elimination tree
/// @returns         vector of fundamental Supernode objects in column/postorder
std::vector<Supernode> detect_fundamental_supernodes(
    Int n,
    const std::vector<Int>& col_ptr,
    const std::vector<Int>& row_idx,
    const EliminationTree& etree);

/// Amalgamate supernodes using the nemin threshold.
///
/// Two adjacent supernodes S (child) and P (parent) are merged when:
///   S.width() < nemin  AND  P.width() < nemin  AND  P.children.size() == 1
///
/// Merging absorbs S into P: P.col_start = S.col_start; P inherits S's children.
/// Repeated bottom-up until stable.
///
/// @param supernodes  fundamental supernodes (taken by value; modified internally)
/// @param nemin       minimum supernode width threshold (from Control::nemin)
/// @returns           compacted vector of non-merged supernodes in column order
std::vector<Supernode> amalgamate_supernodes(
    std::vector<Supernode> supernodes,
    Int nemin);

} // namespace smf
