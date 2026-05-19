#pragma once
#include "smf/types.hpp"
#include "smf/csc_matrix.hpp"
#include <vector>
#include <utility>

namespace smf {

/// Build symmetric adjacency graph from a lower-CSC matrix.
/// Result: CSR format {xadj, adjncy} with:
///   - no self-loops
///   - each undirected edge (i,j) stored as i->j AND j->i
///   - each row's adjacency sorted ascending
///   - xadj size = n+1, adjncy size = xadj[n]
std::pair<std::vector<Int>, std::vector<Int>>
build_symmetric_adjacency(const CscLower& A);

/// Structural rank estimate: upper bound via greedy row matching.
/// Documents: this is NOT the true structural rank (which requires maximum matching).
/// It is a greedy estimate: iterate columns, assign first unmatched row in that column.
/// Returns the number of matched pivots (<=  true structural rank <= n).
Int structural_rank(const CscLower& A);

} // namespace smf
