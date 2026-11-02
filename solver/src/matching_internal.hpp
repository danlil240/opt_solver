#pragma once

// The matching's cost graph and per-row state, shared by SymmetricMatcher
// (scaling_matching.cpp) and the auction kernel (auction_matching.cpp).

#include "smf/csc_matrix.hpp"
#include "smf/error.hpp"
#include "smf/types.hpp"

#include <limits>
#include <vector>

namespace smf::detail {

// Full symmetric pattern (both triangles) in CSC form, duplicates merged and
// explicit zeros dropped, with the matching cost of every entry:
//   cost(i, j) = log(col_max[j]) - log|a_ij| >= 0.
struct CostGraph {
    std::vector<Int> ptr;
    std::vector<int> row;
    std::vector<double> cost;
    std::vector<double> col_max;
};

// Builds g from A (shape already validated). IllegalValue on a non-finite
// entry, SingularMatrix on a column without nonzeros.
ErrorCode build_cost_graph(const CscLower &A, CostGraph &g);

// The per-row state every scanned edge reads: the row dual and the search
// distance, 16 bytes, so four rows share a cache line (the searches are
// bound by these random row accesses). The row's match, parent and flags,
// needed only once a row's distance improves or it is finalized, live in
// separate arrays of the matcher.
struct RowState {
    double u;      // row dual
    double dist;   // tentative search distance (inf when untouched)
};

constexpr double kInf = std::numeric_limits<double>::infinity();

} // namespace smf::detail
