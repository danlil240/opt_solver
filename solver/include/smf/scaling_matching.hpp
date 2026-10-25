#pragma once

#include "smf/csc_matrix.hpp"
#include "smf/error.hpp"
#include "smf/types.hpp"

#include <vector>

namespace smf {

struct MatchingResult {
    ErrorCode status = ErrorCode::Success;
    std::vector<double> scale;      ///< positive symmetric diagonal scaling S (length n)
    std::vector<Int> row_to_col;    ///< matched column for each row i (or -1 if unmatched)
    std::vector<Int> col_to_row;    ///< matched row for each col j (or -1 if unmatched)
    int matched_rank = 0;           ///< number of matched columns
};

/// Compute maximum-product bipartite matching and symmetric positive diagonal scaling
/// S = diag(scale) such that S * A * S has unit-bounded entries and unit-magnitude
/// matched entries, preserving symmetry and Sylvester inertia.
/// Returns ErrorCode::SingularMatrix if A is structurally singular (matched_rank < n).
MatchingResult compute_symmetric_matching(const CscLower &A);

ErrorCode compute_matching_scale(const CscLower &A,
                                 std::vector<double> &scale,
                                 int print_level = 0);

} // namespace smf
