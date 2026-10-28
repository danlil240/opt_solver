#pragma once
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include <vector>

namespace smf {

/// Result of cleaning a lower-CSC matrix.
struct CleanedPattern {
    CscLower          clean;             ///< Deduplicated, sorted lower CSC
    std::vector<Int>  original_to_clean; ///< Maps original entry index -> clean entry index (-1 = removed)
    int               duplicates    = 0; ///< Number of duplicate (i,j) pairs summed
    int               out_of_range  = 0; ///< Entries discarded (j<0, i<j, i>=n, j>=n)
    int               missing_diag  = 0; ///< Number of columns without a diagonal entry
    bool              was_clean     = false; ///< Input already clean (cleaned is a plain copy)
};

/// Clean a lower-CSC matrix:
///   - Discard entries where row < col (upper triangle), row >= n, col < 0, or row < 0.
///   - Sort each column's entries by row index (ascending).
///   - Sum duplicate (row, col) values.
///   - Count missing diagonal entries (columns where row_idx has no entry equal to col).
///   - Populate info.matrix_duplicates, info.matrix_out_of_range, info.matrix_missing_diag.
///
/// Does NOT insert missing diagonal entries.
/// Does NOT call any ordering or symbolic analysis.
///
/// check = false (Control::check_input): the caller guarantees clean input,
/// so the O(nnz) sortedness/range scan is skipped and A is taken as is (only
/// the shape checks and the O(n) missing-diagonal count run).
CleanedPattern clean_lower_csc(const CscLower& A, Info& info, bool check = true);

} // namespace smf
