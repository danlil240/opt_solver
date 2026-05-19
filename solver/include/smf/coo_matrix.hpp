#pragma once
#include "smf/csc_matrix.hpp"
#include "smf/types.hpp"
#include <vector>

namespace smf {

/// Coordinate (COO) format for symmetric sparse matrices.
/// Stores ALL explicitly provided entries — caller may include
/// upper-triangle entries or duplicates; coo_to_lower_csc() normalises.
struct CooMatrix {
    Int n{0};                   ///< matrix dimension
    std::vector<Int> row;       ///< row indices (0-based)
    std::vector<Int> col;       ///< column indices (0-based)
    std::vector<double> val;    ///< values

    Int nnz() const noexcept { return static_cast<Int>(val.size()); }
};

/// Convert coordinate format to lower-triangular CSC.
///
/// Rules applied (in order):
///  1. Entries with row < 0, col < 0, row >= n, col >= n are silently discarded.
///  2. Upper-triangle entries (col > row) are discarded (NOT reflected).
///  3. Diagonal entries (row == col) are kept.
///  4. Duplicate (row, col) pairs in the lower triangle are summed.
///  5. Within each column the resulting row indices are sorted ascending.
///
/// The returned CscLower is ready to pass directly to Solver::analyse().
CscLower coo_to_lower_csc(const CooMatrix& coo);

} // namespace smf
