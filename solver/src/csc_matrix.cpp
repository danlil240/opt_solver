#include "smf/csc_matrix.hpp"

namespace smf {

ErrorCode CscLower::validate_shape() const noexcept {
    // col_ptr must have exactly n+1 entries
    if (col_ptr.size() != static_cast<std::size_t>(n + 1)) {
        return ErrorCode::IllegalValue;
    }

    // First entry must be zero (0-based indexing)
    if (col_ptr[0] != 0) {
        return ErrorCode::IllegalValue;
    }

    // col_ptr must be monotonically non-decreasing
    for (Int j = 0; j < n; ++j) {
        if (col_ptr[j + 1] < col_ptr[j]) {
            return ErrorCode::IllegalValue;
        }
    }

    const auto nnz_count = static_cast<std::size_t>(col_ptr[n]);

    // row_idx size must match nnz derived from col_ptr
    if (row_idx.size() != nnz_count) {
        return ErrorCode::IllegalValue;
    }

    // values size must match nnz
    if (values.size() != nnz_count) {
        return ErrorCode::IllegalValue;
    }

    // Every row index must be in [col, n) — lower triangle only
    for (Int j = 0; j < n; ++j) {
        for (Int k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            const Int row = row_idx[static_cast<std::size_t>(k)];
            if (row < j || row >= n) {
                return ErrorCode::IllegalValue;
            }
        }
    }

    return ErrorCode::Success;
}

} // namespace smf
