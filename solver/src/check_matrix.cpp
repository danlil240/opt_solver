#include "smf/check_matrix.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace smf {

CleanedPattern clean_lower_csc(const CscLower& A, Info& info) {
    CleanedPattern result;

    // Validate n
    if (A.n < 0) {
        info.status = ErrorCode::IllegalValue;
        return result;
    }
    const Int  n    = A.n;
    const auto n_sz = static_cast<std::size_t>(n);

    // col_ptr must have exactly n+1 entries
    if (A.col_ptr.size() != n_sz + 1u) {
        info.status = ErrorCode::IllegalValue;
        return result;
    }

    // col_ptr[0] must be zero
    if (A.col_ptr[0] != 0) {
        info.status = ErrorCode::IllegalValue;
        return result;
    }

    // col_ptr must be monotonically non-decreasing
    for (Int j = 0; j < n; ++j) {
        if (A.col_ptr[static_cast<std::size_t>(j) + 1u] <
            A.col_ptr[static_cast<std::size_t>(j)]) {
            info.status = ErrorCode::IllegalValue;
            return result;
        }
    }

    const Int  nnz    = A.col_ptr[n_sz];
    const auto nnz_sz = static_cast<std::size_t>(nnz);

    // row_idx and values sizes must match nnz
    if (A.row_idx.size() != nnz_sz || A.values.size() != nnz_sz) {
        info.status = ErrorCode::IllegalValue;
        return result;
    }

    // Initialise output
    result.original_to_clean.assign(nnz_sz, -1);
    result.clean.n = n;
    result.clean.col_ptr.resize(n_sz + 1u);
    result.clean.col_ptr[0] = 0;

    // Per-column working buffer
    struct Entry {
        Int    row;
        double val;
        Int    orig_k;
    };
    std::vector<Entry> entries;

    Int clean_idx = 0;

    for (Int j = 0; j < n; ++j) {
        entries.clear();

        const Int col_start = A.col_ptr[static_cast<std::size_t>(j)];
        const Int col_end   = A.col_ptr[static_cast<std::size_t>(j) + 1u];

        for (Int k = col_start; k < col_end; ++k) {
            const Int    i = A.row_idx[static_cast<std::size_t>(k)];
            const double v = A.values[static_cast<std::size_t>(k)];

            // i < j also catches i < 0 because j >= 0
            if (i < j || i >= n) {
                ++result.out_of_range;
                // original_to_clean[k] stays -1
            } else {
                entries.push_back({i, v, k});
            }
        }

        // Sort valid entries by row (ascending)
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) noexcept {
                      return a.row < b.row;
                  });

        // Single-pass deduplication: merge runs with same row index
        bool        has_diagonal = false;
        std::size_t pos          = 0u;

        while (pos < entries.size()) {
            // Accumulate all entries in this row-run
            std::size_t run_end = pos + 1u;
            double      sum_val = entries[pos].val;

            while (run_end < entries.size() &&
                   entries[run_end].row == entries[pos].row) {
                sum_val += entries[run_end].val;
                ++result.duplicates;
                ++run_end;
            }

            // Emit the surviving merged entry
            const Int clean_pos = clean_idx;
            result.clean.row_idx.push_back(entries[pos].row);
            result.clean.values.push_back(sum_val);

            // Map every original entry in the run to this clean position
            for (std::size_t m = pos; m < run_end; ++m) {
                result.original_to_clean[
                    static_cast<std::size_t>(entries[m].orig_k)] = clean_pos;
            }

            if (entries[pos].row == j) {
                has_diagonal = true;
            }

            ++clean_idx;
            pos = run_end;
        }

        // Column without any diagonal entry counts as missing
        if (!has_diagonal) {
            ++result.missing_diag;
        }

        result.clean.col_ptr[static_cast<std::size_t>(j) + 1u] = clean_idx;
    }

    // Propagate counts to caller's Info struct
    info.matrix_duplicates   = result.duplicates;
    info.matrix_out_of_range = result.out_of_range;
    info.matrix_missing_diag = result.missing_diag;

    return result;
}

} // namespace smf
