#include "smf/coo_matrix.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace smf {

CscLower coo_to_lower_csc(const CooMatrix& coo) {
    CscLower result;

    // Handle empty / zero-dimension matrix
    if (coo.n <= 0) {
        result.n = 0;
        result.col_ptr.assign(1, 0);
        return result;
    }

    const Int  n          = coo.n;
    const Int  num_input  = coo.nnz();
    const auto n_sz       = static_cast<std::size_t>(n);

    // ---------------------------------------------------------------
    // Step 1: count valid lower-triangle entries per column (pass 1).
    // Valid: row >= 0, col >= 0, row < n, col < n, row >= col.
    // ---------------------------------------------------------------
    std::vector<Int> col_count(n_sz, 0);

    for (Int k = 0; k < num_input; ++k) {
        const Int r = coo.row[static_cast<std::size_t>(k)];
        const Int c = coo.col[static_cast<std::size_t>(k)];
        if (r < 0 || c < 0 || r >= n || c >= n) continue;
        if (c > r) continue;  // discard upper triangle
        ++col_count[static_cast<std::size_t>(c)];
    }

    // ---------------------------------------------------------------
    // Step 2: build col_ptr from counts (prefix sum).
    // ---------------------------------------------------------------
    std::vector<Int> col_ptr(n_sz + 1u, 0);
    for (std::size_t j = 0; j < n_sz; ++j) {
        col_ptr[j + 1u] = col_ptr[j] + col_count[j];
    }
    const Int total_valid = col_ptr[n_sz];

    // ---------------------------------------------------------------
    // Step 3: scatter (row, val) pairs into columns (pass 2).
    // ---------------------------------------------------------------
    std::vector<Int>    tmp_row(static_cast<std::size_t>(total_valid));
    std::vector<double> tmp_val(static_cast<std::size_t>(total_valid));

    // cursor[j] = next write position within column j's range
    std::vector<Int> cursor(col_ptr.begin(), col_ptr.end());

    for (Int k = 0; k < num_input; ++k) {
        const Int    r = coo.row[static_cast<std::size_t>(k)];
        const Int    c = coo.col[static_cast<std::size_t>(k)];
        const double v = coo.val[static_cast<std::size_t>(k)];
        if (r < 0 || c < 0 || r >= n || c >= n) continue;
        if (c > r) continue;
        const Int pos = cursor[static_cast<std::size_t>(c)]++;
        tmp_row[static_cast<std::size_t>(pos)] = r;
        tmp_val[static_cast<std::size_t>(pos)] = v;
    }

    // ---------------------------------------------------------------
    // Step 4: for each column, sort by row index then sum duplicates.
    // Build final row_idx / values directly.
    // ---------------------------------------------------------------
    struct Pair { Int row; double val; };
    std::vector<Pair> pairs;

    result.n = n;
    result.col_ptr.resize(n_sz + 1u);
    result.col_ptr[0] = 0;
    result.row_idx.reserve(static_cast<std::size_t>(total_valid));
    result.values .reserve(static_cast<std::size_t>(total_valid));

    for (Int j = 0; j < n; ++j) {
        const Int col_start = col_ptr[static_cast<std::size_t>(j)];
        const Int col_end   = col_ptr[static_cast<std::size_t>(j) + 1u];
        const Int count     = col_end - col_start;

        if (count == 0) {
            result.col_ptr[static_cast<std::size_t>(j) + 1u] =
                static_cast<Int>(result.row_idx.size());
            continue;
        }

        if (count == 1) {
            result.row_idx.push_back(tmp_row[static_cast<std::size_t>(col_start)]);
            result.values .push_back(tmp_val[static_cast<std::size_t>(col_start)]);
            result.col_ptr[static_cast<std::size_t>(j) + 1u] =
                static_cast<Int>(result.row_idx.size());
            continue;
        }

        // Multiple entries: sort then merge duplicates
        pairs.clear();
        pairs.reserve(static_cast<std::size_t>(count));
        for (Int k = col_start; k < col_end; ++k) {
            const auto ks = static_cast<std::size_t>(k);
            pairs.push_back({tmp_row[ks], tmp_val[ks]});
        }

        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& a, const Pair& b) noexcept {
                      return a.row < b.row;
                  });

        // Merge runs with the same row index
        std::size_t read = 0u;
        while (read < pairs.size()) {
            std::size_t run_end = read + 1u;
            double      sum     = pairs[read].val;
            while (run_end < pairs.size() &&
                   pairs[run_end].row == pairs[read].row) {
                sum += pairs[run_end].val;
                ++run_end;
            }
            result.row_idx.push_back(pairs[read].row);
            result.values .push_back(sum);
            read = run_end;
        }

        result.col_ptr[static_cast<std::size_t>(j) + 1u] =
            static_cast<Int>(result.row_idx.size());
    }

    return result;
}

} // namespace smf
