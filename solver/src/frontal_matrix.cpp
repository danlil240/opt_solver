#include "smf/frontal_matrix.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace smf {

FrontalMatrix::FrontalMatrix(Int front_size, Int pivot_cols,
                             const Int* row_map, const Int* col_map,
                             AlignedArena& arena)
    : f_(front_size), p_(pivot_cols),
      row_map_(row_map), col_map_(col_map)
{
    assert(p_ >= 0 && p_ <= f_);

    const std::size_t nbytes =
        static_cast<std::size_t>(f_) * static_cast<std::size_t>(p_) * sizeof(double);
    if (nbytes > 0) {
        data_ = static_cast<double*>(arena.allocate(nbytes, 64));
        std::memset(data_, 0, nbytes);
    } else {
        data_ = nullptr;
    }
}

FrontalMatrix::FrontalMatrix(Int front_size, Int pivot_cols,
                             const std::vector<Int>& row_indices,
                             const std::vector<Int>& col_indices,
                             AlignedArena& arena)
    : FrontalMatrix(front_size, pivot_cols,
                    row_indices.data(), col_indices.data(), arena)
{
    owned_row_map_ = row_indices;
    owned_col_map_ = col_indices;
    row_map_ = owned_row_map_.data();
    col_map_ = owned_col_map_.data();
}

void FrontalMatrix::zero() noexcept
{
    const std::size_t nbytes =
        static_cast<std::size_t>(f_) * static_cast<std::size_t>(p_) * sizeof(double);
    if (nbytes > 0)
        std::memset(data_, 0, nbytes);
}

void FrontalMatrix::scatter_original(const std::vector<Int>& col_ptr,
                                     const std::vector<Int>& row_idx,
                                     const std::vector<double>& values)
{
    const Int* const row_end = row_map_ + f_;
    for (Int j = 0; j < p_; ++j) {
        const Int orig_col = col_map_[j];
        for (Int k = col_ptr[static_cast<std::size_t>(orig_col)];
             k < col_ptr[static_cast<std::size_t>(orig_col) + 1]; ++k) {
            const Int orig_row = row_idx[static_cast<std::size_t>(k)];
            // Binary search for orig_row in sorted row_map_
            const Int* it = std::lower_bound(row_map_, row_end, orig_row);
            if (it != row_end && *it == orig_row) {
                const Int front_row = static_cast<Int>(it - row_map_);
                at(front_row, j) += values[static_cast<std::size_t>(k)];
            }
        }
    }
}

void FrontalMatrix::assemble_contrib(const double* contrib, Int q,
                                     const std::vector<Int>& parent_rows)
{
    for (Int j = 0; j < q; ++j) {
        const Int pr_j = parent_rows[static_cast<std::size_t>(j)];
        if (pr_j >= p_) continue;   // extended column — no storage in this front
        const Int pcol_j = pr_j;    // pivot invariant

        // Lower triangle of contribution: entry (i,j) at contrib[j*q + i], i >= j
        for (Int i = j; i < q; ++i) {
            at(parent_rows[static_cast<std::size_t>(i)], pcol_j)
                += contrib[static_cast<std::size_t>(j) * static_cast<std::size_t>(q)
                           + static_cast<std::size_t>(i)];
        }
    }
}

} // namespace smf
