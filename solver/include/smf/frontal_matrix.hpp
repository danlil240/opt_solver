#pragma once
#include "smf/types.hpp"
#include "smf/utils/memory.hpp"
#include <vector>

namespace smf {

/// Dense column-major frontal matrix for a single supernode.
///
/// A front with `f` rows and `p` pivot columns (p ≤ f) stores f*p doubles in
/// column-major order:
///   - Rows 0..p-1 : the p×p fully-summed (pivot) block (lower triangle + diagonal).
///   - Rows p..f-1 : the (f-p)×p update block (extended rows × pivot columns).
///
/// Memory: 64-byte aligned, arena-backed. The arena must outlive this object.
///
/// Invariant: col_map_[k] == row_map_[k] for k in [0, p) — the first p rows of
/// the front are the pivot rows, which are also the pivot columns.
class FrontalMatrix {
public:
    /// Fast pointer-based constructor — zero copies; row_map/col_map must outlive this.
    /// Typically pass fi.row_indices.data() for both: the first p entries serve as
    /// col_map and all f entries serve as row_map.
    FrontalMatrix(Int front_size, Int pivot_cols,
                  const Int* row_map, const Int* col_map,
                  AlignedArena& arena);

    /// Convenience constructor from vectors (delegates to pointer constructor).
    /// The vectors must outlive this object (pointers are stored, no copy).
    FrontalMatrix(Int front_size, Int pivot_cols,
                  const std::vector<Int>& row_indices,
                  const std::vector<Int>& col_indices,
                  AlignedArena& arena)
        : FrontalMatrix(front_size, pivot_cols,
                        row_indices.data(), col_indices.data(), arena) {}

    // ---- Scatter / assemble -------------------------------------------------

    /// Scatter entries from the original permuted lower-CSC into this front.
    void scatter_original(const std::vector<Int>& col_ptr,
                          const std::vector<Int>& row_idx,
                          const std::vector<double>& values);

    /// Add a (q×q) lower-triangular contribution block (column-major) into this front.
    void assemble_contrib(const double* contrib, Int q,
                          const std::vector<Int>& parent_rows);

    // ---- Accessors ----------------------------------------------------------

    Int front_size()  const noexcept { return f_; }
    Int pivot_cols()  const noexcept { return p_; }

    double*       data()       noexcept { return data_; }
    const double* data() const noexcept { return data_; }

    /// Column-major element access: row `row`, column `col`.
    double&       at(Int row, Int col)       noexcept { return data_[col * f_ + row]; }
    const double& at(Int row, Int col) const noexcept { return data_[col * f_ + row]; }

    /// Zero all f*p entries (preserves metadata).
    void zero() noexcept;

private:
    Int          f_;        ///< front size (total rows)
    Int          p_;        ///< pivot columns
    double*      data_;     ///< f*p doubles, column-major, arena-backed (NOT owned)
    const Int*   row_map_;  ///< length-f sorted original row indices (NOT owned)
    const Int*   col_map_;  ///< length-p original col indices (NOT owned)
};

} // namespace smf
