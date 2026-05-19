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
    /// Construct a zeroed front of size f×p backed by `arena`.
    ///
    /// @param front_size    f — total number of rows (pivot + extended)
    /// @param pivot_cols    p — number of pivot columns (p ≤ f)
    /// @param row_indices   length-f sorted original row indices
    /// @param col_indices   length-p sorted original column indices (= row_indices[0..p-1])
    /// @param arena         arena from which f*p*sizeof(double) bytes are allocated
    FrontalMatrix(Int front_size, Int pivot_cols,
                  const std::vector<Int>& row_indices,
                  const std::vector<Int>& col_indices,
                  AlignedArena& arena);

    // ---- Scatter / assemble -------------------------------------------------

    /// Scatter entries from the original permuted lower-CSC into this front.
    ///
    /// For each pivot column j (orig col = col_map_[j]):
    ///   for each (orig_row, val) in column orig_col of A:
    ///     find front_row = binary-search position of orig_row in row_map_ (sorted)
    ///     if found: at(front_row, j) += val
    ///
    /// @param col_ptr  size-(n+1) column pointer array of the permuted lower-CSC
    /// @param row_idx  row indices of the lower-CSC
    /// @param values   values of the lower-CSC
    void scatter_original(const std::vector<Int>& col_ptr,
                          const std::vector<Int>& row_idx,
                          const std::vector<double>& values);

    /// Add a (q×q) lower-triangular contribution block (column-major) into this front.
    ///
    /// Implements scatter-add with a precomputed scatter map (no map lookups in the
    /// inner loop). Each entry (i,j) of the contribution (lower-triangle, i ≥ j) is
    /// added to parent front position (parent_rows[i], parent_rows[j]) provided that
    /// parent_rows[j] < p_ (pivot invariant: col index = parent_rows[j]).
    ///
    /// @param contrib      q×q dense column-major block; entry (i,j) at contrib[j*q+i]
    /// @param q            size of the contribution block
    /// @param parent_rows  precomputed map: contribution index i → row in this front
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

    const std::vector<Int>& row_map() const noexcept { return row_map_; }
    const std::vector<Int>& col_map() const noexcept { return col_map_; }

    /// Zero all f*p entries (preserves metadata).
    void zero() noexcept;

private:
    Int              f_;        ///< front size (total rows)
    Int              p_;        ///< pivot columns
    double*          data_;     ///< f*p doubles, column-major, arena-backed (NOT owned)
    std::vector<Int> row_map_;  ///< row_map_[i] = original row index for front row i (sorted)
    std::vector<Int> col_map_;  ///< col_map_[j] = original col index for front col j (sorted)
};

} // namespace smf
