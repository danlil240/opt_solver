#pragma once
#include "smf/types.hpp"
#include "smf/supernode.hpp"
#include "smf/info.hpp"
#include "smf/csc_matrix.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace smf {

/// Compute every supernode's frontal row indices in CSR form and populate
/// prediction stats in Info. Uses the original lower-CSC sparsity pattern (no
/// symbolic fill): a front's rows are its pivot columns, its columns' pattern
/// and its children's extension rows.
///
/// @param A              cleaned (permuted) lower-CSC matrix
/// @param supernodes     supernode list in column order (postorder)
/// @param child_ptr      children CSR (build_children_csr), nsn + 1 entries
/// @param child_idx      children CSR indices
/// @param front_row_ptr  output: nsn + 1 offsets into front_row_idx
/// @param front_row_idx  output: each front's rows, sorted ascending
/// @param info           output: predicted_factor_entries, predicted_flops,
///                       max_front_size, max_supernode_size, max_tree_depth
/// @param rows_hint      expected total row count (capacity reserved up front;
///                       0 = unknown). Only a hint: any value gives the same result.
void build_assembly_tree(const CscLower &A, std::span<const Supernode> supernodes,
                         std::span<const Int> child_ptr, std::span<const Int> child_idx,
                         std::vector<Int> &front_row_ptr, std::vector<Int> &front_row_idx,
                         Info &info, std::size_t rows_hint = 0);

} // namespace smf
