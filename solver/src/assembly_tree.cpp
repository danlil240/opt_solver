#include "smf/assembly_tree.hpp"
#include <algorithm>
#include <numeric>

namespace smf {

void build_assembly_tree(const CscLower &A, std::span<const Supernode> supernodes,
                         std::span<const Int> child_ptr, std::span<const Int> child_idx,
                         std::vector<Int> &front_row_ptr, std::vector<Int> &front_row_idx,
                         Info &info, std::size_t rows_hint)
{
    const Int nsn = static_cast<Int>(supernodes.size());
    front_row_ptr.assign(static_cast<std::size_t>(nsn) + 1, 0);
    front_row_idx.clear();

    if (nsn == 0) {
        info.predicted_factor_entries = 0;
        info.predicted_flops = 0.0;
        info.max_front_size = 0;
        info.max_supernode_size = 0;
        info.max_tree_depth = 0;
        return;
    }

    // For each supernode, collect its row indices using:
    //   1. Direct pattern: all row_idx entries in columns col_start..col_end-1 of A
    //   2. Fill propagation: extension rows of all children (rows >= child.col_end)
    //      are inherited by the parent supernode.
    //
    // Supernodes are numbered in postorder so children always have smaller indices
    // than their parent. Processing in order 0..nsn-1 guarantees children are
    // complete before we read their rows for propagation.
    //
    // Rows are appended straight into front_row_idx (one array for all fronts)
    // and each front's range is sorted in place. mark[r] == s means row r was
    // added for supernode s; the generation number s is an implicit epoch, so
    // mark[] is never cleared. Children's rows are read by index, so growing
    // front_row_idx while appending is safe.
    if (rows_hint > 0)
        front_row_idx.reserve(rows_hint);
    std::vector<Int> mark(static_cast<std::size_t>(A.n), -1);

    for (Int s = 0; s < nsn; ++s) {
        const Supernode& sn = supernodes[static_cast<std::size_t>(s)];
        const std::size_t begin = front_row_idx.size();

        // Helper: mark row r as belonging to supernode s (idempotent)
        auto mark_row = [&](Int r) {
            if (mark[static_cast<std::size_t>(r)] != s) {
                mark[static_cast<std::size_t>(r)] = s;
                front_row_idx.push_back(r);
            }
        };

        // --- Step 1: Add diagonal entries (pivot columns of this supernode) ---
        for (Int j = sn.col_start; j < sn.col_end; ++j)
            mark_row(j);

        // --- Step 2: Add off-diagonal entries from the original sparsity pattern ---
        for (Int j = sn.col_start; j < sn.col_end; ++j) {
            for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
                 p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
                mark_row(A.row_idx[static_cast<std::size_t>(p)]);
            }
        }

        // --- Step 3: Fill propagation — inherit extension rows from all children ---
        // Extension rows of child C are its rows >= C.col_end (the child's
        // own pivot rows are skipped).
        for (Int t = child_ptr[static_cast<std::size_t>(s)];
             t < child_ptr[static_cast<std::size_t>(s) + 1]; ++t) {
            const Int c = child_idx[static_cast<std::size_t>(t)];
            const Int c_end = supernodes[static_cast<std::size_t>(c)].col_end;
            const Int r1 = front_row_ptr[static_cast<std::size_t>(c) + 1];
            for (Int k = front_row_ptr[static_cast<std::size_t>(c)]; k < r1; ++k) {
                const Int r = front_row_idx[static_cast<std::size_t>(k)];
                if (r >= c_end)
                    mark_row(r);
            }
        }

        // --- Step 4: Sort this front's rows in place ---
        std::sort(front_row_idx.begin() + static_cast<std::ptrdiff_t>(begin),
                  front_row_idx.end());
        front_row_ptr[static_cast<std::size_t>(s) + 1] =
            static_cast<Int>(front_row_idx.size());
    }

    // Compute prediction stats
    LongInt total_entries = 0;
    double  total_flops   = 0.0;
    int     max_front     = 0;
    int     max_sn_size   = 0;
    int     max_depth     = 0;

    // Compute tree depth for each supernode.
    // Supernodes are numbered in column order (postorder), so parent index > child index.
    // Process 0..nsn-1: for each sn, propagate depth[s]+1 to parent.
    std::vector<int> depth(static_cast<std::size_t>(nsn), 0);
    for (Int s = 0; s < nsn; ++s) {
        const Supernode& sn = supernodes[static_cast<std::size_t>(s)];
        const Int par = sn.parent;
        if (par >= 0)
            depth[static_cast<std::size_t>(par)] = std::max(
                depth[static_cast<std::size_t>(par)],
                depth[static_cast<std::size_t>(s)] + 1);
        max_depth = std::max(max_depth, depth[static_cast<std::size_t>(s)]);
    }
    // Also check roots (they may have been updated after their s-iteration)
    for (Int s = 0; s < nsn; ++s)
        max_depth = std::max(max_depth, depth[static_cast<std::size_t>(s)]);

    for (Int s = 0; s < nsn; ++s) {
        const Supernode& sn = supernodes[static_cast<std::size_t>(s)];
        const Int f = front_row_ptr[static_cast<std::size_t>(s) + 1] -
                      front_row_ptr[static_cast<std::size_t>(s)];
        const Int p = sn.width();
        total_entries += static_cast<LongInt>(f);
        total_flops   += static_cast<double>(p) * static_cast<double>(f) * static_cast<double>(f);
        max_front      = std::max(max_front, static_cast<int>(f));
        max_sn_size    = std::max(max_sn_size, static_cast<int>(p));
    }

    info.predicted_factor_entries = total_entries;
    info.predicted_flops          = total_flops;
    info.max_front_size           = max_front;
    info.max_supernode_size       = max_sn_size;
    info.max_tree_depth           = max_depth;
}

} // namespace smf
