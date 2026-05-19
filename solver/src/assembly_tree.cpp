#include "smf/assembly_tree.hpp"
#include <algorithm>
#include <numeric>

namespace smf {

std::vector<FrontalInfo> build_assembly_tree(
    const CscLower& A,
    const std::vector<Supernode>& supernodes,
    Info& info)
{
    const Int nsn = static_cast<Int>(supernodes.size());
    std::vector<FrontalInfo> fronts(static_cast<std::size_t>(nsn));

    if (nsn == 0) return fronts;

    // For each supernode, collect its row indices using:
    //   1. Direct pattern: all row_idx entries in columns col_start..col_end-1 of A
    //   2. Fill propagation: extension rows of all children (rows >= child.col_end)
    //      are inherited by the parent supernode.
    //
    // Supernodes are numbered in postorder so children always have smaller indices
    // than their parent. Processing in order 0..nsn-1 guarantees children are
    // complete before we read their row_indices for propagation.
    for (Int s = 0; s < nsn; ++s) {
        const Supernode& sn = supernodes[static_cast<std::size_t>(s)];
        FrontalInfo& fi = fronts[static_cast<std::size_t>(s)];

        std::vector<bool> seen(static_cast<std::size_t>(A.n), false);

        // --- Step 1: Add diagonal entries (pivot columns of this supernode) ---
        for (Int j = sn.col_start; j < sn.col_end; ++j)
            seen[static_cast<std::size_t>(j)] = true;

        // --- Step 2: Add off-diagonal entries from the original sparsity pattern ---
        for (Int j = sn.col_start; j < sn.col_end; ++j) {
            for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
                 p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
                seen[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)])] = true;
            }
        }

        // --- Step 3: Fill propagation — inherit extension rows from all children ---
        // For child C, extension rows are those in C.row_indices with index >= C.col_end
        // (i.e. rows that were not pivot columns of C).  All such rows belong in the
        // parent's frontal matrix (they are >= C.col_end >= sn.col_start because
        // elimination is ordered: C.col_end <= sn.col_start in a valid assembly tree).
        for (Int c : sn.children) {
            const FrontalInfo& cfi = fronts[static_cast<std::size_t>(c)];
            const Supernode& csn   = supernodes[static_cast<std::size_t>(c)];
            // Extension rows start where the child's pivot columns end
            for (const Int r : cfi.row_indices) {
                if (r >= csn.col_end)          // skip the child's own pivot rows
                    seen[static_cast<std::size_t>(r)] = true;
            }
        }

        // --- Step 4: Collect sorted row indices ---
        for (Int i = 0; i < A.n; ++i)
            if (seen[static_cast<std::size_t>(i)]) fi.row_indices.push_back(i);
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
        const FrontalInfo& fi = fronts[static_cast<std::size_t>(s)];
        const Int f = fi.front_size();
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

    return fronts;
}

} // namespace smf
