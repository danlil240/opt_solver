// solve_sparse_fwd.cpp — Sparse forward solve exploiting RHS sparsity.
//
// The algorithm is identical to solve_forward() in solve_forward.cpp, but
// only processes supernodes in the "reach" of the non-zero entries of b in
// the supernode assembly tree.  All other supernodes contribute zero to the
// forward solution and are skipped.
//
// See solve_sparse_fwd.hpp for the full API and algorithm description.

#include "smf/solve_sparse_fwd.hpp"

#include "smf/analysis.hpp"
#include "smf/assembly_tree.hpp"  // FrontalInfo
#include "smf/dense_kernel.hpp"   // smf_dtrsv_lower, smf_dtrsv_lower_unit, smf_dgemv
#include "smf/factor_posdef.hpp"  // FactorKeep
#include "smf/supernode.hpp"      // Supernode

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

namespace smf {

// ---------------------------------------------------------------------------
// Internal helper: compute postorder of the supernode assembly tree.
// Identical logic to the helper in solve_forward.cpp, duplicated here so
// solve_sparse_fwd.cpp has no dependency on internal translation units.
// ---------------------------------------------------------------------------
static std::vector<int>
compute_sn_postorder(const std::vector<Supernode> &supernodes) {
    const int ns = static_cast<int>(supernodes.size());
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(ns));

    // Iterative DFS: stack entries are (node_index, next_child_cursor).
    std::vector<std::pair<int, int>> stk;
    stk.reserve(static_cast<std::size_t>(ns));

    // Seed with all roots (parent == -1).
    for (int i = 0; i < ns; ++i) {
        if (supernodes[static_cast<std::size_t>(i)].parent == -1)
            stk.push_back({i, 0});
    }

    while (!stk.empty()) {
        auto &[node, ci] = stk.back();
        const int nch = static_cast<int>(
            supernodes[static_cast<std::size_t>(node)].children.size());
        if (ci < nch) {
            const int child =
                supernodes[static_cast<std::size_t>(node)]
                    .children[static_cast<std::size_t>(ci)];
            ++ci;
            stk.push_back({child, 0});
        } else {
            order.push_back(node);
            stk.pop_back();
        }
    }
    return order;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void solve_sparse_forward(const AnalysisKeep     &akeep,
                          const FactorKeep        &fkeep,
                          std::span<const double>  b,
                          std::vector<int>        &reach,
                          std::vector<double>     &x_out) {
    reach.clear();
    const int n = static_cast<int>(akeep.n);
    x_out.assign(static_cast<std::size_t>(n), 0.0);

    if (n == 0)
        return;

    const auto &supernodes = akeep.supernodes;
    const auto &fronts     = akeep.fronts;
    const auto &iperm      = fkeep.iperm;   // iperm[old] = new (permuted index)
    const int   ns         = static_cast<int>(supernodes.size());

    // ------------------------------------------------------------------
    // Step 1: Apply P^T permutation to b.
    //   x_out_perm[iperm[j]] = b[j]  for j = 0..n-1.
    // This matches the permutation step in solve_forward().
    // ------------------------------------------------------------------
    assert(static_cast<int>(b.size()) == n);
    for (int j = 0; j < n; ++j)
        x_out[static_cast<std::size_t>(
            iperm[static_cast<std::size_t>(j)])] =
            b[static_cast<std::size_t>(j)];

    // ------------------------------------------------------------------
    // Step 2: Build a column-to-supernode map.
    //   col_to_sn[c] = s  iff  supernodes[s].col_start <= c < col_end.
    // ------------------------------------------------------------------
    std::vector<int> col_to_sn(static_cast<std::size_t>(n), -1);
    for (int s = 0; s < ns; ++s) {
        const Supernode &sn = supernodes[static_cast<std::size_t>(s)];
        for (int c = sn.col_start; c < sn.col_end; ++c)
            col_to_sn[static_cast<std::size_t>(c)] = s;
    }

    // ------------------------------------------------------------------
    // Step 3: Identify seed supernodes.
    //   In permuted ordering, a column c is a "seed" if x_out[c] != 0.
    //   The supernode containing c is a seed supernode.
    //   Traverse each seed supernode's parent chain upward, marking every
    //   ancestor (including the seed itself) as being in the reach.
    // ------------------------------------------------------------------
    std::vector<bool> in_reach(static_cast<std::size_t>(ns), false);

    for (int c = 0; c < n; ++c) {
        if (x_out[static_cast<std::size_t>(c)] == 0.0)
            continue;
        // Walk supernode ancestor chain.
        int s = col_to_sn[static_cast<std::size_t>(c)];
        while (s != -1 && !in_reach[static_cast<std::size_t>(s)]) {
            in_reach[static_cast<std::size_t>(s)] = true;
            s = supernodes[static_cast<std::size_t>(s)].parent;
        }
    }

    // ------------------------------------------------------------------
    // Step 4: Collect reach supernodes in postorder.
    // ------------------------------------------------------------------
    const std::vector<int> postorder_all = compute_sn_postorder(supernodes);
    reach.reserve(static_cast<std::size_t>(ns));
    for (int s : postorder_all) {
        if (in_reach[static_cast<std::size_t>(s)])
            reach.push_back(s);
    }

    // ------------------------------------------------------------------
    // Step 5: Sparse forward substitution — process only reach supernodes.
    //   Logic is identical to solve_forward() for each supernode:
    //     (a) gather p pivot entries from x_out
    //     (b) solve L11 * y = b_loc  (non-unit or unit lower triangular)
    //     (c) scatter pivot entries back
    //     (d) update q extended rows: x_out[ext] -= L21 * y
    // ------------------------------------------------------------------
    std::vector<double> b_loc;
    std::vector<double> b_ext;

    for (int s : reach) {
        const FrontalInfo &fi = fronts[static_cast<std::size_t>(s)];
        const Supernode   &sn = supernodes[static_cast<std::size_t>(s)];

        const int p = static_cast<int>(sn.width());       // pivot columns
        const int f = static_cast<int>(fi.front_size());  // total front rows
        const int q = f - p;                               // extended rows

        if (p <= 0)
            continue;

        // Pointer to L data for supernode s: f×p col-major block, lda = f.
        const double *L =
            &fkeep.factor_values[static_cast<std::size_t>(
                fkeep.factor_col_ptr[static_cast<std::size_t>(s)])];

        // (a) Gather pivot values.
        b_loc.resize(static_cast<std::size_t>(p));
        for (int k = 0; k < p; ++k)
            b_loc[static_cast<std::size_t>(k)] =
                x_out[fi.row_indices[static_cast<std::size_t>(k)]];

        // (b) Solve L11 * y = b_loc.
        //   SPD path:  non-unit lower triangular (dpotrf stores full L).
        //   Indef path: unit lower triangular (diagonal stores D blocks).
        if (fkeep.is_posdef)
            smf_dtrsv_lower(b_loc.data(), p, L, f);
        else
            smf_dtrsv_lower_unit(b_loc.data(), p, L, f);

        // (c) Scatter pivot values back.
        for (int k = 0; k < p; ++k)
            x_out[fi.row_indices[static_cast<std::size_t>(k)]] =
                b_loc[static_cast<std::size_t>(k)];

        // (d) Update extended rows: x_out[ext] -= L21 * b_loc.
        if (q > 0) {
            b_ext.resize(static_cast<std::size_t>(q));

            // Gather extended rows.
            for (int k = 0; k < q; ++k)
                b_ext[static_cast<std::size_t>(k)] =
                    x_out[fi.row_indices[static_cast<std::size_t>(p + k)]];

            // L21 is rows [p, f) of the f×p block (col-major, lda = f).
            // b_ext = -1 * L21 * b_loc + 1 * b_ext
            smf_dgemv(b_ext.data(), q, p, L + p, f, b_loc.data(), -1.0, 1.0);

            // Scatter extended rows back.
            for (int k = 0; k < q; ++k)
                x_out[fi.row_indices[static_cast<std::size_t>(p + k)]] =
                    b_ext[static_cast<std::size_t>(k)];
        }
    }

    // x_out now holds L^{-1}(P^T b) in permuted ordering.
    // The caller is responsible for applying the inverse permutation P if
    // the result in original ordering is required (e.g., via solve_backward).
}

} // namespace smf
