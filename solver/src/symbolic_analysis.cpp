#include "smf/analysis.hpp"
#include "smf/assembly_tree.hpp"
#include "smf/check_matrix.hpp"
#include "smf/error.hpp"
#include "smf/etree.hpp"
#include "smf/ordering.hpp"
#include "smf/solver.hpp"
#include "smf/supernode.hpp"
#include "smf/sym_graph.hpp"
#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>

namespace smf
{

    static bool is_auto_ordering(OrderingMethod m)
    {
        return (m == OrderingMethod::AutoSerial ||
                m == OrderingMethod::AutoParallel);
    }

    static Int max_lower_bandwidth(const CscLower &A)
    {
        Int bw = 0;
        for (Int j = 0; j < A.n; ++j)
        {
            for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
                 p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            {
                const Int i = A.row_idx[static_cast<std::size_t>(p)];
                const Int d = i - j; // lower CSC => i >= j
                if (d > bw)
                    bw = d;
            }
        }
        return bw;
    }

    // Permute a lower-CSC matrix: given perm[new]=old, build the matrix
    // in the new ordering. For lower-CSC: entry (i,j) with i>=j in old ordering
    // becomes (iperm[i], iperm[j]) in new ordering; we keep only new_i >= new_j.
    static CscLower permute_lower_csc(const CscLower &A,
                                      const std::vector<Int> & /* perm */, // perm[new]=old
                                      const std::vector<Int> &iperm)       // iperm[old]=new
    {
        const Int n = A.n;
        // Count entries per new column
        std::vector<Int> count(static_cast<std::size_t>(n), 0);
        for (Int old_j = 0; old_j < n; ++old_j)
        {
            const Int new_j = iperm[static_cast<std::size_t>(old_j)];
            for (Int p = A.col_ptr[static_cast<std::size_t>(old_j)]; p < A.col_ptr[static_cast<std::size_t>(old_j) + 1];
                 ++p)
            {
                const Int old_i = A.row_idx[static_cast<std::size_t>(p)];
                const Int new_i = iperm[static_cast<std::size_t>(old_i)];
                if (new_i >= new_j)
                    ++count[static_cast<std::size_t>(new_j)];
                else
                    ++count[static_cast<std::size_t>(new_i)];
            }
        }

        CscLower B;
        B.n = n;
        B.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
        for (Int j = 0; j < n; ++j)
            B.col_ptr[static_cast<std::size_t>(j) + 1] =
                B.col_ptr[static_cast<std::size_t>(j)] + count[static_cast<std::size_t>(j)];
        const Int nnz = B.col_ptr[static_cast<std::size_t>(n)];
        B.row_idx.resize(static_cast<std::size_t>(nnz));
        B.values.resize(static_cast<std::size_t>(nnz), 0.0);

        std::vector<Int> pos(B.col_ptr.begin(), B.col_ptr.begin() + n);

        for (Int old_j = 0; old_j < n; ++old_j)
        {
            const Int new_j = iperm[static_cast<std::size_t>(old_j)];
            for (Int p = A.col_ptr[static_cast<std::size_t>(old_j)]; p < A.col_ptr[static_cast<std::size_t>(old_j) + 1];
                 ++p)
            {
                const Int old_i = A.row_idx[static_cast<std::size_t>(p)];
                const Int new_i = iperm[static_cast<std::size_t>(old_i)];
                Int col, row;
                if (new_i >= new_j)
                {
                    col = new_j;
                    row = new_i;
                }
                else
                {
                    col = new_i;
                    row = new_j;
                }
                const Int slot = pos[static_cast<std::size_t>(col)]++;
                B.row_idx[static_cast<std::size_t>(slot)] = row;
                B.values[static_cast<std::size_t>(slot)] = A.values[static_cast<std::size_t>(p)];
            }
        }

        // Sort each column's row indices (and values accordingly)
        // Workspace allocated once outside the loop; resize() does not reallocate
        // when shrinking, so at most O(max_col_width) allocations total.
        std::vector<std::pair<Int, double>> ws;
        for (Int j = 0; j < n; ++j)
        {
            const Int start = B.col_ptr[static_cast<std::size_t>(j)];
            const Int end = B.col_ptr[static_cast<std::size_t>(j) + 1];
            const Int len = end - start;
            if (len <= 1)
                continue;

            ws.resize(static_cast<std::size_t>(len));
            for (Int k = 0; k < len; ++k)
                ws[static_cast<std::size_t>(k)] = {B.row_idx[static_cast<std::size_t>(start + k)],
                                                    B.values[static_cast<std::size_t>(start + k)]};
            std::sort(ws.begin(), ws.end(),
                      [](const std::pair<Int, double> &a, const std::pair<Int, double> &b) {
                          return a.first < b.first;
                      });
            for (Int k = 0; k < len; ++k)
            {
                B.row_idx[static_cast<std::size_t>(start + k)] = ws[static_cast<std::size_t>(k)].first;
                B.values[static_cast<std::size_t>(start + k)] = ws[static_cast<std::size_t>(k)].second;
            }
        }

        return B;
    }

    static std::vector<Int> compute_supernode_postorder(const std::vector<Supernode> &supernodes)
    {
        const Int ns = static_cast<Int>(supernodes.size());
        std::vector<Int> order;
        order.reserve(static_cast<std::size_t>(ns));

        std::vector<std::pair<Int, Int>> stack;
        stack.reserve(static_cast<std::size_t>(ns));

        for (Int i = 0; i < ns; ++i)
        {
            if (supernodes[static_cast<std::size_t>(i)].parent == -1)
                stack.push_back({i, 0});
        }

        while (!stack.empty())
        {
            auto &[node, child_pos] = stack.back();
            const auto &children = supernodes[static_cast<std::size_t>(node)].children;
            if (child_pos < static_cast<Int>(children.size()))
            {
                const Int child = children[static_cast<std::size_t>(child_pos)];
                ++child_pos;
                stack.push_back({child, 0});
            }
            else
            {
                order.push_back(node);
                stack.pop_back();
            }
        }

        return order;
    }

    std::unique_ptr<AnalysisKeep> Solver::analyse(const CscLower &A, const Control &control, Info &info)
    {
        const auto t0 = std::chrono::steady_clock::now();

        info = Info{}; // zero-initialise
        info.status = ErrorCode::Success;

        // Step 1: clean the input matrix
        auto cleaned = clean_lower_csc(A, info);
        const Int n = cleaned.clean.n;

        if (n == 0)
        {
            // Trivially empty analysis
            auto keep = std::make_unique<AnalysisKeep>();
            keep->n = 0;
            keep->cleaned = std::move(cleaned.clean);
            const auto t1 = std::chrono::steady_clock::now();
            info.analyse_seconds = std::chrono::duration<double>(t1 - t0).count();
            return keep;
        }

        // Step 2: choose ordering strategy.
        // For narrow-band matrices in Auto modes, identity ordering preserves
        // structure and avoids expensive ordering/permutation overhead.
        const bool auto_order = is_auto_ordering(control.ordering);
        const Int half_bw = max_lower_bandwidth(cleaned.clean);
        const bool use_identity_ordering = auto_order && (half_bw <= 64);

        std::vector<Int> perm(static_cast<std::size_t>(n));
        if (use_identity_ordering)
        {
            std::iota(perm.begin(), perm.end(), 0);
            info.ordering_used = static_cast<int>(OrderingMethod::UserSupplied);
        }
        else
        {
            auto backend = make_ordering_backend(control.ordering, n, info);

            // Step 3: build symmetric adjacency graph for the cleaned pattern.
            auto [xadj, adjncy] = build_symmetric_adjacency(cleaned.clean);

            // Step 4: compute fill-reducing permutation.
            perm = backend->compute_ordering(n, xadj, adjncy);
            if (perm.empty())
            {
                // Ordering failed — fall back to identity permutation.
                info.status = ErrorCode::Success;
                perm.resize(static_cast<std::size_t>(n));
                std::iota(perm.begin(), perm.end(), 0);
                info.ordering_used = static_cast<int>(OrderingMethod::UserSupplied);
            }
            else
            {
#if defined(SMF_HAS_METIS) && SMF_HAS_METIS
                info.ordering_used = static_cast<int>(
                    control.ordering == OrderingMethod::AutoParallel
                        ? OrderingMethod::METIS
                        : (control.ordering == OrderingMethod::AutoSerial
                               ? OrderingMethod::AMD
                               : control.ordering));
#else
                info.ordering_used = static_cast<int>(
                    control.ordering == OrderingMethod::AutoParallel
                        ? OrderingMethod::AMD
                        : (control.ordering == OrderingMethod::AutoSerial
                               ? OrderingMethod::AMD
                               : control.ordering));
#endif
            }
        }

        // Build inverse permutation
        std::vector<Int> iperm(static_cast<std::size_t>(n));
        for (Int i = 0; i < n; ++i)
            iperm[static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])] = i;

        // Step 5: permute the cleaned matrix.
        CscLower A_perm;
        const CscLower *A_ord = nullptr;
        if (use_identity_ordering)
            A_ord = &cleaned.clean;
        else
        {
            A_perm = permute_lower_csc(cleaned.clean, perm, iperm);
            A_ord = &A_perm;
        }

        // Step 6: build elimination tree
        EliminationTree etree = build_elimination_tree(n, A_ord->col_ptr, A_ord->row_idx);

        // Step 7: detect and amalgamate supernodes
        auto snodes = detect_fundamental_supernodes(n, A_ord->col_ptr, A_ord->row_idx, etree);
        snodes = amalgamate_supernodes(std::move(snodes), static_cast<Int>(control.nemin));

        // Step 8: build assembly tree and fill info predictions
        auto fronts = build_assembly_tree(*A_ord, snodes, info);

        // Step 8.5a: compute supernode postorder (children before parents) via iterative DFS.
        const Int nsn = static_cast<Int>(snodes.size());
        std::vector<Int> sn_postorder;
        sn_postorder.reserve(static_cast<std::size_t>(nsn));
        {
            std::vector<std::pair<Int,Int>> stk; // (node, next_child_idx)
            stk.reserve(static_cast<std::size_t>(nsn));
            for (Int i = 0; i < nsn; ++i)
                if (snodes[static_cast<std::size_t>(i)].parent == -1)
                    stk.push_back({i, 0});
            while (!stk.empty()) {
                auto& [node, ci] = stk.back();
                const Int nch = static_cast<Int>(snodes[static_cast<std::size_t>(node)].children.size());
                if (ci < nch) {
                    const Int child = snodes[static_cast<std::size_t>(node)].children[static_cast<std::size_t>(ci)];
                    ++ci;
                    stk.push_back({child, 0});
                } else {
                    sn_postorder.push_back(node);
                    stk.pop_back();
                }
            }
        }

        // Step 8.5b: build inverse postorder for sort key.
        std::vector<Int> po_idx(static_cast<std::size_t>(nsn));
        for (Int i = 0; i < nsn; ++i)
            po_idx[static_cast<std::size_t>(sn_postorder[static_cast<std::size_t>(i)])] = i;

        // Step 8.5c: pre-sort each supernode's children by descending postorder index.
        // Factor's LIFO fstack freeing requires this order; doing it once in analysis
        // eliminates the per-supernode sorted_ch vector + std::sort in the factor hot loop.
        for (Int s = 0; s < nsn; ++s) {
            auto& ch = snodes[static_cast<std::size_t>(s)].children;
            if (ch.size() > 1)
                std::sort(ch.begin(), ch.end(), [&](Int a, Int b) {
                    return po_idx[static_cast<std::size_t>(a)] >
                           po_idx[static_cast<std::size_t>(b)];
                });
        }

        // Step 8.5d: pre-compute child-parent row scatter maps only for smaller
        // matrices; for larger cases this can dominate one-shot analyse time.
        const bool precompute_child_maps = (n <= 1200);
        if (precompute_child_maps) {
            for (Int s = 0; s < nsn; ++s) {
                const Supernode& sn = snodes[static_cast<std::size_t>(s)];
                FrontalInfo& fi = fronts[static_cast<std::size_t>(s)];
                const std::size_t nch = sn.children.size();
                fi.child_parent_rows.resize(nch);
                for (std::size_t k = 0; k < nch; ++k) {
                    const Int c = sn.children[k];
                    const FrontalInfo& cfi = fronts[static_cast<std::size_t>(c)];
                    const Int cp = snodes[static_cast<std::size_t>(c)].width();
                    const Int q_c = cfi.front_size() - cp;
                    fi.child_parent_rows[k].resize(static_cast<std::size_t>(q_c));
                    for (Int i = 0; i < q_c; ++i) {
                        const Int ext_row = cfi.row_indices[static_cast<std::size_t>(cp + i)];
                        const auto it = std::lower_bound(fi.row_indices.begin(),
                                                         fi.row_indices.end(), ext_row);
                        fi.child_parent_rows[k][static_cast<std::size_t>(i)] =
                            static_cast<Int>(it - fi.row_indices.begin());
                    }
                }
            }
        }

        // Step 9: record timing
        const auto t1 = std::chrono::steady_clock::now();
        info.analyse_seconds = std::chrono::duration<double>(t1 - t0).count();

        // Step 9.5: build orig_to_perm_idx scatter map (before moving A_perm).
        // For each entry k in cleaned (0-based), compute the slot in A_perm where it lands.
        // Since A_perm.row_idx is sorted within each column (post-sort from permute_lower_csc),
        // we use std::lower_bound: O(nnz * log(max_col_width)).
        // This map enables O(nnz) value-only permutation on every factorization call.
        std::vector<Int> otp;
        if (!use_identity_ordering) {
            const Int nnz_orig = static_cast<Int>(cleaned.clean.values.size());
            otp.resize(static_cast<std::size_t>(nnz_orig));
            for (Int old_j = 0; old_j < n; ++old_j)
            {
                const Int new_j = iperm[static_cast<std::size_t>(old_j)];
                for (Int k = cleaned.clean.col_ptr[static_cast<std::size_t>(old_j)];
                     k < cleaned.clean.col_ptr[static_cast<std::size_t>(old_j) + 1]; ++k)
                {
                    const Int old_i =
                        cleaned.clean.row_idx[static_cast<std::size_t>(k)];
                    const Int new_i = iperm[static_cast<std::size_t>(old_i)];
                    // Determine which column this entry lands in after lower-triangle
                    // enforcement: col = min(new_i, new_j), row = max(new_i, new_j).
                    const Int perm_col = (new_i >= new_j) ? new_j : new_i;
                    const Int perm_row = (new_i >= new_j) ? new_i : new_j;
                    // Binary-search within the sorted column of A_perm.
                    const Int cs =
                        A_perm.col_ptr[static_cast<std::size_t>(perm_col)];
                    const Int ce =
                        A_perm.col_ptr[static_cast<std::size_t>(perm_col) + 1];
                    auto it = std::lower_bound(
                        A_perm.row_idx.begin() + cs,
                        A_perm.row_idx.begin() + ce,
                        perm_row);
                    otp[static_cast<std::size_t>(k)] =
                        static_cast<Int>(it - A_perm.row_idx.begin());
                }
            }
        }

        // Step 10: assemble AnalysisKeep
        auto keep = std::make_unique<AnalysisKeep>();
        keep->n = n;
        keep->cleaned = std::move(cleaned.clean);
        // Store permuted pattern for fast value-only permutation in factor.
        // For identity ordering we keep these empty and factor reuses cleaned pattern.
        if (!use_identity_ordering) {
            keep->perm_col_ptr = std::move(A_perm.col_ptr);
            keep->perm_row_idx = std::move(A_perm.row_idx);
        }
        keep->orig_to_perm_idx = std::move(otp);
        keep->perm = std::move(perm);
        keep->iperm = std::move(iperm);
        keep->etree = std::move(etree);
        keep->supernodes = std::move(snodes);
        keep->fronts = std::move(fronts);
        keep->postorder = std::move(sn_postorder);
        keep->solve_postorder = keep->postorder;

        return keep;
    }

} // namespace smf
