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

    // Rows are sorted within each cleaned column, so the last one is the
    // farthest from the diagonal.
    static Int max_lower_bandwidth(const CscLower &A)
    {
        Int bw = 0;
        for (Int j = 0; j < A.n; ++j)
        {
            const Int p1 = A.col_ptr[static_cast<std::size_t>(j) + 1];
            if (p1 > A.col_ptr[static_cast<std::size_t>(j)])
                bw = std::max(bw, A.row_idx[static_cast<std::size_t>(p1) - 1] - j);
        }
        return bw;
    }

    // Lower triangle of P A P^T by rows (ip = inverse permutation): row r
    // lists the columns c <= r with an entry, and with `src` also the cleaned
    // entry it came from. With `col_count`, the entries per permuted column
    // are counted in the same sweep.
    static void build_row_form(const CscLower &C, const Int *ip, Int *row_ptr,
                               Int *row_col, Int *src, Int *col_count, Int *head)
    {
        const Int n = C.n;
        const Int *cp = C.col_ptr.data();
        const Int *ri = C.row_idx.data();
        std::fill(row_ptr, row_ptr + n + 1, 0);
        if (col_count != nullptr)
            std::fill(col_count, col_count + n, 0);
        for (Int j = 0; j < n; ++j)
        {
            const Int pj = ip[j];
            for (Int k = cp[j]; k < cp[j + 1]; ++k)
            {
                const Int pi = ip[ri[k]];
                ++row_ptr[std::max(pi, pj) + 1];
                if (col_count != nullptr)
                    ++col_count[std::min(pi, pj)];
            }
        }
        for (Int r = 0; r < n; ++r)
            row_ptr[r + 1] += row_ptr[r];
        std::copy(row_ptr, row_ptr + n, head);
        for (Int j = 0; j < n; ++j)
        {
            const Int pj = ip[j];
            for (Int k = cp[j]; k < cp[j + 1]; ++k)
            {
                const Int pi = ip[ri[k]];
                const Int slot = head[std::max(pi, pj)]++;
                row_col[slot] = std::min(pi, pj);
                if (src != nullptr)
                    src[slot] = k;
            }
        }
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
            keep->orig_col_ptr = A.col_ptr;
            keep->orig_row_idx = A.row_idx;
            keep->cleaned = std::move(cleaned.clean);
            const auto t1 = std::chrono::steady_clock::now();
            info.analyse_seconds = std::chrono::duration<double>(t1 - t0).count();
            return keep;
        }

        // Step 2: choose ordering strategy.
        // Identity ordering is optimal for genuinely banded matrices (the
        // band is densely occupied, e.g. tridiagonal / block-banded KKT), and
        // ordering overhead does not amortise on small problems.  A matrix
        // like 2D Poisson has a *sparse* band (bw~63 but only ~5 nnz/col), so
        // it falls through to the fill-reducing backend on large n.
        const bool auto_order = is_auto_ordering(control.ordering);
        const Int half_bw = max_lower_bandwidth(cleaned.clean);
        const LongInt nnz_lower = cleaned.clean.col_ptr[static_cast<std::size_t>(n)];
        const bool dense_band =
            static_cast<LongInt>(half_bw + 1) * static_cast<LongInt>(n) <=
            2 * nnz_lower;
        const bool use_identity_ordering =
            auto_order && (half_bw <= 64) && (dense_band || n <= 2048);

        std::vector<Int> perm(static_cast<std::size_t>(n));
        if (use_identity_ordering)
        {
            std::iota(perm.begin(), perm.end(), 0);
            info.ordering_used = static_cast<int>(OrderingMethod::UserSupplied);
        }
        else
        {
            // AutoSerial / AMD, and AutoParallel below the METIS size
            // threshold, resolve to AMD: run it straight off the cleaned
            // pattern with the solver-owned workspace.
            const bool amd_backend =
                control.ordering == OrderingMethod::AutoSerial ||
                control.ordering == OrderingMethod::AMD ||
                (control.ordering == OrderingMethod::AutoParallel && n <= 8192);
            if (!(amd_backend && amd_order_lower_csc(n, cleaned.clean.col_ptr,
                                                     cleaned.clean.row_idx, perm,
                                                     analyse_scratch_)))
            {
                auto backend = make_ordering_backend(control.ordering, n, info);

                // Step 3: build symmetric adjacency graph for the cleaned pattern.
                auto [xadj, adjncy] = build_symmetric_adjacency(cleaned.clean);

                // Step 4: compute fill-reducing permutation.
                perm = backend->compute_ordering(n, xadj, adjncy);
            }
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

        // Temporaries live in a solver-owned buffer reused across analyses
        // (fresh allocations are dominated by page faults at these sizes).
        const auto sn_sz = static_cast<std::size_t>(n);
        const CscLower &C = cleaned.clean;
        const std::size_t nnz_c = C.row_idx.size();
        const std::size_t work_need = 7 * sn_sz + 1 + 2 * nnz_c;
        if (analyse_work_.size() < work_need)
            analyse_work_.resize(work_need);
        Int *row_ptr = analyse_work_.data();     // n + 1
        Int *head = row_ptr + sn_sz + 1;         // n
        Int *col_count = head + sn_sz;           // n
        Int *inv_po = col_count + sn_sz;         // n
        Int *parent1 = inv_po + sn_sz;           // n
        Int *colcount1 = parent1 + sn_sz;        // n
        Int *inv_order = colcount1 + sn_sz;      // n
        Int *row_col = inv_order + sn_sz;        // nnz
        Int *row_src = row_col + nnz_c;          // nnz

        // Step 5: elimination tree and exact column counts of the ordering.
        std::vector<Int> iperm(sn_sz);
        for (Int i = 0; i < n; ++i)
            iperm[static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])] = i;
        build_row_form(C, iperm.data(), row_ptr, row_col, nullptr, nullptr, head);
        EliminationTree etree0 = build_elimination_tree_from_rows(n, row_ptr, row_col);
        const std::vector<Int> colcount0 =
            filled_column_counts_from_rows(n, row_ptr, row_col, etree0);

        // Step 6: relabel by the etree postorder (fundamental supernodes
        // become contiguous), then relaxed amalgamation, which returns a
        // further topological reordering making merged groups contiguous.
        for (Int k = 0; k < n; ++k)
            inv_po[etree0.postorder[static_cast<std::size_t>(k)]] = k;
        for (Int k = 0; k < n; ++k)
        {
            const Int old = etree0.postorder[static_cast<std::size_t>(k)];
            const Int par = etree0.parent[static_cast<std::size_t>(old)];
            parent1[k] = (par >= 0) ? inv_po[par] : -1;
            colcount1[k] = colcount0[static_cast<std::size_t>(old)];
        }
        std::vector<Int> order;
        std::vector<Supernode> snodes;
        amalgamate_relaxed(n, parent1, colcount1, static_cast<Int>(control.nemin),
                           order, snodes);

        // Final permutation and its elimination tree (relabelled, not rebuilt).
        for (Int k = 0; k < n; ++k)
            inv_order[order[static_cast<std::size_t>(k)]] = k;
        bool identity_final = true;
        std::vector<Int> parentf(sn_sz);
        {
            const std::vector<Int> perm0 = perm;
            for (Int k = 0; k < n; ++k)
            {
                const Int k1 = order[static_cast<std::size_t>(k)];
                const Int p1 = parent1[k1];
                parentf[static_cast<std::size_t>(k)] = (p1 >= 0) ? inv_order[p1] : -1;
                const Int old = perm0[static_cast<std::size_t>(
                    etree0.postorder[static_cast<std::size_t>(k1)])];
                perm[static_cast<std::size_t>(k)] = old;
                identity_final = identity_final && (old == k);
            }
        }
        for (Int i = 0; i < n; ++i)
            iperm[static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])] = i;
        // Relabelling preserves the tree: numbering 1 *is* the postorder of
        // etree0, so k -> inv_order[k] lists the final labels in postorder.
        EliminationTree etree;
        etree.parent = std::move(parentf);
        etree.postorder.assign(inv_order, inv_order + n);
        etree.max_depth = etree0.max_depth;

        // Step 7: permuted pattern by double transpose. Walking the final
        // row form in increasing row order emits every permuted column
        // already sorted and records where each cleaned entry lands (otp).
        // When the final order is the identity the cleaned matrix is used.
        CscLower A_perm;
        std::vector<Int> otp;
        if (!identity_final)
        {
            build_row_form(C, iperm.data(), row_ptr, row_col, row_src, col_count, head);
            A_perm.n = n;
            A_perm.col_ptr.resize(sn_sz + 1);
            A_perm.col_ptr[0] = 0;
            for (std::size_t c = 0; c < sn_sz; ++c)
                A_perm.col_ptr[c + 1] = A_perm.col_ptr[c] + col_count[c];
            A_perm.row_idx.resize(nnz_c);
            otp.resize(nnz_c);
            std::copy(A_perm.col_ptr.begin(), A_perm.col_ptr.end() - 1, head);
            for (Int r = 0; r < n; ++r)
            {
                for (Int t = row_ptr[r]; t < row_ptr[r + 1]; ++t)
                {
                    const Int slot = head[row_col[t]]++;
                    A_perm.row_idx[static_cast<std::size_t>(slot)] = r;
                    otp[static_cast<std::size_t>(row_src[t])] = slot;
                }
            }
        }

        // Step 8: build assembly tree and fill info predictions
        auto fronts = build_assembly_tree(identity_final ? C : A_perm, snodes, info);

        // Step 8.5: supernodes are numbered in postorder by construction;
        // the factor's LIFO contribution stack wants children in descending
        // postorder.
        const Int nsn = static_cast<Int>(snodes.size());
        std::vector<Int> sn_postorder(static_cast<std::size_t>(nsn));
        std::iota(sn_postorder.begin(), sn_postorder.end(), 0);
        for (Int s = 0; s < nsn; ++s) {
            auto& ch = snodes[static_cast<std::size_t>(s)].children;
            std::reverse(ch.begin(), ch.end());
        }

        // Step 9: record timing
        const auto t1 = std::chrono::steady_clock::now();
        info.analyse_seconds = std::chrono::duration<double>(t1 - t0).count();

        // Step 10: assemble AnalysisKeep
        auto keep = std::make_unique<AnalysisKeep>();
        keep->n = n;
        keep->orig_col_ptr = A.col_ptr;
        keep->orig_row_idx = A.row_idx;
        keep->orig_to_clean_idx = std::move(cleaned.original_to_clean);
        keep->orig_to_perm_direct.resize(keep->orig_to_clean_idx.size(), -1);
        for (std::size_t k = 0; k < keep->orig_to_clean_idx.size(); ++k) {
            const Int c_idx = keep->orig_to_clean_idx[k];
            if (c_idx >= 0) {
                keep->orig_to_perm_direct[k] =
                    identity_final ? c_idx : otp[static_cast<std::size_t>(c_idx)];
            }
        }
        keep->cleaned = std::move(cleaned.clean);
        // Store permuted pattern for fast value-only permutation in factor.
        // For identity ordering we keep these empty and factor reuses cleaned pattern.
        if (!identity_final) {
            keep->perm_col_ptr = std::move(A_perm.col_ptr);
            keep->perm_row_idx = std::move(A_perm.row_idx);
            keep->orig_to_perm_idx = std::move(otp);
        }
        keep->perm = std::move(perm);
        keep->iperm = std::move(iperm);
        keep->etree = std::move(etree);
        keep->supernodes = std::move(snodes);
        keep->fronts = std::move(fronts);
        keep->postorder = sn_postorder;
        keep->solve_postorder = std::move(sn_postorder);

        return keep;
    }

} // namespace smf
