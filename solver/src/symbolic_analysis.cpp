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
#include <cmath>
#include <memory>
#include <numeric>

namespace smf
{

#if defined(SMF_HAS_METIS) && SMF_HAS_METIS
    static constexpr bool kHaveMetis = true;
#else
    static constexpr bool kHaveMetis = false;
#endif

    // Matching ordering: a matched pair is kept together when either scaled
    // diagonal is below this (on the GHS_indef KKT set every pair has a zero
    // diagonal member, so the value barely matters).
    static constexpr double kMatchingPairTol = 0.1;

    // Auto policy: keep the matching ordering while its predicted flops are
    // within this factor of AMD's. Measured on GHS_indef and optim20: it wins
    // at ratios <= ~1.15 (bratu3d, boyd1/2, cont-201/300, kkt_eq_qp), where
    // AMD's factor delays pivots by the thousand, is neutral to slower from
    // ~2.2 (kkt_network_flow, ncvxqp9) and loses from ~3.9 (mario002,
    // aug3dcqp, stokes128).
    static constexpr double kMatchingFlopsRatio = 1.5;

    // KKT recognition for the auto policy: a diagonal below kSmallDiagonal
    // times its column's largest off-diagonal cannot be a 1x1 pivot; the
    // matrix is treated as a saddle point when such columns are at least
    // kKktSmallDiagonalShare of n (optim20/GHS KKT: 10-87%; GHS indefinite
    // PDE matrices: 0.3-0.9%) or any diagonal is exactly zero.
    static constexpr double kSmallDiagonal = 0.01;
    static constexpr double kKktSmallDiagonalShare = 0.05;
    // AutoParallel tries METIS when AMD's predicted factor flops (times
    // Control::expected_refactors) reach this many per row. One-off SuiteSparse
    // SPD factorizations: METIS won at 8.8e5-1.1e6 (bmwcra_1, cfd2), tied at
    // 2.1e5 (pwtk) and lost at 1.3e5-4.0e5 (af_shell7, G3_circuit, apache2).
    static constexpr double kMetisFlopsPerRow = 4.5e5;

    // Below this order delays stay cheap (fronts are bounded by n) and the
    // matching's fixed cost would dominate (optim20: n <= 4000).
    static constexpr Int kAutoMatchingMinN = 10000;

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
        // Phase timing: lap(slot) charges the time since the previous lap.
        AnalysePhaseSeconds phases;
        auto t_lap = t0;
        const auto lap = [&](double &slot) {
            const auto now = std::chrono::steady_clock::now();
            slot += std::chrono::duration<double>(now - t_lap).count();
            t_lap = now;
        };

        info = Info{}; // zero-initialise
        info.status = ErrorCode::Success;

        // Step 1: clean the input matrix
        auto cleaned = clean_lower_csc(A, info);
        const Int n = cleaned.clean.n;

        // UserSupplied: user_perm must be a permutation of 0..n-1 and each
        // pair's two columns adjacent in it. user_iperm[old] = position.
        const bool user_order = control.ordering == OrderingMethod::UserSupplied;
        std::vector<Int> user_iperm;
        if (user_order)
        {
            const Int np = control.user_num_pairs;
            const Int *pf = control.user_pair_first;
            bool ok = control.user_perm != nullptr && np >= 0 && (np == 0 || pf != nullptr);
            if (ok)
            {
                user_iperm.assign(static_cast<std::size_t>(n), -1);
                for (Int k = 0; ok && k < n; ++k)
                {
                    const Int old = control.user_perm[k];
                    ok = old >= 0 && old < n && user_iperm[static_cast<std::size_t>(old)] < 0;
                    if (ok)
                        user_iperm[static_cast<std::size_t>(old)] = k;
                }
            }
            for (Int k = 0; ok && k < np; ++k)
            {
                const Int a = pf[2 * static_cast<std::size_t>(k)];
                const Int b = pf[2 * static_cast<std::size_t>(k) + 1];
                ok = a >= 0 && a < n && b >= 0 && b < n &&
                     std::abs(user_iperm[static_cast<std::size_t>(a)] -
                              user_iperm[static_cast<std::size_t>(b)]) == 1;
            }
            if (!ok)
            {
                info.status = ErrorCode::IllegalValue;
                return nullptr;
            }
        }

        if (n == 0)
        {
            // Trivially empty analysis
            auto keep = std::make_unique<AnalysisKeep>();
            keep->n = 0;
            keep->child_ptr.assign(1, 0);
            keep->front_row_ptr.assign(1, 0);
            keep->orig_col_ptr = A.col_ptr;
            keep->orig_row_idx = A.row_idx;
            keep->cleaned = std::move(cleaned.clean);
            lap(phases.check);
            const auto t1 = t_lap;
            info.analyse_seconds = std::chrono::duration<double>(t1 - t0).count();
            info.analyse_phases = phases;
            return keep;
        }

        lap(phases.check);

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
        bool amd_ordered = false;
        MatchingOrderStats matching_stats;

        // Saddle-point / KKT matrices in auto mode get the maximum-product
        // matching: its scaling (applied by ScalingMethod::Auto) rescues badly
        // scaled systems, and its compressed ordering replaces AMD unless it
        // costs more flops (decided in step 5). They are recognised by
        // diagonals that cannot carry a 1x1 pivot: any exact zero, or many
        // (regularised constraint blocks) below 1% of their column's largest
        // off-diagonal. Indefinite PDE matrices have under 1% such columns.
        const CscLower &C = cleaned.clean;

        // Temporaries live in a solver-owned buffer reused across analyses
        // (fresh allocations are dominated by page faults at these sizes).
        // The symmetric adjacency of C is built once: AMD consumes a copy and
        // the etree and column counts of every candidate ordering read it
        // directly, so no permuted copy of A is formed for them.
        const auto sn_sz = static_cast<std::size_t>(n);
        const std::size_t nnz_c = C.row_idx.size();
        const std::size_t nz_adj = lower_csc_adjacency_size(n, C.col_ptr.data(), C.row_idx.data());
        const std::size_t work_need = 19 * sn_sz + 3 + nz_adj + 2 * nnz_c;
        if (analyse_work_.size() < work_need)
            analyse_work_.resize(work_need);
        Int *xadj = analyse_work_.data();        // n + 1
        Int *adj = xadj + sn_sz + 1;             // nz_adj
        Int *iperm_w = adj + nz_adj;             // n
        Int *sym_a = iperm_w + sn_sz;            // 3n: parent, postorder, counts
        Int *sym_b = sym_a + 3 * sn_sz;          // 3n: a candidate ordering's
        Int *sym_work = sym_b + 3 * sn_sz;       // 4n + 1
        Int *row_ptr = sym_work + 4 * sn_sz + 1; // n + 1
        Int *head = row_ptr + sn_sz + 1;         // n
        Int *col_count = head + sn_sz;           // n
        Int *inv_po = col_count + sn_sz;         // n
        Int *parent1 = inv_po + sn_sz;           // n
        Int *colcount1 = parent1 + sn_sz;        // n
        Int *inv_order = colcount1 + sn_sz;      // n
        Int *row_col = inv_order + sn_sz;        // nnz
        Int *row_src = row_col + nnz_c;          // nnz
        lower_csc_adjacency(n, C.col_ptr.data(), C.row_idx.data(), xadj, adj);

        bool auto_matching = false;
        lap(phases.ordering);
        if (auto_order && !use_identity_ordering && n >= kAutoMatchingMinN &&
            control.matrix_type == MatrixType::RealSymmetricIndefinite &&
            C.values.size() == C.row_idx.size())
        {
            std::vector<double> diag(static_cast<std::size_t>(n), 0.0), off_max(static_cast<std::size_t>(n), 0.0);
            for (Int j = 0; j < n; ++j)
            {
                for (Int p = C.col_ptr[static_cast<std::size_t>(j)]; p < C.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
                {
                    const Int i = C.row_idx[static_cast<std::size_t>(p)];
                    const double a = std::abs(C.values[static_cast<std::size_t>(p)]);
                    if (i == j)
                    {
                        diag[static_cast<std::size_t>(j)] = a;
                        continue;
                    }
                    off_max[static_cast<std::size_t>(i)] = std::max(off_max[static_cast<std::size_t>(i)], a);
                    off_max[static_cast<std::size_t>(j)] = std::max(off_max[static_cast<std::size_t>(j)], a);
                }
            }
            Int small = 0;
            bool zero = false;
            for (std::size_t j = 0; j < diag.size(); ++j)
            {
                zero = zero || diag[j] == 0.0;
                small += (diag[j] < kSmallDiagonal * off_max[j]) ? 1 : 0;
            }
            auto_matching = zero || static_cast<double>(small) >= kKktSmallDiagonalShare * static_cast<double>(n);
        }
        std::vector<Int> matching_perm;
        if (auto_matching)
            auto_matching = matching_order_lower_csc(C, kMatchingPairTol, matching_perm,
                                                     analyse_scratch_, &matching_stats);
        lap(phases.matching);

        if (user_order)
        {
            // Used as given (validated above). Not an auto ordering, so the
            // identity-band test, auto matching (and its scaling) and the
            // METIS trial below are all skipped.
            perm.assign(control.user_perm, control.user_perm + n);
            info.ordering_used = static_cast<int>(OrderingMethod::UserSupplied);
        }
        else if (use_identity_ordering)
        {
            std::iota(perm.begin(), perm.end(), 0);
            info.ordering_used = static_cast<int>(OrderingMethod::UserSupplied);
        }
        else
        {
            // AutoSerial / AMD resolve to AMD: run it straight off the
            // cleaned pattern with the solver-owned workspace. AutoParallel
            // starts from AMD too; step 5 tries METIS only when AMD's factor
            // looks expensive.
            // Matching falls back to plain AMD when it cannot run (no values,
            // structurally singular).
            lap(phases.ordering);
            const bool matching_ordered =
                control.ordering == OrderingMethod::Matching &&
                matching_order_lower_csc(C, kMatchingPairTol, perm,
                                         analyse_scratch_, &matching_stats);
            lap(phases.matching);
            const bool amd_backend =
                control.ordering == OrderingMethod::AutoSerial ||
                control.ordering == OrderingMethod::AMD ||
                control.ordering == OrderingMethod::AutoParallel ||
                control.ordering == OrderingMethod::Matching;
            if (!matching_ordered && amd_backend)
            {
                const std::size_t amd_need = amd_adjacency_work_size(n, nz_adj);
                if (analyse_scratch_.size() < amd_need)
                    analyse_scratch_.resize(amd_need);
                amd_ordered = amd_order_adjacency(n, xadj, adj, perm.data(),
                                                  analyse_scratch_.data());
            }
            if (!matching_ordered && !amd_ordered)
            {
                auto backend = make_ordering_backend(control.ordering, n, info);

                // Step 3: build symmetric adjacency graph for the cleaned pattern.
                auto [xadj_g, adjncy_g] = build_symmetric_adjacency(C);

                // Step 4: compute fill-reducing permutation.
                perm = backend->compute_ordering(n, xadj_g, adjncy_g);
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
                OrderingMethod used = control.ordering;
                if (matching_ordered)
                    used = OrderingMethod::Matching;
                else if (amd_ordered || control.ordering == OrderingMethod::AutoSerial)
                    used = OrderingMethod::AMD;
                else if (control.ordering == OrderingMethod::AutoParallel)
                    used = (kHaveMetis && n > 8192) ? OrderingMethod::METIS : OrderingMethod::AMD;
                info.ordering_used = static_cast<int>(used);
            }
        }

        lap(phases.ordering);

        std::vector<Int> iperm(sn_sz);
        // Etree (parent, postorder) and exact column counts of an ordering,
        // into one of the 3n blocks sym_a / sym_b.
        const auto symbolic_counts = [&](const Int *p, Int *sym) {
            for (Int i = 0; i < n; ++i)
                iperm_w[p[i]] = i;
            etree_from_adjacency(n, xadj, adj, p, iperm_w, sym, sym_work);
            const Int depth = etree_postorder(n, sym, sym + sn_sz, sym_work);
            filled_column_counts_from_adjacency(n, xadj, adj, p, iperm_w, sym, sym + sn_sz,
                                                sym + 2 * sn_sz, sym_work);
            return depth;
        };
        const auto factor_flops = [&](const Int *sym) {
            double fl = 0.0;
            for (Int j = 0; j < n; ++j)
            {
                const double c = static_cast<double>(sym[2 * sn_sz + static_cast<std::size_t>(j)]);
                fl += c * c;
            }
            return fl;
        };

        std::vector<Int> order;
        std::vector<Supernode> snodes;
        std::vector<Int> parentf(sn_sz);
        // Step 5: elimination tree and exact column counts of the ordering.
        Int *sym0 = sym_a;
        Int *sym1 = sym_b;
        Int depth0 = symbolic_counts(perm.data(), sym0);
        lap(phases.etree_counts);

        // AutoParallel on large matrices: METIS costs several us per vertex
        // (up to ~85% of analysis) and only pays off when AMD's factor is
        // expensive per row; then it is tried and the cheaper one kept.
        if (kHaveMetis && amd_ordered &&
            control.ordering == OrderingMethod::AutoParallel && n > 8192)
        {
            const double fl = factor_flops(sym0);
            const double refactors = std::max(1, control.expected_refactors);
            if (fl * refactors >= kMetisFlopsPerRow * static_cast<double>(n))
            {
                auto backend = make_ordering_backend(OrderingMethod::METIS, n, info);
                auto [xadj_m, adjncy_m] = build_symmetric_adjacency(C);
                std::vector<Int> mperm = backend->compute_ordering(n, xadj_m, adjncy_m);
                if (mperm.size() == sn_sz)
                {
                    const Int depth_m = symbolic_counts(mperm.data(), sym1);
                    if (factor_flops(sym1) < fl)
                    {
                        perm = std::move(mperm);
                        std::swap(sym0, sym1);
                        depth0 = depth_m;
                        info.ordering_used = static_cast<int>(OrderingMethod::METIS);
                    }
                }
            }
        }

        lap(phases.ordering);

        // Auto matching: the compressed ordering keeps 2x2 pivot pairs
        // together, which AMD's lower predicted fill ignores (AMD's factor
        // then delays pivots by the thousand). Prefer it unless its exact
        // flops exceed AMD's by more than a constant factor.
        if (auto_matching && info.ordering_used != static_cast<int>(OrderingMethod::Matching))
        {
            const Int depth_m = symbolic_counts(matching_perm.data(), sym1);
            if (factor_flops(sym1) <= kMatchingFlopsRatio * factor_flops(sym0))
            {
                perm = std::move(matching_perm);
                std::swap(sym0, sym1);
                depth0 = depth_m;
                info.ordering_used = static_cast<int>(OrderingMethod::Matching);
            }
        }
        const Int *parent0 = sym0;
        const Int *post0 = sym0 + sn_sz;
        const Int *colcount0 = sym0 + 2 * sn_sz;

        lap(phases.matching);

        // Step 6: relabel by the etree postorder (fundamental supernodes
        // become contiguous), then relaxed amalgamation, which returns a
        // further topological reordering making merged groups contiguous.
        for (Int k = 0; k < n; ++k)
            inv_po[post0[k]] = k;
        for (Int k = 0; k < n; ++k)
        {
            const Int old = post0[k];
            const Int par = parent0[old];
            parent1[k] = (par >= 0) ? inv_po[par] : -1;
            colcount1[k] = colcount0[old];
        }
        // User 2x2 pairs: perm is still user_perm here, so a pair's
        // positions in the permuted numbering are user_iperm; the earlier
        // one is flagged (in the postordered numbering, via inv_po) when the
        // later one is its etree parent, i.e. L(later, earlier) != 0.
        std::vector<std::uint8_t> keep_with_parent;
        if (user_order && control.user_num_pairs > 0)
        {
            keep_with_parent.assign(sn_sz, 0);
            const Int *pf = control.user_pair_first;
            for (Int k = 0; k < control.user_num_pairs; ++k)
            {
                Int pa = user_iperm[static_cast<std::size_t>(pf[2 * static_cast<std::size_t>(k)])];
                Int pb = user_iperm[static_cast<std::size_t>(pf[2 * static_cast<std::size_t>(k) + 1])];
                if (pa > pb)
                    std::swap(pa, pb);
                if (parent0[pa] == pb)
                    keep_with_parent[static_cast<std::size_t>(inv_po[pa])] = 1;
                else
                    ++info.pairs_split;
            }
        }
        amalgamate_relaxed(n, parent1, colcount1, static_cast<Int>(control.nemin),
                           order, snodes,
                           keep_with_parent.empty() ? nullptr : keep_with_parent.data());

        // Final permutation and its elimination tree (relabelled, not rebuilt).
        for (Int k = 0; k < n; ++k)
            inv_order[order[static_cast<std::size_t>(k)]] = k;
        bool identity_final = true;
        {
            const std::vector<Int> perm0 = perm;
            for (Int k = 0; k < n; ++k)
            {
                const Int k1 = order[static_cast<std::size_t>(k)];
                const Int p1 = parent1[k1];
                parentf[static_cast<std::size_t>(k)] = (p1 >= 0) ? inv_order[p1] : -1;
                const Int old = perm0[static_cast<std::size_t>(post0[k1])];
                perm[static_cast<std::size_t>(k)] = old;
                identity_final = identity_final && (old == k);
            }
        }
        for (Int i = 0; i < n; ++i)
            iperm[static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])] = i;

        // Relabelling preserves the tree: numbering 1 *is* the postorder of
        // the ordering's etree, so k -> inv_order[k] lists the final labels in postorder.
        EliminationTree etree;
        etree.parent = std::move(parentf);
        etree.postorder.assign(inv_order, inv_order + n);
        etree.max_depth = depth0;

        lap(phases.amalgamate);

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

        // Step 8: supernode tree and front rows in flat CSR form. Children
        // come out in descending postorder (the factor's LIFO contribution
        // stack consumes them that way). A front holds its columns plus the
        // rows below its last column, so the column counts give the exact
        // total and front_row_idx is allocated once.
        const Int nsn = static_cast<Int>(snodes.size());
        std::vector<Int> child_ptr, child_idx, front_row_ptr, front_row_idx;
        build_children_csr(snodes, child_ptr, child_idx);
        std::size_t rows_total = 0;
        for (const Supernode &sn : snodes)
            rows_total += static_cast<std::size_t>(sn.width()) +
                          static_cast<std::size_t>(
                              colcount1[order[static_cast<std::size_t>(sn.col_end) - 1]]) - 1;
        build_assembly_tree(identity_final ? C : A_perm, snodes, child_ptr, child_idx,
                            front_row_ptr, front_row_idx, info, rows_total);

        std::vector<Int> sn_postorder(static_cast<std::size_t>(nsn));
        std::iota(sn_postorder.begin(), sn_postorder.end(), 0);

        lap(phases.assembly_tree);

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
        keep->child_ptr = std::move(child_ptr);
        keep->child_idx = std::move(child_idx);
        keep->front_row_ptr = std::move(front_row_ptr);
        keep->front_row_idx = std::move(front_row_idx);
        keep->postorder = sn_postorder;
        keep->solve_postorder = std::move(sn_postorder);
        // The matching's scaling only pays off with the matching ordering
        // (its 2x2 pairs kept adjacent); with AMD or METIS kept it makes the
        // factor less accurate, so ScalingMethod::Auto then applies none.
        if (info.ordering_used == static_cast<int>(OrderingMethod::Matching))
            keep->matching_scale = std::move(matching_stats.scale);

        // Step 9: record timing (after the keep is assembled, so the phases
        // partition analyse_seconds).
        lap(phases.keep_build);
        info.analyse_seconds = std::chrono::duration<double>(t_lap - t0).count();
        info.analyse_phases = phases;

        return keep;
    }

} // namespace smf
