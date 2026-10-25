#include "smf/ordering.hpp"
#include "smf/scaling_matching.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace smf {

// Matching-based compressed ordering for symmetric indefinite (KKT) matrices
// (Duff & Pralet 2005; the idea behind MA97 orderings 7/8 and MUMPS
// ICNTL(12) = 2). A fill-reducing ordering of the plain graph happily
// separates a zero-diagonal constraint row from the variable it must be
// pivoted with, so the factorization delays it, front after front. Here
// the cycles of the maximum-product matching are split into 2x2 pivot
// candidates; each pair whose diagonal cannot carry a 1x1 pivot becomes one
// node of a compressed graph, AMD orders that graph, and the expanded order
// keeps both members of a pair adjacent.
bool matching_order_lower_csc(const CscLower &A, double pair_tol, std::vector<Int> &perm,
                              std::vector<Int> &amd_work, MatchingOrderStats *stats) {
    const Int n = A.n;
    const auto sn = static_cast<std::size_t>(n);
    if (n == 0 || A.values.size() != A.row_idx.size()) return false;
    const MatchingResult mr = compute_symmetric_matching(A);
    if (mr.status != ErrorCode::Success) return false;
    const auto &sigma = mr.col_to_row;  // column j is matched to row sigma[j]
    const auto &s = mr.scale;

    // Scaled |diagonal| and scaled |matched entry| of every column.
    std::vector<double> diag(sn, 0.0), matched(sn, 0.0);
    for (Int j = 0; j < n; ++j) {
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int i = A.row_idx[static_cast<std::size_t>(p)];
            const double v = std::abs(A.values[static_cast<std::size_t>(p)]) *
                             s[static_cast<std::size_t>(i)] * s[static_cast<std::size_t>(j)];
            if (i == j) diag[static_cast<std::size_t>(j)] = v;
            if (sigma[static_cast<std::size_t>(j)] == i) matched[static_cast<std::size_t>(j)] = v;
            if (i != j && sigma[static_cast<std::size_t>(i)] == j) matched[static_cast<std::size_t>(i)] = v;
        }
    }

    // Split every matching cycle c0 -> c1 = sigma(c0) -> ... into adjacent
    // pairs (c_t, c_t+1), whose coupling a(c_t+1, c_t) is a matched entry.
    // Even cycles take the alternation with the larger product of couplings;
    // odd cycles leave out the member with the best diagonal.
    std::vector<Int> partner(sn, -1);
    std::vector<std::uint8_t> seen(sn, 0);
    std::vector<Int> cyc;
    const auto log_coupling = [&](Int c) {
        return std::log(std::max(matched[static_cast<std::size_t>(c)], 1e-300));
    };
    for (Int start = 0; start < n; ++start) {
        if (seen[static_cast<std::size_t>(start)]) continue;
        cyc.clear();
        for (Int c = start; !seen[static_cast<std::size_t>(c)]; c = sigma[static_cast<std::size_t>(c)]) {
            seen[static_cast<std::size_t>(c)] = 1;
            cyc.push_back(c);
        }
        const std::size_t k = cyc.size();
        if (k < 2) continue;
        std::size_t first = 0;  // pairs start at cyc[first]
        std::size_t npairs = k / 2;
        if (k % 2 == 0) {
            double w0 = 0.0, w1 = 0.0;
            for (std::size_t t = 0; t < k; ++t) (t % 2 == 0 ? w0 : w1) += log_coupling(cyc[t]);
            first = (w1 > w0) ? 1 : 0;
        } else {
            std::size_t single = 0;
            for (std::size_t t = 1; t < k; ++t)
                if (diag[static_cast<std::size_t>(cyc[t])] > diag[static_cast<std::size_t>(cyc[single])]) single = t;
            first = single + 1;
        }
        for (std::size_t q = 0; q < npairs; ++q) {
            const Int a = cyc[(first + 2 * q) % k];
            const Int b = cyc[(first + 2 * q + 1) % k];
            // Pair only when one diagonal is too small for a 1x1 pivot.
            if (std::min(diag[static_cast<std::size_t>(a)], diag[static_cast<std::size_t>(b)]) < pair_tol) {
                partner[static_cast<std::size_t>(a)] = b;
                partner[static_cast<std::size_t>(b)] = a;
            }
        }
    }

    // Compressed graph: one node per pair or unpaired variable.
    std::vector<Int> node_of(sn), first_of;
    first_of.reserve(sn);
    Int nnodes = 0, npaired = 0;
    for (Int i = 0; i < n; ++i) {
        const Int pi = partner[static_cast<std::size_t>(i)];
        if (pi >= 0 && pi < i) {
            node_of[static_cast<std::size_t>(i)] = node_of[static_cast<std::size_t>(pi)];
            ++npaired;
            continue;
        }
        node_of[static_cast<std::size_t>(i)] = nnodes++;
        first_of.push_back(i);
    }
    // Each compressed edge once (row > col), duplicates removed per column.
    std::vector<Int> cnt(static_cast<std::size_t>(nnodes) + 1, 0);
    for (Int j = 0; j < n; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int I = node_of[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)])];
            const Int J = node_of[static_cast<std::size_t>(j)];
            if (I != J) ++cnt[static_cast<std::size_t>(std::min(I, J)) + 1];
        }
    for (Int J = 0; J < nnodes; ++J) cnt[static_cast<std::size_t>(J) + 1] += cnt[static_cast<std::size_t>(J)];
    std::vector<Int> cptr(cnt), ridx(static_cast<std::size_t>(cnt[static_cast<std::size_t>(nnodes)]));
    for (Int j = 0; j < n; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int I = node_of[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)])];
            const Int J = node_of[static_cast<std::size_t>(j)];
            if (I != J) ridx[static_cast<std::size_t>(cnt[static_cast<std::size_t>(std::min(I, J))]++)] = std::max(I, J);
        }
    std::vector<Int> mark(static_cast<std::size_t>(nnodes), -1);
    Int out = 0;
    for (Int J = 0; J < nnodes; ++J) {
        const Int begin = cptr[static_cast<std::size_t>(J)], end = cptr[static_cast<std::size_t>(J) + 1];
        cptr[static_cast<std::size_t>(J)] = out;
        for (Int p = begin; p < end; ++p) {
            const Int I = ridx[static_cast<std::size_t>(p)];
            if (mark[static_cast<std::size_t>(I)] == J) continue;
            mark[static_cast<std::size_t>(I)] = J;
            ridx[static_cast<std::size_t>(out++)] = I;
        }
    }
    cptr[static_cast<std::size_t>(nnodes)] = out;
    ridx.resize(static_cast<std::size_t>(out));

    std::vector<Int> node_perm;
    if (!amd_order_lower_csc(nnodes, cptr, ridx, node_perm, amd_work)) return false;

    perm.clear();
    perm.reserve(sn);
    for (Int J : node_perm) {
        const Int a = first_of[static_cast<std::size_t>(J)];
        perm.push_back(a);
        const Int b = partner[static_cast<std::size_t>(a)];
        if (b >= 0) perm.push_back(b);
    }
    if (stats) {
        stats->pairs = npaired;
        stats->nodes = nnodes;
        stats->scale = mr.scale;
    }
    return true;
}

} // namespace smf
