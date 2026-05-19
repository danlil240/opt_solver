#include "smf/solver.hpp"
#include "smf/analysis.hpp"
#include "smf/check_matrix.hpp"
#include "smf/ordering.hpp"
#include "smf/sym_graph.hpp"
#include "smf/etree.hpp"
#include "smf/supernode.hpp"
#include "smf/assembly_tree.hpp"
#include "smf/error.hpp"
#include <chrono>
#include <memory>
#include <algorithm>
#include <numeric>

namespace smf {

// Permute a lower-CSC matrix: given perm[new]=old, build the matrix
// in the new ordering. For lower-CSC: entry (i,j) with i>=j in old ordering
// becomes (iperm[i], iperm[j]) in new ordering; we keep only new_i >= new_j.
static CscLower permute_lower_csc(
    const CscLower& A,
    const std::vector<Int>& perm,    // perm[new]=old
    const std::vector<Int>& iperm)   // iperm[old]=new
{
    const Int n = A.n;
    // Count entries per new column
    std::vector<Int> count(static_cast<std::size_t>(n), 0);
    for (Int old_j = 0; old_j < n; ++old_j) {
        const Int new_j = iperm[static_cast<std::size_t>(old_j)];
        for (Int p = A.col_ptr[static_cast<std::size_t>(old_j)];
             p < A.col_ptr[static_cast<std::size_t>(old_j) + 1]; ++p) {
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

    for (Int old_j = 0; old_j < n; ++old_j) {
        const Int new_j = iperm[static_cast<std::size_t>(old_j)];
        for (Int p = A.col_ptr[static_cast<std::size_t>(old_j)];
             p < A.col_ptr[static_cast<std::size_t>(old_j) + 1]; ++p) {
            const Int old_i = A.row_idx[static_cast<std::size_t>(p)];
            const Int new_i = iperm[static_cast<std::size_t>(old_i)];
            Int col, row;
            if (new_i >= new_j) { col = new_j; row = new_i; }
            else                 { col = new_i; row = new_j; }
            const Int slot = pos[static_cast<std::size_t>(col)]++;
            B.row_idx[static_cast<std::size_t>(slot)] = row;
            B.values[static_cast<std::size_t>(slot)] = A.values[static_cast<std::size_t>(p)];
        }
    }

    // Sort each column's row indices (and values accordingly)
    for (Int j = 0; j < n; ++j) {
        const Int start = B.col_ptr[static_cast<std::size_t>(j)];
        const Int end   = B.col_ptr[static_cast<std::size_t>(j) + 1];
        const Int len   = end - start;
        if (len <= 1) continue;

        std::vector<Int> idx(static_cast<std::size_t>(len));
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](Int a, Int b){
            return B.row_idx[static_cast<std::size_t>(start + a)]
                 < B.row_idx[static_cast<std::size_t>(start + b)];
        });

        std::vector<Int>    sorted_r(static_cast<std::size_t>(len));
        std::vector<double> sorted_v(static_cast<std::size_t>(len));
        for (Int k = 0; k < len; ++k) {
            sorted_r[static_cast<std::size_t>(k)] = B.row_idx[static_cast<std::size_t>(start + idx[static_cast<std::size_t>(k)])];
            sorted_v[static_cast<std::size_t>(k)] = B.values[static_cast<std::size_t>(start + idx[static_cast<std::size_t>(k)])];
        }
        for (Int k = 0; k < len; ++k) {
            B.row_idx[static_cast<std::size_t>(start + k)] = sorted_r[static_cast<std::size_t>(k)];
            B.values[static_cast<std::size_t>(start + k)]  = sorted_v[static_cast<std::size_t>(k)];
        }
    }

    return B;
}

std::unique_ptr<AnalysisKeep> Solver::analyse(
    const CscLower& A,
    const Control& control,
    Info& info)
{
    const auto t0 = std::chrono::steady_clock::now();

    info = Info{};  // zero-initialise
    info.status = ErrorCode::Success;

    // Step 1: clean the input matrix
    auto cleaned = clean_lower_csc(A, info);
    const Int n = cleaned.clean.n;

    if (n == 0) {
        // Trivially empty analysis
        auto keep = std::make_unique<AnalysisKeep>();
        keep->n = 0;
        keep->cleaned = std::move(cleaned.clean);
        const auto t1 = std::chrono::steady_clock::now();
        info.analyse_seconds = std::chrono::duration<double>(t1 - t0).count();
        return keep;
    }

    // Step 2: choose ordering backend
    auto backend = make_ordering_backend(control.ordering, n, info);

    // Step 3: build symmetric adjacency graph for the cleaned pattern
    auto [xadj, adjncy] = build_symmetric_adjacency(cleaned.clean);

    // Step 4: compute fill-reducing permutation
    std::vector<Int> perm = backend->compute_ordering(n, xadj, adjncy);
    if (perm.empty()) {
        // Ordering failed — fall back to identity permutation
        info.status = ErrorCode::Success;
        perm.resize(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), 0);
    }

    // Build inverse permutation
    std::vector<Int> iperm(static_cast<std::size_t>(n));
    for (Int i = 0; i < n; ++i)
        iperm[static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])] = i;

    // Step 5: permute the cleaned matrix
    CscLower A_perm = permute_lower_csc(cleaned.clean, perm, iperm);

    // Step 6: build elimination tree
    EliminationTree etree = build_elimination_tree(n, A_perm.col_ptr, A_perm.row_idx);

    // Step 7: detect and amalgamate supernodes
    auto snodes = detect_fundamental_supernodes(n, A_perm.col_ptr, A_perm.row_idx, etree);
    snodes = amalgamate_supernodes(std::move(snodes), static_cast<Int>(control.nemin));

    // Step 8: build assembly tree and fill info predictions
    auto fronts = build_assembly_tree(A_perm, snodes, info);

    // Step 9: record timing
    const auto t1 = std::chrono::steady_clock::now();
    info.analyse_seconds = std::chrono::duration<double>(t1 - t0).count();

    // Step 10: assemble AnalysisKeep
    auto keep = std::make_unique<AnalysisKeep>();
    keep->n          = n;
    keep->cleaned    = std::move(cleaned.clean);
    keep->perm       = std::move(perm);
    keep->iperm      = std::move(iperm);
    keep->etree      = std::move(etree);
    keep->supernodes = std::move(snodes);
    keep->fronts     = std::move(fronts);

    return keep;
}

} // namespace smf
