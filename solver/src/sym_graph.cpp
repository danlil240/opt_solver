#include "smf/sym_graph.hpp"
#include <algorithm>

namespace smf {

std::pair<std::vector<Int>, std::vector<Int>>
build_symmetric_adjacency(const CscLower& A)
{
    const Int n = A.n;
    std::vector<Int> deg(n, 0);

    for (Int j = 0; j < n; ++j) {
        for (Int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k) {
            const Int i = A.row_idx[k];
            if (i != j) {
                ++deg[j];
                ++deg[i];
            }
        }
    }

    std::vector<Int> xadj(n + 1, 0);
    for (Int v = 0; v < n; ++v) {
        xadj[v + 1] = xadj[v] + deg[v];
    }

    const Int total_edges = xadj[n];
    std::vector<Int> adjncy(static_cast<std::size_t>(total_edges));
    std::vector<Int> cursor(xadj.begin(), xadj.begin() + n);

    for (Int j = 0; j < n; ++j) {
        for (Int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k) {
            const Int i = A.row_idx[k];
            if (i != j) {
                adjncy[static_cast<std::size_t>(cursor[j]++)] = i;
                adjncy[static_cast<std::size_t>(cursor[i]++)] = j;
            }
        }
    }

    for (Int v = 0; v < n; ++v) {
        std::sort(adjncy.begin() + xadj[v], adjncy.begin() + xadj[v + 1]);
    }

    return {std::move(xadj), std::move(adjncy)};
}

// This greedy upper bound equals the true structural rank when the matrix is
// permuted to have a zero-free diagonal, but may overestimate when rows must
// be shared.
Int structural_rank(const CscLower& A)
{
    std::vector<bool> row_used(static_cast<std::size_t>(A.n), false);
    Int matched = 0;

    for (Int j = 0; j < A.n; ++j) {
        for (Int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k) {
            const Int i = A.row_idx[k];
            if (!row_used[static_cast<std::size_t>(i)]) {
                row_used[static_cast<std::size_t>(i)] = true;
                ++matched;
                break;
            }
        }
    }

    return matched;
}

} // namespace smf
