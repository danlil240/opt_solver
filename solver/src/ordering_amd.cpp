#include "smf/ordering.hpp"
#include <suitesparse/amd.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace smf {

// Build symmetric CSR (no self-loops) from lower-CSC.
// For each off-diagonal entry (i,j) with i>j in the lower CSC:
//   add edge j->i and edge i->j to the symmetric graph.
// Sort each row's adjacency.
std::pair<std::vector<Int>, std::vector<Int>>
build_symmetric_csr(Int n,
                    const std::vector<Int>& col_ptr,
                    const std::vector<Int>& row_idx)
{
    // Count degree of each node (excluding self-loops)
    std::vector<Int> deg(static_cast<std::size_t>(n), 0);
    for (Int j = 0; j < n; ++j) {
        for (Int k = col_ptr[static_cast<std::size_t>(j)];
             k < col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            Int i = row_idx[static_cast<std::size_t>(k)];
            if (i != j) {
                ++deg[static_cast<std::size_t>(j)];
                ++deg[static_cast<std::size_t>(i)];
            }
        }
    }
    // Build xadj
    std::vector<Int> xadj(static_cast<std::size_t>(n) + 1, 0);
    for (Int i = 0; i < n; ++i)
        xadj[static_cast<std::size_t>(i) + 1] =
            xadj[static_cast<std::size_t>(i)] + deg[static_cast<std::size_t>(i)];
    // Fill adjncy
    std::vector<Int> adjncy(static_cast<std::size_t>(xadj[static_cast<std::size_t>(n)]));
    // current fill position per row
    std::vector<Int> pos(xadj.begin(), xadj.begin() + n);
    for (Int j = 0; j < n; ++j) {
        for (Int k = col_ptr[static_cast<std::size_t>(j)];
             k < col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            Int i = row_idx[static_cast<std::size_t>(k)];
            if (i != j) {
                adjncy[static_cast<std::size_t>(pos[static_cast<std::size_t>(j)]++)] = i;
                adjncy[static_cast<std::size_t>(pos[static_cast<std::size_t>(i)]++)] = j;
            }
        }
    }
    // Sort each row's adjacency
    for (Int i = 0; i < n; ++i)
        std::sort(adjncy.begin() + xadj[static_cast<std::size_t>(i)],
                  adjncy.begin() + xadj[static_cast<std::size_t>(i) + 1]);
    return {xadj, adjncy};
}

std::vector<Int> AmdOrdering::compute_ordering(
    Int n,
    const std::vector<Int>& xadj,
    const std::vector<Int>& adjncy)
{
    std::vector<Int> perm(static_cast<std::size_t>(n));
    // Handle empty adjacency (diagonal-only matrix): AMD returns AMD_INVALID when
    // adjncy.data() is nullptr (empty vector). Return identity permutation instead.
    if (adjncy.empty()) {
        std::iota(perm.begin(), perm.end(), 0);
        return perm;
    }
    // amd_order takes (n, Ap, Ai, P, Control, Info)
    // Ap and Ai are const int* — our Int is int32_t = int, so direct cast is fine.
    int result = amd_order(
        static_cast<int>(n),
        xadj.data(),
        adjncy.data(),
        perm.data(),
        nullptr,   // use default AMD control
        nullptr    // don't capture AMD info
    );
    if (result != AMD_OK && result != AMD_OK_BUT_JUMBLED) {
        throw std::runtime_error("amd_order failed with code " + std::to_string(result));
    }
    return perm;
}

} // namespace smf
