#include "smf/etree.hpp"
#include <vector>
#include <stack>
#include <algorithm>

namespace smf {

EliminationTree build_elimination_tree(
    Int n,
    const std::vector<Int>& col_ptr,
    const std::vector<Int>& row_idx)
{
    EliminationTree et;
    if (n == 0) return et;

    // Phase 1: parent array via Liu's path-compressed algorithm.
    //
    // Input is lower-triangle CSC (col j contains rows i > j).
    // We first transpose to upper-triangle CSC (col i contains rows j < i),
    // then run the standard upper-triangle etree algorithm.
    //
    // Transposition: lower entry (col=j, row=i, i>j) -> upper entry (col=i, row=j).

    // Count entries per upper-triangle column
    const auto sn = static_cast<std::size_t>(n);
    std::vector<Int> ucol(sn + 1, 0);
    for (Int j = 0; j < n; ++j)
        for (Int p = col_ptr[static_cast<std::size_t>(j)];
             p < col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            Int i_ = row_idx[static_cast<std::size_t>(p)];
            if (i_ == j) continue;  // skip diagonal (not an etree edge)
            ucol[static_cast<std::size_t>(i_) + 1]++;
        }

    // Prefix sum -> ucol[k] = start of column k in upper CSC
    for (Int k = 0; k < n; ++k)
        ucol[static_cast<std::size_t>(k) + 1] += ucol[static_cast<std::size_t>(k)];

    // Fill upper row indices
    std::vector<Int> urow(static_cast<std::size_t>(ucol[sn]));
    {
        std::vector<Int> pos(ucol.begin(), ucol.begin() + static_cast<std::ptrdiff_t>(n));
        for (Int j = 0; j < n; ++j)
            for (Int p = col_ptr[static_cast<std::size_t>(j)];
                 p < col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
                Int i = row_idx[static_cast<std::size_t>(p)];
                if (i == j) continue;  // skip diagonal
                urow[static_cast<std::size_t>(pos[static_cast<std::size_t>(i)]++)] = j;
            }
    }

    // Liu's upper-triangle etree algorithm:
    // For each column k, for each row r in upper CSC column k (r < k):
    //   walk r upward via ancestor[], path-compressing to k,
    //   until ancestor[r]==-1 (unrooted) or ancestor[r]==k (already linked).
    //   On unrooted: set parent[r]=k and ancestor[r]=k.
    std::vector<Int> parent(sn, -1);
    std::vector<Int> ancestor(sn, -1);
    for (Int k = 0; k < n; ++k) {
        for (Int p = ucol[static_cast<std::size_t>(k)];
             p < ucol[static_cast<std::size_t>(k) + 1]; ++p) {
            Int r = urow[static_cast<std::size_t>(p)]; // r < k
            while (ancestor[static_cast<std::size_t>(r)] != -1 &&
                   ancestor[static_cast<std::size_t>(r)] != k) {
                Int t = ancestor[static_cast<std::size_t>(r)];
                ancestor[static_cast<std::size_t>(r)] = k;
                r = t;
            }
            if (ancestor[static_cast<std::size_t>(r)] == -1) {
                parent[static_cast<std::size_t>(r)]   = k;
                ancestor[static_cast<std::size_t>(r)] = k;
            }
        }
    }

    // Phase 2: build children lists (flat CSR) for postorder traversal
    std::vector<Int> ch_start(sn + 1, 0);
    for (Int j = 0; j < n; ++j)
        if (parent[static_cast<std::size_t>(j)] >= 0)
            ++ch_start[static_cast<std::size_t>(parent[static_cast<std::size_t>(j)]) + 1];
    for (Int k = 0; k < n; ++k)
        ch_start[static_cast<std::size_t>(k) + 1] += ch_start[static_cast<std::size_t>(k)];
    std::vector<Int> children(static_cast<std::size_t>(ch_start[sn]));
    {
        std::vector<Int> pos(ch_start.begin(), ch_start.begin() + static_cast<std::ptrdiff_t>(n));
        for (Int j = 0; j < n; ++j)
            if (parent[static_cast<std::size_t>(j)] >= 0)
                children[static_cast<std::size_t>(pos[static_cast<std::size_t>(parent[static_cast<std::size_t>(j)])]++)] = j;
    }

    // Iterative postorder DFS (leaves before parents) with index cursors —
    // no per-node pair pushes and no vector-of-vectors allocations.
    std::vector<Int> postorder;
    postorder.reserve(sn);
    {
        std::vector<Int> stk_node;
        std::vector<Int> stk_ci;
        stk_node.reserve(sn);
        stk_ci.reserve(sn);
        // Push roots in reverse index order so smallest-index root is
        // processed first (gives ascending postorder for chains).
        for (Int j = n - 1; j >= 0; --j)
            if (parent[static_cast<std::size_t>(j)] == -1) {
                stk_node.push_back(j);
                stk_ci.push_back(0);
            }
        while (!stk_node.empty()) {
            const Int node = stk_node.back();
            const Int ci = stk_ci.back();
            const Int c0 = ch_start[static_cast<std::size_t>(node)];
            const Int c1 = ch_start[static_cast<std::size_t>(node) + 1];
            if (c0 + ci < c1) {
                stk_ci.back() = ci + 1;
                stk_node.push_back(children[static_cast<std::size_t>(c0 + ci)]);
                stk_ci.push_back(0);
            } else {
                postorder.push_back(node);
                stk_node.pop_back();
                stk_ci.pop_back();
            }
        }
    }

    // Phase 3: compute max_depth via bottom-up pass in postorder
    // depth[j] = length of longest path from j down to a leaf
    std::vector<Int> depth(sn, 0);
    Int max_depth = 0;
    for (Int j : postorder) {
        if (parent[static_cast<std::size_t>(j)] >= 0)
            depth[static_cast<std::size_t>(parent[static_cast<std::size_t>(j)])] =
                std::max(depth[static_cast<std::size_t>(parent[static_cast<std::size_t>(j)])],
                         depth[static_cast<std::size_t>(j)] + 1);
        max_depth = std::max(max_depth, depth[static_cast<std::size_t>(j)]);
    }

    et.parent    = std::move(parent);
    et.postorder = std::move(postorder);
    et.max_depth = max_depth;
    return et;
}

} // namespace smf
