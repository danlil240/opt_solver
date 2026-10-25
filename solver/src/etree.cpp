#include "smf/etree.hpp"
#include <vector>
#include <stack>
#include <algorithm>

namespace smf {

namespace {

// Children lists, postorder and depth for a parent array (shared by both
// etree builders).
void finish_elimination_tree(Int n, std::vector<Int> parent, EliminationTree &et)
{
    const auto sn = static_cast<std::size_t>(n);
    // All temporaries in one block: children CSR (ch_start, children), a
    // fill cursor, and the DFS stacks (node, child cursor) reused as the
    // depth array afterwards.
    std::vector<Int> work(5 * sn + 1, 0);
    Int *ch_start = work.data();         // n + 1
    Int *children = ch_start + sn + 1;   // n
    Int *cursor = children + sn;         // n
    Int *stk_node = cursor + sn;         // n
    Int *stk_ci = stk_node + sn;         // n

    // Phase 2: build children lists (flat CSR) for postorder traversal
    for (Int j = 0; j < n; ++j)
        if (parent[static_cast<std::size_t>(j)] >= 0)
            ++ch_start[parent[static_cast<std::size_t>(j)] + 1];
    for (Int k = 0; k < n; ++k)
        ch_start[k + 1] += ch_start[k];
    std::copy(ch_start, ch_start + n, cursor);
    for (Int j = 0; j < n; ++j)
        if (parent[static_cast<std::size_t>(j)] >= 0)
            children[cursor[parent[static_cast<std::size_t>(j)]]++] = j;

    // Iterative postorder DFS (leaves before parents) with index cursors.
    std::vector<Int> postorder;
    postorder.reserve(sn);
    {
        Int top = 0;
        // Push roots in reverse index order so smallest-index root is
        // processed first (gives ascending postorder for chains).
        for (Int j = n - 1; j >= 0; --j)
            if (parent[static_cast<std::size_t>(j)] == -1) {
                stk_node[top] = j;
                stk_ci[top] = 0;
                ++top;
            }
        while (top > 0) {
            const Int node = stk_node[top - 1];
            const Int ci = stk_ci[top - 1];
            const Int c0 = ch_start[node];
            const Int c1 = ch_start[node + 1];
            if (c0 + ci < c1) {
                stk_ci[top - 1] = ci + 1;
                stk_node[top] = children[c0 + ci];
                stk_ci[top] = 0;
                ++top;
            } else {
                postorder.push_back(node);
                --top;
            }
        }
    }

    // Phase 3: compute max_depth via bottom-up pass in postorder
    // depth[j] = length of longest path from j down to a leaf
    Int *depth = stk_node;
    std::fill(depth, depth + n, 0);
    Int max_depth = 0;
    for (Int j : postorder) {
        const Int pj = parent[static_cast<std::size_t>(j)];
        if (pj >= 0)
            depth[pj] = std::max(depth[pj], depth[j] + 1);
        max_depth = std::max(max_depth, depth[j]);
    }

    et.parent    = std::move(parent);
    et.postorder = std::move(postorder);
    et.max_depth = max_depth;
}

} // namespace


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

    finish_elimination_tree(n, std::move(parent), et);
    return et;
}

EliminationTree build_elimination_tree_from_rows(Int n, const Int *row_ptr,
                                                 const Int *row_cols)
{
    EliminationTree et;
    if (n == 0) return et;
    const auto sn = static_cast<std::size_t>(n);
    std::vector<Int> parent(sn, -1);
    std::vector<Int> ancestor(sn, -1);
    for (Int k = 0; k < n; ++k) {
        for (Int p = row_ptr[k]; p < row_ptr[k + 1]; ++p) {
            Int r = row_cols[p];
            if (r >= k) continue; // diagonal
            while (ancestor[static_cast<std::size_t>(r)] != -1 &&
                   ancestor[static_cast<std::size_t>(r)] != k) {
                const Int t = ancestor[static_cast<std::size_t>(r)];
                ancestor[static_cast<std::size_t>(r)] = k;
                r = t;
            }
            if (ancestor[static_cast<std::size_t>(r)] == -1) {
                parent[static_cast<std::size_t>(r)]   = k;
                ancestor[static_cast<std::size_t>(r)] = k;
            }
        }
    }
    finish_elimination_tree(n, std::move(parent), et);
    return et;
}

} // namespace smf
