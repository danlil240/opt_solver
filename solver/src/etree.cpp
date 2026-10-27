#include "smf/etree.hpp"
#include <vector>
#include <algorithm>

namespace smf {

namespace {

// Children lists, postorder and depth for a parent array (shared by the
// vector-returning etree builders).
void finish_elimination_tree(Int n, std::vector<Int> parent, EliminationTree &et)
{
    const auto sn = static_cast<std::size_t>(n);
    std::vector<Int> work(4 * sn + 1);
    std::vector<Int> postorder(sn);
    et.max_depth = etree_postorder(n, parent.data(), postorder.data(), work.data());
    et.parent    = std::move(parent);
    et.postorder = std::move(postorder);
}

} // namespace

Int etree_postorder(Int n, const Int *parent, Int *postorder, Int *work)
{
    const auto sn = static_cast<std::size_t>(n);
    // Children CSR (ch_start, children) and the DFS stacks (node, child
    // cursor); the child-cursor stack is the fill cursor before the DFS and
    // the node stack is the depth array after it.
    Int *ch_start = work;                // n + 1
    Int *children = ch_start + sn + 1;   // n
    Int *stk_node = children + sn;       // n
    Int *stk_ci = stk_node + sn;         // n
    Int *cursor = stk_ci;

    std::fill(ch_start, ch_start + n + 1, 0);
    for (Int j = 0; j < n; ++j)
        if (parent[j] >= 0)
            ++ch_start[parent[j] + 1];
    for (Int k = 0; k < n; ++k)
        ch_start[k + 1] += ch_start[k];
    std::copy(ch_start, ch_start + n, cursor);
    for (Int j = 0; j < n; ++j)
        if (parent[j] >= 0)
            children[cursor[parent[j]]++] = j;

    // Iterative postorder DFS (leaves before parents) with index cursors.
    // Roots are pushed in reverse index order so the smallest-index root is
    // processed first (gives ascending postorder for chains).
    Int npost = 0;
    {
        Int top = 0;
        for (Int j = n - 1; j >= 0; --j)
            if (parent[j] == -1) {
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
                postorder[npost++] = node;
                --top;
            }
        }
    }

    // depth[j] = length of the longest path from j down to a leaf.
    Int *depth = stk_node;
    std::fill(depth, depth + n, 0);
    Int max_depth = 0;
    for (Int k = 0; k < npost; ++k) {
        const Int j = postorder[k];
        const Int pj = parent[j];
        if (pj >= 0)
            depth[pj] = std::max(depth[pj], depth[j] + 1);
        max_depth = std::max(max_depth, depth[j]);
    }
    return max_depth;
}

void etree_from_adjacency(Int n, const Int *xadj, const Int *adj, const Int *perm,
                          const Int *iperm, Int *parent, Int *ancestor)
{
    // Liu's algorithm over the rows of P A P^T: row k is old vertex perm[k],
    // its entries left of the diagonal the neighbours numbered below k. The
    // tree does not depend on the order a row's entries are visited in.
    std::fill(parent, parent + n, -1);
    std::fill(ancestor, ancestor + n, -1);
    for (Int k = 0; k < n; ++k) {
        const Int v = perm[k];
        for (Int p = xadj[v]; p < xadj[v + 1]; ++p) {
            Int r = iperm[adj[p]];
            if (r >= k) continue;
            while (ancestor[r] != -1 && ancestor[r] != k) {
                const Int t = ancestor[r];
                ancestor[r] = k;
                r = t;
            }
            if (ancestor[r] == -1) {
                parent[r] = k;
                ancestor[r] = k;
            }
        }
    }
}


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
