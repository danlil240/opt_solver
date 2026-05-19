#include "smf/etree.hpp"
#include <gtest/gtest.h>
#include <numeric>

namespace smf {

// Helper: build a simple lower CSC from dense row indices per column
// columns 0..n-1, row_indices_per_col[j] = list of row indices > j in col j
static std::pair<std::vector<Int>, std::vector<Int>>
make_lower_csc(Int n, const std::vector<std::vector<Int>>& rows_per_col) {
    std::vector<Int> col_ptr(n + 1, 0);
    for (Int j = 0; j < n; ++j)
        col_ptr[j + 1] = col_ptr[j] + (Int)rows_per_col[j].size();
    std::vector<Int> row_idx;
    for (auto& r : rows_per_col)
        for (Int i : r) row_idx.push_back(i);
    return {col_ptr, row_idx};
}

// Arrow matrix: column 0 connects to all other columns (rows 1..n-1)
// This gives a tree of depth n-1 rooted at n-1
TEST(ETree, ArrowMatrix) {
    const Int n = 6;
    // Lower triangle: col 0 has rows 1,2,3,4,5; others have no off-diag entries
    std::vector<std::vector<Int>> rows = {{1,2,3,4,5},{},{},{},{},{}};
    auto [cp, ri] = make_lower_csc(n, rows);
    auto et = build_elimination_tree(n, cp, ri);
    ASSERT_EQ((Int)et.parent.size(), n);
    ASSERT_EQ((Int)et.postorder.size(), n);
    // All columns 1..n-2 should have parent = n-1 (or chain from 0)
    // Arrow: each off-diag row i>j in col 0 means col 0 is child of...
    // Actually for lower CSC arrow: col 0 has rows 1..5 (fill edges),
    // so etree: parent[0]=1, parent[1]=2, ..., parent[4]=5, parent[5]=-1 (depth=5)
    EXPECT_GE(et.max_depth, n - 1);
}

// Banded tridiagonal: col j has row j+1 only (for j < n-1)
TEST(ETree, BandedMatrix) {
    const Int n = 8;
    std::vector<std::vector<Int>> rows(n);
    for (Int j = 0; j < n - 1; ++j) rows[j] = {j + 1};
    auto [cp, ri] = make_lower_csc(n, rows);
    auto et = build_elimination_tree(n, cp, ri);
    // Linear chain: parent[j] = j+1, parent[n-1] = -1, depth = n-1
    for (Int j = 0; j < n - 1; ++j)
        EXPECT_EQ(et.parent[j], j + 1) << "j=" << j;
    EXPECT_EQ(et.parent[n - 1], -1);
    EXPECT_EQ(et.max_depth, n - 1);
    // Postorder for a chain = 0, 1, ..., n-1
    for (Int j = 0; j < n; ++j)
        EXPECT_EQ(et.postorder[j], j) << "postorder[" << j << "]";
}

// Block-diagonal 2x2 blocks: two separate trees (forest)
TEST(ETree, BlockDiagonalForest) {
    // 4x4 with two 2x2 dense blocks: (0,1) and (2,3)
    // Col 0: rows {1}, col 1: rows {}, col 2: rows {3}, col 3: rows {}
    std::vector<std::vector<Int>> rows = {{1},{},{3},{}};
    auto [cp, ri] = make_lower_csc(4, rows);
    auto et = build_elimination_tree(4, cp, ri);
    EXPECT_EQ(et.parent[0], 1);
    EXPECT_EQ(et.parent[1], -1);
    EXPECT_EQ(et.parent[2], 3);
    EXPECT_EQ(et.parent[3], -1);
    // max_depth = 1 (one level below root in each tree)
    EXPECT_EQ(et.max_depth, 1);
    ASSERT_EQ((Int)et.postorder.size(), 4);
}

// Empty matrix (n=0)
TEST(ETree, EmptyMatrix) {
    std::vector<Int> cp = {0}, ri = {};
    auto et = build_elimination_tree(0, cp, ri);
    EXPECT_TRUE(et.parent.empty());
    EXPECT_TRUE(et.postorder.empty());
    EXPECT_EQ(et.max_depth, 0);
}

// Diagonal matrix (no off-diag entries): forest of n isolated nodes
TEST(ETree, DiagonalMatrix) {
    const Int n = 5;
    std::vector<Int> cp(n + 1, 0), ri = {};
    auto et = build_elimination_tree(n, cp, ri);
    for (Int j = 0; j < n; ++j)
        EXPECT_EQ(et.parent[j], -1) << "j=" << j;
    EXPECT_EQ(et.max_depth, 0);
    ASSERT_EQ((Int)et.postorder.size(), n);
}

} // namespace smf
