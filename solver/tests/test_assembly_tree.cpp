#include "smf/assembly_tree.hpp"
#include "smf/supernode.hpp"
#include "smf/etree.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include <gtest/gtest.h>

namespace smf {

// Helper: 5x5 dense lower CSC (all entries)
static CscLower make_dense5() {
    CscLower A;
    A.n = 5;
    A.col_ptr = {0, 5, 9, 12, 14, 15};
    A.row_idx = {0,1,2,3,4, 1,2,3,4, 2,3,4, 3,4, 4};
    A.values.resize(15, 1.0);
    return A;
}

// Helper: single supernode covering all 5 columns
static std::vector<Supernode> one_supernode5() {
    Supernode sn;
    sn.col_start = 0;
    sn.col_end = 5;
    sn.parent = -1;
    return {sn};
}

// Dense 5x5: one supernode, front_size = 5, predicted_factor_entries = 5
TEST(AssemblyTree, Dense5x5) {
    CscLower A = make_dense5();
    auto snodes = one_supernode5();
    Info info{};
    auto fronts = build_assembly_tree(A, snodes, info);

    ASSERT_EQ((Int)fronts.size(), 1);
    EXPECT_EQ(fronts[0].front_size(), 5);
    EXPECT_EQ(info.predicted_factor_entries, 5);
    // flops = width * front_size^2 = 5 * 25 = 125
    EXPECT_DOUBLE_EQ(info.predicted_flops, 125.0);
    EXPECT_EQ(info.max_front_size, 5);
    EXPECT_EQ(info.max_supernode_size, 5);
}

// 10x10 banded (tridiagonal): each column has 2 entries (diag + 1 subdiag),
// except last which has 1. With n singleton supernodes, each front has size 2
// (except last = size 1).
TEST(AssemblyTree, Banded10x10Singletons) {
    const Int n = 10;
    CscLower A;
    A.n = n;
    A.col_ptr.resize(n + 1, 0);
    for (Int j = 0; j < n; ++j)
        A.col_ptr[j + 1] = A.col_ptr[j] + (j < n - 1 ? 2 : 1);
    A.row_idx.clear();
    for (Int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        if (j < n - 1) A.row_idx.push_back(j + 1);
    }
    A.values.resize(A.row_idx.size(), 1.0);

    // Make n singleton supernodes in a chain
    std::vector<Supernode> snodes(n);
    for (Int j = 0; j < n; ++j) {
        snodes[j].col_start = j;
        snodes[j].col_end = j + 1;
        snodes[j].parent = (j < n - 1) ? j + 1 : -1;
    }
    for (auto& s : snodes) s.children.clear();
    for (Int j = 0; j < n - 1; ++j)
        snodes[j + 1].children.push_back(j);

    Info info{};
    auto fronts = build_assembly_tree(A, snodes, info);

    ASSERT_EQ((Int)fronts.size(), n);
    for (Int j = 0; j < n - 1; ++j)
        EXPECT_EQ(fronts[j].front_size(), 2) << "j=" << j;
    EXPECT_EQ(fronts[n - 1].front_size(), 1);

    EXPECT_GT(info.predicted_factor_entries, 0);
    EXPECT_GT(info.predicted_flops, 0.0);
    EXPECT_EQ(info.max_front_size, 2);
    EXPECT_EQ(info.max_supernode_size, 1);
}

// Empty matrix
TEST(AssemblyTree, Empty) {
    CscLower A;
    A.n = 0;
    A.col_ptr = {0};
    std::vector<Supernode> snodes;
    Info info{};
    auto fronts = build_assembly_tree(A, snodes, info);
    EXPECT_TRUE(fronts.empty());
    EXPECT_EQ(info.predicted_factor_entries, 0);
    EXPECT_DOUBLE_EQ(info.predicted_flops, 0.0);
}

// Predictions are non-negative and max values are positive for non-trivial input
TEST(AssemblyTree, PredictionsNonNegative) {
    CscLower A = make_dense5();
    auto snodes = one_supernode5();
    Info info{};
    build_assembly_tree(A, snodes, info);
    EXPECT_GE(info.predicted_factor_entries, 0);
    EXPECT_GE(info.predicted_flops, 0.0);
    EXPECT_GE(info.max_front_size, 0);
    EXPECT_GE(info.max_supernode_size, 0);
    EXPECT_GE(info.max_tree_depth, 0);
}

} // namespace smf
