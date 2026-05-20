#include "smf/supernode.hpp"
#include "smf/etree.hpp"
#include <gtest/gtest.h>

namespace smf {

// Helper: build lower CSC for a tridiagonal n×n matrix.
// Column j has rows: j (diagonal), j+1 (subdiagonal, if j < n-1).
static std::pair<std::vector<Int>, std::vector<Int>>
tridiag_lower_csc(Int n) {
    std::vector<Int> cp(static_cast<std::size_t>(n) + 1, 0);
    for (Int j = 0; j < n; ++j)
        cp[static_cast<std::size_t>(j) + 1] = cp[static_cast<std::size_t>(j)] + (j < n - 1 ? 2 : 1);
    std::vector<Int> ri;
    for (Int j = 0; j < n; ++j) {
        ri.push_back(j);
        if (j < n - 1) ri.push_back(j + 1);
    }
    return {cp, ri};
}

// Helper: build dense lower CSC for an n×n matrix.
// Column j has rows j..n-1.
static std::pair<std::vector<Int>, std::vector<Int>>
dense_lower_csc(Int n) {
    std::vector<Int> cp(static_cast<std::size_t>(n) + 1, 0);
    for (Int j = 0; j < n; ++j)
        cp[static_cast<std::size_t>(j) + 1] = cp[static_cast<std::size_t>(j)] + (n - j);
    std::vector<Int> ri;
    for (Int j = 0; j < n; ++j)
        for (Int i = j; i < n; ++i)
            ri.push_back(i);
    return {cp, ri};
}

// Dense n×n lower triangle: col_count[j]=n-j, arithmetic sequence.
// Etree is a linear chain 0→1→...→n-1.
// Fundamental-supernode conditions are satisfied for all consecutive pairs,
// so all columns collapse into ONE supernode.
TEST(Supernode, DenseOneGroup) {
    const Int n = 5;
    auto [cp, ri] = dense_lower_csc(n);

    EliminationTree et;
    et.parent.resize(static_cast<std::size_t>(n));
    for (Int j = 0; j < n - 1; ++j) et.parent[static_cast<std::size_t>(j)] = j + 1;
    et.parent[static_cast<std::size_t>(n - 1)] = -1;
    et.postorder.resize(static_cast<std::size_t>(n));
    for (Int j = 0; j < n; ++j) et.postorder[static_cast<std::size_t>(j)] = j;
    et.max_depth = n - 1;

    auto snodes = detect_fundamental_supernodes(n, cp, ri, et);

    // Dense + linear chain: col_count[j-1] = col_count[j]+1 ✓, parent[j-1]=j ✓,
    // nchildren[j]=1 (child j-1) ✓ → all columns continue → ONE supernode.
    ASSERT_EQ(static_cast<Int>(snodes.size()), 1) << "Dense matrix should have 1 supernode";
    EXPECT_EQ(snodes[0].col_start, 0);
    EXPECT_EQ(snodes[0].col_end,   n);
    EXPECT_EQ(snodes[0].parent,   -1);
}

// Tridiagonal n=6: col_count[j]=2 for j<n-1, col_count[n-1]=1.
// For j=1..n-2: col_count[j-1]=2 ≠ col_count[j]+1=3 → new supernode each step.
// For j=n-1:   col_count[n-2]=2 = col_count[n-1]+1=2 → continues prev supernode.
// Result: n-1 supernodes (last two columns merge into one of width 2).
TEST(Supernode, TridiagManySingletons) {
    const Int n = 6;
    auto [cp, ri] = tridiag_lower_csc(n);

    EliminationTree et;
    et.parent.resize(static_cast<std::size_t>(n));
    for (Int j = 0; j < n - 1; ++j) et.parent[static_cast<std::size_t>(j)] = j + 1;
    et.parent[static_cast<std::size_t>(n - 1)] = -1;
    et.postorder.resize(static_cast<std::size_t>(n));
    for (Int j = 0; j < n; ++j) et.postorder[static_cast<std::size_t>(j)] = j;
    et.max_depth = n - 1;

    auto snodes = detect_fundamental_supernodes(n, cp, ri, et);

    // n-1 supernodes: first n-2 are singletons, last has width 2
    ASSERT_EQ(static_cast<Int>(snodes.size()), n - 1);
    for (Int s = 0; s < n - 2; ++s)
        EXPECT_EQ(snodes[static_cast<std::size_t>(s)].width(), 1) << "s=" << s;
    EXPECT_EQ(snodes[static_cast<std::size_t>(n - 2)].width(), 2);

    // Total column coverage must equal n
    Int total = 0;
    for (auto& sn : snodes) total += sn.width();
    EXPECT_EQ(total, n);
}

// Amalgamation: tridiagonal n=6 with nemin=3 should reduce to fewer supernodes.
TEST(Supernode, AmalgamationReduces) {
    const Int n = 6;
    auto [cp, ri] = tridiag_lower_csc(n);

    EliminationTree et;
    et.parent.resize(static_cast<std::size_t>(n));
    for (Int j = 0; j < n - 1; ++j) et.parent[static_cast<std::size_t>(j)] = j + 1;
    et.parent[static_cast<std::size_t>(n - 1)] = -1;
    et.postorder.resize(static_cast<std::size_t>(n));
    for (Int j = 0; j < n; ++j) et.postorder[static_cast<std::size_t>(j)] = j;
    et.max_depth = n - 1;

    auto snodes = detect_fundamental_supernodes(n, cp, ri, et);
    const Int before = static_cast<Int>(snodes.size());
    auto amalg = amalgamate_supernodes(std::move(snodes), 3);

    // Count must be strictly smaller after amalgamation
    EXPECT_LT(static_cast<Int>(amalg.size()), before);

    // Total column coverage must still equal n
    Int total = 0;
    for (auto& s : amalg) total += s.width();
    EXPECT_EQ(total, n);
}

TEST(Supernode, AmalgamationKeepsParentSiblings) {
    std::vector<Supernode> supernodes(4);
    for (Int i = 0; i < 4; ++i) {
        supernodes[static_cast<std::size_t>(i)].col_start = i;
        supernodes[static_cast<std::size_t>(i)].col_end = i + 1;
        supernodes[static_cast<std::size_t>(i)].parent = -1;
    }

    supernodes[1].parent = 2;
    supernodes[0].parent = 2;
    supernodes[2].children = {0, 1};

    auto amalg = amalgamate_supernodes(std::move(supernodes), 2);

    ASSERT_EQ(static_cast<Int>(amalg.size()), 3);
    EXPECT_EQ(amalg[1].col_start, 1);
    EXPECT_EQ(amalg[1].col_end, 3);
    ASSERT_EQ(amalg[1].children.size(), 1u);
    EXPECT_EQ(amalg[1].children[0], 0);
    EXPECT_EQ(amalg[0].parent, 1);
}

// n=0 edge case
TEST(Supernode, EmptyMatrix) {
    std::vector<Int> cp = {0};
    std::vector<Int> ri = {};
    EliminationTree et;

    auto snodes = detect_fundamental_supernodes(0, cp, ri, et);
    EXPECT_TRUE(snodes.empty());

    auto amalg = amalgamate_supernodes(std::move(snodes), 8);
    EXPECT_TRUE(amalg.empty());
}

// Amalgamation with nemin=1 must not change anything
TEST(Supernode, AmalgamationNemin1NoChange) {
    const Int n = 4;
    auto [cp, ri] = tridiag_lower_csc(n);

    EliminationTree et;
    et.parent   = {1, 2, 3, -1};
    et.postorder = {0, 1, 2, 3};
    et.max_depth = 3;

    auto snodes = detect_fundamental_supernodes(n, cp, ri, et);
    const Int before = static_cast<Int>(snodes.size());

    auto amalg = amalgamate_supernodes(std::move(snodes), 1);
    EXPECT_EQ(static_cast<Int>(amalg.size()), before);
}

} // namespace smf
