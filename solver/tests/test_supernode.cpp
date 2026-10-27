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

    auto amalg = amalgamate_supernodes(std::move(supernodes), 2);

    ASSERT_EQ(static_cast<Int>(amalg.size()), 3);
    EXPECT_EQ(amalg[1].col_start, 1);
    EXPECT_EQ(amalg[1].col_end, 3);
    std::vector<Int> child_ptr, child_idx;
    build_children_csr(amalg, child_ptr, child_idx);
    ASSERT_EQ(child_ptr[2] - child_ptr[1], 1);
    EXPECT_EQ(child_idx[static_cast<std::size_t>(child_ptr[1])], 0);
    EXPECT_EQ(amalg[0].parent, 1);
}

// Children CSR from parent links: every non-root appears once, under its
// parent, in descending index order; roots and leaves get empty ranges.
TEST(Supernode, ChildrenCsrDescending) {
    // Forest: 0,1,3 -> 4; 2 -> 3; 4 root; 5 root (isolated); 6 -> 7; 7 root.
    std::vector<Supernode> sn(8);
    const Int parent[8] = {4, 4, 3, 4, -1, -1, 7, -1};
    for (Int s = 0; s < 8; ++s) {
        sn[static_cast<std::size_t>(s)].col_start = s;
        sn[static_cast<std::size_t>(s)].col_end = s + 1;
        sn[static_cast<std::size_t>(s)].parent = parent[s];
    }
    std::vector<Int> child_ptr, child_idx;
    build_children_csr(sn, child_ptr, child_idx);
    ASSERT_EQ(child_ptr.size(), 9u);
    EXPECT_EQ(child_ptr, (std::vector<Int>{0, 0, 0, 0, 1, 4, 4, 4, 5}));
    EXPECT_EQ(child_idx, (std::vector<Int>{2, 3, 1, 0, 6}));

    std::vector<Supernode> none;
    build_children_csr(none, child_ptr, child_idx);
    EXPECT_EQ(child_ptr, (std::vector<Int>{0}));
    EXPECT_TRUE(child_idx.empty());
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

// Relaxed amalgamation bounds the zeros a merge stores. Leaf column 0
// (rows 0, 1) hangs off column 1, whose structure is rows 1, 3..41; column
// 2 is a leaf of the dense chain 3..41. Merging {0} into {1} would store
// 2 x 41 values for 42 nonzeros: 40 added zeros, 49% of the node, so it is
// refused although both nodes are narrower than nemin. With column 1's
// structure only rows 1, 3, the same merge adds 2 zeros and is taken.
TEST(Supernode, RelaxedAmalgamationBoundsZeros) {
    const Int n = 42;
    std::vector<Int> parent(static_cast<std::size_t>(n)), colcount(static_cast<std::size_t>(n));
    parent[0] = 1;
    parent[1] = 3;
    parent[2] = 3;
    colcount[0] = 2;
    colcount[2] = 2;
    for (Int j = 3; j < n; ++j) {
        parent[static_cast<std::size_t>(j)] = (j + 1 < n) ? j + 1 : -1;
        colcount[static_cast<std::size_t>(j)] = n - j;
    }

    std::vector<Int> order;
    std::vector<Supernode> snodes;
    colcount[1] = 40;  // rows 1, 3..41
    amalgamate_relaxed(n, parent.data(), colcount.data(), 8, order, snodes);
    EXPECT_EQ(snodes.size(), 4u);

    colcount[1] = 2;   // rows 1, 3
    amalgamate_relaxed(n, parent.data(), colcount.data(), 8, order, snodes);
    ASSERT_EQ(snodes.size(), 3u);
    EXPECT_EQ(snodes[0].width(), 2);
}

} // namespace smf

#include <random>

// Exact column counts (Gilbert-Ng-Peyton) against a brute-force symbolic
// Cholesky on random sparse patterns, including disconnected forests.
TEST(ColumnCounts, MatchBruteForceSymbolicFactor) {
    for (unsigned seed = 1; seed <= 25; ++seed) {
        std::mt19937 rng(seed);
        const int n = 20 + static_cast<int>(seed * 7 % 140);
        const double density = (seed % 3 == 0) ? 0.01 : 0.05;
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        std::vector<std::vector<char>> L(static_cast<std::size_t>(n), std::vector<char>(static_cast<std::size_t>(n), 0));
        std::vector<smf::Int> row_ptr{0}, row_cols;
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < i; ++k) {
                if (unif(rng) < density) {
                    row_cols.push_back(k);
                    L[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] = 1;
                }
            }
            row_cols.push_back(i);  // diagonal (ignored by the algorithms)
            row_ptr.push_back(static_cast<smf::Int>(row_cols.size()));
        }
        // Symbolic elimination: struct(L_j) rows fill every later column in it.
        for (int j = 0; j < n; ++j)
            for (int i = j + 1; i < n; ++i)
                if (L[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)])
                    for (int k = j + 1; k < i; ++k)
                        if (L[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)])
                            L[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] = 1;
        const smf::EliminationTree et =
            smf::build_elimination_tree_from_rows(n, row_ptr.data(), row_cols.data());
        const std::vector<smf::Int> got =
            smf::filled_column_counts_from_rows(n, row_ptr.data(), row_cols.data(), et);
        ASSERT_EQ(got.size(), static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) {
            int expect = 1;
            for (int i = j + 1; i < n; ++i) expect += L[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            ASSERT_EQ(got[static_cast<std::size_t>(j)], expect) << "seed=" << seed << " col=" << j;
        }
    }
}
