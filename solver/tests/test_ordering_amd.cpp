#include <gtest/gtest.h>
#include "smf/ordering.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

using smf::Int;
using smf::AmdOrdering;
using smf::build_symmetric_csr;

// Helper: check that perm is a valid permutation of [0, n-1].
static void check_valid_permutation(const std::vector<Int>& perm, Int n) {
    ASSERT_EQ(static_cast<Int>(perm.size()), n);
    std::vector<Int> sorted_perm(perm);
    std::sort(sorted_perm.begin(), sorted_perm.end());
    for (Int i = 0; i < n; ++i) {
        EXPECT_EQ(sorted_perm[static_cast<std::size_t>(i)], i);
    }
}

// Test 1: SmallTriangle — 5×5 complete lower-triangular (all entries present).
TEST(OrderingAmd, SmallTriangle) {
    const Int n = 5;
    // Lower CSC: column j has rows j..n-1
    // col_ptr[j+1] - col_ptr[j] = n - j
    std::vector<Int> col_ptr = {0, 5, 9, 12, 14, 15};
    std::vector<Int> row_idx = {0,1,2,3,4,  1,2,3,4,  2,3,4,  3,4,  4};

    auto [xadj, adjncy] = build_symmetric_csr(n, col_ptr, row_idx);

    AmdOrdering amd;
    auto perm = amd.compute_ordering(n, xadj, adjncy);

    check_valid_permutation(perm, n);
}

// Test 2: BuildSymmetricCSR_Triangle — 3×3 lower triangle.
// Lower CSC for 3×3:
//   col 0: rows 0,1,2  → diag + 2 off-diag
//   col 1: rows 1,2    → diag + 1 off-diag
//   col 2: rows 2      → diag only
// Off-diagonal entries: (1,0), (2,0), (2,1) → 3 off-diagonal entries.
// Each appears twice in the symmetric CSR → xadj[3] == 6.
TEST(OrderingAmd, BuildSymmetricCSR_Triangle) {
    const Int n = 3;
    std::vector<Int> col_ptr = {0, 3, 5, 6};
    std::vector<Int> row_idx = {0,1,2, 1,2, 2};

    auto [xadj, adjncy] = build_symmetric_csr(n, col_ptr, row_idx);

    const Int num_offdiag = 3;
    EXPECT_EQ(xadj[static_cast<std::size_t>(n)], 2 * num_offdiag);

    // No self-loops
    for (Int v : adjncy) {
        // We'd need the row index to verify i != v; instead check value range.
        EXPECT_GE(v, 0);
        EXPECT_LT(v, n);
    }

    // Each row's adjacency is sorted
    for (Int i = 0; i < n; ++i) {
        EXPECT_TRUE(std::is_sorted(
            adjncy.begin() + xadj[static_cast<std::size_t>(i)],
            adjncy.begin() + xadj[static_cast<std::size_t>(i) + 1]));
    }

    // Verify no self-loops explicitly: row i should not contain i
    for (Int i = 0; i < n; ++i) {
        for (Int k = xadj[static_cast<std::size_t>(i)];
             k < xadj[static_cast<std::size_t>(i) + 1]; ++k) {
            EXPECT_NE(adjncy[static_cast<std::size_t>(k)], i);
        }
    }
}

// Test 3: BuildSymmetricCSR_Diagonal — n×n diagonal matrix (only diagonal entries).
TEST(OrderingAmd, BuildSymmetricCSR_Diagonal) {
    const Int n = 6;
    // col_ptr[j+1] = col_ptr[j] + 1 (one entry per column: the diagonal)
    std::vector<Int> col_ptr(static_cast<std::size_t>(n) + 1);
    std::iota(col_ptr.begin(), col_ptr.end(), 0);
    std::vector<Int> row_idx(static_cast<std::size_t>(n));
    std::iota(row_idx.begin(), row_idx.end(), 0);

    auto [xadj, adjncy] = build_symmetric_csr(n, col_ptr, row_idx);

    EXPECT_EQ(xadj[static_cast<std::size_t>(n)], 0);
    EXPECT_TRUE(adjncy.empty());
}

// Test 4: Complete lower triangle 10×10 — valid permutation only.
TEST(OrderingAmd, CompleteLowerTriangle10x10) {
    const Int n = 10;
    // Build complete lower-triangular CSC
    // col j: rows j, j+1, ..., n-1  → (n - j) entries
    std::vector<Int> col_ptr(static_cast<std::size_t>(n) + 1, 0);
    for (Int j = 0; j < n; ++j)
        col_ptr[static_cast<std::size_t>(j) + 1] =
            col_ptr[static_cast<std::size_t>(j)] + (n - j);
    std::vector<Int> row_idx;
    row_idx.reserve(static_cast<std::size_t>(col_ptr[static_cast<std::size_t>(n)]));
    for (Int j = 0; j < n; ++j)
        for (Int i = j; i < n; ++i)
            row_idx.push_back(i);

    auto [xadj, adjncy] = build_symmetric_csr(n, col_ptr, row_idx);

    AmdOrdering amd;
    auto perm = amd.compute_ordering(n, xadj, adjncy);

    check_valid_permutation(perm, n);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
