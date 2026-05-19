#include "smf/solver.hpp"
#include "smf/analysis.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <cmath>

namespace smf {

// Build a 50x50 SPD banded lower CSC (bandwidth 3)
static CscLower make_spd50() {
    const Int n = 50;
    const Int bw = 3;
    CscLower A;
    A.n = n;
    A.col_ptr.resize(n + 1, 0);
    for (Int j = 0; j < n; ++j) {
        Int cnt = 1; // diagonal
        for (Int k = 1; k <= bw && j + k < n; ++k) ++cnt;
        A.col_ptr[j + 1] = A.col_ptr[j] + cnt;
    }
    A.row_idx.reserve(A.col_ptr[n]);
    A.values.reserve(A.col_ptr[n]);
    for (Int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);          // diagonal
        A.values.push_back(10.0 + bw);   // dominant diagonal
        for (Int k = 1; k <= bw && j + k < n; ++k) {
            A.row_idx.push_back(j + k);
            A.values.push_back(-1.0);
        }
    }
    return A;
}

// End-to-end: analyse a 50x50 SPD matrix, check all predictions populated
TEST(Symbolic, Analyse50x50) {
    Solver solver;
    CscLower A = make_spd50();
    Control ctrl{};
    Info info{};

    auto keep = solver.analyse(A, ctrl, info);

    ASSERT_NE(keep.get(), nullptr) << "analyse returned nullptr";
    EXPECT_EQ(info.status, ErrorCode::Success);
    EXPECT_EQ(keep->n, 50);

    // Permutation vectors should be valid
    ASSERT_EQ((Int)keep->perm.size(), 50);
    ASSERT_EQ((Int)keep->iperm.size(), 50);
    // perm and iperm are inverses
    for (Int i = 0; i < 50; ++i) {
        EXPECT_EQ(keep->iperm[keep->perm[i]], i) << "i=" << i;
    }

    // Predictions should be positive
    EXPECT_GT(info.predicted_factor_entries, 0);
    EXPECT_GT(info.predicted_flops, 0.0);
    EXPECT_GT(info.max_front_size, 0);
    EXPECT_GT(info.max_supernode_size, 0);

    // Timing should be non-negative
    EXPECT_GE(info.analyse_seconds, 0.0);
}

// n=0 edge case
TEST(Symbolic, AnalyseEmpty) {
    Solver solver;
    CscLower A;
    A.n = 0;
    A.col_ptr = {0};
    Control ctrl{};
    Info info{};
    auto keep = solver.analyse(A, ctrl, info);
    ASSERT_NE(keep.get(), nullptr);
    EXPECT_EQ(keep->n, 0);
    EXPECT_EQ(info.status, ErrorCode::Success);
}

// Diagonal 5x5 matrix (no off-diagonal entries)
TEST(Symbolic, AnalyseDiagonal5) {
    Solver solver;
    CscLower A;
    A.n = 5;
    A.col_ptr = {0, 1, 2, 3, 4, 5};
    A.row_idx = {0, 1, 2, 3, 4};
    A.values  = {1, 1, 1, 1, 1};
    Control ctrl{};
    Info info{};
    auto keep = solver.analyse(A, ctrl, info);
    ASSERT_NE(keep.get(), nullptr);
    EXPECT_EQ(keep->n, 5);
    EXPECT_EQ(info.status, ErrorCode::Success);
    EXPECT_GT(info.predicted_factor_entries, 0);
}

} // namespace smf
