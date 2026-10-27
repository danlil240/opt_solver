#include "smf/solver.hpp"
#include "smf/analysis.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
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

namespace smf {
namespace {

// OPF-like saddle-point KKT: n_x primal variables (5% with tiny curvature),
// m constraints with zero diagonal and 2-4 random Jacobian entries each.
CscLower opf_like_kkt(int n_x, int m, unsigned seed) {
    std::uint64_t state = seed;
    auto next = [&]() { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return state >> 11; };
    auto unit = [&]() { return static_cast<double>(next()) / 9007199254740992.0; };
    const int n = n_x + m;
    std::vector<std::vector<std::pair<Int, double>>> cols(static_cast<std::size_t>(n));
    for (int j = 0; j < n_x; ++j)
        cols[static_cast<std::size_t>(j)].push_back({j, unit() < 0.05 ? 1e-5 : 1.0 + unit()});
    for (int r = 0; r < m; ++r) {
        const int row = n_x + r;
        const int k = 2 + static_cast<int>(next() % 3);
        std::vector<int> used;
        for (int t = 0; t < k; ++t) {
            const int j = static_cast<int>(next() % static_cast<std::uint64_t>(n_x));
            if (std::find(used.begin(), used.end(), j) != used.end()) continue;
            used.push_back(j);
            cols[static_cast<std::size_t>(j)].push_back({row, 0.5 + unit()});
        }
        cols[static_cast<std::size_t>(row)].push_back({row, 0.0});
    }
    CscLower A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        auto &c = cols[static_cast<std::size_t>(j)];
        std::sort(c.begin(), c.end());
        A.col_ptr[static_cast<std::size_t>(j) + 1] = A.col_ptr[static_cast<std::size_t>(j)] + static_cast<Int>(c.size());
        for (const auto &e : c) {
            A.row_idx.push_back(e.first);
            A.values.push_back(e.second);
        }
    }
    return A;
}

} // namespace

// ScalingMethod::Auto applies the analysis' matching scaling. That scaling
// only helps together with the matching ordering that keeps 2x2 pairs
// adjacent; with AMD kept it degrades accuracy (pglib case9241 failed the
// replay accuracy gate on 17 of 73 steps through it). An analysis that kept
// AMD must therefore carry no matching scale.
TEST(Symbolic, AutoMatchingScaleOnlyWithMatchingOrdering) {
    const CscLower A = opf_like_kkt(6000, 5000, 7);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.ordering = OrderingMethod::AutoSerial;
    Info info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak, nullptr);
    ASSERT_EQ(info.ordering_used, static_cast<int>(OrderingMethod::AMD))
        << "fixture must make the auto policy keep AMD";
    EXPECT_TRUE(ak->matching_scale.empty());
}

// Per-step analyse timing: every phase is non-negative, ordering is measured,
// and the phases account for (nearly) all of analyse_seconds.
TEST(Symbolic, AnalysePhasesSumToTotal) {
    const Int g = 60, n = g * g;
    CscLower A;
    A.n = n;
    A.col_ptr.assign(1, 0);
    for (Int j = 0; j < n; ++j) {
        const Int x = j % g, y = j / g;
        A.row_idx.push_back(j);
        A.values.push_back(4.0);
        if (x + 1 < g) { A.row_idx.push_back(j + 1); A.values.push_back(-1.0); }
        if (y + 1 < g) { A.row_idx.push_back(j + g); A.values.push_back(-1.0); }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    Info info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak, nullptr);
    const AnalysePhaseSeconds &p = info.analyse_phases;
    EXPECT_GE(p.check, 0.0);
    EXPECT_GT(p.ordering, 0.0);
    EXPECT_GE(p.matching, 0.0);
    EXPECT_GE(p.etree_counts, 0.0);
    EXPECT_GE(p.amalgamate, 0.0);
    EXPECT_GE(p.assembly_tree, 0.0);
    EXPECT_GE(p.keep_build, 0.0);
    const double sum = p.check + p.ordering + p.matching + p.etree_counts +
                       p.amalgamate + p.assembly_tree + p.keep_build;
    ASSERT_GT(info.analyse_seconds, 0.0);
    EXPECT_LE(sum, info.analyse_seconds * 1.0000001);
    EXPECT_GE(sum, 0.8 * info.analyse_seconds);
}

} // namespace smf
