#include <gtest/gtest.h>
#include "smf/etree.hpp"
#include "smf/ordering.hpp"
#include "smf/ordering_metis.hpp"
#include "smf/supernode.hpp"
#include "smf/scaling_matching.hpp"
#include "smf/check_matrix.hpp"
#include "smf/matrix_market.hpp"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <set>
#include <vector>
#include <utility>

static std::pair<std::vector<int>, std::vector<int>>
make_csr(int n, std::vector<std::pair<int,int>> edges)
{
    std::vector<int> deg(static_cast<std::size_t>(n), 0);
    for (auto [i, j] : edges) {
        ++deg[static_cast<std::size_t>(i)];
        ++deg[static_cast<std::size_t>(j)];
    }
    std::vector<int> xadj(static_cast<std::size_t>(n + 1), 0);
    for (int k = 0; k < n; ++k)
        xadj[static_cast<std::size_t>(k + 1)] =
            xadj[static_cast<std::size_t>(k)] + deg[static_cast<std::size_t>(k)];
    std::vector<int> adjncy(static_cast<std::size_t>(xadj[static_cast<std::size_t>(n)]));
    std::vector<int> pos(xadj.begin(), xadj.begin() + n);
    for (auto [i, j] : edges) {
        adjncy[static_cast<std::size_t>(pos[static_cast<std::size_t>(i)]++)] = j;
        adjncy[static_cast<std::size_t>(pos[static_cast<std::size_t>(j)]++)] = i;
    }
    for (int k = 0; k < n; ++k)
        std::sort(adjncy.begin() + xadj[static_cast<std::size_t>(k)],
                  adjncy.begin() + xadj[static_cast<std::size_t>(k + 1)]);
    return {xadj, adjncy};
}

static bool is_valid_permutation(const std::vector<smf::Int>& perm, int n)
{
    if (static_cast<int>(perm.size()) != n) return false;
    std::set<smf::Int> seen(perm.begin(), perm.end());
    if (static_cast<int>(seen.size()) != n) return false;
    if (*seen.begin() != 0) return false;
    if (*seen.rbegin() != n - 1) return false;
    return true;
}

TEST(MetisOrdering, ValidPermutation_Small)
{
    std::vector<std::pair<int,int>> edges;
    for (int i = 1; i < 5; ++i)
        for (int j = 0; j < i; ++j)
            edges.push_back({i, j});
    auto [xadj, adjncy] = make_csr(5, edges);
    smf::MetisOrdering ord;
    smf::Info info;
    auto perm = ord.compute_ordering(5, xadj, adjncy, info);
    ASSERT_FALSE(perm.empty());
    EXPECT_EQ(static_cast<int>(perm.size()), 5);
    EXPECT_TRUE(is_valid_permutation(perm, 5));
    EXPECT_EQ(info.status, smf::ErrorCode::Success);
}

TEST(MetisOrdering, Chain100_ValidPermutation)
{
    std::vector<std::pair<int,int>> edges;
    for (int i = 1; i < 100; ++i)
        edges.push_back({i, i - 1});
    auto [xadj, adjncy] = make_csr(100, edges);
    smf::MetisOrdering ord;
    smf::Info info;
    auto perm = ord.compute_ordering(100, xadj, adjncy, info);
    ASSERT_FALSE(perm.empty());
    EXPECT_TRUE(is_valid_permutation(perm, 100));
}

TEST(MetisOrdering, DiagonalMatrix_EmptyGraph)
{
    std::vector<int> xadj(6, 0);
    std::vector<int> adjncy;
    smf::MetisOrdering ord;
    smf::Info info;
    auto perm = ord.compute_ordering(5, xadj, adjncy, info);
    if (info.status == smf::ErrorCode::Success) {
        EXPECT_TRUE(is_valid_permutation(perm, 5));
    } else {
        EXPECT_EQ(info.status, smf::ErrorCode::OrderingFailed);
        EXPECT_TRUE(perm.empty());
    }
}

TEST(MetisOrdering, FillReduction_Grid)
{
    const int N = 100;
    std::vector<std::pair<int,int>> edges;
    for (int r = 0; r < 10; ++r) {
        for (int c = 0; c < 10; ++c) {
            int node = r * 10 + c;
            if (c + 1 < 10) edges.push_back({r * 10 + (c + 1), node});
            if (r + 1 < 10) edges.push_back({(r + 1) * 10 + c, node});
        }
    }
    auto [xadj, adjncy] = make_csr(N, edges);
    smf::MetisOrdering ord;
    smf::Info info;
    auto perm = ord.compute_ordering(N, xadj, adjncy, info);
    ASSERT_FALSE(perm.empty());
    EXPECT_TRUE(is_valid_permutation(perm, N));
}


// ---------------------------------------------------------------------------
// Matching-based compressed ordering (MA97 ordering 7/8 parity).

namespace {

// KKT [H B^T; B 0]: H the 5-point Laplacian of a g x g grid (diagonal 4),
// then m constraint rows with an explicit zero diagonal, row c coupling
// variables (3c, 3c + 1, 3c + 2) mod g^2. Lower CSC, rows sorted.
smf::CscLower kkt_grid(int g, int m)
{
    const int nx = g * g, n = nx + m;
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n));
    for (int y = 0; y < g; ++y)
        for (int x = 0; x < g; ++x) {
            const int j = y * g + x;
            auto &c = cols[static_cast<std::size_t>(j)];
            c.push_back({j, 4.0});
            if (x + 1 < g) c.push_back({j + 1, -1.0});
            if (y + 1 < g) c.push_back({j + g, -1.0});
        }
    const double coef[3] = {1.0, -0.5, 0.25};
    for (int k = 0; k < m; ++k) {
        const int row = nx + k;
        for (int t = 0; t < 3; ++t)
            cols[static_cast<std::size_t>((3 * k + t) % nx)].push_back({row, coef[t]});
        cols[static_cast<std::size_t>(row)].push_back({row, 0.0});
    }
    smf::CscLower A;
    A.n = n;
    A.col_ptr.assign(1, 0);
    for (auto &c : cols) {
        std::sort(c.begin(), c.end());
        for (auto [i, v] : c) {
            A.row_idx.push_back(i);
            A.values.push_back(v);
        }
        A.col_ptr.push_back(static_cast<smf::Int>(A.row_idx.size()));
    }
    return A;
}

} // namespace

TEST(MetisOrdering, MatchingKeepsPairsAdjacent)
{
    const int g = 30, m = 200, nx = g * g, n = nx + m;
    const smf::CscLower A = kkt_grid(g, m);
    for (smf::CompressedOrder how : {smf::CompressedOrder::Amd, smf::CompressedOrder::Metis,
                                     smf::CompressedOrder::Auto}) {
        std::vector<smf::Int> perm, work;
        smf::MatchingOrderStats st;
        ASSERT_TRUE(smf::matching_order_lower_csc(A, 0.1, perm, work, &st, how));
        ASSERT_TRUE(is_valid_permutation(perm, n));
        EXPECT_EQ(st.pairs, m);
        EXPECT_EQ(st.nodes, n - m);
        EXPECT_GT(st.flops, 0.0);
        std::vector<int> pos(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k)
            pos[static_cast<std::size_t>(perm[static_cast<std::size_t>(k)])] = k;
        // Every zero-diagonal row sits next to one of its own variables.
        for (int k = 0; k < m; ++k) {
            const int pz = pos[static_cast<std::size_t>(nx + k)];
            bool adjacent = false;
            for (int t = 0; t < 3; ++t) {
                const int v = (3 * k + t) % nx;
                adjacent = adjacent || std::abs(pos[static_cast<std::size_t>(v)] - pz) == 1;
            }
            EXPECT_TRUE(adjacent) << "constraint " << k << " how " << static_cast<int>(how);
        }
    }
}

// The compressed ordering also takes a partial matching (the matching's
// initial phase leaves free columns, and freeing a cycle's columns leaves
// open paths): it stays a valid permutation, and freed constraints lose
// their pairs.
TEST(MetisOrdering, MatchingOrderFromPartialMatching)
{
    const int g = 30, m = 200, nx = g * g, n = nx + m;
    const smf::CscLower A = kkt_grid(g, m);
    smf::MatchingResult mr = smf::compute_symmetric_matching(A);
    ASSERT_EQ(mr.status, smf::ErrorCode::Success);
    std::vector<smf::Int> perm, work;
    smf::MatchingOrderStats full;
    ASSERT_TRUE(smf::matching_order_from(A, mr, 0.1, perm, work, &full));
    ASSERT_TRUE(is_valid_permutation(perm, n));
    EXPECT_EQ(full.pairs, m);

    // Free every 4th constraint, as a column and as a row.
    int freed = 0;
    for (int k = 0; k < m; k += 4) {
        const smf::Int c = nx + k;
        const smf::Int r = mr.col_to_row[static_cast<std::size_t>(c)];
        const smf::Int c2 = mr.row_to_col[static_cast<std::size_t>(c)];
        mr.col_to_row[static_cast<std::size_t>(c)] = -1;
        mr.row_to_col[static_cast<std::size_t>(r)] = -1;
        mr.col_to_row[static_cast<std::size_t>(c2)] = -1;
        mr.row_to_col[static_cast<std::size_t>(c)] = -1;
        mr.matched_rank -= (c2 == c) ? 1 : 2;
        ++freed;
    }
    for (smf::CompressedOrder how : {smf::CompressedOrder::Amd, smf::CompressedOrder::Auto}) {
        smf::MatchingOrderStats part;
        ASSERT_TRUE(smf::matching_order_from(A, mr, 0.1, perm, work, &part, how));
        ASSERT_TRUE(is_valid_permutation(perm, n));
        EXPECT_GT(part.pairs, 0);
        EXPECT_LT(part.pairs, full.pairs);
        EXPECT_EQ(part.nodes, n - part.pairs);
    }
}

// stats->etree / etree_depth are the returned ordering's etree, postorder,
// column counts and depth, exactly as analyse() computes them (analyse()
// adopts them instead of recomputing), and a caller-supplied adjacency of A
// gives the same result as the one matching_order_from builds itself.
TEST(MetisOrdering, MatchingOrderStatsCarryTheOrderingsEtree)
{
    const int g = 30, m = 200, n = g * g + m;
    const smf::CscLower A = kkt_grid(g, m);
    const smf::MatchingResult mr = smf::compute_symmetric_matching(A);
    ASSERT_EQ(mr.status, smf::ErrorCode::Success);
    const auto sn = static_cast<std::size_t>(n);
    std::vector<smf::Int> xadj(sn + 1), adj(smf::lower_csc_adjacency_size(n, A.col_ptr.data(), A.row_idx.data()));
    smf::lower_csc_adjacency(n, A.col_ptr.data(), A.row_idx.data(), xadj.data(), adj.data());
    for (smf::CompressedOrder how : {smf::CompressedOrder::Amd, smf::CompressedOrder::Metis,
                                     smf::CompressedOrder::Auto}) {
        std::vector<smf::Int> perm, perm2, work;
        smf::MatchingOrderStats st, st2;
        ASSERT_TRUE(smf::matching_order_from(A, mr, 0.1, perm, work, &st, how));
        ASSERT_TRUE(smf::matching_order_from(A, mr, 0.1, perm2, work, &st2, how, 1, xadj.data(), adj.data()));
        EXPECT_EQ(perm, perm2);
        EXPECT_EQ(st.etree, st2.etree);
        EXPECT_EQ(st.etree_depth, st2.etree_depth);
        EXPECT_EQ(st.flops, st2.flops);
        EXPECT_EQ(st.used, st2.used);

        std::vector<smf::Int> iperm(sn), tree(3 * sn), w(4 * sn + 1);
        for (int i = 0; i < n; ++i) iperm[static_cast<std::size_t>(perm[static_cast<std::size_t>(i)])] = i;
        smf::Int *parent = tree.data(), *post = parent + sn, *cc = post + sn;
        smf::etree_from_adjacency(n, xadj.data(), adj.data(), perm.data(), iperm.data(), parent, w.data());
        const smf::Int depth = smf::etree_postorder(n, parent, post, w.data());
        smf::filled_column_counts_from_adjacency(n, xadj.data(), adj.data(), perm.data(), iperm.data(), parent,
                                                 post, cc, w.data());
        EXPECT_EQ(st.etree, tree);
        EXPECT_EQ(st.etree_depth, depth);
        double fl = 0.0;
        for (std::size_t j = 0; j < sn; ++j) fl += static_cast<double>(cc[j]) * static_cast<double>(cc[j]);
        EXPECT_EQ(st.flops, fl);
    }
}

TEST(MetisOrdering, MatchingFallsBackWhenSingular)
{
    // Columns 0 and 1 only reach row 2: no perfect matching.
    smf::CscLower A;
    A.n = 3;
    A.col_ptr = {0, 1, 2, 3};
    A.row_idx = {2, 2, 2};
    A.values = {1.0, 1.0, 1.0};
    for (smf::CompressedOrder how : {smf::CompressedOrder::Amd, smf::CompressedOrder::Metis,
                                     smf::CompressedOrder::Auto}) {
        std::vector<smf::Int> perm, work;
        EXPECT_FALSE(smf::matching_order_lower_csc(A, 0.1, perm, work, nullptr, how));
    }
}

TEST(MetisOrdering, MatchingCutsFlopsOnNcvxqp5)
{
#if !(defined(SMF_HAS_METIS) && SMF_HAS_METIS)
    GTEST_SKIP() << "METIS not compiled in";
#endif
    namespace fs = std::filesystem;
    const fs::path mtx = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" /
                         "fixtures" / "large" / "ncvxqp5" / "ncvxqp5.mtx";
    std::error_code ec;
    if (!fs::exists(mtx, ec)) GTEST_SKIP() << "ncvxqp5 fixture not present";
    auto mm = smf::read_matrix_market(mtx.string());
    ASSERT_TRUE(mm.ok);
    smf::Info info;
    const smf::CscLower A = smf::clean_lower_csc(mm.matrix, info).clean;

    std::vector<smf::Int> perm, work;
    smf::MatchingOrderStats amd, metis, autom;
    ASSERT_TRUE(smf::matching_order_lower_csc(A, 0.1, perm, work, &amd, smf::CompressedOrder::Amd));
    ASSERT_TRUE(smf::matching_order_lower_csc(A, 0.1, perm, work, &metis, smf::CompressedOrder::Metis));
    // A METIS failure at run time (AMD kept) fails the test below.
    ASSERT_EQ(metis.used, smf::CompressedOrder::Metis);
    ASSERT_TRUE(smf::matching_order_lower_csc(A, 0.1, perm, work, &autom, smf::CompressedOrder::Auto));
    std::printf("[MatchingCutsFlops] ncvxqp5 flops: amd %.4g metis %.4g auto %.4g (ratio %.3f)\n",
                amd.flops, metis.flops, autom.flops, metis.flops / amd.flops);
    EXPECT_EQ(amd.used, smf::CompressedOrder::Amd);
    EXPECT_LT(metis.flops, 0.5 * amd.flops);
    EXPECT_EQ(autom.used, smf::CompressedOrder::Metis);
    EXPECT_DOUBLE_EQ(autom.flops, metis.flops);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
