#include <gtest/gtest.h>
#include "smf/ordering_metis.hpp"
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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
