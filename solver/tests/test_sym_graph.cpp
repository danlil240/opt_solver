#include <gtest/gtest.h>
#include "smf/sym_graph.hpp"
#include <algorithm>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: build CscLower from (row, col, val) triplets
// ---------------------------------------------------------------------------
static smf::CscLower make_csc(int n, std::vector<std::tuple<int,int,double>> entries)
{
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n + 1), 0);

    std::sort(entries.begin(), entries.end(),
        [](const auto& a, const auto& b) {
            return std::get<1>(a) < std::get<1>(b) ||
                   (std::get<1>(a) == std::get<1>(b) && std::get<0>(a) < std::get<0>(b));
        });

    for (auto [i, j, v] : entries) {
        A.col_ptr[static_cast<std::size_t>(j + 1)]++;
    }
    for (int k = 0; k < n; ++k) {
        A.col_ptr[static_cast<std::size_t>(k + 1)] += A.col_ptr[static_cast<std::size_t>(k)];
    }

    A.row_idx.resize(entries.size());
    A.values.resize(entries.size());
    std::vector<int> pos(A.col_ptr.begin(), A.col_ptr.begin() + n);

    for (auto [i, j, v] : entries) {
        A.row_idx[static_cast<std::size_t>(pos[j])] = i;
        A.values[static_cast<std::size_t>(pos[j]++)] = v;
    }
    return A;
}

// ---------------------------------------------------------------------------
// Test 1: Triangle3x3
// Lower entries: (1,0), (2,0), (2,1)  -- 3 off-diagonal undirected edges
// ---------------------------------------------------------------------------
TEST(SymGraph, Triangle3x3)
{
    auto A = make_csc(3, {
        {0, 0, 1.0},  // diagonal col 0
        {1, 0, 1.0},  // edge 0-1
        {2, 0, 1.0},  // edge 0-2
        {1, 1, 1.0},  // diagonal col 1
        {2, 1, 1.0},  // edge 1-2
        {2, 2, 1.0},  // diagonal col 2
    });

    auto [xadj, adjncy] = smf::build_symmetric_adjacency(A);

    ASSERT_EQ(static_cast<int>(xadj.size()), 4);
    EXPECT_EQ(xadj[3], 6);  // 3 undirected edges -> 6 directed

    // Node 0: adjacency {1,2}
    std::vector<smf::Int> adj0(adjncy.begin() + xadj[0], adjncy.begin() + xadj[1]);
    EXPECT_EQ(adj0, (std::vector<smf::Int>{1, 2}));

    // Node 1: adjacency {0,2}
    std::vector<smf::Int> adj1(adjncy.begin() + xadj[1], adjncy.begin() + xadj[2]);
    EXPECT_EQ(adj1, (std::vector<smf::Int>{0, 2}));

    // Node 2: adjacency {0,1}
    std::vector<smf::Int> adj2(adjncy.begin() + xadj[2], adjncy.begin() + xadj[3]);
    EXPECT_EQ(adj2, (std::vector<smf::Int>{0, 1}));

    // All rows sorted (already checked by equality, but be explicit)
    for (int v = 0; v < 3; ++v) {
        EXPECT_TRUE(std::is_sorted(adjncy.begin() + xadj[v], adjncy.begin() + xadj[v + 1]));
    }
}

// ---------------------------------------------------------------------------
// Test 2: DiagonalOnly - 4x4
// ---------------------------------------------------------------------------
TEST(SymGraph, DiagonalOnly)
{
    auto A = make_csc(4, {
        {0, 0, 1.0},
        {1, 1, 1.0},
        {2, 2, 1.0},
        {3, 3, 1.0},
    });

    auto [xadj, adjncy] = smf::build_symmetric_adjacency(A);

    ASSERT_EQ(static_cast<int>(xadj.size()), 5);
    EXPECT_EQ(xadj[4], 0);   // no off-diagonal entries
    EXPECT_TRUE(adjncy.empty());
}

// ---------------------------------------------------------------------------
// Test 3: DisconnectedBlocks - two 2x2 dense lower blocks
// Block 0: nodes {0,1}, Block 1: nodes {2,3}
// ---------------------------------------------------------------------------
TEST(SymGraph, DisconnectedBlocks)
{
    auto A = make_csc(4, {
        {0, 0, 1.0},
        {1, 0, 1.0},  // edge 0-1
        {1, 1, 1.0},
        {2, 2, 1.0},
        {3, 2, 1.0},  // edge 2-3
        {3, 3, 1.0},
    });

    auto [xadj, adjncy] = smf::build_symmetric_adjacency(A);

    // Each block has 1 undirected edge -> 2 directed each -> 4 total
    EXPECT_EQ(xadj[4], 4);

    // Node 0 adjacent to 1 only
    std::vector<smf::Int> adj0(adjncy.begin() + xadj[0], adjncy.begin() + xadj[1]);
    EXPECT_EQ(adj0, (std::vector<smf::Int>{1}));

    // Node 1 adjacent to 0 only
    std::vector<smf::Int> adj1(adjncy.begin() + xadj[1], adjncy.begin() + xadj[2]);
    EXPECT_EQ(adj1, (std::vector<smf::Int>{0}));

    // Node 2 adjacent to 3 only
    std::vector<smf::Int> adj2(adjncy.begin() + xadj[2], adjncy.begin() + xadj[3]);
    EXPECT_EQ(adj2, (std::vector<smf::Int>{3}));

    // Node 3 adjacent to 2 only
    std::vector<smf::Int> adj3(adjncy.begin() + xadj[3], adjncy.begin() + xadj[4]);
    EXPECT_EQ(adj3, (std::vector<smf::Int>{2}));
}

// ---------------------------------------------------------------------------
// Test 4: WithExplicitZeroDiag - no self-loops
// 3x3: diagonal everywhere but no off-diagonal in column 2
// ---------------------------------------------------------------------------
TEST(SymGraph, WithExplicitZeroDiag)
{
    auto A = make_csc(3, {
        {0, 0, 1.0},
        {1, 0, 1.0},  // edge 0-1
        {1, 1, 1.0},
        {2, 2, 0.0},  // explicit (possibly zero) diagonal entry only
    });

    auto [xadj, adjncy] = smf::build_symmetric_adjacency(A);

    // 1 undirected edge (0-1) -> 2 directed
    EXPECT_EQ(xadj[3], 2);

    // Verify no self-loops
    for (std::size_t idx = 0; idx < adjncy.size(); ++idx) {
        // Find which node owns this position
        int owner = -1;
        for (int v = 0; v < 3; ++v) {
            if (static_cast<int>(idx) >= xadj[v] && static_cast<int>(idx) < xadj[v + 1]) {
                owner = v;
                break;
            }
        }
        EXPECT_NE(adjncy[idx], owner);
    }
}

// ---------------------------------------------------------------------------
// Test 5: StructuralRank_Full - n x n lower triangular full pattern
// ---------------------------------------------------------------------------
TEST(SymGraph, StructuralRank_Full)
{
    const int n = 5;
    std::vector<std::tuple<int,int,double>> entries;
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            entries.emplace_back(i, j, 1.0);
        }
    }
    auto A = make_csc(n, entries);
    EXPECT_EQ(smf::structural_rank(A), n);
}

// ---------------------------------------------------------------------------
// Test 6: StructuralRank_Diagonal - diagonal only
// ---------------------------------------------------------------------------
TEST(SymGraph, StructuralRank_Diagonal)
{
    const int n = 6;
    std::vector<std::tuple<int,int,double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 1.0);
    }
    auto A = make_csc(n, entries);
    EXPECT_EQ(smf::structural_rank(A), n);
}

// ---------------------------------------------------------------------------
// Test 7: StructuralRank_MissingDiag - only entry (2,0)
// Greedy: column 0 matches row 2. Columns 1,2 have no entries -> matched=1.
// ---------------------------------------------------------------------------
TEST(SymGraph, StructuralRank_MissingDiag)
{
    auto A = make_csc(3, {
        {2, 0, 1.0},  // only entry
    });
    smf::Int sr = smf::structural_rank(A);
    EXPECT_LE(sr, 2);
    EXPECT_GE(sr, 0);
    EXPECT_EQ(sr, 1);  // greedy: column 0 gets row 2; cols 1,2 have no entries
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
