// detail::auction_advance (solver/src/auction_matching.hpp): the
// epsilon-scaling auction on the sparse part, on its own invariants. Every
// test starts from the matcher constructor's duals on the cost graph:
// u_i = min_j c_ij, v_j = min_i (c_ij - u_i), and (warm) its greedy matching
// on tight edges.

#include "../src/auction_matching.hpp"
#include "../src/matching_internal.hpp"

#include "smf/scaling_matching.hpp"
#include "kkt_test_matrices.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace smf {
namespace {

using detail::AuctionState;
using detail::AuctionStatus;
using detail::CostGraph;
using detail::RowState;

// Everything auction_advance reads and updates, as SymmetricMatcher holds it.
struct Kernel {
    CostGraph g;
    std::vector<std::uint8_t> dense;
    std::vector<RowState> rs;
    std::vector<double> v;
    std::vector<int> row_match;
    std::vector<Int> col_to_row;
    std::vector<int> first_queue;  // free columns by decreasing degree (stable)
    AuctionState st;
    long long work = 0;

    AuctionStatus advance(const AuctionOptions &opt, long long budget = -1) {
        return detail::auction_advance(g, dense.data(), rs, v, row_match, col_to_row, first_queue, opt, st,
                                       work, budget);
    }
};

// The constructor's duals and greedy matching (no dense rows: every matrix
// here is far below 10 sqrt(n) entries per column). False if the cost graph
// fails (a column without nonzeros).
bool setup(const CscLower &A, Kernel &k) {
    if (detail::build_cost_graph(A, k.g) != ErrorCode::Success) return false;
    const int n = static_cast<int>(A.n);
    const CostGraph &g = k.g;
    k.dense.assign(static_cast<std::size_t>(n), 0);
    k.rs.assign(static_cast<std::size_t>(n), RowState{detail::kInf, detail::kInf});
    for (int j = 0; j < n; ++j)
        for (Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q) {
            double &u = k.rs[static_cast<std::size_t>(g.row[q])].u;
            u = std::min(u, g.cost[q]);
        }
    for (RowState &r : k.rs)
        if (r.u == detail::kInf) r.u = 0.0;
    k.v.assign(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        double vj = detail::kInf;
        for (Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q)
            vj = std::min(vj, g.cost[q] - k.rs[static_cast<std::size_t>(g.row[q])].u);
        k.v[static_cast<std::size_t>(j)] = vj == detail::kInf ? 0.0 : vj;
    }
    k.row_match.assign(static_cast<std::size_t>(n), -1);
    k.col_to_row.assign(static_cast<std::size_t>(n), -1);
    for (int j = 0; j < n; ++j)
        for (Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q) {
            const int i = g.row[q];
            if (k.row_match[static_cast<std::size_t>(i)] < 0 &&
                g.cost[q] - k.rs[static_cast<std::size_t>(i)].u - k.v[static_cast<std::size_t>(j)] <= 0.0) {
                k.row_match[static_cast<std::size_t>(i)] = j;
                k.col_to_row[static_cast<std::size_t>(j)] = i;
                break;
            }
        }
    k.first_queue.clear();
    for (int j = 0; j < n; ++j)
        if (k.col_to_row[static_cast<std::size_t>(j)] < 0) k.first_queue.push_back(j);
    std::stable_sort(k.first_queue.begin(), k.first_queue.end(),
                     [&g](int a, int b) { return g.ptr[a + 1] - g.ptr[a] > g.ptr[b + 1] - g.ptr[b]; });
    return true;
}

// eps-complementary slackness on every edge: c_ij - u_i - v_j >= -eps for
// v_j = min_i (c_ij - u_i), and each matched edge within eps of its
// column's minimum. Also the kernel's own v: feasible (c - u - v >= 0 up to
// rounding), the duals the exact repair starts from.
void expect_eps_cs(const Kernel &k, double eps) {
    const CostGraph &g = k.g;
    const int n = static_cast<int>(k.v.size());
    for (int j = 0; j < n; ++j) {
        double vj = detail::kInf;
        for (Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q)
            vj = std::min(vj, g.cost[q] - k.rs[static_cast<std::size_t>(g.row[q])].u);
        const Int r = k.col_to_row[static_cast<std::size_t>(j)];
        for (Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q) {
            const double cu = g.cost[q] - k.rs[static_cast<std::size_t>(g.row[q])].u;
            ASSERT_GE(cu - vj, -eps - 1e-12) << "j=" << j;
            ASSERT_GE(cu - k.v[static_cast<std::size_t>(j)], -1e-12) << "j=" << j;
            if (g.row[q] == r) ASSERT_LE(cu - vj, eps + 1e-12) << "j=" << j;
        }
    }
}

// Sum of log|a_ij| over the matched entries (duplicates summed).
double log_product(const CostGraph &g, const std::vector<Int> &col_to_row) {
    double s = 0.0;
    const int n = static_cast<int>(col_to_row.size());
    for (int j = 0; j < n; ++j) {
        const Int r = col_to_row[static_cast<std::size_t>(j)];
        for (Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q)
            if (g.row[q] == r) s += std::log(g.col_max[static_cast<std::size_t>(j)]) - g.cost[q];
    }
    return s;
}

bool all_matched(const std::vector<Int> &col_to_row) {
    return std::all_of(col_to_row.begin(), col_to_row.end(), [](Int i) { return i >= 0; });
}

// Random symmetric matrix in lower CSC form, as Matching.OptimalAgainstBruteForce
// (test_refinement.cpp) builds it.
CscLower make_random_sym(int n, double density, bool with_diag, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    CscLower A;
    A.n = n;
    A.col_ptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            if ((i == j && with_diag) || (i != j && unif(rng) < density)) {
                const double mag = std::pow(10.0, -4.0 + 8.0 * unif(rng));
                A.row_idx.push_back(i);
                A.values.push_back(unif(rng) < 0.5 ? -mag : mag);
            }
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

template <class T>
bool bitwise_equal(const std::vector<T> &a, const std::vector<T> &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

} // namespace

TEST(AuctionMatching, KernelReachesEpsCsAndPerfectMatching) {
    for (unsigned seed = 1; seed <= 3; ++seed) {
        const CscLower A = opf_like_kkt(3000, 1200, seed);
        Kernel k;
        ASSERT_TRUE(setup(A, k));
        const AuctionOptions opt;
        ASSERT_EQ(k.advance(opt), AuctionStatus::Done) << "seed=" << seed;
        ASSERT_TRUE(all_matched(k.col_to_row));
        for (int j = 0; j < static_cast<int>(A.n); ++j)
            ASSERT_EQ(k.row_match[static_cast<std::size_t>(k.col_to_row[static_cast<std::size_t>(j)])], j);
        expect_eps_cs(k, opt.eps_final);
        EXPECT_GT(k.work, 0);

        const MatchingResult sp = compute_symmetric_matching(A);
        ASSERT_EQ(sp.status, ErrorCode::Success);
        EXPECT_NEAR(log_product(k.g, k.col_to_row), log_product(k.g, sp.col_to_row),
                    static_cast<double>(A.n) * opt.eps_final)
            << "seed=" << seed;
    }
}

TEST(AuctionMatching, KernelOptimalAgainstBruteForce) {
    int compared = 0, capped = 0;
    for (unsigned seed = 1; seed <= 60; ++seed) {
        const int n = 3 + static_cast<int>(seed % 5);  // 3..7
        const CscLower A = make_random_sym(n, 0.45, seed % 3 == 0, seed);
        Kernel k;
        // A column without nonzeros: the cost graph already reports it.
        if (!setup(A, k)) continue;
        const CostGraph &g = k.g;
        std::vector<double> c(static_cast<std::size_t>(n) * n, detail::kInf);
        for (int j = 0; j < n; ++j)
            for (Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q) c[static_cast<std::size_t>(g.row[q]) * n + j] = g.cost[q];

        // Brute force over perm[j] = row matched to column j.
        std::vector<int> perm(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), 0);
        double best = detail::kInf;
        do {
            double s = 0.0;
            for (int j = 0; j < n; ++j) s += c[static_cast<std::size_t>(perm[static_cast<std::size_t>(j)]) * n + j];
            best = std::min(best, s);
        } while (std::next_permutation(perm.begin(), perm.end()));

        AuctionOptions opt;
        opt.eps_final = 1e-9 / n;
        const AuctionStatus status = k.advance(opt);
        if (best == detail::kInf) {
            EXPECT_EQ(status, AuctionStatus::Capped) << "seed=" << seed;
            ++capped;
            continue;
        }
        ASSERT_EQ(status, AuctionStatus::Done) << "seed=" << seed;
        double got = 0.0;
        for (int j = 0; j < n; ++j) {
            const Int i = k.col_to_row[static_cast<std::size_t>(j)];
            ASSERT_GE(i, 0);
            ASSERT_EQ(k.row_match[static_cast<std::size_t>(i)], j);
            got += c[static_cast<std::size_t>(i) * n + j];
        }
        EXPECT_NEAR(got, best, 1e-9 * (1.0 + std::abs(best))) << "seed=" << seed << " n=" << n;
        ++compared;
    }
    EXPECT_GT(compared, 30);
    EXPECT_GT(capped, 0);
}

TEST(AuctionMatching, KernelPausesAtBidBoundaries) {
    for (unsigned seed = 1; seed <= 3; ++seed) {
        const CscLower A = opf_like_kkt(3000, 1200, seed);
        const AuctionOptions opt;
        Kernel one;
        ASSERT_TRUE(setup(A, one));
        ASSERT_EQ(one.advance(opt), AuctionStatus::Done);

        Kernel k;
        ASSERT_TRUE(setup(A, k));
        AuctionStatus status = AuctionStatus::Paused;
        int pauses = 0;
        while (status == AuctionStatus::Paused) {
            const long long budget = k.work + 50;
            status = k.advance(opt, budget);
            if (status == AuctionStatus::Paused) {
                ASSERT_GE(k.work, budget);
                ++pauses;
            }
        }
        ASSERT_EQ(status, AuctionStatus::Done);
        EXPECT_GT(pauses, 10);
        EXPECT_EQ(k.work, one.work);
        std::vector<double> u1, u2;
        for (const RowState &r : one.rs) u1.push_back(r.u);
        for (const RowState &r : k.rs) u2.push_back(r.u);
        EXPECT_TRUE(bitwise_equal(u1, u2)) << "seed=" << seed;
        EXPECT_TRUE(bitwise_equal(one.v, k.v)) << "seed=" << seed;
        EXPECT_TRUE(bitwise_equal(one.row_match, k.row_match)) << "seed=" << seed;
        EXPECT_TRUE(bitwise_equal(one.col_to_row, k.col_to_row)) << "seed=" << seed;
    }
}

TEST(AuctionMatching, KernelColdAndWarmStartsAgreeOnProduct) {
    for (unsigned seed = 1; seed <= 3; ++seed) {
        const CscLower A = opf_like_kkt(3000, 1200, seed);
        AuctionOptions warm, cold;
        cold.warm = false;
        Kernel kw, kc;
        ASSERT_TRUE(setup(A, kw));
        ASSERT_TRUE(setup(A, kc));
        ASSERT_EQ(kw.advance(warm), AuctionStatus::Done) << "seed=" << seed;
        ASSERT_EQ(kc.advance(cold), AuctionStatus::Done) << "seed=" << seed;
        ASSERT_TRUE(all_matched(kw.col_to_row));
        ASSERT_TRUE(all_matched(kc.col_to_row));
        expect_eps_cs(kc, cold.eps_final);
        EXPECT_NEAR(log_product(kw.g, kw.col_to_row), log_product(kc.g, kc.col_to_row),
                    static_cast<double>(A.n) * warm.eps_final)
            << "seed=" << seed;
    }
}

} // namespace smf
