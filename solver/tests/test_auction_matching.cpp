// detail::auction_advance (solver/src/auction_matching.hpp): the
// epsilon-scaling auction on the sparse part, on its own invariants; then
// SymmetricMatcher with MatchingAlgorithm::Auction (the auction plus its
// exact shortest-path repair) against the ShortestPath matcher. Every
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
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
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
// rounding), the duals the exact repair starts from. Columns and rows
// marked dense are skipped, as the kernel skips them.
void expect_eps_cs(const Kernel &k, double eps) {
    const CostGraph &g = k.g;
    const int n = static_cast<int>(k.v.size());
    const auto dense = [&k](int i) { return k.dense[static_cast<std::size_t>(i)] != 0; };
    for (int j = 0; j < n; ++j) {
        if (dense(j)) continue;
        double vj = detail::kInf;
        for (Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q)
            if (!dense(g.row[q])) vj = std::min(vj, g.cost[q] - k.rs[static_cast<std::size_t>(g.row[q])].u);
        const Int r = k.col_to_row[static_cast<std::size_t>(j)];
        for (Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q) {
            if (dense(g.row[q])) continue;
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

// The matcher's dense threshold (scaling_matching.cpp): columns of the full
// pattern with more than max(16, 10 sqrt(n)) entries, for n > 50.
int count_dense(const CscLower &A) {
    CostGraph g;
    if (A.n <= 50 || detail::build_cost_graph(A, g) != ErrorCode::Success) return 0;
    const double threshold = std::max(16.0, 10.0 * std::sqrt(static_cast<double>(A.n)));
    int dense = 0;
    for (Int j = 0; j < A.n; ++j) dense += static_cast<double>(g.ptr[j + 1] - g.ptr[j]) > threshold ? 1 : 0;
    return dense;
}

// Both matchings perfect and of the same product (summed duplicates).
void same_optimum(const CscLower &A, const MatchingResult &a, const MatchingResult &b) {
    ASSERT_EQ(a.status, ErrorCode::Success);
    ASSERT_EQ(b.status, ErrorCode::Success);
    ASSERT_EQ(a.matched_rank, A.n);
    ASSERT_EQ(b.matched_rank, A.n);
    CostGraph g;
    ASSERT_EQ(detail::build_cost_graph(A, g), ErrorCode::Success);
    const double pa = log_product(g, a.col_to_row), pb = log_product(g, b.col_to_row);
    EXPECT_NEAR(pa, pb, 1e-9 * std::abs(pb) + 1e-9);
}

// The scaling's guarantees, as MatchingTrial.DegreeSearchOrderIsAMaximumProductMatching
// checks them: no scaled entry above 1, every matched one 1 (on the merged
// entries, |a_ij| = col_max_j exp(-cost)).
void scaling_bounds(const CscLower &A, const MatchingResult &r) {
    CostGraph g;
    ASSERT_EQ(detail::build_cost_graph(A, g), ErrorCode::Success);
    ASSERT_EQ(r.scale.size(), static_cast<std::size_t>(A.n));
    for (Int j = 0; j < A.n; ++j) {
        for (Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q) {
            const int i = g.row[q];
            const double a = g.col_max[static_cast<std::size_t>(j)] * std::exp(-g.cost[q]);
            const double s = r.scale[static_cast<std::size_t>(i)] * a * r.scale[static_cast<std::size_t>(j)];
            ASSERT_LE(s, 1.0 + 1e-10) << "i=" << i << " j=" << j;
            if (r.col_to_row[static_cast<std::size_t>(j)] == i) ASSERT_NEAR(s, 1.0, 1e-10) << "j=" << j;
        }
    }
}

// A boyd1-like KKT whose sparse part needs the auction: a tridiagonal block
// over n_x variables with magnitudes 10^[-2, 2] from a seeded LCG (the
// diagonal is not the matching) and k dense constraint rows with zero
// diagonal, row r holding every variable j with j % k == r.
CscLower dense_row_kkt(Int n_x, Int k, unsigned seed) {
    std::uint64_t st = seed;
    const auto mag = [&st] {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return std::pow(10.0, 4.0 * (static_cast<double>(st >> 11) / 9007199254740992.0) - 2.0);
    };
    CscLower A;
    A.n = n_x + k;
    A.col_ptr.push_back(0);
    for (Int j = 0; j < n_x; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(mag());
        if (j + 1 < n_x) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(-mag());
        }
        A.row_idx.push_back(n_x + j % k);
        A.values.push_back(mag());
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    for (Int r = 0; r < k; ++r) A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    return A;
}

// Lower CSC from (row, col, value) entries with row >= col, in input order
// per column (duplicates and zeros kept).
CscLower from_entries(int n, std::vector<std::tuple<int, int, double>> e) {
    std::stable_sort(e.begin(), e.end(), [](const auto &a, const auto &b) { return std::get<1>(a) < std::get<1>(b); });
    CscLower A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto &[i, j, x] : e) {
        A.row_idx.push_back(i);
        A.values.push_back(x);
        ++A.col_ptr[static_cast<std::size_t>(j) + 1];
    }
    for (int j = 0; j < n; ++j) A.col_ptr[static_cast<std::size_t>(j) + 1] += A.col_ptr[static_cast<std::size_t>(j)];
    return A;
}

MatchingResult run_auction(const CscLower &A, const AuctionOptions *opt = nullptr) {
    SymmetricMatcher m(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Auction, opt);
    EXPECT_TRUE(m.advance());
    return m.current();
}

void expect_bitwise(const MatchingResult &a, const MatchingResult &b) {
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.matched_rank, b.matched_rank);
    EXPECT_TRUE(bitwise_equal(a.col_to_row, b.col_to_row));
    EXPECT_TRUE(bitwise_equal(a.row_to_col, b.row_to_col));
    EXPECT_TRUE(bitwise_equal(a.scale, b.scale));
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
        expect_eps_cs(kw, warm.eps_final);
        expect_eps_cs(kc, cold.eps_final);
        EXPECT_NEAR(log_product(kw.g, kw.col_to_row), log_product(kc.g, kc.col_to_row),
                    static_cast<double>(A.n) * warm.eps_final)
            << "seed=" << seed;
    }
}

// A sparse column whose every entry lies in a dense row has nothing to bid
// on: the kernel drops it (it stays free for the searches) and finishes the
// other columns, then reports Capped, again on every later call. Resuming
// from bid-boundary pauses gives the one-shot state bitwise.
TEST(AuctionMatching, KernelDropsAColumnWithoutSparseRows) {
    // Lower triangle, n = 6: column 0 only meets row 5, the dense one;
    // columns 1-4 form a path with diagonals, the cheap edges off-diagonal.
    CscLower A;
    A.n = 6;
    A.col_ptr = {0, 1, 3, 5, 7, 8, 9};
    A.row_idx = {5, 1, 2, 2, 3, 3, 4, 4, 5};
    A.values = {1.0, 1e-3, 1.0, 1e-3, 1.0, 1e-3, 1.0, 1e-3, 1.0};
    const auto fresh = [&A](Kernel &k) {
        ASSERT_TRUE(setup(A, k));
        // Mark 5 dense and start every sparse column free (as a cold start
        // does), column 0 first in the queue.
        k.dense[5] = 1;
        std::fill(k.row_match.begin(), k.row_match.end(), -1);
        std::fill(k.col_to_row.begin(), k.col_to_row.end(), -1);
        k.first_queue = {0, 1, 2, 3, 4};
    };
    const AuctionOptions opt;
    Kernel one;
    fresh(one);
    ASSERT_EQ(one.advance(opt), AuctionStatus::Capped);
    EXPECT_EQ(one.col_to_row[0], -1);
    for (int j = 1; j <= 4; ++j) EXPECT_GE(one.col_to_row[static_cast<std::size_t>(j)], 0) << "j=" << j;
    expect_eps_cs(one, opt.eps_final);
    const long long work_done = one.work;
    const std::vector<Int> c2r = one.col_to_row;
    EXPECT_EQ(one.advance(opt), AuctionStatus::Capped);
    EXPECT_EQ(one.work, work_done);
    EXPECT_EQ(one.col_to_row, c2r);

    Kernel k;
    fresh(k);
    AuctionStatus status = AuctionStatus::Paused;
    int pauses = 0;
    while (status == AuctionStatus::Paused) {
        status = k.advance(opt, k.work + 1);
        if (status == AuctionStatus::Paused) ++pauses;
    }
    EXPECT_EQ(status, AuctionStatus::Capped);
    EXPECT_GT(pauses, 2);
    EXPECT_EQ(k.work, one.work);
    EXPECT_TRUE(bitwise_equal(one.v, k.v));
    EXPECT_TRUE(bitwise_equal(one.col_to_row, k.col_to_row));
    EXPECT_TRUE(bitwise_equal(one.row_match, k.row_match));
}

// SymmetricMatcher under MatchingAlgorithm::Auction reaches a maximum-product
// matching with the scaling guarantees: on the scaled OPF KKT (phase 1
// only) and on a KKT with 3 dense rows (phases 1-3).
TEST(AuctionMatching, SameOptimumAsShortestPathOnKkt) {
    std::vector<CscLower> cases;
    for (unsigned seed = 1; seed <= 3; ++seed) cases.push_back(scaled_opf_kkt(3000, 1200, seed, 4.0));
    cases.push_back(dense_row_kkt(6000, 3, 5));
    ASSERT_EQ(count_dense(cases.back()), 3);  // phases 2-3 run
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const CscLower &A = cases[c];
        SymmetricMatcher m(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Auction);
        // The auction has free sparse columns to bid for.
        ASSERT_LT(m.initial().matched_rank, A.n - count_dense(A)) << "case " << c;
        ASSERT_TRUE(m.advance()) << "case " << c;
        const MatchingResult r = m.current();
        same_optimum(A, r, compute_symmetric_matching(A));
        scaling_bounds(A, r);
        EXPECT_GT(m.work(), 0) << "case " << c;
    }
}

TEST(AuctionMatching, InitialIsTheShortestPathInitial) {
    for (const CscLower &A : {scaled_opf_kkt(3000, 1200, 1, 4.0), dense_row_kkt(6000, 3, 5)}) {
        SymmetricMatcher a(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Auction);
        SymmetricMatcher sp(A, MatchingSearchOrder::Degree);
        const MatchingResult ia = a.initial(), isp = sp.initial();
        EXPECT_TRUE(bitwise_equal(ia.col_to_row, isp.col_to_row));
        EXPECT_TRUE(bitwise_equal(ia.scale, isp.scale));
        EXPECT_EQ(a.diagonal(), sp.diagonal());
        EXPECT_EQ(a.repaired(), 0);
        EXPECT_EQ(sp.repaired(), 0);
    }
}

// Pausing every 50 edges (inside the auction, the repair's searches and
// phase 3) and resuming gives the one-shot matching, scaling and work.
TEST(AuctionMatching, ResumedAdvanceIsOneShot) {
    for (const CscLower &A : {scaled_opf_kkt(3000, 1200, 1, 4.0), dense_row_kkt(6000, 3, 5)}) {
        SymmetricMatcher one(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Auction);
        ASSERT_TRUE(one.advance());
        SymmetricMatcher m(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Auction);
        int pauses = 0;
        for (long long budget = 0; !m.advance(budget); budget = m.work() + 50) ++pauses;
        EXPECT_GT(pauses, 1);
        expect_bitwise(m.current(), one.current());
        EXPECT_EQ(m.work(), one.work());
        EXPECT_EQ(m.repaired(), one.repaired());
    }
}

TEST(AuctionMatching, StructurallySingularIsReported) {
    // Matching.StructurallySingular: row/column 1 is empty.
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 1, 1, 2};
    A.row_idx = {0, 2};
    A.values = {4.0, 5.0};
    // Matching.StructurallySingularFailsAfterSearches: columns 0 and 1
    // only reach row 2 (a price war in the auction).
    CscLower B;
    B.n = 3;
    B.col_ptr = {0, 1, 2, 3};
    B.row_idx = {2, 2, 2};
    B.values = {1.0, 1.0, 1.0};
    for (const CscLower *M : std::vector<const CscLower *>{&A, &B}) {
        SymmetricMatcher sp(*M);
        ASSERT_TRUE(sp.advance());
        SymmetricMatcher m(*M, MatchingSearchOrder::Degree, MatchingAlgorithm::Auction);
        EXPECT_TRUE(m.advance());
        EXPECT_EQ(m.status(), ErrorCode::SingularMatrix);
        EXPECT_EQ(m.status(), sp.status());
        EXPECT_EQ(m.current().status, ErrorCode::SingularMatrix);
        EXPECT_LT(m.current().matched_rank, 3);
    }
}

// Columns 0 and 1 (zero diagonal) share the single sparse row 2, and each
// also lies in a dense row: the sparse part has no perfect matching, the
// auction caps its price war, and phase 3 matches one of them to a dense
// row. The scaling stays bounded (the war's prices do not reach it).
TEST(AuctionMatching, InfeasibleSparsePartFinishesInPhase3) {
    const int n_x = 2000, k = 3;
    std::vector<std::tuple<int, int, double>> e;
    for (int j = 0; j < n_x; ++j) {
        if (j >= 2) e.emplace_back(j, j, 1.0 + 0.25 * (j % 7));
        e.emplace_back(n_x + j % k, j, 0.5 + 0.125 * (j % 5));
    }
    e.emplace_back(2, 0, 3.0);
    e.emplace_back(2, 1, 2.0);
    const CscLower A = from_entries(n_x + k, e);
    ASSERT_EQ(count_dense(A), k);
    SymmetricMatcher m(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Auction);
    ASSERT_TRUE(m.advance());
    EXPECT_GE(m.repaired(), 1);  // a column the auction could not match
    const MatchingResult r = m.current();
    same_optimum(A, r, compute_symmetric_matching(A));
    scaling_bounds(A, r);
}

// Entries spanning 40 decades: the same product and bounds, and no price war
// (eps0 comes from the largest cost): the auction scans at most 1.5x the
// edges it scans at a 4-decade spread of the same matrix. (Not "within 4x
// the Degree-order searches": those scan 5079 edges here, under one pass
// over the ~10k-edge sparse part, while the auction's start, phase passes
// and repair are full passes.)
TEST(AuctionMatching, WideMagnitudeRange) {
    const CscLower A = scaled_opf_kkt(3000, 1200, 4, 40.0);
    SymmetricMatcher m(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Auction);
    ASSERT_TRUE(m.advance());
    const MatchingResult r = m.current();
    same_optimum(A, r, compute_symmetric_matching(A));
    scaling_bounds(A, r);
    SymmetricMatcher narrow(scaled_opf_kkt(3000, 1200, 4, 4.0), MatchingSearchOrder::Degree,
                            MatchingAlgorithm::Auction);
    ASSERT_TRUE(narrow.advance());
    EXPECT_LE(static_cast<double>(m.work()), 1.5 * static_cast<double>(narrow.work()))
        << "wide " << m.work() << " narrow " << narrow.work();
}

TEST(AuctionMatching, SingleEntryColumnsAndUncleanedInput) {
    // A chain of single off-diagonal columns: column 2t holds only (2t+1, 2t),
    // column 2t+1 its own diagonal and (2t+2, 2t+1).
    std::vector<std::tuple<int, int, double>> e;
    const int n = 400;
    for (int j = 0; j < n; j += 2) {
        e.emplace_back(j + 1, j, 1.0 + 0.01 * j);
        e.emplace_back(j + 1, j + 1, 0.5);
        if (j + 2 < n) e.emplace_back(j + 2, j + 1, 3.0);
    }
    const CscLower chain = from_entries(n, e);
    // Matching.ExplicitZerosKeepLaterEntries: explicit zeros at (0,0) and
    // (2,0).
    CscLower zeros;
    zeros.n = 3;
    zeros.col_ptr = {0, 3, 4, 5};
    zeros.row_idx = {0, 1, 2, 2, 2};
    zeros.values = {0.0, 1.0, 0.0, 1.0, 1.0};
    // Duplicates summed: (1,0) is 0.25 + 0.75 = 1, (2,2) is 2 - 1.5 = 0.5,
    // and (1,1) 3 - 3 = 0 (dropped).
    const CscLower dups = from_entries(
        3, {{1, 0, 0.25}, {0, 0, 0.5}, {1, 0, 0.75}, {1, 1, 3.0}, {2, 1, 4.0}, {1, 1, -3.0}, {2, 2, 2.0}, {2, 2, -1.5}});
    for (const CscLower *A : std::vector<const CscLower *>{&chain, &zeros, &dups}) {
        const MatchingResult r = run_auction(*A);
        same_optimum(*A, r, compute_symmetric_matching(*A));
        scaling_bounds(*A, r);
    }
}

TEST(AuctionMatching, EmptyAndTiny) {
    CscLower empty;
    empty.n = 0;
    empty.col_ptr = {0};
    CscLower diag5;
    diag5.n = 5;
    diag5.col_ptr = {0, 1, 2, 3, 4, 5};
    diag5.row_idx = {0, 1, 2, 3, 4};
    diag5.values = {1, 1, 1, 1, 1};
    for (const CscLower *A : std::vector<const CscLower *>{&empty, &diag5}) {
        SymmetricMatcher sp(*A, MatchingSearchOrder::Degree);
        ASSERT_TRUE(sp.advance());
        const MatchingResult r = run_auction(*A);
        EXPECT_EQ(r.status, ErrorCode::Success);
        expect_bitwise(r, sp.current());
    }
}

// Invalid options fall back to the defaults (the public boundary validates
// them), so the auction still ends with the optimum.
TEST(AuctionMatching, InvalidOptionsFallBackToDefaults) {
    const CscLower A = scaled_opf_kkt(3000, 1200, 2, 4.0);
    const MatchingResult ref = run_auction(A);
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    std::vector<AuctionOptions> bad(6);
    bad[0].eps_factor = 1.0;
    bad[1].eps_factor = nan;
    bad[2].eps_final = 0.0;
    bad[3].eps_final = inf;
    bad[4].bid_cap = -1.0;
    bad[5].bid_cap = nan;
    for (const AuctionOptions &o : bad) expect_bitwise(run_auction(A, &o), ref);
    // Out of range but valid: clamped, and still the optimum.
    AuctionOptions extreme;
    extreme.eps_factor = 1.0 + 1e-12;
    extreme.eps_final = 1e-300;
    extreme.bid_cap = 1e300;
    const MatchingResult r = run_auction(A, &extreme);
    same_optimum(A, r, compute_symmetric_matching(A));
}

// Random KKTs whose sparse part has no perfect matching (30% of the
// variables lack a diagonal, off-diagonals of 6 decades): the auction's
// price wars must not reach the scaling. Before repair() reset the war's
// rows, 6 of these 12 seeds failed the scaling bounds (scales clamped at
// exp(+-340)); the products were already the optimum.
TEST(AuctionMatching, PriceWarsDoNotReachTheScaling) {
    int deficient = 0;
    for (unsigned seed = 1; seed <= 12; ++seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        const int n_x = 2000, k = 1 + static_cast<int>(seed % 3);
        std::vector<std::tuple<int, int, double>> e;
        for (int j = 0; j < n_x; ++j) {
            if (unif(rng) < 0.7) e.emplace_back(j, j, std::pow(10.0, 6.0 * unif(rng) - 3.0));
            const int t = 1 + static_cast<int>(unif(rng) * 2.0);
            for (int q = 0; q < t; ++q) {
                const int i = static_cast<int>(unif(rng) * n_x);
                if (i != j) e.emplace_back(std::max(i, j), std::min(i, j), std::pow(10.0, 6.0 * unif(rng) - 3.0));
            }
            e.emplace_back(n_x + j % k, j, std::pow(10.0, 2.0 * unif(rng) - 1.0));
        }
        const CscLower A = from_entries(n_x + k, e);
        ASSERT_EQ(count_dense(A), k) << "seed " << seed;
        SymmetricMatcher m(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Auction);
        ASSERT_TRUE(m.advance());
        const MatchingResult r = m.current(), ref = compute_symmetric_matching(A);
        EXPECT_EQ(r.status, ref.status) << "seed " << seed;
        if (ref.status != ErrorCode::Success) continue;
        ++deficient;
        same_optimum(A, r, ref);
        scaling_bounds(A, r);
    }
    EXPECT_GE(deficient, 8);
}

// ---- MatchingAlgorithm::Hybrid: Degree searches, then (past
// switch_passes of phase-1 search work) the auction on the free columns
// left, the exact repair and the remaining searches. ----

namespace {

// The Hybrid defaults with the switch moved to `passes`.
AuctionOptions hybrid_at(double passes) {
    AuctionOptions o = hybrid_auction_options();
    o.switch_passes = passes;
    return o;
}

// The random deficient KKTs of AuctionMatching.PriceWarsDoNotReachTheScaling.
CscLower deficient_kkt(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const int n_x = 2000, k = 1 + static_cast<int>(seed % 3);
    std::vector<std::tuple<int, int, double>> e;
    for (int j = 0; j < n_x; ++j) {
        if (unif(rng) < 0.7) e.emplace_back(j, j, std::pow(10.0, 6.0 * unif(rng) - 3.0));
        const int t = 1 + static_cast<int>(unif(rng) * 2.0);
        for (int q = 0; q < t; ++q) {
            const int i = static_cast<int>(unif(rng) * n_x);
            if (i != j) e.emplace_back(std::max(i, j), std::min(i, j), std::pow(10.0, 6.0 * unif(rng) - 3.0));
        }
        e.emplace_back(n_x + j % k, j, std::pow(10.0, 2.0 * unif(rng) - 1.0));
    }
    return from_entries(n_x + k, e);
}

} // namespace

TEST(AuctionMatching, HybridDefaults) {
    const AuctionOptions h = hybrid_auction_options();
    EXPECT_EQ(h.eps_factor, 10.0);
    EXPECT_EQ(h.eps_final, 1e-2);
    EXPECT_TRUE(h.warm);
    EXPECT_EQ(h.eps_start, 0.1);
    EXPECT_EQ(h.switch_passes, 40.0);
    // Auction's own defaults are unchanged (eps_start 0: max cost / eps_factor).
    const AuctionOptions a;
    EXPECT_EQ(a.eps_factor, 5.0);
    EXPECT_EQ(a.eps_final, 1e-3);
    EXPECT_EQ(a.eps_start, 0.0);
}

// A switch forced early (a small fraction of a pass) on KKTs the searches
// finish in under one pass: the same optimum as compute_symmetric_matching
// and the scaling guarantees, phases 1 only and 1-3 (dense rows).
TEST(AuctionMatching, HybridSameOptimumAsShortestPath) {
    std::vector<CscLower> cases;
    for (unsigned seed = 1; seed <= 3; ++seed) cases.push_back(scaled_opf_kkt(3000, 1200, seed, 4.0));
    cases.push_back(scaled_opf_kkt(3000, 1200, 4, 40.0));
    cases.push_back(dense_row_kkt(6000, 3, 5));
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const CscLower &A = cases[c];
        for (double passes : {0.0, 0.01, 0.05}) {
            const AuctionOptions opt = hybrid_at(passes);
            SymmetricMatcher m(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid, &opt);
            ASSERT_TRUE(m.advance()) << "case " << c;
            EXPECT_TRUE(m.switched()) << "case " << c << " passes " << passes;
            const MatchingResult r = m.current();
            same_optimum(A, r, compute_symmetric_matching(A));
            scaling_bounds(A, r);
        }
    }
}

// Canonical duals (ncvxqp1-probe-report.md, candidate d): after a switch the
// auction leaves its own prices on the rows of its displacement chains, a
// different point of the optimal dual face than the searches reach (on
// ncvxqp1 rec the scale drifted by up to e^8.5 and omega failed, 4.6e-8).
// At the end of phase 1 the engine moves the sparse duals to the pointwise
// largest u below the duals at the switch that keeps them feasible and the
// matching tight. That point does not depend on which maximum-product
// matching was found, up to the switch Hybrid is the Degree ShortestPath
// matcher bitwise, and the ShortestPath searches from there end at it. So
// phases 2-3 start from ShortestPath's duals (to rounding) and the scales
// agree, also where the matchings differ (ties).
//
// Except where phase 3 meets an exact tie: on the deficient KKTs with dense
// rows (deficient_kkt seeds 9 and 11, cases 13 and 15) the duals at the
// start of phase 3 agree to 1e-15 (checked by a dump of every u, v and
// row_match), and a phase-3 search then picks another of its equal-length
// paths. Its 4-8 indices, along one alternating path ending at a dense
// row, differ by up to 1.4 in log scale; both scales are optimal and
// bounded (same_optimum, scaling_bounds). Every other switched run (42 of
// 48 today) agrees to 1e-12 relative on every index, and every run on all
// but at most 1% of them.
TEST(AuctionMatching, HybridCanonicalDualsMatchShortestPathScale) {
    std::vector<CscLower> cases;
    for (unsigned seed = 1; seed <= 3; ++seed) cases.push_back(scaled_opf_kkt(3000, 1200, seed, 4.0));
    cases.push_back(scaled_opf_kkt(3000, 1200, 4, 40.0));
    cases.push_back(dense_row_kkt(6000, 3, 5));
    const std::size_t n_full_rank = cases.size();
    for (unsigned seed = 1; seed <= 12; ++seed) cases.push_back(deficient_kkt(seed));
    int compared = 0, equal = 0, differ = 0;
    double worst = 0.0;
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const CscLower &A = cases[c];
        SymmetricMatcher sp(A, MatchingSearchOrder::Degree);
        ASSERT_TRUE(sp.advance());
        if (sp.status() != ErrorCode::Success) continue;
        const MatchingResult ref = sp.current();
        for (double passes : {0.0, 0.01, 0.05}) {
            SCOPED_TRACE("case " + std::to_string(c) + " passes " + std::to_string(passes));
            const AuctionOptions opt = hybrid_at(passes);
            SymmetricMatcher h(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid, &opt);
            ASSERT_TRUE(h.advance());
            ASSERT_TRUE(h.switched());
            const MatchingResult r = h.current();
            same_optimum(A, r, ref);
            scaling_bounds(A, r);
            int off = 0;
            for (Int i = 0; i < A.n; ++i) {
                const double ls = std::log(ref.scale[static_cast<std::size_t>(i)]);
                const double d = std::abs(std::log(r.scale[static_cast<std::size_t>(i)]) - ls);
                if (d > 1e-12 * std::max(1.0, std::abs(ls))) {
                    ++off;
                    worst = std::max(worst, d);
                }
            }
            if (c < n_full_rank) EXPECT_EQ(off, 0);
            EXPECT_LE(off, A.n / 100);
            ++compared;
            equal += off == 0 ? 1 : 0;
            differ += r.col_to_row != ref.col_to_row ? 1 : 0;
        }
    }
    EXPECT_EQ(compared, 48);
    EXPECT_GE(equal, 42);
    std::printf("[ hybrid ] canonical duals: %d switched runs, %d with the ShortestPath scale (1e-12), %d with another "
                "optimal matching, max |dlog s| elsewhere %.3g\n",
                compared, equal, differ, worst);
}

// Below the switch point Hybrid is the Degree ShortestPath matcher bitwise:
// the default 40 passes is never reached on these KKTs (each finishes in
// under a pass), nor is a switch point just above the run's own work.
TEST(AuctionMatching, HybridBitwiseWhenTheSwitchDoesNotFire) {
    std::vector<CscLower> cases;
    for (unsigned seed = 1; seed <= 3; ++seed) cases.push_back(opf_like_kkt(3000, 1200, seed));
    for (unsigned seed = 1; seed <= 3; ++seed) cases.push_back(scaled_opf_kkt(3000, 1200, seed, 4.0));
    cases.push_back(dense_row_kkt(6000, 3, 5));
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const CscLower &A = cases[c];
        SymmetricMatcher sp(A, MatchingSearchOrder::Degree);
        ASSERT_TRUE(sp.advance());
        const double pass = 2.0 * static_cast<double>(A.nnz());
        ASSERT_GT(sp.work(), 0) << "case " << c;
        const AuctionOptions above = hybrid_at(static_cast<double>(sp.work()) / pass + 1e-9);
        for (const AuctionOptions *opt : {static_cast<const AuctionOptions *>(nullptr), &above}) {
            SymmetricMatcher h(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid, opt);
            ASSERT_TRUE(h.advance()) << "case " << c;
            EXPECT_FALSE(h.switched()) << "case " << c;
            EXPECT_EQ(h.status(), sp.status());
            EXPECT_EQ(h.work(), sp.work()) << "case " << c;
            EXPECT_EQ(h.repaired(), 0);
            expect_bitwise(h.current(), sp.current());
            expect_bitwise(h.initial(), sp.initial());
            EXPECT_EQ(h.diagonal(), sp.diagonal());
        }
    }
}

// The switch point depends on the search work alone: pausing every 50
// edges (inside the searches, the auction and the repair's searches), at
// every search boundary (so also right before the canonical duals at the
// end of phase 1), in one-pass slices (as the ConcurrentTrial runs it) or
// one-shot give the same matching, scaling and work. deficient_kkt(9)
// adds failed phase-1 searches (free columns at the canonical duals) and
// phase 3.
TEST(AuctionMatching, HybridResumedAdvanceIsOneShot) {
    int k = 0;
    for (const CscLower &A : {scaled_opf_kkt(3000, 1200, 1, 4.0), dense_row_kkt(6000, 3, 5), deficient_kkt(9)}) {
        SCOPED_TRACE("matrix " + std::to_string(k++));
        const AuctionOptions opt = hybrid_at(0.02);
        SymmetricMatcher one(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid, &opt);
        ASSERT_TRUE(one.advance());
        ASSERT_TRUE(one.switched());
        ASSERT_EQ(one.status(), ErrorCode::Success);
        // Premise: the canonical duals ran (the scale is ShortestPath's;
        // without them it is not, from index 2 of matrix 0 on).
        if (k <= 2) {
            SymmetricMatcher sp(A, MatchingSearchOrder::Degree);
            ASSERT_TRUE(sp.advance());
            const MatchingResult rs = sp.current(), r1 = one.current();
            for (Int i = 0; i < A.n; ++i) {
                const double ls = std::log(rs.scale[static_cast<std::size_t>(i)]);
                ASSERT_NEAR(std::log(r1.scale[static_cast<std::size_t>(i)]), ls, 1e-12 * std::max(1.0, std::abs(ls)))
                    << "i=" << i;
            }
        }
        {
            SymmetricMatcher m(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid, &opt);
            for (long long budget = 0; !m.advance(budget); budget = m.work() + 1) {
            }
            expect_bitwise(m.current(), one.current());
            EXPECT_EQ(m.work(), one.work());
            EXPECT_EQ(m.repaired(), one.repaired());
        }
        {
            SymmetricMatcher m(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid, &opt);
            int pauses = 0;
            for (long long budget = 0; !m.advance(budget); budget = m.work() + 50) ++pauses;
            EXPECT_GT(pauses, 10);
            EXPECT_TRUE(m.switched());
            expect_bitwise(m.current(), one.current());
            EXPECT_EQ(m.work(), one.work());
            EXPECT_EQ(m.repaired(), one.repaired());
        }
        {
            // Slices of a tenth of a pass (these runs are a few passes long).
            const long long slice = std::max<long long>(1, A.nnz() / 5);
            SymmetricMatcher m(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid, &opt);
            for (long long budget = slice; !m.advance(budget); budget += slice) {
            }
            expect_bitwise(m.current(), one.current());
            EXPECT_EQ(m.work(), one.work());
        }
    }
}

// The engine is the bench probe it replaces (bench_matching_trial
// --hybrid-probe, rule w<k>): the probe pausing at every search boundary
// and switching at the first one past switch_passes gives the same
// matching, scaling and work bitwise.
TEST(AuctionMatching, HybridIsTheProbe) {
    for (const CscLower &A : {scaled_opf_kkt(3000, 1200, 2, 4.0), dense_row_kkt(6000, 3, 5)}) {
        const AuctionOptions opt = hybrid_at(0.03);
        SymmetricMatcher h(A, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid, &opt);
        ASSERT_TRUE(h.advance());
        ASSERT_TRUE(h.switched());

        SymmetricMatcher p(A, MatchingSearchOrder::Degree);
        const double target = opt.switch_passes * 2.0 * static_cast<double>(A.nnz());
        bool done = false;
        while (!(done = p.advance(p.work() + 1)) && static_cast<double>(p.work()) <= target) {
        }
        ASSERT_FALSE(done);
        p.probe_switch_to_auction(opt);
        ASSERT_TRUE(p.advance());
        expect_bitwise(h.current(), p.current());
        EXPECT_EQ(h.work(), p.work());
        EXPECT_EQ(h.repaired(), p.repaired());
    }
}

// Structurally singular matrices and deficient sparse parts under a switch
// at the first search boundary: the ShortestPath status, and on success the
// optimum with bounded scaling (the auction's price wars do not reach it).
TEST(AuctionMatching, HybridSingularAndDense) {
    CscLower A;  // Matching.StructurallySingular: row/column 1 is empty.
    A.n = 3;
    A.col_ptr = {0, 1, 1, 2};
    A.row_idx = {0, 2};
    A.values = {4.0, 5.0};
    CscLower B;  // Columns 0 and 1 only reach row 2.
    B.n = 3;
    B.col_ptr = {0, 1, 2, 3};
    B.row_idx = {2, 2, 2};
    B.values = {1.0, 1.0, 1.0};
    std::vector<CscLower> cases = {A, B, dense_row_kkt(6000, 3, 5)};
    for (unsigned seed = 1; seed <= 12; ++seed) cases.push_back(deficient_kkt(seed));
    const AuctionOptions opt = hybrid_at(0.0);
    int switched = 0, success = 0, singular = 0;
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const CscLower &M = cases[c];
        SymmetricMatcher sp(M, MatchingSearchOrder::Degree);
        ASSERT_TRUE(sp.advance());
        SymmetricMatcher h(M, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid, &opt);
        ASSERT_TRUE(h.advance()) << "case " << c;
        EXPECT_EQ(h.status(), sp.status()) << "case " << c;
        switched += h.switched() ? 1 : 0;
        const MatchingResult r = h.current();
        EXPECT_EQ(r.status, sp.status()) << "case " << c;
        if (sp.status() != ErrorCode::Success) {
            ++singular;
            EXPECT_LT(r.matched_rank, M.n);
            continue;
        }
        ++success;
        same_optimum(M, r, sp.current());
        scaling_bounds(M, r);
    }
    EXPECT_GE(singular, 2);
    EXPECT_GE(success, 9);
    EXPECT_GE(switched, 10);
}

} // namespace smf
