// Control::matching_search_budget: analyse()'s auto matching trial past its
// search budget (MatchingTrial), against the exact trial (budget -1).

#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/scaling_matching.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "kkt_test_matrices.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace smf {
namespace {

// Too small for the trial.
CscLower diagonal5() {
    CscLower A;
    A.n = 5;
    A.col_ptr = {0, 1, 2, 3, 4, 5};
    A.row_idx = {0, 1, 2, 3, 4};
    A.values = {1, 1, 1, 1, 1};
    return A;
}

// A boyd1-like KKT: a tridiagonal block (diagonal 4, off-diagonals -1)
// over n_x variables and k dense constraint rows (zero diagonal), row r
// holding every variable j with j % k == r (n_x / k entries, past the
// matcher's 10 sqrt(n) dense threshold).
CscLower diagonal_kkt(Int n_x, Int k) {
    CscLower A;
    A.n = n_x + k;
    A.col_ptr.push_back(0);
    for (Int j = 0; j < n_x; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(4.0);
        if (j + 1 < n_x) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(-1.0);
        }
        A.row_idx.push_back(n_x + j % k);
        A.values.push_back(1.0);
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    for (Int r = 0; r < k; ++r) A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    return A;
}

} // namespace

// A matrix whose matching matches every sparse column to its own diagonal
// (the dense constraint columns aside) skips the trial: the searches could
// only match the dense columns, so the matching ordering would be AMD's.
// The analysis keeps AMD with the matching's scaling (dense columns scaled
// by their duals); the factor has the KKT inertia and solves accurately,
// serially and with threads.
TEST(MatchingTrial, DiagonalMatchingSkipsTheTrial) {
    const Int n_x = 10000, k = 3;
    const CscLower A = diagonal_kkt(n_x, k);
    const int n = static_cast<int>(A.n);
    // b = A * ones (A symmetric, lower triangle stored).
    std::vector<double> b(static_cast<std::size_t>(n), 0.0);
    for (Int j = 0; j < A.n; ++j) {
        for (Int q = A.col_ptr[j]; q < A.col_ptr[j + 1]; ++q) {
            const Int i = A.row_idx[q];
            b[static_cast<std::size_t>(i)] += A.values[q];
            if (i != j) b[static_cast<std::size_t>(j)] += A.values[q];
        }
    }
    std::vector<double> serial_scale;
    for (int threads : {1, 4}) {
        Control c;
        c.matrix_type = MatrixType::RealSymmetricIndefinite;
        c.ordering = OrderingMethod::AutoSerial;
        c.num_threads = threads;
        c.auto_threads = false;  // small test trees
        Solver s;
        Info i;
        auto a = s.analyse(A, c, i);
        ASSERT_NE(a, nullptr) << "threads " << threads;
        EXPECT_EQ(i.matching_trial, MatchingTrial::Diagonal) << "threads " << threads;
        EXPECT_EQ(i.ordering_used, static_cast<int>(OrderingMethod::AMD)) << "threads " << threads;
        EXPECT_FALSE(a->matching_rejected) << "threads " << threads;
        EXPECT_TRUE(a->matching_scale_partial) << "threads " << threads;
        ASSERT_EQ(a->matching_scale.size(), static_cast<std::size_t>(n)) << "threads " << threads;
        for (double v : a->matching_scale) EXPECT_TRUE(std::isfinite(v) && v > 0.0) << "threads " << threads;
        // The concurrent trial decides as the serial one does.
        if (threads == 1) serial_scale = a->matching_scale;
        else EXPECT_EQ(a->matching_scale, serial_scale);

        FactorKeep f;
        ASSERT_EQ(s.factor(*a, c, i, f), FactorStatus::Success) << "threads " << threads;
        EXPECT_EQ(i.num_positive, n_x) << "threads " << threads;
        EXPECT_EQ(i.num_negative, k) << "threads " << threads;
        EXPECT_EQ(i.num_zero, 0) << "threads " << threads;
        std::vector<double> x = b;
        ASSERT_EQ(s.solve(f, c, i, x.data(), n), 0) << "threads " << threads;
        double err = 0.0;
        for (double v : x) err = std::max(err, std::abs(v - 1.0));
        EXPECT_LT(err, 1e-10) << "threads " << threads;
    }
}

// MatchingSearchOrder: column order is compute_symmetric_matching (the API
// contract: same code path, so this is no check of the pre-change loop);
// degree order reaches a matching of the same (maximum) product, and its
// scaling still has unit matched entries and no entry above 1.
TEST(MatchingTrial, DegreeSearchOrderIsAMaximumProductMatching) {
    const auto entry = [](const CscLower &A, Int r, Int c) {
        const Int lo = std::min(r, c), hi = std::max(r, c);
        for (Int q = A.col_ptr[lo]; q < A.col_ptr[lo + 1]; ++q)
            if (A.row_idx[q] == hi) return A.values[q];
        return 0.0;
    };
    int differ = 0;  // seeds whose two orders reach different optimal matchings or duals
    for (unsigned seed : {1u, 2u, 3u}) {
        const CscLower A = scaled_opf_kkt(3000, 1200, seed, 4.0);
        SymmetricMatcher col(A, MatchingSearchOrder::Column), deg(A, MatchingSearchOrder::Degree);
        ASSERT_TRUE(col.advance());
        ASSERT_TRUE(deg.advance());
        const MatchingResult mc = col.current(), md = deg.current();
        const MatchingResult ref = compute_symmetric_matching(A);
        EXPECT_EQ(mc.col_to_row, ref.col_to_row) << "seed " << seed;
        EXPECT_EQ(mc.scale, ref.scale) << "seed " << seed;
        ASSERT_EQ(md.status, ErrorCode::Success) << "seed " << seed;
        ASSERT_EQ(md.matched_rank, A.n) << "seed " << seed;
        double log_c = 0.0, log_d = 0.0;
        for (Int j = 0; j < A.n; ++j) {
            log_c += std::log(std::abs(entry(A, mc.col_to_row[j], j)));
            log_d += std::log(std::abs(entry(A, md.col_to_row[j], j)));
        }
        EXPECT_NEAR(log_d, log_c, 1e-9 * std::abs(log_c) + 1e-9) << "seed " << seed;
        differ += (md.col_to_row != mc.col_to_row || md.scale != mc.scale) ? 1 : 0;
        for (Int j = 0; j < A.n; ++j) {
            const Int r = md.col_to_row[j];
            EXPECT_NEAR(std::abs(md.scale[r] * entry(A, r, j) * md.scale[j]), 1.0, 1e-10) << "seed " << seed;
            for (Int q = A.col_ptr[j]; q < A.col_ptr[j + 1]; ++q)
                EXPECT_LE(std::abs(md.scale[A.row_idx[q]] * A.values[q] * md.scale[j]), 1.0 + 1e-10);
        }
    }
    // The orders search differently here: the product check is not vacuous.
    EXPECT_GT(differ, 0);
}

// Auto applying the partial scale of a diagonal trial to the analysed values
// is not their own matching scaling: a costly factorization past
// Control::scale_on_delay_next_ratio still leaves the next one the tighter
// limit (FactorKeep::delay_next), as an unscaled one does.
TEST(MatchingTrial, PartialScaleKeepsTheNextLimitArmed) {
    const CscLower A = diagonal_kkt(10000, 3);
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.ordering = OrderingMethod::AutoSerial;
    c.num_threads = 1;
    c.auto_threads = false;  // small test trees
    c.scale_on_delay_next_ratio = 0.0;  // any factorization is costly
    Solver s;
    Info i;
    auto a = s.analyse(A, c, i);
    ASSERT_NE(a, nullptr);
    ASSERT_TRUE(a->matching_scale_partial);
    FactorKeep f;
    ASSERT_EQ(s.factor(*a, c, i, f), FactorStatus::Success);
    ASSERT_EQ(i.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_TRUE(f.delay_next);
    EXPECT_EQ(f.delay_analysis, a->serial);
}

// One sparse column whose diagonal the max-product matching does not take
// makes the matching not diagonal: the trial runs.
TEST(MatchingTrial, OffDiagonalSparseColumnRunsTheTrial) {
    CscLower A = diagonal_kkt(10000, 3);
    // Column 0: diagonal 4 -> 1e-3, far below its sub-diagonal |-1|, so
    // the max-product matching takes (1, 0) and (0, 1) instead.
    ASSERT_EQ(A.row_idx[0], 0);
    A.values[0] = 1e-3;
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.ordering = OrderingMethod::AutoSerial;
    c.num_threads = 1;
    c.auto_threads = false;  // small test trees
    Solver s;
    Info i;
    auto a = s.analyse(A, c, i);
    ASSERT_NE(a, nullptr);
    EXPECT_NE(i.matching_trial, MatchingTrial::Diagonal);
    EXPECT_NE(i.matching_trial, MatchingTrial::None);
    EXPECT_FALSE(a->matching_scale_partial);
}

// The auto matching trial's searches have a work budget (default: about two
// passes over the matrix's symmetric graph). Past it, the trial first orders
// the matching's initial phase and, that ordering being far too dear,
// rejects on it without the exact matching; the analysis is the exact
// trial's (AMD kept, no matching scale).
TEST(MatchingTrial, RejectsOnInitialMatchingPastBudget) {
    EXPECT_DOUBLE_EQ(Control{}.matching_search_budget, 2.0);
    const CscLower A = opf_like_kkt(6000, 5000, 7);
    Control exact;
    exact.matrix_type = MatrixType::RealSymmetricIndefinite;
    exact.ordering = OrderingMethod::AutoSerial;
    exact.matching_search_budget = -1.0;
    Control budgeted = exact;
    budgeted.matching_search_budget = 0.0;
    Solver s1, s2;
    Info i1, i2;
    auto a1 = s1.analyse(A, exact, i1);
    auto a2 = s2.analyse(A, budgeted, i2);
    ASSERT_NE(a1, nullptr);
    ASSERT_NE(a2, nullptr);
    EXPECT_EQ(i1.matching_trial, MatchingTrial::Exact);
    EXPECT_EQ(i2.matching_trial, MatchingTrial::RejectedInitial);
    EXPECT_EQ(i1.ordering_used, static_cast<int>(OrderingMethod::AMD));
    EXPECT_EQ(i2.ordering_used, static_cast<int>(OrderingMethod::AMD));
    EXPECT_EQ(a2->perm, a1->perm);
    EXPECT_TRUE(a2->matching_rejected);
    EXPECT_TRUE(a2->matching_scale.empty());
}

// The trial records the flop ratios it decided on, so the initial phase's
// margin to the rule is observable: the exact trial its exact ratio only,
// a trial rejected on the initial matching its initial ratio only.
TEST(MatchingTrial, RecordsFlopRatios) {
    const CscLower A = opf_like_kkt(6000, 5000, 7);
    Control exact;
    exact.matrix_type = MatrixType::RealSymmetricIndefinite;
    exact.ordering = OrderingMethod::AutoSerial;
    exact.matching_search_budget = -1.0;
    Control budgeted = exact;
    budgeted.matching_search_budget = 0.0;
    Solver s1, s2;
    Info i1, i2;
    ASSERT_NE(s1.analyse(A, exact, i1), nullptr);
    ASSERT_NE(s2.analyse(A, budgeted, i2), nullptr);
    EXPECT_GT(i1.matching_exact_ratio, 1.5);
    EXPECT_EQ(i1.matching_initial_ratio, 0.0);
    ASSERT_EQ(i2.matching_trial, MatchingTrial::RejectedInitial);
    EXPECT_GT(i2.matching_initial_ratio, 2.4);  // past the guard band
    EXPECT_EQ(i2.matching_exact_ratio, 0.0);
    // No trial: nothing recorded.
    Solver s3;
    Info i3;
    ASSERT_NE(s3.analyse(diagonal5(), exact, i3), nullptr);
    EXPECT_EQ(i3.matching_trial, MatchingTrial::None);
    EXPECT_EQ(i3.matching_initial_ratio, 0.0);
    EXPECT_EQ(i3.matching_exact_ratio, 0.0);
}

// Two analyses of A that must agree: the exact trial's and a budgeted one's.
static void expect_same_analysis(const AnalysisKeep &exact, const AnalysisKeep &budgeted, const Info &ie,
                                 const Info &ib) {
    EXPECT_EQ(ib.ordering_used, ie.ordering_used);
    EXPECT_EQ(budgeted.perm, exact.perm);
    EXPECT_EQ(budgeted.matching_scale, exact.matching_scale);
    EXPECT_EQ(budgeted.matching_rejected, exact.matching_rejected);
    EXPECT_EQ(ib.matching_exact_ratio, ie.matching_exact_ratio);
}

// The ResumedExact path on a synthetic KKT (no fixture): budget 0 decides on
// the initial matching first, which passes, so the searches resume and the
// exact matching is adopted. The analysis is bitwise the exact trial's.
TEST(MatchingTrial, ResumesOnSyntheticKkt) {
    const CscLower A = opf_like_kkt(10000, 1000, 7);
    Control exact;
    exact.matrix_type = MatrixType::RealSymmetricIndefinite;
    exact.ordering = OrderingMethod::AutoSerial;
    exact.matching_search_budget = -1.0;
    Control budgeted = exact;
    budgeted.matching_search_budget = 0.0;
    Solver s1, s2;
    Info i1, i2;
    auto a1 = s1.analyse(A, exact, i1);
    auto a2 = s2.analyse(A, budgeted, i2);
    ASSERT_NE(a1, nullptr);
    ASSERT_NE(a2, nullptr);
    EXPECT_EQ(i1.matching_trial, MatchingTrial::Exact);
    EXPECT_EQ(i2.matching_trial, MatchingTrial::ResumedExact);
    EXPECT_EQ(i1.ordering_used, static_cast<int>(OrderingMethod::Matching));
    ASSERT_FALSE(a1->matching_scale.empty());
    EXPECT_FALSE(a2->matching_rejected);
    expect_same_analysis(*a1, *a2, i1, i2);
    EXPECT_GT(i2.matching_initial_ratio, 0.0);
    EXPECT_LE(i2.matching_initial_ratio, 1.5);
}

// Guard band: an initial matching ordering a little above the 1.5 rule is
// not final (the exact one may still pass: corpus turon_m, 2.27 initial vs
// 1.35 exact). The searches resume and the exact matching decides; here it
// rejects too, and the analysis is the exact trial's.
TEST(MatchingTrial, ResumesInsideGuardBand) {
    const CscLower A = opf_like_kkt(8000, 2000, 1);
    Control exact;
    exact.matrix_type = MatrixType::RealSymmetricIndefinite;
    exact.ordering = OrderingMethod::AutoSerial;
    exact.matching_search_budget = -1.0;
    Control budgeted = exact;
    budgeted.matching_search_budget = 0.0;
    Solver s1, s2;
    Info i1, i2;
    auto a1 = s1.analyse(A, exact, i1);
    auto a2 = s2.analyse(A, budgeted, i2);
    ASSERT_NE(a1, nullptr);
    ASSERT_NE(a2, nullptr);
    ASSERT_GT(i2.matching_initial_ratio, 1.5);
    ASSERT_LT(i2.matching_initial_ratio, 2.4) << "fixture must lie inside the guard band";
    EXPECT_EQ(i1.matching_trial, MatchingTrial::Exact);
    EXPECT_EQ(i2.matching_trial, MatchingTrial::ResumedExact);
    EXPECT_EQ(i1.ordering_used, static_cast<int>(OrderingMethod::AMD));
    EXPECT_TRUE(a2->matching_rejected);
    EXPECT_TRUE(a2->matching_scale.empty());
    expect_same_analysis(*a1, *a2, i1, i2);
}

// Guard band, the other way: the initial matching ordering sits above the
// 1.5 rule (1.64 here) but inside the 2.4 guard band, so it is not rejected;
// the resumed exact matching (1.44) passes and is adopted, with the same
// ordering and matching scale as the exact trial (budget -1). Corpus
// analogue: turon_m (2.27 initial, 1.35 exact). The matrix was found by a
// sweep (scaled_opf_kkt), so the test checks its own band first.
TEST(MatchingTrial, ResumesInsideGuardBandAndAdopts) {
    const CscLower A = scaled_opf_kkt(9000, 2000, 6, 1.0);
    Control exact;
    exact.matrix_type = MatrixType::RealSymmetricIndefinite;
    exact.ordering = OrderingMethod::AutoSerial;
    exact.matching_search_budget = -1.0;
    Control budgeted = exact;
    budgeted.matching_search_budget = 0.0;
    Solver s1, s2;
    Info i1, i2;
    auto a1 = s1.analyse(A, exact, i1);
    auto a2 = s2.analyse(A, budgeted, i2);
    ASSERT_NE(a1, nullptr);
    ASSERT_NE(a2, nullptr);
    ASSERT_GT(i2.matching_initial_ratio, 1.5) << "fixture must start above the rule";
    ASSERT_LE(i2.matching_initial_ratio, 2.4) << "fixture must lie inside the guard band";
    ASSERT_LE(i1.matching_exact_ratio, 1.5) << "fixture must pass on the exact matching";
    EXPECT_EQ(i1.matching_trial, MatchingTrial::Exact);
    EXPECT_EQ(i2.matching_trial, MatchingTrial::ResumedExact);
    EXPECT_EQ(i1.ordering_used, static_cast<int>(OrderingMethod::Matching));
    EXPECT_EQ(i2.ordering_used, static_cast<int>(OrderingMethod::Matching));
    EXPECT_FALSE(a1->matching_rejected);
    EXPECT_FALSE(a2->matching_rejected);
    ASSERT_FALSE(a1->matching_scale.empty());
    expect_same_analysis(*a1, *a2, i1, i2);
}

// A NaN, infinite or huge budget means no budget: the exact trial. The
// analysis (trial verdict, ratios, ordering, matching scale) and the
// factorization it feeds (inertia, delays, factor size, solution) are
// bitwise those of budget -1.
TEST(MatchingTrial, NonFiniteBudgetIsNoBudget) {
    const CscLower A = opf_like_kkt(10000, 1000, 7);
    const int n = static_cast<int>(A.n);
    Control ref;
    ref.matrix_type = MatrixType::RealSymmetricIndefinite;
    ref.ordering = OrderingMethod::AutoSerial;
    ref.num_threads = 1;
    ref.auto_threads = false;  // small test trees
    ref.matching_search_budget = -1.0;
    Solver sr;
    Info ir;
    auto ar = sr.analyse(A, ref, ir);
    ASSERT_NE(ar, nullptr);
    ASSERT_EQ(ir.matching_trial, MatchingTrial::Exact);
    ASSERT_EQ(ir.ordering_used, static_cast<int>(OrderingMethod::Matching));
    ASSERT_FALSE(ar->matching_scale.empty());
    FactorKeep fr;
    ASSERT_EQ(sr.factor(*ar, ref, ir, fr), FactorStatus::Success);
    std::vector<double> xr(static_cast<std::size_t>(n), 1.0);
    ASSERT_EQ(sr.solve(fr, ref, ir, xr.data(), n), 0);

    for (double b : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), 1e300}) {
        Control c = ref;
        c.matching_search_budget = b;
        Solver s;
        Info i;
        auto a = s.analyse(A, c, i);
        ASSERT_NE(a, nullptr) << "budget " << b;
        EXPECT_EQ(i.matching_trial, MatchingTrial::Exact) << "budget " << b;
        EXPECT_EQ(i.matching_initial_ratio, 0.0) << "budget " << b;  // never paused
        EXPECT_EQ(i.matching_exact_ratio, ir.matching_exact_ratio) << "budget " << b;
        EXPECT_EQ(i.ordering_used, ir.ordering_used) << "budget " << b;
        EXPECT_EQ(a->perm, ar->perm) << "budget " << b;
        EXPECT_FALSE(a->matching_rejected) << "budget " << b;
        EXPECT_EQ(a->matching_scale, ar->matching_scale) << "budget " << b;

        FactorKeep f;
        ASSERT_EQ(s.factor(*a, c, i, f), FactorStatus::Success) << "budget " << b;
        EXPECT_EQ(i.num_negative, ir.num_negative) << "budget " << b;
        EXPECT_EQ(i.num_zero, ir.num_zero) << "budget " << b;
        EXPECT_EQ(i.num_positive, ir.num_positive) << "budget " << b;
        EXPECT_EQ(i.delayed_pivots, ir.delayed_pivots) << "budget " << b;
        EXPECT_EQ(i.actual_factor_entries, ir.actual_factor_entries) << "budget " << b;
        std::vector<double> x(static_cast<std::size_t>(n), 1.0);
        ASSERT_EQ(s.solve(f, c, i, x.data(), n), 0) << "budget " << b;
        EXPECT_EQ(x, xr) << "budget " << b;
    }
}

// The budget counts edges scanned by serial searches, so how the trial is
// decided does not depend on the thread count (AutoSerial: the base
// ordering does not either).
TEST(MatchingTrial, DecisionDoesNotDependOnThreads) {
    for (const CscLower &A : {opf_like_kkt(10000, 1000, 7), opf_like_kkt(8000, 2000, 1),
                              opf_like_kkt(6000, 5000, 7)}) {
        Control c1;
        c1.matrix_type = MatrixType::RealSymmetricIndefinite;
        c1.ordering = OrderingMethod::AutoSerial;
        c1.matching_search_budget = 0.0;
        c1.num_threads = 1;
        c1.auto_threads = false;  // small test trees
        Control c4 = c1;
        c4.num_threads = 4;
        c4.auto_threads = false;  // small test trees
        Solver s1, s4;
        Info i1, i4;
        auto a1 = s1.analyse(A, c1, i1);
        auto a4 = s4.analyse(A, c4, i4);
        ASSERT_NE(a1, nullptr);
        ASSERT_NE(a4, nullptr);
        EXPECT_NE(i1.matching_trial, MatchingTrial::None);
        EXPECT_EQ(i4.matching_trial, i1.matching_trial);
        EXPECT_EQ(i4.matching_initial_ratio, i1.matching_initial_ratio);
        EXPECT_EQ(a4->perm, a1->perm);
    }
}

// A dear base factor (predicted flops times expected_refactors at least
// 4.5e5 per row) resumes the paused searches without the initial
// matching's verdict: no initial ratio, and the analysis is the exact
// trial's. Here the many expected refactors make a cheap factor dear, and
// the matrix is one whose initial matching rejects otherwise (see
// RejectsOnInitialMatchingPastBudget), so the exact matching decides
// instead; serially and with threads.
TEST(MatchingTrial, DearBaseFactorResumesWithoutInitialVerdict) {
    const CscLower A = opf_like_kkt(6000, 5000, 7);
    for (int threads : {1, 4}) {
        Control exact;
        exact.matrix_type = MatrixType::RealSymmetricIndefinite;
        exact.ordering = OrderingMethod::AutoSerial;
        exact.matching_search_budget = -1.0;
        exact.num_threads = threads;
        exact.auto_threads = false;  // small test trees
        Control budgeted = exact;
        budgeted.matching_search_budget = 0.0;
        Control cheap = budgeted;
        exact.expected_refactors = 1000000;
        budgeted.expected_refactors = 1000000;
        Solver s1, s2, s3;
        Info i1, i2, i3;
        auto a1 = s1.analyse(A, exact, i1);
        auto a2 = s2.analyse(A, budgeted, i2);
        auto a3 = s3.analyse(A, cheap, i3);
        ASSERT_NE(a1, nullptr);
        ASSERT_NE(a2, nullptr);
        ASSERT_NE(a3, nullptr);
        EXPECT_EQ(i3.matching_trial, MatchingTrial::RejectedInitial) << "threads " << threads;
        EXPECT_EQ(i1.matching_trial, MatchingTrial::Exact) << "threads " << threads;
        EXPECT_EQ(i2.matching_trial, MatchingTrial::ResumedExact) << "threads " << threads;
        EXPECT_EQ(i2.matching_initial_ratio, 0.0) << "threads " << threads;
        EXPECT_EQ(a2->rejected_matching_scale, a1->rejected_matching_scale) << "threads " << threads;
        expect_same_analysis(*a1, *a2, i1, i2);
    }
}

// With threads the trial runs on a worker thread, concurrently with the base
// ordering (and stopped once the initial matching rejects). Every budget
// path (exact within the budget, resumed, rejected on the initial matching,
// guard band) gives the serial analysis: verdict, both ratios, ordering,
// matching scales and rejected flag.
TEST(MatchingTrial, ConcurrentTrialIsTheSerialOne) {
    const CscLower mats[] = {opf_like_kkt(10000, 1000, 7), opf_like_kkt(8000, 2000, 1),
                             opf_like_kkt(6000, 5000, 7), scaled_opf_kkt(9000, 2000, 6, 1.0)};
    for (const CscLower &A : mats) {
        for (double budget : {-1.0, 0.0, 2.0}) {
            Control c1;
            c1.matrix_type = MatrixType::RealSymmetricIndefinite;
            c1.ordering = OrderingMethod::AutoSerial;
            c1.matching_search_budget = budget;
            c1.num_threads = 1;
            c1.auto_threads = false;  // small test trees
            Control c4 = c1;
            c4.num_threads = 4;
            c4.auto_threads = false;  // small test trees
            Solver s1, s4;
            Info i1, i4;
            auto a1 = s1.analyse(A, c1, i1);
            ASSERT_NE(a1, nullptr);
            ASSERT_NE(i1.matching_trial, MatchingTrial::None);
            for (int rep = 0; rep < 2; ++rep) {  // the second reuses s4's buffers
                auto a4 = s4.analyse(A, c4, i4);
                ASSERT_NE(a4, nullptr);
                EXPECT_EQ(i4.matching_trial, i1.matching_trial) << "budget " << budget;
                EXPECT_EQ(i4.matching_initial_ratio, i1.matching_initial_ratio) << "budget " << budget;
                EXPECT_EQ(i4.matching_exact_ratio, i1.matching_exact_ratio) << "budget " << budget;
                EXPECT_EQ(i4.ordering_used, i1.ordering_used) << "budget " << budget;
                EXPECT_EQ(a4->perm, a1->perm) << "budget " << budget;
                EXPECT_EQ(a4->matching_scale, a1->matching_scale) << "budget " << budget;
                EXPECT_EQ(a4->rejected_matching_scale, a1->rejected_matching_scale) << "budget " << budget;
                EXPECT_EQ(a4->matching_rejected, a1->matching_rejected) << "budget " << budget;
            }
        }
    }
}

namespace {

Control hybrid_test_control(double budget, MatchingAlgorithm alg, int threads) {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.ordering = OrderingMethod::AutoSerial;
    c.matching_search_budget = budget;
    // Hybrid is the default (AuctionMatching.HybridDefaults checks it): the
    // Hybrid runs leave the field alone, so they test the default.
    if (alg != MatchingAlgorithm::Hybrid) c.matching_algorithm = alg;
    c.num_threads = threads;
    c.auto_threads = false;  // small test trees
    return c;
}

// The trial's matcher under Hybrid run to the end on the matrix analyse()
// matched (AnalysisKeep::cleaned).
bool hybrid_switches(const CscLower &C) {
    SymmetricMatcher m(C, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid);
    m.advance(-1);
    return m.switched();
}

double matched_log_product(const CscLower &A, const std::vector<Int> &col_to_row) {
    double p = 0.0;
    for (Int j = 0; j < A.n; ++j) {
        const Int r = col_to_row[static_cast<std::size_t>(j)];
        const Int lo = std::min(r, j), hi = std::max(r, j);
        for (Int q = A.col_ptr[lo]; q < A.col_ptr[lo + 1]; ++q)
            if (A.row_idx[q] == hi) p += std::log(std::abs(A.values[q]));
    }
    return p;
}

// The verdict of two analyses of one matrix: equal, and the initial ratio
// bitwise equal (both are 0 when unset).
void expect_same_verdict(const AnalysisKeep &a, const Info &ia, const AnalysisKeep &b, const Info &ib) {
    EXPECT_EQ(ib.matching_trial, ia.matching_trial);
    EXPECT_EQ(ib.ordering_used, ia.ordering_used);
    EXPECT_EQ(b.matching_rejected, a.matching_rejected);
    EXPECT_EQ(std::memcmp(&ib.matching_initial_ratio, &ia.matching_initial_ratio, sizeof(double)), 0)
        << ib.matching_initial_ratio << " vs " << ia.matching_initial_ratio;
}

} // namespace

// Control::matching_algorithm = Hybrid runs the trial's searches under the
// hybrid engine. It switches to the auction only past 40 passes of search
// work (AuctionOptions::switch_passes) and is ShortestPath bitwise before
// that; the default budget (2 passes) pauses far earlier, so no verdict can
// move. On matrices whose searches stay below 40 passes the whole analysis
// is bitwise ShortestPath's. On two that switch inside analyse()
// (opf_like_kkt(20000, 18000, 1): 43.9 passes, full rank; opf_like_kkt(10000,
// 9200, 3): 44.7 passes, structurally singular) the verdict is the same and
// the matching is again a maximum-product one (same status, same log
// product); when the standalone matchers also give the same matching and
// scale, the analysis is bitwise ShortestPath's (otherwise the max-product
// matching is not unique, and perm may differ).
TEST(MatchingTrial, HybridTrialKeepsVerdicts) {
    const double budgets[] = {-1.0, 0.0, Control{}.matching_search_budget};
    const CscLower quiet[] = {scaled_opf_kkt(3000, 1200, 1, 4.0), opf_like_kkt(10000, 1000, 7),
                              opf_like_kkt(6000, 5000, 7), diagonal_kkt(10000, 3)};
    int k = 0;
    for (const CscLower &A : quiet) {
        for (double budget : budgets) {
            SCOPED_TRACE("matrix " + std::to_string(k) + " budget " + std::to_string(budget));
            Solver ss, sh;
            Info is, ih;
            auto as = ss.analyse(A, hybrid_test_control(budget, MatchingAlgorithm::ShortestPath, 1), is);
            auto ah = sh.analyse(A, hybrid_test_control(budget, MatchingAlgorithm::Hybrid, 1), ih);
            ASSERT_NE(as, nullptr);
            ASSERT_NE(ah, nullptr);
            ASSERT_FALSE(hybrid_switches(ah->cleaned)) << "fixture must stay below the switch";
            expect_same_verdict(*as, is, *ah, ih);
            EXPECT_EQ(ih.matching_exact_ratio, is.matching_exact_ratio);
            EXPECT_EQ(ah->perm, as->perm);
            EXPECT_EQ(ah->matching_scale, as->matching_scale);
            EXPECT_EQ(ah->rejected_matching_scale, as->rejected_matching_scale);
            EXPECT_EQ(ah->matching_scale_partial, as->matching_scale_partial);
        }
        ++k;
    }

    const CscLower switching[] = {opf_like_kkt(20000, 18000, 1), opf_like_kkt(10000, 9200, 3)};
    int switched = 0, success = 0, same = 0;
    k = 0;
    for (const CscLower &A : switching) {
        bool same_matching = false;
        for (double budget : budgets) {
            SCOPED_TRACE("switching matrix " + std::to_string(k) + " budget " + std::to_string(budget));
            Solver ss, sh;
            Info is, ih;
            auto as = ss.analyse(A, hybrid_test_control(budget, MatchingAlgorithm::ShortestPath, 1), is);
            auto ah = sh.analyse(A, hybrid_test_control(budget, MatchingAlgorithm::Hybrid, 1), ih);
            ASSERT_NE(as, nullptr);
            ASSERT_NE(ah, nullptr);
            if (budget == budgets[0]) {
                // The hybrid path really runs inside this analyse(): the
                // exact trial's matcher, on its own cleaned matrix, switches.
                const CscLower &C = ah->cleaned;
                SymmetricMatcher ms(C, MatchingSearchOrder::Degree);
                SymmetricMatcher mh(C, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid);
                ms.advance(-1);
                mh.advance(-1);
                ASSERT_TRUE(mh.switched());
                ++switched;
                ASSERT_EQ(mh.status(), ms.status());
                if (ms.status() == ErrorCode::Success) {
                    ++success;
                    const MatchingResult rs = ms.current(), rh = mh.current();
                    const double ps = matched_log_product(C, rs.col_to_row);
                    EXPECT_NEAR(matched_log_product(C, rh.col_to_row), ps, 1e-9 * std::abs(ps) + 1e-9);
                    // The matchings coincide here; the scales differ in
                    // rounding only since the canonical duals (31323 of
                    // 38000 indices, max |dlog| 1.2e-14; 5.6 without them).
                    // That rounding is what makes the plumbing check below
                    // (each analysis keeps its own matcher's scale)
                    // observable; it is not asserted, since a compiler or FP
                    // flag change could make the scales bitwise equal on a
                    // correct build.
                    if (rh.scale == rs.scale)
                        std::printf("[ hybrid ] note: Hybrid and ShortestPath scales are bitwise equal here; "
                                    "the kept-scale plumbing check is not observable\n");
                    for (std::size_t i = 0; i < rs.scale.size(); ++i) {
                        const double ls = std::log(rs.scale[i]);
                        ASSERT_NEAR(std::log(rh.scale[i]), ls, 1e-12 * std::max(1.0, std::abs(ls))) << "i=" << i;
                    }
                    same_matching = rs.col_to_row == rh.col_to_row && rs.scale == rh.scale;
                    same += same_matching ? 1 : 0;
                    // Each analysis kept its own matcher's exact scale
                    // (adopted, or recorded as rejected).
                    const auto kept = [](const AnalysisKeep &a) -> const std::vector<double> & {
                        return a.matching_scale.empty() ? a.rejected_matching_scale : a.matching_scale;
                    };
                    EXPECT_EQ(kept(*as), rs.scale);
                    EXPECT_EQ(kept(*ah), rh.scale);
                }
            }
            expect_same_verdict(*as, is, *ah, ih);
            if (same_matching) {
                EXPECT_EQ(ah->perm, as->perm);
                EXPECT_EQ(ah->matching_scale, as->matching_scale);
            }
        }
        ++k;
    }
    EXPECT_EQ(switched, 2);
    EXPECT_EQ(success, 1);
    std::printf("[ hybrid ] switching matrices with the ShortestPath matching and scale: %d of %d\n", same, success);
}

// ConcurrentTrialIsTheSerialOne under Hybrid, plus a matrix whose searches
// switch to the auction (43.9 passes): the concurrent trial's one-pass slices
// pause the hybrid engine around its switch and auction, and 1 vs 4 threads
// still give the same perm, scales, rejected flag and verdict.
TEST(MatchingTrial, HybridConcurrentTrialIsTheSerialOne) {
    const CscLower mats[] = {opf_like_kkt(10000, 1000, 7), opf_like_kkt(8000, 2000, 1),
                             opf_like_kkt(6000, 5000, 7), scaled_opf_kkt(9000, 2000, 6, 1.0),
                             opf_like_kkt(20000, 18000, 1)};
    for (const CscLower &A : mats) {
        for (double budget : {-1.0, 0.0, 2.0}) {
            SCOPED_TRACE("n " + std::to_string(A.n) + " budget " + std::to_string(budget));
            const Control c1 = hybrid_test_control(budget, MatchingAlgorithm::Hybrid, 1);
            const Control c4 = hybrid_test_control(budget, MatchingAlgorithm::Hybrid, 4);
            Solver s1, s4;
            Info i1, i4;
            auto a1 = s1.analyse(A, c1, i1);
            ASSERT_NE(a1, nullptr);
            ASSERT_NE(i1.matching_trial, MatchingTrial::None);
            // Premise: the last matrix really switches (and so runs the
            // auction, the repair and the canonical duals inside the trial).
            if (&A == &mats[std::size(mats) - 1]) ASSERT_TRUE(hybrid_switches(a1->cleaned));
            for (int rep = 0; rep < 2; ++rep) {  // the second reuses s4's buffers
                auto a4 = s4.analyse(A, c4, i4);
                ASSERT_NE(a4, nullptr);
                EXPECT_EQ(i4.matching_trial, i1.matching_trial);
                EXPECT_EQ(i4.matching_initial_ratio, i1.matching_initial_ratio);
                EXPECT_EQ(i4.matching_exact_ratio, i1.matching_exact_ratio);
                EXPECT_EQ(i4.ordering_used, i1.ordering_used);
                EXPECT_EQ(a4->perm, a1->perm);
                EXPECT_EQ(a4->matching_scale, a1->matching_scale);
                EXPECT_EQ(a4->rejected_matching_scale, a1->rejected_matching_scale);
                EXPECT_EQ(a4->matching_rejected, a1->matching_rejected);
            }
            if (budget < 0.0) {
                // The exact trial kept the Hybrid matcher's scale (adopted,
                // or recorded as rejected), on both thread counts.
                SymmetricMatcher mh(a1->cleaned, MatchingSearchOrder::Degree, MatchingAlgorithm::Hybrid);
                ASSERT_TRUE(mh.advance(-1));
                const std::vector<double> &kept =
                    a1->matching_scale.empty() ? a1->rejected_matching_scale : a1->matching_scale;
                EXPECT_EQ(kept, mh.current().scale);
            }
        }
    }
}

// The same under AutoParallel (the IPOPT plugin's ordering). Its base
// ordering can come from a thread-count dependent path (serial METIS at 1
// thread, parallel nested dissection above), so the flop ratios and the
// verdict are thread independent only where that path is not taken. These
// synthetic KKTs have AMD factors too cheap to try METIS, which this test
// pins.
TEST(MatchingTrial, DecisionDoesNotDependOnThreadsAutoParallel) {
    for (const CscLower &A : {opf_like_kkt(10000, 1000, 7), opf_like_kkt(8000, 2000, 1),
                              opf_like_kkt(6000, 5000, 7)}) {
        Control c1;
        c1.matrix_type = MatrixType::RealSymmetricIndefinite;
        c1.ordering = OrderingMethod::AutoParallel;
        c1.matching_search_budget = 0.0;
        c1.num_threads = 1;
        c1.auto_threads = false;  // small test trees
        Control c4 = c1;
        c4.num_threads = 4;
        c4.auto_threads = false;  // small test trees
        Solver s1, s4;
        Info i1, i4;
        auto a1 = s1.analyse(A, c1, i1);
        auto a4 = s4.analyse(A, c4, i4);
        ASSERT_NE(a1, nullptr);
        ASSERT_NE(a4, nullptr);
        EXPECT_NE(i1.matching_trial, MatchingTrial::None);
        EXPECT_EQ(i4.matching_trial, i1.matching_trial);
        EXPECT_EQ(i4.matching_initial_ratio, i1.matching_initial_ratio);
        EXPECT_EQ(i4.matching_exact_ratio, i1.matching_exact_ratio);
        EXPECT_EQ(i4.ordering_used, i1.ordering_used);
        EXPECT_EQ(a4->perm, a1->perm);
    }
}

// TSOPF_FS_b162_c1 adopts the matching ordering, and its searches exceed the
// default budget: the initial matching passes, the searches resume, and the
// analysis is bitwise the exact trial's.
TEST(MatchingTrial, ResumesWhenInitialMatchingPasses) {
    namespace fs = std::filesystem;
    const fs::path mtx = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" / "fixtures" /
                         "large" / "TSOPF_FS_b162_c1" / "TSOPF_FS_b162_c1.mtx";
    std::error_code ec;
    if (!fs::exists(mtx, ec)) GTEST_SKIP() << "fixture not present: " << mtx.string();
    auto mm = read_matrix_market(mtx.string());
    ASSERT_TRUE(mm.ok) << mm.error_message;
    Control def;
    def.matrix_type = MatrixType::RealSymmetricIndefinite;
    def.ordering = OrderingMethod::AutoParallel;
    Control exact = def;
    exact.matching_search_budget = -1.0;
    Solver s1, s2;
    Info i1, i2;
    auto a1 = s1.analyse(mm.matrix, exact, i1);
    auto a2 = s2.analyse(mm.matrix, def, i2);
    ASSERT_NE(a1, nullptr);
    ASSERT_NE(a2, nullptr);
    EXPECT_EQ(i1.matching_trial, MatchingTrial::Exact);
    EXPECT_EQ(i2.matching_trial, MatchingTrial::ResumedExact);
    EXPECT_EQ(i1.ordering_used, static_cast<int>(OrderingMethod::Matching));
    EXPECT_EQ(i2.ordering_used, static_cast<int>(OrderingMethod::Matching));
    EXPECT_EQ(a2->perm, a1->perm);
    ASSERT_FALSE(a1->matching_scale.empty());
    EXPECT_EQ(a2->matching_scale, a1->matching_scale);
    EXPECT_FALSE(a2->matching_rejected);
}

// Control::matching_initial_decide_passes (decide on the initial matching
// first, after a short first budget). Fixtures (Degree order, default
// Hybrid matcher; free = the initial phase's free-column share, passes =
// the full searches' work):
//   opf_like_kkt(10000, 1000, 7):      free 2.4%,  0.07 passes, Exact, adopted
//   opf_like_kkt(10000, 3000, 1):      free 7.1%,  0.27 passes, Exact, AMD kept
//                                      (exact ratio 3.96, initial 3.46)
//   opf_like_kkt(6000, 5000, 7):       free 18.6%, 15.3 passes, RejectedInitial
//   scaled_opf_kkt(9000, 2000, 1, 4):  free 5.2%,  0.14 passes, Exact, adopted
//                                      (exact ratio 1.21): boyd2-like
namespace {

Control decide_control(double passes, int threads) {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.ordering = OrderingMethod::AutoSerial;
    c.matching_initial_decide_passes = passes;
    c.num_threads = threads;
    c.auto_threads = false;  // small test trees
    return c;
}

// The whole analysis of two runs agrees (verdict, ratios, ordering, scales).
void expect_same_trial_analysis(const AnalysisKeep &a, const Info &ia, const AnalysisKeep &b, const Info &ib) {
    EXPECT_EQ(ib.matching_trial, ia.matching_trial);
    EXPECT_EQ(ib.matching_initial_ratio, ia.matching_initial_ratio);
    EXPECT_EQ(ib.matching_exact_ratio, ia.matching_exact_ratio);
    EXPECT_EQ(ib.ordering_used, ia.ordering_used);
    EXPECT_EQ(b.perm, a.perm);
    EXPECT_EQ(b.matching_scale, a.matching_scale);
    EXPECT_EQ(b.rejected_matching_scale, a.rejected_matching_scale);
    EXPECT_EQ(b.matching_scale_partial, a.matching_scale_partial);
    EXPECT_EQ(b.matching_rejected, a.matching_rejected);
}

} // namespace

// Off (the default, negative, NaN, infinite) and on-but-not-firing (the
// free share below the threshold, or a NaN threshold) give the default
// analysis and factorization bitwise: verdict, ratios, ordering, scales,
// factor size, delays, inertia and solution. The off path computes no free
// share.
TEST(MatchingTrial, DecideFirstOffIsTheDefaultTrial) {
    EXPECT_LT(Control{}.matching_initial_decide_passes, 0.0);  // off by default
    EXPECT_DOUBLE_EQ(Control{}.matching_initial_decide_free, 0.05);
    const CscLower mats[] = {opf_like_kkt(10000, 1000, 7), opf_like_kkt(10000, 3000, 1),
                             opf_like_kkt(6000, 5000, 7), scaled_opf_kkt(9000, 2000, 1, 4.0)};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    int k = 0;
    for (const CscLower &A : mats) {
        const int n = static_cast<int>(A.n);
        const Control ref = decide_control(-1.0, 1);
        Solver sr;
        Info ir;
        auto ar = sr.analyse(A, ref, ir);
        ASSERT_NE(ar, nullptr);
        ASSERT_NE(ir.matching_trial, MatchingTrial::None);
        EXPECT_EQ(ir.matching_decide_first, 0);
        EXPECT_EQ(ir.matching_initial_free, -1.0);
        FactorKeep fr;
        ASSERT_EQ(sr.factor(*ar, ref, ir, fr), FactorStatus::Success);
        std::vector<double> xr(static_cast<std::size_t>(n), 1.0);
        ASSERT_EQ(sr.solve(fr, ref, ir, xr.data(), n), 0);

        struct Variant {
            const char *name;
            double passes, free;
        };
        // On but not firing: a threshold above any share (1.5), or NaN.
        const Variant variants[] = {{"-3", -3.0, 0.05}, {"nan", nan, 0.05}, {"inf", inf, 0.05},
                                    {"on, free 1.5", 0.0, 1.5}, {"on, free nan", 0.0, nan}};
        for (const Variant &v : variants) {
            SCOPED_TRACE("matrix " + std::to_string(k) + " " + v.name);
            Control c = decide_control(v.passes, 1);
            c.matching_initial_decide_free = v.free;
            Solver s;
            Info i;
            auto a = s.analyse(A, c, i);
            ASSERT_NE(a, nullptr);
            expect_same_trial_analysis(*ar, ir, *a, i);
            EXPECT_EQ(i.matching_decide_first, 0);
            EXPECT_EQ(i.matching_paused_passes, ir.matching_paused_passes);
            if (v.passes < 0.0 || !(v.passes < inf)) EXPECT_EQ(i.matching_initial_free, -1.0);
            else EXPECT_GE(i.matching_initial_free, 0.0);
            FactorKeep f;
            ASSERT_EQ(s.factor(*a, c, i, f), FactorStatus::Success);
            EXPECT_EQ(i.actual_factor_entries, ir.actual_factor_entries);
            EXPECT_EQ(i.delayed_pivots, ir.delayed_pivots);
            EXPECT_EQ(i.num_negative, ir.num_negative);
            EXPECT_EQ(i.num_positive, ir.num_positive);
            EXPECT_EQ(i.num_zero, ir.num_zero);
            std::vector<double> x(static_cast<std::size_t>(n), 1.0);
            ASSERT_EQ(s.solve(f, c, i, x.data(), n), 0);
            EXPECT_EQ(x, xr);
        }
        ++k;
    }
}

// Below the free-share threshold the knob does not fire: opf_like_kkt(10000,
// 1000, 7) leaves 2.4% free, so even 0 passes runs the default budget and
// the exact matching decides; a threshold below its share fires it.
TEST(MatchingTrial, DecideFirstNeedsTheFreeShare) {
    const CscLower A = opf_like_kkt(10000, 1000, 7);
    Solver s1, s2;
    Info i1, i2;
    Control c = decide_control(0.0, 1);
    ASSERT_NE(s1.analyse(A, c, i1), nullptr);
    EXPECT_EQ(i1.matching_decide_first, 0);
    EXPECT_GT(i1.matching_initial_free, 0.0);
    EXPECT_LT(i1.matching_initial_free, 0.05);
    EXPECT_EQ(i1.matching_trial, MatchingTrial::Exact);
    EXPECT_EQ(i1.matching_paused_passes, -1.0);
    c.matching_initial_decide_free = 0.01;
    ASSERT_NE(s2.analyse(A, c, i2), nullptr);
    EXPECT_EQ(i2.matching_decide_first, 1);
    EXPECT_EQ(i2.matching_initial_free, i1.matching_initial_free);
    EXPECT_EQ(i2.matching_paused_passes, 0.0);
    EXPECT_EQ(i2.matching_trial, MatchingTrial::ResumedExact);  // adopter: the initial phase passes
}

// Knob 0 where the initial matching decides: no search runs (paused at 0
// passes) and the trial rejects on the initial matching, keeping the
// ordering the default trial keeps (AMD). On opf_like_kkt(10000, 3000, 1)
// the default's searches finish (0.27 passes, Exact, rejected at 3.96); on
// opf_like_kkt(6000, 5000, 7) the default pauses past its 2-pass budget and
// rejects there, so knob 0 spares those 2 passes. Serial and threaded.
TEST(MatchingTrial, DecideFirstAtZeroSkipsTheSearches) {
    const CscLower mats[] = {opf_like_kkt(10000, 3000, 1), opf_like_kkt(6000, 5000, 7)};
    const MatchingTrial default_verdict[] = {MatchingTrial::Exact, MatchingTrial::RejectedInitial};
    for (int k = 0; k < 2; ++k) {
        const CscLower &A = mats[k];
        for (int threads : {1, 4}) {
            SCOPED_TRACE("matrix " + std::to_string(k) + " threads " + std::to_string(threads));
            Solver sd, sz;
            Info id, iz;
            auto ad = sd.analyse(A, decide_control(-1.0, threads), id);
            auto az = sz.analyse(A, decide_control(0.0, threads), iz);
            ASSERT_NE(ad, nullptr);
            ASSERT_NE(az, nullptr);
            EXPECT_EQ(id.matching_trial, default_verdict[k]);
            EXPECT_EQ(id.ordering_used, static_cast<int>(OrderingMethod::AMD));
            EXPECT_EQ(iz.matching_decide_first, 1);
            EXPECT_GE(iz.matching_initial_free, 0.05);
            EXPECT_EQ(iz.matching_paused_passes, 0.0);  // no search ran
            if (k == 1) EXPECT_GE(id.matching_paused_passes, 2.0);
            EXPECT_EQ(iz.matching_trial, MatchingTrial::RejectedInitial);
            EXPECT_GT(iz.matching_initial_ratio, 2.4);
            EXPECT_EQ(iz.matching_exact_ratio, 0.0);
            EXPECT_EQ(iz.ordering_used, static_cast<int>(OrderingMethod::AMD));
            EXPECT_EQ(az->perm, ad->perm);
            EXPECT_TRUE(az->matching_rejected);
            EXPECT_TRUE(az->matching_scale.empty());
        }
    }
}

// boyd2-like: an adopter whose exact matching is a few searches away (0.14
// passes) and whose initial phase leaves 5.2% free. Knob 0 decides on the
// initial matching first, which passes, so it pays the initial candidate and
// resumes (ResumedExact). A first budget past its searches (0.25, 0.5, 1)
// finishes them inside it: Exact, as the default trial, with no pause and
// no initial candidate. One below them (0.05) pauses there and resumes.
// Every variant is bitwise the default analysis (ordering, scale).
TEST(MatchingTrial, DecideFirstShortBudgetKeepsExact) {
    const CscLower A = scaled_opf_kkt(9000, 2000, 1, 4.0);
    Solver sd;
    Info id;
    auto ad = sd.analyse(A, decide_control(-1.0, 1), id);
    ASSERT_NE(ad, nullptr);
    ASSERT_EQ(id.matching_trial, MatchingTrial::Exact);
    ASSERT_EQ(id.ordering_used, static_cast<int>(OrderingMethod::Matching));
    ASSERT_FALSE(ad->matching_scale.empty());
    for (double passes : {0.0, 0.05, 0.25, 0.5, 1.0}) {
        SCOPED_TRACE("passes " + std::to_string(passes));
        Solver s;
        Info i;
        auto a = s.analyse(A, decide_control(passes, 1), i);
        ASSERT_NE(a, nullptr);
        ASSERT_EQ(i.matching_decide_first, 1) << "fixture must leave >= 5% free";
        EXPECT_EQ(i.ordering_used, id.ordering_used);
        EXPECT_EQ(i.matching_exact_ratio, id.matching_exact_ratio);
        EXPECT_EQ(a->perm, ad->perm);
        EXPECT_EQ(a->matching_scale, ad->matching_scale);
        EXPECT_FALSE(a->matching_rejected);
        if (passes >= 0.25) {
            EXPECT_EQ(i.matching_trial, MatchingTrial::Exact);
            EXPECT_EQ(i.matching_paused_passes, -1.0);
            EXPECT_EQ(i.matching_initial_ratio, 0.0);  // no initial candidate built
        } else {
            EXPECT_EQ(i.matching_trial, MatchingTrial::ResumedExact);
            EXPECT_GE(i.matching_paused_passes, passes);
            EXPECT_LT(i.matching_paused_passes, 0.14);
            EXPECT_GT(i.matching_initial_ratio, 0.0);
            EXPECT_LE(i.matching_initial_ratio, 2.4);
        }
    }
}

// The first budget is capped by matching_search_budget: past it (5 passes
// against the default 2) the trial pauses where the default one does.
TEST(MatchingTrial, DecideFirstIsCappedByTheBudget) {
    const CscLower A = opf_like_kkt(6000, 5000, 7);
    Solver sd, s5;
    Info id, i5;
    auto ad = sd.analyse(A, decide_control(-1.0, 1), id);
    auto a5 = s5.analyse(A, decide_control(5.0, 1), i5);
    ASSERT_NE(ad, nullptr);
    ASSERT_NE(a5, nullptr);
    EXPECT_EQ(i5.matching_decide_first, 1);
    EXPECT_EQ(i5.matching_paused_passes, id.matching_paused_passes);
    expect_same_trial_analysis(*ad, id, *a5, i5);
}

// The first budget counts edges scanned by serial searches, and the free
// share is the constructor's: every path (rejected at 0, resumed below the
// searches, exact within the first budget, not firing, dear factor) gives
// the same analysis and telemetry at 1, 2 and 4 threads (ConcurrentTrial),
// under AutoSerial and AutoParallel (these KKTs never try METIS).
TEST(MatchingTrial, DecideFirstDoesNotDependOnThreads) {
    const CscLower mats[] = {opf_like_kkt(10000, 1000, 7), opf_like_kkt(10000, 3000, 1),
                             opf_like_kkt(6000, 5000, 7), scaled_opf_kkt(9000, 2000, 1, 4.0)};
    int k = 0;
    for (const CscLower &A : mats) {
        for (OrderingMethod ord : {OrderingMethod::AutoSerial, OrderingMethod::AutoParallel}) {
            for (double passes : {0.0, 0.05, 0.25}) {
                for (int refactors : {1, 1000000}) {  // 1e6: a dear base factor resumes outright
                    SCOPED_TRACE("matrix " + std::to_string(k) + " ord " + std::to_string(static_cast<int>(ord)) +
                                 " passes " + std::to_string(passes) + " refactors " + std::to_string(refactors));
                    Control c1 = decide_control(passes, 1);
                    c1.ordering = ord;
                    c1.expected_refactors = refactors;
                    Solver s1;
                    Info i1;
                    auto a1 = s1.analyse(A, c1, i1);
                    ASSERT_NE(a1, nullptr);
                    ASSERT_NE(i1.matching_trial, MatchingTrial::None);
                    for (int threads : {2, 4}) {
                        Control ct = c1;
                        ct.num_threads = threads;
                        Solver st;
                        Info it;
                        for (int rep = 0; rep < (threads == 4 ? 2 : 1); ++rep) {  // rep 1 reuses st's buffers
                            auto at = st.analyse(A, ct, it);
                            ASSERT_NE(at, nullptr);
                            expect_same_trial_analysis(*a1, i1, *at, it);
                            EXPECT_EQ(it.matching_decide_first, i1.matching_decide_first);
                            EXPECT_EQ(it.matching_initial_free, i1.matching_initial_free);
                            EXPECT_EQ(it.matching_paused_passes, i1.matching_paused_passes);
                        }
                    }
                }
            }
        }
        ++k;
    }
}

// A diagonal initial matching skips the trial before the knob: no free
// share, no first budget, the default Diagonal analysis; serial and threaded.
TEST(MatchingTrial, DecideFirstLeavesTheDiagonalPath) {
    const CscLower A = diagonal_kkt(10000, 3);
    for (int threads : {1, 4}) {
        SCOPED_TRACE("threads " + std::to_string(threads));
        Solver sd, sz;
        Info id, iz;
        auto ad = sd.analyse(A, decide_control(-1.0, threads), id);
        Control z = decide_control(0.0, threads);
        z.matching_initial_decide_free = 0.0;  // would fire on any share
        auto az = sz.analyse(A, z, iz);
        ASSERT_NE(ad, nullptr);
        ASSERT_NE(az, nullptr);
        ASSERT_EQ(id.matching_trial, MatchingTrial::Diagonal);
        expect_same_trial_analysis(*ad, id, *az, iz);
        EXPECT_EQ(iz.matching_decide_first, 0);
        EXPECT_EQ(iz.matching_initial_free, -1.0);
        EXPECT_EQ(iz.matching_paused_passes, -1.0);
    }
}

} // namespace smf
