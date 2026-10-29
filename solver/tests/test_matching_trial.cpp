// Control::matching_search_budget: analyse()'s auto matching trial past its
// search budget (MatchingTrial), against the exact trial (budget -1).

#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "kkt_test_matrices.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
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

// opf_like_kkt with every Jacobian entry (an off-diagonal) scaled by
// 10^(spread * (u - 1/2)), u uniform in [0, 1) from a seeded LCG. The spread
// of magnitudes makes the matching's greedy initial phase pair columns
// worse than the full searches do.
CscLower scaled_opf_kkt(int n_x, int m, unsigned seed, double spread) {
    CscLower A = opf_like_kkt(n_x, m, seed);
    std::uint64_t st = seed * 7919u + 13u;
    for (Int j = 0; j < A.n; ++j) {
        for (Int q = A.col_ptr[j]; q < A.col_ptr[j + 1]; ++q) {
            if (A.row_idx[q] == j) continue;
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            const double u = static_cast<double>(st >> 11) / 9007199254740992.0;
            A.values[q] *= std::pow(10.0, spread * (u - 0.5));
        }
    }
    return A;
}

} // namespace

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
        Control c4 = c1;
        c4.num_threads = 4;
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
            Control c4 = c1;
            c4.num_threads = 4;
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
        Control c4 = c1;
        c4.num_threads = 4;
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

} // namespace smf
