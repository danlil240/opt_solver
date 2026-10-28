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

#include <filesystem>
#include <limits>
#include <string>

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

// A NaN, infinite or huge budget means no budget: the exact trial.
TEST(MatchingTrial, NonFiniteBudgetIsNoBudget) {
    const CscLower A = opf_like_kkt(10000, 1000, 7);
    for (double b : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), 1e300}) {
        Control c;
        c.matrix_type = MatrixType::RealSymmetricIndefinite;
        c.matching_search_budget = b;
        Solver s;
        Info i;
        ASSERT_NE(s.analyse(A, c, i), nullptr);
        EXPECT_EQ(i.matching_trial, MatchingTrial::Exact) << "budget " << b;
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
