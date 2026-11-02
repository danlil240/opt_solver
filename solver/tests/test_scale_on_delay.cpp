// Control::scale_on_delay_ratio: under ScalingMethod::Auto, a threshold-
// pivoting factorization (pivot_u >= 1e-6) whose delays inflate its front
// work past the ratio stops and is redone with the matching scaling of its
// values; later factorizations of the analysis into the same FactorKeep
// start from that scale while it fits their values
// (Control::scale_on_delay_reuse_tol).
#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/determinism.hpp"
#include "smf/factor_indef.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/scaling.hpp"
#include "smf/scaling_matching.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"
#include "kkt_test_matrices.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <cmath>
#include <cstdio>
#include <random>
#include <span>
#include <utility>
#include <vector>

using namespace smf;

namespace {

// Regularized KKT [H B^T; B creg I] (H: 5-point Laplacian of a g x g grid,
// diagonal hdiag; m constraint rows, row k coupling three pseudo-random
// variables; the pattern is fixed) as D A D, D_i = 10^U(-spread, spread)
// drawn from `seed` (spread 0: A itself). With creg < 0 its inertia is exact
// whatever D: g^2 positive, m negative (H is SPD, the Schur complement
// creg I - B H^-1 B^T negative definite). With a large spread the scaling
// hides every pivot behind its row's large entries: unscaled threshold
// pivoting delays most columns, many repeatedly.
CscLower kkt(int g, int m, double hdiag, double spread, double creg = -0.1, unsigned seed = 7) {
    const int nx = g * g, n = nx + m;
    std::mt19937 rng(7);          // the pattern
    std::mt19937 drng(seed);      // the scaling D
    std::uniform_int_distribution<int> var(0, nx - 1);
    std::uniform_real_distribution<double> expo(-spread, spread);
    std::vector<double> d(static_cast<std::size_t>(n));
    for (double &v : d) v = std::pow(10.0, expo(drng));
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n));
    for (int y = 0; y < g; ++y)
        for (int x = 0; x < g; ++x) {
            const int j = y * g + x;
            auto &c = cols[static_cast<std::size_t>(j)];
            c.push_back({j, hdiag});
            if (x + 1 < g) c.push_back({j + 1, -1.0});
            if (y + 1 < g) c.push_back({j + g, -1.0});
        }
    const double coef[3] = {1.0, -0.5, 0.25};
    for (int k = 0; k < m; ++k) {
        const int row = nx + k;
        int v[3] = {var(rng), var(rng), var(rng)};
        while (v[1] == v[0]) v[1] = var(rng);
        while (v[2] == v[0] || v[2] == v[1]) v[2] = var(rng);
        for (int t = 0; t < 3; ++t) cols[static_cast<std::size_t>(v[t])].push_back({row, coef[t]});
        cols[static_cast<std::size_t>(row)].push_back({row, creg});
    }
    CscLower A;
    A.n = n;
    A.col_ptr.assign(1, 0);
    for (int j = 0; j < n; ++j) {
        auto &c = cols[static_cast<std::size_t>(j)];
        std::sort(c.begin(), c.end());
        for (auto [i, v] : c) {
            A.row_idx.push_back(i);
            A.values.push_back(d[static_cast<std::size_t>(i)] * v * d[static_cast<std::size_t>(j)]);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

constexpr int kG = 40, kM = 600;  // n = 2200

void expect_accurate_solve(const Solver &s, const FactorKeep &fk, const Control &ctrl,
                           const CscLower &A) {
    const int n = static_cast<int>(A.n);
    std::vector<double> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = 1.0 + 0.01 * (i % 97);
    std::vector<double> x = b;
    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, x.data(), n, 1), 0);
    const auto val = smf::bench::validate(A, b, x, 1);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val))
        << "eta_inf=" << val.eta_inf << " omega=" << val.omega;
}

// Front work of a factorization of ak relative to the delay-free one, the
// trigger's measure (Control::scale_on_delay_ratio).
double work_ratio(const AnalysisKeep &ak, const Info &info) {
    return info.front_work / delay_free_front_work(ak);
}

// Work ratio of a plain factorization (scale on delay off), to pin a premise.
double unscaled_ratio(const CscLower &A, Control ctrl) {
    ctrl.scale_on_delay_ratio = -1.0;
    Solver s;
    Info i;
    auto ak = s.analyse(A, ctrl, i);
    EXPECT_NE(ak.get(), nullptr);
    FactorKeep fk;
    EXPECT_EQ(s.factor(*ak, ctrl, i, fk), FactorStatus::Success);
    std::printf("[ScaleOnDelay] unscaled: %d delays, work ratio %.2f, n = %d\n", i.delayed_pivots,
                work_ratio(*ak, i), static_cast<int>(A.n));
    return work_ratio(*ak, i);
}

// The test KKTs are sized for the trigger on the Smf rule's tree: with
// CostModel's wider fronts the same delays inflate the front work less
// (FiresOnBadlyScaledKkt: work ratio 9.5 against 24+), and these tests are
// about the trigger, not the tree.
Control base_control() {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.amalgamation = Amalgamation::Smf;
    return c;  // Auto scaling, pivot_u = 0.01, default trigger ratio
}

// The rescue's matching scaling of M's values, M being a clean matrix of
// ak's pattern. Call it before ak's first armed factor (which releases
// ak.rescue_matcher).
std::vector<double> matching_scale_of(const AnalysisKeep &ak, const CscLower &M) {
    // The analysed values reuse the analysis' own matching scale, adopted or
    // rejected (a degree-order search's, see MatchingSearchOrder), as
    // factor() does; else, when the analysis kept its matcher
    // (AnalysisKeep::rescue_matcher, MatchingTrial::RejectedInitial), the
    // analysed values resume it; other values get a fresh matching. Its
    // searches end with at most 2% of the columns free (solve.cpp's
    // kRescueMaxFree).
    const std::vector<double> &analysed =
        !ak.matching_scale.empty() && !ak.matching_scale_partial ? ak.matching_scale : ak.rejected_matching_scale;
    if (!analysed.empty() && M.values == ak.cleaned.values) return analysed;
    const int max_free = static_cast<int>(0.02 * static_cast<double>(ak.n));
    MatchingResult mr;
    if (ak.rescue_matcher && M.values == ak.cleaned.values) {
        std::unique_ptr<SymmetricMatcher> m = ak.rescue_matcher->fork();
        m->set_stop_free(max_free);
        m->advance();
        mr = m->current();
    } else {
        CscLower clean = ak.cleaned;
        clean.values = M.values;
        SymmetricMatcher m(clean);
        m.set_stop_free(max_free);
        m.advance();
        mr = m.current();
    }
    EXPECT_EQ(mr.status, ErrorCode::Success);
    return mr.scale;
}

// The kept-matcher fixture: an OPF-like KKT whose initial matching rejects
// (MatchingTrial::RejectedInitial) under a 0.5-pass search budget, so the
// analysis keeps its matcher at the budget point
// (AnalysisKeep::rescue_matcher). spread: the Jacobian's magnitude spread
// (decades).
CscLower kept_matcher_kkt(double spread) { return scaled_opf_kkt(6000, 5000, 7, spread); }

// The spread at which the rescue fires (unscaled work ratio 89 on the Smf
// tree) and its scale stays accurate. At 3.0 neither the kept nor a fresh
// partial matching's scale passes expect_accurate_solve (omega 8e-5 and
// 4e-7: a 2%-free matching of these values leaves the delays at a work
// ratio of 97).
constexpr double kKeptSpread = 1.0;

Control kept_matcher_control() {
    Control c = base_control();
    c.matching_search_budget = 0.5;
    c.scale_on_delay_keep_matcher = true;  // off by default (gate failed)
    return c;
}

// The analysis' search budget in edges scanned
// (Control::matching_search_budget passes over the 2 nnz edges of the cost
// graph, as MatchingTrial.RejectedInitialKeepsTheBudgetMatcher).
long long budget_edges(const AnalysisKeep &ak, const Control &c) {
    return static_cast<long long>(c.matching_search_budget * 2.0 *
                                  static_cast<double>(ak.cleaned.row_idx.size()));
}

} // namespace

TEST(ScaleOnDelay, FiresOnBadlyScaledKktAndKeepsTheScale) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const Control ctrl = base_control();
    ASSERT_GE(ctrl.scale_on_delay_ratio, 1.0);
    ASSERT_GE(ctrl.scale_on_delay_reuse_tol, 0.0);

    // Premise: unscaled, the delays inflate the work well past the threshold.
    ASSERT_GT(unscaled_ratio(A, ctrl), 1.5 * ctrl.scale_on_delay_ratio);
    ASSERT_GT(ctrl.scale_on_delay_ratio, 4.0);

    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_TRUE(ak->matching_scale.empty());  // Auto alone would not scale

    NumericWorkspace ws;
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), ctrl, i1, fk, ws),
              FactorStatus::Success);
    std::printf("[ScaleOnDelay] rescaled: %d delays, work ratio %.2f\n", i1.delayed_pivots,
                work_ratio(*ak, i1));
    EXPECT_EQ(i1.delay_rescales, 1);
    EXPECT_EQ(i1.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_LE(work_ratio(*ak, i1), ctrl.scale_on_delay_ratio);
    EXPECT_EQ(i1.num_positive, kG * kG);
    EXPECT_EQ(i1.num_negative, kM);
    EXPECT_EQ(i1.num_zero, 0);
    EXPECT_TRUE(i1.inertia_exact);
    expect_accurate_solve(s, fk, ctrl, A);
    // The scale is the matching scaling of A's values, and the FactorKeep
    // keeps it for this analysis.
    const std::vector<double> scale_a = matching_scale_of(*ak, A);
    EXPECT_EQ(fk.scale, scale_a);
    EXPECT_EQ(fk.delay_analysis, ak->serial);
    EXPECT_EQ(fk.delay_scale, scale_a);

    // Next factorization into the same FactorKeep, values that the kept
    // scale still fits (the same bad scaling, a slightly larger H
    // diagonal): the kept scale is reused, no stopped try, no matching.
    const CscLower A1 = kkt(kG, kM, 4.15, 3.0);
    Info i1b;
    ASSERT_EQ(s.factor(*ak, A1, ctrl, i1b, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i1b.delay_rescales, 0);
    EXPECT_EQ(i1b.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_EQ(fk.scale, scale_a);
    EXPECT_EQ(fk.delay_scale, scale_a);
    EXPECT_EQ(i1b.num_positive, kG * kG);
    EXPECT_EQ(i1b.num_negative, kM);
    expect_accurate_solve(s, fk, ctrl, A1);

    // Values the kept scale no longer fits (another bad scaling): a new
    // matching scaling of them, kept in its place.
    const CscLower A2 = kkt(kG, kM, 4.6, 3.0, -0.1, 11);
    ASSERT_GT(unscaled_ratio(A2, ctrl), ctrl.scale_on_delay_ratio);
    Info i2;
    ASSERT_EQ(s.factor(*ak, A2, ctrl, i2, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i2.delay_rescales, 1);
    EXPECT_EQ(i2.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_LE(work_ratio(*ak, i2), ctrl.scale_on_delay_ratio);
    const std::vector<double> scale_a2 = matching_scale_of(*ak, A2);
    EXPECT_EQ(fk.scale, scale_a2);
    EXPECT_EQ(fk.delay_scale, scale_a2);
    EXPECT_EQ(i2.num_positive, kG * kG);
    EXPECT_EQ(i2.num_negative, kM);
    expect_accurate_solve(s, fk, ctrl, A2);

    // With reuse off, every later factorization computes a new matching.
    {
        Control rc = ctrl;
        rc.scale_on_delay_reuse_tol = -1.0;
        Info ir;
        ASSERT_EQ(s.factor(*ak, A2, rc, ir, fk, ws), FactorStatus::Success);
        EXPECT_EQ(ir.delay_rescales, 1);
        EXPECT_EQ(fk.scale, scale_a2);
    }

    // Another FactorKeep, or another analysis, starts over.
    FactorKeep fk_other;
    Info i_other;
    ASSERT_EQ(s.factor(*ak, A2, ctrl, i_other, fk_other, ws), FactorStatus::Success);
    EXPECT_EQ(i_other.delay_rescales, 1);
    Info ib;
    auto ak_b = s.analyse(A, ctrl, ib);
    ASSERT_NE(ak_b.get(), nullptr);
    EXPECT_NE(ak_b->serial, ak->serial);
    Info i3;
    ASSERT_EQ(s.factor(*ak_b, ctrl, i3, fk), FactorStatus::Success);
    EXPECT_EQ(i3.delay_rescales, 1);
    EXPECT_EQ(fk.delay_analysis, ak_b->serial);
}

// Well scaled: no delays, nothing fires, no scaling.
TEST(ScaleOnDelay, DoesNotFireOnWellScaledKkt) {
    const CscLower A = kkt(kG, kM, 4.1, 0.0);
    const Control ctrl = base_control();
    ASSERT_LE(unscaled_ratio(A, ctrl), ctrl.scale_on_delay_ratio);

    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, ctrl, i1, fk), FactorStatus::Success);
    EXPECT_EQ(i1.delay_rescales, 0);
    EXPECT_EQ(i1.scaling_used, static_cast<int>(ScalingMethod::None));
    EXPECT_TRUE(fk.scale.empty());
    EXPECT_EQ(fk.delay_analysis, 0u);
    expect_accurate_solve(s, fk, ctrl, A);
}

// Delays that cost little (the zero (2,2) block of a well scaled KKT:
// constraint rows pivot one level up) do not fire the trigger: a factor that
// delays but stays under the work ratio is the plain one, bit for bit.
TEST(ScaleOnDelay, CheapDelaysDoNotFire) {
    const CscLower A = kkt(kG, kM, 4.1, 0.0, 0.0);
    Control ctrl = base_control();
    Control off = ctrl;
    off.scale_on_delay_ratio = -1.0;
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk, fk_off;
    Info i1, i_off;
    ASSERT_EQ(s.factor(*ak, off, i_off, fk_off), FactorStatus::Success);
    std::printf("[ScaleOnDelay] zero (2,2): %d delays, work ratio %.2f\n", i_off.delayed_pivots,
                work_ratio(*ak, i_off));
    ASSERT_GT(i_off.delayed_pivots, 0) << "premise: it delays";
    ASSERT_LE(work_ratio(*ak, i_off), ctrl.scale_on_delay_ratio) << "premise: cheaply";
    ASSERT_EQ(s.factor(*ak, ctrl, i1, fk), FactorStatus::Success);
    EXPECT_EQ(i1.delay_rescales, 0);
    EXPECT_EQ(i1.delayed_pivots, i_off.delayed_pivots);
    EXPECT_EQ(fk.factor_values, fk_off.factor_values);
}

// Below pivot u = 1e-6 (IPOPT's 1e-8: the ipopt track, the MA97 plugin,
// IpmSession) the trigger is never armed: the factorization is the one with
// the feature off, bit for bit, even with a ratio of 0 (which would stop
// after the first front) on a badly scaled KKT with a zero (2,2) block.
TEST(ScaleOnDelay, NeverFiresBelowWeakPivotRatio) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0, 0.0);
    for (const double u : {1e-8, 9.9e-7}) {
        Control ctrl = base_control();
        ctrl.pivot_u = u;
        ctrl.scale_on_delay_ratio = 0.0;
        Control off = ctrl;
        off.scale_on_delay_ratio = -1.0;

        Solver s;
        Info ia;
        auto ak = s.analyse(A, ctrl, ia);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk, fk_off;
        Info i1, i_off;
        const FactorStatus st = s.factor(*ak, ctrl, i1, fk);
        ASSERT_EQ(s.factor(*ak, off, i_off, fk_off), st);
        std::printf("[ScaleOnDelay] u = %g: %d delays, work ratio %.2f\n", u, i1.delayed_pivots,
                    work_ratio(*ak, i1));
        EXPECT_GT(i1.delayed_pivots, 0) << "premise: it delays";
        EXPECT_GT(i1.front_work, 0.0);
        EXPECT_EQ(i1.delay_rescales, 0);
        EXPECT_EQ(i1.scaling_used, static_cast<int>(ScalingMethod::None));
        EXPECT_TRUE(fk.scale.empty());
        EXPECT_EQ(fk.delay_analysis, 0u);
        EXPECT_EQ(i1.delayed_pivots, i_off.delayed_pivots);
        EXPECT_EQ(i1.num_negative, i_off.num_negative);
        EXPECT_EQ(fk.factor_values, fk_off.factor_values);
        EXPECT_EQ(fk.perm, fk_off.perm);
    }
}

// Not armed outside plain threshold pivoting with Auto scaling, with
// max_negative_pivots, or on the legacy tree_parallel engine. A ratio of 0
// would stop after the first front if it were armed, and the badly scaled
// KKT with a zero (2,2) block delays under every pivoting variant below
// (static pivoting perturbs instead, but its fronts still count work).
TEST(ScaleOnDelay, OnlyUnderAutoScalingAndThresholdPivoting) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0, 0.0);
    for (int variant = 0; variant < 6; ++variant) {
        Control ctrl = base_control();
        ctrl.scale_on_delay_ratio = 0.0;
        if (variant == 0) ctrl.scaling = ScalingMethod::None;
        if (variant == 1) ctrl.static_pivot_tol = 1e-8;
        if (variant == 2) ctrl.fallback_static_pivot_tol = 1e-8;
        if (variant == 3) ctrl.delay_budget = 1000000;
        if (variant == 4) ctrl.max_negative_pivots = static_cast<int>(A.n);
        if (variant == 5) {
            ctrl.tree_parallel = true;
            ctrl.num_threads = 2;
            ctrl.auto_threads = false;  // small test trees
        }
        Solver s;
        Info ia;
        auto ak = s.analyse(A, ctrl, ia);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        Info i1;
        (void)s.factor(*ak, ctrl, i1, fk);
        // Deterministic mode ignores tree_parallel (the lean engine runs).
        if (variant == 5 && deterministic_mode(ctrl)) continue;
        EXPECT_GT(i1.front_work, 0.0) << "variant " << variant;
        EXPECT_EQ(i1.delay_rescales, 0) << "variant " << variant;
        EXPECT_EQ(fk.delay_analysis, 0u) << "variant " << variant;
    }
}

// max_negative_pivots arms an early stop of its own; with it the trigger is
// off, so the factorization, its status and the FactorKeep state are the
// same at 1, 2 and 4 threads (deterministic mode), whichever limit would
// have been crossed first.
TEST(ScaleOnDelay, MaxNegativePivotsDisarmsTheTrigger) {
    const CscLower A = kkt(90, 300, 4.1, 3.0);  // n < 10000: no auto matching
    for (const int neg_limit : {300 - 1, 300}) {  // trips / does not trip
        std::vector<double> x_ref;
        FactorStatus st_ref = FactorStatus::Success;
        for (const int threads : {1, 2, 4}) {
            Control ctrl = base_control();
            ctrl.deterministic = true;
            ctrl.num_threads = threads;
            ctrl.auto_threads = false;  // small test trees
            ctrl.max_negative_pivots = neg_limit;
            Solver s;
            Info ia;
            auto ak = s.analyse(A, ctrl, ia);
            ASSERT_NE(ak.get(), nullptr);
            FactorKeep fk;
            Info i1;
            const FactorStatus st = s.factor(*ak, ctrl, i1, fk);
            EXPECT_EQ(i1.delay_rescales, 0);
            EXPECT_EQ(fk.delay_analysis, 0u);
            EXPECT_TRUE(fk.scale.empty());
            if (threads > 1) EXPECT_GT(i1.parallel_subtrees, 1) << "threads " << threads;
            if (threads == 1) st_ref = st;
            EXPECT_EQ(st, st_ref) << "threads " << threads;
            if (st != FactorStatus::Success) continue;
            std::vector<double> x(static_cast<std::size_t>(A.n), 1.0);
            Info si;
            ASSERT_EQ(s.solve(fk, ctrl, si, x.data(), static_cast<int>(A.n), 1), 0);
            if (x_ref.empty()) x_ref = x;
            else EXPECT_EQ(x, x_ref) << "threads " << threads;
        }
        EXPECT_EQ(st_ref, neg_limit < 300 ? FactorStatus::InertiaExceeded : FactorStatus::Success);
    }
}

// The engine's work limit: stops only past it, and a run within it is the
// unarmed run bit for bit; the workspace stays fit after a stop.
TEST(ScaleOnDelay, WorkLimitStopsOnlyPastTheCount) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    Control ctrl = base_control();
    ctrl.scale_on_delay_ratio = -1.0;
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    FactorKeep f_ref, f_hi, f_lo;
    Info i_ref, i_hi, i_lo;
    ASSERT_EQ(factor_indef(*ak, A, ctrl, i_ref, f_ref, ws), FactorStatus::Success);
    const double work = i_ref.front_work;
    ASSERT_GT(work, delay_free_front_work(*ak));  // it delays
    ASSERT_EQ(factor_indef(*ak, A, ctrl, i_hi, f_hi, ws, work), FactorStatus::Success);
    EXPECT_EQ(i_hi.front_work, work);
    EXPECT_EQ(i_hi.delayed_pivots, i_ref.delayed_pivots);
    EXPECT_EQ(f_hi.factor_values, f_ref.factor_values);
    EXPECT_EQ(factor_indef(*ak, A, ctrl, i_lo, f_lo, ws, work - 1.0),
              FactorStatus::MaxPivotDelays);
    EXPECT_GT(i_lo.front_work, work - 1.0);
    EXPECT_TRUE(f_lo.factor_values.empty());
    FactorKeep f_again;
    Info i_again;
    ASSERT_EQ(factor_indef(*ak, A, ctrl, i_again, f_again, ws), FactorStatus::Success);
    EXPECT_EQ(i_again.delayed_pivots, i_ref.delayed_pivots);
    EXPECT_EQ(f_again.factor_values, f_ref.factor_values);
}

// A factor the work limit stopped is never solved with, not even under
// continue_on_singular (which lets Singular factors through).
TEST(ScaleOnDelay, SolveRejectsWorkLimitStoppedFactor) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    Control ctrl = base_control();
    ctrl.scale_on_delay_ratio = -1.0;
    ctrl.continue_on_singular = true;
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    FactorKeep f;
    Info info;
    ASSERT_EQ(factor_indef(*ak, A, ctrl, info, f, ws, 0.0), FactorStatus::MaxPivotDelays);
    std::vector<double> b(static_cast<std::size_t>(A.n), 1.0);
    SolveWorkspace sws;
    Info is;
    EXPECT_EQ(s.solve(f, ctrl, is, b.data(), static_cast<int>(A.n), 1, sws),
              static_cast<int>(ErrorCode::WorkLimitExceeded));
    EXPECT_EQ(is.status, ErrorCode::WorkLimitExceeded);
    EXPECT_TRUE(std::all_of(b.begin(), b.end(), [](double x) { return x == 1.0; }));
}

// The same on the subtree-parallel engine (a tree split at 4 threads; the
// threads count the work into one shared total). A limit of a tenth of the
// work stops while the subtrees are being factored, before the top of the
// tree; the workspace is fit for the next factorization after that.
TEST(ScaleOnDelay, WorkLimitOnParallelSubtrees) {
    const CscLower A = kkt(100, 200, 4.1, 0.0);
    for (const bool det : {false, true}) {
        Control ctrl = base_control();
        ctrl.scale_on_delay_ratio = -1.0;
        ctrl.num_threads = 4;
        ctrl.auto_threads = false;  // small test trees
        ctrl.deterministic = det;
        Solver s;
        Info ia;
        auto ak = s.analyse(A, ctrl, ia);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace ws;
        FactorKeep f_ref, f_hi, f_lo, f_again;
        Info i_ref, i_hi, i_lo, i_again;
        ASSERT_EQ(factor_indef(*ak, A, ctrl, i_ref, f_ref, ws), FactorStatus::Success);
        ASSERT_GT(i_ref.parallel_subtrees, 1) << "premise: the subtree engine ran";
        ASSERT_GT(i_ref.top_part_fronts, 0) << "premise: it has a top part";
        const double work = i_ref.front_work;
        ASSERT_GT(work, 0.0);
        ASSERT_EQ(factor_indef(*ak, A, ctrl, i_hi, f_hi, ws, work), FactorStatus::Success);
        EXPECT_EQ(i_hi.front_work, work);
        EXPECT_EQ(i_hi.delayed_pivots, i_ref.delayed_pivots);
        EXPECT_EQ(i_hi.num_negative, i_ref.num_negative);
        const double limit = 0.1 * work;
        EXPECT_EQ(factor_indef(*ak, A, ctrl, i_lo, f_lo, ws, limit), FactorStatus::MaxPivotDelays);
        EXPECT_GT(i_lo.front_work, limit);
        // Stopped in the subtree phase: no front of the top part ran.
        EXPECT_EQ(i_lo.top_part_fronts, 0);
        EXPECT_LT(i_lo.front_work, 0.5 * work);
        EXPECT_TRUE(f_lo.factor_values.empty());
        ASSERT_EQ(factor_indef(*ak, A, ctrl, i_again, f_again, ws), FactorStatus::Success);
        EXPECT_EQ(i_again.front_work, work);
        EXPECT_EQ(i_again.delayed_pivots, i_ref.delayed_pivots);
        EXPECT_EQ(i_again.num_negative, i_ref.num_negative);
        if (det) {
            // Deterministic mode: the same solution, bit for bit.
            const auto solve = [&](const FactorKeep &f) {
                std::vector<double> x(static_cast<std::size_t>(A.n), 1.0);
                Info si;
                EXPECT_EQ(s.solve(f, ctrl, si, x.data(), static_cast<int>(A.n), 1), 0);
                return x;
            };
            EXPECT_EQ(solve(f_again), solve(f_ref));
        }
        std::printf("[ScaleOnDelay] subtrees %d (det %d): work %.3e, stopped at %.3e\n",
                    i_ref.parallel_subtrees, det ? 1 : 0, work, i_lo.front_work);
    }
}

// Deterministic mode: the trigger decision, and the solution, are the same
// at every thread count, the subtree engine running at 2 and 4.
TEST(ScaleOnDelay, DeterministicAcrossThreadCounts) {
    const CscLower A = kkt(90, 300, 4.1, 3.0);  // n < 10000: no auto matching
    std::vector<double> x_ref;
    for (const int threads : {1, 2, 4}) {
        Control ctrl = base_control();
        ctrl.deterministic = true;
        ctrl.num_threads = threads;
        ctrl.auto_threads = false;  // small test trees
        Solver s;
        Info ia;
        auto ak = s.analyse(A, ctrl, ia);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        Info i1;
        ASSERT_EQ(s.factor(*ak, ctrl, i1, fk), FactorStatus::Success);
        EXPECT_EQ(i1.delay_rescales, 1) << "threads " << threads;
        if (threads > 1) EXPECT_GT(i1.parallel_subtrees, 1) << "threads " << threads;
        std::vector<double> x(static_cast<std::size_t>(A.n), 1.0);
        Info si;
        ASSERT_EQ(s.solve(fk, ctrl, si, x.data(), static_cast<int>(A.n), 1), 0);
        if (x_ref.empty()) x_ref = x;
        else EXPECT_EQ(x, x_ref) << "threads " << threads;
    }
}

// A singular matrix (one variable's row and column numerically zero, so it
// has no matching scaling: the no-matching branch) whose delays fire the
// trigger: the result keeps the plain factorization's status and inertia (a
// zero pivot), and the FactorKeep remembers that there is no matching
// scaling, so a later factorization runs unarmed instead of paying the
// stopped try again.
TEST(ScaleOnDelay, SingularMatrixKeepsStatusAndInertia) {
    CscLower A = kkt(kG, kM, 4.1, 3.0);
    const Int z = 5;
    for (Int j = 0; j < A.n; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            if (j == z || A.row_idx[static_cast<std::size_t>(p)] == z) A.values[static_cast<std::size_t>(p)] = 0.0;
    Control ctrl = base_control();
    ctrl.continue_on_singular = true;
    Control off = ctrl;
    off.scale_on_delay_ratio = -1.0;
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk, fk_off;
    Info i1, i_off;
    const FactorStatus st_off = s.factor(*ak, off, i_off, fk_off);
    std::printf("[ScaleOnDelay] singular: off %d delays, work ratio %.2f, status %d\n",
                i_off.delayed_pivots, work_ratio(*ak, i_off), static_cast<int>(st_off));
    ASSERT_GT(work_ratio(*ak, i_off), ctrl.scale_on_delay_ratio);
    for (int call = 0; call < 2; ++call) {
        const FactorStatus st = s.factor(*ak, ctrl, i1, fk);
        EXPECT_EQ(st, st_off);
        EXPECT_EQ(i1.status, i_off.status);
        EXPECT_EQ(i1.num_zero, i_off.num_zero);
        EXPECT_EQ(i1.num_positive, i_off.num_positive);
        EXPECT_EQ(i1.num_negative, i_off.num_negative);
        EXPECT_GE(i1.num_zero, 1);
        EXPECT_EQ(i1.delay_rescales, 0);
        // The first call stopped a try; the second, same zero pattern, ran
        // unarmed: the memo held.
        EXPECT_EQ(i1.delay_stops, call == 0 ? 1 : 0) << "call " << call;
        EXPECT_EQ(fk.delay_analysis, ak->serial);  // no matching scaling: unarmed from now
        EXPECT_TRUE(fk.delay_scale.empty());
        EXPECT_EQ(fk.factor_values, fk_off.factor_values);
    }
    // The memo holds only for that zero pattern: values with the zero
    // row and column filled in again (the plain badly scaled KKT) are tried
    // and rescued.
    const CscLower full = kkt(kG, kM, 4.1, 3.0);
    Info i2;
    NumericWorkspace ws;
    ASSERT_EQ(s.factor(*ak, full, ctrl, i2, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i2.delay_rescales, 1);
    EXPECT_EQ(fk.delay_analysis, ak->serial);
    EXPECT_FALSE(fk.delay_scale.empty());
    EXPECT_EQ(i2.num_negative, kM);
}

// The reuse decision on a probe error: at most 10x the fresh one (floored
// at 1e-14) and at most 1e-9; no fresh error recorded: the cap only; a NaN
// or infinite error fails.
TEST(ScaleOnDelay, ReuseProbeRule) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // Relative: 10x the fresh error.
    EXPECT_TRUE(scale_on_delay_reuse_passes(9.9e-11, 1e-11));
    EXPECT_FALSE(scale_on_delay_reuse_passes(1.1e-10, 1e-11));
    // The floor: a fresh error near eps does not make rounding fail it.
    EXPECT_TRUE(scale_on_delay_reuse_passes(9e-14, 1e-16));
    EXPECT_FALSE(scale_on_delay_reuse_passes(1.1e-13, 1e-16));
    // The cap, whatever the fresh error.
    EXPECT_TRUE(scale_on_delay_reuse_passes(1e-9, 1.0));
    EXPECT_FALSE(scale_on_delay_reuse_passes(1.1e-9, 1.0));
    EXPECT_FALSE(scale_on_delay_reuse_passes(1.1e-9, -1.0));
    // No fresh error recorded (negative): the cap only.
    EXPECT_TRUE(scale_on_delay_reuse_passes(9e-10, -1.0));
    // Not finite.
    EXPECT_FALSE(scale_on_delay_reuse_passes(nan, 1.0));
    EXPECT_FALSE(scale_on_delay_reuse_passes(inf, -1.0));
}

// The probe: tiny on an accurate factor of its values, and infinite as
// soon as one row's error is not finite (values with a NaN), never a finite
// maximum over the other rows.
TEST(ScaleOnDelay, ProbeErrorAndNaN) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const Control ctrl = base_control();
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, ctrl, i1, fk), FactorStatus::Success);
    ASSERT_EQ(i1.delay_rescales, 1);
    const double omega = scale_on_delay_probe(*ak, A.values, fk, ctrl);
    std::printf("[ScaleOnDelay] probe of a fresh factor: %.3e (recorded %.3e)\n", omega,
                fk.delay_omega);
    EXPECT_GT(omega, 0.0);
    EXPECT_LT(omega, 1e-12);
    EXPECT_EQ(omega, fk.delay_omega);  // what the trigger recorded
    std::vector<double> bad = A.values;
    bad[bad.size() / 2] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(scale_on_delay_probe(*ak, bad, fk, ctrl), std::numeric_limits<double>::infinity());
}

// A real reuse that the probe rejects: PGLib case500's IPOPT capture,
// factored at the replay's 5 evenly spaced steps into one FactorKeep. On
// the tuning split the kept scale of step 13 fits step 52 but its factor
// probes 75x the fresh error there, so step 52 is refactored with a new
// matching scaling; every step solves to the accuracy gate. Skipped without
// the (untracked) fixture.
TEST(ScaleOnDelay, ProbeRejectsAStaleReuseOnPglib) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" /
                         "fixtures" / "ipopt_hard" / "pglib_opf_case500_goc__api";
    const fs::path mtx = dir / "pglib_opf_case500_goc__api.mtx";
    const fs::path vals_path = dir / "pglib_opf_case500_goc__api.mtx.vals";
    if (!fs::exists(mtx) || !fs::exists(vals_path)) GTEST_SKIP() << "fixture missing";
    auto mm = read_matrix_market(mtx.string());
    ASSERT_TRUE(mm.ok) << mm.error_message;
    const CscLower A = std::move(mm.matrix);
    const std::size_t nnz = A.values.size();
    std::ifstream in(vals_path, std::ios::binary);
    const Control ctrl = base_control();
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    FactorKeep fk;
    std::vector<double> vals(nnz);
    const std::size_t steps[] = {0, 13, 26, 39, 52};
    int rescales_at_52 = -1, stops_at_52 = -1, rescales_at_39 = -1;
    for (const std::size_t step : steps) {
        in.seekg(static_cast<std::streamoff>(step * nnz * sizeof(double)));
        in.read(reinterpret_cast<char *>(vals.data()), static_cast<std::streamsize>(nnz * sizeof(double)));
        ASSERT_TRUE(static_cast<bool>(in)) << "step " << step;
        Info info;
        ASSERT_EQ(s.factor(*ak, std::span<const double>(vals), ctrl, info, fk, ws),
                  FactorStatus::Success)
            << "step " << step;
        std::printf("[ScaleOnDelay] case500 step %zu: rescales %d stops %d\n", step,
                    info.delay_rescales, info.delay_stops);
        if (step == 39) rescales_at_39 = info.delay_rescales;
        if (step == 52) {
            rescales_at_52 = info.delay_rescales;
            stops_at_52 = info.delay_stops;
        }
        CscLower M = A;
        M.values = vals;
        expect_accurate_solve(s, fk, ctrl, M);
    }
    EXPECT_EQ(rescales_at_39, 0);  // reused
    EXPECT_EQ(rescales_at_52, 1);  // rejected by the probe, recomputed
    EXPECT_EQ(stops_at_52, 1);
}

// A kept scale that passes the fit test but whose factorization trips the
// work limit (the scale of other values, the fit test opened up): the try
// is stopped and the matching scaling of the new values computed and kept.
TEST(ScaleOnDelay, ReuseThatTripsTheLimitRecomputes) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const CscLower A2 = kkt(kG, kM, 4.1, 3.0, -0.1, 11);  // another bad scaling
    Control ctrl = base_control();
    ctrl.scale_on_delay_reuse_tol = 100.0;  // every kept scale "fits"
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, A, ctrl, i1, fk, ws), FactorStatus::Success);
    ASSERT_EQ(i1.delay_rescales, 1);
    // Premise: A2 with A's scale blows up past the limit.
    {
        Control uc = ctrl;
        uc.scale_on_delay_ratio = -1.0;
        uc.scaling = ScalingMethod::UserSupplied;
        uc.user_scale = fk.delay_scale.data();
        FactorKeep f0;
        Info i0;
        NumericWorkspace w0;
        ASSERT_EQ(s.factor(*ak, A2, uc, i0, f0, w0), FactorStatus::Success);
        ASSERT_GT(work_ratio(*ak, i0), ctrl.scale_on_delay_ratio);
    }
    // The final factorization alone: A2 with its own matching scaling.
    double fresh_seconds = 0.0;
    {
        const std::vector<double> scale2 = matching_scale_of(*ak, A2);
        Control uc = ctrl;
        uc.scale_on_delay_ratio = -1.0;
        uc.scaling = ScalingMethod::UserSupplied;
        uc.user_scale = scale2.data();
        FactorKeep f0;
        Info i0;
        NumericWorkspace w0;
        ASSERT_EQ(s.factor(*ak, A2, uc, i0, f0, w0), FactorStatus::Success);
        fresh_seconds = i0.factor_seconds;
    }
    Info i2;
    ASSERT_EQ(s.factor(*ak, A2, ctrl, i2, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i2.delay_rescales, 1);
    EXPECT_EQ(i2.delay_stops, 1);
    EXPECT_EQ(i2.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_LE(work_ratio(*ak, i2), ctrl.scale_on_delay_ratio);
    EXPECT_EQ(fk.delay_scale, matching_scale_of(*ak, A2));
    EXPECT_EQ(i2.num_negative, kM);
    // factor_seconds covers the spent try (16x the work) and the matching
    // as well as that final factorization.
    std::printf("[ScaleOnDelay] tripped reuse: %.3f ms, final factorization alone %.3f ms\n",
                1e3 * i2.factor_seconds, 1e3 * fresh_seconds);
    EXPECT_GT(i2.factor_seconds, 1.5 * fresh_seconds);
    expect_accurate_solve(s, fk, ctrl, A2);
}

// With the matching ordering Auto applies the analysis' matching scale, of
// the analysed values. New values scaled differently make it stale: the
// factorization blows up past the threshold and is redone with the new
// values' own matching scaling. The analysed values themselves never stop.
TEST(ScaleOnDelay, ReplacesStaleAnalysisMatchingScale) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const CscLower A2 = kkt(kG, kM, 4.1, 3.0, -0.1, 99);  // another D
    Control ctrl = base_control();
    ctrl.ordering = OrderingMethod::Matching;
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_EQ(ia.ordering_used, static_cast<int>(OrderingMethod::Matching));
    ASSERT_FALSE(ak->matching_scale.empty());

    // Premise: A2 with the analysis' (stale) scale blows up past the threshold.
    {
        Control off = ctrl;
        off.scale_on_delay_ratio = -1.0;
        FactorKeep f0;
        Info i0;
        NumericWorkspace w0;
        ASSERT_EQ(s.factor(*ak, A2, off, i0, f0, w0), FactorStatus::Success);
        std::printf("[ScaleOnDelay] stale analysis scale: %d delays, work ratio %.2f\n",
                    i0.delayed_pivots, work_ratio(*ak, i0));
        ASSERT_GT(work_ratio(*ak, i0), ctrl.scale_on_delay_ratio);
    }
    NumericWorkspace ws;
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, A, ctrl, i1, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i1.delay_rescales, 0);
    EXPECT_EQ(i1.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_EQ(fk.scale, ak->matching_scale);
    EXPECT_EQ(fk.delay_analysis, 0u);

    Info i2;
    ASSERT_EQ(s.factor(*ak, A2, ctrl, i2, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i2.delay_rescales, 1);
    EXPECT_LE(work_ratio(*ak, i2), ctrl.scale_on_delay_ratio);
    EXPECT_NE(fk.scale, ak->matching_scale);
    EXPECT_EQ(fk.delay_analysis, ak->serial);
    EXPECT_EQ(i2.num_positive, kG * kG);
    EXPECT_EQ(i2.num_negative, kM);
    expect_accurate_solve(s, fk, ctrl, A2);
}


// Control::scale_on_delay_next_ratio: a factorization the work limit lets
// through, but past the next ratio, is kept as it is; the next one of that
// analysis into that FactorKeep runs under the next ratio as its limit. If
// its delays stay heavy that tighter try stops and the matching scaling
// takes over; values that factor cheaply pass it unscaled. Off (negative),
// the next one runs under the full limit.
TEST(ScaleOnDelay, CostlyButLetThroughTightensTheNextLimit) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const CscLower A1 = kkt(kG, kM, 4.15, 3.0);  // the same bad scaling
    const CscLower E = kkt(kG, kM, 4.1, 0.0);    // well scaled
    Control ctrl = base_control();
    const double r = unscaled_ratio(A, ctrl);
    ASSERT_GT(r, 4.0) << "premise: the delays inflate the work";
    ASSERT_LT(unscaled_ratio(E, ctrl), 0.5 * r) << "premise: E factors cheaply";
    ctrl.scale_on_delay_ratio = 2.0 * r;  // lets A through
    ctrl.scale_on_delay_next_ratio = 0.5 * r;

    struct Case { const CscLower *next_values; bool on; bool rescales; };
    for (const Case cs : {Case{&A1, true, true}, Case{&A1, false, false}, Case{&E, true, false}}) {
        Control c = ctrl;
        if (!cs.on) c.scale_on_delay_next_ratio = -1.0;
        Solver s;
        Info info;
        auto ak = s.analyse(A, c, info);
        ASSERT_NE(ak.get(), nullptr);
        ASSERT_TRUE(ak->matching_scale.empty());
        NumericWorkspace ws;
        FactorKeep fk;
        Info i1;
        ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), c, i1, fk, ws),
                  FactorStatus::Success);
        EXPECT_EQ(i1.delay_rescales, 0);
        EXPECT_EQ(i1.delay_stops, 0);
        EXPECT_EQ(i1.scaling_used, static_cast<int>(ScalingMethod::None));
        EXPECT_EQ(fk.delay_next, cs.on);
        expect_accurate_solve(s, fk, c, A);

        const CscLower &B = *cs.next_values;
        Info i2;
        ASSERT_EQ(s.factor(*ak, B, c, i2, fk, ws), FactorStatus::Success);
        EXPECT_FALSE(fk.delay_next);
        EXPECT_EQ(i2.delay_stops, cs.rescales ? 1 : 0);
        EXPECT_EQ(i2.delay_rescales, cs.rescales ? 1 : 0);
        EXPECT_EQ(i2.scaling_used, static_cast<int>(cs.rescales ? ScalingMethod::Matching
                                                                : ScalingMethod::None));
        if (cs.rescales) EXPECT_EQ(fk.delay_scale, matching_scale_of(*ak, B));
        EXPECT_EQ(i2.num_positive, kG * kG);
        EXPECT_EQ(i2.num_negative, kM);
        expect_accurate_solve(s, fk, c, B);
    }
}

TEST(ScaleOnDelay, RescuePiecesTimed) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);  // fires (see FiresOnBadlyScaledKkt)
    const Control ctrl = base_control();
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    NumericWorkspace ws;
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), ctrl, i1, fk, ws),
              FactorStatus::Success);
    EXPECT_EQ(i1.delay_stops, 1);
    EXPECT_GT(i1.delay_try_seconds, 0.0);
    EXPECT_GT(i1.delay_matching_seconds, 0.0);
    EXPECT_LE(i1.delay_try_seconds + i1.delay_matching_seconds, i1.factor_seconds);

    // Second factor, same values, same FactorKeep: the kept scale is reused.
    Info i1b;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), ctrl, i1b, fk, ws),
              FactorStatus::Success);
    EXPECT_EQ(i1b.delay_try_seconds, 0.0);
    EXPECT_EQ(i1b.delay_matching_seconds, 0.0);

    // Well scaled values: no try, no matching.
    const CscLower B = kkt(kG, kM, 4.1, 0.0);
    FactorKeep fk2;
    Info i2;
    auto ak2 = s.analyse(B, ctrl, info);
    ASSERT_NE(ak2.get(), nullptr);
    ASSERT_EQ(s.factor(*ak2, std::span<const double>(B.values), ctrl, i2, fk2, ws),
              FactorStatus::Success);
    EXPECT_EQ(i2.delay_try_seconds, 0.0);
    EXPECT_EQ(i2.delay_matching_seconds, 0.0);
}

// The analysed values of a RejectedInitial analysis: the rescue resumes the
// analysis' matcher from its budget point (the same matching as a Degree-
// order matcher advanced to the budget, then stopped at 2% free) instead of
// matching from scratch, and releases it.
TEST(ScaleOnDelay, RescueResumesTheAnalysisMatcher) {
    const CscLower A = kept_matcher_kkt(kKeptSpread);
    const Control ctrl = kept_matcher_control();
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    // Premise: the trial rejects on its initial matching, and unscaled the
    // delays trip the work limit.
    ASSERT_EQ(info.matching_trial, MatchingTrial::RejectedInitial);
    ASSERT_GT(unscaled_ratio(A, ctrl), ctrl.scale_on_delay_ratio);

    ASSERT_NE(ak->rescue_matcher, nullptr);
    const int n = static_cast<int>(ak->n);
    SymmetricMatcher ref(ak->cleaned, MatchingSearchOrder::Degree);
    ASSERT_FALSE(ref.advance(budget_edges(*ak, ctrl)));
    ref.set_stop_free(static_cast<int>(0.02 * static_cast<double>(n)));
    ref.advance();
    const std::vector<double> expect = ref.current().scale;
    EXPECT_EQ(matching_scale_of(*ak, A), expect);

    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, ctrl, i1, fk), FactorStatus::Success);
    std::printf("[ScaleOnDelay] kept matcher: %d delays, work ratio %.2f\n", i1.delayed_pivots,
                work_ratio(*ak, i1));
    EXPECT_EQ(i1.delay_rescales, 1);
    EXPECT_EQ(i1.delay_matcher_reused, 1);
    EXPECT_EQ(fk.scale, expect);  // bitwise
    EXPECT_EQ(ak->rescue_matcher, nullptr);  // released
    expect_accurate_solve(s, fk, ctrl, A);
}

// Values other than the analysed ones (same pattern): the kept matcher
// describes the analysed values only, so the rescue matches them afresh
// (Column order), and still releases the kept matcher.
TEST(ScaleOnDelay, OtherValuesGetAFreshMatching) {
    const CscLower A = kept_matcher_kkt(kKeptSpread);
    const Control ctrl = kept_matcher_control();
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_EQ(info.matching_trial, MatchingTrial::RejectedInitial);
    ASSERT_NE(ak->rescue_matcher, nullptr);

    CscLower B = A;  // H's diagonal x1.01
    for (Int j = 0; j < 6000; ++j)
        for (Int p = B.col_ptr[static_cast<std::size_t>(j)]; p < B.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            if (B.row_idx[static_cast<std::size_t>(p)] == j) B.values[static_cast<std::size_t>(p)] *= 1.01;
    const std::vector<double> fresh = matching_scale_of(*ak, B);

    NumericWorkspace ws;
    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, B, ctrl, i, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i.delay_rescales, 1);
    EXPECT_EQ(i.delay_matcher_reused, 0);
    EXPECT_EQ(fk.delay_scale, fresh);
    EXPECT_EQ(ak->rescue_matcher, nullptr);
}

// A RejectedInitial analysis whose factor stays under the work limit: no
// rescue, and the kept matcher is released all the same.
TEST(ScaleOnDelay, KeptMatcherReleasedWhenTheRescueDoesNotFire) {
    const CscLower A = kept_matcher_kkt(0.0);
    // The default (CostModel) tree: on the Smf rule's narrower fronts the
    // same delays trip the limit (work ratio 52 against 3.3).
    Control ctrl = kept_matcher_control();
    ctrl.amalgamation = Control{}.amalgamation;
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_EQ(info.matching_trial, MatchingTrial::RejectedInitial);
    ASSERT_LE(unscaled_ratio(A, ctrl), ctrl.scale_on_delay_ratio);
    ASSERT_NE(ak->rescue_matcher, nullptr);

    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, ctrl, i, fk), FactorStatus::Success);
    EXPECT_EQ(i.delay_stops, 0);
    EXPECT_EQ(ak->rescue_matcher, nullptr);
}

// A kept matcher whose resumed searches turn out structurally singular:
// the rescue falls back as an empty scale does, running unarmed and
// recording the zero signature, so the next factor of the same values runs
// no try. (An all-zero column cannot be kept: the matcher's cost graph
// rejects it, so the analysis runs no trial. Instead the constraint rows
// that touch the first 200 variables keep only those entries, pattern
// kept: about 450 such columns over 200 rows, a structural deficiency
// past the rescue's 2% stop, so the resumed matching fails in its last
// phase.)
TEST(ScaleOnDelay, KeptMatcherSingularFallsBack) {
    CscLower A = kept_matcher_kkt(kKeptSpread);
    const Int nv = 200;
    std::vector<char> touched(static_cast<std::size_t>(A.n), 0);
    for (Int j = 0; j < nv; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            if (A.row_idx[static_cast<std::size_t>(p)] != j)
                touched[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)])] = 1;
    for (Int j = nv; j < 6000; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            if (A.row_idx[static_cast<std::size_t>(p)] != j &&
                touched[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)])])
                A.values[static_cast<std::size_t>(p)] = 0.0;
    Control ctrl = kept_matcher_control();
    ctrl.continue_on_singular = true;
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_EQ(info.matching_trial, MatchingTrial::RejectedInitial);
    ASSERT_NE(ak->rescue_matcher, nullptr);

    FactorKeep fk;
    Info i1;
    const FactorStatus st = s.factor(*ak, ctrl, i1, fk);
    EXPECT_TRUE(st == FactorStatus::Singular || st == FactorStatus::Success) << static_cast<int>(st);
    EXPECT_EQ(i1.delay_stops, 1);
    EXPECT_EQ(i1.delay_matcher_reused, 1);
    EXPECT_EQ(i1.delay_rescales, 0);
    EXPECT_EQ(fk.delay_analysis, ak->serial);
    EXPECT_TRUE(fk.delay_scale.empty());
    EXPECT_NE(fk.delay_signature, 0u);
    EXPECT_EQ(ak->rescue_matcher, nullptr);

    Info i2;
    const FactorStatus st2 = s.factor(*ak, ctrl, i2, fk);
    EXPECT_EQ(st2, st);
    EXPECT_EQ(i2.delay_stops, 0);
    EXPECT_EQ(i2.delay_rescales, 0);
}

// Deterministic mode: the rescue that resumes the kept matcher gives the
// same factor and solution at every thread count (the kept matcher is the
// budget point's whatever the analysis' threads,
// MatchingTrial.ConcurrentTrialIsTheSerialOne).
TEST(ScaleOnDelay, ReusedMatcherDeterministicAcrossThreads) {
    const CscLower A = kept_matcher_kkt(kKeptSpread);
    std::vector<double> scale_ref, x_ref;
    Info i_ref;
    for (const int threads : {1, 4}) {
        Control ctrl = kept_matcher_control();
        ctrl.deterministic = true;
        ctrl.num_threads = threads;
        Solver s;
        Info ia;
        auto ak = s.analyse(A, ctrl, ia);
        ASSERT_NE(ak.get(), nullptr);
        ASSERT_EQ(ia.matching_trial, MatchingTrial::RejectedInitial) << "threads " << threads;
        FactorKeep fk;
        Info i1;
        ASSERT_EQ(s.factor(*ak, ctrl, i1, fk), FactorStatus::Success);
        EXPECT_EQ(i1.delay_matcher_reused, 1) << "threads " << threads;
        std::vector<double> x(static_cast<std::size_t>(A.n), 1.0);
        Info si;
        ASSERT_EQ(s.solve(fk, ctrl, si, x.data(), static_cast<int>(A.n), 1), 0);
        if (threads == 1) {
            scale_ref = fk.scale;
            x_ref = x;
            i_ref = i1;
            continue;
        }
        EXPECT_EQ(fk.scale, scale_ref);
        EXPECT_EQ(i1.delayed_pivots, i_ref.delayed_pivots);
        EXPECT_EQ(i1.num_positive, i_ref.num_positive);
        EXPECT_EQ(i1.num_negative, i_ref.num_negative);
        EXPECT_EQ(i1.num_zero, i_ref.num_zero);
        EXPECT_EQ(x, x_ref);
    }
}

// Control::scale_on_delay_first_ratio: the first armed factor of a
// RejectedInitial analysis into a FactorKeep tries under
// min(limit, first_ratio * delay-free work), or (0) not at all.
namespace {

// The kept-matcher fixture's analysis with the keep flag at its default
// (off: the rescue matches afresh, main's behaviour).
Control first_ratio_control(double first_ratio) {
    Control c = base_control();
    c.matching_search_budget = 0.5;
    c.scale_on_delay_first_ratio = first_ratio;
    return c;
}

struct FirstFactor {
    Info info;
    FactorKeep fk;
    bool matcher_released = false;
};

// A fresh analysis of A and its first factor into a fresh FactorKeep.
FirstFactor first_factor(const CscLower &A, const Control &ctrl) {
    FirstFactor r;
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    EXPECT_NE(ak.get(), nullptr);
    if (!ak) return r;
    EXPECT_EQ(ia.matching_trial, MatchingTrial::RejectedInitial);
    EXPECT_EQ(ak->matching_trial, MatchingTrial::RejectedInitial);
    EXPECT_EQ(s.factor(*ak, ctrl, r.info, r.fk), FactorStatus::Success);
    r.matcher_released = ak->rescue_matcher == nullptr;
    expect_accurate_solve(s, r.fk, ctrl, A);
    return r;
}

} // namespace

TEST(ScaleOnDelay, SkippedFirstTryGivesTheSameRescue) {
    const CscLower A = kept_matcher_kkt(kKeptSpread);
    const Control c_def = first_ratio_control(-1.0);
    ASSERT_FALSE(c_def.scale_on_delay_keep_matcher);
    ASSERT_GT(unscaled_ratio(A, c_def), c_def.scale_on_delay_ratio);  // premise: the try stops

    const FirstFactor def = first_factor(A, c_def);
    const FirstFactor skip = first_factor(A, first_ratio_control(0.0));
    const Info &i_def = def.info;
    const Info &i_skip = skip.info;
    EXPECT_EQ(i_def.delay_stops, 1);
    EXPECT_EQ(i_def.delay_rescales, 1);
    EXPECT_EQ(i_skip.delay_stops, 0);
    EXPECT_EQ(i_skip.delay_rescales, 1);
    EXPECT_EQ(i_skip.delay_try_seconds, 0.0);
    EXPECT_GT(i_skip.delay_matching_seconds, 0.0);
    EXPECT_EQ(i_skip.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_EQ(skip.fk.scale, def.fk.scale);  // the scale does not depend on the try
    EXPECT_EQ(skip.fk.delay_scale, def.fk.delay_scale);
    EXPECT_EQ(i_skip.delayed_pivots, i_def.delayed_pivots);
    EXPECT_EQ(i_skip.num_positive, i_def.num_positive);
    EXPECT_EQ(i_skip.num_negative, i_def.num_negative);

    // With the matcher kept, the skipped try still releases it.
    Control c_keep = first_ratio_control(0.0);
    c_keep.scale_on_delay_keep_matcher = true;
    const FirstFactor kept = first_factor(A, c_keep);
    EXPECT_EQ(kept.info.delay_stops, 0);
    EXPECT_EQ(kept.info.delay_rescales, 1);
    EXPECT_EQ(kept.info.delay_matcher_reused, 1);
    EXPECT_TRUE(kept.matcher_released);
}

TEST(ScaleOnDelay, FirstRatioCapsTheTry) {
    const CscLower A = kept_matcher_kkt(kKeptSpread);
    const Control c_def = first_ratio_control(-1.0);
    ASSERT_GT(unscaled_ratio(A, c_def), c_def.scale_on_delay_ratio);  // premise: the try stops

    const FirstFactor def = first_factor(A, c_def);
    const FirstFactor cap = first_factor(A, first_ratio_control(4.0));
    EXPECT_EQ(def.info.delay_stops, 1);
    EXPECT_EQ(cap.info.delay_stops, 1);
    EXPECT_EQ(cap.info.delay_rescales, 1);
    EXPECT_GT(cap.info.delay_try_seconds, 0.0);
    EXPECT_LE(cap.info.delay_try_seconds, def.info.delay_try_seconds);
    EXPECT_EQ(cap.fk.scale, def.fk.scale);
    EXPECT_EQ(cap.info.delayed_pivots, def.info.delayed_pivots);
    EXPECT_EQ(cap.info.num_negative, def.info.num_negative);
}

// A keep that is not an initial rejection is untouched: it still tries
// first, and later factors into the same FactorKeep run as without the
// option.
TEST(ScaleOnDelay, FirstRatioIgnoredWithoutInitialRejection) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const CscLower A1 = kkt(kG, kM, 4.15, 3.0);       // the same bad scaling
    const CscLower A2 = kkt(kG, kM, 4.1, 3.0, -0.1, 11);  // another bad scaling
    struct Run { int stops, rescales, delayed, negative; std::vector<double> scale; };
    std::vector<Run> runs[2];
    const double ratios[2] = {-1.0, 0.0};
    for (int v = 0; v < 2; ++v) {
        Control ctrl = base_control();
        ctrl.scale_on_delay_first_ratio = ratios[v];
        Solver s;
        Info ia;
        auto ak = s.analyse(A, ctrl, ia);
        ASSERT_NE(ak.get(), nullptr);
        ASSERT_EQ(ia.matching_trial, MatchingTrial::None) << "premise: no trial";
        ASSERT_EQ(ak->matching_trial, MatchingTrial::None);
        NumericWorkspace ws;
        FactorKeep fk;
        for (const CscLower *M : {&A, &A1, &A2, &A}) {
            Info i;
            ASSERT_EQ(s.factor(*ak, *M, ctrl, i, fk, ws), FactorStatus::Success);
            runs[v].push_back({i.delay_stops, i.delay_rescales, i.delayed_pivots,
                               i.num_negative, fk.scale});
            expect_accurate_solve(s, fk, ctrl, *M);
        }
    }
    EXPECT_EQ(runs[1][0].stops, 1);  // still tried
    EXPECT_EQ(runs[1][0].rescales, 1);
    for (std::size_t k = 0; k < runs[0].size(); ++k) {
        EXPECT_EQ(runs[1][k].stops, runs[0][k].stops) << "factor " << k;
        EXPECT_EQ(runs[1][k].rescales, runs[0][k].rescales) << "factor " << k;
        EXPECT_EQ(runs[1][k].delayed, runs[0][k].delayed) << "factor " << k;
        EXPECT_EQ(runs[1][k].negative, runs[0][k].negative) << "factor " << k;
        EXPECT_EQ(runs[1][k].scale, runs[0][k].scale) << "factor " << k;
    }
}
