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
#include "smf/refinement.hpp"
#include "smf/scaling.hpp"
#include "smf/scaling_matching.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
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
    // These tests are about the rescue itself; the static fallback that
    // replaces it after a cascade has its own tests (StaticFallback.*).
    c.static_fallback_after_cascade = false;
    return c;  // Auto scaling, pivot_u = 0.01, default trigger ratio
}

// The rescue's matching scaling of M's values, M being a clean matrix of
// ak's pattern.
std::vector<double> matching_scale_of(const AnalysisKeep &ak, const CscLower &M) {
    // The analysed values reuse the analysis' own matching scale, adopted or
    // rejected (a degree-order search's, see MatchingSearchOrder), as
    // factor() does; other values get a matching whose searches end with at
    // most 2% of the columns free (solve.cpp's kRescueMaxFree).
    const std::vector<double> &analysed =
        !ak.matching_scale.empty() && !ak.matching_scale_partial ? ak.matching_scale : ak.rejected_matching_scale;
    if (!analysed.empty() && M.values == ak.cleaned.values) return analysed;
    CscLower clean = ak.cleaned;
    clean.values = M.values;
    SymmetricMatcher m(clean);
    m.set_stop_free(static_cast<int>(0.02 * static_cast<double>(clean.n)));
    m.advance();
    const MatchingResult mr = m.current();
    EXPECT_EQ(mr.status, ErrorCode::Success);
    return mr.scale;
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
    // The full ratio alone: at the 4x crossing this one projects to 13.8x
    // (scale_on_delay_early_proj 8) and the early rule stops it, though it
    // ends at 9.96x (top-heavy tree: the root front carries most of the
    // delay-free work).
    ctrl.scale_on_delay_early_ratio = -1.0;
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

    // The defaults (early stop on): the try stops at the early crossing and
    // the matching scaling follows. Pinned: one early stop, one rescale,
    // and the work spent (the stopped try plus the rescued factor) against
    // the factor let through.
    const Control dflt = base_control();
    ASSERT_GE(dflt.scale_on_delay_early_ratio, 0.0);
    FactorKeep fk_d;
    Info i_d;
    ASSERT_EQ(s.factor(*ak, dflt, i_d, fk_d), FactorStatus::Success);
    EXPECT_EQ(i_d.delay_stops, 1);
    EXPECT_EQ(i_d.delay_early_stops, 1);
    EXPECT_EQ(i_d.delay_rescales, 1);
    const double F = delay_free_front_work(*ak);
    NumericWorkspace ws;
    FactorKeep f_try;
    Info i_try;
    ASSERT_EQ(factor_indef(*ak, A, dflt, i_try, f_try, ws,
                           std::floor(dflt.scale_on_delay_ratio * F),
                           EarlyWorkStop{F, dflt.scale_on_delay_early_ratio,
                                         dflt.scale_on_delay_early_proj}),
              FactorStatus::MaxPivotDelays);
    std::printf("[ScaleOnDelay] zero (2,2), defaults: try %.2f F + rescued %.2f F = %.2f F "
                "(%d delays) against %.2f F let through; factor %.2f ms vs %.2f ms\n",
                i_try.front_work / F, i_d.front_work / F, (i_try.front_work + i_d.front_work) / F,
                i_d.delayed_pivots, i_off.front_work / F, 1e3 * i_d.factor_seconds,
                1e3 * i_off.factor_seconds);
    EXPECT_EQ(i_d.num_positive, kG * kG);
    EXPECT_EQ(i_d.num_negative, kM);
    // Measured: try 4.25 F + rescued 7.73 F = 11.98 F against 9.96 F let
    // through. The rescued factor is cheaper than the one let through (so
    // are the later factorizations that reuse its scale), but the stopped
    // try makes this factor() cost more work than letting it through. With
    // the early stop off the 9.96x factor arms the next limit, and the next
    // factor() is rescaled without a try (Control::scale_on_delay_carry);
    // with it on the next one reuses the kept scale.
    EXPECT_LT(i_d.front_work, i_off.front_work);
    EXPECT_GT(i_try.front_work + i_d.front_work, i_off.front_work);
    Info i_d2, i_2;
    ASSERT_EQ(s.factor(*ak, dflt, i_d2, fk_d), FactorStatus::Success);
    EXPECT_EQ(i_d2.delay_stops, 0);
    EXPECT_EQ(i_d2.delay_rescales, 0);
    ASSERT_EQ(s.factor(*ak, ctrl, i_2, fk), FactorStatus::Success);  // early stop off
    ASSERT_TRUE(ctrl.scale_on_delay_carry);
    EXPECT_EQ(i_2.delay_stops, 0);
    EXPECT_EQ(i_2.delay_try_seconds, 0.0);
    EXPECT_EQ(i_2.delay_early_stops, 0);
    EXPECT_EQ(i_2.delay_rescales, 1);
    EXPECT_EQ(i_2.front_work, i_d2.front_work);
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

// Solve M x = b and refine it with smf::refine, as the IPOPT path solves
// these captures; the refined x must meet the accuracy gate. The unrefined
// backward error is only printed: at PGLib case500 step 26 it sits at the
// gate, omega 6.5e-9 on a Ryzen 9950X3D and 1.12e-8 on an i9-12900K with the
// same binary and the same rescale decisions, so the test cannot assert on it.
void expect_accurate_refined_solve(const Solver &s, const FactorKeep &fk, const Control &ctrl,
                                   const CscLower &M, std::size_t step) {
    const int n = static_cast<int>(M.n);
    std::vector<double> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = 1.0 + 0.01 * (i % 97);
    std::vector<double> x = b;
    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, x.data(), n, 1), 0) << "step " << step;
    const auto raw = smf::bench::validate(M, b, x, 1);
    SolveWorkspace sws;
    const RefinementResult rr = refine(M, fk, ctrl, b, x, 1, sws);
    const auto val = smf::bench::validate(M, b, x, 1);
    std::printf("[ScaleOnDelay] case500 step %zu: omega %.2e unrefined, %.2e after %d refinement steps\n",
                step, raw.omega, val.omega, rr.iterations);
    EXPECT_EQ(rr.status, ErrorCode::Success) << "step " << step;
    EXPECT_TRUE(smf::bench::is_qualified_validation(val))
        << "step " << step << ": eta_inf=" << val.eta_inf << " omega=" << val.omega;
}

// A real reuse that the probe rejects: PGLib case500's IPOPT capture,
// factored at the replay's 5 evenly spaced steps into one FactorKeep. On
// the tuning split the kept scale of step 13 fits step 52 but its factor
// probes 75x the fresh error there, so step 52 is refactored with a new
// matching scaling; every step solves to the accuracy gate after refinement.
// Skipped without the (untracked) fixture.
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
        expect_accurate_refined_solve(s, fk, ctrl, M, step);
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
    ASSERT_GT(ctrl.scale_on_delay_next_ratio, ctrl.scale_on_delay_early_ratio);

    struct Case { const CscLower *next_values; bool on; bool rescales; };
    for (const Case cs : {Case{&A1, true, true}, Case{&A1, false, false}, Case{&E, true, false}}) {
        Control c = ctrl;
        if (!cs.on) c.scale_on_delay_next_ratio = -1.0;
        // The first factorization with the early stop off (A is a cascade:
        // the early stop would end its first try). The second under the
        // tighter next limit keeps the default early stop on: next tries are
        // never early-stopped, even with a next ratio above the early one.
        // Without the next limit (cs.on false) the second is a first try
        // under the full limit, which the early stop would end: off there.
        Control c1 = c;
        c1.scale_on_delay_early_ratio = -1.0;
        if (!cs.on) c.scale_on_delay_early_ratio = -1.0;
        // The next try itself (scale_on_delay_carry skips it).
        c.scale_on_delay_carry = false;
        Solver s;
        Info info;
        auto ak = s.analyse(A, c, info);
        ASSERT_NE(ak.get(), nullptr);
        ASSERT_TRUE(ak->matching_scale.empty());
        NumericWorkspace ws;
        FactorKeep fk;
        Info i1;
        ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), c1, i1, fk, ws),
                  FactorStatus::Success);
        EXPECT_EQ(i1.delay_rescales, 0);
        EXPECT_EQ(i1.delay_stops, 0);
        EXPECT_EQ(i1.scaling_used, static_cast<int>(ScalingMethod::None));
        EXPECT_EQ(fk.delay_next, cs.on);
        expect_accurate_solve(s, fk, c1, A);

        const CscLower &B = *cs.next_values;
        Info i2;
        ASSERT_EQ(s.factor(*ak, B, c, i2, fk, ws), FactorStatus::Success);
        EXPECT_FALSE(fk.delay_next);
        EXPECT_EQ(i2.delay_stops, cs.rescales ? 1 : 0);
        EXPECT_EQ(i2.delay_rescales, cs.rescales ? 1 : 0);
        EXPECT_EQ(i2.delay_early_stops, 0);
        EXPECT_EQ(i2.scaling_used, static_cast<int>(cs.rescales ? ScalingMethod::Matching
                                                                : ScalingMethod::None));
        if (cs.rescales) EXPECT_EQ(fk.delay_scale, matching_scale_of(*ak, B));
        EXPECT_EQ(i2.num_positive, kG * kG);
        EXPECT_EQ(i2.num_negative, kM);
        expect_accurate_solve(s, fk, c, B);
    }
}

// Control::scale_on_delay_carry: after a costly let-through the next
// factorization skips its unscaled try and starts from the matching scaling
// of its own values (no stopped try, one rescale); later ones go on from
// that kept scale as after any rescue. A sequence that never let a costly
// factor through is untouched.
TEST(ScaleOnDelay, CarryRescalesAfterACostlyLetThrough) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const CscLower A1 = kkt(kG, kM, 4.15, 3.0);  // the same bad scaling
    const CscLower E = kkt(kG, kM, 4.1, 0.0);    // well scaled
    Control ctrl = base_control();
    const double r = unscaled_ratio(A, ctrl);
    ASSERT_GT(r, 4.0) << "premise: the delays inflate the work";
    ctrl.scale_on_delay_ratio = 2.0 * r;  // lets A through
    ctrl.scale_on_delay_next_ratio = 0.5 * r;
    ctrl.scale_on_delay_early_ratio = -1.0;
    ctrl.scale_on_delay_carry = true;
    Solver s;
    {
        Info info;
        auto ak = s.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace ws;
        FactorKeep fk;
        Info i1;
        ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), ctrl, i1, fk, ws),
                  FactorStatus::Success);
        EXPECT_EQ(i1.delay_rescales, 0);
        EXPECT_TRUE(fk.delay_next);

        Info i2;
        ASSERT_EQ(s.factor(*ak, A1, ctrl, i2, fk, ws), FactorStatus::Success);
        EXPECT_FALSE(fk.delay_next);
        EXPECT_EQ(i2.delay_stops, 0);
        EXPECT_EQ(i2.delay_early_stops, 0);
        EXPECT_EQ(i2.delay_rescales, 1);
        EXPECT_EQ(i2.delay_try_seconds, 0.0);
        EXPECT_EQ(i2.scaling_used, static_cast<int>(ScalingMethod::Matching));
        EXPECT_EQ(fk.delay_scale, matching_scale_of(*ak, A1));
        EXPECT_EQ(fk.delay_analysis, ak->serial);
        EXPECT_EQ(i2.num_positive, kG * kG);
        EXPECT_EQ(i2.num_negative, kM);
        expect_accurate_solve(s, fk, ctrl, A1);

        // The next one goes on from the kept scale, never an unscaled try.
        Info i3;
        ASSERT_EQ(s.factor(*ak, A1, ctrl, i3, fk, ws), FactorStatus::Success);
        EXPECT_EQ(i3.delay_stops, 0);
        EXPECT_EQ(i3.scaling_used, static_cast<int>(ScalingMethod::Matching));
        EXPECT_EQ(fk.delay_analysis, ak->serial);
        expect_accurate_solve(s, fk, ctrl, A1);
    }
    {
        // Without the next limit there is no carry: a plain first try.
        Control c = ctrl;
        Info info;
        auto ak = s.analyse(A, c, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace ws;
        FactorKeep fk;
        Info i1;
        ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), c, i1, fk, ws),
                  FactorStatus::Success);
        ASSERT_TRUE(fk.delay_next);
        c.scale_on_delay_next_ratio = -1.0;
        Info i2;
        ASSERT_EQ(s.factor(*ak, A1, c, i2, fk, ws), FactorStatus::Success);
        EXPECT_EQ(i2.delay_rescales, 0);
        EXPECT_EQ(i2.scaling_used, static_cast<int>(ScalingMethod::None));
    }
    {
        // Cheap factorizations never arm the carry.
        Info info;
        auto ak = s.analyse(E, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace ws;
        FactorKeep fk;
        for (int k = 0; k < 2; ++k) {
            Info ik;
            ASSERT_EQ(s.factor(*ak, std::span<const double>(E.values), ctrl, ik, fk, ws),
                      FactorStatus::Success);
            EXPECT_FALSE(fk.delay_next);
            EXPECT_EQ(ik.delay_rescales, 0);
            EXPECT_EQ(ik.delay_stops, 0);
            EXPECT_EQ(ik.scaling_used, static_cast<int>(ScalingMethod::None));
        }
    }
}

// Control::scale_on_delay_next_early_ratio: the next try after a costly
// let-through may stop at its own early crossing (an early stop, then the
// matching scaling), well before the next limit.
TEST(ScaleOnDelay, NextTryEarlyStop) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const CscLower A1 = kkt(kG, kM, 4.15, 3.0);
    Control ctrl = base_control();
    const double r = unscaled_ratio(A, ctrl);
    ASSERT_GT(r, 4.0) << "premise: the delays inflate the work";
    ctrl.scale_on_delay_ratio = 2.0 * r;
    ctrl.scale_on_delay_next_ratio = 0.9 * r;
    ctrl.scale_on_delay_early_ratio = -1.0;
    ctrl.scale_on_delay_carry = false;
    ctrl.scale_on_delay_next_early_ratio = 1.5;
    ctrl.scale_on_delay_next_early_proj = 1.2;
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), ctrl, i1, fk, ws),
              FactorStatus::Success);
    ASSERT_TRUE(fk.delay_next);
    Info i2;
    ASSERT_EQ(s.factor(*ak, A1, ctrl, i2, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i2.delay_stops, 1);
    EXPECT_EQ(i2.delay_early_stops, 1);
    EXPECT_EQ(i2.delay_rescales, 1);
    EXPECT_EQ(i2.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_EQ(i2.num_positive, kG * kG);
    EXPECT_EQ(i2.num_negative, kM);
    expect_accurate_solve(s, fk, ctrl, A1);
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

namespace {

// The rescue's conditional early stop of a first try
// (Control::scale_on_delay_early_ratio) as Solver::factor arms it.
EarlyWorkStop early_of(const AnalysisKeep &ak, const Control &c) {
    return {delay_free_front_work(ak), c.scale_on_delay_early_ratio, c.scale_on_delay_early_proj};
}

// Solver::factor outcomes that must not depend on where a try stopped.
void expect_same_outcome(const Info &a, const FactorKeep &fa, const Info &b,
                         const FactorKeep &fb) {
    EXPECT_EQ(a.delay_stops, b.delay_stops);
    EXPECT_EQ(a.delay_rescales, b.delay_rescales);
    EXPECT_EQ(a.scaling_used, b.scaling_used);
    EXPECT_EQ(a.front_work, b.front_work);
    EXPECT_EQ(a.delayed_pivots, b.delayed_pivots);
    EXPECT_EQ(a.num_positive, b.num_positive);
    EXPECT_EQ(a.num_negative, b.num_negative);
    EXPECT_EQ(a.num_zero, b.num_zero);
    EXPECT_EQ(fa.scale, fb.scale);
    EXPECT_EQ(fa.factor_values, fb.factor_values);
    EXPECT_EQ(fa.delay_scale, fb.delay_scale);
    EXPECT_EQ(fa.delay_next, fb.delay_next);
}

} // namespace

// A first try headed past the full limit (a cascade) stops at the early
// crossing when its inflation so far projects past the threshold: less
// front work spent, and the same rescue after it (the rescued factor does
// not depend on where the try stopped).
TEST(ScaleOnDelay, EarlyStopEndsATryHeadedPastTheLimit) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const Control ctrl = base_control();
    ASSERT_GE(ctrl.scale_on_delay_early_ratio, 0.0);
    ASSERT_GT(ctrl.scale_on_delay_ratio, ctrl.scale_on_delay_early_ratio);
    ASSERT_GT(unscaled_ratio(A, ctrl), ctrl.scale_on_delay_ratio) << "premise: past the limit";

    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_TRUE(ak->matching_scale.empty());
    const double F = delay_free_front_work(*ak);
    const double cap = std::floor(ctrl.scale_on_delay_ratio * F);
    const double at = std::floor(ctrl.scale_on_delay_early_ratio * F);

    // The engine: the full limit alone stops past it; the rule stops at the
    // early crossing, with less work.
    NumericWorkspace ws;
    FactorKeep f_cap, f_early;
    Info i_cap, i_early;
    ASSERT_EQ(factor_indef(*ak, A, ctrl, i_cap, f_cap, ws, cap), FactorStatus::MaxPivotDelays);
    ASSERT_EQ(factor_indef(*ak, A, ctrl, i_early, f_early, ws, cap, early_of(*ak, ctrl)),
              FactorStatus::MaxPivotDelays);
    std::printf("[ScaleOnDelay] early stop: try work %.2f F (limit alone %.2f F)\n",
                i_early.front_work / F, i_cap.front_work / F);
    EXPECT_GT(i_cap.front_work, cap);
    EXPECT_GT(i_early.front_work, at);
    EXPECT_LE(i_early.front_work, cap);
    EXPECT_LT(i_early.front_work, i_cap.front_work);
    EXPECT_TRUE(f_early.factor_values.empty());

    // Solver::factor, rule on and off: one stopped try and one rescale each,
    // the same rescued factor.
    Control off = ctrl;
    off.scale_on_delay_early_ratio = -1.0;
    FactorKeep fk_on, fk_off;
    Info on_i, off_i;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), ctrl, on_i, fk_on, ws),
              FactorStatus::Success);
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), off, off_i, fk_off, ws),
              FactorStatus::Success);
    EXPECT_EQ(on_i.delay_stops, 1);
    EXPECT_EQ(on_i.delay_early_stops, 1);
    EXPECT_EQ(on_i.delay_rescales, 1);
    EXPECT_EQ(off_i.delay_early_stops, 0);  // a limit trip
    expect_same_outcome(on_i, fk_on, off_i, fk_off);
    expect_accurate_solve(s, fk_on, ctrl, A);
}

// A try whose delays inflate it past the early ratio but not past the full
// limit, at a modest rate, is let through: the same factor as with the rule
// off, no stop, no rescale.
TEST(ScaleOnDelay, EarlyStopLetsAModestTryThrough) {
    // Mildly scaled, a weak (2,2) block: unscaled work 4.84x the delay-free
    // one, about 6x projected at the 4x crossing.
    const CscLower A = kkt(kG, kM, 4.1, 0.5, -0.01);
    const Control ctrl = base_control();
    const double r = unscaled_ratio(A, ctrl);
    ASSERT_GT(r, ctrl.scale_on_delay_early_ratio) << "premise: crosses the early ratio";
    ASSERT_LT(r, ctrl.scale_on_delay_ratio) << "premise: within the limit";

    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_TRUE(ak->matching_scale.empty());
    Control off = ctrl;
    off.scale_on_delay_early_ratio = -1.0;
    NumericWorkspace ws;
    FactorKeep fk_on, fk_off;
    Info on_i, off_i;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), ctrl, on_i, fk_on, ws),
              FactorStatus::Success);
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), off, off_i, fk_off, ws),
              FactorStatus::Success);
    EXPECT_EQ(on_i.delay_stops, 0);
    EXPECT_EQ(on_i.delay_rescales, 0);
    EXPECT_EQ(on_i.scaling_used, static_cast<int>(ScalingMethod::None));
    EXPECT_GT(work_ratio(*ak, on_i), ctrl.scale_on_delay_early_ratio);
    expect_same_outcome(on_i, fk_on, off_i, fk_off);
}

// A negative early ratio is the plain work limit, bit for bit (engine and
// Solver::factor), as is a projection threshold nothing reaches.
TEST(ScaleOnDelay, EarlyStopOffIsThePlainLimit) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const Control ctrl = base_control();
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    const double F = delay_free_front_work(*ak);
    const double cap = std::floor(ctrl.scale_on_delay_ratio * F);
    NumericWorkspace ws;
    FactorKeep f_cap, f_neg, f_inf;
    Info i_cap, i_neg, i_inf;
    ASSERT_EQ(factor_indef(*ak, A, ctrl, i_cap, f_cap, ws, cap), FactorStatus::MaxPivotDelays);
    ASSERT_EQ(factor_indef(*ak, A, ctrl, i_neg, f_neg, ws, cap, EarlyWorkStop{F, -1.0, 0.0}),
              FactorStatus::MaxPivotDelays);
    ASSERT_EQ(factor_indef(*ak, A, ctrl, i_inf, f_inf, ws, cap,
                           EarlyWorkStop{F, ctrl.scale_on_delay_early_ratio,
                                         std::numeric_limits<double>::infinity()}),
              FactorStatus::MaxPivotDelays);
    EXPECT_EQ(i_neg.front_work, i_cap.front_work);
    EXPECT_EQ(i_neg.delayed_pivots, i_cap.delayed_pivots);
    EXPECT_EQ(i_inf.front_work, i_cap.front_work);
    EXPECT_EQ(i_inf.delayed_pivots, i_cap.delayed_pivots);

    Control off = ctrl;
    off.scale_on_delay_early_ratio = -1.0;
    Control never = ctrl;
    never.scale_on_delay_early_proj = std::numeric_limits<double>::infinity();
    FactorKeep fk_off, fk_never;
    Info off_i, never_i;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), off, off_i, fk_off, ws),
              FactorStatus::Success);
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), never, never_i, fk_never, ws),
              FactorStatus::Success);
    expect_same_outcome(off_i, fk_off, never_i, fk_never);
}

// Deterministic mode: the rule's decision is the same at every thread count
// (the subtree engine, at 2 and 4, evaluates it over the fronts in
// postorder): a stop under a limit nothing reaches when the projection
// passes the threshold, the run to the end when nothing can pass it; and
// Solver::factor's solution is the same.
TEST(ScaleOnDelay, EarlyStopDeterministicAcrossThreadCounts) {
    const CscLower A = kkt(90, 300, 4.1, 3.0);  // fires (DeterministicAcrossThreadCounts)
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
        const double F = delay_free_front_work(*ak);
        const double big = 1e6 * F;  // a limit nothing reaches
        NumericWorkspace ws;
        FactorKeep f_ref, f_stop, f_go;
        Info i_ref, i_stop, i_go;
        ASSERT_EQ(factor_indef(*ak, A, ctrl, i_ref, f_ref, ws), FactorStatus::Success);
        if (threads > 1) EXPECT_GT(i_ref.parallel_subtrees, 1) << "threads " << threads;
        ASSERT_GT(i_ref.front_work, ctrl.scale_on_delay_ratio * F) << "premise: a cascade";
        EXPECT_EQ(factor_indef(*ak, A, ctrl, i_stop, f_stop, ws, big, early_of(*ak, ctrl)),
                  FactorStatus::MaxPivotDelays)
            << "threads " << threads;
        EXPECT_LT(i_stop.front_work, i_ref.front_work) << "threads " << threads;
        EarlyWorkStop never = early_of(*ak, ctrl);
        never.proj = std::numeric_limits<double>::infinity();
        EXPECT_EQ(factor_indef(*ak, A, ctrl, i_go, f_go, ws, big, never), FactorStatus::Success)
            << "threads " << threads;
        EXPECT_EQ(i_go.front_work, i_ref.front_work) << "threads " << threads;
        EXPECT_EQ(i_go.delayed_pivots, i_ref.delayed_pivots) << "threads " << threads;

        FactorKeep fk;
        Info i1;
        ASSERT_EQ(s.factor(*ak, ctrl, i1, fk), FactorStatus::Success);
        EXPECT_EQ(i1.delay_stops, 1) << "threads " << threads;
        EXPECT_EQ(i1.delay_rescales, 1) << "threads " << threads;
        std::vector<double> x(static_cast<std::size_t>(A.n), 1.0);
        Info si;
        ASSERT_EQ(s.solve(fk, ctrl, si, x.data(), static_cast<int>(A.n), 1), 0);
        if (x_ref.empty()) x_ref = x;
        else EXPECT_EQ(x, x_ref) << "threads " << threads;
    }
}

// --- Control::static_fallback_after_cascade, Pivoting::Static and the
// backward-error refinement trigger (Control::refine_omega_target,
// refine_growth_threshold) ---

namespace {

Control fallback_control() {
    Control c = base_control();
    c.static_fallback_after_cascade = true;
    return c;
}

// x = A^-1 b with fk, refined when refinement_advised(fk); returns the
// refinement result (iterations -1: not refined).
RefinementResult solve_refined(const Solver &s, const FactorKeep &fk, const Control &ctrl,
                               const CscLower &A, std::vector<double> &x) {
    const int n = static_cast<int>(A.n);
    std::vector<double> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = 1.0 + 0.01 * (i % 97);
    x = b;
    Info si;
    EXPECT_EQ(s.solve(fk, ctrl, si, x.data(), n, 1), 0);
    RefinementResult rr;
    rr.iterations = -1;
    if (refinement_advised(fk)) {
        SolveWorkspace ws;
        rr = refine(A, fk, ctrl, b, x, 1, ws);
    }
    const auto val = smf::bench::validate(A, b, x, 1);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val))
        << "eta_inf=" << val.eta_inf << " omega=" << val.omega;
    return rr;
}

} // namespace

// A moderately scaled KKT whose unscaled delays still cascade (work ratio
// ~21 against the limit's 16) but whose unscaled static factorization
// perturbs no pivot: the static fallback holds there.
CscLower static_ok_kkt(double hdiag = 2.5) { return kkt(90, 300, hdiag, 1.5); }

// A sequence whose first factorization trips the work limit: that factor is
// redone on the static path (no matching), and every later one of the
// analysis into that FactorKeep runs on it directly, with no try. The same
// decisions, factors and solutions at 1, 2 and 4 threads in deterministic
// mode.
TEST(StaticFallback, CascadeSwitchesTheSequenceDeterministically) {
    const CscLower A = static_ok_kkt();  // n < 10000: no auto matching
    // Later steps with the same values: a nearby H diagonal (2.55) already
    // perturbs pivots unscaled, which PerturbedStaticFactorFallsBackToTheRescue covers.
    const CscLower A1 = A;
    std::pair<int, int> ref_inertia;
    {
        const Control t = base_control();
        Solver s;
        Info ia, i;
        auto ak = s.analyse(A, t, ia);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(s.factor(*ak, t, i, fk), FactorStatus::Success);
        ASSERT_EQ(i.num_zero, 0);
        ref_inertia = {i.num_positive, i.num_negative};
    }
    std::vector<std::vector<double>> x_ref;
    std::vector<int> static_ref;
    for (const int threads : {1, 2, 4}) {
        Control ctrl = fallback_control();
        ctrl.deterministic = true;
        ctrl.num_threads = threads;
        ctrl.auto_threads = false;  // small test trees
        Solver s;
        Info ia;
        auto ak = s.analyse(A, ctrl, ia);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace ws;
        FactorKeep fk;
        std::vector<std::vector<double>> xs;
        std::vector<int> statics;
        int step = 0;
        for (const CscLower *M : {&A, &A1, &A1}) {
            Info i;
            ASSERT_EQ(s.factor(*ak, *M, ctrl, i, fk, ws), FactorStatus::Success);
            std::printf("[StaticFallback] threads %d step %d: stops %d rescales %d static %d "
                        "perturbed %d delays %d growth %.3g\n",
                        threads, step, i.delay_stops, i.delay_rescales, i.static_fallback,
                        i.static_pivots, i.delayed_pivots, i.pivot_growth);
            // Step 0 trips (a stopped try), later steps have no try at all.
            EXPECT_EQ(i.delay_stops, step == 0 ? 1 : 0) << "threads " << threads;
            EXPECT_EQ(i.delay_rescales, 0);
            EXPECT_EQ(i.static_fallback, 1);
            if (step > 0) EXPECT_EQ(i.delay_try_seconds, 0.0);
            EXPECT_EQ(i.delayed_pivots, 0);
            EXPECT_EQ(i.scaling_used, static_cast<int>(ScalingMethod::None));
            EXPECT_TRUE(fk.static_path);
            EXPECT_TRUE(fk.delay_static);
            EXPECT_EQ(fk.delay_analysis, ak->serial);
            EXPECT_TRUE(fk.delay_scale.empty());
            EXPECT_EQ(i.static_rejects, 0);
            EXPECT_EQ(i.static_pivots, 0);
            EXPECT_TRUE(i.inertia_exact);
            // H (diagonal 2.5) is indefinite: the threshold factor's inertia.
            EXPECT_EQ(i.num_positive, ref_inertia.first);
            EXPECT_EQ(i.num_negative, ref_inertia.second);
            EXPECT_EQ(i.num_zero, 0);
            std::vector<double> x;
            const RefinementResult rr = solve_refined(s, fk, ctrl, *M, x);
            EXPECT_GE(rr.iterations, 0) << "a static-path factor is refined";
            xs.push_back(x);
            statics.push_back(i.static_pivots);
            ++step;
        }
        if (x_ref.empty()) {
            x_ref = xs;
            static_ref = statics;
        } else {
            EXPECT_EQ(xs, x_ref) << "threads " << threads;
            EXPECT_EQ(statics, static_ref) << "threads " << threads;
        }
    }
}

// A factorization let through past scale_on_delay_next_ratio stays as it is
// and switches the later ones to the static path (instead of the next try or
// the carried rescale); cheap sequences never switch.
TEST(StaticFallback, CostlyLetThroughSwitchesTheNextFactor) {
    const CscLower A = static_ok_kkt();
    // Later steps with the same values: a nearby H diagonal (2.55) already
    // perturbs pivots unscaled, which PerturbedStaticFactorFallsBackToTheRescue covers.
    const CscLower A1 = A;
    const CscLower E = kkt(kG, kM, 4.1, 0.0);
    Control ctrl = fallback_control();
    const double r = unscaled_ratio(A, ctrl);
    ASSERT_GT(r, 4.0) << "premise: the delays inflate the work";
    ctrl.scale_on_delay_ratio = 2.0 * r;  // lets A through
    ctrl.scale_on_delay_next_ratio = 0.5 * r;
    ctrl.scale_on_delay_early_ratio = -1.0;
    Solver s;
    {
        Info info;
        auto ak = s.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace ws;
        FactorKeep fk;
        Info i1;
        ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), ctrl, i1, fk, ws),
                  FactorStatus::Success);
        EXPECT_EQ(i1.static_fallback, 0);
        EXPECT_EQ(i1.delay_stops, 0);
        EXPECT_GT(i1.delayed_pivots, 0);
        EXPECT_FALSE(fk.static_path);
        EXPECT_FALSE(fk.delay_next);
        EXPECT_TRUE(fk.delay_static);
        Info i2;
        ASSERT_EQ(s.factor(*ak, A1, ctrl, i2, fk, ws), FactorStatus::Success);
        EXPECT_EQ(i2.static_fallback, 1);
        EXPECT_EQ(i2.delay_stops, 0);
        EXPECT_EQ(i2.delay_rescales, 0);
        EXPECT_EQ(i2.delayed_pivots, 0);
        EXPECT_TRUE(fk.static_path);
        std::vector<double> x;
        solve_refined(s, fk, ctrl, A1, x);
        // Later values whose unscaled static factor perturbs pivots: that
        // static factor is thrown away and the rescue follows; no static
        // tries after it.
        const CscLower A2 = static_ok_kkt(2.55);
        Info i4;
        ASSERT_EQ(s.factor(*ak, A2, ctrl, i4, fk, ws), FactorStatus::Success);
        EXPECT_EQ(i4.static_rejects, 1);
        EXPECT_EQ(i4.static_fallback, 0);
        EXPECT_EQ(i4.delay_rescales, 1);
        EXPECT_FALSE(fk.static_path);
        EXPECT_FALSE(fk.delay_static);
        EXPECT_EQ(fk.static_rejected_analysis, ak->serial);
        expect_accurate_solve(s, fk, ctrl, A2);
        Info i5;
        ASSERT_EQ(s.factor(*ak, A2, ctrl, i5, fk, ws), FactorStatus::Success);
        EXPECT_EQ(i5.static_rejects, 0);
        EXPECT_EQ(i5.static_fallback, 0);
        // Another FactorKeep starts over (a plain first try).
        FactorKeep fk2;
        Info i3;
        ASSERT_EQ(s.factor(*ak, A1, ctrl, i3, fk2, ws), FactorStatus::Success);
        EXPECT_EQ(i3.static_fallback, 0);
    }
    {
        Info info;
        auto ak = s.analyse(E, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace ws;
        FactorKeep fk;
        for (int k = 0; k < 3; ++k) {
            Info ik;
            ASSERT_EQ(s.factor(*ak, std::span<const double>(E.values), ctrl, ik, fk, ws),
                      FactorStatus::Success);
            EXPECT_EQ(ik.static_fallback, 0);
            EXPECT_FALSE(fk.delay_static);
            EXPECT_FALSE(fk.static_path);
        }
    }
}

// A badly scaled KKT: unscaled static pivoting perturbs many pivots, so the
// static factor of the tripped factorization is thrown away (a spent try,
// Info::static_rejects) and the rescue's matching scaling follows; the
// sequence then goes on from the kept scale and never tries static again.
TEST(StaticFallback, PerturbedStaticFactorFallsBackToTheRescue) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const CscLower A1 = kkt(kG, kM, 4.15, 3.0);
    const Control ctrl = fallback_control();
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), ctrl, i1, fk, ws),
              FactorStatus::Success);
    EXPECT_EQ(i1.static_rejects, 1);
    EXPECT_EQ(i1.static_fallback, 0);
    EXPECT_EQ(i1.delay_stops, 1);
    EXPECT_EQ(i1.delay_rescales, 1);
    EXPECT_EQ(i1.static_pivots, 0);
    EXPECT_EQ(i1.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_FALSE(fk.static_path);
    EXPECT_FALSE(fk.delay_static);
    EXPECT_EQ(fk.scale, matching_scale_of(*ak, A));
    EXPECT_EQ(i1.num_positive, kG * kG);
    EXPECT_EQ(i1.num_negative, kM);
    expect_accurate_solve(s, fk, ctrl, A);
    for (int k = 0; k < 2; ++k) {
        Info i2;
        ASSERT_EQ(s.factor(*ak, A1, ctrl, i2, fk, ws), FactorStatus::Success);
        EXPECT_EQ(i2.static_rejects, 0);
        EXPECT_EQ(i2.static_fallback, 0);
        EXPECT_EQ(i2.scaling_used, static_cast<int>(ScalingMethod::Matching));
        EXPECT_FALSE(fk.static_path);
        expect_accurate_solve(s, fk, ctrl, A1);
    }
}

// Off, a cascading sequence takes the rescue exactly as before the fallback
// existed (the ScaleOnDelay tests pin its details): a matching scaling of
// the values, no static path; refine() on such a factor ignores
// refine_omega_target (bitwise the same refined x).
TEST(StaticFallback, OffIsTheRescue) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const CscLower A1 = kkt(kG, kM, 4.15, 3.0);
    Control ctrl = base_control();
    ASSERT_FALSE(ctrl.static_fallback_after_cascade);
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, std::span<const double>(A.values), ctrl, i1, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i1.delay_rescales, 1);
    EXPECT_EQ(i1.static_fallback, 0);
    EXPECT_FALSE(fk.static_path);
    EXPECT_FALSE(fk.delay_static);
    EXPECT_EQ(fk.scale, matching_scale_of(*ak, A));
    Info i2;
    ASSERT_EQ(s.factor(*ak, A1, ctrl, i2, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i2.static_fallback, 0);
    EXPECT_EQ(i2.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_FALSE(refinement_advised(fk));
    const int n = static_cast<int>(A1.n);
    std::vector<double> b(static_cast<std::size_t>(n), 1.0);
    std::vector<double> x0 = b;
    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, x0.data(), n, 1), 0);
    std::vector<double> xa = x0, xb = x0;
    SolveWorkspace wa, wb;
    Control tight = ctrl;
    tight.refine_omega_target = 1e-300;
    tight.refine_growth_threshold = 0.0;
    const RefinementResult ra = refine(A1, fk, ctrl, b, xa, 1, wa);
    const RefinementResult rb = refine(A1, fk, tight, b, xb, 1, wb);
    EXPECT_EQ(xa, xb);
    EXPECT_EQ(ra.iterations, rb.iterations);
}

// The knob is off by default, and a default Control then factors a cascading
// sequence bitwise like one with the knob set off explicitly (the pre-fallback
// path: the rescue, no static path).
TEST(StaticFallback, DefaultIsOffAndBitwiseTheRescue) {
    Control def;
    def.matrix_type = MatrixType::RealSymmetricIndefinite;
    def.amalgamation = Amalgamation::Smf;
    ASSERT_FALSE(def.static_fallback_after_cascade);
    ASSERT_EQ(def.pivoting, Pivoting::Threshold);
    Control off = def;
    off.static_fallback_after_cascade = false;
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const CscLower A1 = kkt(kG, kM, 4.15, 3.0);
    std::vector<std::vector<double>> ref;
    for (const Control &ctrl : {off, def}) {
        Solver s;
        Info info;
        auto ak = s.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace ws;
        FactorKeep fk;
        std::vector<std::vector<double>> got;
        for (const CscLower *M : {&A, &A1, &A1}) {
            Info i;
            ASSERT_EQ(s.factor(*ak, *M, ctrl, i, fk, ws), FactorStatus::Success);
            EXPECT_EQ(i.static_fallback, 0);
            EXPECT_EQ(i.static_rejects, 0);
            EXPECT_FALSE(fk.static_path);
            got.push_back(fk.factor_values);
        }
        if (ref.empty()) ref = got;
        else EXPECT_EQ(got, ref);
    }
}

// The ipopt track (pivot_u = 1e-8, where the rescue is never armed) is
// untouched by the fallback: bitwise the same factors over a sequence.
TEST(StaticFallback, IpoptTrackUntouched) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const CscLower A1 = kkt(kG, kM, 4.15, 3.0);
    std::vector<std::vector<double>> ref;
    for (const bool on : {false, true}) {
        Control ctrl = base_control();
        ctrl.pivot_u = 1e-8;
        ctrl.static_fallback_after_cascade = on;
        Solver s;
        Info info;
        auto ak = s.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace ws;
        FactorKeep fk;
        std::vector<std::vector<double>> got;
        for (const CscLower *M : {&A, &A1, &A1}) {
            Info i;
            ASSERT_EQ(s.factor(*ak, *M, ctrl, i, fk, ws), FactorStatus::Success);
            EXPECT_EQ(i.static_fallback, 0);
            EXPECT_FALSE(fk.static_path);
            EXPECT_FALSE(fk.delay_static);
            got.push_back(fk.factor_values);
        }
        if (ref.empty()) ref = got;
        else EXPECT_EQ(got, ref);
    }
}

// The backward-error trigger: a static-path factor (Pivoting::Static) at a
// high pivot threshold forces pivots that fail it without perturbing any
// (static_pivots 0, pivot growth above 1 / u). refinement_advised() holds
// although nothing was perturbed; refine() drives omega to
// refine_omega_target, and takes a step when the growth passes
// refine_growth_threshold even if omega already met the target.
TEST(StaticFallback, BackwardErrorTriggerRefinesGrowthWithoutPerturbation) {
    const CscLower E = kkt(kG, kM, 4.1, 0.0);
    Control ctrl = fallback_control();
    ctrl.pivoting = Pivoting::Static;
    ctrl.pivot_u = 0.4;
    Solver s;
    Info info;
    auto ak = s.analyse(E, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, ctrl, i, fk), FactorStatus::Success);
    std::printf("[StaticFallback] static u=0.4: perturbed %d growth %.3g delays %d\n", i.static_pivots,
                i.pivot_growth, i.delayed_pivots);
    EXPECT_EQ(i.static_fallback, 1);
    EXPECT_EQ(i.static_pivots, 0);
    EXPECT_TRUE(i.inertia_exact);
    EXPECT_EQ(i.delayed_pivots, 0);
    EXPECT_GT(i.pivot_growth, 1.0 / ctrl.pivot_u);
    EXPECT_EQ(fk.pivot_growth, i.pivot_growth);
    EXPECT_TRUE(fk.static_path);
    EXPECT_TRUE(refinement_advised(fk));
    EXPECT_EQ(i.num_positive, kG * kG);
    EXPECT_EQ(i.num_negative, kM);

    const int n = static_cast<int>(E.n);
    std::vector<double> b(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) b[static_cast<std::size_t>(k)] = 1.0 + 0.01 * (k % 97);
    std::vector<double> x0 = b;
    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, x0.data(), n, 1), 0);
    {
        // Omega target: refined until omega <= refine_omega_target.
        Control c = ctrl;
        c.refine_growth_threshold = std::numeric_limits<double>::infinity();
        c.refine_omega_target = 1e-14;
        std::vector<double> x = x0;
        SolveWorkspace ws;
        const RefinementResult rr = refine(E, fk, c, b, x, 1, ws);
        std::printf("[StaticFallback] omega %.3g -> %.3g in %d steps\n", rr.omega_start, rr.omega,
                    rr.iterations);
        EXPECT_EQ(rr.status, ErrorCode::Success);
        if (rr.omega_start > c.refine_omega_target) {
            EXPECT_GE(rr.iterations, 1);
        }
        EXPECT_TRUE(rr.omega <= c.refine_omega_target || rr.iterations >= c.refinement_max_iters)
            << "omega " << rr.omega;
    }
    {
        // Growth past the threshold: at least one step even when omega
        // already meets the target.
        Control c = ctrl;
        c.refine_omega_target = 1.0;
        c.refine_growth_threshold = 1.0;
        ASSERT_GT(fk.pivot_growth, c.refine_growth_threshold);
        std::vector<double> x = x0;
        SolveWorkspace ws;
        const RefinementResult rr = refine(E, fk, c, b, x, 1, ws);
        EXPECT_GE(rr.iterations, 1);
        EXPECT_EQ(rr.status, ErrorCode::Success);
        // Below the growth threshold and at a loose target: no step.
        c.refine_growth_threshold = 2.0 * fk.pivot_growth;
        std::vector<double> y = x0;
        const RefinementResult r0 = refine(E, fk, c, b, y, 1, ws);
        EXPECT_EQ(r0.iterations, 0);
        EXPECT_EQ(y, x0);
    }
    // The default threshold-pivoting factor of the same matrix: no static
    // path, nothing advised.
    Control t = base_control();
    FactorKeep ft;
    Info it;
    auto akt = s.analyse(E, t, it);
    ASSERT_NE(akt.get(), nullptr);
    ASSERT_EQ(s.factor(*akt, t, it, ft), FactorStatus::Success);
    EXPECT_FALSE(ft.static_path);
    EXPECT_EQ(it.static_fallback, 0);
    EXPECT_FALSE(refinement_advised(ft));
}


// The growth-triggered forced step (refine_growth_threshold) under every
// refine_method: a high-growth static-path factor whose first solve already
// meets the omega target still takes (and counts) at least one step with
// Classical, Auto and Fgmres (Fgmres used to return before its first
// iteration).
TEST(StaticFallback, ForcedStepUnderEveryRefineMethod) {
    const CscLower E = kkt(kG, kM, 4.1, 0.0);
    Control ctrl = fallback_control();
    ctrl.pivoting = Pivoting::Static;
    ctrl.pivot_u = 0.4;
    Solver s;
    Info ia;
    auto ak = s.analyse(E, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, ctrl, i, fk), FactorStatus::Success);
    ASSERT_TRUE(fk.static_path);
    const int n = static_cast<int>(E.n);
    std::vector<double> b(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) b[static_cast<std::size_t>(k)] = 1.0 + 0.01 * (k % 97);
    std::vector<double> x0 = b;
    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, x0.data(), n, 1), 0);
    for (const RefineMethod m :
         {RefineMethod::Classical, RefineMethod::Auto, RefineMethod::Fgmres}) {
        Control c = ctrl;
        c.refine_method = m;
        c.refine_omega_target = 1.0;  // met before any step
        c.refine_growth_threshold = 1.0;
        ASSERT_GT(fk.pivot_growth, c.refine_growth_threshold);
        std::vector<double> x = x0;
        SolveWorkspace ws;
        const RefinementResult rr = refine(E, fk, c, b, x, 1, ws);
        std::printf("[StaticFallback] forced step, method %d: %d steps, omega %.3g -> %.3g\n",
                    static_cast<int>(m), rr.iterations, rr.omega_start, rr.omega);
        EXPECT_GE(rr.iterations, 1) << "method " << static_cast<int>(m);
        EXPECT_EQ(rr.status, ErrorCode::Success) << "method " << static_cast<int>(m);
        EXPECT_LE(rr.omega, rr.omega_start) << "method " << static_cast<int>(m);
        if (m == RefineMethod::Fgmres) {
            EXPECT_EQ(rr.method_used, RefineMethod::Fgmres);
            EXPECT_EQ(rr.iterations, 1) << "one forced iteration from a converged start";
        }
        // Below the growth threshold: no step.
        c.refine_growth_threshold = 2.0 * fk.pivot_growth;
        std::vector<double> y = x0;
        const RefinementResult r0 = refine(E, fk, c, b, y, 1, ws);
        EXPECT_EQ(r0.iterations, 0) << "method " << static_cast<int>(m);
        EXPECT_EQ(y, x0) << "method " << static_cast<int>(m);
    }
}

namespace {

// 3 x 3 tridiagonal [1e-12 1 0; 1 1 1; 0 1 1] (lower CSC): under the
// identity order with no amalgamation every column is its own front, and the
// first has a tiny diagonal and no 2x2 partner inside its front, so threshold
// pivoting delays it and the static path perturbs it instead (raised to
// 1e-8 * max|A|; an exact zero would stay a zero eigenvalue).
CscLower tridiag3() {
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 2, 4, 5};
    A.row_idx = {0, 1, 1, 2, 2};
    A.values = {1e-12, 1.0, 1.0, 1.0, 1.0};
    return A;
}

Control tridiag3_control(const Int *perm) {
    Control c = base_control();
    c.ordering = OrderingMethod::UserSupplied;
    c.user_perm = perm;
    c.nemin = 1;
    c.scaling = ScalingMethod::None;
    c.num_threads = 2;
    c.auto_threads = false;
    return c;
}

} // namespace

// Pivoting::Static with tree_parallel at num_threads > 1 (outside
// deterministic mode) used to run the legacy task-parallel engine, which
// ignores the delay budget: it threshold-pivoted (one delay, no
// perturbation) while reporting static_fallback = 1 and static_path. The
// static path now always runs on the default engine: the same factor as
// without tree_parallel.
TEST(StaticFallback, StaticPivotingIgnoresTreeParallel) {
    const CscLower A = tridiag3();
    const std::vector<Int> perm = {0, 1, 2};
    Solver s;
    // Reference: Pivoting::Static on the default engine.
    Control ref_ctrl = tridiag3_control(perm.data());
    ref_ctrl.pivoting = Pivoting::Static;
    Info ia;
    auto ak = s.analyse(A, ref_ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk_ref;
    Info ir;
    ASSERT_EQ(s.factor(*ak, ref_ctrl, ir, fk_ref), FactorStatus::Success);
    std::printf("[StaticFallback] tridiag static default engine: perturbed %d delays %d\n",
                ir.static_pivots, ir.delayed_pivots);
    EXPECT_EQ(ir.delayed_pivots, 0);
    EXPECT_EQ(ir.static_pivots, 1);
    EXPECT_EQ(ir.static_fallback, 1);
    EXPECT_TRUE(fk_ref.static_path);

    // The repro is meaningful: threshold pivoting on the legacy engine delays.
    Control tp = tridiag3_control(perm.data());
    tp.tree_parallel = true;
    if (!deterministic_mode(tp)) {
        FactorKeep fk_thr;
        Info it;
        ASSERT_EQ(s.factor(*ak, tp, it, fk_thr), FactorStatus::Success);
        std::printf("[StaticFallback] tridiag threshold tree_parallel: perturbed %d delays %d\n",
                    it.static_pivots, it.delayed_pivots);
        EXPECT_GE(it.delayed_pivots, 1);
        EXPECT_EQ(it.static_pivots, 0);
        EXPECT_FALSE(fk_thr.static_path);
    }

    // Pivoting::Static with tree_parallel: the static path, really.
    tp.pivoting = Pivoting::Static;
    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, tp, i, fk), FactorStatus::Success);
    std::printf("[StaticFallback] tridiag static tree_parallel: perturbed %d delays %d\n",
                i.static_pivots, i.delayed_pivots);
    EXPECT_EQ(i.delayed_pivots, 0);
    EXPECT_EQ(i.static_pivots, ir.static_pivots);
    EXPECT_EQ(i.static_fallback, 1);
    EXPECT_TRUE(fk.static_path);
    EXPECT_EQ(i.num_positive, ir.num_positive);
    EXPECT_EQ(i.num_negative, ir.num_negative);
    EXPECT_EQ(i.num_zero, ir.num_zero);
    EXPECT_EQ(fk.factor_values, fk_ref.factor_values);
}

// The static fallback (default off) is never armed on the legacy
// tree_parallel engine (scale-on-delay does not apply there), so a cascading
// sequence there reports no static-path factor: threshold pivoting, reported
// as such.
TEST(StaticFallback, FallbackNotArmedUnderTreeParallel) {
    const CscLower A = static_ok_kkt();
    Control ctrl = fallback_control();
    ctrl.tree_parallel = true;
    ctrl.num_threads = 2;
    ctrl.auto_threads = false;  // small test trees
    if (deterministic_mode(ctrl)) GTEST_SKIP() << "deterministic mode ignores tree_parallel";
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    for (int step = 0; step < 3; ++step) {
        Info i;
        ASSERT_EQ(s.factor(*ak, ctrl, i, fk), FactorStatus::Success);
        EXPECT_EQ(i.static_fallback, 0) << "step " << step;
        EXPECT_EQ(i.static_rejects, 0) << "step " << step;
        EXPECT_EQ(i.delay_stops, 0) << "step " << step;
        EXPECT_FALSE(fk.static_path) << "step " << step;
        EXPECT_FALSE(fk.delay_static) << "step " << step;
        EXPECT_GT(i.delayed_pivots, 0) << "the cascade, threshold-pivoted";
    }
}
