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
#include "smf/scaling.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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

Control base_control() {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    return c;  // Auto scaling, pivot_u = 0.01, default trigger ratio
}

// The matching scaling (ScalingMethod::Matching's) of M's values, M being
// a clean matrix of ak's pattern.
std::vector<double> matching_scale_of(const AnalysisKeep &ak, const CscLower &M) {
    CscLower clean = ak.cleaned;
    clean.values = M.values;
    std::vector<double> ref;
    EXPECT_EQ(compute_scaling_for_method(clean, ScalingMethod::Matching, ref), ErrorCode::Success);
    return ref;
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

// A kept scale that fits but whose factor fails the accuracy probe is not
// kept: the matching scaling is computed anew (the probe compares with the
// backward error the kept scale had when fresh, FactorKeep::delay_omega).
TEST(ScaleOnDelay, ReuseThatFailsTheProbeRecomputes) {
    const CscLower A = kkt(kG, kM, 4.1, 3.0);
    const Control ctrl = base_control();
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(s.factor(*ak, A, ctrl, i1, fk, ws), FactorStatus::Success);
    ASSERT_EQ(i1.delay_rescales, 1);
    ASSERT_EQ(fk.delay_analysis, ak->serial);
    EXPECT_GT(fk.delay_omega, 0.0);
    EXPECT_LE(fk.delay_omega, 1e-12);  // the fresh factor's probe
    const std::vector<double> scale_a = fk.delay_scale;

    // The kept scale fits A exactly (it is A's own); with the recorded
    // fresh error at 1, the probe passes: reuse.
    fk.delay_omega = 1.0;
    Info i2;
    ASSERT_EQ(s.factor(*ak, A, ctrl, i2, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i2.delay_rescales, 0);
    EXPECT_EQ(i2.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_EQ(fk.delay_scale, scale_a);

    // A recorded fresh error of 0 makes any probe error a failure: the
    // factor is redone with a newly computed matching scaling (the same
    // one, the values being the same) and the new probe error recorded.
    fk.delay_omega = 0.0;
    Info i3;
    ASSERT_EQ(s.factor(*ak, A, ctrl, i3, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i3.delay_rescales, 1);
    EXPECT_EQ(fk.delay_scale, scale_a);
    EXPECT_GT(fk.delay_omega, 0.0);
    EXPECT_EQ(i3.num_negative, kM);
    expect_accurate_solve(s, fk, ctrl, A);
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
    Info i2;
    ASSERT_EQ(s.factor(*ak, A2, ctrl, i2, fk, ws), FactorStatus::Success);
    EXPECT_EQ(i2.delay_rescales, 1);
    EXPECT_EQ(i2.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_LE(work_ratio(*ak, i2), ctrl.scale_on_delay_ratio);
    EXPECT_EQ(fk.delay_scale, matching_scale_of(*ak, A2));
    EXPECT_EQ(i2.num_negative, kM);
    EXPECT_GT(i2.factor_seconds, 0.0);
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
