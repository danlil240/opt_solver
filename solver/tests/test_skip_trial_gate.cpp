// Control::skip_trial_gate: analyse() skips the auto matching trial at
// IPOPT-style pivot thresholds when the base AMD factor's predicted flops
// per row lie in the gate's band, and the scale-on-delay rescue is then
// armed below kWeakPivotRatio as its safety net
// (Control::skip_trial_rescue_ratio).

#include "../benchmarks/common/validation.hpp"
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/factor_indef.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "kkt_test_matrices.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace smf {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// A boyd1-like KKT whose matcher's initial phase is diagonal
// (test_matching_trial.cpp's diagonal_kkt).
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

// test_scale_on_delay.cpp's badly scaled regularized KKT D [H B^T; B creg I] D
// (H: 5-point Laplacian of a g x g grid; m constraint rows of three
// variables; D_i = 10^U(-spread, spread)). With creg < 0 its inertia is g^2
// positive, m negative whatever D. Unscaled threshold pivoting delays most
// columns.
CscLower scaled_kkt(int g, int m, double hdiag, double spread, double creg, unsigned seed = 7) {
    const int nx = g * g, n = nx + m;
    std::mt19937 rng(7);
    std::mt19937 drng(seed);
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

// IPOPT's setting (the bench's ipopt track): u = 1e-8.
Control ipopt_control(int threads) {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.ordering = OrderingMethod::AutoSerial;
    c.pivot_u = 1e-8;
    c.skip_trial_gate = false;  // gate_on() arms it
    c.num_threads = threads;
    c.auto_threads = false;  // small test trees
    // Bitwise comparisons with threads: deterministic mode (multi-threaded
    // factorizations otherwise vary run to run at ~1e-14).
    c.deterministic = threads > 1;
    return c;
}

Control gate_on(Control c, double lo, double hi) {
    c.skip_trial_gate = true;
    c.skip_trial_min_flops_per_row = lo;
    c.skip_trial_max_flops_per_row = hi;
    return c;
}

std::vector<double> rhs_ones(const CscLower &A) {
    std::vector<double> b(static_cast<std::size_t>(A.n), 0.0);
    for (Int j = 0; j < A.n; ++j)
        for (Int q = A.col_ptr[j]; q < A.col_ptr[j + 1]; ++q) {
            const Int i = A.row_idx[q];
            b[static_cast<std::size_t>(i)] += A.values[q];
            if (i != j) b[static_cast<std::size_t>(j)] += A.values[q];
        }
    return b;
}

struct Run {
    std::unique_ptr<AnalysisKeep> ak;
    Info ia, ifac;
    FactorStatus st = FactorStatus::Success;
    FactorKeep fk;
    std::vector<double> x;
};

Run run(const CscLower &A, const Control &c) {
    Run r;
    Solver s;
    r.ak = s.analyse(A, c, r.ia);
    EXPECT_NE(r.ak, nullptr);
    if (!r.ak) return r;
    r.st = s.factor(*r.ak, c, r.ifac, r.fk);
    r.x = rhs_ones(A);
    if (r.st == FactorStatus::Success) {
        Info si;
        EXPECT_EQ(s.solve(r.fk, c, si, r.x.data(), static_cast<int>(A.n)), 0);
    }
    return r;
}

// Bitwise the same analysis and factorization.
void expect_same(const Run &a, const Run &b, const char *what) {
    EXPECT_EQ(b.ia.matching_trial, a.ia.matching_trial) << what;
    EXPECT_EQ(b.ia.matching_initial_ratio, a.ia.matching_initial_ratio) << what;
    EXPECT_EQ(b.ia.matching_exact_ratio, a.ia.matching_exact_ratio) << what;
    EXPECT_EQ(b.ia.ordering_used, a.ia.ordering_used) << what;
    EXPECT_FALSE(b.ia.matching_trial_skipped) << what;
    EXPECT_EQ(b.ak->perm, a.ak->perm) << what;
    EXPECT_EQ(b.ak->matching_scale, a.ak->matching_scale) << what;
    EXPECT_EQ(b.ak->matching_scale_partial, a.ak->matching_scale_partial) << what;
    EXPECT_EQ(b.ak->rejected_matching_scale, a.ak->rejected_matching_scale) << what;
    EXPECT_EQ(b.ak->matching_rejected, a.ak->matching_rejected) << what;
    EXPECT_FALSE(b.ak->matching_trial_skipped) << what;
    EXPECT_EQ(b.st, a.st) << what;
    EXPECT_EQ(b.ifac.delayed_pivots, a.ifac.delayed_pivots) << what;
    EXPECT_EQ(b.ifac.num_negative, a.ifac.num_negative) << what;
    EXPECT_EQ(b.ifac.num_zero, a.ifac.num_zero) << what;
    EXPECT_EQ(b.ifac.delay_rescales, a.ifac.delay_rescales) << what;
    EXPECT_EQ(b.fk.factor_values, a.fk.factor_values) << what;
    EXPECT_EQ(b.x, a.x) << what;
}

std::vector<CscLower> trial_matrices() {
    // Exact adopt, guard band reject, initial reject, guard band adopt
    // (test_matching_trial.cpp), and a badly scaled KKT.
    return {opf_like_kkt(10000, 1000, 7), opf_like_kkt(8000, 2000, 1), opf_like_kkt(6000, 5000, 7),
            scaled_opf_kkt(9000, 2000, 6, 1.0), scaled_kkt(100, 1500, 4.1, 3.0, -0.1)};
}

} // namespace

// The default Control has the gate on, with the fitted band, at IPOPT's
// threshold only; the IPOPT plugin's Control inherits it.
TEST(SkipTrialGate, DefaultIsOn) {
    const Control c;
    EXPECT_TRUE(c.skip_trial_gate);
    EXPECT_EQ(c.skip_trial_max_pivot_u, 1e-6);
    EXPECT_EQ(c.skip_trial_min_flops_per_row, 20.0);
    EXPECT_EQ(c.skip_trial_max_flops_per_row, 8e3);
    EXPECT_EQ(c.skip_trial_rescue_ratio, 16.0);
    EXPECT_GT(c.pivot_u, c.skip_trial_max_pivot_u) << "default pivot_u: not armed";
}

// Gate off, gate on but not armed (pivot_u above its bound), and gate armed
// with a band no matrix falls in (the deferred serial trial, the sliced
// concurrent one) all give the same analysis and factorization, bitwise:
// serially and with threads, at every budget path.
TEST(SkipTrialGate, OffOrOutsideTheBandIsBitwiseUnchanged) {
    for (const CscLower &A : trial_matrices()) {
        for (int threads : {1, 4}) {
            for (double budget : {-1.0, 0.0, 2.0}) {
                Control off = ipopt_control(threads);
                off.matching_search_budget = budget;
                const Run ref = run(A, off);
                ASSERT_NE(ref.ak, nullptr);
                ASSERT_NE(ref.ia.matching_trial, MatchingTrial::None) << "premise: the trial runs";
                EXPECT_EQ(ref.ia.matching_gate_flops_per_row, 0.0);

                Control strong = gate_on(off, 0.0, kInf);
                strong.pivot_u = 0.01;  // above skip_trial_max_pivot_u: not armed
                Control strong_off = off;
                strong_off.pivot_u = 0.01;
                const Run rs = run(A, strong_off);
                const Run gs = run(A, strong);
                expect_same(rs, gs, "pivot_u above the gate's bound");
                EXPECT_EQ(gs.ia.matching_gate_flops_per_row, 0.0);

                for (auto [lo, hi] : {std::pair{kInf, kInf}, std::pair{0.0, 0.0}}) {
                    const Run g = run(A, gate_on(off, lo, hi));
                    expect_same(ref, g, "armed, outside the band");
                    EXPECT_GT(g.ia.matching_gate_flops_per_row, 0.0);
                    EXPECT_EQ(g.ifac.skip_trial_rescues, 0);
                }
            }
        }
    }
}

// Inside the band the trial is skipped: no verdict, the base AMD ordering
// (that of OrderingMethod::AMD), no matching scale, matching_rejected set
// as after a rejection; the factor has the matrix's inertia and solves.
// Just outside it (the band ending below or starting above the matrix's
// flops per row) the trial runs.
TEST(SkipTrialGate, InsideTheBandSkipsTheTrial) {
    for (const CscLower &A : trial_matrices()) {
        for (int threads : {1, 4}) {
            const Control off = ipopt_control(threads);
            const Run ref = run(A, off);
            const Run probe = run(A, gate_on(off, kInf, kInf));
            const double flr = probe.ia.matching_gate_flops_per_row;
            ASSERT_GT(flr, 0.0);

            const Run g = run(A, gate_on(off, flr, flr));
            ASSERT_NE(g.ak, nullptr);
            EXPECT_TRUE(g.ia.matching_trial_skipped) << "threads " << threads;
            EXPECT_TRUE(g.ak->matching_trial_skipped);
            EXPECT_EQ(g.ia.matching_gate_flops_per_row, flr);
            EXPECT_EQ(g.ak->matching_gate_flops_per_row, flr);
            EXPECT_EQ(g.ia.matching_trial, MatchingTrial::None);
            EXPECT_EQ(g.ia.matching_initial_ratio, 0.0);
            EXPECT_EQ(g.ia.matching_exact_ratio, 0.0);
            EXPECT_EQ(g.ia.ordering_used, static_cast<int>(OrderingMethod::AMD));
            EXPECT_TRUE(g.ak->matching_scale.empty());
            EXPECT_TRUE(g.ak->rejected_matching_scale.empty());
            EXPECT_TRUE(g.ak->matching_rejected);

            Control amd = off;
            amd.ordering = OrderingMethod::AMD;
            Solver sa;
            Info ia;
            auto aa = sa.analyse(A, amd, ia);
            ASSERT_NE(aa, nullptr);
            EXPECT_EQ(g.ak->perm, aa->perm) << "the base AMD ordering";

            ASSERT_EQ(g.st, FactorStatus::Success);
            EXPECT_EQ(g.ifac.num_negative, ref.ifac.num_negative);
            EXPECT_EQ(g.ifac.num_zero, ref.ifac.num_zero);
            EXPECT_EQ(g.ifac.num_positive, ref.ifac.num_positive);
            double err = 0.0;
            for (double v : g.x) err = std::max(err, std::abs(v - 1.0));
            EXPECT_LT(err, 1e-6) << "threads " << threads;

            // Band below or above the matrix: the trial runs as with the gate off.
            expect_same(ref, run(A, gate_on(off, 0.0, std::nextafter(flr, 0.0))), "band below");
            expect_same(ref, run(A, gate_on(off, std::nextafter(flr, kInf), kInf)), "band above");
        }
    }
}

// The diagonal verdict and its partial scale are kept whatever the band
// (the gate only skips the searches of a non-diagonal trial).
TEST(SkipTrialGate, DiagonalTrialIsKept) {
    const CscLower A = diagonal_kkt(10000, 3);
    for (int threads : {1, 4}) {
        const Control off = ipopt_control(threads);
        const Run ref = run(A, off);
        ASSERT_EQ(ref.ia.matching_trial, MatchingTrial::Diagonal);
        const Run g = run(A, gate_on(off, 0.0, kInf));
        expect_same(ref, g, "diagonal");
        EXPECT_EQ(g.ia.matching_gate_flops_per_row, 0.0);
        EXPECT_TRUE(g.ak->matching_scale_partial);
        EXPECT_FALSE(g.ak->matching_scale.empty());
    }
}

// The safety net: a badly scaled KKT (D_i = 10^U(-6, 6)) whose skipped
// trial leaves the base AMD ordering unscaled, so even at u = 1e-8 its
// delays inflate the front work past the rescue's ratio (86x on the Smf
// rule's tree; the trial's matching ordering and scale factor it with no
// delay). Without the net (ratio < 0) nothing stops it;
// with it the try stops and is redone with the matching scaling of the
// values: one rescue, far fewer delays, the KKT inertia, an accurate solve.
// A later factorization of the same values reuses the kept scale.
TEST(SkipTrialGate, SafetyNetRescuesASkippedTrialThatNeededMatching) {
    const CscLower A = scaled_kkt(100, 1500, 4.0, 6.0, -0.1);
    const Int nx = 100 * 100, m = 1500;
    Control base = ipopt_control(1);
    base.amalgamation = Amalgamation::Smf;  // as test_scale_on_delay.cpp: narrower fronts, larger ratios
    Control on = gate_on(base, 0.0, kInf);
    Control bare = on;
    bare.skip_trial_rescue_ratio = -1.0;

    const Run ref = run(A, base);
    std::printf("[SkipTrialGate] trial: %d delays, front work %.3g\n", ref.ifac.delayed_pivots,
                ref.ifac.front_work);
    const Run b = run(A, bare);
    ASSERT_TRUE(b.ia.matching_trial_skipped);
    const double free_work = delay_free_front_work(*b.ak);
    const double ratio = b.ifac.front_work / free_work;
    std::printf("[SkipTrialGate] skipped, no net: %d delays, work ratio %.1f\n", b.ifac.delayed_pivots, ratio);
    ASSERT_GT(ratio, on.skip_trial_rescue_ratio) << "premise: the skip lets the delays cascade";
    EXPECT_EQ(b.ifac.delay_rescales, 0);
    EXPECT_EQ(b.ifac.skip_trial_rescues, 0);

    Solver s;
    Info ia, i1, i2;
    auto ak = s.analyse(A, on, ia);
    ASSERT_NE(ak, nullptr);
    ASSERT_TRUE(ia.matching_trial_skipped);
    FactorKeep fk;
    ASSERT_EQ(s.factor(*ak, on, i1, fk), FactorStatus::Success);
    std::printf("[SkipTrialGate] skipped, net: %d delays, rescues %d\n", i1.delayed_pivots, i1.skip_trial_rescues);
    EXPECT_EQ(i1.delay_rescales, 1);
    EXPECT_EQ(i1.skip_trial_rescues, 1);
    EXPECT_EQ(i1.delay_stops, 1);
    EXPECT_EQ(i1.delay_early_stops, 0) << "the gate's rescue has no early stop";
    EXPECT_EQ(i1.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_LT(i1.delayed_pivots, b.ifac.delayed_pivots / 4);
    EXPECT_EQ(i1.num_positive, nx);
    EXPECT_EQ(i1.num_negative, m);
    EXPECT_EQ(i1.num_zero, 0);
    std::vector<double> b1 = rhs_ones(A), x = b1;
    Info si;
    ASSERT_EQ(s.solve(fk, on, si, x.data(), static_cast<int>(A.n)), 0);
    // As accurate as the trial's factor (both unrefined; the scaling spans
    // 24 orders of magnitude).
    const auto val = smf::bench::validate(A, b1, x, 1);
    const auto val_ref = smf::bench::validate(A, b1, ref.x, 1);
    std::printf("[SkipTrialGate] eta_inf %.3g omega %.3g (trial: %.3g, %.3g)\n", val.eta_inf, val.omega,
                val_ref.eta_inf, val_ref.omega);
    EXPECT_LE(val.eta_inf, 1e-10);
    EXPECT_LE(val.omega, 10.0 * val_ref.omega);

    // The next factorization starts from the kept scale: no new rescue.
    ASSERT_EQ(s.factor(*ak, on, i2, fk), FactorStatus::Success);
    EXPECT_EQ(i2.delay_rescales, 0);
    EXPECT_EQ(i2.skip_trial_rescues, 0);
    EXPECT_EQ(i2.scaling_used, static_cast<int>(ScalingMethod::Matching));
    EXPECT_EQ(i2.delayed_pivots, i1.delayed_pivots);

    // Not armed without a skip: the same matrix with the gate off never
    // rescues at u = 1e-8 (scale_on_delay_armed needs u >= 1e-6).
    EXPECT_EQ(ref.ifac.delay_rescales, 0);
    EXPECT_EQ(ref.ifac.skip_trial_rescues, 0);
    // Nor under ScalingMethod::None (the IPOPT plugin's MA97 scaling 0).
    Control none = on;
    none.scaling = ScalingMethod::None;
    const Run nn = run(A, none);
    EXPECT_TRUE(nn.ia.matching_trial_skipped);
    EXPECT_EQ(nn.ifac.delay_rescales, 0);
    EXPECT_EQ(nn.ifac.skip_trial_rescues, 0);
}

// The gate decides on AMD's ordering, the same at every thread count, and
// the rescue's decision is the finished count's: in deterministic mode the
// skip, the gate's flops per row, the ordering, the rescue and the solution
// are identical at 1, 2 and 4 threads, AutoSerial and AutoParallel.
TEST(SkipTrialGate, DeterministicAcrossThreadCounts) {
    // The last one fires the safety net (SafetyNetRescuesASkippedTrialThatNeededMatching).
    const std::vector<CscLower> mats = {opf_like_kkt(10000, 1000, 7), opf_like_kkt(6000, 5000, 7),
                                        scaled_kkt(100, 1500, 4.0, 6.0, -0.1)};
    for (const CscLower &A : mats) {
        for (OrderingMethod ord : {OrderingMethod::AutoSerial, OrderingMethod::AutoParallel}) {
            Run first;
            bool have = false;
            for (int threads : {1, 2, 4}) {
                Control c = gate_on(ipopt_control(threads), 0.0, kInf);
                c.ordering = ord;
                c.amalgamation = Amalgamation::Smf;
                c.deterministic = true;
                Run r = run(A, c);
                ASSERT_NE(r.ak, nullptr);
                EXPECT_TRUE(r.ia.matching_trial_skipped) << "threads " << threads;
                if (!have) {
                    first = std::move(r);
                    have = true;
                    continue;
                }
                EXPECT_EQ(r.ia.matching_gate_flops_per_row, first.ia.matching_gate_flops_per_row);
                EXPECT_EQ(r.ak->perm, first.ak->perm) << "threads " << threads;
                EXPECT_EQ(r.st, first.st);
                EXPECT_EQ(r.ifac.skip_trial_rescues, first.ifac.skip_trial_rescues) << "threads " << threads;
                EXPECT_EQ(r.ifac.delayed_pivots, first.ifac.delayed_pivots) << "threads " << threads;
                EXPECT_EQ(r.ifac.num_negative, first.ifac.num_negative);
                EXPECT_EQ(r.x, first.x) << "threads " << threads;
            }
        }
    }
}

} // namespace smf
