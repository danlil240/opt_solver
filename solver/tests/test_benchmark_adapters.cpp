#include "../benchmarks/adapters/adapter.hpp"
#include "../benchmarks/common/ipopt_dynamic_scaling.hpp"
#include "../benchmarks/common/replay_refine.hpp"
#include "../benchmarks/common/replay_steps.hpp"
#include "../benchmarks/common/replay_track.hpp"
#include "../benchmarks/common/validation.hpp"
#include "smf/ipopt_plugin_control.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace smf;
using namespace smf::bench;

namespace {

CscLower make_identity_spd_3x3() {
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 1, 2, 3};
    A.row_idx = {0, 1, 2};
    A.values = {1.0, 1.0, 1.0};
    return A;
}

CscLower make_exchange_2x2_indef() {
    CscLower A;
    A.n = 2;
    A.col_ptr = {0, 2, 3};
    A.row_idx = {0, 1, 1};
    A.values = {0.0, 1.0, 0.0};
    return A;
}

CscLower make_singular_2x2() {
    CscLower A;
    A.n = 2;
    A.col_ptr = {0, 1, 2};
    A.row_idx = {0, 1};
    A.values = {2.0, 0.0};
    return A;
}

} // namespace

TEST(BenchmarkAdapters, SmfAdapterAndAllRegisteredAdapters) {
    auto ids = all_registered_adapter_ids();
    ASSERT_FALSE(ids.empty());

    for (const auto &id : ids) {
        SCOPED_TRACE("adapter=" + id);
        auto adapter = make_solver_adapter(id);
        ASSERT_NE(adapter.get(), nullptr) << "Null adapter for " << id;
        AdapterIdentity ident = adapter->identity();
        EXPECT_EQ(ident.solver_id, id);
        EXPECT_FALSE(ident.integer_abi.empty());
        // Never label libsmf_ma97 as genuine HSL ma97!
        if (id == "ma97") {
            EXPECT_EQ(ident.library_path.find("smf_ma97"), std::string::npos);
        }

        if (!ident.available) {
            CscLower I3 = make_identity_spd_3x3();
            EXPECT_EQ(adapter->analyse(I3), Outcome::Unavailable);
            EXPECT_EQ(adapter->factor(I3.values), Outcome::Unavailable);
            std::vector<double> b = {1.0, 2.0, 3.0};
            EXPECT_EQ(adapter->solve(b, 1), Outcome::Unavailable);
            continue;
        }

        // Test 1: Identity SPD
        if (ident.supports_spd) {
            Control ctrl;
            ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
            adapter->configure(ctrl, "recommended");
            CscLower I3 = make_identity_spd_3x3();
            ASSERT_EQ(adapter->analyse(I3), Outcome::Success);
            ASSERT_EQ(adapter->factor(I3.values), Outcome::Success);
            std::vector<double> b = {1.0, 2.0, 3.0};
            std::vector<double> x = b;
            ASSERT_EQ(adapter->solve(x, 1), Outcome::Success);
            auto v = validate(I3, b, x, 1);
            EXPECT_TRUE(is_qualified_validation(v));

            // Changed-value replay on the same pattern: diag(2, 4, 8)
            std::vector<double> updated_vals = {2.0, 4.0, 8.0};
            ASSERT_EQ(adapter->factor(updated_vals), Outcome::Success);
            // Multi-RHS (nrhs = 2)
            std::vector<double> b2 = {2.0, 8.0, 24.0, 4.0, 12.0, 16.0};
            std::vector<double> x2 = b2;
            ASSERT_EQ(adapter->solve(x2, 2), Outcome::Success);
            CscLower A_upd = I3;
            A_upd.values = updated_vals;
            auto v2 = validate(A_upd, b2, x2, 2);
            EXPECT_TRUE(is_qualified_validation(v2));
        }

        // Test 2: 2x2 exchange matrix where indefinite is supported
        if (ident.supports_indefinite) {
            Control ctrl;
            ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
            // cuDSS has no 2x2 pivots (static pivoting only) and, as on the
            // recommended track, does no refinement; its exchange-matrix solve
            // is only exact with its own iterative refinement (ipopt track).
            adapter->configure(ctrl, id == "cudss" ? "ipopt" : "recommended");
            CscLower E2 = make_exchange_2x2_indef();
            ASSERT_EQ(adapter->analyse(E2), Outcome::Success);
            ASSERT_EQ(adapter->factor(E2.values), Outcome::Success);
            std::vector<double> b = {1.0, 2.0};
            std::vector<double> x = b;
            ASSERT_EQ(adapter->solve(x, 1), Outcome::Success);
            EXPECT_NEAR(x[0], 2.0, 1e-12);
            EXPECT_NEAR(x[1], 1.0, 1e-12);
            AdapterStats st = adapter->stats();
            EXPECT_EQ(st.inertia_pos, 1);
            EXPECT_EQ(st.inertia_neg, 1);
            EXPECT_EQ(st.inertia_zero, 0);
        }

        // Test 3: Singularity must never produce a falsely successful solve.
        // Factor-time detection (MUMPS/CHOLMOD/smf) is preferred, but solvers
        // that internally regularize (e.g. MKL PARDISO pivot perturbation) may
        // report factor Success; in that case the solution must fail the shared
        // backward-error qualification on the original system.
        {
            Control ctrl;
            ctrl.matrix_type = ident.supports_indefinite
                                   ? MatrixType::RealSymmetricIndefinite
                                   : MatrixType::RealSymmetricPositiveDefinite;
            adapter->configure(ctrl, "recommended");
            CscLower S2 = make_singular_2x2();
            ASSERT_EQ(adapter->analyse(S2), Outcome::Success);
            Outcome f_out = adapter->factor(S2.values);
            if (f_out == Outcome::Success) {
                std::vector<double> b = {1.0, 1.0};
                std::vector<double> x = b;
                Outcome s_out = adapter->solve(x, 1);
                if (s_out == Outcome::Success) {
                    auto v = validate(S2, b, x, 1);
                    EXPECT_FALSE(is_qualified_validation(v))
                        << "singular solve passed qualification for " << id;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// bench_replay's IPM option: the smf adapter routed through smf::IpmSession
// with a static mode (--ipm-static off|fallback|regularized,
// --ipm-static-zero), and the replayed steps of a recorded sequence
// (--seq-steps N|all).

namespace {

// diag(1, d, 0.5) with no scaling: max|A| = 1, so Regularized's rho = 1e-8.
CscLower make_tiny_diag(double d) {
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 1, 2, 3};
    A.row_idx = {0, 1, 2};
    A.values = {1.0, d, 0.5};
    return A;
}

Control replay_control() {
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.scaling = ScalingMethod::None;
    ctrl.pivot_u = 1e-8;
    return ctrl;
}

} // namespace

TEST(BenchmarkAdapters, ParseIpmStatic) {
    EXPECT_EQ(parse_ipm_static("off"), StaticMode::Off);
    EXPECT_EQ(parse_ipm_static("fallback"), StaticMode::Fallback);
    EXPECT_EQ(parse_ipm_static("regularized"), StaticMode::Regularized);
    EXPECT_FALSE(parse_ipm_static("").has_value());
    EXPECT_FALSE(parse_ipm_static("reg").has_value());
}

TEST(BenchmarkAdapters, IpoptTrackFactorizesLikeThePlugin) {
    // bench_replay --track ipopt: IPOPT's pivot threshold; a replayed IPOPT
    // sequence (warm_sequence) is also analysed for the plugin's
    // refactorization estimate (which steers the analysis: the compressed
    // METIS trial, the matching resume), as the plugin would analyse it.
    Control c;
    smf::bench::apply_replay_track(c, "ipopt", /*sequence=*/true);
    EXPECT_EQ(c.pivot_u, 1e-8);
    EXPECT_EQ(c.expected_refactors, kIpoptExpectedRefactors);
    EXPECT_EQ(c.expected_refactors, ipopt_plugin_control()->expected_refactors);
    // One cold factorization is analysed for one factorization.
    Control one;
    smf::bench::apply_replay_track(one, "ipopt", /*sequence=*/false);
    EXPECT_EQ(one.pivot_u, 1e-8);
    EXPECT_EQ(one.expected_refactors, Control{}.expected_refactors);
    // Other tracks keep the library's defaults.
    Control r;
    smf::bench::apply_replay_track(r, "recommended", /*sequence=*/true);
    EXPECT_EQ(r.pivot_u, Control{}.pivot_u);
    EXPECT_EQ(r.expected_refactors, Control{}.expected_refactors);
}

TEST(BenchmarkAdapters, IpoptPluginControlDefaults) {
    // The Control the MA97 plugin factorizes with under IPOPT's defaults
    // (bench_replay --track ipopt-plugin).
    const auto ctrl = ipopt_plugin_control();
    ASSERT_TRUE(ctrl.has_value());
    EXPECT_EQ(ctrl->matrix_type, MatrixType::RealSymmetricIndefinite);
    // An IPOPT run refactors its pattern once per iteration or more: 25 is
    // about the lower quartile of the Mittelmann AMPL-NLP runs' iterations
    // (21) and the median of the PR #64 probe runs' factorizations (24.5).
    EXPECT_EQ(kIpoptExpectedRefactors, 25);
    EXPECT_EQ(ctrl->expected_refactors, kIpoptExpectedRefactors);
    EXPECT_EQ(ctrl->ordering, OrderingMethod::AutoParallel);
    EXPECT_EQ(ctrl->scaling, ScalingMethod::None);
    EXPECT_EQ(ctrl->pivot_u, 1e-8);
    EXPECT_EQ(ctrl->small_pivot, 1e-20);
    EXPECT_EQ(ctrl->nemin, 8);
    EXPECT_EQ(ctrl->factor_memory_multiplier, 1.2);
    EXPECT_TRUE(ctrl->continue_on_singular);
    EXPECT_EQ(ctrl->delay_budget, -1);
    // The plugin's thread count (SMF_NUM_THREADS / OMP_NUM_THREADS /
    // performance cores); the rest of the parallel set-up stays at the defaults.
    EXPECT_EQ(ctrl->num_threads, plugin_num_threads());
    EXPECT_GE(ctrl->num_threads, 1);
    EXPECT_EQ(ctrl->auto_threads, Control{}.auto_threads);
    EXPECT_EQ(ctrl->top_tiles, Control{}.top_tiles);
    EXPECT_EQ(ctrl->deterministic, Control{}.deterministic);
    EXPECT_EQ(ipopt_plugin_control(Ma97Options{}, -1, 3)->num_threads, 3);
    Control c;
    ASSERT_TRUE(apply_ma97_options(c, Ma97Options{}, -1, 5));
    EXPECT_EQ(c.num_threads, 5);
    ASSERT_TRUE(apply_ma97_options(c, Ma97Options{}));
    EXPECT_EQ(c.num_threads, plugin_num_threads());

    Ma97Options o;
    o.ordering = 1;
    o.scaling = 1;
    EXPECT_EQ(ipopt_plugin_control(o)->ordering, OrderingMethod::AMD);
    EXPECT_EQ(ipopt_plugin_control(o)->scaling, ScalingMethod::Matching);
    o.ordering = 3;
    o.scaling = 3;
    EXPECT_EQ(ipopt_plugin_control(o)->ordering, OrderingMethod::METIS);
    EXPECT_EQ(ipopt_plugin_control(o)->scaling, ScalingMethod::Equilibration);
    o.scaling = 7;
    EXPECT_FALSE(ipopt_plugin_control(o).has_value());
}

namespace {

void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value != nullptr ? value : "");  // "" removes it
#else
    if (value != nullptr) setenv(name, value, 1); else unsetenv(name);
#endif
}

// Sets / unsets variables for its lifetime and restores their old values.
struct EnvScope {
    std::vector<std::pair<std::string, std::optional<std::string>>> saved;
    void set(const char *name, const char *value) {
        const char *old = std::getenv(name);
        saved.emplace_back(name, old != nullptr ? std::optional<std::string>(old) : std::nullopt);
        set_env(name, value);
    }
    ~EnvScope() {
        for (auto it = saved.rbegin(); it != saved.rend(); ++it)
            set_env(it->first.c_str(), it->second ? it->second->c_str() : nullptr);
    }
};

} // namespace

TEST(BenchmarkAdapters, IpoptPluginThreadResolution) {
    // Pure resolution: SMF_NUM_THREADS, else OMP_NUM_THREADS (its first
    // entry), else the performance cores; invalid values count as unset.
    using S = PluginThreadsSource;
    const auto check = [](const char *smf_v, const char *omp_v, int cores, int want, S src) {
        const PluginThreads t = plugin_threads_resolve(smf_v, omp_v, cores);
        EXPECT_EQ(t.threads, want) << (smf_v ? smf_v : "unset") << " / " << (omp_v ? omp_v : "unset");
        EXPECT_EQ(t.source, src) << (smf_v ? smf_v : "unset") << " / " << (omp_v ? omp_v : "unset");
    };
    check(nullptr, nullptr, 8, 8, S::PerformanceCores);
    check(nullptr, nullptr, 0, 1, S::PerformanceCores);
    check(nullptr, "4", 8, 4, S::OmpNumThreads);
    check(nullptr, "6,2", 8, 6, S::OmpNumThreads);
    check(nullptr, " 3 ", 8, 3, S::OmpNumThreads);
    check("2", "4", 8, 2, S::SmfNumThreads);
    check("2", nullptr, 8, 2, S::SmfNumThreads);
    check("0", "4", 8, 4, S::OmpNumThreads);
    check("-3", "4", 8, 4, S::OmpNumThreads);
    check("abc", "4", 8, 4, S::OmpNumThreads);
    check("", "4", 8, 4, S::OmpNumThreads);
    check("2x", nullptr, 8, 8, S::PerformanceCores);
    check(nullptr, "0", 8, 8, S::PerformanceCores);
    check(nullptr, ",4", 8, 8, S::PerformanceCores);
    check(nullptr, "", 8, 8, S::PerformanceCores);
    check(nullptr, "+4", 8, 8, S::PerformanceCores);
    check(nullptr, "3\r\n", 8, 3, S::OmpNumThreads);
    check("5\n", "3", 8, 5, S::SmfNumThreads);
    check(nullptr, " \t", 8, 8, S::PerformanceCores);
    check("8000", nullptr, 8, kPluginMaxThreads, S::SmfNumThreads);
    check(nullptr, "99999999999999999999999", 8, kPluginMaxThreads, S::OmpNumThreads);
    check(nullptr, "4096", 8, 4096, S::OmpNumThreads);
    EXPECT_EQ(std::string(plugin_threads_source_name(S::SmfNumThreads)), "SMF_NUM_THREADS");
    EXPECT_EQ(std::string(plugin_threads_source_name(S::OmpNumThreads)), "OMP_NUM_THREADS");
    EXPECT_EQ(std::string(plugin_threads_source_name(S::PerformanceCores)), "performance cores");

    // The same order through the environment (read now, not the cached
    // plugin_threads()); the variables are restored afterwards. (No upper
    // bound from hardware_concurrency(): it sees one processor group only
    // on Windows machines with more than 64 logical processors.)
    const int cores = performance_core_count();
    EXPECT_GE(cores, 1);
    std::printf("[IpoptPluginThreadResolution] performance_core_count() = %d\n", cores);
    const auto read = [](const char *name) {
        const char *v = std::getenv(name);
        return v != nullptr ? std::optional<std::string>(v) : std::nullopt;
    };
    const auto smf_before = read("SMF_NUM_THREADS");
    const auto omp_before = read("OMP_NUM_THREADS");
    {
        EnvScope env;
        env.set("SMF_NUM_THREADS", nullptr);
        env.set("OMP_NUM_THREADS", nullptr);
        EXPECT_EQ(plugin_threads_from_env().threads, cores);
        EXPECT_EQ(plugin_threads_from_env().source, S::PerformanceCores);
        set_env("OMP_NUM_THREADS", "3");
        EXPECT_EQ(plugin_threads_from_env().threads, 3);
        EXPECT_EQ(plugin_threads_from_env().source, S::OmpNumThreads);
        set_env("OMP_NUM_THREADS", "5,1");
        EXPECT_EQ(plugin_threads_from_env().threads, 5);
        set_env("SMF_NUM_THREADS", "2");
        EXPECT_EQ(plugin_threads_from_env().threads, 2);
        EXPECT_EQ(plugin_threads_from_env().source, S::SmfNumThreads);
        set_env("OMP_NUM_THREADS", nullptr);
        EXPECT_EQ(plugin_threads_from_env().threads, 2);
        set_env("SMF_NUM_THREADS", "0");
        EXPECT_EQ(plugin_threads_from_env().threads, cores);
        EXPECT_EQ(plugin_threads_from_env().source, S::PerformanceCores);
    }
    EXPECT_EQ(read("SMF_NUM_THREADS"), smf_before);
    EXPECT_EQ(read("OMP_NUM_THREADS"), omp_before);
}

TEST(BenchmarkAdapters, PhysicalCoresFollowAffinity) {
    // physical_core_count()'s rule on a synthetic topology: 4 cores with 2
    // SMT siblings each (CPUs 0-7), plus a core of processor group 1
    // (Windows ids: group * 64 + bit).
    const std::vector<std::vector<int>> cores = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {64, 65}};
    EXPECT_EQ(count_allowed_cores(cores, {}), 5);                       // affinity unknown: all
    EXPECT_EQ(count_allowed_cores(cores, {0, 1, 2, 3, 4, 5, 6, 7, 64, 65}), 5);
    EXPECT_EQ(count_allowed_cores(cores, {0, 1}), 1);                   // taskset -c 0,1: one core
    EXPECT_EQ(count_allowed_cores(cores, {0, 2}), 2);                   // one sibling of two cores
    EXPECT_EQ(count_allowed_cores(cores, {7, 3, 5}), 3);                // unsorted
    EXPECT_EQ(count_allowed_cores(cores, {64}), 1);                     // group 1 only
    EXPECT_EQ(count_allowed_cores(cores, {0, 1, 2, 3, 4, 5, 6, 7}), 4); // primary group only
    EXPECT_EQ(count_allowed_cores(cores, {100}), 0);                    // ids from elsewhere
}

TEST(BenchmarkAdapters, PerformanceCoresTakeTopClass) {
    // performance_core_count()'s rule on synthetic topologies.
    // Non-hybrid: 4 cores with SMT siblings, all of one class: every core.
    const std::vector<std::vector<int>> smt = {{0, 1}, {2, 3}, {4, 5}, {6, 7}};
    EXPECT_EQ(count_allowed_performance_cores(smt, {0, 0, 0, 0}, {}), 4);
    EXPECT_EQ(count_allowed_performance_cores(smt, {0, 0, 0, 0}, {0, 2, 5}), 3);
    // No class information (empty, or not one per core): as count_allowed_cores().
    EXPECT_EQ(count_allowed_performance_cores(smt, {}, {}), 4);
    EXPECT_EQ(count_allowed_performance_cores(smt, {}, {0, 1}), 1);
    EXPECT_EQ(count_allowed_performance_cores(smt, {1, 0}, {}), 4);
    // i9-12900K: 8 P-cores with 2 CPUs each (0-15, EfficiencyClass 1) and 8
    // E-cores with one (16-23, class 0).
    std::vector<std::vector<int>> hyb;
    std::vector<int> cls;
    for (int c = 0; c < 8; ++c) hyb.push_back({2 * c, 2 * c + 1}), cls.push_back(1);
    for (int c = 16; c < 24; ++c) hyb.push_back({c}), cls.push_back(0);
    std::vector<int> all(24);
    for (int c = 0; c < 24; ++c) all[static_cast<std::size_t>(c)] = c;
    EXPECT_EQ(count_allowed_cores(hyb, all), 16);
    EXPECT_EQ(count_allowed_performance_cores(hyb, cls, {}), 8);
    EXPECT_EQ(count_allowed_performance_cores(hyb, cls, all), 8);
    // Affinity: two P-cores and every E-core: the two P-cores.
    EXPECT_EQ(count_allowed_performance_cores(hyb, cls, {0, 3, 16, 17, 18, 19, 20, 21, 22, 23}), 2);
    // Affinity without any P-core: the allowed E-cores (top class among the allowed).
    EXPECT_EQ(count_allowed_performance_cores(hyb, cls, {16, 17, 18, 19, 20, 21, 22, 23}), 8);
    EXPECT_EQ(count_allowed_performance_cores(hyb, cls, {20, 22}), 2);
    // More than two classes (e.g. a CPU with low-power E-cores): only the top one.
    EXPECT_EQ(count_allowed_performance_cores({{0}, {1}, {2}, {3}}, {2, 1, 0, 2}, {}), 2);
    EXPECT_EQ(count_allowed_performance_cores({{0}, {1}, {2}, {3}}, {2, 1, 0, 2}, {1, 2}), 1);
    // Ids from elsewhere: nothing allowed (the caller falls back).
    EXPECT_EQ(count_allowed_performance_cores(hyb, cls, {100}), 0);

    // This machine's counts (the plugin's default is the second).
    const int physical = physical_core_count();
    const int performance = performance_core_count();
    std::printf("[PerformanceCoresTakeTopClass] physical_core_count() = %d, performance_core_count() = %d\n",
                physical, performance);
    EXPECT_GE(performance, 1);
    EXPECT_LE(performance, physical);
}

TEST(BenchmarkAdapters, SmfIpoptPluginAdapterDefersAnalysis) {
    auto ad = make_smf_ipopt_plugin_adapter(ipopt_plugin_ipm_control());
    ASSERT_NE(ad.get(), nullptr);
    EXPECT_EQ(ad->identity().solver_id, "smf");
    ad->configure(*ipopt_plugin_control(), "ipopt-plugin");
    CscLower E2 = make_exchange_2x2_indef();
    ASSERT_EQ(ad->analyse(E2), Outcome::Success);
    ASSERT_EQ(ad->factor(E2.values), Outcome::Success);
    std::vector<double> x = {1.0, 2.0};
    ASSERT_EQ(ad->solve(x, 1), Outcome::Success);
    EXPECT_NEAR(x[0], 2.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0, 1e-12);
    EXPECT_EQ(ad->stats().inertia_neg, 1);
    EXPECT_TRUE(refine_on_track("ipopt-plugin", 1));
    EXPECT_FALSE(refine_on_track("ipopt-plugin", 2));

    // A pattern large enough to time: analyse() only records the pattern,
    // the first factor() runs the analysis and books it as analyse_ms.
    CscLower T;
    T.n = 3000;
    T.col_ptr.push_back(0);
    for (Int j = 0; j < T.n; ++j) {
        T.row_idx.push_back(j);
        T.values.push_back(j % 2 == 0 ? 2.0 : -2.0);
        if (j + 1 < T.n) {
            T.row_idx.push_back(j + 1);
            T.values.push_back(1.0);
        }
        T.col_ptr.push_back(static_cast<Int>(T.row_idx.size()));
    }
    auto big = make_smf_ipopt_plugin_adapter(ipopt_plugin_ipm_control());
    big->configure(*ipopt_plugin_control(), "ipopt-plugin");
    ASSERT_EQ(big->analyse(T), Outcome::Success);
    EXPECT_FALSE(big->stats().has_analyse_phases);
    const double deferred_ms = big->stats().analyse_ms;
    ASSERT_EQ(big->factor(T.values), Outcome::Success);
    EXPECT_TRUE(big->stats().has_analyse_phases);
    EXPECT_GT(big->stats().analyse_ms, deferred_ms);
    EXPECT_GT(big->stats().analyse_ms, 0.0);
}

TEST(BenchmarkAdapters, Ma97ScalingCallMatchesPlugin) {
    const std::vector<double> unit(4, 1.0), s = {1.0, 2.0, 0.5, 1.0};
    // MC64 request after the analysis rejected the matching: unit scale.
    Control c = *ipopt_plugin_control();
    c.scaling = ScalingMethod::Matching;
    Ma97ScalingCall r = ma97_scaling_call(c, true, nullptr, 4);
    EXPECT_TRUE(r.unit_scale);
    EXPECT_FALSE(r.computed);
    EXPECT_EQ(c.scaling, ScalingMethod::None);
    // MC64 request otherwise: computed.
    c.scaling = ScalingMethod::Matching;
    r = ma97_scaling_call(c, false, s.data(), 4);
    EXPECT_TRUE(r.computed);
    EXPECT_FALSE(r.reuse);
    EXPECT_EQ(c.scaling, ScalingMethod::Matching);
    // No request, a non-unit scale passed back: reuse.
    c.scaling = ScalingMethod::None;
    r = ma97_scaling_call(c, false, s.data(), 4);
    EXPECT_TRUE(r.reuse);
    EXPECT_EQ(c.scaling, ScalingMethod::UserSupplied);
    // A unit scale passed back is no scaling.
    c.scaling = ScalingMethod::None;
    r = ma97_scaling_call(c, false, unit.data(), 4);
    EXPECT_FALSE(r.reuse || r.computed || r.unit_scale);
    EXPECT_EQ(c.scaling, ScalingMethod::None);
    // MC64 request after a rejected matching, but the analysis' unscaled
    // factorizations cascade: honoured.
    c.scaling = ScalingMethod::Matching;
    r = ma97_scaling_call(c, true, unit.data(), 4, true);
    EXPECT_FALSE(r.unit_scale);
    EXPECT_TRUE(r.computed);
    EXPECT_TRUE(r.cascade_honoured);
    EXPECT_EQ(c.scaling, ScalingMethod::Matching);
    // The cascade flag changes nothing else: an accepted matching's request,
    // a reuse call.
    c.scaling = ScalingMethod::Matching;
    r = ma97_scaling_call(c, false, nullptr, 4, true);
    EXPECT_TRUE(r.computed);
    EXPECT_FALSE(r.cascade_honoured);
    c.scaling = ScalingMethod::None;
    r = ma97_scaling_call(c, true, s.data(), 4, true);
    EXPECT_TRUE(r.reuse);
    EXPECT_FALSE(r.computed || r.cascade_honoured || r.unit_scale);
}

TEST(BenchmarkAdapters, Ma97CascadeStateRemembersACascade) {
    Ma97CascadeState st;
    EXPECT_FALSE(st.cascade());
    // Delays that inflate the fronts a few times over are no cascade.
    st.after_factor(8.0e6, 1.0e6, false);
    EXPECT_DOUBLE_EQ(st.last_work_ratio, 8.0);
    EXPECT_FALSE(st.cascade());
    // At kPluginCascadeWorkRatio times the delay-free work they are.
    st.after_factor(kPluginCascadeWorkRatio * 1.0e6, 1.0e6, false);
    EXPECT_TRUE(st.cascade());
    // The answer follows the latest factorization until a request is honoured...
    st.after_factor(1.0e6, 1.0e6, false);
    EXPECT_FALSE(st.cascade());
    // ...and stays once one was: the scaled factorizations no longer cascade.
    st.after_factor(40.0e6, 1.0e6, false);
    st.after_factor(1.0e6, 1.0e6, true);
    EXPECT_TRUE(st.honoured);
    st.after_factor(1.0e6, 1.0e6, false);
    EXPECT_TRUE(st.cascade());
    // No delay-free work (an empty analysis): no ratio.
    Ma97CascadeState empty;
    empty.after_factor(5.0, 0.0, false);
    EXPECT_DOUBLE_EQ(empty.last_work_ratio, 0.0);
    // The delay-free work is computed once per analysis (its serial).
    int computed = 0;
    const auto work = [&computed] { return 100.0 * ++computed; };
    EXPECT_DOUBLE_EQ(empty.free_work_of(7, work), 100.0);
    EXPECT_DOUBLE_EQ(empty.free_work_of(7, work), 100.0);
    EXPECT_DOUBLE_EQ(empty.free_work_of(8, work), 200.0);
    EXPECT_EQ(computed, 2);
}

TEST(BenchmarkAdapters, IpoptDynamicScalingStateMachine) {
    // IPOPT 3.14 defaults (dynamic, od_hd_reuse): n = 100, threshold 5 delays
    // over the last MC64 factor's count.
    const int n = 100;
    const std::vector<double> none, s1(100, 2.0), s2(100, 3.0);
    IpoptDynamicScaling d;
    Ma97ScalingCall plain;                      // unscaled / reuse call: no scale computed
    Ma97ScalingCall computed;
    computed.computed = true;                   // an MC64 call (or a stale-scale retry)
    Ma97ScalingCall reuse;
    reuse.reuse = true;

    // Unscaled factors below the threshold: nothing requested, no scale held.
    d.after_factor(IpmFactorStatus::Ok, plain, none, 5, n);
    EXPECT_FALSE(d.mc64_requested());
    EXPECT_EQ(d.scale(), nullptr);
    // 6 > 0.05 n: MC64 on the next factor.
    d.after_factor(IpmFactorStatus::Ok, plain, none, 6, n);
    EXPECT_TRUE(d.mc64_requested());
    // A singular factor returns before IPOPT's check: the request stays,
    // the scale it computed is still handed back.
    d.after_factor(IpmFactorStatus::Singular, computed, s1, 50, n);
    EXPECT_TRUE(d.mc64_requested());
    ASSERT_NE(d.scale(), nullptr);
    EXPECT_EQ(d.scale()[0], 2.0);
    // A failed call hands nothing back and changes nothing.
    d.after_factor(IpmFactorStatus::Error, computed, s2, 50, n);
    EXPECT_TRUE(d.mc64_requested());
    EXPECT_EQ(d.scale()[0], 2.0);
    // The MC64 factor: scale kept, its delays (40) become numdelay_.
    d.after_factor(IpmFactorStatus::Ok, computed, s2, 40, n);
    EXPECT_FALSE(d.mc64_requested());
    EXPECT_EQ(d.numdelay(), 40);
    EXPECT_EQ(d.scale()[0], 3.0);
    // Reuse while delays stay within numdelay_ + 0.05 n.
    d.after_factor(IpmFactorStatus::Ok, reuse, s2, 45, n);
    EXPECT_FALSE(d.mc64_requested());
    // Delays grow by more than 0.05 n: MC64 again.
    d.after_factor(IpmFactorStatus::Ok, reuse, s2, 46, n);
    EXPECT_TRUE(d.mc64_requested());
    // That MC64 request answered with the unit scale (rejected matching).
    Ma97ScalingCall unit;
    unit.unit_scale = true;
    d.after_factor(IpmFactorStatus::Ok, unit, none, 10, n);
    EXPECT_FALSE(d.mc64_requested());
    EXPECT_EQ(d.numdelay(), 10);
    EXPECT_EQ(d.scale()[0], 1.0);
    // A stale-scale retry (reuse turned computed) replaces the held scale but
    // is not an MC64 request: numdelay_ stays.
    d.after_factor(IpmFactorStatus::Ok, computed, s1, 12, n);
    EXPECT_EQ(d.numdelay(), 10);
    EXPECT_EQ(d.scale()[0], 2.0);
    d.reset();
    EXPECT_FALSE(d.mc64_requested());
    EXPECT_EQ(d.scale(), nullptr);
    EXPECT_EQ(d.numdelay(), 0);
}

TEST(BenchmarkAdapters, ReplayStepPicks) {
    // 5 evenly spaced steps (the default), first to last.
    EXPECT_EQ(replay_step_picks(9, 5), (std::vector<std::size_t>{0, 2, 4, 6, 8}));
    EXPECT_EQ(replay_step_picks(1, 5), (std::vector<std::size_t>{0, 0, 0, 0, 0}));
    // Every stored step.
    EXPECT_EQ(replay_step_picks(4, -1), (std::vector<std::size_t>{0, 1, 2, 3}));
    EXPECT_TRUE(replay_step_picks(0, -1).empty());
    EXPECT_TRUE(replay_step_picks(0, 5).empty());
}

TEST(BenchmarkAdapters, SmfIpmAdapterOffMatchesPlainSmf) {
    auto ipm = make_smf_ipm_adapter(StaticMode::Off, false);
    ASSERT_NE(ipm.get(), nullptr);
    EXPECT_EQ(ipm->identity().solver_id, "smf");
    ipm->configure(replay_control(), "ipopt");
    CscLower E2 = make_exchange_2x2_indef();
    ASSERT_EQ(ipm->analyse(E2), Outcome::Success);
    ASSERT_EQ(ipm->factor(E2.values), Outcome::Success);
    std::vector<double> x = {1.0, 2.0};
    ASSERT_EQ(ipm->solve(x, 1), Outcome::Success);
    EXPECT_NEAR(x[0], 2.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0, 1e-12);
    EXPECT_EQ(ipm->stats().inertia_pos, 1);
    EXPECT_EQ(ipm->stats().inertia_neg, 1);
    EXPECT_EQ(ipm->stats().static_pivots, 0);
}

TEST(BenchmarkAdapters, SmfIpmAdapterRegularizedRaisesTinyPivot) {
    // A primal row with diagonal -rho/2: regularized to +rho/2 < rho, raised
    // to rho (one static pivot). Hybrid accounting keeps its sign; with
    // static-zero it is a zero eigenvalue (singular factor).
    const CscLower T = make_tiny_diag(-0.5e-8);

    auto plain = make_smf_adapter();
    plain->configure(replay_control(), "ipopt");
    ASSERT_EQ(plain->analyse(T), Outcome::Success);
    ASSERT_EQ(plain->factor(T.values), Outcome::Success);
    EXPECT_EQ(plain->stats().static_pivots, 0);
    EXPECT_EQ(plain->stats().inertia_neg, 1);

    auto hybrid = make_smf_ipm_adapter(StaticMode::Regularized, false);
    hybrid->configure(replay_control(), "ipopt");
    ASSERT_EQ(hybrid->analyse(T), Outcome::Success);
    ASSERT_EQ(hybrid->factor(T.values), Outcome::Success);
    EXPECT_EQ(hybrid->stats().static_pivots, 1);
    EXPECT_EQ(hybrid->stats().inertia_pos, 3);
    EXPECT_EQ(hybrid->stats().inertia_neg, 0);
    EXPECT_EQ(hybrid->stats().inertia_zero, 0);

    auto zero = make_smf_ipm_adapter(StaticMode::Regularized, true);
    zero->configure(replay_control(), "ipopt");
    ASSERT_EQ(zero->analyse(T), Outcome::Success);
    EXPECT_EQ(zero->factor(T.values), Outcome::Singular);
    EXPECT_EQ(zero->stats().static_pivots, 1);
    EXPECT_EQ(zero->stats().inertia_zero, 1);
}

TEST(BenchmarkAdapters, SmfIpmAdapterRegularizedSolveIsRefined) {
    // An indefinite KKT-like matrix (zero constraint diagonal): the factor is
    // of the regularized matrix, the solve is refined against the original.
    CscLower K;
    K.n = 3;
    K.col_ptr = {0, 3, 5, 6};
    K.row_idx = {0, 1, 2, 1, 2, 2};
    K.values = {4.0, 1.0, 1.0, 3.0, 1.0, 0.0};
    auto reg = make_smf_ipm_adapter(StaticMode::Regularized, false);
    reg->configure(replay_control(), "ipopt");
    ASSERT_EQ(reg->analyse(K), Outcome::Success);
    for (int f = 0; f < 2; ++f) {
        ASSERT_EQ(reg->factor(K.values), Outcome::Success);
        const std::vector<double> b = {1.0, -2.0, 0.5};
        std::vector<double> x = b;
        ASSERT_EQ(reg->solve(x, 1), Outcome::Success);
        EXPECT_TRUE(is_qualified_validation(validate(K, b, x, 1))) << "factor " << f;
        EXPECT_EQ(reg->stats().inertia_neg, 1);
    }
}

TEST(BenchmarkAdapters, RecordCarriesFirstFactorAndRefactorOnlyWhenMeasured) {
    RunRecord rec;
    EXPECT_EQ(rec.to_json_line().find("first_factor_ms"), std::string::npos);
    EXPECT_EQ(rec.to_json_line().find("\"refactor_ms\""), std::string::npos);
    rec.first_factor_ms = 12.5;
    rec.refactor_ms = 3.25;
    const std::string line = rec.to_json_line();
    EXPECT_NE(line.find("\"first_factor_ms\":12.5"), std::string::npos) << line;
    EXPECT_NE(line.find("\"refactor_ms\":3.25"), std::string::npos) << line;
}

// R29: the cold path on the ipopt track refines like warm_sequence does. A
// solver whose solve carries a 1e-6 relative error misses the accuracy gate
// unrefined; harness refinement (ipopt track only, nrhs == 1) recovers it.
TEST(BenchmarkAdapters, ColdIpoptTrackRefinesRecommendedDoesNot) {
    CscLower A;
    const int n = 200;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(4.0);
        if (j + 1 < n) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(-1.0);
        }
        A.col_ptr[static_cast<std::size_t>(j) + 1] = static_cast<Int>(A.row_idx.size());
    }
    const std::vector<double> b(static_cast<std::size_t>(n), 1.0);
    // Exact tridiagonal solve (Thomas) times (1 + 1e-6): an inaccurate "factorization".
    auto inexact_solve = [&](std::vector<double> &r) {
        std::vector<double> c(n), d(r);
        double m = 4.0;
        c[0] = -1.0 / m;
        d[0] /= m;
        for (int i = 1; i < n; ++i) {
            m = 4.0 + c[i - 1];
            c[i] = -1.0 / m;
            d[i] = (d[i] + d[i - 1]) / m;
        }
        for (int i = n - 2; i >= 0; --i) d[i] -= c[i] * d[i + 1];
        for (int i = 0; i < n; ++i) r[i] = d[i] * (1.0 + 1e-6);
        return true;
    };

    std::vector<double> x0 = b;
    inexact_solve(x0);
    ASSERT_FALSE(is_qualified_validation(validate(A, b, x0, 1)));

    std::vector<double> xi = x0;
    const double ms = refine_timed("ipopt", 1, inexact_solve, A, A.values, b, xi, 3);
    EXPECT_GT(ms, 0.0);
    EXPECT_TRUE(is_qualified_validation(validate(A, b, xi, 1)));

    std::vector<double> xr = x0;
    EXPECT_EQ(refine_timed("recommended", 1, inexact_solve, A, A.values, b, xr, 3), 0.0);
    EXPECT_EQ(xr, x0);
    std::vector<double> xm = x0;
    EXPECT_EQ(refine_timed("ipopt", 2, inexact_solve, A, A.values, b, xm, 3), 0.0);
    EXPECT_EQ(xm, x0);
}

// ---------------------------------------------------------------------------
// cuDSS (GPU reference solver), built only with -DSMF_BENCH_CUDSS=ON.

#if defined(SMF_BENCH_CUDSS)

namespace {

// 5-point 2D Poisson on an m x m grid, lower triangle in CSC.
CscLower make_poisson2d(int m) {
    CscLower A;
    A.n = m * m;
    A.col_ptr.assign(static_cast<std::size_t>(A.n) + 1, 0);
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            const int col = j * m + i;
            A.row_idx.push_back(col);
            A.values.push_back(4.0);
            if (i + 1 < m) {
                A.row_idx.push_back(col + 1);
                A.values.push_back(-1.0);
            }
            if (j + 1 < m) {
                A.row_idx.push_back(col + m);
                A.values.push_back(-1.0);
            }
            A.col_ptr[static_cast<std::size_t>(col) + 1] =
                static_cast<Int>(A.row_idx.size());
        }
    }
    return A;
}

} // namespace

TEST(BenchmarkAdapters, CudssSolvesPoisson) {
    auto adapter = make_solver_adapter("cudss");
    ASSERT_NE(adapter.get(), nullptr) << "cudss is not a registered adapter";
    const AdapterIdentity ident = adapter->identity();
    EXPECT_EQ(ident.solver_id, "cudss");
    if (!ident.available) GTEST_SKIP() << "no usable CUDA device: " << adapter->stats().reason;

    const CscLower A = make_poisson2d(30);
    std::vector<double> b(static_cast<std::size_t>(A.n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = 1.0 + 0.001 * static_cast<double>(i % 17);

    // Both matrix types: SPD (Cholesky) and symmetric indefinite (LDL^T).
    for (MatrixType mt : {MatrixType::RealSymmetricPositiveDefinite,
                          MatrixType::RealSymmetricIndefinite}) {
        SCOPED_TRACE(mt == MatrixType::RealSymmetricIndefinite ? "ldlt" : "spd");
        Control ctrl;
        ctrl.matrix_type = mt;
        adapter->configure(ctrl, "recommended");
        ASSERT_EQ(adapter->analyse(A), Outcome::Success) << adapter->stats().reason;
        ASSERT_EQ(adapter->factor(A.values), Outcome::Success) << adapter->stats().reason;
        std::vector<double> x = b;
        ASSERT_EQ(adapter->solve(x, 1), Outcome::Success) << adapter->stats().reason;
        const Validation v = validate(A, b, x, 1);
        EXPECT_LE(v.eta_inf, 1e-12);
        EXPECT_TRUE(is_qualified_validation(v));
        EXPECT_GT(adapter->stats().factor_ms, 0.0);

        // Refactor with changed values on the same analysis (diag scaled by 1.5).
        CscLower A2 = A;
        for (Int j = 0; j < A2.n; ++j) {
            A2.values[static_cast<std::size_t>(A2.col_ptr[static_cast<std::size_t>(j)])] *= 1.5;
        }
        ASSERT_EQ(adapter->factor(A2.values), Outcome::Success);
        std::vector<double> x2 = b;
        ASSERT_EQ(adapter->solve(x2, 1), Outcome::Success);
        EXPECT_LE(validate(A2, b, x2, 1).eta_inf, 1e-12);

        // ipopt track, one rhs: cuDSS iterative refinement (3 steps) runs inside solve.
        adapter->configure(ctrl, "ipopt");
        ASSERT_EQ(adapter->analyse(A), Outcome::Success);
        ASSERT_EQ(adapter->factor(A.values), Outcome::Success);
        std::vector<double> x3 = b;
        ASSERT_EQ(adapter->solve(x3, 1), Outcome::Success);
        EXPECT_LE(validate(A, b, x3, 1).eta_inf, 1e-12);
    }
}

#endif // SMF_BENCH_CUDSS
