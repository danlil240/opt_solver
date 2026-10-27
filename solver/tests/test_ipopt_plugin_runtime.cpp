#include <gtest/gtest.h>

#include "smf/csc_matrix.hpp"
#include "smf/matrix_market.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef small
#undef small
#endif
#else
#include <dlfcn.h>
#endif

struct ma97_control_d {
    int f_arrays;
    int action;
    int nemin;
    double multiplier;
    int ordering;
    int print_level;
    int scaling;
    double small;
    double u;
    int unit_diagnostics;
    int unit_error;
    int unit_warning;
    long factor_min;
    int solve_blas3;
    int solve_min;
    int solve_mf;
    double consist_tol;
    int ispare[5];
    double rspare[10];
};

struct ma97_info_d {
    int flag;
    int flag68;
    int flag77;
    int matrix_dup;
    int matrix_rank;
    int matrix_outrange;
    int matrix_missing_diag;
    int maxdepth;
    int maxfront;
    int num_delay;
    long num_factor;
    long num_flops;
    int num_neg;
    int num_sup;
    int num_two;
    int ordering;
    int stat;
    int maxsupernode;
    int ispare[4];
    double rspare[10];
};

namespace {

struct PluginSymbols {
#ifdef _WIN32
    HMODULE handle = nullptr;
#else
    void *handle = nullptr;
#endif
    void (*default_control)(ma97_control_d *) = nullptr;
    void (*analyse)(int, int, const int *, const int *, double *, void **,
                    const ma97_control_d *, ma97_info_d *, int *) = nullptr;
    void (*factor)(int, const int *, const int *, const double *, void **,
                   void **, const ma97_control_d *, ma97_info_d *, double *) = nullptr;
    void (*factor_solve)(int, const int *, const int *, const double *, int,
                         double *, int, void **, void **, const ma97_control_d *,
                         ma97_info_d *, double *) = nullptr;
    void (*solve)(int, int, double *, int, void **, void **,
                  const ma97_control_d *, ma97_info_d *) = nullptr;
    void (*free_akeep)(void **) = nullptr;
    void (*free_fkeep)(void **) = nullptr;
    void (*finalise)(void **, void **) = nullptr;

    bool load() {
#ifdef SMF_MA97_PLUGIN_PATH
        const char *path = SMF_MA97_PLUGIN_PATH;
#elif defined(_WIN32)
        const char *path = "smf_ma97.dll";
#else
        const char *path = "./libsmf_ma97.so";
#endif
#ifdef _WIN32
        handle = LoadLibraryA(path);
        if (!handle) return false;
        default_control = reinterpret_cast<decltype(default_control)>(
            GetProcAddress(handle, "ma97_default_control_d"));
        analyse = reinterpret_cast<decltype(analyse)>(
            GetProcAddress(handle, "ma97_analyse_d"));
        factor = reinterpret_cast<decltype(factor)>(
            GetProcAddress(handle, "ma97_factor_d"));
        factor_solve = reinterpret_cast<decltype(factor_solve)>(
            GetProcAddress(handle, "ma97_factor_solve_d"));
        solve = reinterpret_cast<decltype(solve)>(
            GetProcAddress(handle, "ma97_solve_d"));
        free_akeep = reinterpret_cast<decltype(free_akeep)>(
            GetProcAddress(handle, "ma97_free_akeep_d"));
        free_fkeep = reinterpret_cast<decltype(free_fkeep)>(
            GetProcAddress(handle, "ma97_free_fkeep_d"));
        finalise = reinterpret_cast<decltype(finalise)>(
            GetProcAddress(handle, "ma97_finalise_d"));
#else
        handle = dlopen(path, RTLD_NOW);
        if (!handle) return false;
        default_control = reinterpret_cast<decltype(default_control)>(
            dlsym(handle, "ma97_default_control_d"));
        analyse = reinterpret_cast<decltype(analyse)>(
            dlsym(handle, "ma97_analyse_d"));
        factor = reinterpret_cast<decltype(factor)>(
            dlsym(handle, "ma97_factor_d"));
        factor_solve = reinterpret_cast<decltype(factor_solve)>(
            dlsym(handle, "ma97_factor_solve_d"));
        solve = reinterpret_cast<decltype(solve)>(
            dlsym(handle, "ma97_solve_d"));
        free_akeep = reinterpret_cast<decltype(free_akeep)>(
            dlsym(handle, "ma97_free_akeep_d"));
        free_fkeep = reinterpret_cast<decltype(free_fkeep)>(
            dlsym(handle, "ma97_free_fkeep_d"));
        finalise = reinterpret_cast<decltype(finalise)>(
            dlsym(handle, "ma97_finalise_d"));
#endif
        return default_control && analyse && factor && factor_solve &&
               solve && free_akeep && free_fkeep && finalise;
    }

    ~PluginSymbols() {
#ifdef _WIN32
        if (handle) FreeLibrary(handle);
#else
        if (handle) dlclose(handle);
#endif
    }
};

} // namespace

TEST(Ma97PluginRuntime, ChangedValuesAndInertia) {
    PluginSymbols sym;
    ASSERT_TRUE(sym.load()) << "Failed to load shared plugin library or symbols (including ma97_free_fkeep_d)";

    ma97_control_d ctrl{};
    sym.default_control(&ctrl);
    ctrl.f_arrays = 1; // 1-based Fortran indexing (as IPOPT uses)

    // 3x3 symmetric matrix with duplicate entry in col 1:
    // col 1: (1,1), (1,1 dup), (2,1), (3,1) -> 4 entries
    // col 2: (2,2), (3,2) -> 2 entries
    // col 3: (3,3) -> 1 entry
    const int n = 3;
    const int ptr[4] = {1, 5, 7, 8};
    const int row[7] = {1, 1, 2, 3, 2, 3, 3};

    void *akeep = nullptr;
    void *fkeep = nullptr;
    ma97_info_d info{};

    sym.analyse(1, n, ptr, row, nullptr, &akeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    ASSERT_NE(akeep, nullptr);

    // Update 1: diag = [2+2=4, -3, 5], off-diag = 1 -> 2 positive, 1 negative
    const double val1[7] = {2.0, 2.0, 1.0, 0.5, -3.0, 1.0, 5.0};
    sym.factor(4, ptr, row, val1, &akeep, &fkeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    EXPECT_EQ(info.num_neg, 1);
    EXPECT_EQ(info.num_two >= 0, true);

    double rhs1[3] = {5.5, -1.0, 6.5}; // x = [1, 1, 1]
    sym.solve(0, 1, rhs1, 3, &akeep, &fkeep, &ctrl, &info);
    ASSERT_EQ(info.flag, 0);
    EXPECT_NEAR(rhs1[0], 1.0, 1e-11);
    EXPECT_NEAR(rhs1[1], 1.0, 1e-11);
    EXPECT_NEAR(rhs1[2], 1.0, 1e-11);

    // Update 2: flip sign of (3,3) to -5 -> 1 positive, 2 negative
    const double val2[7] = {2.0, 2.0, 1.0, 0.5, -3.0, 1.0, -5.0};
    sym.factor(4, ptr, row, val2, &akeep, &fkeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    EXPECT_EQ(info.num_neg, 2);

    sym.finalise(&akeep, &fkeep);
    EXPECT_EQ(akeep, nullptr);
    EXPECT_EQ(fkeep, nullptr);
}

TEST(Ma97PluginRuntime, ScalingComputedThenReused) {
    // IPOPT's "reuse" scaling: factor once with scaling = 1 (MC64), which
    // returns the scale vector, then refactor with scaling = 0 passing that
    // vector back, which MA97 applies as user-supplied scaling.
    PluginSymbols sym;
    ASSERT_TRUE(sym.load());
    ma97_control_d ctrl{};
    sym.default_control(&ctrl);
    ctrl.f_arrays = 0;

    // [[4e6, 2e3, 0], [2e3, 0, 1e-3], [0, 1e-3, -1e-5]] (det 36), x = [1, 1, 1]
    const int n = 3;
    const int ptr[4] = {0, 2, 4, 5};
    const int row[5] = {0, 1, 1, 2, 2};
    const double val[5] = {4e6, 2e3, 0.0, 1e-3, -1e-5};
    const double rhs[3] = {4e6 + 2e3, 2e3 + 1e-3, 1e-3 - 1e-5};

    void *akeep = nullptr;
    void *fkeep = nullptr;
    ma97_info_d info{};
    sym.analyse(1, n, ptr, row, nullptr, &akeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);

    double scale[3] = {0.0, 0.0, 0.0};
    ctrl.scaling = 1;
    sym.factor(4, ptr, row, val, &akeep, &fkeep, &ctrl, &info, scale);
    ASSERT_EQ(info.flag, 0);
    for (double s : scale) EXPECT_GT(s, 0.0);

    const auto solve_and_check = [&](const char *what) {
        double x[3] = {rhs[0], rhs[1], rhs[2]};
        sym.solve(0, 1, x, n, &akeep, &fkeep, &ctrl, &info);
        ASSERT_EQ(info.flag, 0) << what;
        for (double xi : x) EXPECT_NEAR(xi, 1.0, 1e-8) << what;
    };
    solve_and_check("mc64");

    // Reuse a (perturbed) user scale: applied as given, not overwritten (a
    // reused scale is only replaced when it makes the factor singular, see
    // StaleReusedScaleIsRefreshedOnSpuriousSingularity).
    ctrl.scaling = 0;
    scale[1] *= 2.0;
    const double kept = scale[1];
    sym.factor(4, ptr, row, val, &akeep, &fkeep, &ctrl, &info, scale);
    ASSERT_EQ(info.flag, 0);
    EXPECT_EQ(scale[1], kept);
    solve_and_check("reuse");

    // The reused vector is really read: a zero factor is rejected.
    double zero_scale[3] = {scale[0], 0.0, scale[2]};
    sym.factor(4, ptr, row, val, &akeep, &fkeep, &ctrl, &info, zero_scale);
    EXPECT_LT(info.flag, 0) << "flag=" << info.flag;

    // scaling = 0 without a vector is unscaled.
    sym.factor(4, ptr, row, val, &akeep, &fkeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    solve_and_check("none");

    sym.finalise(&akeep, &fkeep);
}

TEST(Ma97PluginRuntime, MultipleRhsLeadingDimension) {
    PluginSymbols sym;
    ASSERT_TRUE(sym.load());

    ma97_control_d ctrl{};
    sym.default_control(&ctrl);
    ctrl.f_arrays = 0; // 0-based C indexing

    // 2x2 exchange matrix [[0, 1], [1, 0]]
    const int n = 2;
    const int ptr[3] = {0, 2, 3};
    const int row[3] = {0, 1, 1};
    const double val[3] = {0.0, 1.0, 0.0};

    void *akeep = nullptr;
    void *fkeep = nullptr;
    ma97_info_d info{};

    sym.analyse(1, n, ptr, row, nullptr, &akeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    sym.factor(4, ptr, row, val, &akeep, &fkeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    EXPECT_EQ(info.num_neg, 1);
    EXPECT_EQ(info.num_two, 1);

    // nrhs = 2 with padded leading dimension ldx = 4 > n = 2!
    // Sentinel values at rows 2 and 3 must NOT be overwritten!
    std::vector<double> x = {
        1.0, 2.0, 99.0, 88.0,
        -3.0, 5.0, 77.0, 66.0
    };
    sym.solve(0, 2, x.data(), 4, &akeep, &fkeep, &ctrl, &info);
    ASSERT_EQ(info.flag, 0);
    EXPECT_NEAR(x[0], 2.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(x[2], 99.0);
    EXPECT_DOUBLE_EQ(x[3], 88.0);
    EXPECT_NEAR(x[4], 5.0, 1e-12);
    EXPECT_NEAR(x[5], -3.0, 1e-12);
    EXPECT_DOUBLE_EQ(x[6], 77.0);
    EXPECT_DOUBLE_EQ(x[7], 66.0);

    // Invalid ldx < n must set negative flag!
    sym.solve(0, 1, x.data(), 1, &akeep, &fkeep, &ctrl, &info);
    EXPECT_LT(info.flag, 0);

    sym.finalise(&akeep, &fkeep);
}

TEST(Ma97PluginRuntime, QualityRetryAndTeardown) {
    PluginSymbols sym;
    ASSERT_TRUE(sym.load());

    ma97_control_d ctrl{};
    sym.default_control(&ctrl);
    ctrl.f_arrays = 0;
    ctrl.scaling = 1; // Matching scaling
    ctrl.u = 1e-8;

    // 3x3 matrix where first pivot has tiny diagonal 1e-7 and large coupling 10.0:
    // Retry with larger u = 0.1 triggers pivot delay/2x2 and improves quality!
    const int n = 3;
    const int ptr[4] = {0, 2, 4, 5};
    const int row[5] = {0, 2, 1, 2, 2};
    const double val[5] = {1e-7, 10.0, 4.0, 1.0, 2.0};

    void *akeep = nullptr;
    void *fkeep = nullptr;
    ma97_info_d info{};

    sym.analyse(1, n, ptr, row, nullptr, &akeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);

    ctrl.u = 0.1; // quality retry escalation
    double x[3] = {10.0, 5.0, 13.0};
    sym.factor_solve(4, ptr, row, val, 1, x, 3, &akeep, &fkeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);

    // Free akeep first while keeping fkeep alive, then solve with fkeep, then free fkeep!
    sym.free_akeep(&akeep);
    EXPECT_EQ(akeep, nullptr);
    double x2[3] = {10.0, 5.0, 13.0};
    sym.solve(0, 1, x2, 3, &akeep, &fkeep, &ctrl, &info);
    EXPECT_EQ(info.flag, 0);
    EXPECT_NEAR(x2[0], x[0], 1e-12);

    sym.free_fkeep(&fkeep);
    EXPECT_EQ(fkeep, nullptr);
}

namespace {

// Lower-CSC KKT of test_ipm_session.cpp: [[H, J^T], [J, 0]], 40 primal and
// 15 constraint rows, H = diag(-1, -1, -1, 2, ...); inertia (37, 18, 0).
// With `bad`, H_jj = -20 for j = 3, 4, 5: inertia (34, 21, 0).
struct SmallKkt {
    int n = 55;
    std::vector<int> ptr, row;
    std::vector<double> val;
    explicit SmallKkt(bool bad) {
        std::vector<std::vector<std::pair<int, double>>> cols(55);
        for (int j = 0; j < 40; ++j)
            cols[j].push_back({j, j < 3 ? -1.0 : (bad && j < 6) ? -20.0 : 2.0 + 0.01 * j});
        for (int i = 0; i < 15; ++i) {
            cols[3 + i].push_back({40 + i, 1.0 + 0.1 * i});
            cols[18 + i].push_back({40 + i, 0.5});
            cols[33 + i % 7].push_back({40 + i, 0.25});
        }
        ptr.push_back(0);
        for (auto &c : cols) {
            std::sort(c.begin(), c.end());
            for (const auto &[i, x] : c) {
                row.push_back(i);
                val.push_back(x);
            }
            ptr.push_back(static_cast<int>(row.size()));
        }
    }
    // A * ones (symmetric, from the lower triangle).
    std::vector<double> rhs_of_ones() const {
        std::vector<double> b(n, 0.0);
        for (int j = 0; j < n; ++j)
            for (int p = ptr[j]; p < ptr[j + 1]; ++p) {
                b[row[p]] += val[p];
                if (row[p] != j) b[j] += val[p];
            }
        return b;
    }
};

void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value) setenv(name, value, 1); else unsetenv(name);
#endif
}

} // namespace

// SMF_IPM_ABORT=1: once a factor has been solved with, a later one with more
// negative pivots stops early. MA97 has no status for that: flag 0 and
// num_neg = the partial count (> the learned 18), which IPOPT rejects. A
// scale computed by that factor is still returned, and a solve with its
// fkeep completes the factorization first.
TEST(Ma97PluginRuntime, EarlyAbortOnWrongInertia) {
    // The plugin reads SMF_IPM_ABORT once, at its first analyse after it is
    // loaded (the earlier tests unload it again).
    set_env("SMF_IPM_ABORT", "1");
    PluginSymbols sym;
    ASSERT_TRUE(sym.load());

    const SmallKkt good(false), bad(true);
    const int n = good.n;
    ma97_control_d ctrl{};
    sym.default_control(&ctrl);
    ctrl.f_arrays = 0;
    void *akeep = nullptr;
    void *fkeep = nullptr;
    ma97_info_d info{};
    sym.analyse(1, n, good.ptr.data(), good.row.data(), nullptr, &akeep, &ctrl, &info, nullptr);
    set_env("SMF_IPM_ABORT", "");
    ASSERT_EQ(info.flag, 0);

    // No expectation yet: the bad matrix factors in full.
    sym.factor(4, bad.ptr.data(), bad.row.data(), bad.val.data(), &akeep, &fkeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    EXPECT_EQ(info.num_neg, 21);

    // Accepted (solved with): 18 is learned.
    sym.factor(4, good.ptr.data(), good.row.data(), good.val.data(), &akeep, &fkeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    ASSERT_EQ(info.num_neg, 18);
    std::vector<double> x = good.rhs_of_ones();
    sym.solve(0, 1, x.data(), n, &akeep, &fkeep, &ctrl, &info);
    ASSERT_EQ(info.flag, 0);

    // Wrong inertia, in a factor that also computes the scaling.
    std::vector<double> scale(n, -1.0);
    ctrl.scaling = 1;
    sym.factor(4, bad.ptr.data(), bad.row.data(), bad.val.data(), &akeep, &fkeep, &ctrl, &info, scale.data());
    ASSERT_EQ(info.flag, 0);
    EXPECT_GT(info.num_neg, 18);
    EXPECT_LE(info.num_neg, 21);
    for (double s : scale) EXPECT_GT(s, 0.0);

    // Solving with the aborted factor completes it: the bad system's answer.
    ctrl.scaling = 0;
    x = bad.rhs_of_ones();
    sym.solve(0, 1, x.data(), n, &akeep, &fkeep, &ctrl, &info);
    ASSERT_EQ(info.flag, 0);
    EXPECT_EQ(info.num_neg, 21);
    for (double xi : x) EXPECT_NEAR(xi, 1.0, 1e-9);

    // The returned scale is reusable (IPOPT's next factor passes it back).
    sym.factor(4, good.ptr.data(), good.row.data(), good.val.data(), &akeep, &fkeep, &ctrl, &info, scale.data());
    ASSERT_EQ(info.flag, 0);
    EXPECT_LE(info.num_neg, 21); // expectation is now 21: no abort
    x = good.rhs_of_ones();
    sym.solve(0, 1, x.data(), n, &akeep, &fkeep, &ctrl, &info);
    ASSERT_EQ(info.flag, 0);
    for (double xi : x) EXPECT_NEAR(xi, 1.0, 1e-9);

    sym.finalise(&akeep, &fkeep);
}

// Two fkeeps factored from one akeep share the session's factor: a solve
// with the older fkeep, whose factor has been replaced, fails with MA97's
// "error in the sequence of calls" (flag -1) instead of solving with the
// other matrix. Refactoring into the same fkeep (IPOPT's pattern) is unchanged.
TEST(Ma97PluginRuntime, StaleFkeepFromSameAkeepIsRefused) {
    PluginSymbols sym;
    ASSERT_TRUE(sym.load());
    ma97_control_d ctrl{};
    sym.default_control(&ctrl);
    ctrl.f_arrays = 1;

    // The 3x3 matrices of ChangedValuesAndInertia: A1 = [[4,1,.5],[1,-3,1],
    // [.5,1,5]], A2 the same with (3,3) = -5. x = [1, 1, 1] for both.
    const int n = 3;
    const int ptr[4] = {1, 5, 7, 8};
    const int row[7] = {1, 1, 2, 3, 2, 3, 3};
    const double val1[7] = {2.0, 2.0, 1.0, 0.5, -3.0, 1.0, 5.0};
    const double val2[7] = {2.0, 2.0, 1.0, 0.5, -3.0, 1.0, -5.0};
    const auto rhs1 = [] { return std::vector<double>{5.5, -1.0, 6.5}; };
    const auto rhs2 = [] { return std::vector<double>{5.5, -1.0, -3.5}; };

    void *akeep = nullptr;
    void *fkeep1 = nullptr;
    void *fkeep2 = nullptr;
    ma97_info_d info{};
    sym.analyse(1, n, ptr, row, nullptr, &akeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);

    sym.factor(4, ptr, row, val1, &akeep, &fkeep1, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    ASSERT_EQ(info.num_neg, 1);
    sym.factor(4, ptr, row, val2, &akeep, &fkeep2, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    ASSERT_EQ(info.num_neg, 2);
    ASSERT_NE(fkeep1, fkeep2);

    // fkeep1's factor was replaced by A2's: refused, x untouched.
    std::vector<double> x = rhs1();
    sym.solve(0, 1, x.data(), n, &akeep, &fkeep1, &ctrl, &info);
    EXPECT_EQ(info.flag, -1);
    EXPECT_EQ(x, rhs1());

    // fkeep2 holds the current factor.
    x = rhs2();
    sym.solve(0, 1, x.data(), n, &akeep, &fkeep2, &ctrl, &info);
    ASSERT_EQ(info.flag, 0);
    EXPECT_EQ(info.num_neg, 2);
    for (double xi : x) EXPECT_NEAR(xi, 1.0, 1e-11);

    // Refactoring into fkeep1 makes it current again and fkeep2 stale.
    sym.factor(4, ptr, row, val1, &akeep, &fkeep1, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);
    x = rhs1();
    sym.solve(0, 1, x.data(), n, &akeep, &fkeep1, &ctrl, &info);
    ASSERT_EQ(info.flag, 0);
    EXPECT_EQ(info.num_neg, 1);
    for (double xi : x) EXPECT_NEAR(xi, 1.0, 1e-11);
    x = rhs2();
    sym.solve(0, 1, x.data(), n, &akeep, &fkeep2, &ctrl, &info);
    EXPECT_EQ(info.flag, -1);

    // IPOPT's pattern: one fkeep, refactored and solved repeatedly.
    for (int k = 0; k < 3; ++k) {
        const bool second = (k % 2) == 1;
        sym.factor(4, ptr, row, second ? val2 : val1, &akeep, &fkeep1, &ctrl, &info, nullptr);
        ASSERT_EQ(info.flag, 0);
        x = second ? rhs2() : rhs1();
        sym.solve(0, 1, x.data(), n, &akeep, &fkeep1, &ctrl, &info);
        ASSERT_EQ(info.flag, 0) << "k=" << k;
        for (double xi : x) EXPECT_NEAR(xi, 1.0, 1e-11);
    }

    sym.free_fkeep(&fkeep2);
    sym.finalise(&akeep, &fkeep1);
    EXPECT_EQ(fkeep2, nullptr);
    EXPECT_EQ(fkeep1, nullptr);
}

TEST(Ma97PluginRuntime, StaleReusedScaleIsRefreshedOnSpuriousSingularity) {
    // IPOPT's reuse strategy passes back a scale computed at an earlier
    // iterate. One computed on a blown-up iterate (cops_chain_800: x-scales
    // 1e-14) shrinks the x-pivots of a later, nonsingular KKT below `small`:
    // the factor reports zero pivots, flag 7, and IPOPT never rescales after
    // a singular factor. The plugin refactors once with a fresh matching
    // scaling and hands that scale back.
    PluginSymbols sym;
    ASSERT_TRUE(sym.load());
    ma97_control_d ctrl{};
    sym.default_control(&ctrl);
    ctrl.f_arrays = 0;
    ctrl.u = 1e-8;
    ctrl.small = 1e-20;
    ctrl.scaling = 0;

    // [delta_w I, J^T; J, -delta_c] with J = [1, -1]: inertia (2, 1, 0).
    const int n = 3;
    const int ptr[4] = {0, 2, 4, 5};
    const int row[5] = {0, 2, 1, 2, 2};
    const double val[5] = {1e7, 1.0, 1e7, -1.0, -2e-7};
    void *akeep = nullptr;
    void *fkeep = nullptr;
    ma97_info_d info{};
    sym.analyse(1, n, ptr, row, nullptr, &akeep, &ctrl, &info, nullptr);
    ASSERT_EQ(info.flag, 0);

    double scale[3] = {1e-14, 1e-14, 5e4};
    sym.factor(4, ptr, row, val, &akeep, &fkeep, &ctrl, &info, scale);
    EXPECT_EQ(info.flag, 0);
    EXPECT_EQ(info.num_neg, 1);
    EXPECT_EQ(info.matrix_rank, 3);
    EXPECT_NE(scale[0], 1e-14);
    for (double s : scale) EXPECT_TRUE(std::isfinite(s) && s > 0.0) << s;

    const double b[3] = {1.0, 2.0, 3.0};
    double x[3] = {b[0], b[1], b[2]};
    sym.solve(0, 1, x, n, &akeep, &fkeep, &ctrl, &info);
    ASSERT_EQ(info.flag, 0);
    const double r[3] = {b[0] - (1e7 * x[0] + x[2]), b[1] - (1e7 * x[1] - x[2]),
                         b[2] - (x[0] - x[1] - 2e-7 * x[2])};
    for (double ri : r) EXPECT_LT(std::abs(ri), 1e-12 * 1e7 * (std::abs(x[0]) + std::abs(x[1]) + std::abs(x[2])));

    // The refreshed scale is reused as is: nonsingular again, not replaced.
    const double refreshed[3] = {scale[0], scale[1], scale[2]};
    sym.factor(4, ptr, row, val, &akeep, &fkeep, &ctrl, &info, scale);
    EXPECT_EQ(info.flag, 0);
    for (int i = 0; i < n; ++i) EXPECT_EQ(scale[i], refreshed[i]);

    // A genuinely singular KKT (W = 0, delta_c = 0, rank 2) stays singular.
    const double sing[5] = {0.0, 1.0, 0.0, -1.0, 0.0};
    double stale[3] = {1e-14, 1e-14, 5e4};
    sym.factor(4, ptr, row, sing, &akeep, &fkeep, &ctrl, &info, stale);
    EXPECT_EQ(info.flag, 7);
    EXPECT_EQ(info.matrix_rank, 2);

    sym.finalise(&akeep, &fkeep);
}

namespace {

// A KKT matrix [H J^T; J 0] in 0-based lower CSC: H tridiagonal (diagonal
// 2 + 0.01 (i mod 7), off-diagonal -0.5) on nx primal rows, and m
// constraints, constraint k coupling x[2k] (1), x[2k + 1] (-1) and
// x[(2k + 37) mod nx] (0.25). Constraint rows have no diagonal entry.
struct ChainKkt {
    int n = 0;
    int m = 0;
    std::vector<int> ptr, row;
    std::vector<double> val;
    ChainKkt(int nx, int nc) : n(nx + nc), m(nc) {
        std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n));
        for (int i = 0; i < nx; ++i) {
            cols[i].push_back({i, 2.0 + 0.01 * (i % 7)});
            if (i + 1 < nx) cols[i].push_back({i + 1, -0.5});
        }
        for (int k = 0; k < nc; ++k) {
            const int c = nx + k;
            cols[2 * k].push_back({c, 1.0});
            cols[2 * k + 1].push_back({c, -1.0});
            const int away = (2 * k + 37) % nx;
            if (away != 2 * k && away != 2 * k + 1) cols[away].push_back({c, 0.25});
        }
        ptr.push_back(0);
        for (auto &col : cols) {
            std::sort(col.begin(), col.end());
            for (const auto &[i, x] : col) {
                row.push_back(i);
                val.push_back(x);
            }
            ptr.push_back(static_cast<int>(row.size()));
        }
    }
    std::vector<double> rhs_of_ones() const {
        std::vector<double> b(static_cast<std::size_t>(n), 0.0);
        for (int j = 0; j < n; ++j)
            for (int p = ptr[j]; p < ptr[j + 1]; ++p) {
                b[row[p]] += val[p];
                if (row[p] != j) b[j] += val[p];
            }
        return b;
    }
};

// IPOPT's dynamic MA97 scaling on `kkt`: an unscaled factorization (which
// runs the deferred analysis), an MC64 request (scaling = 1, returned in
// `scale`), then the returned scale reused (scaling = 0). Every solve of
// A x = A 1 must give x = 1 and every factor m negative pivots. Returns the
// scale and the analysis' ordering (MA97 info.ordering).
template <class Kkt>
std::vector<double> mc64_request_sequence(PluginSymbols &sym, const Kkt &kkt, int expect_neg,
                                          int &ordering) {
    const int n = kkt.n;
    ma97_control_d ctrl{};
    sym.default_control(&ctrl);
    ctrl.f_arrays = 0;
    void *akeep = nullptr;
    void *fkeep = nullptr;
    ma97_info_d info{};
    sym.analyse(1, n, kkt.ptr.data(), kkt.row.data(), nullptr, &akeep, &ctrl, &info, nullptr);
    EXPECT_EQ(info.flag, 0);
    const auto solve_ones = [&](const char *what) {
        std::vector<double> x = kkt.rhs_of_ones();
        sym.solve(0, 1, x.data(), n, &akeep, &fkeep, &ctrl, &info);
        EXPECT_EQ(info.flag, 0) << what;
        double err = 0.0;
        for (double xi : x) err = std::max(err, std::abs(xi - 1.0));
        EXPECT_LT(err, 1e-9) << what;
    };
    std::vector<double> scale(static_cast<std::size_t>(n), -3.0);
    for (int call = 0; call < 3; ++call) {
        ctrl.scaling = call == 1 ? 1 : 0;
        sym.factor(4, kkt.ptr.data(), kkt.row.data(), kkt.val.data(), &akeep, &fkeep, &ctrl, &info,
                   call == 0 ? nullptr : scale.data());
        EXPECT_EQ(info.flag, 0) << "call " << call;
        EXPECT_EQ(info.num_neg, expect_neg) << "call " << call;
        ordering = info.ordering;
        ctrl.scaling = 0;
        solve_ones(call == 0 ? "none" : call == 1 ? "mc64 request" : "reuse");
    }
    sym.finalise(&akeep, &fkeep);
    return scale;
}

} // namespace

// IPOPT asks for MC64 scaling (scaling = 1) after the pattern was analysed.
// When the analysis computed the matching ordering and kept AMD (n above the
// auto matching's minimum, zero-diagonal constraint rows), the plugin
// follows ScalingMethod::Auto: the request is answered with the unit scale,
// which IPOPT then passes back for reuse. A pattern the analysis did not try
// the matching on (n below that minimum) still gets the matching scaling.
TEST(Ma97PluginRuntime, Mc64RequestAfterRejectedMatchingReturnsUnitScale) {
    PluginSymbols sym;
    ASSERT_TRUE(sym.load());

    int ordering = -1;
    const ChainKkt large(9000, 4500);
    const std::vector<double> unit = mc64_request_sequence(sym, large, large.m, ordering);
    ASSERT_NE(ordering, 5) << "the matching ordering was adopted: no rejection to test";
    for (double s : unit) ASSERT_EQ(s, 1.0);

    const SmallKkt small(false);
    const std::vector<double> mc64 = mc64_request_sequence(sym, small, 18, ordering);
    EXPECT_FALSE(std::all_of(mc64.begin(), mc64.end(), [](double s) { return s == 1.0; }));
    for (double s : mc64) EXPECT_GT(s, 0.0);
}

// ---------------------------------------------------------------------------
// Task 17: PGLib AC-OPF IPOPT captures replayed through the plugin.
// ---------------------------------------------------------------------------

namespace {

// Backward errors of x for A x = b (A symmetric, lower CSC, 0-based), as the
// replay harness gates them: normwise eta = ||r||inf / (||A||inf ||x||inf +
// ||b||inf) and componentwise omega = max_i |r_i| / ((|A||x|)_i + |b_i|).
// r is left holding b - A x.
struct BackwardError {
    double eta = 0.0;
    double omega = 0.0;
};

BackwardError backward_error(const smf::CscLower &A, const std::vector<double> &vals,
                             const std::vector<double> &b, const std::vector<double> &x,
                             std::vector<double> &r) {
    const std::size_t n = static_cast<std::size_t>(A.n);
    std::vector<double> ax(n, 0.0), aax(n, 0.0), rowsum(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        for (smf::Int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k) {
            const std::size_t i = static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(k)]);
            const double a = vals[static_cast<std::size_t>(k)];
            ax[i] += a * x[j];
            aax[i] += std::abs(a) * std::abs(x[j]);
            rowsum[i] += std::abs(a);
            if (i != j) {
                ax[j] += a * x[i];
                aax[j] += std::abs(a) * std::abs(x[i]);
                rowsum[j] += std::abs(a);
            }
        }
    }
    double anorm = 0.0, rnorm = 0.0, xnorm = 0.0, bnorm = 0.0;
    BackwardError e;
    r.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = b[i] - ax[i];
        anorm = std::max(anorm, rowsum[i]);
        rnorm = std::max(rnorm, std::abs(r[i]));
        xnorm = std::max(xnorm, std::abs(x[i]));
        bnorm = std::max(bnorm, std::abs(b[i]));
        const double den = aax[i] + std::abs(b[i]);
        if (den > 0.0) e.omega = std::max(e.omega, std::abs(r[i]) / den);
        else if (r[i] != 0.0) e.omega = HUGE_VAL;
    }
    e.eta = rnorm / (anorm * xnorm + bnorm);
    return e;
}

struct PglibCapture {
    const char *name;
    // MA27's negative-pivot count. It is the same at every stored
    // factorization of these captures, with no zero pivots. Ruling R39: the
    // captures record no inertia, so it was computed once with
    //   bench_replay --solver ma27 --track ipopt --seq-steps all --print-steps
    // on the same fixtures and is kept here.
    int ma27_neg;
};

} // namespace

// The three largest PGLib tuning captures (benchmarks/manifests/ipopt_hard.json)
// replayed factorization by factorization through the plugin, the way IPOPT
// drives it: u = 1e-8, MA97 scaling as IPOPT requests it by default (below),
// every solve wrapped in up to 3 refinement steps.
// Every stored step must factor with MA27's inertia and solve to the replay
// harness's gate (eta <= 1e-10, omega <= 1e-8). Skipped when the fixtures
// (untracked, benchmarks/fixtures/ipopt_hard) are absent.
TEST(IpoptPluginRuntime, PglibReplayAccurate) {
    namespace fs = std::filesystem;
    const fs::path dir =
        fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" / "fixtures" / "ipopt_hard";
    const PglibCapture captures[] = {
        {"pglib_opf_case500_goc__api", 9274},
        {"pglib_opf_case2869_pegase", 57173},
        {"pglib_opf_case4661_sdet", 79212},
    };
    PluginSymbols sym;
    ASSERT_TRUE(sym.load());
    int replayed = 0;
    for (const PglibCapture &cap : captures) {
        const fs::path mtx = dir / cap.name / (std::string(cap.name) + ".mtx");
        const fs::path vals_path = dir / cap.name / (std::string(cap.name) + ".mtx.vals");
        if (!fs::exists(mtx) || !fs::exists(vals_path)) {
            std::printf("[ NOTE     ] %s: fixture missing, skipped\n", cap.name);
            continue;
        }
        auto mm = smf::read_matrix_market(mtx.string());
        ASSERT_TRUE(mm.ok) << cap.name << ": " << mm.error_message;
        const smf::CscLower A = std::move(mm.matrix);
        const int n = static_cast<int>(A.n);
        const std::size_t nnz = A.values.size();
        std::ifstream in(vals_path, std::ios::binary | std::ios::ate);
        const std::size_t steps = static_cast<std::size_t>(in.tellg()) / (nnz * sizeof(double));
        ASSERT_GT(steps, 0u) << cap.name;
        in.seekg(0);

        // 1-based CSC (f_arrays = 1), as IPOPT passes it.
        std::vector<int> ptr(static_cast<std::size_t>(n) + 1), row(nnz);
        for (std::size_t j = 0; j <= static_cast<std::size_t>(n); ++j)
            ptr[j] = static_cast<int>(A.col_ptr[j]) + 1;
        for (std::size_t k = 0; k < nnz; ++k) row[k] = static_cast<int>(A.row_idx[k]) + 1;

        ma97_control_d ctrl{};
        sym.default_control(&ctrl);
        ctrl.f_arrays = 1;
        ctrl.u = 1e-8;
        ctrl.scaling = 0;
        void *akeep = nullptr;
        void *fkeep = nullptr;
        ma97_info_d info{};
        sym.analyse(1, n, ptr.data(), row.data(), nullptr, &akeep, &ctrl, &info, nullptr);
        ASSERT_EQ(info.flag, 0) << cap.name;

        std::vector<double> vals(nnz), scale(static_cast<std::size_t>(n), 1.0);
        const std::vector<double> b(static_cast<std::size_t>(n), 1.0);
        std::vector<double> x, r;
        int failing = 0, wrong_inertia = 0;
        for (std::size_t s = 0; s < steps; ++s) {
            in.read(reinterpret_cast<char *>(vals.data()),
                    static_cast<std::streamsize>(nnz * sizeof(double)));
            ASSERT_TRUE(static_cast<bool>(in)) << cap.name << " step " << s;
            // IPOPT's default MA97 scaling ("dynamic", switch od_hd_reuse):
            // none at first, then MC64 scaling computed once (scaling = 1,
            // returned in `scale`) and passed back for reuse (scaling = 0).
            ctrl.scaling = s == 1 ? 1 : 0;
            sym.factor(4, ptr.data(), row.data(), vals.data(), &akeep, &fkeep, &ctrl, &info,
                       s == 0 ? nullptr : scale.data());
            ctrl.scaling = 0;
            ASSERT_EQ(info.flag, 0) << cap.name << " step " << s;
            if (info.num_neg != cap.ma27_neg || info.matrix_rank != n) {
                ++wrong_inertia;
                ADD_FAILURE() << cap.name << " step " << s << ": num_neg " << info.num_neg << " (MA27 "
                              << cap.ma27_neg << "), rank " << info.matrix_rank << " of " << n;
            }
            x = b;
            sym.solve(0, 1, x.data(), n, &akeep, &fkeep, &ctrl, &info);
            ASSERT_EQ(info.flag, 0) << cap.name << " step " << s;
            BackwardError e = backward_error(A, vals, b, x, r);
            for (int it = 0; it < 3 && !(e.eta <= 1e-14 && e.omega <= 1e-10); ++it) {
                sym.solve(0, 1, r.data(), n, &akeep, &fkeep, &ctrl, &info);
                ASSERT_EQ(info.flag, 0) << cap.name << " step " << s;
                for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) x[i] += r[i];
                e = backward_error(A, vals, b, x, r);
            }
            if (!(e.eta <= 1e-10 && e.omega <= 1e-8)) {
                ++failing;
                ADD_FAILURE() << cap.name << " step " << s << ": eta " << e.eta << " omega " << e.omega;
            }
        }
        std::printf("[ PGLIB    ] %s: %zu steps, %d failing, %d with wrong inertia\n", cap.name, steps,
                    failing, wrong_inertia);
        sym.finalise(&akeep, &fkeep);
        ++replayed;
    }
    if (replayed == 0) GTEST_SKIP() << "no PGLib capture under " << dir.string();
}
