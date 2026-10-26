#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

    // Reuse a (perturbed) user scale: applied as given, never overwritten.
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
