#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef SMF_MA97_PLUGIN_PATH
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

TEST(Ma97PluginAbi, ControlAndInfoLayoutAndOffsets) {
    static_assert(std::is_standard_layout_v<ma97_control_d>);
    static_assert(std::is_standard_layout_v<ma97_info_d>);

    EXPECT_EQ(alignof(ma97_control_d), alignof(double));
    EXPECT_EQ(alignof(ma97_info_d), alignof(double));

    // Critical field offsets invariant across LP64 and LLP64
    EXPECT_EQ(offsetof(ma97_control_d, f_arrays), 0u);
    EXPECT_EQ(offsetof(ma97_control_d, action), 4u);
    EXPECT_EQ(offsetof(ma97_control_d, nemin), 8u);
    EXPECT_EQ(offsetof(ma97_control_d, multiplier), 16u);
    EXPECT_EQ(offsetof(ma97_control_d, ordering), 24u);
    EXPECT_EQ(offsetof(ma97_control_d, print_level), 28u);
    EXPECT_EQ(offsetof(ma97_control_d, scaling), 32u);
    EXPECT_EQ(offsetof(ma97_control_d, small), 40u);
    EXPECT_EQ(offsetof(ma97_control_d, u), 48u);

    EXPECT_EQ(offsetof(ma97_info_d, flag), 0u);
    EXPECT_EQ(offsetof(ma97_info_d, matrix_dup), 12u);
    EXPECT_EQ(offsetof(ma97_info_d, matrix_rank), 16u);
    EXPECT_EQ(offsetof(ma97_info_d, maxfront), 32u);
    EXPECT_EQ(offsetof(ma97_info_d, num_delay), 36u);

    // Explicit platform ABI check: never assume `long` has Linux LP64 width on Windows LLP64
#if defined(_WIN32)
    static_assert(sizeof(long) == 4, "Windows LLP64 has 32-bit long");
    EXPECT_EQ(offsetof(ma97_info_d, num_neg), 48u);
#else
    static_assert(sizeof(long) == 8, "Linux LP64 has 64-bit long");
    EXPECT_EQ(offsetof(ma97_info_d, num_neg), 56u);
#endif
}

#ifdef SMF_MA97_PLUGIN_PATH
namespace {

// The plugin's analyse / factor / solve / finalise entry points.
struct Plugin {
#ifdef _WIN32
    HMODULE handle = nullptr;
    void *sym(const char *name) { return reinterpret_cast<void *>(GetProcAddress(handle, name)); }
#else
    void *handle = nullptr;
    void *sym(const char *name) { return dlsym(handle, name); }
#endif
    void (*default_control)(ma97_control_d *) = nullptr;
    void (*analyse)(int, int, const int *, const int *, double *, void **,
                    const ma97_control_d *, ma97_info_d *, int *) = nullptr;
    void (*factor)(int, const int *, const int *, const double *, void **,
                   void **, const ma97_control_d *, ma97_info_d *, double *) = nullptr;
    void (*finalise)(void **, void **) = nullptr;

    bool load() {
#ifdef _WIN32
        handle = LoadLibraryA(SMF_MA97_PLUGIN_PATH);
#else
        handle = dlopen(SMF_MA97_PLUGIN_PATH, RTLD_NOW);
#endif
        if (!handle) return false;
        default_control = reinterpret_cast<decltype(default_control)>(sym("ma97_default_control_d"));
        analyse = reinterpret_cast<decltype(analyse)>(sym("ma97_analyse_d"));
        factor = reinterpret_cast<decltype(factor)>(sym("ma97_factor_d"));
        finalise = reinterpret_cast<decltype(finalise)>(sym("ma97_finalise_d"));
        return default_control && analyse && factor && finalise;
    }
    ~Plugin() {
#ifdef _WIN32
        if (handle) FreeLibrary(handle);
#else
        if (handle) dlclose(handle);
#endif
    }
};

void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value) setenv(name, value, 1); else unsetenv(name);
#endif
}

// KKT [H B^T; B 0], 0-based lower CSC: H the 5-point Laplacian of a g x g
// grid (diagonal 4), m zero-diagonal constraint rows, row c coupling
// variables 3c, 3c+1, 3c+2 mod g^2. Nonsingular; with AMD the constraint
// rows are eliminated early, where they have no acceptable pivot.
struct DelayKkt {
    int n = 0;
    std::vector<int> ptr, row;
    std::vector<double> val;
    DelayKkt(int g, int m) {
        const int nx = g * g;
        n = nx + m;
        std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n));
        for (int y = 0; y < g; ++y)
            for (int x = 0; x < g; ++x) {
                const int j = y * g + x;
                auto &c = cols[static_cast<std::size_t>(j)];
                c.push_back({j, 4.0});
                if (x + 1 < g) c.push_back({j + 1, -1.0});
                if (y + 1 < g) c.push_back({j + g, -1.0});
            }
        const double coef[3] = {1.0, -0.5, 0.25};
        for (int k = 0; k < m; ++k) {
            for (int t = 0; t < 3; ++t)
                cols[static_cast<std::size_t>((3 * k + t) % nx)].push_back({nx + k, coef[t]});
            cols[static_cast<std::size_t>(nx + k)].push_back({nx + k, 0.0});
        }
        ptr.push_back(0);
        for (auto &c : cols) {
            std::sort(c.begin(), c.end());
            for (const auto &[i, v] : c) {
                row.push_back(i);
                val.push_back(v);
            }
            ptr.push_back(static_cast<int>(row.size()));
        }
    }
};

// Analyse and factor the KKT once with a freshly loaded plugin (its
// environment switches are read at the first analyse after loading).
ma97_info_d factor_once(const char *delay_budget, const char *static_mode) {
    set_env("SMF_TEST_DELAY_BUDGET", delay_budget);
    set_env("SMF_IPM_STATIC", static_mode);
    Plugin plugin;
    ma97_info_d info{};
    info.flag = -100;
    if (!plugin.load()) {
        ADD_FAILURE() << "cannot load " << SMF_MA97_PLUGIN_PATH;
        return info;
    }
    const DelayKkt A(20, 120);
    ma97_control_d ctrl{};
    plugin.default_control(&ctrl);
    ctrl.f_arrays = 0;
    ctrl.ordering = 1;  // AMD
    ctrl.nemin = 1;     // no amalgamation: the constraint rows keep their own fronts
    ctrl.u = 0.01;
    void *akeep = nullptr;
    void *fkeep = nullptr;
    plugin.analyse(1, A.n, A.ptr.data(), A.row.data(), nullptr, &akeep, &ctrl, &info, nullptr);
    set_env("SMF_TEST_DELAY_BUDGET", "");
    set_env("SMF_IPM_STATIC", "");
    if (info.flag != 0) {
        ADD_FAILURE() << "analyse flag " << info.flag;
        plugin.finalise(&akeep, &fkeep);
        return info;
    }
    plugin.factor(4, A.ptr.data(), A.row.data(), A.val.data(), &akeep, &fkeep, &ctrl, &info, nullptr);
    plugin.finalise(&akeep, &fkeep);
    return info;
}

} // namespace

// A factor whose inertia is not exact (a pivot perturbed by static pivoting)
// must never reach IPOPT as a success: the plugin returns MA97's singular
// flag (7), so IPOPT increases its regularization. The test hook
// SMF_TEST_DELAY_BUDGET=0 makes every front static-pivot; the plugin itself
// never sets a delay budget.
TEST(Ma97PluginAbi, InexactInertiaReportsFailure) {
    // Without the hook: exact inertia, plain success.
    const ma97_info_d plain = factor_once("", "");
    EXPECT_EQ(plain.flag, 0);
    EXPECT_GT(plain.num_delay, 0);  // the constraint rows delay
    EXPECT_EQ(plain.num_neg, 120);

    const ma97_info_d forced = factor_once("0", "");
    EXPECT_NE(forced.flag, 0);
    EXPECT_EQ(forced.flag, 7);
    EXPECT_EQ(forced.num_delay, 0);

    // A static mode the plugin requested itself (SMF_IPM_STATIC) perturbs
    // pivots on purpose: those factors keep their own status.
    const ma97_info_d requested = factor_once("0", "fallback");
    EXPECT_EQ(requested.flag, 0);
    EXPECT_EQ(requested.num_delay, 0);
    EXPECT_EQ(requested.num_neg, 120);
}
#endif
