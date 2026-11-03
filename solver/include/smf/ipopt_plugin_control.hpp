#pragma once
// ipopt_plugin_control.hpp — the Control and IpmControl the smf_ma97 plugin
// (src/smf_ma97_plugin.cpp, libsmf_ma97) factorizes IPOPT's KKT systems with.
//
// One source of truth for the plugin and the benchmark harness
// (bench_replay --track ipopt-plugin), header-only so both use it without a
// new library source. The plugin maps every MA97 control block it receives
// through apply_ma97_options(), and the harness builds its Control with
// ipopt_plugin_control() from the same defaults.

#include "smf/control.hpp"
#include "smf/ipm_session.hpp"
#include "smf/threading.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace smf {

/// The HSL MA97 control fields (ma97_control_d) the plugin reads. The member
/// defaults are the plugin's ma97_default_control_d(), which IPOPT calls and
/// then overrides with its own ma97_* options; under IPOPT's defaults
/// (ma97_nemin 8, ma97_small 1e-20, ma97_u 1e-8, ma97_order auto,
/// ma97_scaling dynamic: the first factorizations unscaled) the values IPOPT
/// passes equal these.
struct Ma97Options {
    int    action       = 0;     // 0: continue on singular (Control::continue_on_singular)
    int    nemin        = 8;     // > 0: Control::nemin
    double multiplier   = 1.2;   // > 0: Control::factor_memory_multiplier
    int    ordering     = 5;     // 0/1 AMD, 3 METIS, else AutoParallel
    int    print_level  = -1;
    int    scaling      = 0;     // <= 0 none, 1 matching, 2-4 equilibration, else invalid
    double small        = 1e-20; // > 0: Control::small_pivot
    double u            = 1e-8;  // in [0, 0.5]: Control::pivot_u
    int    solve_blas3  = 1;
    int    solve_mf     = 1;
};

/// Where the plugin's thread count came from (plugin_threads_resolve()).
enum class PluginThreadsSource { SmfNumThreads, OmpNumThreads, PhysicalCores };

struct PluginThreads {
    int threads = 1;
    PluginThreadsSource source = PluginThreadsSource::PhysicalCores;
};

/// A positive int at the start of `v` (OMP_NUM_THREADS may be a list per
/// nesting level, "a,b": the first, the outer level's, counts), else 0.
inline int plugin_threads_parse(const char *v)
{
    if (v == nullptr)
        return 0;
    while (*v == ' ' || *v == '\t')
        ++v;
    char *end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || n <= 0 || n > 4096)
        return 0;
    while (*end == ' ' || *end == '\t')
        ++end;
    return (*end == '\0' || *end == ',') ? static_cast<int>(n) : 0;
}

/// The plugin's Control::num_threads from these environment values (null:
/// unset), first match wins:
///  1. SMF_NUM_THREADS, a positive int: smf's own override;
///  2. (no IPOPT option: IPOPT has no ma97 thread option, and the plugin
///     sees only MA97's control block, which has no thread field either);
///  3. OMP_NUM_THREADS, a positive int (a list "a,b": its first entry);
///  4. `physical_cores`.
/// Why OpenMP's setting: real MA97 has no thread knob and factors on
/// OpenMP's default team (omp_get_max_threads(), from OMP_NUM_THREADS), so
/// a drop-in MA97 replacement inside IPOPT takes the same count. Without
/// OMP_NUM_THREADS the OpenMP runtimes default to every logical processor;
/// smf takes the physical cores instead (SMT siblings add nothing to dense
/// front kernels, and auto_threads trims the count to what the tree repays
/// anyway). Invalid values (0, negative, text) are skipped as unset.
inline PluginThreads plugin_threads_resolve(const char *smf_num_threads, const char *omp_num_threads,
                                            int physical_cores)
{
    if (const int t = plugin_threads_parse(smf_num_threads))
        return {t, PluginThreadsSource::SmfNumThreads};
    if (const int t = plugin_threads_parse(omp_num_threads))
        return {t, PluginThreadsSource::OmpNumThreads};
    return {std::max(1, physical_cores), PluginThreadsSource::PhysicalCores};
}

/// plugin_threads_resolve() of the current environment and
/// smf::physical_core_count(), read now (tests); the plugin uses
/// plugin_num_threads().
inline PluginThreads plugin_threads_from_env()
{
    return plugin_threads_resolve(std::getenv("SMF_NUM_THREADS"), std::getenv("OMP_NUM_THREADS"),
                                  physical_core_count());
}

/// The plugin's thread count: plugin_threads_from_env(), read once per
/// process (as the plugin's other environment knobs).
inline PluginThreads plugin_threads()
{
    static const PluginThreads t = plugin_threads_from_env();
    return t;
}

inline int plugin_num_threads() { return plugin_threads().threads; }

inline const char *plugin_threads_source_name(PluginThreadsSource s)
{
    switch (s)
    {
    case PluginThreadsSource::SmfNumThreads:
        return "SMF_NUM_THREADS";
    case PluginThreadsSource::OmpNumThreads:
        return "OMP_NUM_THREADS";
    case PluginThreadsSource::PhysicalCores:
    default:
        return "physical cores";
    }
}

/// Apply one MA97 control block to `ctrl`, as the plugin does at every
/// analyse, factor and solve call: the mapped fields above plus the plugin's
/// fixed settings (expected_refactors = 10, Control::delay_budget =
/// `delay_budget`; the plugin passes -1 except through its test seam;
/// Control::num_threads = `num_threads`, or plugin_num_threads() when it is
/// <= 0, as the plugin passes). The rest of the parallel configuration stays
/// at Control's defaults (auto_threads, top_tiles), as on the board's
/// multi-thread runs.
/// Returns false for an unsupported scaling value (MA97 flag -6); `ctrl` is
/// then partially updated, as before.
inline bool apply_ma97_options(Control &out_ctrl, const Ma97Options &opt, int delay_budget = -1,
                               int num_threads = 0)
{
    out_ctrl.num_threads = num_threads > 0 ? num_threads : plugin_num_threads();
    if (opt.nemin > 0)
        out_ctrl.nemin = opt.nemin;
    if (opt.u >= 0.0 && opt.u <= 0.5)
        out_ctrl.pivot_u = opt.u;
    if (opt.small > 0.0)
        out_ctrl.small_pivot = opt.small;
    if (opt.multiplier > 0.0)
        out_ctrl.factor_memory_multiplier = opt.multiplier;
    out_ctrl.solve_use_blas3_single_rhs = (opt.solve_blas3 != 0);
    out_ctrl.solve_multifrontal_forward = (opt.solve_mf != 0);
    out_ctrl.continue_on_singular = (opt.action == 0);
    out_ctrl.print_level = opt.print_level;
    // MA97 callers (IPOPT) refactor one pattern every iteration, which pays
    // for a costlier ordering.
    out_ctrl.expected_refactors = 10;
    // No delay budget: delays are never traded for static pivots (inexact
    // inertia), except through the plugin's test seam.
    out_ctrl.delay_budget = delay_budget;

    switch (opt.ordering)
    {
    case 0:
    case 1:
        out_ctrl.ordering = OrderingMethod::AMD;
        break;
    case 3:
        out_ctrl.ordering = OrderingMethod::METIS;
        break;
    case 5:
    default:
        out_ctrl.ordering = OrderingMethod::AutoParallel;
        break;
    }

    // scaling <= 0 is "none", or the caller's own scale vector when factor
    // receives one (applied in the plugin's ma97_factor_d).
    if (opt.scaling <= 0)
    {
        out_ctrl.scaling = ScalingMethod::None;
        return true;
    }
    switch (opt.scaling)
    {
    case 1:
        out_ctrl.scaling = ScalingMethod::Matching;
        break;
    case 2:
    case 3:
    case 4:
        out_ctrl.scaling = ScalingMethod::Equilibration;
        break;
    default:
        return false;
    }
    return true;
}

/// The Control of the plugin's analysis and factorizations for the MA97
/// control block `opt`: a default Control, matrix_type = indefinite (IPOPT's
/// KKT systems), then apply_ma97_options() (num_threads: `num_threads`, or
/// plugin_num_threads() when <= 0). std::nullopt for an invalid scaling
/// value. Everything else stays at Control's defaults, as in the plugin.
inline std::optional<Control> ipopt_plugin_control(const Ma97Options &opt = Ma97Options{}, int delay_budget = -1,
                                                   int num_threads = 0)
{
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    if (!apply_ma97_options(ctrl, opt, delay_budget, num_threads))
        return std::nullopt;
    return ctrl;
}

/// The IpmControl of the plugin's IpmSession: every IPM feature off unless
/// enabled through the environment (read once per process):
/// SMF_IPM_ABORT=1 (inertia learning + early abort), SMF_IPM_REANALYSE=1,
/// SMF_IPM_STATIC=off|fallback|regularized, SMF_IPM_STATIC_ZERO=1.
inline IpmControl ipopt_plugin_ipm_control()
{
    const auto flag = [](const char *name) {
        const char *v = std::getenv(name);
        return v != nullptr && *v != '\0' && std::atoi(v) != 0;
    };
    static const bool abort_on = flag("SMF_IPM_ABORT");
    static const bool reanalyse_on = flag("SMF_IPM_REANALYSE");
    static const StaticMode static_mode = [] {
        const char *v = std::getenv("SMF_IPM_STATIC");
        if (v != nullptr && std::strcmp(v, "fallback") == 0)
            return StaticMode::Fallback;
        if (v != nullptr && std::strcmp(v, "regularized") == 0)
            return StaticMode::Regularized;
        return StaticMode::Off;
    }();
    static const bool static_zero = flag("SMF_IPM_STATIC_ZERO");
    IpmControl ipm;
    ipm.learn_inertia = abort_on;
    ipm.abort_on_wrong_inertia = abort_on;
    ipm.reanalyse = reanalyse_on;
    ipm.static_mode = static_mode;
    ipm.static_as_zero = static_zero;
    return ipm;
}

/// SMF_PLUGIN_MC64=always: the plugin answers every MC64 request (scaling =
/// 1) with the matching scaling, also after the analysis rejected the
/// matching ordering (read once per process).
inline bool plugin_mc64_always()
{
    static const bool always = [] {
        const char *v = std::getenv("SMF_PLUGIN_MC64");
        return v != nullptr && std::strcmp(v, "always") == 0;
    }();
    return always;
}

/// SMF_STALE_SCALE_RETRY (default 1): a factor made singular by a reused
/// scale is refactored once with a fresh matching scaling (read once per
/// process).
inline bool plugin_stale_scale_retry()
{
    static const bool retry = [] {
        const char *v = std::getenv("SMF_STALE_SCALE_RETRY");
        return (v == nullptr || *v == '\0') ? true : std::atoi(v) != 0;
    }();
    return retry;
}

/// The plugin's per-factorization scaling decision (ma97_factor_d), applied
/// to `ctrl` (the Control after apply_ma97_options for this call):
///  - unit_scale: an MC64 request (Matching) after the analysis rejected the
///    matching ordering is answered with no scaling, and the unit scale is
///    handed back (unless plugin_mc64_always());
///  - computed: the factorization computes a scaling, handed back in `scale`;
///  - reuse: no scaling requested and the caller passed a non-unit `scale`
///    (IPOPT's "reuse" strategies): ctrl.scaling = UserSupplied, and the
///    caller sets the session's user scale to `scale`.
/// `matching_rejected`: an analysis exists and its matching_rejected is set.
struct Ma97ScalingCall {
    bool unit_scale = false;
    bool computed = false;
    bool reuse = false;
};

inline Ma97ScalingCall ma97_scaling_call(Control &ctrl, bool matching_rejected, const double *scale, int n)
{
    Ma97ScalingCall r;
    r.unit_scale = !plugin_mc64_always() && ctrl.scaling == ScalingMethod::Matching && matching_rejected;
    if (r.unit_scale)
        ctrl.scaling = ScalingMethod::None;
    r.computed = ctrl.scaling != ScalingMethod::None;
    r.reuse = !r.computed && !r.unit_scale && scale != nullptr &&
              !std::all_of(scale, scale + n, [](double s) { return s == 1.0; });
    if (r.reuse)
        ctrl.scaling = ScalingMethod::UserSupplied;
    return r;
}

} // namespace smf
