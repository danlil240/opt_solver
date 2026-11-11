#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "smf/analysis.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/factor_indef.hpp"
#include "smf/info.hpp"
#include "smf/ipm_session.hpp"
#include "smf/ipopt_plugin_control.hpp"
#include "smf/slack_elimination.hpp"
#include "smf/types.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <string>
#include <vector>

struct ma97_control_d
{
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

struct ma97_info_d
{
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

// A new IpmSession whose statistics are added to the unload summary when the
// last of akeep / fkeep frees it.
static std::shared_ptr<smf::IpmSession> make_session();

// The plugin's own per-session state, beside the IpmSession and shared by
// akeep and fkeep like it. Only the SMF_PLUGIN_STATS trace reads it; it never
// changes a factorization or a solve. It is per session, not process-wide, so
// that MA97 sessions running concurrently or interleaved in one process
// (concurrent IPOPT instances, IPOPT's restoration phase next to its main
// solver) do not see each other's calls. As with MA97, one session is used by
// one thread at a time.
struct PluginSession
{
    bool last_call_was_factor = false; // no solve since this session's previous factorization => IPOPT inertia correction
    long last_factor_id = 0;           // trace id of this session's latest factorization
    smf::Ma97CascadeState cascade;     // do this analysis' unscaled factorizations cascade (MC64 requests)
};

// The IpmSession owns the analysis, the factor and the workspaces. akeep and
// fkeep share it: MA97 callers may free akeep and still solve with fkeep.
struct SmfAkeep
{
    std::shared_ptr<smf::IpmSession> session = make_session();
    std::shared_ptr<PluginSession> state = std::make_shared<PluginSession>();
    smf::Control ctrl;          // factor control, updated from every MA97 control block
    smf::CscLower mat;          // pattern + this factorization's values (CSC order)
    std::vector<int> sort_perm; // CSC entry -> caller's entry
    int n = 0;
    int capture_id = 0;          // KKT dump sequence id (SMF_KKT_DUMP_DIR)
    int dump_factorizations = 0; // factorizations dumped for this analysis
};

// The factor itself lives in the session, so two fkeeps made from one akeep
// would share it: `generation` is the session's factor_generation() when this
// fkeep was factored, and a solve with an fkeep whose factor has since been
// replaced (another fkeep factored from the same akeep) fails with flag -1
// (MA97's "error in the sequence of calls"). IPOPT reuses one fkeep handle per
// akeep, which always holds the current generation.
struct SmfFkeep
{
    std::shared_ptr<smf::IpmSession> session;
    std::shared_ptr<PluginSession> state; // the akeep's, set by the factorization
    smf::Control ctrl; // solve control: the factor's, updated by each solve's control block
    int n = 0;
    int num_neg = 0;
    int num_two = 0;
    std::uint64_t generation = 0; // session factor generation of this fkeep's factor
    bool aborted = false; // factor stopped on wrong inertia: num_neg is a partial count
};

// Threads: Control::num_threads is smf::plugin_num_threads()
// (smf/ipopt_plugin_control.hpp, read once per process at the first call:
// later environment changes or omp_set_num_threads() have no effect):
// SMF_NUM_THREADS, else OMP_NUM_THREADS (its first entry), else the
// performance cores the process affinity allows (smf::performance_core_count():
// every physical core on a non-hybrid CPU, only the P-cores on a hybrid one;
// on an i9-12900K 16 threads were slower than 8 on 51 of 72 plugin-track
// cases, see ipopt_plugin_control.hpp). The variables are taken as given.
// Real MA97 has no thread option (nor has IPOPT one for it): it factors on
// OpenMP's default team, i.e. OMP_NUM_THREADS. Every analyse, factor and
// solve call runs under a BlasThreadGuard of that count (the calling
// thread's local MKL setting, its previous one restored on return), so BLAS
// sees what it sees on bench_replay's N-thread runs (mkl_set_num_threads(N),
// num_threads = N): the factorization then splits those threads itself
// (serial BLAS in the subtrees and tiles, shares at the top; deterministic
// mode: one), and IPOPT's own BLAS setting outside the calls is left alone.
// More than one thread changes the analysis too, not only its speed: under
// IPOPT's default ordering (AutoParallel) large matrices get parallel nested
// dissection instead of serial METIS and the matching trial runs
// concurrently, so orderings, delays, fill and IPOPT's iterates can differ
// from the serial plugin and depend on the core count. SMF_NUM_THREADS=1
// gives the serial plugin back (also the reproducible choice; deterministic
// mode, Control::deterministic, is the other one). Concurrent IPOPT
// instances in one machine (multistart, parallel batches) now get P threads
// each (P: the performance cores): set SMF_NUM_THREADS (or OMP_NUM_THREADS)
// to share the cores.
//
// The IPM features are opt-in through the environment (the MA97 interface
// has no slots for them). Their defaults were decided on the held-out IPOPT
// models (docs/performance/2026-10-26-ipm-mode.md): every feature failed its
// gate there, so every default is off and the plugin behaves as plain smf.
//   SMF_IPM_ABORT=1      inertia learning and early abort (Feature 1), default
//                        off: identical logs, but saves < 5% (the excess
//                        negatives appear in the root front); tuning gate failed too.
//   SMF_IPM_REANALYSE=1  re-analysis from the realized pivot order (Feature 2),
//                        default off; IpmControl's default thresholds. Held-out
//                        wall geomean 1.017 > 1; tuning failed the iteration band.
//   SMF_IPM_STATIC=off|fallback|regularized  static mode (Feature 3), default
//                        off; an unknown value is off. Both static modes fail
//                        the held-out outcome gate (new failures, iterations).
//   SMF_IPM_STATIC_ZERO=1  static pivots count as zero eigenvalues instead of
//                        by the hybrid rule (IpmControl::static_as_zero), default 0.
// The plugin's IpmControl: smf::ipopt_plugin_ipm_control()
// (smf/ipopt_plugin_control.hpp), shared with bench_replay's ipopt-plugin
// track, reads the variables above.
static smf::IpmControl plugin_ipm_control()
{
    return smf::ipopt_plugin_ipm_control();
}

// Test seam, never set in production: SMF_TEST_DELAY_BUDGET=k gives every
// factorization Control::delay_budget = k (k = 0: every front static-pivots),
// so a test can force static pivots without requesting a static mode. Read
// once, at the first analyse after the plugin is loaded; unset, the plugin
// keeps the default delay_budget = -1 (unlimited delays, exact inertia).
static int test_delay_budget()
{
    static const int budget = [] {
        const char* v = std::getenv("SMF_TEST_DELAY_BUDGET");
        return (v == nullptr || *v == '\0') ? -1 : std::atoi(v);
    }();
    return budget;
}

// Call statistics for benchmarking (SMF_PLUGIN_STATS=1): counts and time of
// the factorizations and solves of every session in the process, printed to
// stderr when the plugin unloads. Sessions may run concurrently (one per
// IPOPT instance), so the totals are atomics or guarded by `mutex`; the
// per-session trace state is PluginSession's.
struct PluginStats
{
    const bool enabled = std::getenv("SMF_PLUGIN_STATS") != nullptr;
    const bool trace = std::getenv("SMF_PLUGIN_STATS") != nullptr && std::atoi(std::getenv("SMF_PLUGIN_STATS")) > 1;
    std::atomic<long> factors{0}; // also the trace id of the latest factorization
    std::atomic<long> solves{0};
    std::atomic<long> rhs{0};
    std::mutex mutex; // guards the members below
    double factor_s = 0.0;
    double solve_s = 0.0;
    // Solve calls and their time by right-hand-side count.
    std::map<int, std::pair<long, double>> solves_by_nrhs;
    // IpmSession statistics of the sessions freed so far (one per analysis).
    long sessions = 0;
    smf::IpmStats ipm;
    // Counts a factorization; returns its trace id (1, 2, ... over the process).
    long add_factor(double seconds)
    {
        const long id = factors.fetch_add(1) + 1;
        const std::lock_guard<std::mutex> lock(mutex);
        factor_s += seconds;
        return id;
    }
    void add_solve(int nrhs, double seconds)
    {
        solves.fetch_add(1);
        rhs.fetch_add(nrhs);
        const std::lock_guard<std::mutex> lock(mutex);
        solve_s += seconds;
        auto& [calls, s] = solves_by_nrhs[nrhs];
        ++calls;
        s += seconds;
    }
    void add_session(const smf::IpmStats& st)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        ++sessions;
        ipm.factorizations += st.factorizations;
        ipm.accepted += st.accepted;
        ipm.aborted += st.aborted;
        ipm.reanalyses_tried += st.reanalyses_tried;
        ipm.reanalyses_adopted += st.reanalyses_adopted;
        ipm.reanalyses_reverted += st.reanalyses_reverted;
        ipm.static_pivots += st.static_pivots;
        ipm.flops_avoided += st.flops_avoided;
        ipm.reanalyse_seconds += st.reanalyse_seconds;
    }
    ~PluginStats()
    {
        if (!enabled)
            return;
        const std::lock_guard<std::mutex> lock(mutex);
        const long n_factors = factors.load();
        const long n_solves = solves.load();
        const long n_rhs = rhs.load();
        std::fprintf(stderr,
                     "[smf_ma97] threads=%d (%s) factors=%ld (%.3f s) solves=%ld rhs=%ld (%.3f s, %.1f us/solve)\n",
                     smf::plugin_threads().threads, smf::plugin_threads_source_name(smf::plugin_threads().source),
                     n_factors, factor_s, n_solves, n_rhs, solve_s,
                     n_solves > 0 ? 1e6 * solve_s / static_cast<double>(n_solves) : 0.0);
        for (const auto& [k, v] : solves_by_nrhs)
            std::fprintf(stderr, "[smf_ma97] solves nrhs=%d: %ld calls, %.3f ms, %.1f us/call\n", k, v.first,
                         1e3 * v.second, 1e6 * v.second / static_cast<double>(v.first));
        std::fprintf(stderr,
                     "[smf_ma97] ipm: sessions=%ld factorizations=%d accepted=%d aborted=%d "
                     "reanalyses_tried=%d reanalyses_adopted=%d reanalyses_reverted=%d reanalyse_s=%.6f "
                     "flops_avoided=%.6e\n",
                     sessions, ipm.factorizations, ipm.accepted, ipm.aborted, ipm.reanalyses_tried,
                     ipm.reanalyses_adopted, ipm.reanalyses_reverted, ipm.reanalyse_seconds, ipm.flops_avoided);
    }
};
static PluginStats g_stats;

// IPOPT loads the hsllib for every solve and frees it afterwards, so an
// application that optimizes over and over (a replanning robot) loaded smf
// and its dependencies (MKL, the OpenMP runtimes) again and re-ran their
// initialisation every time: about 2 ms of each 9-11 ms trajectory replan.
// The module pins itself on its first load; later loads only count a
// reference, and the statistics above print once, at process exit. The
// SMF_* settings the plugin reads once per load are then read once per
// process: SMF_PLUGIN_UNLOAD=1 keeps the old unload per free (the plugin
// tests change those settings between loads).
static void pin_this_module()
{
    if (const char* v = std::getenv("SMF_PLUGIN_UNLOAD"); v != nullptr && std::atoi(v) != 0)
        return;
#if defined(_WIN32)
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                       reinterpret_cast<LPCWSTR>(&pin_this_module), &self);
#else
    Dl_info di{};
    if (dladdr(reinterpret_cast<void*>(&pin_this_module), &di) != 0 && di.dli_fname != nullptr)
        dlopen(di.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
#endif
}
static const bool g_pinned = (pin_this_module(), true);

static std::shared_ptr<smf::IpmSession> make_session()
{
    return std::shared_ptr<smf::IpmSession>(new smf::IpmSession(), [](smf::IpmSession* s) {
        if (g_stats.enabled)
            g_stats.add_session(s->stats());
        delete s;
    });
}

// KKT capture for benchmarking (SMF_KKT_DUMP_DIR): the first factorization of
// each analysis writes <dir>/kkt_s<id>.mtx (cleaned lower triangle, entries in
// column-major order); every factorization appends its values, in that same
// order, to <dir>/kkt_s<id>.vals as raw doubles. SMF_KKT_DUMP_MAX (default
// 200) caps the factorizations recorded per analysis.
static void dump_kkt(SmfAkeep* ak)
{
    static const char* dir = std::getenv("SMF_KKT_DUMP_DIR");
    if (dir == nullptr || *dir == '\0')
        return;
    static std::atomic<int> next_id{0};
    static const int max_dumps = std::getenv("SMF_KKT_DUMP_MAX") ? std::atoi(std::getenv("SMF_KKT_DUMP_MAX")) : 200;
    if (ak->dump_factorizations == 0)
        ak->capture_id = next_id++;
    if (ak->dump_factorizations >= max_dumps)
        return;
    const std::filesystem::path base =
        std::filesystem::path(dir) / ("kkt_s" + std::to_string(ak->capture_id));
    const smf::CscLower& A = ak->mat;
    if (ak->dump_factorizations == 0)
    {
        std::ofstream mtx(base.string() + ".mtx");
        mtx << "%%MatrixMarket matrix coordinate real symmetric\n"
            << "% IPOPT KKT matrix captured by the smf_ma97 plugin\n"
            << A.n << " " << A.n << " " << A.row_idx.size() << "\n"
            << std::setprecision(17);
        for (smf::Int j = 0; j < A.n; ++j)
            for (smf::Int k = A.col_ptr[static_cast<size_t>(j)]; k < A.col_ptr[static_cast<size_t>(j) + 1]; ++k)
                mtx << A.row_idx[static_cast<size_t>(k)] + 1 << " " << j + 1 << " "
                    << A.values[static_cast<size_t>(k)] << "\n";
    }
    std::ofstream vals(base.string() + ".vals", std::ios::binary | std::ios::app);
    vals.write(reinterpret_cast<const char*>(A.values.data()),
               static_cast<std::streamsize>(A.values.size() * sizeof(double)));
    ++ak->dump_factorizations;
}

static bool build_csc_pattern(SmfAkeep* ak, int n, const int* ptr, const int* row, int base)
{
    if (!ak || n <= 0 || !ptr || !row)
        return false;

    struct Entry
    {
        int col, row, orig;
    };
    std::vector<Entry> entries;
    const int nnz_total = ptr[n] - base;
    if (nnz_total < 0)
        return false;
    entries.reserve(static_cast<size_t>(nnz_total));

    for (int j = 0; j < n; ++j)
    {
        const int cs = ptr[j] - base;
        const int ce = ptr[j + 1] - base;
        if (cs < 0 || ce < cs)
            return false;
        for (int p = cs; p < ce; ++p)
        {
            const int r = row[p] - base;
            if (r < 0 || r >= n)
                continue;
            if (r >= j)
                entries.push_back({j, r, p});
            else
                entries.push_back({r, j, p});
        }
    }
    const int m = static_cast<int>(entries.size());

    std::sort(entries.begin(), entries.end(),
              [](const Entry &a, const Entry &b) noexcept { return a.col != b.col ? a.col < b.col : a.row < b.row; });

    ak->mat.n = n;
    ak->mat.col_ptr.assign(static_cast<size_t>(n + 1), 0);
    ak->mat.row_idx.resize(static_cast<size_t>(m));
    ak->mat.values.assign(static_cast<size_t>(m), 0.0);
    ak->sort_perm.resize(static_cast<size_t>(m));
    ak->n = n;

    for (const auto &e : entries)
        ak->mat.col_ptr[static_cast<size_t>(e.col + 1)]++;
    for (int j = 0; j < n; ++j)
        ak->mat.col_ptr[static_cast<size_t>(j + 1)] += ak->mat.col_ptr[static_cast<size_t>(j)];
    for (int k = 0; k < m; ++k)
    {
        ak->mat.row_idx[static_cast<size_t>(k)] = entries[static_cast<size_t>(k)].row;
        ak->sort_perm[static_cast<size_t>(k)] = entries[static_cast<size_t>(k)].orig;
    }
    return true;
}

static void fill_values(SmfAkeep* ak, const double* val)
{
    const int m = static_cast<int>(ak->mat.values.size());
    for (int k = 0; k < m; ++k)
        ak->mat.values[static_cast<size_t>(k)] = val[static_cast<size_t>(ak->sort_perm[static_cast<size_t>(k)])];
}

// The MA97 control fields the plugin reads, as smf::Ma97Options.
static smf::Ma97Options ma97_options(const ma97_control_d& c)
{
    smf::Ma97Options o;
    o.action = c.action;
    o.nemin = c.nemin;
    o.multiplier = c.multiplier;
    o.ordering = c.ordering;
    o.print_level = c.print_level;
    o.scaling = c.scaling;
    o.small = c.small;
    o.u = c.u;
    o.solve_blas3 = c.solve_blas3;
    o.solve_mf = c.solve_mf;
    return o;
}

// Every MA97 control block maps through smf::apply_ma97_options()
// (smf/ipopt_plugin_control.hpp), the single source of the plugin's Control
// that bench_replay's ipopt-plugin track builds from too.
static bool apply_ma97_control(smf::Control &out_ctrl, const ma97_control_d* ctrl)
{
    // No control block: the MA97 defaults stay, but the thread count and
    // the pivoting policy (SMF_PIVOTING) are the plugin's all the same.
    out_ctrl.num_threads = smf::plugin_num_threads();
    out_ctrl.pivoting = smf::plugin_pivoting();
    out_ctrl.slack_elimination = smf::plugin_slack_elimination();
    if (!ctrl)
        return true;
    return smf::apply_ma97_options(out_ctrl, ma97_options(*ctrl), test_delay_budget());
}

extern "C"
{

    void ma97_solve_d(int job,
                      int nrhs,
                      double* x,
                      int ldx,
                      void** akeep,
                      void** fkeep,
                      const ma97_control_d* ctrl,
                      ma97_info_d* info);

    void ma97_default_control_d(ma97_control_d* ctrl)
    {
        if (!ctrl)
            return;
        std::memset(ctrl, 0, sizeof(*ctrl));
        // The fields the plugin reads default to smf::Ma97Options{}.
        const smf::Ma97Options d;
        ctrl->f_arrays = 0;
        ctrl->action = d.action;
        ctrl->nemin = d.nemin;
        ctrl->multiplier = d.multiplier;
        ctrl->ordering = d.ordering;
        ctrl->print_level = d.print_level;
        ctrl->scaling = d.scaling;
        ctrl->small = d.small;
        ctrl->u = d.u;
        ctrl->unit_diagnostics = 6;
        ctrl->unit_error = 6;
        ctrl->unit_warning = 6;
        ctrl->factor_min = 0;
        ctrl->solve_blas3 = d.solve_blas3;
        ctrl->solve_min = 1;
        ctrl->solve_mf = d.solve_mf;
        ctrl->consist_tol = 0.0;
    }

    void ma97_analyse_d(int /*check*/,
                        int n,
                        const int* ptr,
                        const int* row,
                        double* val,
                        void** akeep,
                        const ma97_control_d* ctrl,
                        ma97_info_d* info,
                        int* /*order*/)
    {
        if (!info)
            return;
        std::memset(info, 0, sizeof(*info));
        if (!akeep || n <= 0 || !ptr || !row)
        {
            info->flag = -1;
            if (akeep) *akeep = nullptr;
            return;
        }

        const int base = (ctrl && ctrl->f_arrays) ? 1 : 0;

        auto* ak = new SmfAkeep();
        ak->ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
        if (!apply_ma97_control(ak->ctrl, ctrl))
        {
            info->flag = -6;
            delete ak;
            *akeep = nullptr;
            return;
        }

        if (!build_csc_pattern(ak, n, ptr, row, base))
        {
            info->flag = -1;
            delete ak;
            *akeep = nullptr;
            return;
        }

        // smf's analysis needs values to recognise saddle-point matrices
        // (their matching ordering keeps 2x2 pivots together; AMD alone
        // delays ~1 pivot per row on ACOPF KKT systems). IPOPT analyses the
        // pattern only, so without values the analysis is deferred to the
        // first factorization.
        std::span<const double> values;
        if (val)
        {
            fill_values(ak, val);
            values = ak->mat.values;
        }
        const smf::BlasThreadGuard blas(ak->ctrl.num_threads);
        if (ak->session->analyse(ak->mat, values, ak->ctrl, plugin_ipm_control()) != smf::ErrorCode::Success)
        {
            info->flag = -3;
            delete ak;
            *akeep = nullptr;
            return;
        }

        info->flag = 0;
        info->matrix_rank = n;
        if (const smf::AnalysisKeep* a = ak->session->analysis_keep())
        {
            const smf::Info& si = ak->session->info();
            info->matrix_dup = si.matrix_duplicates;
            info->matrix_outrange = si.matrix_out_of_range;
            info->matrix_missing_diag = si.matrix_missing_diag;
            info->num_sup = static_cast<int>(a->supernodes.size());
            info->maxdepth = si.max_tree_depth;
            info->maxfront = si.max_front_size;
            info->maxsupernode = si.max_supernode_size;
            info->ordering = si.ordering_used;
        }
        *akeep = ak;
    }

    void ma97_factor_d(int matrix_type,
                       const int* /*ptr*/,
                       const int* /*row*/,
                       const double* val,
                       void** akeep,
                       void** fkeep,
                       const ma97_control_d* ctrl,
                       ma97_info_d* info,
                       double* scale)
    {
        if (!info)
            return;
        std::memset(info, 0, sizeof(*info));
        if (!akeep || !*akeep || !fkeep || !val)
        {
            info->flag = -5;
            return;
        }

        auto* ak = static_cast<SmfAkeep*>(*akeep);
        ak->ctrl.matrix_type = (matrix_type == static_cast<int>(smf::MatrixType::RealSymmetricPositiveDefinite))
                                   ? smf::MatrixType::RealSymmetricPositiveDefinite
                                   : smf::MatrixType::RealSymmetricIndefinite;
        if (!apply_ma97_control(ak->ctrl, ctrl))
        {
            info->flag = -6;
            return;
        }
        // IPOPT's MC64 request (scaling = 1) once the pattern is analysed:
        // when the analysis computed the matching ordering and rejected it
        // (kept AMD or METIS), the matching scaling is not applied, as under
        // ScalingMethod::Auto. Without its ordering it makes the factor less
        // accurate (triage-case9241-notes.md; on PGLib case2869 / case4661 a
        // reused matching scale gives wrong inertia and omega up to 1 on most
        // steps), and recomputing it costs ~10x a refactorization. The
        // request is then answered with the unit scale, which IPOPT stores
        // and passes back. Patterns the analysis did not try the matching on
        // (n below its minimum, no zero/small diagonal) keep the matching
        // scaling, which helps there (COPS elec / chain iteration counts).
        // Exception: a delay cascade. IPOPT asks for MC64 when the delays
        // jump; when the latest factorization's front work also reached
        // kPluginCascadeWorkRatio times the delay-free work (or an earlier
        // request of this analysis was honoured for that), the unscaled
        // factorization has stopped working and the request is honoured
        // (Ma97CascadeState; Mittelmann qcqp1500-1c: 0.03 -> 7 s per
        // factorization without it).
        // SMF_PLUGIN_MC64=always restores the matching scaling on every
        // request, =never the unit answer after every rejected ordering.
        // Known limitation: a request on the very first factorization (no
        // analysis yet; the analysis is deferred into session.factor below)
        // still gets the matching scaling. IPOPT's default ma97_scaling
        // (dynamic) factors unscaled first, so its requests always come after
        // the analysis; with ma97_scaling = mc64 the first scale is matching
        // and its reuse keeps it, the pre-Task-17 behaviour.
        // IPOPT's "reuse" scaling strategies compute MC64 scaling once
        // (scaling = 1, returned in `scale`) and pass it back with
        // scaling = 0 on later factorizations. A unit scale passed back is
        // no scaling. The decision is smf::ma97_scaling_call()
        // (smf/ipopt_plugin_control.hpp), shared with bench_replay's
        // ipopt-plugin track; SMF_PLUGIN_MC64 is smf::plugin_mc64_always().
        const smf::AnalysisKeep* analysed = ak->session->analysis_keep();
        // Whether the matching was rejected is the full matrix's own analysis'
        // finding (with a slack plan, made now when a matching is requested).
        const bool rejected = analysed != nullptr &&
                              (ak->ctrl.scaling == smf::ScalingMethod::Matching
                                   ? smf::slack_full_analysis(*analysed).matching_rejected
                                   : analysed->matching_rejected);
        const smf::Ma97ScalingCall scaling_call =
            smf::ma97_scaling_call(ak->ctrl, rejected, scale, ak->n, ak->state->cascade.cascade());
        const bool unit_scale = scaling_call.unit_scale;
        bool computed_scaling = scaling_call.computed;
        const bool reuse_scaling = scaling_call.reuse;

        fill_values(ak, val);
        dump_kkt(ak);
        smf::IpmSession& session = *ak->session;
        // A scale computed here goes back to IPOPT for reuse, so it must be
        // the full matrix's: such factorizations do not take the slack plan.
        smf::Control fc = ak->ctrl;
        if (computed_scaling) fc.slack_elimination = smf::SlackElimination::Off;
        session.set_control(fc);
        if (reuse_scaling)
            session.set_user_scale(scale);

        // The first factorization also runs the deferred analysis; that time
        // is not counted as factorization time.
        const bool analysed_before = session.analysis_keep() != nullptr;
        // ra= trace: reverts decided before this factorization (a revert
        // decided at its end does not change the analysis it ran on).
        const int reverted_before = session.stats().reanalyses_reverted;
        const smf::BlasThreadGuard blas(ak->ctrl.num_threads);
        const auto t_factor = std::chrono::steady_clock::now();
        smf::IpmFactorStatus status = session.factor(ak->mat.values);
        // A reused scale that makes the factor singular is presumed stale: one
        // computed at a blown-up iterate (entries ~1e23 on cops_chain_800)
        // shrinks pivots of later, nonsingular KKT matrices below `small`.
        // IPOPT never rescales after a singular factor (it returns before its
        // high-delay check), so refactor once with a fresh matching scaling
        // and hand that scale back for its later reuses. If the fresh scaling
        // cannot be computed, the stale-scale result stands.
        if (smf::plugin_stale_scale_retry() && reuse_scaling && status == smf::IpmFactorStatus::Singular)
        {
            // The id is the one this factorization will get; with other
            // sessions factoring concurrently it is approximate.
            if (g_stats.trace)
                std::fprintf(stderr, "[smf_ma97] factor %ld: singular with a reused scale (zero=%d): rescaling\n",
                             g_stats.factors.load() + 1, session.info().num_zero);
            smf::Control fresh = ak->ctrl;
            fresh.scaling = smf::ScalingMethod::Matching;
            smf::Control ffc = fresh;
            ffc.slack_elimination = smf::SlackElimination::Off;
            session.set_control(ffc);
            const smf::IpmFactorStatus retried = session.factor(ak->mat.values);
            if (retried == smf::IpmFactorStatus::Ok || retried == smf::IpmFactorStatus::Singular)
            {
                status = retried;
                ak->ctrl = fresh;
                computed_scaling = true;
            }
            else
            {
                session.set_control(ak->ctrl);
                session.set_user_scale(scale);
                status = session.factor(ak->mat.values);
            }
        }
        const bool aborted = status == smf::IpmFactorStatus::WrongInertia;
        const smf::Info& si = session.info();
        if (session.analysis_keep() == nullptr)
        {
            info->flag = -3;
            return;
        }
        const double analyse_s = analysed_before ? 0.0 : si.analyse_seconds;
        const smf::FactorStatus fs = si.factor_status;
        const smf::FactorKeep& fk = session.factor_keep();

        // Refactorizations reuse the previous fkeep handle; the factor's
        // storage itself lives in the session.
        auto* fw = *fkeep ? static_cast<SmfFkeep*>(*fkeep) : nullptr;
        *fkeep = nullptr;
        if (!fw)
            fw = new SmfFkeep();
        fw->session = ak->session;
        fw->state = ak->state;
        fw->n = ak->n;
        fw->ctrl = ak->ctrl;
        fw->generation = session.factor_generation();
        fw->aborted = false;
        PluginSession& state = *ak->state;
        // Work of the factorization that ran, over the delay-free work: the
        // next MC64 request's cascade test.
        const double work_ratio_before = state.cascade.last_work_ratio;
        if (status == smf::IpmFactorStatus::Ok || status == smf::IpmFactorStatus::Singular)
        {
            const smf::AnalysisKeep& fak = *session.analysis_keep();
            const double free_work =
                state.cascade.free_work_of(fak.serial, [&fak] { return smf::delay_free_front_work(fak); });
            state.cascade.after_factor(si.front_work, free_work, scaling_call.cascade_honoured && computed_scaling);
        }
        if (g_stats.enabled)
            state.last_factor_id = g_stats.add_factor(
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_factor).count() - analyse_s);
        // A successful SPD factor reports no inertia counts of its own.
        const bool pd = ak->ctrl.matrix_type == smf::MatrixType::RealSymmetricPositiveDefinite &&
                        fs == smf::FactorStatus::Success;
        // Per-factorization trace (SMF_PLUGIN_STATS>=2); corr=1 when the previous
        // call on this session (akeep) was also a factorization (IPOPT inertia
        // correction); other sessions' calls do not count, and a new analysis
        // (IPOPT's restoration phase included) starts at corr=0.
        // abort=1: stopped on wrong inertia (partial counts); skipped = the
        // predicted flops of the fronts it did not factor; flops = actual.
        // work = front work over the delay-free work (prev: of the session's
        // previous factorization, what an MC64 request here was judged on);
        // scale=out-cascade: an MC64 request honoured after a delay cascade.
        // ra=tried/adopted/reverted: the session's re-analysis counts, tried
        // and adopted including this factorization, reverted as of its start,
        // so adopted > reverted exactly when it ran on a re-analysed order
        // (a probation factor that is then reverted included).
        if (g_stats.trace)
        {
            const smf::IpmStats& ist = session.stats();
            std::fprintf(stderr,
                         "[smf_ma97] factor %ld: corr=%d delays=%d static=%d pos=%d neg=%d zero=%d L=%lld t=%.6f "
                         "scaling=%d scale=%s u=%.1e ordering=%d(req %d) nemin=%d two=%d abort=%d flops=%.6e "
                         "skipped=%.6e work=%.2f(prev %.2f) ra=%d/%d/%d\n",
                         state.last_factor_id, state.last_call_was_factor ? 1 : 0, si.delayed_pivots,
                         si.static_pivots, pd ? ak->n : si.num_positive, pd ? 0 : si.num_negative,
                         pd ? 0 : si.num_zero, static_cast<long long>(si.actual_factor_entries),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t_factor).count() - analyse_s,
                         ctrl ? ctrl->scaling : -1,
                         unit_scale ? "unit"
                                    : computed_scaling ? (scaling_call.cascade_honoured ? "out-cascade" : "out")
                                                       : (reuse_scaling ? "in" : "-"),
                         ctrl ? ctrl->u : 0.0, si.ordering_used, ctrl ? ctrl->ordering : -1, ak->ctrl.nemin,
                         si.two_by_two_pivots,
                         aborted ? 1 : 0, si.actual_flops, aborted ? si.aborted_flops_skipped : 0.0,
                         state.cascade.last_work_ratio, work_ratio_before,
                         ist.reanalyses_tried, ist.reanalyses_adopted, reverted_before);
        }
        state.last_call_was_factor = true;

        // This early return comes before the inexact-inertia guard below on
        // purpose: an aborted factor (SMF_IPM_ABORT only, off by default)
        // is already rejected by IPOPT through its num_neg (above the
        // learned expectation), and the solve that completes it refactors.
        // Wrong inertia: the factorization stopped once its negative pivots
        // exceeded the learned expectation. MA97 has no status for it, so it
        // is a successful factor whose num_neg (the partial count, already
        // above the expectation) IPOPT rejects; it then refactors with a
        // larger delta. A solve with this fkeep completes the factorization.
        // The scaling ran before the abort: return it as for any factor
        // (IPOPT's reuse strategies keep it for later factorizations).
        if (aborted)
        {
            if (computed_scaling && scale && !fk.scale.empty())
                std::copy(fk.scale.begin(), fk.scale.end(), scale);
            if (unit_scale && scale)
                std::fill(scale, scale + ak->n, 1.0);
            fw->num_neg = si.num_negative;
            fw->num_two = 0;
            fw->aborted = true;
            info->flag = 0;
            info->num_neg = fw->num_neg;
            info->num_two = 0;
            info->matrix_rank = ak->n;
            info->num_delay = si.delayed_pivots;
            info->maxfront = si.max_front_size;
            info->num_factor = static_cast<long>(si.actual_factor_entries);
            info->num_flops = static_cast<long>(si.actual_flops);
            info->maxsupernode = si.max_supernode_size;
            info->ordering = si.ordering_used;
            *fkeep = fw;
            return;
        }

        // Every other status but Ok / Singular is a failure; smf reports
        // input/scaling/allocation failures as FactorStatus::Singular too,
        // which IpmSession maps to Error (only SingularMatrix is the MA97
        // "singular" warning).
        if (status != smf::IpmFactorStatus::Ok && status != smf::IpmFactorStatus::Singular)
        {
            info->flag = -5;
            delete fw;
            return;
        }

        const int num_pos = pd ? ak->n : si.num_positive;
        const int num_neg = pd ? 0 : si.num_negative;
        const int num_zero = pd ? 0 : si.num_zero;
        const int inertia_total = num_pos + num_neg + num_zero;
        if (inertia_total != static_cast<int>(session.analysis_keep()->n))
        {
            info->flag = -5;
            delete fw;
            return;
        }

        // Counted by the factorization (Control::lazy_diag_entries leaves the
        // D entries unmerged until a solve needs them).
        const int two_count = si.two_by_two_pivots;

        fw->num_neg = num_neg;
        fw->num_two = two_count;
        info->num_neg = fw->num_neg;
        info->num_two = fw->num_two;
        info->matrix_rank = ak->n - num_zero;
        info->num_delay = si.delayed_pivots;
        info->maxfront = si.max_front_size;
        info->num_factor = static_cast<long>(si.actual_factor_entries);
        info->num_flops = static_cast<long>(si.actual_flops);
        info->maxsupernode = si.max_supernode_size;
        info->ordering = si.ordering_used;

        if (computed_scaling && scale && !fk.scale.empty())
        {
            std::copy(fk.scale.begin(), fk.scale.end(), scale);
        }
        if (unit_scale && scale)
            std::fill(scale, scale + ak->n, 1.0);

        if (status == smf::IpmFactorStatus::Singular)
        {
            info->flag = 7;
            *fkeep = fw;
            return;
        }

        // Honest inertia: a factor with a static pivot (Info::inertia_exact
        // false) has the inertia of a perturbed matrix, which IPOPT must not
        // accept as the KKT inertia. MA97 has no flag for that, so it is
        // reported as MA97's singular warning (flag 7), on which IPOPT
        // increases its regularization and refactors. This applies only when
        // the plugin did not ask for static pivots itself: the opt-in static
        // modes (SMF_IPM_STATIC=fallback|regularized, off by default) perturb
        // pivots on purpose and report their inertia by their own
        // (hybrid / static_as_zero) accounting, unchanged. With the defaults
        // (no static mode, delay_budget -1) no pivot is ever static, so the
        // guard never fires.
        if (!si.inertia_exact && plugin_ipm_control().static_mode == smf::StaticMode::Off)
        {
            if (g_stats.trace)
                std::fprintf(stderr, "[smf_ma97] factor %ld: %d static pivots, inexact inertia: flag 7\n",
                             state.last_factor_id, si.static_pivots);
            info->flag = 7;
            *fkeep = fw;
            return;
        }

        info->flag = 0;
        *fkeep = fw;
    }

    void ma97_factor_solve_d(int matrix_type,
                             const int* ptr,
                             const int* row,
                             const double* val,
                             int nrhs,
                             double* x,
                             int ldx,
                             void** akeep,
                             void** fkeep,
                             const ma97_control_d* ctrl,
                             ma97_info_d* info,
                             double* scale)
    {
        ma97_factor_d(matrix_type, ptr, row, val, akeep, fkeep, ctrl, info, scale);
        if (!info || info->flag < 0)
            return;
        ma97_solve_d(0, nrhs, x, ldx, akeep, fkeep, ctrl, info);
    }

    void ma97_solve_d(int job,
                      int nrhs,
                      double* x,
                      int ldx,
                      void** /*akeep*/,
                      void** fkeep,
                      const ma97_control_d* ctrl,
                      ma97_info_d* info)
    {
        if (!info)
            return;
        std::memset(info, 0, sizeof(*info));
        if (!fkeep || !*fkeep || !x)
        {
            info->flag = -5;
            return;
        }

        auto* fw = static_cast<SmfFkeep*>(*fkeep);
        const int n = fw->n;
        // This fkeep's factor was replaced by a later factorization into
        // another fkeep from the same akeep (they share one session).
        if (!fw->session || fw->generation != fw->session->factor_generation())
        {
            info->flag = -1;
            return;
        }
        if (nrhs < 1)
        {
            info->flag = -2;
            return;
        }
        if (ldx < n)
        {
            info->flag = -4;
            return;
        }
        if (job < 0 || job > 4)
        {
            info->flag = -5;
            return;
        }

        apply_ma97_control(fw->ctrl, ctrl);
        const smf::SolveJob sjob = static_cast<smf::SolveJob>(job);
        smf::IpmSession& session = *fw->session;
        session.set_control(fw->ctrl);
        int rc = 0;
        const smf::BlasThreadGuard blas(fw->ctrl.num_threads);

        const auto t_solve = std::chrono::steady_clock::now();
        if (ldx == n)
        {
            rc = session.solve(x, n, nrhs, sjob);
        }
        else
        {
            std::vector<double> buf(static_cast<size_t>(n) * static_cast<size_t>(nrhs));
            for (int j = 0; j < nrhs; ++j)
                std::copy(x + static_cast<size_t>(j) * ldx,
                          x + static_cast<size_t>(j) * ldx + n,
                          buf.data() + static_cast<size_t>(j) * n);
            rc = session.solve(buf.data(), n, nrhs, sjob);
            if (rc == 0)
            {
                for (int j = 0; j < nrhs; ++j)
                    std::copy(buf.data() + static_cast<size_t>(j) * n,
                              buf.data() + static_cast<size_t>(j + 1) * n,
                              x + static_cast<size_t>(j) * ldx);
            }
        }

        // A solve after an aborted factor completed it first: report the
        // full inertia from now on.
        if (fw->aborted && rc == 0)
        {
            const smf::InertiaCounts in = session.inertia();
            fw->num_neg = in.negative;
            const auto& de = session.factor_keep().diag();
            fw->num_two = static_cast<int>(std::count_if(de.begin(), de.end(), [](const auto& d) { return d.tag == 2; }));
            fw->aborted = false;
            if (g_stats.trace)
                std::fprintf(stderr, "[smf_ma97] completed aborted factor %ld before a solve: neg=%d\n",
                             fw->state ? fw->state->last_factor_id : 0L, fw->num_neg);
        }

        if (fw->state)
            fw->state->last_call_was_factor = false;
        if (g_stats.enabled)
            g_stats.add_solve(nrhs, std::chrono::duration<double>(std::chrono::steady_clock::now() - t_solve).count());
        info->flag = (rc == 0) ? 0 : -5;
        info->num_neg = fw->num_neg;
        info->num_two = fw->num_two;
    }

    void ma97_free_akeep_d(void** akeep)
    {
        if (akeep && *akeep)
        {
            delete static_cast<SmfAkeep*>(*akeep);
            *akeep = nullptr;
        }
    }

    void ma97_free_fkeep_d(void** fkeep)
    {
        if (fkeep && *fkeep)
        {
            delete static_cast<SmfFkeep*>(*fkeep);
            *fkeep = nullptr;
        }
    }

    void ma97_finalise_d(void** akeep, void** fkeep)
    {
        ma97_free_fkeep_d(fkeep);
        ma97_free_akeep_d(akeep);
    }

} // extern "C"
