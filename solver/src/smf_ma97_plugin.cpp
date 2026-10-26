#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/ipm_session.hpp"
#include "smf/types.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
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

// The IpmSession owns the analysis, the factor and the workspaces. akeep and
// fkeep share it: MA97 callers may free akeep and still solve with fkeep.
struct SmfAkeep
{
    std::shared_ptr<smf::IpmSession> session = std::make_shared<smf::IpmSession>();
    smf::Control ctrl;          // factor control, updated from every MA97 control block
    smf::CscLower mat;          // pattern + this factorization's values (CSC order)
    std::vector<int> sort_perm; // CSC entry -> caller's entry
    int n = 0;
    int capture_id = 0;          // KKT dump sequence id (SMF_KKT_DUMP_DIR)
    int dump_factorizations = 0; // factorizations dumped for this analysis
};

struct SmfFkeep
{
    std::shared_ptr<smf::IpmSession> session;
    smf::Control ctrl; // solve control: the factor's, updated by each solve's control block
    int n = 0;
    int num_neg = 0;
    int num_two = 0;
    bool aborted = false; // factor stopped on wrong inertia: num_neg is a partial count
};

static bool env_flag(const char* name, bool fallback)
{
    const char* v = std::getenv(name);
    return (v == nullptr || *v == '\0') ? fallback : std::atoi(v) != 0;
}

// The IPM features are read from the environment until their defaults are
// decided (the MA97 interface has no slots for them):
//   SMF_IPM_ABORT=1  inertia learning and early abort (Feature 1), default off.
static smf::IpmControl plugin_ipm_control()
{
    static const bool abort_on = env_flag("SMF_IPM_ABORT", false);
    smf::IpmControl ipm;
    ipm.learn_inertia = abort_on;
    ipm.abort_on_wrong_inertia = abort_on;
    ipm.static_mode = smf::StaticMode::Off;
    return ipm;
}

// Call statistics for benchmarking (SMF_PLUGIN_STATS=1): counts and time of
// factorizations and solves, printed to stderr when the plugin unloads.
struct PluginStats
{
    bool enabled = std::getenv("SMF_PLUGIN_STATS") != nullptr;
    bool trace = std::getenv("SMF_PLUGIN_STATS") != nullptr && std::atoi(std::getenv("SMF_PLUGIN_STATS")) > 1;
    bool last_call_was_factor = false;  // no solve since the previous factorization => IPOPT inertia correction
    long factors = 0;
    long solves = 0;
    long rhs = 0;
    double factor_s = 0.0;
    double solve_s = 0.0;
    ~PluginStats()
    {
        if (enabled)
            std::fprintf(stderr,
                         "[smf_ma97] factors=%ld (%.3f s) solves=%ld rhs=%ld (%.3f s, %.1f us/solve)\n",
                         factors, factor_s, solves, rhs, solve_s,
                         solves > 0 ? 1e6 * solve_s / static_cast<double>(solves) : 0.0);
    }
};
static PluginStats g_stats;

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
    static int next_id = 0;
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

static bool apply_ma97_control(smf::Control &out_ctrl, const ma97_control_d* ctrl)
{
    if (!ctrl)
        return true;

    if (ctrl->nemin > 0)
        out_ctrl.nemin = ctrl->nemin;
    if (ctrl->u >= 0.0 && ctrl->u <= 0.5)
        out_ctrl.pivot_u = ctrl->u;
    if (ctrl->small > 0.0)
        out_ctrl.small_pivot = ctrl->small;
    if (ctrl->multiplier > 0.0)
        out_ctrl.factor_memory_multiplier = ctrl->multiplier;
    out_ctrl.solve_use_blas3_single_rhs = (ctrl->solve_blas3 != 0);
    out_ctrl.solve_multifrontal_forward = (ctrl->solve_mf != 0);
    out_ctrl.continue_on_singular = (ctrl->action == 0);
    out_ctrl.print_level = ctrl->print_level;
    // MA97 callers (IPOPT) refactor one pattern every iteration, which pays
    // for a costlier ordering.
    out_ctrl.expected_refactors = 10;

    switch (ctrl->ordering)
    {
    case 0:
    case 1:
        out_ctrl.ordering = smf::OrderingMethod::AMD;
        break;
    case 3:
        out_ctrl.ordering = smf::OrderingMethod::METIS;
        break;
    case 5:
    default:
        out_ctrl.ordering = smf::OrderingMethod::AutoParallel;
        break;
    }

    // scaling <= 0 is "none", or the caller's own scale vector when factor
    // receives one (applied in ma97_factor_d).
    if (ctrl->scaling <= 0)
    {
        out_ctrl.scaling = smf::ScalingMethod::None;
        return true;
    }
    switch (ctrl->scaling)
    {
    case 1:
        out_ctrl.scaling = smf::ScalingMethod::Matching;
        break;
    case 2:
    case 3:
    case 4:
        out_ctrl.scaling = smf::ScalingMethod::Equilibration;
        break;
    default:
        return false;
    }
    return true;
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
        ctrl->f_arrays = 0;
        ctrl->action = 0;
        ctrl->nemin = 8;
        ctrl->multiplier = 1.2;
        ctrl->ordering = 5;
        ctrl->print_level = -1;
        ctrl->scaling = 0;
        ctrl->small = 1e-20;
        ctrl->u = 1e-8;
        ctrl->unit_diagnostics = 6;
        ctrl->unit_error = 6;
        ctrl->unit_warning = 6;
        ctrl->factor_min = 0;
        ctrl->solve_blas3 = 1;
        ctrl->solve_min = 1;
        ctrl->solve_mf = 1;
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
        // IPOPT's "reuse" scaling strategies compute MC64 scaling once
        // (scaling = 1, returned in `scale`) and pass it back with
        // scaling = 0 on later factorizations.
        const bool computed_scaling = ak->ctrl.scaling != smf::ScalingMethod::None;
        const bool reuse_scaling = !computed_scaling && scale;
        if (reuse_scaling)
            ak->ctrl.scaling = smf::ScalingMethod::UserSupplied;

        fill_values(ak, val);
        dump_kkt(ak);
        smf::IpmSession& session = *ak->session;
        session.set_control(ak->ctrl);
        if (reuse_scaling)
            session.set_user_scale(scale);

        // The first factorization also runs the deferred analysis; that time
        // is not counted as factorization time.
        const bool analysed_before = session.analysis_keep() != nullptr;
        const auto t_factor = std::chrono::steady_clock::now();
        const smf::IpmFactorStatus status = session.factor(ak->mat.values);
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
        fw->n = ak->n;
        fw->ctrl = ak->ctrl;
        fw->aborted = false;
        if (g_stats.enabled)
        {
            ++g_stats.factors;
            g_stats.factor_s +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_factor).count() - analyse_s;
        }
        // A successful SPD factor reports no inertia counts of its own.
        const bool pd = ak->ctrl.matrix_type == smf::MatrixType::RealSymmetricPositiveDefinite &&
                        fs == smf::FactorStatus::Success;
        // Per-factorization trace (SMF_PLUGIN_STATS>=2); corr=1 when the previous
        // plugin call was also a factorization (IPOPT inertia correction).
        // abort=1: stopped on wrong inertia (partial counts); skipped = the
        // predicted flops of the fronts it did not factor; flops = actual.
        if (g_stats.trace)
        {
            std::fprintf(stderr,
                         "[smf_ma97] factor %ld: corr=%d delays=%d static=%d pos=%d neg=%d zero=%d L=%lld t=%.6f "
                         "scaling=%d scale=%s u=%.1e ordering=%d(req %d) nemin=%d two=%d abort=%d flops=%.6e "
                         "skipped=%.6e\n",
                         g_stats.factors, g_stats.last_call_was_factor ? 1 : 0, si.delayed_pivots,
                         si.static_pivots, pd ? ak->n : si.num_positive, pd ? 0 : si.num_negative,
                         pd ? 0 : si.num_zero, static_cast<long long>(si.actual_factor_entries),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t_factor).count() - analyse_s,
                         ctrl ? ctrl->scaling : -1, computed_scaling ? "out" : (scale ? "in" : "-"),
                         ctrl ? ctrl->u : 0.0, si.ordering_used, ctrl ? ctrl->ordering : -1, ak->ctrl.nemin,
                         static_cast<int>(std::count_if(fk.diag_entries.begin(), fk.diag_entries.end(),
                                                        [](const auto& de) { return de.tag == 2; })),
                         aborted ? 1 : 0, si.actual_flops, aborted ? si.aborted_flops_skipped : 0.0);
        }
        g_stats.last_call_was_factor = true;

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

        int two_count = 0;
        for (const auto &de : fk.diag_entries)
        {
            if (de.tag == 2)
                ++two_count;
        }

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

        if (status == smf::IpmFactorStatus::Singular)
        {
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
            const auto& de = session.factor_keep().diag_entries;
            fw->num_two = static_cast<int>(std::count_if(de.begin(), de.end(), [](const auto& d) { return d.tag == 2; }));
            fw->aborted = false;
            if (g_stats.trace)
                std::fprintf(stderr, "[smf_ma97] completed aborted factor %ld before a solve: neg=%d\n",
                             g_stats.factors, fw->num_neg);
        }

        g_stats.last_call_was_factor = false;
        if (g_stats.enabled)
        {
            ++g_stats.solves;
            g_stats.rhs += nrhs;
            g_stats.solve_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_solve).count();
        }
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
