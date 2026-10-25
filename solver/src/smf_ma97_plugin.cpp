#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
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

struct SmfAkeep
{
    smf::Solver solver;
    smf::Control ctrl;
    smf::CscLower mat;
    smf::Info info;
    std::unique_ptr<smf::AnalysisKeep> ak;
    smf::NumericWorkspace nws;
    std::vector<int> sort_perm;
    int n = 0;
    int capture_id = 0;          // KKT dump sequence id (SMF_KKT_DUMP_DIR)
    int dump_factorizations = 0; // factorizations dumped for this analysis
};

struct SmfFkeep
{
    smf::FactorKeep fk;
    smf::Solver solver;
    smf::Control ctrl;
    smf::Info info;
    smf::SolveWorkspace sws;
    int n = 0;
    int num_neg = 0;
    int num_two = 0;
    int capture_id = 0;
    bool rhs_captured = false;
};

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

        if (val)
            fill_values(ak, val);

        ak->ak = ak->solver.analyse(ak->mat, ak->ctrl, ak->info);
        if (!ak->ak || ak->info.status != smf::ErrorCode::Success)
        {
            info->flag = -3;
            delete ak;
            *akeep = nullptr;
            return;
        }

        info->flag = 0;
        info->matrix_dup = ak->info.matrix_duplicates;
        info->matrix_outrange = ak->info.matrix_out_of_range;
        info->matrix_missing_diag = ak->info.matrix_missing_diag;
        info->matrix_rank = n;
        info->num_sup = static_cast<int>(ak->ak->supernodes.size());
        info->maxdepth = ak->info.max_tree_depth;
        info->maxfront = ak->info.max_front_size;
        info->maxsupernode = ak->info.max_supernode_size;
        info->ordering = ak->info.ordering_used;
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
        if (!ak->ak)
        {
            info->flag = -5;
            return;
        }

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
        if (!computed_scaling && scale)
        {
            ak->ctrl.scaling = smf::ScalingMethod::UserSupplied;
            ak->ctrl.user_scale = scale;
        }

        fill_values(ak, val);
        dump_kkt(ak);

        // Refactorizations reuse the previous factor object so its storage
        // (factor values, solve layout) is not reallocated every iteration.
        auto* fw = *fkeep ? static_cast<SmfFkeep*>(*fkeep) : nullptr;
        if (fw && fw->n != ak->n)
        {
            delete fw;
            fw = nullptr;
        }
        *fkeep = nullptr;
        if (!fw)
            fw = new SmfFkeep();
        fw->n = ak->n;
        fw->ctrl = ak->ctrl;
        const smf::FactorStatus fs = ak->solver.factor(*ak->ak, ak->mat, ak->ctrl, ak->info, fw->fk, ak->nws);
        ak->ctrl.user_scale = nullptr;
        fw->ctrl.user_scale = nullptr;

        // smf reports input/scaling/allocation failures as Singular too;
        // only SingularMatrix is the MA97 "singular" warning.
        if (fs == smf::FactorStatus::Singular && ak->info.status != smf::ErrorCode::SingularMatrix)
        {
            info->flag = -5;
            delete fw;
            return;
        }

        if (ak->ctrl.matrix_type == smf::MatrixType::RealSymmetricPositiveDefinite &&
            fs == smf::FactorStatus::Success)
        {
            ak->info.num_positive = ak->n;
            ak->info.num_negative = 0;
            ak->info.num_zero = 0;
        }

        const int inertia_total = ak->info.num_positive + ak->info.num_negative + ak->info.num_zero;
        if (inertia_total != static_cast<int>(ak->ak->n))
        {
            info->flag = -5;
            delete fw;
            return;
        }

        int two_count = 0;
        for (const auto &de : fw->fk.diag_entries)
        {
            if (de.tag == 2)
                ++two_count;
        }

        fw->num_neg = ak->info.num_negative;
        fw->num_two = two_count;
        info->num_neg = fw->num_neg;
        info->num_two = fw->num_two;
        info->matrix_rank = ak->n - ak->info.num_zero;
        info->num_delay = ak->info.delayed_pivots;
        info->maxfront = ak->info.max_front_size;
        info->num_factor = static_cast<long>(ak->info.actual_factor_entries);
        info->num_flops = static_cast<long>(ak->info.actual_flops);
        info->maxsupernode = ak->info.max_supernode_size;
        info->ordering = ak->info.ordering_used;

        if (computed_scaling && scale && !fw->fk.scale.empty())
        {
            std::copy(fw->fk.scale.begin(), fw->fk.scale.end(), scale);
        }

        if (fs == smf::FactorStatus::Singular)
        {
            info->flag = 7;
            *fkeep = fw;
            return;
        }
        if (fs != smf::FactorStatus::Success && fs != smf::FactorStatus::MaxPivotDelays)
        {
            info->flag = -5;
            delete fw;
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
        int rc = 0;

        if (ldx == n)
        {
            rc = fw->solver.solve(fw->fk, fw->ctrl, fw->info, x, n, nrhs, fw->sws, sjob);
        }
        else
        {
            std::vector<double> buf(static_cast<size_t>(n) * static_cast<size_t>(nrhs));
            for (int j = 0; j < nrhs; ++j)
                std::copy(x + static_cast<size_t>(j) * ldx,
                          x + static_cast<size_t>(j) * ldx + n,
                          buf.data() + static_cast<size_t>(j) * n);
            rc = fw->solver.solve(fw->fk, fw->ctrl, fw->info, buf.data(), n, nrhs, fw->sws, sjob);
            if (rc == 0)
            {
                for (int j = 0; j < nrhs; ++j)
                    std::copy(buf.data() + static_cast<size_t>(j) * n,
                              buf.data() + static_cast<size_t>(j + 1) * n,
                              x + static_cast<size_t>(j) * ldx);
            }
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
