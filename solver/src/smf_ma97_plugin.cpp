#include <cmath>
#include <cstdio>
/**
 * smf_ma97_plugin.cpp  —  HSL MA97 C interface backed by smf.
 *
 * Build as a shared library (libsmf_ma97.so) with
 *   cmake -DSMF_BUILD_MA97_PLUGIN=ON
 *   cmake --build build --target smf_ma97
 *   cmake --install build --prefix /usr/local
 *
 * Then tell IPOPT to use it by adding two solver options:
 *   s_opts["linear_solver"] = "ma97";
 *   s_opts["hsllib"]        = "/usr/local/lib/libsmf_ma97.so";
 *
 * IPOPT loads the 7 symbols below via dlsym at runtime.  The structs
 * ma97_control_d / ma97_info_d are laid out to match the HSL MA97 3.x
 * C interface specification (same layout IPOPT 3.14 was compiled against).
 *
 * Compatibility note
 * ------------------
 * The only fields read at well-known offsets are:
 *   control.f_arrays  (offset 0, int)  — safe to read regardless of layout
 *   info.flag         (offset 0, int)  — safe to write regardless of layout
 * For solvers using hessian_approximation=limited-memory (L-BFGS) IPOPT
 * does not check inertia, so info.num_neg is never consulted.  For
 * exact-Hessian mode the struct layout must match HSL MA97 exactly.
 */

#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// MA97 struct definitions  (HSL MA97 2.8 / IPOPT 3.14 C interface spec)
// ============================================================================

struct ma97_control_d
{
    // CRITICAL: field order must match HSL MA97 2.8 / IPOPT 3.14 exactly
    // or IPOPT will misread control parameters like u (pivot tolerance)
    int f_arrays;         // [0]  0=C 0-based, 1=Fortran 1-based
    int action;           // [4]  0=continue on singular, 1=abort
    int nemin;            // [8]  node amalgamation minimum
    double multiplier;    // [16] factor memory multiplier (8-byte aligned)
    int ordering;         // [24] ordering method: 5=METIS/auto
    int print_level;      // [28] -1=silent, 0=errors, 1=warnings, 2+=diagnostics
    int scaling;          // [32] 0=user/none, 1-4=HSL scaling methods
    double small;         // [40] small pivot threshold (8-byte aligned)
    double u;             // [48] pivot tolerance (CRITICAL: IPOPT default 1e-8)
    int unit_diagnostics; // [56]
    int unit_error;       // [60]
    int unit_warning;     // [64]
    long factor_min;      // [72] minimum factor allocation (8-byte aligned)
    int solve_blas3;      // [80] use BLAS3 for solve
    int solve_min;        // [84] minimum solve work for threading
    int solve_mf;         // [88] solve multifrontal control
    double consist_tol;   // [96] consistency tolerance (8-byte aligned)
    int ispare[5];        // [104] reserved
    double rspare[10];    // [124] reserved (8-byte aligned)
};

struct ma97_info_d
{
    // CRITICAL: field order must match HSL MA97 2.8 / IPOPT 3.14 exactly
    int flag;                // [0]  return code: 0=success, <0=error, 7=singular
    int flag68;              // [4]  out-of-range warning count
    int flag77;              // [8]  dup/rank deficiency warning count
    int matrix_dup;          // [12] duplicate entries
    int matrix_rank;         // [16] estimated rank
    int matrix_outrange;     // [20] out-of-range entries
    int matrix_missing_diag; // [24] missing diagonal entries
    int maxdepth;            // [28] max supernode tree depth
    int maxfront;            // [32] max frontal matrix size
    int num_delay;           // [36] delayed pivots
    long num_factor;         // [40] # nonzeros in factor (8-byte aligned)
    long num_flops;          // [48] # floating-point ops
    int num_neg;             // [56] negative eigenvalues (CRITICAL for IPOPT)
    int num_sup;             // [60] supernodes
    int num_two;             // [64] 2×2 pivots
    int ordering;            // [68] ordering used
    int stat;                // [72] Fortran allocation status
    int maxsupernode;        // [76] largest supernode size
    int ispare[4];           // [80] reserved
    double rspare[10];       // [96] reserved (8-byte aligned)
};

// ============================================================================
// Internal per-problem state kept behind the opaque akeep / fkeep pointers
// ============================================================================

struct SmfAkeep
{
    smf::Solver solver;
    smf::Control ctrl;
    smf::CscLower mat; // sorted lower-CSC copy
    smf::Info info;
    std::unique_ptr<smf::AnalysisKeep> ak;

    // Map: position k in mat → original index in the IPOPT val[] array
    std::vector<int> sort_perm;
    // Map: position k in mat → position in ak->cleaned (for value scatter)
    std::vector<int> csc_to_cleaned;
    int n = 0;
};

struct SmfFkeep
{
    smf::FactorKeep fk;
    int num_neg = 0;
    int capture_id = 0;
    bool rhs_captured = false;
};

// ============================================================================
// Internal helpers
// ============================================================================

static bool build_csc_pattern(SmfAkeep* ak, int n, const int* ptr, const int* row, int base)
{
    // Collect (col, row, orig_pos) for lower-triangle entries (row >= col)
    struct Entry
    {
        int col, row, orig;
    };
    std::vector<Entry> entries;
    const int nnz_total = ptr[n] - base;
    entries.reserve(static_cast<size_t>(nnz_total));

    for (int j = 0; j < n; ++j)
    {
        const int cs = ptr[j] - base;
        const int ce = ptr[j + 1] - base;
        for (int p = cs; p < ce; ++p)
        {
            const int r = row[p] - base;
            if (r < 0 || r >= n)
                continue; // out-of-range guard
            if (r >= j)   // lower triangle only
                entries.push_back({j, r, p});
        }
    }
    const int m = static_cast<int>(entries.size());

    // Sort into column-major order for CscLower
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

static void build_csc_to_cleaned(SmfAkeep* ak)
{
    // ak->ak->cleaned is the ORIGINAL-ordered cleaned matrix (from
    // symbolic_analysis.cpp: keep->cleaned = std::move(cleaned.clean)).
    // Factor permutes it internally via permute_lower_csc_indef(cleaned, iperm).
    // Therefore we must search by ORIGINAL (old_j, old_i) — no iperm needed.
    const int n = ak->n;
    const int mat_nnz = static_cast<int>(ak->mat.values.size());

    ak->csc_to_cleaned.assign(static_cast<size_t>(mat_nnz), -1);

    for (int old_j = 0; old_j < n; ++old_j)
    {
        const int cs = ak->mat.col_ptr[static_cast<size_t>(old_j)];
        const int ce = ak->mat.col_ptr[static_cast<size_t>(old_j + 1)];
        for (int p = cs; p < ce; ++p)
        {
            const int old_i = ak->mat.row_idx[static_cast<size_t>(p)];

            // Search for (old_j, old_i) in cleaned (original ordering).
            const int cl_cs = static_cast<int>(ak->ak->cleaned.col_ptr[static_cast<size_t>(old_j)]);
            const int cl_ce = static_cast<int>(ak->ak->cleaned.col_ptr[static_cast<size_t>(old_j + 1)]);

            const auto beg = ak->ak->cleaned.row_idx.begin() + cl_cs;
            const auto end = ak->ak->cleaned.row_idx.begin() + cl_ce;
            const auto it = std::lower_bound(beg, end, old_i);

            ak->csc_to_cleaned[static_cast<size_t>(p)] = static_cast<int>(it - ak->ak->cleaned.row_idx.begin());
        }
    }
}

static void fill_values(SmfAkeep* ak, const double* val)
{
    const int m = static_cast<int>(ak->mat.values.size());
    for (int k = 0; k < m; ++k)
        ak->mat.values[static_cast<size_t>(k)] = val[static_cast<size_t>(ak->sort_perm[static_cast<size_t>(k)])];
}

static void push_to_cleaned(SmfAkeep* ak)
{
    // Defensive: ensure cleaned.values is properly sized — analyse may leave
    // it empty if the input was zero-valued at analyse time.
    const auto cleaned_nnz = static_cast<size_t>(ak->ak->cleaned.nnz());
    if (ak->ak->cleaned.values.size() != cleaned_nnz)
        ak->ak->cleaned.values.assign(cleaned_nnz, 0.0);
    else
        std::fill(ak->ak->cleaned.values.begin(), ak->ak->cleaned.values.end(), 0.0);

    const int m = static_cast<int>(ak->mat.values.size());
    for (int k = 0; k < m; ++k)
    {
        const int pos = ak->csc_to_cleaned[static_cast<size_t>(k)];
        if (pos >= 0 && pos < static_cast<int>(ak->ak->cleaned.values.size()))
        {
            ak->ak->cleaned.values[static_cast<size_t>(pos)] += ak->mat.values[static_cast<size_t>(k)];
        }
    }
}

static int capture_limit()
{
    const char* env = std::getenv("SMF_MA97_CAPTURE_LIMIT");
    if (!env || env[0] == '\0')
        return 1;
    const int value = std::atoi(env);
    return value > 0 ? value : 1;
}

static std::string capture_path(const char* dir, int id, const char* suffix)
{
    std::ostringstream os;
    os << dir << "/kkt_" << std::setw(4) << std::setfill('0') << id << suffix;
    return os.str();
}

static int maybe_capture_matrix(const SmfAkeep* ak)
{
    const char* dir = std::getenv("SMF_MA97_CAPTURE_DIR");
    if (!dir || dir[0] == '\0' || !ak || !ak->ak)
        return 0;

    static int capture_count = 0;
    if (capture_count >= capture_limit())
        return 0;

    const int id = ++capture_count;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
        return 0;

    const smf::CscLower& A = ak->ak->cleaned;
    const std::string matrix_file = capture_path(dir, id, ".mtx");
    std::ofstream out(matrix_file);
    if (!out)
        return 0;

    out << "%%MatrixMarket matrix coordinate real symmetric\n";
    out << "% Captured from smf_ma97_plugin ma97_factor_d\n";
    out << A.n << ' ' << A.n << ' ' << A.nnz() << '\n';
    out << std::setprecision(17);
    for (int j = 0; j < A.n; ++j)
    {
        for (smf::Int p = A.col_ptr[static_cast<size_t>(j)]; p < A.col_ptr[static_cast<size_t>(j + 1)]; ++p)
        {
            const int i = A.row_idx[static_cast<size_t>(p)];
            out << (i + 1) << ' ' << (j + 1) << ' ' << A.values[static_cast<size_t>(p)] << '\n';
        }
    }

    std::ofstream meta(capture_path(dir, id, ".meta"));
    if (meta)
    {
        meta << "id " << id << '\n';
        meta << "n " << A.n << '\n';
        meta << "nnz_lower " << A.nnz() << '\n';
        meta << "source smf_ma97_plugin ma97_factor_d\n";
    }
    return id;
}

static void maybe_capture_rhs(const SmfAkeep* ak, SmfFkeep* fw, const double* x, int ldx, int nrhs)
{
    const char* dir = std::getenv("SMF_MA97_CAPTURE_DIR");
    if (!dir || dir[0] == '\0' || !ak || !fw || fw->capture_id <= 0 || fw->rhs_captured || !x)
        return;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
        return;

    std::ofstream out(capture_path(dir, fw->capture_id, ".rhs"));
    if (!out)
        return;

    const int n = ak->n;
    out << n << ' ' << nrhs << '\n';
    out << std::setprecision(17);
    for (int rhs = 0; rhs < nrhs; ++rhs)
    {
        for (int i = 0; i < n; ++i)
            out << x[static_cast<size_t>(rhs * ldx + i)] << '\n';
    }
    fw->rhs_captured = true;
}

static void apply_ma97_control(SmfAkeep* ak, const ma97_control_d* ctrl)
{
    if (!ak || !ctrl)
        return;

    // Map HSL MA97 control parameters to smf::Control
    if (ctrl->nemin > 0)
        ak->ctrl.nemin = ctrl->nemin;
    
    // IPOPT owns MA97 pivot-quality escalation.  Do not floor this value here:
    // same-matrix KKT comparison showed the old 0.01 floor caused a real solve
    // residual regression while IPOPT's requested 1e-8 remained accurate.
    if (ctrl->u > 0.0)
        ak->ctrl.pivot_u = ctrl->u;
    
    if (ctrl->small > 0.0)
        ak->ctrl.small_pivot = ctrl->small;
    if (ctrl->multiplier > 0.0)
        ak->ctrl.factor_memory_multiplier = static_cast<int>(ctrl->multiplier);
    ak->ctrl.solve_use_blas3_single_rhs = (ctrl->solve_blas3 != 0);
    // action: 0=continue on singular, 1=abort → map to continue_on_singular
    ak->ctrl.continue_on_singular = (ctrl->action == 0);
    // scaling: 0=user/none, 1-4=HSL methods → conservatively map to None
    // (smf doesn't fully support HSL scaling enums; IPOPT manages its own scaling)
}

// ============================================================================
// Exported C symbols  (loaded by IPOPT via dlsym when linear_solver=ma97)
// ============================================================================

extern "C"
{

    // Forward declarations (allow mutual calls within the extern "C" block)
    void ma97_solve_d(int job,
                      int nrhs,
                      double* x,
                      int ldx,
                      void** akeep,
                      void** fkeep,
                      const ma97_control_d* ctrl,
                      ma97_info_d* info);

    // ---------------------------------------------------------------------------
    // ma97_default_control_d  — called once by IPOPT to get default settings
    // ---------------------------------------------------------------------------
    void ma97_default_control_d(ma97_control_d* ctrl)
    {
        std::memset(ctrl, 0, sizeof(*ctrl));
        // CRITICAL: these defaults must match IPOPT 3.14's registered MA97 defaults
        ctrl->f_arrays = 0;     // 0-based C indexing (IPOPT overwrites to 1)
        ctrl->action = 0;       // 0=continue on singular (IPOPT expects this)
        ctrl->nemin = 8;        // node amalgamation threshold
        ctrl->multiplier = 1.2; // factor memory multiplier
        ctrl->ordering = 5;     // 5=METIS/auto
        ctrl->print_level = -1; // -1=silent
        ctrl->scaling = 0;      // 0=none/user (IPOPT manages its own scaling)
        ctrl->small = 1e-20;    // small pivot threshold
        ctrl->u = 1e-8;         // pivot tolerance (IPOPT default, NOT 0.01)
        ctrl->unit_diagnostics = 6;
        ctrl->unit_error = 6;
        ctrl->unit_warning = 6;
        ctrl->factor_min = 0;
        ctrl->solve_blas3 = 1;  // enable BLAS3 for solve
        ctrl->solve_min = 1;
        ctrl->solve_mf = 1;
        ctrl->consist_tol = 0.0;
    }

    // ---------------------------------------------------------------------------
    // ma97_analyse_d  — symbolic analysis (builds sparsity pattern + ordering)
    // ---------------------------------------------------------------------------
    void ma97_analyse_d(int /*check*/,
                        int n,
                        const int* ptr,
                        const int* row,
                        double* val, // may be NULL (pattern only)
                        void** akeep,
                        const ma97_control_d* ctrl,
                        ma97_info_d* info,
                        int* /*order*/)
    {
        std::memset(info, 0, sizeof(*info));

        const int base = (ctrl && ctrl->f_arrays) ? 1 : 0;

        auto* ak = new SmfAkeep();
        ak->ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
        apply_ma97_control(ak, ctrl);

        if (!build_csc_pattern(ak, n, ptr, row, base))
        {
            info->flag = -1;
            delete ak;
            *akeep = nullptr;
            return;
        }

        // Fill numerical values if provided (optional at analyse stage)
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

        build_csc_to_cleaned(ak);

        // Populate info fields from analysis
        info->flag = 0;
        info->matrix_rank = n;
        info->num_sup = static_cast<int>(ak->ak->supernodes.size());
        info->ordering = ctrl ? ctrl->ordering : 5;
        *akeep = ak;
    }

    // ---------------------------------------------------------------------------
    // ma97_factor_d  — numerical factorization
    // ---------------------------------------------------------------------------
    void ma97_factor_d(int matrix_type,
                       const int* /*ptr*/,
                       const int* /*row*/,
                       const double* val,
                       void** akeep,
                       void** fkeep,
                       const ma97_control_d* ctrl,
                       ma97_info_d* info,
                       double* /*scale*/)
    {
        std::memset(info, 0, sizeof(*info));

        auto* ak = static_cast<SmfAkeep*>(*akeep);
        if (!ak || !ak->ak)
        {
            info->flag = -5;
            return;
        }

        // Set matrix_type BEFORE calling apply_ma97_control so pivot_u clamping works correctly
        ak->ctrl.matrix_type = (matrix_type == static_cast<int>(smf::MatrixType::RealSymmetricPositiveDefinite))
                                   ? smf::MatrixType::RealSymmetricPositiveDefinite
                                   : smf::MatrixType::RealSymmetricIndefinite;
        apply_ma97_control(ak, ctrl);

        // Scatter IPOPT's values into ak->cleaned
        fill_values(ak, val);
        push_to_cleaned(ak);

        // Debug: verify values are non-zero (only if print_level > 1)
        if (ctrl && ctrl->print_level > 1)
        {
            double vsum = 0.0;
            for (double v : ak->ak->cleaned.values)
                vsum += v * v;
            fprintf(stderr, "[smf_ma97] factor: cleaned values norm=%.3e (n=%d nnz=%d)\n", std::sqrt(vsum), ak->n,
                    (int)ak->ak->cleaned.values.size());
        }

        // Free any previously held fkeep to avoid memory leak
        if (*fkeep)
        {
            delete static_cast<SmfFkeep*>(*fkeep);
            *fkeep = nullptr;
        }

        auto* fw = new SmfFkeep();
        fw->capture_id = maybe_capture_matrix(ak);
        const smf::FactorStatus fs = ak->solver.factor(*ak->ak, ak->ctrl, ak->info, fw->fk);

        if (ctrl && ctrl->print_level > 1)
        {
            fprintf(stderr, "[smf_ma97] factor: status=%d pos=%d neg=%d zero=%d (keep.n=%d)\n", (int)fs,
                    ak->info.num_positive, ak->info.num_negative, ak->info.num_zero, (int)ak->ak->n);
        }

        // CRITICAL: sanity check inertia before exposing to IPOPT
        const int inertia_total = ak->info.num_positive + ak->info.num_negative + ak->info.num_zero;
        if (inertia_total != static_cast<int>(ak->ak->n))
        {
            if (ctrl && ctrl->print_level >= 0)
            {
                fprintf(stderr,
                    "[smf_ma97] ERROR: invalid inertia total=%d n=%d pos=%d neg=%d zero=%d\n",
                    inertia_total, static_cast<int>(ak->ak->n), ak->info.num_positive,
                    ak->info.num_negative, ak->info.num_zero);
            }
            info->flag = -5;
            delete fw;
            return;
        }

        fw->num_neg = ak->info.num_negative;
        // Populate info fields from factor
        info->num_neg = fw->num_neg;
        info->matrix_rank = ak->n - ak->info.num_zero;
        info->num_delay = ak->info.delayed_pivots;
        info->maxfront = ak->info.max_front_size;
        info->num_factor = static_cast<long>(ak->info.actual_factor_entries);
        info->num_flops = static_cast<long>(ak->info.actual_flops);
        info->maxsupernode = ak->info.max_supernode_size;

        if (fs == smf::FactorStatus::Singular)
        {
            // flag=7 is the HSL MA97 "singular" warning — positive, not fatal.
            // IPOPT sees SYMSOLVER_SINGULAR and applies inertia correction.
            // We still expose fkeep so solve can proceed if IPOPT chooses to.
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

    // ---------------------------------------------------------------------------
    // ma97_factor_solve_d  — factor then immediately solve (IPOPT calls this
    //                         when new_matrix=true and solve is needed at once)
    // ---------------------------------------------------------------------------
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
        if (ctrl && ctrl->print_level > 1)
            fprintf(stderr, "[smf_ma97] factor_solve_d called\n");
        ma97_factor_d(matrix_type, ptr, row, val, akeep, fkeep, ctrl, info, scale);
        if (ctrl && ctrl->print_level > 1)
            fprintf(stderr, "[smf_ma97] factor_solve_d: after factor flag=%d\n", info->flag);
        if (info->flag < 0)
            return;
        ma97_solve_d(0, nrhs, x, ldx, akeep, fkeep, ctrl, info);
    }

    // ---------------------------------------------------------------------------
    // ma97_solve_d  — triangular solve  (job=0: full AX=B)
    // ---------------------------------------------------------------------------
    void ma97_solve_d(int job,
                      int nrhs,
                      double* x,
                      int ldx,
                      void** akeep,
                      void** fkeep,
                      const ma97_control_d* ctrl,
                      ma97_info_d* info)
    {
        if (ctrl && ctrl->print_level > 1)
            fprintf(stderr, "[smf_ma97] >>> ENTERING ma97_solve_d job=%d\n", job);
        std::memset(info, 0, sizeof(*info));

        auto* ak = static_cast<SmfAkeep*>(*akeep);
        auto* fw = static_cast<SmfFkeep*>(*fkeep);
        if (!ak || !fw)
        {
            info->flag = -5;
            return;
        }

        smf::SolveJob sjob = smf::SolveJob::Full;
        if (job == 1)
            sjob = smf::SolveJob::Forward;
        if (job == 2)
            sjob = smf::SolveJob::DiagOnly;
        if (job == 3)
            sjob = smf::SolveJob::Backward;
        if (job == 4)
            sjob = smf::SolveJob::DiagBack;

        const int n = ak->n;
        int rc = 0;

        maybe_capture_rhs(ak, fw, x, ldx, nrhs);

        // Debug: RHS norm (only if print_level > 1)
        if (ctrl && ctrl->print_level > 1)
        {
            double rhs_norm = 0.0;
            for (int i = 0; i < n * nrhs; ++i)
                rhs_norm += x[i] * x[i];
            fprintf(stderr, "[smf_ma97] solve: n=%d nrhs=%d rhs_norm=%.3e\n", n, nrhs, std::sqrt(rhs_norm));
        }

        if (ldx == n)
        {
            // Fast path: columns are contiguous
            rc = ak->solver.solve(fw->fk, ak->ctrl, ak->info, x, n, nrhs, sjob);
        }
        else
        {
            // General path: copy column-by-column into contiguous buffer
            std::vector<double> buf(static_cast<size_t>(n * nrhs));
            for (int j = 0; j < nrhs; ++j)
                std::copy(x + j * ldx, x + j * ldx + n, buf.data() + j * n);
            rc = ak->solver.solve(fw->fk, ak->ctrl, ak->info, buf.data(), n, nrhs, sjob);
            for (int j = 0; j < nrhs; ++j)
                std::copy(buf.data() + j * n, buf.data() + (j + 1) * n, x + j * ldx);
        }

        info->flag = (rc == 0) ? 0 : -5;
        info->num_neg = fw->num_neg;

        // Debug: solution norm (only if print_level > 1)
        if (ctrl && ctrl->print_level > 1)
        {
            double sol_norm = 0.0;
            for (int i = 0; i < n * nrhs; ++i)
                sol_norm += x[i] * x[i];
            fprintf(stderr, "[smf_ma97] solve: sol_norm=%.3e rc=%d\n", std::sqrt(sol_norm), rc);
        }
    }

    // ---------------------------------------------------------------------------
    // ma97_finalise_d  — free both akeep and fkeep
    // ---------------------------------------------------------------------------
    void ma97_finalise_d(void** akeep, void** fkeep)
    {
        if (fkeep && *fkeep)
        {
            delete static_cast<SmfFkeep*>(*fkeep);
            *fkeep = nullptr;
        }
        if (akeep && *akeep)
        {
            delete static_cast<SmfAkeep*>(*akeep);
            *akeep = nullptr;
        }
    }

    // ---------------------------------------------------------------------------
    // ma97_free_akeep_d  — free only akeep (called when reusing analysis)
    // ---------------------------------------------------------------------------
    void ma97_free_akeep_d(void** akeep)
    {
        if (akeep && *akeep)
        {
            delete static_cast<SmfAkeep*>(*akeep);
            *akeep = nullptr;
        }
    }

} // extern "C"
