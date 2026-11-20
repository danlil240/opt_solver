/**
 * @file smf_c.cpp
 * @brief Pure C ABI implementation — wraps smf::Solver and smf::IpmSession.
 *
 * C++ exceptions are caught at every boundary; never let them propagate into C.
 */

#include "smf/smf_c.h"

#include "smf/analysis.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/ipm_session.hpp"
#include "smf/ipopt_plugin_control.hpp"
#include "smf/refinement.hpp"
#include "smf/slack_elimination.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#ifndef SMF_VERSION_STR
#  define SMF_VERSION_STR "0.1.0"
#endif
#ifndef SMF_BLAS_BACKEND_STR
#  define SMF_BLAS_BACKEND_STR "unknown"
#endif

/* -------------------------------------------------------------------------
 * Version & build introspection
 * ------------------------------------------------------------------------- */

int smf_c_abi_version(void) {
    return SMF_C_ABI_VERSION;
}

const char* smf_version(void) {
    return SMF_VERSION_STR;
}

const char* smf_blas_backend(void) {
    return SMF_BLAS_BACKEND_STR;
}

/* -------------------------------------------------------------------------
 * Internal concrete structs for simple handle API
 * ------------------------------------------------------------------------- */

struct smf_analysis_s {
    std::unique_ptr<smf::AnalysisKeep> keep;   ///< symbolic analysis result
    int                                n;      ///< matrix order (cached)
    std::size_t                        nnz;    ///< entries of the analysed matrix
    smf::Pivoting                      pivoting = smf::Pivoting::Threshold;
};

struct smf_factor_s {
    std::unique_ptr<smf::FactorKeep>   keep;   ///< numerical factor
    int                                pos;    ///< positive eigenvalue count
    int                                neg;    ///< negative eigenvalue count
    int                                zero;   ///< zero     eigenvalue count
    std::unique_ptr<smf::RefineMatrix> refine; ///< static-path refinement matrix
};

static int map_error(smf::ErrorCode ec) noexcept {
    switch (ec) {
    case smf::ErrorCode::Success:
        return SMF_OK;
    case smf::ErrorCode::SingularMatrix:
    case smf::ErrorCode::NotPositiveDefinite:
        return SMF_ERR_SINGULAR;
    case smf::ErrorCode::IllegalValue:
    case smf::ErrorCode::NotInitialised:
        return SMF_ERR_INVALID_ARG;
    default:
        return SMF_ERR_INTERNAL;
    }
}

/* -------------------------------------------------------------------------
 * Simple C API
 * ------------------------------------------------------------------------- */

int smf_analyse(const smf_csc_t* csc, smf_analysis_t* out) {
    if (!csc || !out) {
        return SMF_ERR_INVALID_ARG;
    }
    if (csc->n < 0 || !csc->col_ptr || !csc->row_idx || !csc->values) {
        return SMF_ERR_INVALID_ARG;
    }

    try {
        smf::CscLower A;
        A.n = static_cast<smf::Int>(csc->n);

        const int nnz = csc->col_ptr[csc->n];
        if (csc->col_ptr[0] != 0 || nnz < 0) {
            return SMF_ERR_INVALID_ARG;
        }
        A.col_ptr.assign(csc->col_ptr, csc->col_ptr + csc->n + 1);
        A.row_idx.assign(csc->row_idx, csc->row_idx + nnz);
        A.values.assign(csc->values, csc->values + nnz);

        smf::Control ctrl;
        ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;

        smf::Info info;
        smf::Solver solver;

        auto keep = solver.analyse(A, ctrl, info);
        if (!keep) {
            return map_error(info.status);
        }

        auto* h = new smf_analysis_s;
        h->keep = std::move(keep);
        h->n    = csc->n;
        h->nnz  = static_cast<std::size_t>(nnz);

        *out = h;
        return SMF_OK;
    } catch (...) {
        return SMF_ERR_INTERNAL;
    }
}

int smf_set_pivoting(smf_analysis_t analysis, int pivoting) {
    if (!analysis) {
        return SMF_ERR_INVALID_ARG;
    }
    switch (pivoting) {
    case SMF_PIVOTING_THRESHOLD:
        analysis->pivoting = smf::Pivoting::Threshold;
        return SMF_OK;
    case SMF_PIVOTING_STATIC:
        analysis->pivoting = smf::Pivoting::Static;
        return SMF_OK;
    default:
        return SMF_ERR_INVALID_ARG;
    }
}

int smf_factor(smf_analysis_t analysis,
               const double*  values,
               int            mtype,
               smf_factor_t*  out) {
    if (!analysis || !values || !out) {
        return SMF_ERR_INVALID_ARG;
    }
    if (mtype != SMF_POSDEF && mtype != SMF_INDEF) {
        return SMF_ERR_INVALID_ARG;
    }

    try {
        smf::Control ctrl;
        if (mtype == SMF_POSDEF) {
            ctrl.matrix_type = smf::MatrixType::RealSymmetricPositiveDefinite;
        } else {
            ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
            ctrl.pivoting = analysis->pivoting;
        }

        smf::Info info;
        smf::Solver solver;
        smf::NumericWorkspace ws;

        auto fkeep = std::make_unique<smf::FactorKeep>();
        const smf::FactorStatus fs = solver.factor(
            *analysis->keep, std::span<const double>(values, analysis->nnz),
            ctrl, info, *fkeep, ws);

        if (fs != smf::FactorStatus::Success) {
            if (info.status == smf::ErrorCode::IllegalValue) {
                return SMF_ERR_INVALID_ARG;
            }
            if (fs == smf::FactorStatus::Singular ||
                fs == smf::FactorStatus::NotPositiveDefinite) {
                return SMF_ERR_SINGULAR;
            }
            return SMF_ERR_INTERNAL;
        }

        std::unique_ptr<smf::RefineMatrix> refine;
        if (fkeep->static_path && smf::refinement_advised(*fkeep)) {
            const smf::AnalysisKeep& ak = *analysis->keep;
            std::vector<double> cv;
            if (ak.orig_to_clean_idx.empty()) {
                cv.assign(values, values + analysis->nnz);
            } else {
                if (ak.orig_to_clean_idx.size() != analysis->nnz) {
                    return SMF_ERR_INTERNAL;
                }
                cv.assign(ak.cleaned.row_idx.size(), 0.0);
                for (std::size_t k = 0; k < analysis->nnz; ++k) {
                    const smf::Int c = ak.orig_to_clean_idx[k];
                    if (c >= 0) cv[static_cast<std::size_t>(c)] += values[k];
                }
            }
            refine = std::make_unique<smf::RefineMatrix>();
            if (smf::prepare_refine_matrix(ak.cleaned, *refine) != smf::ErrorCode::Success ||
                cv.size() != refine->slot.size()) {
                return SMF_ERR_INTERNAL;
            }
            smf::set_refine_values(cv, *refine);
        }

        auto* h   = new smf_factor_s;
        h->keep   = std::move(fkeep);
        h->pos    = info.num_positive;
        h->neg    = info.num_negative;
        h->zero   = info.num_zero;
        h->refine = std::move(refine);

        *out = h;
        return SMF_OK;
    } catch (...) {
        return SMF_ERR_INTERNAL;
    }
}

int smf_solve(smf_analysis_t analysis,
              smf_factor_t   factor,
              int            nrhs,
              double*        rhs,
              int            job) {
    if (!analysis || !factor || !rhs || nrhs <= 0) {
        return SMF_ERR_INVALID_ARG;
    }

    smf::SolveJob sjob;
    switch (job) {
    case SMF_SOLVE_FULL:     sjob = smf::SolveJob::Full;     break;
    case SMF_SOLVE_FORWARD:  sjob = smf::SolveJob::Forward;  break;
    case SMF_SOLVE_DIAGONAL: sjob = smf::SolveJob::DiagOnly; break;
    case SMF_SOLVE_BACKWARD: sjob = smf::SolveJob::Backward; break;
    default: return SMF_ERR_INVALID_ARG;
    }

    try {
        smf::Control ctrl;
        smf::Info    info;
        smf::Solver  solver;

        const bool refine = factor->refine != nullptr && sjob == smf::SolveJob::Full &&
                            factor->refine->n == analysis->n;
        const std::size_t len = static_cast<std::size_t>(analysis->n > 0 ? analysis->n : 0) *
                                static_cast<std::size_t>(nrhs);
        std::vector<double> b;
        if (refine) b.assign(rhs, rhs + len);

        int rc = solver.solve(*factor->keep, ctrl, info,
                              rhs, analysis->n, nrhs, sjob);
        if (rc != 0) {
            return info.status == smf::ErrorCode::IllegalValue
                       ? SMF_ERR_INVALID_ARG
                       : SMF_ERR_INTERNAL;
        }
        if (refine) {
            smf::SolveWorkspace ws;
            smf::refine(*factor->refine, *factor->keep, ctrl, b,
                        std::span<double>(rhs, len), nrhs, ws);
        }
        return SMF_OK;
    } catch (...) {
        return SMF_ERR_INTERNAL;
    }
}

int smf_inertia(smf_factor_t factor, smf_inertia_t* inertia) {
    if (!factor || !inertia) {
        return SMF_ERR_INVALID_ARG;
    }
    inertia->pos  = factor->pos;
    inertia->neg  = factor->neg;
    inertia->zero = factor->zero;
    return SMF_OK;
}

void smf_free_analysis(smf_analysis_t* handle) {
    if (!handle || !*handle) return;
    delete *handle;
    *handle = nullptr;
}

void smf_free_factor(smf_factor_t* handle) {
    if (!handle || !*handle) return;
    delete *handle;
    *handle = nullptr;
}

/* -------------------------------------------------------------------------
 * Interior-point session C API (smf_ipm_*)
 * ------------------------------------------------------------------------- */

struct smf_ipm_session_s {
    smf::IpmSession      session;
    smf::Ma97CascadeState cascade;
    smf::CscLower        mat;
    std::vector<int>     sort_perm;     ///< empty when input was already sorted lower-CSC
    std::vector<double>  rhs_pack;      ///< scratch when ldx > n
    smf::Control         last_ctrl;
    int                  n = 0;
    std::size_t          input_nnz = 0;
    bool                 analysed = false;
    bool                 factored = false;
};

static void reset_ipm_info(smf_ipm_info_t* info, int status) noexcept {
    if (!info) return;
    std::memset(info, 0, sizeof(*info));
    info->struct_size = static_cast<int>(sizeof(smf_ipm_info_t));
    info->status = status;
}

void smf_ipm_default_control(smf_ipm_control_t* ctrl) {
    if (!ctrl) return;
    std::memset(ctrl, 0, sizeof(*ctrl));
    ctrl->abi_version          = SMF_C_ABI_VERSION;
    ctrl->struct_size          = static_cast<int>(sizeof(smf_ipm_control_t));
    ctrl->ordering             = SMF_ORDER_AUTO;
    ctrl->scaling              = SMF_SCALE_AUTO;
    ctrl->pivoting             = SMF_PIVOTING_THRESHOLD;
    ctrl->slack_elimination    = SMF_SLACK_OFF;
    ctrl->nemin                = 8;
    ctrl->num_threads          = 0;
    ctrl->deterministic        = 0;
    ctrl->continue_on_singular = 1;
    ctrl->honest_inertia       = 1;
    ctrl->print_level          = -1;
    ctrl->u                    = 1e-8;
    ctrl->small                = 1e-20;
    ctrl->multiplier           = 1.2;
}

static bool validate_and_apply_ipm_control(const smf_ipm_control_t* c,
                                           int mtype,
                                           smf::Control& out) noexcept {
    smf_ipm_control_t def;
    if (!c) {
        smf_ipm_default_control(&def);
        c = &def;
    }
    if (c->abi_version != SMF_C_ABI_VERSION ||
        c->struct_size < static_cast<int>(sizeof(smf_ipm_control_t))) {
        return false;
    }
    if (!std::isfinite(c->u) || c->u < 0.0 || c->u > 0.5) return false;
    if (!std::isfinite(c->small) || c->small <= 0.0) return false;
    if (!std::isfinite(c->multiplier) || c->multiplier <= 0.0) return false;
    if (c->nemin <= 0 || c->num_threads < 0) return false;

    out = smf::Control{};
    out.matrix_type = (mtype == SMF_POSDEF)
                          ? smf::MatrixType::RealSymmetricPositiveDefinite
                          : smf::MatrixType::RealSymmetricIndefinite;
    out.pivot_u = c->u;
    out.small_pivot = c->small;
    out.nemin = c->nemin;
    out.factor_memory_multiplier = c->multiplier;
    out.continue_on_singular = (c->continue_on_singular != 0);
    out.deterministic = (c->deterministic != 0);
    out.print_level = c->print_level;
    out.num_threads = (c->num_threads > 0) ? c->num_threads : smf::plugin_num_threads();

    switch (c->ordering) {
    case SMF_ORDER_AUTO:     out.ordering = smf::OrderingMethod::AutoParallel; break;
    case SMF_ORDER_AMD:      out.ordering = smf::OrderingMethod::AMD;          break;
    case SMF_ORDER_METIS:    out.ordering = smf::OrderingMethod::METIS;        break;
    case SMF_ORDER_MATCHING: out.ordering = smf::OrderingMethod::Matching;     break;
    default: return false;
    }

    switch (c->scaling) {
    case SMF_SCALE_NONE:          out.scaling = smf::ScalingMethod::None;          break;
    case SMF_SCALE_MATCHING:      out.scaling = smf::ScalingMethod::Matching;      break;
    case SMF_SCALE_EQUILIBRATION: out.scaling = smf::ScalingMethod::Equilibration; break;
    case SMF_SCALE_AUTO:          out.scaling = smf::ScalingMethod::Auto;          break;
    default: return false;
    }

    switch (c->pivoting) {
    case SMF_PIVOTING_THRESHOLD: out.pivoting = smf::Pivoting::Threshold; break;
    case SMF_PIVOTING_STATIC:    out.pivoting = smf::Pivoting::Static;    break;
    default: return false;
    }

    switch (c->slack_elimination) {
    case SMF_SLACK_OFF:  out.slack_elimination = smf::SlackElimination::Off;  break;
    case SMF_SLACK_ON:   out.slack_elimination = smf::SlackElimination::On;   break;
    case SMF_SLACK_AUTO: out.slack_elimination = smf::SlackElimination::Auto; break;
    default: return false;
    }

    return true;
}

static bool build_ipm_csc_pattern(smf_ipm_session_s* s,
                                  int n,
                                  const int* col_ptr,
                                  const int* row_idx) {
    if (n <= 0 || !col_ptr || !row_idx) return false;
    if (col_ptr[0] != 0) return false;
    const int nnz = col_ptr[n];
    if (nnz < 0) return false;

    // Fast path check: already valid 0-based lower-triangular CSC with
    // non-decreasing row indices per column (IPOPT CSR_Format_0_Offset).
    bool already_lower_sorted = true;
    for (int j = 0; j < n; ++j) {
        const int start = col_ptr[j];
        const int end   = col_ptr[j + 1];
        if (start < 0 || end < start || end > nnz) return false;
        int prev_r = -1;
        for (int p = start; p < end; ++p) {
            const int r = row_idx[p];
            if (r < 0 || r >= n) return false;
            if (r < j || r < prev_r) {
                already_lower_sorted = false;
            }
            prev_r = r;
        }
    }

    s->n = n;
    s->input_nnz = static_cast<std::size_t>(nnz);
    s->mat.n = static_cast<smf::Int>(n);
    s->sort_perm.clear();

    if (already_lower_sorted) {
        s->mat.col_ptr.assign(col_ptr, col_ptr + n + 1);
        s->mat.row_idx.assign(row_idx, row_idx + nnz);
        s->mat.values.assign(static_cast<std::size_t>(nnz), 0.0);
        return true;
    }

    // Slow path: normalize to the lower triangle and sort rows per column.
    // An entry above the diagonal (row < col) is mirrored, so an upper-
    // triangle input works, unless the lower triangle holds that position
    // too: a matrix passed with both triangles keeps only its lower copy
    // (duplicates are summed, so keeping both would double it).
    struct Entry {
        int r;
        int c;
        int orig;
        bool mirrored;
    };
    std::vector<Entry> entries;
    entries.reserve(static_cast<std::size_t>(nnz));
    for (int j = 0; j < n; ++j) {
        for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            const int r = row_idx[p];
            if (r < j) entries.push_back({j, r, p, true});
            else       entries.push_back({r, j, p, false});
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) noexcept {
        if (a.c != b.c) return a.c < b.c;
        if (a.r != b.r) return a.r < b.r;
        return a.mirrored < b.mirrored;  // lower copies first
    });
    {
        std::size_t out = 0;
        for (std::size_t k = 0; k < entries.size(); ++k) {
            const Entry& e = entries[k];
            // The group of (r, c) starts with its lower copies, if any.
            if (e.mirrored && out > 0 && !entries[out - 1].mirrored &&
                entries[out - 1].r == e.r && entries[out - 1].c == e.c) {
                continue;
            }
            entries[out++] = e;
        }
        entries.resize(out);
    }

    const std::size_t m = entries.size();
    s->mat.col_ptr.assign(static_cast<std::size_t>(n + 1), 0);
    s->mat.row_idx.resize(m);
    s->mat.values.assign(m, 0.0);
    s->sort_perm.resize(m);
    for (const auto& e : entries) {
        s->mat.col_ptr[static_cast<std::size_t>(e.c + 1)]++;
    }
    for (int j = 0; j < n; ++j) {
        s->mat.col_ptr[static_cast<std::size_t>(j + 1)] +=
            s->mat.col_ptr[static_cast<std::size_t>(j)];
    }
    for (std::size_t k = 0; k < m; ++k) {
        s->mat.row_idx[k] = static_cast<smf::Int>(entries[k].r);
        s->sort_perm[k]   = entries[k].orig;
    }
    return true;
}

static void fill_ipm_values(smf_ipm_session_s* s, const double* values) noexcept {
    if (s->sort_perm.empty()) {
        std::copy(values, values + s->mat.values.size(), s->mat.values.begin());
    } else {
        const std::size_t m = s->sort_perm.size();
        for (std::size_t k = 0; k < m; ++k) {
            s->mat.values[k] = values[static_cast<std::size_t>(s->sort_perm[k])];
        }
    }
}

int smf_ipm_create(smf_ipm_session_t* out) {
    if (!out) return SMF_ERR_INVALID_ARG;
    try {
        *out = new smf_ipm_session_s();
        return SMF_OK;
    } catch (...) {
        *out = nullptr;
        return SMF_ERR_INTERNAL;
    }
}

int smf_ipm_analyse(smf_ipm_session_t        session,
                    int                      n,
                    const int*               col_ptr,
                    const int*               row_idx,
                    const double*            values,
                    const smf_ipm_control_t* ctrl,
                    smf_ipm_info_t*          info) {
    reset_ipm_info(info, SMF_ERR_INVALID_ARG);
    if (!session) return SMF_ERR_INVALID_ARG;

    try {
        smf::Control smf_ctrl;
        if (!validate_and_apply_ipm_control(ctrl, SMF_INDEF, smf_ctrl)) {
            return SMF_ERR_INVALID_ARG;
        }
        if (!build_ipm_csc_pattern(session, n, col_ptr, row_idx)) {
            return SMF_ERR_INVALID_ARG;
        }
        session->cascade = smf::Ma97CascadeState{};
        session->last_ctrl = smf_ctrl;
        session->analysed = false;
        session->factored = false;

        std::span<const double> val_span;
        if (values) {
            fill_ipm_values(session, values);
            val_span = session->mat.values;
        }

        const smf::BlasThreadGuard blas(smf_ctrl.num_threads);
        const smf::IpmControl ipm_ctrl{};
        const smf::ErrorCode ec = session->session.analyse(session->mat, val_span, smf_ctrl, ipm_ctrl);
        if (ec != smf::ErrorCode::Success) {
            const int rc = map_error(ec);
            if (info) info->status = rc;
            return rc;
        }

        session->analysed = true;
        if (info) {
            info->status = SMF_OK;
            info->matrix_rank = n;
            if (const smf::AnalysisKeep* ak = session->session.analysis_keep()) {
                const smf::Info& si = session->session.info();
                info->num_sup         = static_cast<int>(ak->supernodes.size());
                info->maxdepth        = si.max_tree_depth;
                info->maxfront        = si.max_front_size;
                info->maxsupernode    = si.max_supernode_size;
                info->ordering_used   = si.ordering_used;
                info->analyse_seconds = si.analyse_seconds;
            }
        }
        return SMF_OK;
    } catch (...) {
        reset_ipm_info(info, SMF_ERR_INTERNAL);
        return SMF_ERR_INTERNAL;
    }
}

int smf_ipm_factor(smf_ipm_session_t        session,
                   int                      mtype,
                   const double*            values,
                   const smf_ipm_control_t* ctrl,
                   smf_ipm_info_t*          info,
                   double*                  scale) {
    reset_ipm_info(info, SMF_ERR_INVALID_ARG);
    if (!session || !session->analysed || !values) return SMF_ERR_INVALID_ARG;
    if (mtype != SMF_POSDEF && mtype != SMF_INDEF) return SMF_ERR_INVALID_ARG;

    try {
        smf::Control smf_ctrl;
        if (!validate_and_apply_ipm_control(ctrl, mtype, smf_ctrl)) {
            return SMF_ERR_INVALID_ARG;
        }
        const bool honest_inertia = (!ctrl || ctrl->honest_inertia != 0);

        // Caller-managed scaling (dynamic/reuse strategies) vs native Auto.
        bool unit_scale = false;
        bool computed_scaling = false;
        bool reuse_scaling = false;
        bool cascade_honoured = false;

        if (scale != nullptr && smf_ctrl.scaling != smf::ScalingMethod::Auto) {
            const smf::AnalysisKeep* analysed = session->session.analysis_keep();
            const bool rejected = analysed != nullptr &&
                                  (smf_ctrl.scaling == smf::ScalingMethod::Matching
                                       ? smf::slack_full_analysis(*analysed).matching_rejected
                                       : analysed->matching_rejected);
            const smf::Ma97ScalingCall sc =
                smf::ma97_scaling_call(smf_ctrl, rejected, scale, session->n,
                                       session->cascade.cascade());
            unit_scale       = sc.unit_scale;
            computed_scaling = sc.computed;
            reuse_scaling    = sc.reuse;
            cascade_honoured = sc.cascade_honoured;
        }

        fill_ipm_values(session, values);
        smf::Control fc = smf_ctrl;
        if (computed_scaling) fc.slack_elimination = smf::SlackElimination::Off;
        session->session.set_control(fc);
        if (reuse_scaling) {
            session->session.set_user_scale(scale);
        }

        const bool analysed_before = (session->session.analysis_keep() != nullptr);
        const smf::BlasThreadGuard blas(smf_ctrl.num_threads);
        const auto t0 = std::chrono::steady_clock::now();
        smf::IpmFactorStatus status = session->session.factor(session->mat.values);

        // Stale-scale retry: if a reused caller scale causes singularity,
        // recompute a fresh matching scale once and return it in scale[].
        if (reuse_scaling && status == smf::IpmFactorStatus::Singular) {
            smf::Control fresh = smf_ctrl;
            fresh.scaling = smf::ScalingMethod::Matching;
            smf::Control ffc = fresh;
            ffc.slack_elimination = smf::SlackElimination::Off;
            session->session.set_control(ffc);
            const smf::IpmFactorStatus retried = session->session.factor(session->mat.values);
            if (retried == smf::IpmFactorStatus::Ok || retried == smf::IpmFactorStatus::Singular) {
                status = retried;
                smf_ctrl = fresh;
                computed_scaling = true;
            } else {
                session->session.set_control(smf_ctrl);
                session->session.set_user_scale(scale);
                status = session->session.factor(session->mat.values);
            }
        }
        const double elapsed_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        const smf::Info& si = session->session.info();
        if (session->session.analysis_keep() == nullptr) {
            reset_ipm_info(info, SMF_ERR_INTERNAL);
            return SMF_ERR_INTERNAL;
        }

        const double analyse_s = analysed_before ? 0.0 : si.analyse_seconds;
        const double factor_s  = std::max(0.0, elapsed_s - analyse_s);

        if (status == smf::IpmFactorStatus::Ok || status == smf::IpmFactorStatus::Singular) {
            const smf::AnalysisKeep& fak = *session->session.analysis_keep();
            const double free_work = session->cascade.free_work_of(
                fak.serial, [&fak] { return smf::delay_free_front_work(fak); });
            session->cascade.after_factor(si.front_work, free_work,
                                          cascade_honoured && computed_scaling);
        }

        session->last_ctrl = smf_ctrl;
        const smf::FactorKeep& fk = session->session.factor_keep();
        if (computed_scaling && scale && !fk.scale.empty()) {
            std::copy(fk.scale.begin(), fk.scale.end(), scale);
        } else if (unit_scale && scale) {
            std::fill(scale, scale + session->n, 1.0);
        }

        const bool pd = (smf_ctrl.matrix_type == smf::MatrixType::RealSymmetricPositiveDefinite) &&
                        (si.factor_status == smf::FactorStatus::Success);
        const int num_pos  = pd ? session->n : si.num_positive;
        const int num_neg  = pd ? 0          : si.num_negative;
        const int num_zero = pd ? 0          : si.num_zero;

        int rc = SMF_OK;
        if (status == smf::IpmFactorStatus::WrongInertia) {
            rc = SMF_ERR_WRONG_INERTIA;
        } else if (status == smf::IpmFactorStatus::Error) {
            rc = (si.status == smf::ErrorCode::IllegalValue) ? SMF_ERR_INVALID_ARG : SMF_ERR_INTERNAL;
        } else if (status == smf::IpmFactorStatus::Singular || num_zero > 0) {
            rc = SMF_ERR_SINGULAR;
        } else if (honest_inertia && !si.inertia_exact) {
            rc = SMF_ERR_INEXACT_INERTIA;
        }

        session->factored = (rc == SMF_OK || rc == SMF_ERR_WRONG_INERTIA || rc == SMF_ERR_SINGULAR);

        if (info) {
            info->status          = rc;
            info->num_pos         = num_pos;
            info->num_neg         = num_neg;
            info->num_zero        = num_zero;
            info->matrix_rank     = session->n - num_zero;
            info->num_delay       = si.delayed_pivots;
            info->num_two         = si.two_by_two_pivots;
            info->maxfront        = si.max_front_size;
            info->maxsupernode    = si.max_supernode_size;
            info->maxdepth        = si.max_tree_depth;
            info->num_sup         = static_cast<int>(session->session.analysis_keep()->supernodes.size());
            info->ordering_used   = si.ordering_used;
            info->inertia_exact   = si.inertia_exact ? 1 : 0;
            info->threads_used    = si.threads_used;
            info->static_pivots   = si.static_pivots;
            info->num_factor      = static_cast<int64_t>(si.actual_factor_entries);
            info->num_flops       = static_cast<int64_t>(si.actual_flops);
            info->analyse_seconds = analyse_s;
            info->factor_seconds  = factor_s;
        }
        return rc;
    } catch (...) {
        reset_ipm_info(info, SMF_ERR_INTERNAL);
        return SMF_ERR_INTERNAL;
    }
}

int smf_ipm_solve(smf_ipm_session_t        session,
                  int                      job,
                  int                      nrhs,
                  double*                  rhs,
                  int                      ldx,
                  const smf_ipm_control_t* ctrl,
                  smf_ipm_info_t*          info) {
    reset_ipm_info(info, SMF_ERR_INVALID_ARG);
    if (!session || !session->factored || !rhs || nrhs <= 0 || ldx < session->n) {
        return SMF_ERR_INVALID_ARG;
    }

    smf::SolveJob sjob;
    switch (job) {
    case SMF_SOLVE_FULL:     sjob = smf::SolveJob::Full;     break;
    case SMF_SOLVE_FORWARD:  sjob = smf::SolveJob::Forward;  break;
    case SMF_SOLVE_DIAGONAL: sjob = smf::SolveJob::DiagOnly; break;
    case SMF_SOLVE_BACKWARD: sjob = smf::SolveJob::Backward; break;
    default: return SMF_ERR_INVALID_ARG;
    }

    try {
        smf::Control solve_ctrl = session->last_ctrl;
        bool restore_ctrl = false;
        if (ctrl) {
            const int mtype = (solve_ctrl.matrix_type == smf::MatrixType::RealSymmetricPositiveDefinite)
                                  ? SMF_POSDEF
                                  : SMF_INDEF;
            if (!validate_and_apply_ipm_control(ctrl, mtype, solve_ctrl)) {
                return SMF_ERR_INVALID_ARG;
            }
            session->session.set_control(solve_ctrl);
            restore_ctrl = true;
        }
        struct ControlRestore {
            smf::IpmSession& sess;
            const smf::Control& saved;
            bool active;
            ~ControlRestore() {
                if (active) sess.set_control(saved);
            }
        } ctrl_restore{session->session, session->last_ctrl, restore_ctrl};

        const smf::BlasThreadGuard blas(solve_ctrl.num_threads);
        const auto t0 = std::chrono::steady_clock::now();
        int rc = 0;
        const int n = session->n;
        if (ldx == n || nrhs == 1) {
            rc = session->session.solve(rhs, n, nrhs, sjob);
        } else {
            session->rhs_pack.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs));
            for (int r = 0; r < nrhs; ++r) {
                std::copy(rhs + static_cast<std::size_t>(r) * static_cast<std::size_t>(ldx),
                          rhs + static_cast<std::size_t>(r) * static_cast<std::size_t>(ldx) + n,
                          session->rhs_pack.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(n));
            }
            rc = session->session.solve(session->rhs_pack.data(), n, nrhs, sjob);
            if (rc == 0) {
                for (int r = 0; r < nrhs; ++r) {
                    std::copy(session->rhs_pack.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(n),
                              session->rhs_pack.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(n) + n,
                              rhs + static_cast<std::size_t>(r) * static_cast<std::size_t>(ldx));
                }
            }
        }
        const double solve_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        if (rc != 0) {
            reset_ipm_info(info, SMF_ERR_INTERNAL);
            return SMF_ERR_INTERNAL;
        }
        if (info) {
            info->status        = SMF_OK;
            info->solve_seconds = solve_s;
        }
        return SMF_OK;
    } catch (...) {
        reset_ipm_info(info, SMF_ERR_INTERNAL);
        return SMF_ERR_INTERNAL;
    }
}

void smf_ipm_free(smf_ipm_session_t* session) {
    if (!session || !*session) return;
    delete *session;
    *session = nullptr;
}
