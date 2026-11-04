/**
 * @file smf_c.cpp
 * @brief C ABI implementation — wraps smf::Solver, CscLower, Control, Info.
 *
 * Compiled only when SMF_BUILD_C_API=1.
 * C++ exceptions are caught at every boundary; never let them propagate into C.
 */

#ifdef SMF_BUILD_C_API

#include "smf/smf_c.h"

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/refinement.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

/* -------------------------------------------------------------------------
 * Internal concrete structs for opaque handles
 * ------------------------------------------------------------------------- */

/**
 * Holds the result of smf::Solver::analyse() together with the order and
 * entry count of the analysed matrix. smf_factor() passes its values to
 * smf::Solver::factor() in the entry order of that matrix.
 */
struct smf_analysis_s {
    std::unique_ptr<smf::AnalysisKeep> keep;   ///< symbolic analysis result
    int                                n;      ///< matrix order (cached)
    std::size_t                        nnz;    ///< entries of the analysed matrix
    /// The analysed pattern (values zeroed, kept only so the shape checks
    /// pass): a static-pivoting factor's refinement matrix is built from it
    /// (smf_set_pivoting).
    smf::CscLower                      pattern;
    /// smf_set_pivoting(): Control::pivoting of the SMF_INDEF factors.
    smf::Pivoting                      pivoting = smf::Pivoting::Threshold;
};

/**
 * Holds the result of smf::Solver::factor() together with the inertia
 * obtained from smf::Info after factorisation.
 */
struct smf_factor_s {
    std::unique_ptr<smf::FactorKeep> keep;     ///< numerical factor
    int                              pos;      ///< positive eigenvalue count
    int                              neg;      ///< negative eigenvalue count
    int                              zero;     ///< zero     eigenvalue count
    /// A static-path factor (SMF_PIVOTING_STATIC): the factored matrix that
    /// smf_solve() refines SMF_SOLVE_FULL solves against. Null otherwise.
    std::unique_ptr<smf::RefineMatrix> refine;
};

/* -------------------------------------------------------------------------
 * Helper: map smf::ErrorCode → SMF_* constants
 * ------------------------------------------------------------------------- */

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
 * smf_analyse
 * ------------------------------------------------------------------------- */

int smf_analyse(const smf_csc_t* csc, smf_analysis_t* out) {
    if (!csc || !out) {
        return SMF_ERR_INVALID_ARG;
    }
    if (csc->n < 0 || !csc->col_ptr || !csc->row_idx || !csc->values) {
        return SMF_ERR_INVALID_ARG;
    }

    try {
        /* Build a CscLower from the caller-supplied arrays */
        smf::CscLower A;
        A.n = static_cast<smf::Int>(csc->n);

        const int nnz = csc->col_ptr[csc->n];
        if (csc->col_ptr[0] != 0 || nnz < 0) {
            return SMF_ERR_INVALID_ARG;
        }
        A.col_ptr.assign(csc->col_ptr, csc->col_ptr + csc->n + 1);
        A.row_idx.assign(csc->row_idx, csc->row_idx + nnz);
        A.values.assign (csc->values,  csc->values  + nnz);

        smf::Control ctrl;
        ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;

        smf::Info info;
        smf::Solver solver;

        auto keep = solver.analyse(A, ctrl, info);
        if (!keep) {
            return map_error(info.status);
        }

        auto* h  = new smf_analysis_s;
        h->keep  = std::move(keep);
        h->n     = csc->n;
        h->nnz   = static_cast<std::size_t>(nnz);
        A.values.assign(A.row_idx.size(), 0.0);  /* the pattern only (shape-valid) */
        A.values.shrink_to_fit();
        h->pattern = std::move(A);

        *out = h;
        return SMF_OK;

    } catch (...) {
        return SMF_ERR_INTERNAL;
    }
}

/* -------------------------------------------------------------------------
 * smf_set_pivoting
 * ------------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------------
 * smf_factor
 * ------------------------------------------------------------------------- */

int smf_factor(smf_analysis_t analysis,
               const double*  values,
               int            mtype,
               smf_factor_t*  out)
{
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

        /* Factor the caller's values (entry order of the analysed matrix).
         * The overload without values factors the values given to
         * smf_analyse(), which ignored `values` here before. */
        auto fkeep = std::make_unique<smf::FactorKeep>();
        const smf::FactorStatus fs = solver.factor(
            *analysis->keep, std::span<const double>(values, analysis->nnz),
            ctrl, info, *fkeep, ws);

        if (fs != smf::FactorStatus::Success) {
            /* Rejected values (NaN/Inf) come back as Singular or
             * NotPositiveDefinite with info.status = IllegalValue. */
            if (info.status == smf::ErrorCode::IllegalValue) {
                return SMF_ERR_INVALID_ARG;
            }
            if (fs == smf::FactorStatus::Singular ||
                fs == smf::FactorStatus::NotPositiveDefinite) {
                return SMF_ERR_SINGULAR;
            }
            return SMF_ERR_INTERNAL;
        }

        /* Fast mode: keep the factored matrix for smf_solve()'s refinement
         * (the backward-error trigger of a static-path factor that forced a
         * pivot, smf::refinement_advised). */
        std::unique_ptr<smf::RefineMatrix> refine;
        if (fkeep->static_path && smf::refinement_advised(*fkeep)) {
            refine = std::make_unique<smf::RefineMatrix>();
            if (smf::prepare_refine_matrix(analysis->pattern, *refine) != smf::ErrorCode::Success) {
                return SMF_ERR_INTERNAL;
            }
            smf::set_refine_values(std::span<const double>(values, analysis->nnz), *refine);
        }

        auto* h  = new smf_factor_s;
        h->keep  = std::move(fkeep);
        h->pos   = info.num_positive;
        h->neg   = info.num_negative;
        h->zero  = info.num_zero;
        h->refine = std::move(refine);

        *out = h;
        return SMF_OK;

    } catch (...) {
        return SMF_ERR_INTERNAL;
    }
}

/* -------------------------------------------------------------------------
 * smf_solve
 * ------------------------------------------------------------------------- */

int smf_solve(smf_analysis_t analysis,
              smf_factor_t   factor,
              int            nrhs,
              double*        rhs,
              int            job)
{
    if (!analysis || !factor || !rhs || nrhs <= 0) {
        return SMF_ERR_INVALID_ARG;
    }

    smf::SolveJob sjob;
    switch (job) {
    case SMF_SOLVE_FULL:     sjob = smf::SolveJob::Full;     break;
    case SMF_SOLVE_FORWARD:  sjob = smf::SolveJob::Forward;  break;
    case SMF_SOLVE_BACKWARD: sjob = smf::SolveJob::Backward; break;
    default: return SMF_ERR_INVALID_ARG;
    }

    try {
        smf::Control ctrl;
        smf::Info    info;
        smf::Solver  solver;

        /* A static-path factor (SMF_PIVOTING_STATIC): refine a full solve
         * against the factored matrix (Control::refine_omega_target). */
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

/* -------------------------------------------------------------------------
 * smf_inertia
 * ------------------------------------------------------------------------- */

int smf_inertia(smf_factor_t factor, smf_inertia_t* inertia) {
    if (!factor || !inertia) {
        return SMF_ERR_INVALID_ARG;
    }
    inertia->pos  = factor->pos;
    inertia->neg  = factor->neg;
    inertia->zero = factor->zero;
    return SMF_OK;
}

/* -------------------------------------------------------------------------
 * smf_free_analysis / smf_free_factor
 * ------------------------------------------------------------------------- */

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

#endif /* SMF_BUILD_C_API */
