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
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <memory>

/* -------------------------------------------------------------------------
 * Internal concrete structs for opaque handles
 * ------------------------------------------------------------------------- */

/**
 * Holds the result of smf::Solver::analyse() together with a copy of the
 * CscLower that was analysed (needed to supply updated values at factor time).
 */
struct smf_analysis_s {
    smf::CscLower                      csc;    ///< copy of the analysed matrix
    std::unique_ptr<smf::AnalysisKeep> keep;   ///< symbolic analysis result
    int                                n;      ///< matrix order (cached)
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
        h->csc   = std::move(A);
        h->keep  = std::move(keep);
        h->n     = csc->n;

        *out = h;
        return SMF_OK;

    } catch (...) {
        return SMF_ERR_INTERNAL;
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
        /* Update the stored CscLower with the new numerical values */
        smf::CscLower A = analysis->csc;   // shallow copy of structure
        const int nnz = static_cast<int>(A.col_ptr.back());
        A.values.assign(values, values + nnz);

        smf::Control ctrl;
        if (mtype == SMF_POSDEF) {
            ctrl.matrix_type = smf::MatrixType::RealSymmetricPositiveDefinite;
        } else {
            ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
        }

        smf::Info info;
        smf::Solver solver;

        auto fkeep = std::make_unique<smf::FactorKeep>();
        smf::FactorStatus fs =
            solver.factor(*analysis->keep, ctrl, info, *fkeep);

        if (fs == smf::FactorStatus::Singular) {
            return SMF_ERR_SINGULAR;
        }
        if (fs != smf::FactorStatus::Success) {
            return SMF_ERR_INTERNAL;
        }

        auto* h  = new smf_factor_s;
        h->keep  = std::move(fkeep);
        h->pos   = info.num_positive;
        h->neg   = info.num_negative;
        h->zero  = info.num_zero;

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

        int rc = solver.solve(*factor->keep, ctrl, info,
                              rhs, analysis->n, nrhs, sjob);
        if (rc != 0) {
            return SMF_ERR_INTERNAL;
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
