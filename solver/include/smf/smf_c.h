/**
 * @file smf_c.h
 * @brief Pure C ABI for the smf sparse symmetric solver.
 *
 * Gate: compiled only when SMF_BUILD_C_API=1 is defined.
 * All integer indices are 0-based.  All dense arrays are column-major.
 */
#pragma once

#ifdef SMF_BUILD_C_API

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* -------------------------------------------------------------------------
 * Opaque handles
 * ------------------------------------------------------------------------- */

/** Opaque handle returned by smf_analyse(). */
typedef struct smf_analysis_s* smf_analysis_t;

/** Opaque handle returned by smf_factor(). */
typedef struct smf_factor_s*   smf_factor_t;

/* -------------------------------------------------------------------------
 * Error codes  (match smf::ErrorCode integer values where applicable)
 * ------------------------------------------------------------------------- */

#define SMF_OK               0   /**< Success                          */
#define SMF_ERR_INVALID_ARG  1   /**< Null pointer or bad argument     */
#define SMF_ERR_SINGULAR     2   /**< Matrix is singular               */
#define SMF_ERR_INTERNAL     3   /**< Unexpected internal error        */

/* -------------------------------------------------------------------------
 * Matrix type  (passed to smf_factor)
 * ------------------------------------------------------------------------- */

#define SMF_POSDEF    0   /**< Real symmetric positive-definite */
#define SMF_INDEF     1   /**< Real symmetric indefinite        */

/* -------------------------------------------------------------------------
 * Solve job  (passed to smf_solve)
 * ------------------------------------------------------------------------- */

#define SMF_SOLVE_FULL     0   /**< Full AX = B solve                       */
#define SMF_SOLVE_FORWARD  1   /**< Forward sweep only   (PLX = SB)         */
#define SMF_SOLVE_BACKWARD 3   /**< Backward sweep only  ((PL)^T S^{-1}X=B) */

/* -------------------------------------------------------------------------
 * Inertia descriptor
 * ------------------------------------------------------------------------- */

/** Counts of positive, negative, and zero eigenvalues from the last factor. */
typedef struct {
    int pos;   /**< Number of positive eigenvalues */
    int neg;   /**< Number of negative eigenvalues */
    int zero;  /**< Number of zero     eigenvalues */
} smf_inertia_t;

/* -------------------------------------------------------------------------
 * CSC matrix descriptor  (caller owns all arrays; not copied here)
 * ------------------------------------------------------------------------- */

/**
 * Compressed-sparse-column descriptor for a symmetric matrix stored as the
 * *lower triangle* (row_idx[k] >= column index for every entry).
 * All indices are 0-based.
 */
typedef struct {
    int           n;        /**< Matrix order (number of rows/columns)   */
    const int*    col_ptr;  /**< Column pointers, length n+1             */
    const int*    row_idx;  /**< Row indices,    length col_ptr[n]       */
    const double* values;   /**< Numerical values, length col_ptr[n]     */
} smf_csc_t;

/* -------------------------------------------------------------------------
 * API functions
 * ------------------------------------------------------------------------- */

/**
 * @brief Analyse the sparsity pattern of A (symbolic phase).
 *
 * @param csc   Pointer to a CSC descriptor (lower triangle, 0-based).
 *              Must not be NULL; col_ptr, row_idx, and values must be valid.
 * @param out   On success, *out is set to a newly allocated analysis handle.
 *              Must not be NULL.
 * @return SMF_OK, SMF_ERR_INVALID_ARG, or SMF_ERR_INTERNAL.
 */
int smf_analyse(const smf_csc_t* csc, smf_analysis_t* out);

/**
 * @brief Perform numerical factorisation of A.
 *
 * @param analysis  Handle obtained from smf_analyse().  Must not be NULL.
 * @param values    Numerical values array (same sparsity pattern as analysed).
 *                  Must not be NULL.
 * @param mtype     SMF_POSDEF or SMF_INDEF.
 * @param out       On success, *out is set to a newly allocated factor handle.
 *                  Must not be NULL.
 * @return SMF_OK, SMF_ERR_INVALID_ARG, SMF_ERR_SINGULAR, or SMF_ERR_INTERNAL.
 */
int smf_factor(smf_analysis_t analysis,
               const double*  values,
               int            mtype,
               smf_factor_t*  out);

/**
 * @brief Solve A X = B in-place (B is overwritten with the solution X).
 *
 * @param analysis  Handle from smf_analyse().  Must not be NULL.
 * @param factor    Handle from smf_factor().   Must not be NULL.
 * @param nrhs      Number of right-hand sides (>= 1).
 * @param rhs       Column-major array of size n * nrhs; overwritten with X.
 *                  Must not be NULL.
 * @param job       SMF_SOLVE_FULL, SMF_SOLVE_FORWARD, or SMF_SOLVE_BACKWARD.
 * @return SMF_OK, SMF_ERR_INVALID_ARG, or SMF_ERR_INTERNAL.
 */
int smf_solve(smf_analysis_t analysis,
              smf_factor_t   factor,
              int            nrhs,
              double*        rhs,
              int            job);

/**
 * @brief Query the inertia computed during the last smf_factor() call.
 *
 * @param factor   Handle from smf_factor().  Must not be NULL.
 * @param inertia  Populated with (pos, neg, zero) counts on success.
 *                 Must not be NULL.
 * @return SMF_OK or SMF_ERR_INVALID_ARG.
 */
int smf_inertia(smf_factor_t factor, smf_inertia_t* inertia);

/**
 * @brief Free an analysis handle.  No-op if handle or *handle is NULL.
 *
 * @param handle  Pointer to the handle variable; set to NULL on return.
 */
void smf_free_analysis(smf_analysis_t* handle);

/**
 * @brief Free a factor handle.  No-op if handle or *handle is NULL.
 *
 * @param handle  Pointer to the handle variable; set to NULL on return.
 */
void smf_free_factor(smf_factor_t* handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SMF_BUILD_C_API */
