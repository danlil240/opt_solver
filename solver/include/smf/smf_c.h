/**
 * @file smf_c.h
 * @brief Pure C ABI for the smf sparse symmetric solver.
 *
 * Provides two layers:
 *  1. Simple handle API (smf_analyse / smf_factor / smf_solve / smf_inertia)
 *     for one-off sparse symmetric solves.
 *  2. Versioned interior-point session API (smf_ipm_*) backed by
 *     smf::IpmSession, designed for direct integration with IPOPT
 *     (linear_solver=smf), MadNLP, CasADi, Julia, and Python bindings.
 *
 * All integer indices are 0-based.  All dense arrays are column-major.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* -------------------------------------------------------------------------
 * Symbol visibility / DLL export
 * ------------------------------------------------------------------------- */
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(SMF_C_EXPORTS)
#    define SMF_C_API __declspec(dllexport)
#  elif defined(SMF_C_SHARED)
#    define SMF_C_API __declspec(dllimport)
#  else
#    define SMF_C_API
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define SMF_C_API __attribute__((visibility("default")))
#else
#  define SMF_C_API
#endif

/** Current C ABI version (bumped only on breaking struct/signature changes). */
#define SMF_C_ABI_VERSION 1

/* -------------------------------------------------------------------------
 * Version & build introspection
 * ------------------------------------------------------------------------- */

/** Returns SMF_C_ABI_VERSION that this library was compiled with. */
SMF_C_API int smf_c_abi_version(void);

/** Returns a static null-terminated library version string (e.g. "0.1.0"). */
SMF_C_API const char* smf_version(void);

/** Returns a static null-terminated BLAS backend string (e.g. "openblas", "mkl"). */
SMF_C_API const char* smf_blas_backend(void);

/* -------------------------------------------------------------------------
 * Opaque handles
 * ------------------------------------------------------------------------- */

/** Opaque handle returned by smf_analyse(). */
typedef struct smf_analysis_s* smf_analysis_t;

/** Opaque handle returned by smf_factor(). */
typedef struct smf_factor_s*   smf_factor_t;

/** Opaque handle returned by smf_ipm_create() for repeated KKT solves. */
typedef struct smf_ipm_session_s* smf_ipm_session_t;

/* -------------------------------------------------------------------------
 * Error codes  (match smf::ErrorCode integer values where applicable)
 * ------------------------------------------------------------------------- */

#define SMF_OK                   0   /**< Success                                          */
#define SMF_ERR_INVALID_ARG      1   /**< Null pointer or bad argument                     */
#define SMF_ERR_SINGULAR         2   /**< Matrix is singular or not positive definite      */
#define SMF_ERR_INTERNAL         3   /**< Unexpected internal error                        */
#define SMF_ERR_WRONG_INERTIA    4   /**< Factor aborted early after exceeding target negs */
#define SMF_ERR_INEXACT_INERTIA  5   /**< Static pivot perturbed inertia; regularize       */

/* -------------------------------------------------------------------------
 * Matrix type  (passed to smf_factor / smf_ipm_factor)
 * ------------------------------------------------------------------------- */

#define SMF_POSDEF    0   /**< Real symmetric positive-definite */
#define SMF_INDEF     1   /**< Real symmetric indefinite        */

/* -------------------------------------------------------------------------
 * Solve job  (passed to smf_solve / smf_ipm_solve)
 * ------------------------------------------------------------------------- */

#define SMF_SOLVE_FULL     0   /**< Full AX = B solve                       */
#define SMF_SOLVE_FORWARD  1   /**< Forward sweep only   (PLX = SB)         */
#define SMF_SOLVE_DIAGONAL 2   /**< Diagonal solve only  (DX = B)           */
#define SMF_SOLVE_BACKWARD 3   /**< Backward sweep only  ((PL)^T S^{-1}X=B) */

/* -------------------------------------------------------------------------
 * Pivoting policy  (passed to smf_set_pivoting / smf_ipm_control_t)
 * ------------------------------------------------------------------------- */

#define SMF_PIVOTING_THRESHOLD 0   /**< Default: accurate threshold pivoting   */
#define SMF_PIVOTING_STATIC    1   /**< Opt-in fast mode: static pivoting      */

/* -------------------------------------------------------------------------
 * Ordering / scaling / slack options (smf_ipm_control_t)
 * ------------------------------------------------------------------------- */

#define SMF_ORDER_AUTO     0   /**< AutoParallel (AMD / METIS / ParallelND / Matching) */
#define SMF_ORDER_AMD      1   /**< Approximate Minimum Degree                         */
#define SMF_ORDER_METIS    2   /**< METIS nested dissection                            */
#define SMF_ORDER_MATCHING 3   /**< Matching-based AMD ordering                        */

#define SMF_SCALE_NONE          0   /**< No scaling (or reuse caller-supplied scale[]) */
#define SMF_SCALE_MATCHING      1   /**< MC64-style max-product matching scaling       */
#define SMF_SCALE_EQUILIBRATION 2   /**< Ruiz equilibration scaling                    */
#define SMF_SCALE_AUTO          3   /**< Native smf scale-on-delay / analysis policy   */

#define SMF_SLACK_OFF   0   /**< Disable slack-pair elimination */
#define SMF_SLACK_ON    1   /**< Always try slack-pair elimination when eligible */
#define SMF_SLACK_AUTO  2   /**< Enable when slack pairs exceed share threshold  */

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
 *
 * Note: For a symmetric matrix, 0-based lower-triangular CSC is identical to
 * 0-based upper-triangular CSR (IPOPT's CSR_Format_0_Offset).
 */
typedef struct {
    int           n;        /**< Matrix order (number of rows/columns)   */
    const int*    col_ptr;  /**< Column pointers, length n+1             */
    const int*    row_idx;  /**< Row indices,    length col_ptr[n]       */
    const double* values;   /**< Numerical values, length col_ptr[n]     */
} smf_csc_t;

/* -------------------------------------------------------------------------
 * Simple C API functions
 * ------------------------------------------------------------------------- */

SMF_C_API int  smf_analyse(const smf_csc_t* csc, smf_analysis_t* out);
SMF_C_API int  smf_set_pivoting(smf_analysis_t analysis, int pivoting);
SMF_C_API int  smf_factor(smf_analysis_t analysis,
                          const double*  values,
                          int            mtype,
                          smf_factor_t*  out);
SMF_C_API int  smf_solve(smf_analysis_t analysis,
                         smf_factor_t   factor,
                         int            nrhs,
                         double*        rhs,
                         int            job);
SMF_C_API int  smf_inertia(smf_factor_t factor, smf_inertia_t* inertia);
SMF_C_API void smf_free_analysis(smf_analysis_t* handle);
SMF_C_API void smf_free_factor(smf_factor_t* handle);

/* -------------------------------------------------------------------------
 * Interior-point session C API (smf_ipm_*)
 * ------------------------------------------------------------------------- */

/**
 * Control parameters for smf_ipm_* calls.
 * Always initialize with smf_ipm_default_control() before modifying fields.
 */
typedef struct {
    int    abi_version;          /**< Set to SMF_C_ABI_VERSION by default_control   */
    int    struct_size;          /**< Set to sizeof(smf_ipm_control_t)              */
    int    ordering;             /**< SMF_ORDER_* (default: SMF_ORDER_AUTO)         */
    int    scaling;              /**< SMF_SCALE_* (default: SMF_SCALE_AUTO)         */
    int    pivoting;             /**< SMF_PIVOTING_* (default: THRESHOLD)           */
    int    slack_elimination;    /**< SMF_SLACK_* (default: SMF_SLACK_OFF)          */
    int    nemin;                /**< Supernode amalgamation threshold (default: 8) */
    int    num_threads;          /**< 0 = auto (SMF_NUM_THREADS/OMP/P-cores), >=1   */
    int    deterministic;        /**< 0 = default, 1 = bitwise deterministic        */
    int    continue_on_singular; /**< 1 = continue past zero pivots (default: 1)    */
    int    honest_inertia;       /**< 1 = return SMF_ERR_INEXACT_INERTIA if static  */
    int    print_level;          /**< <= 0 silent                                   */
    double u;                    /**< Pivot threshold in [0, 0.5] (default: 1e-8)   */
    double small;                /**< Zero-pivot threshold (default: 1e-20)         */
    double multiplier;           /**< Factor memory multiplier (default: 1.2)       */
    int    reserved_i[8];        /**< Reserved for future ABI-compatible fields     */
    double reserved_d[8];        /**< Reserved for future ABI-compatible fields     */
} smf_ipm_control_t;

/**
 * Diagnostics and inertia populated by smf_ipm_analyse / factor / solve.
 */
typedef struct {
    int     struct_size;     /**< sizeof(smf_ipm_info_t)                           */
    int     status;          /**< SMF_OK or SMF_ERR_*                              */
    int     num_pos;         /**< Positive eigenvalue count                        */
    int     num_neg;         /**< Negative eigenvalue count                        */
    int     num_zero;        /**< Zero eigenvalue count                            */
    int     matrix_rank;     /**< n - num_zero                                     */
    int     num_delay;       /**< Delayed pivots                                   */
    int     num_two;         /**< 2x2 pivots                                       */
    int     maxfront;        /**< Maximum front dimension                          */
    int     maxsupernode;    /**< Maximum supernode width                          */
    int     maxdepth;        /**< Assembly tree depth                              */
    int     num_sup;         /**< Number of supernodes                             */
    int     ordering_used;   /**< Ordering actually chosen                         */
    int     inertia_exact;   /**< 1 if inertia is exact, 0 if static-perturbed     */
    int     threads_used;    /**< Thread count used in the last factorization      */
    int     static_pivots;   /**< Number of static-perturbed pivots                */
    int64_t num_factor;      /**< Actual factor entries                            */
    int64_t num_flops;       /**< Actual factorization flops                       */
    double  analyse_seconds; /**< Wall time of symbolic analysis (seconds)         */
    double  factor_seconds;  /**< Wall time of numeric factorization (seconds)     */
    double  solve_seconds;   /**< Wall time of triangular solve (seconds)          */
    int     reserved_i[8];
    double  reserved_d[8];
} smf_ipm_info_t;

/** Fill ctrl with defaults suitable for interior-point KKT systems. */
SMF_C_API void smf_ipm_default_control(smf_ipm_control_t* ctrl);

/** Allocate a new reusable IPM session handle. */
SMF_C_API int smf_ipm_create(smf_ipm_session_t* out);

/**
 * Store a 0-based lower-triangular CSC sparsity pattern (equivalently 0-based
 * upper-triangular CSR) and run symbolic analysis.
 *
 * If values == NULL, symbolic analysis is deferred to the first
 * smf_ipm_factor() call so saddle-point matching sees real numerical values.
 * Entries within each column do not need to be sorted; upper-triangular
 * entries (row < col) are mirrored into the lower triangle, so the lower
 * triangle, the upper triangle or both may be passed; where both triangles
 * hold a position, the upper copy is ignored (not summed).
 */
SMF_C_API int smf_ipm_analyse(smf_ipm_session_t        session,
                              int                      n,
                              const int*               col_ptr,
                              const int*               row_idx,
                              const double*            values,
                              const smf_ipm_control_t* ctrl,
                              smf_ipm_info_t*          info);

/**
 * Factorize new numerical values (in the same entry order passed to
 * smf_ipm_analyse) reusing the session's analysis and workspaces.
 *
 * @param session  Session handle from smf_ipm_create() + smf_ipm_analyse().
 * @param mtype    SMF_INDEF or SMF_POSDEF.
 * @param values   Length col_ptr[n] array of numerical values.
 * @param ctrl     Control options (may be NULL to use defaults).
 * @param info     Optional diagnostics/inertia output (may be NULL).
 * @param scale    Optional length-n scaling vector for caller-managed dynamic
 *                 scaling (e.g. IPOPT reuse strategies). Pass NULL when using
 *                 SMF_SCALE_AUTO or unscaled factorizations.
 * @return SMF_OK, SMF_ERR_SINGULAR, SMF_ERR_INEXACT_INERTIA,
 *         SMF_ERR_WRONG_INERTIA, SMF_ERR_INVALID_ARG, or SMF_ERR_INTERNAL.
 */
SMF_C_API int smf_ipm_factor(smf_ipm_session_t        session,
                             int                      mtype,
                             const double*            values,
                             const smf_ipm_control_t* ctrl,
                             smf_ipm_info_t*          info,
                             double*                  scale);

/**
 * Solve A X = B in-place using the session's latest factorization.
 *
 * @param session  Session handle with a completed factorization.
 * @param job      SMF_SOLVE_FULL, SMF_SOLVE_FORWARD, SMF_SOLVE_DIAGONAL, or
 *                 SMF_SOLVE_BACKWARD.
 * @param nrhs     Number of right-hand sides (>= 1).
 * @param rhs      Column-major array of leading dimension ldx (at least n).
 * @param ldx      Leading dimension of rhs (>= n).
 * @param ctrl     Optional control block (may be NULL).
 * @param info     Optional diagnostics output (may be NULL).
 */
SMF_C_API int smf_ipm_solve(smf_ipm_session_t        session,
                            int                      job,
                            int                      nrhs,
                            double*                  rhs,
                            int                      ldx,
                            const smf_ipm_control_t* ctrl,
                            smf_ipm_info_t*          info);

/** Free an IPM session handle and set *session = NULL. */
SMF_C_API void smf_ipm_free(smf_ipm_session_t* session);

#ifdef __cplusplus
} /* extern "C" */
#endif
