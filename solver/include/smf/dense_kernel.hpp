#pragma once
// Dense BLAS/LAPACK kernel wrappers for smf.
// All parameters use raw pointers and integer sizes — no STL containers.
// Column-major storage throughout (Fortran order).

namespace smf {

// --- LAPACK wrappers ---

/// Cholesky factorization of n×n SPD matrix A (lower triangle).
/// A is overwritten with L such that A = L Lᵀ.
/// Returns LAPACK info: 0 = success, >0 = not positive definite at pivot info.
int smf_dpotrf_lower(double *A, int n, int lda);

/// Triangular solve: solves X Lᵀ = B  (right, lower, transpose)
/// B is n_rows × k, ldb = leading dim of B. L is k×k lower triangular, ldl =
/// leading dim of L. Overwrites B with X.
void smf_dtrsm_right_lower_transpose(double *B, int n_rows, int k,
                                     const double *L, int ldl, int ldb);

// --- BLAS wrappers ---

/// Symmetric rank-k update: C := C - A Aᵀ  (lower, no-transpose)
/// C is k×k symmetric (lower), A is k×n_cols, lda = leading dim of A, ldc =
/// leading dim of C.
void smf_dsyrk_lower(double *C, int k, const double *A, int n_cols, int lda,
                     int ldc);

/// General matrix multiply: C := alpha * A * B + beta * C
/// A is m×k (lda), B is k×n (ldb), C is m×n (ldc). Column-major.
void smf_dgemm(double *C, int m, int n, int k, const double *A, int lda,
               const double *B, int ldb, int ldc, double alpha = 1.0,
               double beta = 0.0);

/// Matrix-vector: y := alpha * A * x + beta * y. A is m×n, lda.
void smf_dgemv(double *y, int m, int n, const double *A, int lda,
               const double *x, double alpha = 1.0, double beta = 0.0);

/// Matrix-vector: y := alpha * Aᵀ * x + beta * y. A is m×n, lda.
/// x has length m, y has length n.
void smf_dgemv_transpose(double *y, int m, int n, const double *A, int lda,
                         const double *x, double alpha = 1.0,
                         double beta = 0.0);

/// Triangular solve: solves L x = b (lower, no-transpose, non-unit).
/// x overwrites b. n = size.
void smf_dtrsv_lower(double *b, int n, const double *L, int lda);

/// Triangular solve: solves Lᵀ x = b (lower, transpose, non-unit).
void smf_dtrsv_lower_transpose(double *b, int n, const double *L, int lda);

/// Triangular solve: solves L x = b (lower, no-transpose, UNIT diagonal).
void smf_dtrsv_lower_unit(double *b, int n, const double *L, int lda);

/// Triangular solve: solves Lᵀ x = b (lower, transpose, UNIT diagonal).
void smf_dtrsv_lower_transpose_unit(double *b, int n, const double *L, int lda);

/// Triangular matrix-matrix multiply: B := L * B (left, lower, no-transpose,
/// non-unit). B is m×n (ldb). L is m×m (ldl). Overwrites B.
void smf_dtrmm_left_lower(double *B, int m, int n, const double *L, int ldl,
                          int ldb);

} // namespace smf
