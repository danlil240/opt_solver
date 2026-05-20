#include "smf/dense_kernel.hpp"

#if defined(SMF_USE_MKL) && (SMF_USE_MKL)
#include <mkl.h>
#else
#include <cblas.h>
// Fortran LAPACK declarations — avoids dependency on liblapacke-dev.
extern "C" {
void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
}
#endif

namespace smf {

int smf_dpotrf_lower(double *A, int n, int lda) {
  char uplo = 'L';
  int info = 0;
  dpotrf_(&uplo, &n, A, &lda, &info);
  return info;
}

void smf_dtrsm_right_lower_transpose(double *B, int n_rows, int k,
                                     const double *L, int ldl, int ldb) {
  // B := B * L^{-T}  — right, lower, transpose, non-unit, alpha=1
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
              n_rows, k, 1.0, L, ldl, B, ldb);
}

void smf_dsyrk_lower(double *C, int k, const double *A, int n_cols, int lda,
                     int ldc) {
  // C := -1 * A * A^T + 1 * C  (lower, no-transpose)
  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, k, n_cols, -1.0, A, lda,
              1.0, C, ldc);
}

void smf_dgemm(double *C, int m, int n, int k, const double *A, int lda,
               const double *B, int ldb, int ldc, double alpha, double beta) {
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda,
              B, ldb, beta, C, ldc);
}

void smf_dgemv(double *y, int m, int n, const double *A, int lda,
               const double *x, double alpha, double beta) {
  cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, A, lda, x, 1, beta, y,
              1);
}

void smf_dgemv_transpose(double *y, int m, int n, const double *A, int lda,
                         const double *x, double alpha, double beta) {
  cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, A, lda, x, 1, beta, y,
              1);
}

void smf_dtrsv_lower(double *b, int n, const double *L, int lda) {
  cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, L, lda,
              b, 1);
}

void smf_dtrsv_lower_transpose(double *b, int n, const double *L, int lda) {
  cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, n, L, lda, b,
              1);
}

void smf_dtrsv_lower_unit(double *b, int n, const double *L, int lda) {
  cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, n, L, lda, b,
              1);
}

void smf_dtrsv_lower_transpose_unit(double *b, int n, const double *L,
                                    int lda) {
  cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasUnit, n, L, lda, b,
              1);
}

void smf_dtrmm_left_lower(double *B, int m, int n, const double *L, int ldl,
                          int ldb) {
  // B := L * B  (left, lower, no-transpose, non-unit, alpha=1)
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
              m, n, 1.0, L, ldl, B, ldb);
}

} // namespace smf
