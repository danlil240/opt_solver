#include "smf/dense_kernel.hpp"

#include <algorithm>
#include <cmath>

#if defined(SMF_BLAS_REFERENCE) && (SMF_BLAS_REFERENCE)

namespace smf {

int smf_dpotrf_lower(double *A, int n, int lda) {
  if (n < 0 || lda < std::max(1, n)) {
    return -1;
  }
  for (int j = 0; j < n; ++j) {
    double sum = A[j * lda + j];
    for (int k = 0; k < j; ++k) {
      const double ljk = A[k * lda + j];
      sum -= ljk * ljk;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
      return j + 1;
    }
    const double ljj = std::sqrt(sum);
    A[j * lda + j] = ljj;
    const double inv_ljj = 1.0 / ljj;
    for (int i = j + 1; i < n; ++i) {
      double s = A[j * lda + i];
      for (int k = 0; k < j; ++k) {
        s -= A[k * lda + i] * A[k * lda + j];
      }
      A[j * lda + i] = s * inv_ljj;
    }
  }
  return 0;
}

int smf_dpbtrf_lower(double *AB, int n, int kd, int ldab) {
  if (n < 0 || kd < 0 || ldab < kd + 1) {
    return -1;
  }
  for (int j = 0; j < n; ++j) {
    const double ajj = AB[j * ldab + 0];
    if (!(ajj > 0.0) || !std::isfinite(ajj)) {
      return j + 1;
    }
    const double ljj = std::sqrt(ajj);
    AB[j * ldab + 0] = ljj;
    const int klen = std::min(kd, n - 1 - j);
    if (klen > 0) {
      const double inv_ljj = 1.0 / ljj;
      for (int r = 1; r <= klen; ++r) {
        AB[j * ldab + r] *= inv_ljj;
      }
      for (int c = 1; c <= klen; ++c) {
        const double l_jc = AB[j * ldab + c];
        const int col = j + c;
        for (int r = c; r <= klen; ++r) {
          AB[col * ldab + (r - c)] -= AB[j * ldab + r] * l_jc;
        }
      }
    }
  }
  return 0;
}

void smf_dtrsm_right_lower_transpose(double *B, int n_rows, int k,
                                     const double *L, int ldl, int ldb) {
  for (int j = 0; j < k; ++j) {
    for (int p = 0; p < j; ++p) {
      const double l_jp = L[p * ldl + j];
      if (l_jp != 0.0) {
        for (int i = 0; i < n_rows; ++i) {
          B[j * ldb + i] -= B[p * ldb + i] * l_jp;
        }
      }
    }
    const double inv_ljj = 1.0 / L[j * ldl + j];
    for (int i = 0; i < n_rows; ++i) {
      B[j * ldb + i] *= inv_ljj;
    }
  }
}

int smf_dpbtrs_lower(double *B, int n, int kd, int nrhs, const double *AB,
                     int ldab, int ldb) {
  if (n < 0 || kd < 0 || nrhs < 0 || ldab < kd + 1 || ldb < std::max(1, n)) {
    return -1;
  }
  for (int rhs = 0; rhs < nrhs; ++rhs) {
    double *x = B + static_cast<std::size_t>(rhs) * ldb;
    for (int j = 0; j < n; ++j) {
      x[j] /= AB[j * ldab + 0];
      const int klen = std::min(kd, n - 1 - j);
      const double xj = x[j];
      for (int r = 1; r <= klen; ++r) {
        x[j + r] -= AB[j * ldab + r] * xj;
      }
    }
    for (int j = n - 1; j >= 0; --j) {
      const int klen = std::min(kd, n - 1 - j);
      double sum = x[j];
      for (int r = 1; r <= klen; ++r) {
        sum -= AB[j * ldab + r] * x[j + r];
      }
      x[j] = sum / AB[j * ldab + 0];
    }
  }
  return 0;
}

void smf_dsyrk_lower(double *C, int k, const double *A, int n_cols, int lda,
                     int ldc) {
  for (int j = 0; j < k; ++j) {
    for (int p = 0; p < n_cols; ++p) {
      const double a_jp = A[p * lda + j];
      if (a_jp != 0.0) {
        for (int i = j; i < k; ++i) {
          C[j * ldc + i] -= A[p * lda + i] * a_jp;
        }
      }
    }
  }
}

void smf_dgemm(double *C, int m, int n, int k, const double *A, int lda,
               const double *B, int ldb, int ldc, double alpha, double beta) {
  for (int j = 0; j < n; ++j) {
    if (beta == 0.0) {
      for (int i = 0; i < m; ++i) {
        C[j * ldc + i] = 0.0;
      }
    } else if (beta != 1.0) {
      for (int i = 0; i < m; ++i) {
        C[j * ldc + i] *= beta;
      }
    }
    for (int p = 0; p < k; ++p) {
      const double b_pj = alpha * B[j * ldb + p];
      if (b_pj != 0.0) {
        for (int i = 0; i < m; ++i) {
          C[j * ldc + i] += A[p * lda + i] * b_pj;
        }
      }
    }
  }
}

void smf_dgemm_nt(double *C, int m, int n, int k, const double *A, int lda,
                  const double *B, int ldb, int ldc, double alpha, double beta) {
  for (int j = 0; j < n; ++j) {
    if (beta == 0.0) {
      for (int i = 0; i < m; ++i) {
        C[j * ldc + i] = 0.0;
      }
    } else if (beta != 1.0) {
      for (int i = 0; i < m; ++i) {
        C[j * ldc + i] *= beta;
      }
    }
    for (int p = 0; p < k; ++p) {
      const double b_jp = alpha * B[p * ldb + j];
      if (b_jp != 0.0) {
        for (int i = 0; i < m; ++i) {
          C[j * ldc + i] += A[p * lda + i] * b_jp;
        }
      }
    }
  }
}

void smf_dgemmt_lower_nt_sub(double *C, int m, int k, const double *A, int lda,
                             const double *B, int ldb, int ldc) {
  for (int j = 0; j < m; ++j) {
    double *c_j = C + static_cast<std::size_t>(j) * ldc;
    for (int p = 0; p < k; ++p) {
      const double b_jp = B[static_cast<std::size_t>(p) * ldb + j];
      if (b_jp == 0.0) continue;
      const double *a_p = A + static_cast<std::size_t>(p) * lda;
      for (int i = j; i < m; ++i) {
        c_j[i] -= a_p[i] * b_jp;
      }
    }
  }
}

void smf_dgemmt_lower_nn_sub(double *C, int m, int k, const double *A, int lda,
                             const double *B, int ldb, int ldc) {
  for (int j = 0; j < m; ++j) {
    double *c_j = C + static_cast<std::size_t>(j) * ldc;
    for (int p = 0; p < k; ++p) {
      const double b_pj = B[static_cast<std::size_t>(j) * ldb + p];
      if (b_pj == 0.0) continue;
      const double *a_p = A + static_cast<std::size_t>(p) * lda;
      for (int i = j; i < m; ++i) {
        c_j[i] -= a_p[i] * b_pj;
      }
    }
  }
}

void smf_dgemv(double *y, int m, int n, const double *A, int lda,
               const double *x, double alpha, double beta) {
  if (beta == 0.0) {
    for (int i = 0; i < m; ++i) {
      y[i] = 0.0;
    }
  } else if (beta != 1.0) {
    for (int i = 0; i < m; ++i) {
      y[i] *= beta;
    }
  }
  for (int j = 0; j < n; ++j) {
    const double axj = alpha * x[j];
    if (axj != 0.0) {
      for (int i = 0; i < m; ++i) {
        y[i] += A[j * lda + i] * axj;
      }
    }
  }
}

void smf_dgemv_transpose(double *y, int m, int n, const double *A, int lda,
                         const double *x, double alpha, double beta) {
  for (int j = 0; j < n; ++j) {
    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
      sum += A[j * lda + i] * x[i];
    }
    if (beta == 0.0) {
      y[j] = alpha * sum;
    } else {
      y[j] = alpha * sum + beta * y[j];
    }
  }
}

void smf_dtrsv_lower(double *b, int n, const double *L, int lda) {
  for (int j = 0; j < n; ++j) {
    b[j] /= L[j * lda + j];
    const double bj = b[j];
    if (bj != 0.0) {
      for (int i = j + 1; i < n; ++i) {
        b[i] -= L[j * lda + i] * bj;
      }
    }
  }
}

void smf_dtrsv_lower_transpose(double *b, int n, const double *L, int lda) {
  for (int j = n - 1; j >= 0; --j) {
    double sum = b[j];
    for (int i = j + 1; i < n; ++i) {
      sum -= L[j * lda + i] * b[i];
    }
    b[j] = sum / L[j * lda + j];
  }
}

void smf_dtrsv_lower_unit(double *b, int n, const double *L, int lda) {
  for (int j = 0; j < n; ++j) {
    const double bj = b[j];
    if (bj != 0.0) {
      for (int i = j + 1; i < n; ++i) {
        b[i] -= L[j * lda + i] * bj;
      }
    }
  }
}

void smf_dtrsv_lower_transpose_unit(double *b, int n, const double *L,
                                    int lda) {
  for (int j = n - 1; j >= 0; --j) {
    double sum = b[j];
    for (int i = j + 1; i < n; ++i) {
      sum -= L[j * lda + i] * b[i];
    }
    b[j] = sum;
  }
}

void smf_dtrmm_left_lower(double *B, int m, int n, const double *L, int ldl,
                          int ldb) {
  for (int j = 0; j < n; ++j) {
    double *col = B + static_cast<std::size_t>(j) * ldb;
    for (int i = m - 1; i >= 0; --i) {
      double sum = L[i * ldl + i] * col[i];
      for (int k = 0; k < i; ++k) {
        sum += L[k * ldl + i] * col[k];
      }
      col[i] = sum;
    }
  }
}

} // namespace smf

#else

#if defined(SMF_USE_MKL) && (SMF_USE_MKL)
#include <mkl.h>
#else
#include <cblas.h>
// Fortran LAPACK declarations — avoids dependency on liblapacke-dev.
extern "C" {
void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
void dpbtrf_(char *uplo, int *n, int *kd, double *ab, int *ldab, int *info);
void dpbtrs_(char *uplo, int *n, int *kd, int *nrhs, double *ab, int *ldab,
             double *b, int *ldb, int *info);
}
#endif

namespace smf {

int smf_dpotrf_lower(double *A, int n, int lda) {
  if (n < 0 || lda < std::max(1, n)) {
    return -1;
  }
  if (n <= 12) {
    for (int j = 0; j < n; ++j) {
      double *col_j = A + static_cast<std::size_t>(j) * lda;
      double sum = col_j[j];
      for (int k = 0; k < j; ++k) {
        const double ljk = A[static_cast<std::size_t>(k) * lda + j];
        sum -= ljk * ljk;
      }
      if (!(sum > 0.0) || !std::isfinite(sum)) {
        return j + 1;
      }
      const double ljj = std::sqrt(sum);
      col_j[j] = ljj;
      const double inv_ljj = 1.0 / ljj;
      for (int i = j + 1; i < n; ++i) {
        double s = col_j[i];
        for (int k = 0; k < j; ++k) {
          const double *col_k = A + static_cast<std::size_t>(k) * lda;
          s -= col_k[i] * col_k[j];
        }
        col_j[i] = s * inv_ljj;
      }
    }
    return 0;
  }
  char uplo = 'L';
  int info = 0;
  dpotrf_(&uplo, &n, A, &lda, &info);
  return info;
}

int smf_dpbtrf_lower(double *AB, int n, int kd, int ldab) {
  if (n < 0 || kd < 0 || ldab < kd + 1) {
    return -1;
  }
  if (kd <= 8 && n <= 512) {
    for (int j = 0; j < n; ++j) {
      double *col_j = AB + static_cast<std::size_t>(j) * ldab;
      const double ajj = col_j[0];
      if (!(ajj > 0.0) || !std::isfinite(ajj)) {
        return j + 1;
      }
      const double ljj = std::sqrt(ajj);
      col_j[0] = ljj;
      const int klen = std::min(kd, n - 1 - j);
      if (klen > 0) {
        const double inv_ljj = 1.0 / ljj;
        for (int r = 1; r <= klen; ++r) {
          col_j[r] *= inv_ljj;
        }
        for (int c = 1; c <= klen; ++c) {
          const double l_jc = col_j[c];
          double *col_jc = AB + static_cast<std::size_t>(j + c) * ldab;
          for (int r = c; r <= klen; ++r) {
            col_jc[r - c] -= col_j[r] * l_jc;
          }
        }
      }
    }
    return 0;
  }
  char uplo = 'L';
  int info = 0;
  dpbtrf_(&uplo, &n, &kd, AB, &ldab, &info);
  return info;
}

void smf_dtrsm_right_lower_transpose(double *B, int n_rows, int k,
                                     const double *L, int ldl, int ldb) {
  if (n_rows <= 0 || k <= 0) return;
  if (k <= 8 && n_rows * k <= 128) {
    for (int j = 0; j < k; ++j) {
      double *b_j = B + static_cast<std::size_t>(j) * ldb;
      for (int p = 0; p < j; ++p) {
        const double l_jp = L[static_cast<std::size_t>(p) * ldl + j];
        if (l_jp != 0.0) {
          const double *b_p = B + static_cast<std::size_t>(p) * ldb;
          for (int i = 0; i < n_rows; ++i) {
            b_j[i] -= b_p[i] * l_jp;
          }
        }
      }
      const double inv_ljj = 1.0 / L[static_cast<std::size_t>(j) * ldl + j];
      for (int i = 0; i < n_rows; ++i) {
        b_j[i] *= inv_ljj;
      }
    }
    return;
  }
  // B := B * L^{-T}  — right, lower, transpose, non-unit, alpha=1
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
              n_rows, k, 1.0, L, ldl, B, ldb);
}

int smf_dpbtrs_lower(double *B, int n, int kd, int nrhs, const double *AB,
                     int ldab, int ldb) {
  if (n < 0 || kd < 0 || nrhs < 0 || ldab < kd + 1 || ldb < std::max(1, n)) {
    return -1;
  }
  if (nrhs == 1 && kd <= 16) {
    double *x = B;
    for (int j = 0; j < n; ++j) {
      const double *col_j = AB + static_cast<std::size_t>(j) * ldab;
      x[j] /= col_j[0];
      const int klen = std::min(kd, n - 1 - j);
      const double xj = x[j];
      for (int r = 1; r <= klen; ++r) {
        x[j + r] -= col_j[r] * xj;
      }
    }
    for (int j = n - 1; j >= 0; --j) {
      const double *col_j = AB + static_cast<std::size_t>(j) * ldab;
      const int klen = std::min(kd, n - 1 - j);
      double sum = x[j];
      for (int r = 1; r <= klen; ++r) {
        sum -= col_j[r] * x[j + r];
      }
      x[j] = sum / col_j[0];
    }
    return 0;
  }
  char uplo = 'L';
  int info = 0;
  dpbtrs_(&uplo, &n, &kd, &nrhs, const_cast<double *>(AB), &ldab, B, &ldb,
          &info);
  return info;
}

void smf_dsyrk_lower(double *C, int k, const double *A, int n_cols, int lda,
                     int ldc) {
  if (k <= 0 || n_cols <= 0) return;
  if (n_cols <= 8 && k * n_cols <= 128) {
    for (int p = 0; p < n_cols; ++p) {
      const double *a_p = A + static_cast<std::size_t>(p) * lda;
      for (int j = 0; j < k; ++j) {
        const double a_jp = a_p[j];
        if (a_jp != 0.0) {
          double *c_j = C + static_cast<std::size_t>(j) * ldc;
          for (int i = j; i < k; ++i) {
            c_j[i] -= a_p[i] * a_jp;
          }
        }
      }
    }
    return;
  }
  // C := -1 * A * A^T + 1 * C  (lower, no-transpose)
  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, k, n_cols, -1.0, A, lda,
              1.0, C, ldc);
}

void smf_dgemm(double *C, int m, int n, int k, const double *A, int lda,
               const double *B, int ldb, int ldc, double alpha, double beta) {
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda,
              B, ldb, beta, C, ldc);
}

void smf_dgemm_nt(double *C, int m, int n, int k, const double *A, int lda,
                  const double *B, int ldb, int ldc, double alpha, double beta) {
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, A, lda,
              B, ldb, beta, C, ldc);
}

void smf_dgemmt_lower_nt_sub(double *C, int m, int k, const double *A, int lda,
                             const double *B, int ldb, int ldc) {
  if (m <= 0 || k <= 0) return;
#if defined(SMF_USE_MKL) && (SMF_USE_MKL)
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION >= 20250000
  cblas_dgemmtr(CblasColMajor, CblasLower, CblasNoTrans, CblasTrans, m, k, -1.0,
                A, lda, B, ldb, 1.0, C, ldc);
#else
  cblas_dgemmt(CblasColMajor, CblasLower, CblasNoTrans, CblasTrans, m, k, -1.0,
               A, lda, B, ldb, 1.0, C, ldc);
#endif
#else
  // Column-blocked GEMM: only the upper half of each diagonal block is wasted.
  constexpr int kBlock = 64;
  for (int j0 = 0; j0 < m; j0 += kBlock) {
    const int nc = std::min(kBlock, m - j0);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m - j0, nc, k, -1.0,
                A + j0, lda, B + j0, ldb, 1.0,
                C + static_cast<std::size_t>(j0) * ldc + j0, ldc);
  }
#endif
}

void smf_dgemmt_lower_nn_sub(double *C, int m, int k, const double *A, int lda,
                             const double *B, int ldb, int ldc) {
  if (m <= 0 || k <= 0) return;
#if defined(SMF_USE_MKL) && (SMF_USE_MKL)
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION >= 20250000
  cblas_dgemmtr(CblasColMajor, CblasLower, CblasNoTrans, CblasNoTrans, m, k, -1.0,
                A, lda, B, ldb, 1.0, C, ldc);
#else
  cblas_dgemmt(CblasColMajor, CblasLower, CblasNoTrans, CblasNoTrans, m, k, -1.0,
               A, lda, B, ldb, 1.0, C, ldc);
#endif
#else
  // Column-blocked GEMM: only the upper half of each diagonal block is wasted.
  constexpr int kBlock = 64;
  for (int j0 = 0; j0 < m; j0 += kBlock) {
    const int nc = std::min(kBlock, m - j0);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m - j0, nc, k, -1.0,
                A + j0, lda, B + static_cast<std::size_t>(j0) * ldb, ldb, 1.0,
                C + static_cast<std::size_t>(j0) * ldc + j0, ldc);
  }
#endif
}

void smf_dgemv(double *y, int m, int n, const double *A, int lda,
               const double *x, double alpha, double beta) {
  if (m <= 0 || n <= 0) return;
  if (m * n <= 256) {
    if (beta == 0.0) {
      for (int i = 0; i < m; ++i) y[i] = 0.0;
    } else if (beta != 1.0) {
      for (int i = 0; i < m; ++i) y[i] *= beta;
    }
    for (int j = 0; j < n; ++j) {
      const double axj = alpha * x[j];
      if (axj != 0.0) {
        const double *col_j = A + static_cast<std::size_t>(j) * lda;
        for (int i = 0; i < m; ++i) {
          y[i] += col_j[i] * axj;
        }
      }
    }
    return;
  }
  cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, A, lda, x, 1, beta, y,
              1);
}

void smf_dgemv_transpose(double *y, int m, int n, const double *A, int lda,
                         const double *x, double alpha, double beta) {
  if (m <= 0 || n <= 0) return;
  if (m * n <= 256) {
    for (int j = 0; j < n; ++j) {
      const double *col_j = A + static_cast<std::size_t>(j) * lda;
      double sum = 0.0;
      for (int i = 0; i < m; ++i) {
        sum += col_j[i] * x[i];
      }
      if (beta == 0.0) {
        y[j] = alpha * sum;
      } else {
        y[j] = alpha * sum + beta * y[j];
      }
    }
    return;
  }
  cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, A, lda, x, 1, beta, y,
              1);
}

void smf_dtrsv_lower(double *b, int n, const double *L, int lda) {
  if (n <= 0) return;
  if (n <= 16) {
    for (int j = 0; j < n; ++j) {
      const double *col_j = L + static_cast<std::size_t>(j) * lda;
      b[j] /= col_j[j];
      const double bj = b[j];
      if (bj != 0.0) {
        for (int i = j + 1; i < n; ++i) {
          b[i] -= col_j[i] * bj;
        }
      }
    }
    return;
  }
  cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, L, lda,
              b, 1);
}

void smf_dtrsv_lower_transpose(double *b, int n, const double *L, int lda) {
  if (n <= 0) return;
  if (n <= 16) {
    for (int j = n - 1; j >= 0; --j) {
      const double *col_j = L + static_cast<std::size_t>(j) * lda;
      double sum = b[j];
      for (int i = j + 1; i < n; ++i) {
        sum -= col_j[i] * b[i];
      }
      b[j] = sum / col_j[j];
    }
    return;
  }
  cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, n, L, lda, b,
              1);
}

void smf_dtrsv_lower_unit(double *b, int n, const double *L, int lda) {
  if (n <= 0) return;
  if (n <= 16) {
    for (int j = 0; j < n; ++j) {
      const double bj = b[j];
      if (bj != 0.0) {
        const double *col_j = L + static_cast<std::size_t>(j) * lda;
        for (int i = j + 1; i < n; ++i) {
          b[i] -= col_j[i] * bj;
        }
      }
    }
    return;
  }
  cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, n, L, lda, b,
              1);
}

void smf_dtrsv_lower_transpose_unit(double *b, int n, const double *L,
                                    int lda) {
  if (n <= 0) return;
  if (n <= 16) {
    for (int j = n - 1; j >= 0; --j) {
      const double *col_j = L + static_cast<std::size_t>(j) * lda;
      double sum = b[j];
      for (int i = j + 1; i < n; ++i) {
        sum -= col_j[i] * b[i];
      }
      b[j] = sum;
    }
    return;
  }
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

#endif
