#include <gtest/gtest.h>
#include "smf/dense_kernel.hpp"

#include <cmath>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Column-major index: element at row r, col c of an lda×* matrix.
static inline double& cm(double* A, int lda, int r, int c) {
    return A[c * lda + r];
}
static inline double cm(const double* A, int lda, int r, int c) {
    return A[c * lda + r];
}

// Frobenius norm of the difference between two n×n matrices.
static double frob_diff(const double* A, const double* B, int n) {
    double acc = 0.0;
    for (int i = 0; i < n * n; ++i) {
        double d = A[i] - B[i];
        acc += d * d;
    }
    return std::sqrt(acc);
}

// ---------------------------------------------------------------------------
// Test 1: DpotrfSmallSPD
// Build a 5×5 SPD matrix, factor it, reconstruct A = L Lᵀ, check residual.
// ---------------------------------------------------------------------------
TEST(BLASWrap, DpotrfSmallSPD) {
    constexpr int N = 5;
    // Construct A = D + small off-diagonal (scaled to ensure SPD).
    // Use A = I * (N+1) + 0.1 * (ones matrix), which is diagonally dominant.
    std::vector<double> A(N * N, 0.0);
    std::vector<double> Aorig(N * N, 0.0);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double val = (i == j) ? static_cast<double>(N + 2) : 0.1;
            cm(A.data(), N, i, j) = val;
            cm(Aorig.data(), N, i, j) = val;
        }
    }

    int info = smf::smf_dpotrf_lower(A.data(), N, N);
    ASSERT_EQ(info, 0) << "dpotrf failed (not SPD?)";

    // Reconstruct Arec = L Lᵀ using only the lower triangle of A (which now holds L).
    std::vector<double> L(N * N, 0.0);
    for (int j = 0; j < N; ++j) {
        for (int i = j; i < N; ++i) {   // lower triangle
            cm(L.data(), N, i, j) = cm(A.data(), N, i, j);
        }
    }

    // Arec = L * Lᵀ  (manual, column-major)
    std::vector<double> Arec(N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < N; ++k) {
                s += cm(L.data(), N, i, k) * cm(L.data(), N, j, k);
            }
            cm(Arec.data(), N, i, j) = s;
        }
    }

    double err = frob_diff(Aorig.data(), Arec.data(), N);
    EXPECT_LT(err, 1e-12) << "Cholesky reconstruction error: " << err;
}

// ---------------------------------------------------------------------------
// Test 2: DsyrkUpdate
// C = I₄, A = 4×2 matrix; after smf_dsyrk_lower, C_lower should equal (I - A*Aᵀ)_lower.
// ---------------------------------------------------------------------------
TEST(BLASWrap, DsyrkUpdate) {
    constexpr int K = 4;
    constexpr int NCOLS = 2;

    // Identity
    std::vector<double> C(K * K, 0.0);
    for (int i = 0; i < K; ++i) cm(C.data(), K, i, i) = 1.0;

    // A (column-major, 4×2)
    std::vector<double> A = {
        0.1, 0.2, 0.3, 0.4,   // column 0
        0.5, 0.6, 0.7, 0.8    // column 1
    };

    smf::smf_dsyrk_lower(C.data(), K, A.data(), NCOLS, K, K);

    // Hand-compute expected = I - A * Aᵀ  (lower triangle)
    std::vector<double> AAT(K * K, 0.0);
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j <= i; ++j) {   // lower only
            double s = 0.0;
            for (int p = 0; p < NCOLS; ++p) {
                s += cm(A.data(), K, i, p) * cm(A.data(), K, j, p);
            }
            AAT[j * K + i] = s;  // (i,j) col-major lower
        }
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j <= i; ++j) {
            double expected = (i == j ? 1.0 : 0.0) - AAT[j * K + i];
            EXPECT_NEAR(cm(C.data(), K, i, j), expected, 1e-14)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: DgemmSmall  — 3×3 matrix multiply
// ---------------------------------------------------------------------------
TEST(BLASWrap, DgemmSmall) {
    constexpr int M = 3, N = 3, K = 3;

    // A (col-major 3×3)
    std::vector<double> A = {
        1.0, 4.0, 7.0,
        2.0, 5.0, 8.0,
        3.0, 6.0, 9.0
    };
    // B = identity
    std::vector<double> B = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    std::vector<double> C(M * N, 0.0);

    smf::smf_dgemm(C.data(), M, N, K,
                   A.data(), M,
                   B.data(), K,
                   M);

    // A * I = A
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], A[i], 1e-14) << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Test 4: DtrsolvesL  — solve Lx = b for a 4×4 lower triangular L
// ---------------------------------------------------------------------------
TEST(BLASWrap, DtrsolvesL) {
    constexpr int N = 4;
    // Lower triangular L (col-major)
    std::vector<double> L(N * N, 0.0);
    // L[i][j] = i + j + 1  for i >= j  (simple non-singular)
    for (int j = 0; j < N; ++j) {
        for (int i = j; i < N; ++i) {
            cm(L.data(), N, i, j) = static_cast<double>(i + j + 2);
        }
    }

    // RHS b
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> b_orig = b;

    smf::smf_dtrsv_lower(b.data(), N, L.data(), N);

    // Verify residual: ‖Lx - b_orig‖ < 1e-14
    double residual = 0.0;
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j <= i; ++j) {
            s += cm(L.data(), N, i, j) * b[j];
        }
        double d = s - b_orig[i];
        residual += d * d;
    }
    EXPECT_LT(std::sqrt(residual), 1e-14) << "dtrsv_lower residual too large";
}

// ---------------------------------------------------------------------------
// Test 5: DtrsvLT  — solve Lᵀx = b for same 4×4 L
// ---------------------------------------------------------------------------
TEST(BLASWrap, DtrsvLT) {
    constexpr int N = 4;
    std::vector<double> L(N * N, 0.0);
    for (int j = 0; j < N; ++j) {
        for (int i = j; i < N; ++i) {
            cm(L.data(), N, i, j) = static_cast<double>(i + j + 2);
        }
    }

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> b_orig = b;

    smf::smf_dtrsv_lower_transpose(b.data(), N, L.data(), N);

    // Verify residual: ‖Lᵀ x - b_orig‖ < 1e-14
    double residual = 0.0;
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        // Lᵀ[i][j] = L[j][i]
        for (int j = i; j < N; ++j) {
            s += cm(L.data(), N, j, i) * b[j];
        }
        double d = s - b_orig[i];
        residual += d * d;
    }
    EXPECT_LT(std::sqrt(residual), 1e-14) << "dtrsv_lower_transpose residual too large";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
