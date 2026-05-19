/**
 * @file test_c_api.cpp
 * @brief GoogleTest suite exercising the smf C ABI (smf_c.h / smf_c.cpp).
 *
 * Compiled only when SMF_BUILD_C_API=1 (gated in CMakeLists.txt).
 */

#include "smf/smf_c.h"

#include <cmath>
#include <gtest/gtest.h>

/* =========================================================================
 * Helpers — small test matrices in CSC lower-triangle format (0-based)
 * ========================================================================= */

/**
 * 5×5 tridiagonal SPD matrix  (diag=4, sub-diag=-1).
 *
 * Full symmetric:
 *   [ 4 -1  0  0  0 ]
 *   [-1  4 -1  0  0 ]
 *   [ 0 -1  4 -1  0 ]
 *   [ 0  0 -1  4 -1 ]
 *   [ 0  0  0 -1  4 ]
 *
 * Lower triangle stored column-by-column:
 *   col 0: (0,0)=4, (1,0)=-1
 *   col 1: (1,1)=4, (2,1)=-1
 *   col 2: (2,2)=4, (3,2)=-1
 *   col 3: (3,3)=4, (4,3)=-1
 *   col 4: (4,4)=4
 */
static const int    tridiag5_col_ptr[] = {0, 2, 4, 6, 8, 9};
static const int    tridiag5_row_idx[] = {0,1, 1,2, 2,3, 3,4, 4};
static const double tridiag5_values[]  = {4.,-1., 4.,-1., 4.,-1., 4.,-1., 4.};

static smf_csc_t make_tridiag5() {
    smf_csc_t m;
    m.n       = 5;
    m.col_ptr = tridiag5_col_ptr;
    m.row_idx = tridiag5_row_idx;
    m.values  = tridiag5_values;
    return m;
}

/**
 * Matrix–vector multiply (full symmetric tridiag5):  y = A * x
 */
static void matvec_tridiag5(const double* x, double* y) {
    y[0] =  4.*x[0] - 1.*x[1];
    y[1] = -1.*x[0] + 4.*x[1] - 1.*x[2];
    y[2] =           -1.*x[1] + 4.*x[2] - 1.*x[3];
    y[3] =                      -1.*x[2] + 4.*x[3] - 1.*x[4];
    y[4] =                                 -1.*x[3] + 4.*x[4];
}

/**
 * 3×3 indefinite diagonal matrix: diag = [1, -1, 1].
 * Inertia: pos=2, neg=1, zero=0.
 *
 * Lower CSC:
 *   col 0: (0,0)= 1
 *   col 1: (1,1)=-1
 *   col 2: (2,2)= 1
 */
static const int    indef3_col_ptr[] = {0, 1, 2, 3};
static const int    indef3_row_idx[] = {0, 1, 2};
static const double indef3_values[]  = {1., -1., 1.};

static smf_csc_t make_indef3() {
    smf_csc_t m;
    m.n       = 3;
    m.col_ptr = indef3_col_ptr;
    m.row_idx = indef3_row_idx;
    m.values  = indef3_values;
    return m;
}

/* =========================================================================
 * Test 1 — smf_analyse returns SMF_OK and non-null handle
 * ========================================================================= */

TEST(CApiTest, CApiAnalyse) {
    smf_csc_t csc = make_tridiag5();
    smf_analysis_t h = nullptr;

    int rc = smf_analyse(&csc, &h);
    EXPECT_EQ(rc, SMF_OK);
    EXPECT_NE(h, nullptr);

    smf_free_analysis(&h);
    EXPECT_EQ(h, nullptr);
}

/* =========================================================================
 * Test 2 — smf_factor returns SMF_OK for SPD matrix
 * ========================================================================= */

TEST(CApiTest, CApiFactor) {
    smf_csc_t      csc = make_tridiag5();
    smf_analysis_t ah  = nullptr;

    ASSERT_EQ(smf_analyse(&csc, &ah), SMF_OK);
    ASSERT_NE(ah, nullptr);

    smf_factor_t fh = nullptr;
    int rc = smf_factor(ah, tridiag5_values, SMF_POSDEF, &fh);
    EXPECT_EQ(rc, SMF_OK);
    EXPECT_NE(fh, nullptr);

    smf_free_factor(&fh);
    smf_free_analysis(&ah);
}

/* =========================================================================
 * Test 3 — full pipeline: analyse + factor + solve; check residual < 1e-12
 * ========================================================================= */

TEST(CApiTest, CApiSolve) {
    smf_csc_t      csc = make_tridiag5();
    smf_analysis_t ah  = nullptr;
    smf_factor_t   fh  = nullptr;

    ASSERT_EQ(smf_analyse(&csc, &ah), SMF_OK);
    ASSERT_EQ(smf_factor(ah, tridiag5_values, SMF_POSDEF, &fh), SMF_OK);

    /* RHS: b = [1, 2, 3, 4, 5] */
    const int n = 5;
    double rhs[5] = {1., 2., 3., 4., 5.};
    const double b_orig[5] = {1., 2., 3., 4., 5.};

    int rc = smf_solve(ah, fh, 1, rhs, SMF_SOLVE_FULL);
    EXPECT_EQ(rc, SMF_OK);

    /* Residual: r = A*x - b_orig */
    double Ax[5];
    matvec_tridiag5(rhs, Ax);

    double resid = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = Ax[i] - b_orig[i];
        resid += d * d;
    }
    resid = std::sqrt(resid);

    EXPECT_LT(resid, 1e-12) << "Residual " << resid << " too large";

    smf_free_factor(&fh);
    smf_free_analysis(&ah);
}

/* =========================================================================
 * Test 4 — inertia for 3×3 indefinite diagonal matrix
 * ========================================================================= */

TEST(CApiTest, CApiInertia) {
    smf_csc_t      csc = make_indef3();
    smf_analysis_t ah  = nullptr;
    smf_factor_t   fh  = nullptr;

    ASSERT_EQ(smf_analyse(&csc, &ah), SMF_OK);
    ASSERT_EQ(smf_factor(ah, indef3_values, SMF_INDEF, &fh), SMF_OK);

    smf_inertia_t iner = {0, 0, 0};
    int rc = smf_inertia(fh, &iner);
    EXPECT_EQ(rc, SMF_OK);

    EXPECT_EQ(iner.pos,  2) << "Expected 2 positive eigenvalues";
    EXPECT_EQ(iner.neg,  1) << "Expected 1 negative eigenvalue";
    EXPECT_EQ(iner.zero, 0) << "Expected 0 zero eigenvalues";

    smf_free_factor(&fh);
    smf_free_analysis(&ah);
}

/* =========================================================================
 * Test 5 — free with null must not crash
 * ========================================================================= */

TEST(CApiTest, CApiFreeNull) {
    /* Passing nullptr directly */
    EXPECT_NO_FATAL_FAILURE(smf_free_analysis(nullptr));
    EXPECT_NO_FATAL_FAILURE(smf_free_factor(nullptr));

    /* Passing pointer-to-null */
    smf_analysis_t ah = nullptr;
    smf_factor_t   fh = nullptr;
    EXPECT_NO_FATAL_FAILURE(smf_free_analysis(&ah));
    EXPECT_NO_FATAL_FAILURE(smf_free_factor(&fh));
}

/* =========================================================================
 * Test 6 — null input pointers must return SMF_ERR_INVALID_ARG
 * ========================================================================= */

TEST(CApiTest, CApiNullInputs) {
    smf_csc_t      csc = make_tridiag5();
    smf_analysis_t h   = nullptr;

    /* Null csc */
    int rc = smf_analyse(nullptr, &h);
    EXPECT_EQ(rc, SMF_ERR_INVALID_ARG);
    EXPECT_EQ(h, nullptr);

    /* Null out-pointer */
    rc = smf_analyse(&csc, nullptr);
    EXPECT_EQ(rc, SMF_ERR_INVALID_ARG);
}
