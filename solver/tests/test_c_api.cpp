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

/* =========================================================================
 * Test 7 — smf_factor factors the values it is given, not the analysed ones
 *
 * Regression: smf_factor copied `values` into a local matrix and then called
 * the Solver::factor overload without it, so it factored the values that
 * were passed to smf_analyse.
 * ========================================================================= */

TEST(CApiTest, CApiFactorUsesNewValuesDiagonal) {
    static const int    col_ptr[] = {0, 1, 2};
    static const int    row_idx[] = {0, 1};
    static const double analysed[] = {1., -1.};
    static const double updated[]  = {2., -3.};

    smf_csc_t csc;
    csc.n       = 2;
    csc.col_ptr = col_ptr;
    csc.row_idx = row_idx;
    csc.values  = analysed;

    smf_analysis_t ah = nullptr;
    smf_factor_t   fh = nullptr;
    ASSERT_EQ(smf_analyse(&csc, &ah), SMF_OK);
    ASSERT_EQ(smf_factor(ah, updated, SMF_INDEF, &fh), SMF_OK);

    double x[2] = {2., -3.};
    ASSERT_EQ(smf_solve(ah, fh, 1, x, SMF_SOLVE_FULL), SMF_OK);
    EXPECT_NEAR(x[0], 1., 1e-12);
    EXPECT_NEAR(x[1], 1., 1e-12);

    smf_inertia_t iner = {-1, -1, -1};
    ASSERT_EQ(smf_inertia(fh, &iner), SMF_OK);
    EXPECT_EQ(iner.pos,  1);
    EXPECT_EQ(iner.neg,  1);
    EXPECT_EQ(iner.zero, 0);

    smf_free_factor(&fh);
    smf_free_analysis(&ah);
}

/* Full symmetric y = A x for a 3x3 lower-CSC matrix with pattern
 * (0,0) (1,0) | (1,1) (2,1) | (2,2). */
static void matvec_tri3(const double* v, const double* x, double* y) {
    y[0] = v[0] * x[0] + v[1] * x[1];
    y[1] = v[1] * x[0] + v[2] * x[1] + v[3] * x[2];
    y[2] =               v[3] * x[1] + v[4] * x[2];
}

/*
 * Off-diagonal pattern: analyse an SPD tridiagonal matrix, then factor values
 * that change the off-diagonals' magnitude and the diagonal's sign pattern
 * (inertia 1+, 2-), then refactor the same analysis with a second set.
 * A copy that missed any entry (diagonal or off-diagonal) gives a wrong
 * solution or inertia.
 */
TEST(CApiTest, CApiFactorUsesNewValuesOffDiagonal) {
    static const int    col_ptr[] = {0, 2, 4, 5};
    static const int    row_idx[] = {0, 1, 1, 2, 2};
    static const double analysed[] = {4., 1., 4., 1., 4.};   // SPD, inertia 3+
    // [[1 2 0] [2 -1 3] [0 3 -2]]: leading minors 1, -5, 1 -> inertia 1+, 2-
    static const double updated1[] = {1., 2., -1., 3., -2.};
    // [[-5 0.5 0] [0.5 7 -4] [0 -4 6]]: leading minors -5, -35.25, -131.5
    // -> inertia 2+, 1-
    static const double updated2[] = {-5., 0.5, 7., -4., 6.};

    smf_csc_t csc;
    csc.n       = 3;
    csc.col_ptr = col_ptr;
    csc.row_idx = row_idx;
    csc.values  = analysed;

    smf_analysis_t ah = nullptr;
    ASSERT_EQ(smf_analyse(&csc, &ah), SMF_OK);

    struct Case { const double* v; int pos; int neg; };
    const Case cases[] = {{updated1, 1, 2}, {updated2, 2, 1}, {analysed, 3, 0}};
    for (const Case& c : cases) {
        smf_factor_t fh = nullptr;
        ASSERT_EQ(smf_factor(ah, c.v, SMF_INDEF, &fh), SMF_OK);

        const double x_true[3] = {1., 2., 3.};
        double x[3];
        matvec_tri3(c.v, x_true, x);
        ASSERT_EQ(smf_solve(ah, fh, 1, x, SMF_SOLVE_FULL), SMF_OK);
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(x[i], x_true[i], 1e-12) << "entry " << i << " for values[0]=" << c.v[0];

        smf_inertia_t iner = {-1, -1, -1};
        ASSERT_EQ(smf_inertia(fh, &iner), SMF_OK);
        EXPECT_EQ(iner.pos,  c.pos) << "values[0]=" << c.v[0];
        EXPECT_EQ(iner.neg,  c.neg) << "values[0]=" << c.v[0];
        EXPECT_EQ(iner.zero, 0);
        smf_free_factor(&fh);
    }

    smf_free_analysis(&ah);
}

/* Positive-definite path: analyse tridiag5, factor different SPD values. */
TEST(CApiTest, CApiFactorUsesNewValuesPosdef) {
    smf_csc_t      csc = make_tridiag5();
    smf_analysis_t ah  = nullptr;
    smf_factor_t   fh  = nullptr;
    ASSERT_EQ(smf_analyse(&csc, &ah), SMF_OK);

    // diag 10, sub-diagonal 3: still SPD, different from the analysed 4 / -1.
    const double updated[] = {10., 3., 10., 3., 10., 3., 10., 3., 10.};
    ASSERT_EQ(smf_factor(ah, updated, SMF_POSDEF, &fh), SMF_OK);

    const double x_true[5] = {1., -2., 3., -4., 5.};
    double x[5];
    for (int i = 0; i < 5; ++i) {
        x[i] = 10. * x_true[i];
        if (i > 0) x[i] += 3. * x_true[i - 1];
        if (i < 4) x[i] += 3. * x_true[i + 1];
    }
    ASSERT_EQ(smf_solve(ah, fh, 1, x, SMF_SOLVE_FULL), SMF_OK);
    for (int i = 0; i < 5; ++i)
        EXPECT_NEAR(x[i], x_true[i], 1e-12) << "entry " << i;

    smf_free_factor(&fh);
    smf_free_analysis(&ah);
}

/* Factoring indefinite values as SMF_POSDEF reports SMF_ERR_SINGULAR. */
TEST(CApiTest, CApiFactorPosdefOnIndefiniteValues) {
    smf_csc_t      csc = make_tridiag5();
    smf_analysis_t ah  = nullptr;
    smf_factor_t   fh  = nullptr;
    ASSERT_EQ(smf_analyse(&csc, &ah), SMF_OK);

    const double indefinite[] = {4., -1., -4., -1., 4., -1., 4., -1., 4.};
    EXPECT_EQ(smf_factor(ah, indefinite, SMF_POSDEF, &fh), SMF_ERR_SINGULAR);
    EXPECT_EQ(fh, nullptr);

    smf_free_analysis(&ah);
}

/*
 * A pattern that analyse cleans: a duplicate (1,0), an out-of-range row and
 * an upper-triangle entry. smf_factor takes values in the caller's entry
 * order: duplicates are summed, the dropped entries' values are ignored.
 */
TEST(CApiTest, CApiFactorUsesNewValuesUncleanPattern) {
    static const int col_ptr[] = {0, 4, 7, 8};
    //                            col 0: (0,0) (1,0) (1,0) (3,0)=out of range
    //                            col 1: (0,1)=upper (1,1) (2,1)   col 2: (2,2)
    static const int row_idx[] = {0, 1, 1, 3, 0, 1, 2, 2};
    static const double analysed[] = {4., .5, .5, 99., 77., 4., 1., 4.};
    // [[1 2 0] [2 -1 3] [0 3 -2]] -> inertia 1+, 2-
    static const double indef[] = {1., 1.5, .5, 1e6, 1e6, -1., 3., -2.};
    // [[10 3 0] [3 10 3] [0 3 10]] -> SPD
    static const double spd[] = {10., 2., 1., 1e6, 1e6, 10., 3., 10.};

    smf_csc_t csc;
    csc.n       = 3;
    csc.col_ptr = col_ptr;
    csc.row_idx = row_idx;
    csc.values  = analysed;

    smf_analysis_t ah = nullptr;
    ASSERT_EQ(smf_analyse(&csc, &ah), SMF_OK);

    struct Case { const double* v; int mtype; double a[5]; int pos; int neg; };
    const Case cases[] = {
        {indef, SMF_INDEF,  {1., 2., -1., 3., -2.}, 1, 2},
        {spd,   SMF_POSDEF, {10., 3., 10., 3., 10.}, 3, 0},
        {spd,   SMF_INDEF,  {10., 3., 10., 3., 10.}, 3, 0},
    };
    for (const Case& c : cases) {
        smf_factor_t fh = nullptr;
        ASSERT_EQ(smf_factor(ah, c.v, c.mtype, &fh), SMF_OK);

        const double x_true[3] = {1., 2., 3.};
        double x[3];
        matvec_tri3(c.a, x_true, x);
        ASSERT_EQ(smf_solve(ah, fh, 1, x, SMF_SOLVE_FULL), SMF_OK);
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(x[i], x_true[i], 1e-12) << "entry " << i << " mtype " << c.mtype;

        if (c.mtype == SMF_INDEF) {
            smf_inertia_t iner = {-1, -1, -1};
            ASSERT_EQ(smf_inertia(fh, &iner), SMF_OK);
            EXPECT_EQ(iner.pos, c.pos);
            EXPECT_EQ(iner.neg, c.neg);
        }
        smf_free_factor(&fh);
    }
    smf_free_analysis(&ah);
}

/* Non-finite values are an invalid argument, not a singular matrix. */
TEST(CApiTest, CApiFactorNonFiniteValues) {
    smf_csc_t      csc = make_tridiag5();
    smf_analysis_t ah  = nullptr;
    ASSERT_EQ(smf_analyse(&csc, &ah), SMF_OK);

    double bad[9];
    for (int i = 0; i < 9; ++i) bad[i] = tridiag5_values[i];
    bad[2] = std::nan("");
    for (int mtype : {SMF_POSDEF, SMF_INDEF}) {
        smf_factor_t fh = nullptr;
        EXPECT_EQ(smf_factor(ah, bad, mtype, &fh), SMF_ERR_INVALID_ARG) << "mtype " << mtype;
        EXPECT_EQ(fh, nullptr);
    }
    smf_free_analysis(&ah);
}
