/// test_dod_regression.cpp — M5.S3 Definition-of-Done regression battery
///
/// Automated tests for all §15 DoD criteria from
/// ma97_solver_implementation_plan.md plus additional M5.S3 acceptance items:
///
///   §15 criterion 1: SPD matrices solved with residual < 1e-10
///   §15 criterion 2: Indefinite KKT matrices solved with residual < 1e-9
///   §15 criterion 3: Correct inertia reported (sum == n, counts non-negative)
///   §15 criterion 4: Analyse-once, factor-many (10 iterations)
///   §15 criterion 5: Singular/rank-deficient handling + continue_on_singular
///   §15 criterion 6: Diagnostics (rank, inertia, factor entries, timing)
///   §15 criterion 7: Deterministic single-thread behavior
///   M5.S3 additions: SolveJob::Forward differs from Full, factor_solve parity
///
/// Label: dod_regression
/// Runtime target: < 30 s total

#include "smf/solver.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace smf;

// ---------------------------------------------------------------------------
// Matrix construction helpers
// ---------------------------------------------------------------------------

/// 1-D Poisson-like tridiagonal SPD lower-CSC of order n.
///   A[j,j] = diag_val,  A[j+1,j] = offdiag_val
static CscLower make_poisson1d(int n, double diag_val = 4.0,
                                double offdiag_val = -1.0) {
    CscLower A;
    A.n = n;
    A.col_ptr.reserve(static_cast<std::size_t>(n + 1));
    A.row_idx.reserve(static_cast<std::size_t>(2 * n - 1));
    A.values.reserve(static_cast<std::size_t>(2 * n - 1));

    int nnz = 0;
    for (int j = 0; j < n; ++j) {
        A.col_ptr.push_back(nnz);
        A.row_idx.push_back(j);
        A.values.push_back(diag_val);
        ++nnz;
        if (j + 1 < n) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(offdiag_val);
            ++nnz;
        }
    }
    A.col_ptr.push_back(nnz);
    return A;
}

/// 50×50 indefinite matrix (lower-CSC).
///
/// 25 independent 2×2 blocks on the diagonal:
///   B = [[4, 1], [1, -1]]  → det = -5 ≠ 0, 1 pos + 1 neg eigenvalue.
///
/// Block-diagonal structure guarantees each 2×2 system is solved exactly
/// (no coupling = no pivoting interaction between blocks).
/// Inertia: 25 positive + 25 negative = 50.
static CscLower make_indef50() {
    const int n = 50;
    CscLower A;
    A.n = n;
    A.col_ptr.reserve(static_cast<std::size_t>(n + 1));
    A.row_idx.reserve(static_cast<std::size_t>(n + 25)); // 25 off-diagonals
    A.values.reserve(static_cast<std::size_t>(n + 25));

    int nnz = 0;
    for (int k = 0; k < 25; ++k) {
        const int j0 = 2 * k;
        const int j1 = 2 * k + 1;
        // Column j0: diagonal (4.0) and sub-diagonal (1.0)
        A.col_ptr.push_back(nnz);
        A.row_idx.push_back(j0); A.values.push_back(4.0); ++nnz;
        A.row_idx.push_back(j1); A.values.push_back(1.0); ++nnz;
        // Column j1: diagonal (-1.0)
        A.col_ptr.push_back(nnz);
        A.row_idx.push_back(j1); A.values.push_back(-1.0); ++nnz;
    }
    A.col_ptr.push_back(nnz);
    return A;
}

// ---------------------------------------------------------------------------
// Utility: general symmetric matrix-vector product y = A*x (lower-CSC A).
// ---------------------------------------------------------------------------
static void sym_matvec(const CscLower &A, const double *x, double *y) {
    const int n = A.n;
    std::fill(y, y + n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
            const int    i = A.row_idx[p];
            const double v = A.values[p];
            y[i] += v * x[j];
            if (i != j)
                y[j] += v * x[i];
        }
    }
}

/// Relative residual: ‖Ax − b‖ / (‖A‖_F · ‖x‖ + ‖b‖).
static double rel_residual(const CscLower &A, const double *x,
                            const double *b) {
    const int n = A.n;
    std::vector<double> Ax(static_cast<std::size_t>(n));
    sym_matvec(A, x, Ax.data());

    double res2 = 0.0, b2 = 0.0, x2 = 0.0, AF2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double r = Ax[i] - b[i];
        res2 += r * r;
        b2   += b[i] * b[i];
        x2   += x[i] * x[i];
    }
    for (int j = 0; j < n; ++j) {
        for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
            const double v = A.values[p];
            AF2 += (A.row_idx[p] == j) ? v * v : 2.0 * v * v;
        }
    }
    const double denom = std::sqrt(AF2) * std::sqrt(x2) + std::sqrt(b2);
    return (denom > 0.0) ? std::sqrt(res2) / denom : std::sqrt(res2);
}

// ===========================================================================
// DoD_SPD_Residual
// §15 criterion 1: 30×30 Poisson SPD, relative residual < 1e-10
// ===========================================================================
TEST(DodRegression, DoD_SPD_Residual) {
    const int n = 30;
    CscLower A = make_poisson1d(n);     // diag=4, offdiag=-1

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    Info   ai;
    Solver s;

    auto ak = s.analyse(A, ctrl, ai);
    ASSERT_NE(ak.get(), nullptr) << "analyse failed";
    ASSERT_EQ(ai.status, ErrorCode::Success);

    FactorKeep fk;
    Info fi;
    ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success);

    std::vector<double> b(static_cast<std::size_t>(n));
    std::vector<double> b0(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) b[i] = b0[i] = static_cast<double>(i + 1);

    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, b.data(), n), 0);

    const double rr = rel_residual(A, b.data(), b0.data());
    EXPECT_LT(rr, 1e-10) << "SPD30 residual=" << rr;
}

// ===========================================================================
// DoD_SPD_MultiRHS
// §15 criterion 1: same 30×30 system, 3 RHS vectors, each residual < 1e-10
// ===========================================================================
TEST(DodRegression, DoD_SPD_MultiRHS) {
    const int n    = 30;
    const int nrhs = 3;
    CscLower A = make_poisson1d(n);

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    Info   ai;
    Solver s;

    auto ak = s.analyse(A, ctrl, ai);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    Info fi;
    ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success);

    // Column-major n×nrhs buffer
    std::vector<double> B(static_cast<std::size_t>(n * nrhs));
    std::vector<double> B0(static_cast<std::size_t>(n * nrhs));
    for (int rhs = 0; rhs < nrhs; ++rhs) {
        for (int i = 0; i < n; ++i) {
            const double v = static_cast<double>(i + rhs + 1);
            B[static_cast<std::size_t>(rhs * n + i)]  = v;
            B0[static_cast<std::size_t>(rhs * n + i)] = v;
        }
    }

    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, B.data(), n, nrhs), 0);

    for (int rhs = 0; rhs < nrhs; ++rhs) {
        const double *x = B.data()  + rhs * n;
        const double *b = B0.data() + rhs * n;
        const double rr = rel_residual(A, x, b);
        EXPECT_LT(rr, 1e-10)
            << "MultiRHS rhs=" << rhs << " residual=" << rr;
    }
}

// ===========================================================================
// DoD_Indef_Residual
// §15 criterion 2: 50×50 indefinite KKT-like system, residual < 1e-9
// ===========================================================================
TEST(DodRegression, DoD_Indef_Residual) {
    CscLower A = make_indef50();
    const int n = A.n; // 50

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    Info   ai;
    Solver s;

    auto ak = s.analyse(A, ctrl, ai);
    ASSERT_NE(ak.get(), nullptr) << "analyse failed for indef50";

    FactorKeep fk;
    Info fi;
    const FactorStatus fs = s.factor(*ak, ctrl, fi, fk);
    ASSERT_EQ(fs, FactorStatus::Success)
        << "factor returned " << static_cast<int>(fs)
        << " (expected Success for invertible indefinite matrix)";

    std::vector<double> b(static_cast<std::size_t>(n));
    std::vector<double> b0(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        b[i] = b0[i] = static_cast<double>((i % 7) + 1);

    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, b.data(), n), 0);

    const double rr = rel_residual(A, b.data(), b0.data());
    EXPECT_LT(rr, 1e-9) << "indef50 residual=" << rr;
}

// ===========================================================================
// DoD_Indef_Inertia
// §15 criterion 3: inertia sum == n, all counts non-negative, negatives > 0
// ===========================================================================
TEST(DodRegression, DoD_Indef_Inertia) {
    // 4×4: tridiag(2,-1) for first 3 cols, last diagonal = -2
    // Full matrix:
    //   [ 2 -1  0  0 ]
    //   [-1  2 -1  0 ]
    //   [ 0 -1  2 -1 ]
    //   [ 0  0 -1 -2 ]
    // Eigenvalues: 3 positive, 1 negative.
    CscLower A;
    A.n       = 4;
    A.col_ptr = {0, 2, 4, 6, 7};
    A.row_idx = {0, 1, 1, 2, 2, 3, 3};
    A.values  = {2.0, -1.0, 2.0, -1.0, 2.0, -1.0, -2.0};

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    Info   ai;
    Solver s;

    auto ak = s.analyse(A, ctrl, ai);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    Info fi;
    ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success);

    // Inertia completeness: must sum to n
    EXPECT_EQ(fi.num_positive + fi.num_negative + fi.num_zero, 4)
        << "Inertia sum != n=4  (pos=" << fi.num_positive
        << " neg=" << fi.num_negative << " zero=" << fi.num_zero << ")";
    // All counts non-negative
    EXPECT_GE(fi.num_positive, 0);
    EXPECT_GE(fi.num_negative, 0);
    EXPECT_GE(fi.num_zero,     0);
    // Matrix has a negative eigenvalue
    EXPECT_GE(fi.num_negative, 1)
        << "Expected ≥1 negative eigenvalue; got " << fi.num_negative;
}

// ===========================================================================
// DoD_RepeatedFactor
// §15 criterion 4: analyse once, factor 10× with perturbed values,
//                  relative residual < 1e-9 each time
// ===========================================================================
TEST(DodRegression, DoD_RepeatedFactor) {
    const int n = 20;
    // diag=10 → very well-conditioned; ±1% diagonal perturbations stay SPD
    CscLower base = make_poisson1d(n, 10.0, -1.0);

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;

    Solver s;
    Info   ai;
    auto ak = s.analyse(base, ctrl, ai);
    ASSERT_NE(ak.get(), nullptr) << "analyse failed";
    ASSERT_EQ(ai.status, ErrorCode::Success);

    const std::vector<double> b0(static_cast<std::size_t>(n), 1.0);

    for (int iter = 0; iter < 10; ++iter) {
        // Slightly perturb diagonal (stay SPD, same sparsity pattern)
        CscLower Ap = base;
        for (int j = 0; j < n; ++j) {
            // diagonal is always first entry in each column for this matrix
            const int dp = Ap.col_ptr[j];
            Ap.values[static_cast<std::size_t>(dp)] +=
                0.01 * static_cast<double>(iter + 1);
        }
        // Update cached cleaned values to new numeric values
        ASSERT_EQ(ak->cleaned.values.size(), Ap.values.size())
            << "value-size mismatch at iter=" << iter;
        ak->cleaned.values = Ap.values;

        FactorKeep fk;
        Info fi;
        ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success)
            << "factor failed at iter=" << iter;

        std::vector<double> b(b0.begin(), b0.end());
        Info si;
        ASSERT_EQ(s.solve(fk, ctrl, si, b.data(), n), 0);

        const double rr = rel_residual(Ap, b.data(), b0.data());
        EXPECT_LT(rr, 1e-9) << "iter=" << iter << " residual=" << rr;
    }
}

// ===========================================================================
// DoD_Singular_Status
// §15 criterion 5: rank-deficient matrix returns FactorStatus::Singular
// ===========================================================================
TEST(DodRegression, DoD_Singular_Status) {
    // 4×4 with row/col 2 entirely absent → structurally zero
    //   col 0: (0,2),(1,1)  col 1: (1,3)  col 2: (empty)  col 3: (3,4)
    CscLower A;
    A.n       = 4;
    A.col_ptr = {0, 2, 3, 3, 4};
    A.row_idx = {0, 1, 1, 3};
    A.values  = {2.0, 1.0, 3.0, 4.0};

    Control ctrl;
    ctrl.matrix_type          = MatrixType::RealSymmetricIndefinite;
    ctrl.continue_on_singular = false;
    Info   info;
    Solver s;

    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    const FactorStatus st = s.factor(*ak, ctrl, info, fk);
    EXPECT_EQ(st, FactorStatus::Singular)
        << "Expected Singular for zero-column matrix";
}

// ===========================================================================
// DoD_Singular_Rank
// §15 criterion 6: info.numerical_rank < n for rank-deficient matrix
// ===========================================================================
TEST(DodRegression, DoD_Singular_Rank) {
    CscLower A;
    A.n       = 4;
    A.col_ptr = {0, 2, 3, 3, 4};
    A.row_idx = {0, 1, 1, 3};
    A.values  = {2.0, 1.0, 3.0, 4.0};

    Control ctrl;
    ctrl.matrix_type          = MatrixType::RealSymmetricIndefinite;
    ctrl.continue_on_singular = false;
    Info   info;
    Solver s;

    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    s.factor(*ak, ctrl, info, fk);
    EXPECT_LT(info.numerical_rank, 4)
        << "numerical_rank must be < n=4 for rank-deficient matrix";
}

// ===========================================================================
// DoD_ContinueOnSingular
// §15 criterion 5: continue_on_singular=true — factorisation completes,
//                  status is still Singular, solve does not crash
// ===========================================================================
TEST(DodRegression, DoD_ContinueOnSingular) {
    // 3×3 with zero row/col at index 1
    CscLower A;
    A.n       = 3;
    A.col_ptr = {0, 1, 1, 2};
    A.row_idx = {0,       2};
    A.values  = {2.0,     3.0};

    Control ctrl;
    ctrl.matrix_type          = MatrixType::RealSymmetricIndefinite;
    ctrl.small_pivot          = 1e-20;
    ctrl.continue_on_singular = true;   // ← key flag
    Info   info;
    Solver s;

    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    const FactorStatus st = s.factor(*ak, ctrl, info, fk);

    // Status must still be Singular (flag doesn't suppress the report)
    EXPECT_EQ(st, FactorStatus::Singular);
    // At least one zero pivot detected
    EXPECT_GE(info.num_zero, 1) << "expected ≥1 zero pivot";
    EXPECT_LT(info.numerical_rank, 3) << "numerical_rank must be < n=3";
    // Factor storage must be non-empty (factorisation completed)
    EXPECT_FALSE(fk.factor_values.empty())
        << "factor_values must be populated even for singular matrix";

    // Solve must not crash (result may be degraded for singular variable)
    double rhs[3] = {1.0, 0.0, 1.0};
    EXPECT_EQ(s.solve(fk, ctrl, info, rhs, 3), 0)
        << "solve must return 0 even for singular factorisation";
}

// ===========================================================================
// DoD_Info_Timing
// §15 criterion 6: analyse_seconds >= 0 and factor_seconds >= 0
// ===========================================================================
TEST(DodRegression, DoD_Info_Timing) {
    CscLower A = make_poisson1d(20);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;

    Solver s;
    Info   ai;
    auto ak = s.analyse(A, ctrl, ai);
    ASSERT_NE(ak.get(), nullptr);
    // Timers are populated; >= 0 is the safe bound (fast CPUs may report 0 ns)
    EXPECT_GE(ai.analyse_seconds, 0.0)
        << "analyse_seconds must be non-negative";

    FactorKeep fk;
    Info fi;
    ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success);
    EXPECT_GE(fi.factor_seconds, 0.0)
        << "factor_seconds must be non-negative";
}

// ===========================================================================
// DoD_Info_FactorEntries
// §15 criterion 6: predicted_factor_entries > 0 after analyse;
//                  actual_factor_entries > 0 after factor
// ===========================================================================
TEST(DodRegression, DoD_Info_FactorEntries) {
    CscLower A = make_poisson1d(20);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;

    Solver s;
    Info   ai;
    auto ak = s.analyse(A, ctrl, ai);
    ASSERT_NE(ak.get(), nullptr);
    EXPECT_GT(ai.predicted_factor_entries, 0LL)
        << "predicted_factor_entries must be > 0 after analyse";

    FactorKeep fk;
    Info fi;
    ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success);
    EXPECT_GT(fi.actual_factor_entries, 0LL)
        << "actual_factor_entries must be > 0 after factor";
}

// ===========================================================================
// DoD_Info_PredictionAccuracy
// §15 criterion 6: both predicted_factor_entries and actual_factor_entries are
// positive and actual is bounded by n² (the full dense matrix).
//
// Implementation note: predicted_factor_entries = Σ front_size_i  (counts
// rows per supernode), while actual_factor_entries = Σ (front_size_i × p_i)
// (counts stored doubles per supernode, where p_i is pivot width).  For p_i=1
// they are equal; for larger supernodes actual > predicted.  The n² bound is
// a hard sanity ceiling independent of supernode width.
// ===========================================================================
TEST(DodRegression, DoD_Info_PredictionAccuracy) {
    CscLower A = make_poisson1d(20);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;

    Solver s;
    Info   ai;
    auto ak = s.analyse(A, ctrl, ai);
    ASSERT_NE(ak.get(), nullptr);
    const LongInt predicted = ai.predicted_factor_entries;
    ASSERT_GT(predicted, 0LL) << "predicted_factor_entries must be > 0";

    FactorKeep fk;
    Info fi;
    ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success);
    const LongInt actual = fi.actual_factor_entries;
    ASSERT_GT(actual, 0LL) << "actual_factor_entries must be > 0";

    // actual >= predicted  (each supernode contributes f*p >= f entries)
    EXPECT_GE(actual, predicted)
        << "actual_factor_entries should be >= predicted";

    // Hard upper bound: no sparse factor can exceed the full dense matrix
    const LongInt n2 = static_cast<LongInt>(A.n) * static_cast<LongInt>(A.n);
    EXPECT_LE(actual, n2)
        << "actual_factor_entries=" << actual
        << " exceeds full-matrix bound n²=" << n2;
}

// ===========================================================================
// DoD_SolveJob_Forward
// Forward-only solve (SolveJob::Forward) gives a different result from Full
// ===========================================================================
TEST(DodRegression, DoD_SolveJob_Forward) {
    const int n = 10;
    CscLower A = make_poisson1d(n);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;

    Solver s;
    Info   ai;
    auto ak = s.analyse(A, ctrl, ai);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    Info fi;
    ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success);

    const std::vector<double> b0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Full solve  (ax = b → x)
    std::vector<double> b_full(b0);
    Info si_full;
    ASSERT_EQ(
        s.solve(fk, ctrl, si_full, b_full.data(), n, 1, SolveJob::Full), 0);

    // Forward-only solve  (Ly = b → y)
    std::vector<double> b_fwd(b0);
    Info si_fwd;
    ASSERT_EQ(
        s.solve(fk, ctrl, si_fwd, b_fwd.data(), n, 1, SolveJob::Forward), 0);

    // The two results must differ (forward-only omits D⁻¹ and L^T sweeps)
    bool all_equal = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(b_full[i] - b_fwd[i]) > 1e-14) {
            all_equal = false;
            break;
        }
    }
    EXPECT_FALSE(all_equal)
        << "SolveJob::Forward must differ from SolveJob::Full";
}

// ===========================================================================
// DoD_SolveJob_Full
// SolveJob::Full pipeline gives relative residual < 1e-10
// ===========================================================================
TEST(DodRegression, DoD_SolveJob_Full) {
    const int n = 15;
    CscLower A = make_poisson1d(n);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;

    Solver s;
    Info   ai;
    auto ak = s.analyse(A, ctrl, ai);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    Info fi;
    ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success);

    std::vector<double> b(static_cast<std::size_t>(n), 1.0);
    std::vector<double> b0(b);
    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, b.data(), n, 1, SolveJob::Full), 0);

    const double rr = rel_residual(A, b.data(), b0.data());
    EXPECT_LT(rr, 1e-10) << "SolveJob::Full residual=" << rr;
}

// ===========================================================================
// DoD_FactorSolve
// Combined factor_solve result matches separate analyse+factor+solve result
// ===========================================================================
TEST(DodRegression, DoD_FactorSolve) {
    const int n = 12;
    CscLower A = make_poisson1d(n);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;

    std::vector<double> b0(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) b0[i] = static_cast<double>(i + 1);

    // --- Separate path: analyse → factor → solve ---
    std::vector<double> b_sep(b0);
    {
        Solver s;
        Info ai, fi, si;
        auto ak = s.analyse(A, ctrl, ai);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success);
        ASSERT_EQ(s.solve(fk, ctrl, si, b_sep.data(), n), 0);
    }

    // --- Combined path: factor_solve ---
    std::vector<double> b_comb(b0);
    {
        Solver s;
        Info info;
        auto fk = s.factor_solve(A, ctrl, info, b_comb.data(), n);
        ASSERT_NE(fk.get(), nullptr) << "factor_solve returned null";
    }

    // Solutions must agree to near-machine precision
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(b_comb[i], b_sep[i], 1e-12)
            << "factor_solve vs separate mismatch at i=" << i;
}

// ===========================================================================
// DoD_Deterministic
// §15 criterion 7: two identical single-threaded runs produce byte-identical
//                  factor_values vectors
// ===========================================================================
TEST(DodRegression, DoD_Deterministic) {
    const int n = 20;
    CscLower A = make_poisson1d(n);
    Control ctrl;
    ctrl.matrix_type   = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.num_threads   = 1;
    ctrl.deterministic = true;

    std::vector<double> vals1, vals2;

    for (int run = 0; run < 2; ++run) {
        Solver s;
        Info   ai;
        auto ak = s.analyse(A, ctrl, ai);
        ASSERT_NE(ak.get(), nullptr);

        FactorKeep fk;
        Info fi;
        ASSERT_EQ(s.factor(*ak, ctrl, fi, fk), FactorStatus::Success);

        (run == 0 ? vals1 : vals2) = fk.factor_values;
    }

    ASSERT_EQ(vals1.size(), vals2.size())
        << "factor_values size changed between runs";
    for (std::size_t i = 0; i < vals1.size(); ++i)
        EXPECT_EQ(vals1[i], vals2[i])
            << "factor_values differ at index " << i;
}
