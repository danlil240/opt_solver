#include "smf/scaling.hpp"
#include "smf/solver.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace smf;

// Helper: build a diagonal CscLower matrix
static CscLower make_diag(const std::vector<double>& diag_vals) {
    int n = static_cast<int>(diag_vals.size());
    CscLower A;
    A.n = n;
    A.col_ptr.resize(n + 1);
    A.row_idx.resize(n);
    A.values.resize(n);
    for (int i = 0; i < n; ++i) {
        A.col_ptr[i] = i;
        A.row_idx[i] = i;
        A.values[i] = diag_vals[i];
    }
    A.col_ptr[n] = n;
    return A;
}

// Compute ||Ax - b||_inf / ||b||_inf
static double rel_residual(const CscLower& A, const double* x, const double* b) {
    int n = static_cast<int>(A.n);
    double num = 0.0, denom = 0.0;
    // A is lower triangular CSC representing symmetric matrix
    std::vector<double> Ax(n, 0.0);
    for (int col = 0; col < n; ++col) {
        for (Int k = A.col_ptr[col]; k < A.col_ptr[col + 1]; ++k) {
            int row = static_cast<int>(A.row_idx[k]);
            Ax[row] += A.values[k] * x[col];
            if (row != col) {
                Ax[col] += A.values[k] * x[row]; // symmetric contribution
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        num = std::max(num, std::abs(Ax[i] - b[i]));
        denom = std::max(denom, std::abs(b[i]));
    }
    return (denom > 0.0) ? num / denom : num;
}

TEST(ScalingEquilib, IllConditioned_DiagMatrix) {
    // Matrix: diag(1, 1e8, 1, 1e-8)
    // RHS b = A * ones => b = diag values
    std::vector<double> diag_vals = {1.0, 1e8, 1.0, 1e-8};
    int n = 4;
    std::vector<double> b = {1.0, 1e8, 1.0, 1e-8}; // A * [1,1,1,1]

    Solver solver;
    Control ctrl;
    Info info;

    // Without scaling
    {
        auto A = make_diag(diag_vals);
        std::vector<double> rhs = b;
        ctrl.scaling = ScalingMethod::None;
        auto fk = solver.factor_solve(A, ctrl, info, rhs.data(), n, 1);
        ASSERT_NE(fk, nullptr);
        // For diagonal matrix, no scaling still works (just checking it runs)
        // residual should be near machine epsilon
        auto A2 = make_diag(diag_vals);
        double resid = rel_residual(A2, rhs.data(), b.data());
        EXPECT_LT(resid, 1e-6) << "Without scaling residual too large: " << resid;
    }

    // With equilibration scaling
    {
        auto A = make_diag(diag_vals);
        std::vector<double> rhs = b;
        ctrl.scaling = ScalingMethod::Equilibration;
        auto fk = solver.factor_solve(A, ctrl, info, rhs.data(), n, 1);
        ASSERT_NE(fk, nullptr);
        auto A2 = make_diag(diag_vals);
        double resid = rel_residual(A2, rhs.data(), b.data());
        EXPECT_LT(resid, 1e-9) << "With scaling residual too large: " << resid;
    }
}

TEST(ScalingEquilib, ComputeScale_SimpleMatrix) {
    // 3x3 diagonal matrix: diag(4, 9, 1)
    // scale[i] = 1/sqrt(max|a_ij|) for diagonal = 1/sqrt(d_i)
    // After 1 iteration: scale = [0.5, 1/3, 1.0]
    // After convergence (max-norm, diag): scale[i] = 1/sqrt(d_i)
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 1, 2, 3};
    A.row_idx = {0, 1, 2};
    A.values  = {4.0, 9.0, 1.0};

    auto scale = compute_equilibration_scale(A, 1);
    ASSERT_EQ(static_cast<int>(scale.size()), 3);

    // After 1 iteration: scale[i] = 1/sqrt(d_i)
    EXPECT_NEAR(scale[0], 1.0 / std::sqrt(4.0), 1e-12);
    EXPECT_NEAR(scale[1], 1.0 / std::sqrt(9.0), 1e-12);
    EXPECT_NEAR(scale[2], 1.0 / std::sqrt(1.0), 1e-12);
}

TEST(ScalingEquilib, ApplyScaleRoundTrip) {
    // Solve: use scaling, verify solution is correct
    // A = diag(1e6, 1e-6), b = [1e6, 1e-6], solution = [1, 1]
    std::vector<double> diag_vals = {1e6, 1e-6};
    int n = 2;
    std::vector<double> b = {1e6, 1e-6};

    Solver solver;
    Control ctrl;
    Info info;
    ctrl.scaling = ScalingMethod::Equilibration;

    auto A = make_diag(diag_vals);
    std::vector<double> rhs = b;
    auto fk = solver.factor_solve(A, ctrl, info, rhs.data(), n, 1);
    ASSERT_NE(fk, nullptr);

    // solution should be [1, 1]
    EXPECT_NEAR(rhs[0], 1.0, 1e-9);
    EXPECT_NEAR(rhs[1], 1.0, 1e-9);
}
