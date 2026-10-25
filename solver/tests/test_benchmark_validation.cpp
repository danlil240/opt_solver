#include "../benchmarks/common/validation.hpp"
#include "smf/csc_matrix.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

smf::CscLower make_test_spd_3x3() {
    // Symmetric matrix:
    // [ 4 -1  0 ]
    // [-1  4 -1 ]
    // [ 0 -1  4 ]
    smf::CscLower A;
    A.n = 3;
    A.col_ptr = {0, 2, 4, 5};
    A.row_idx = {0, 1, 1, 2, 2};
    A.values = {4.0, -1.0, 4.0, -1.0, 4.0};
    return A;
}

} // namespace

TEST(BenchmarkValidation, ExactManufacturedSolutionPasses) {
    smf::CscLower A = make_test_spd_3x3();
    std::vector<double> x = {1.0, 2.0, 3.0};
    // b = A * x = [4*1 - 2, -1 + 8 - 3, -2 + 12] = [2, 4, 10]
    std::vector<double> b = {2.0, 4.0, 10.0};

    smf::bench::Validation v = smf::bench::validate(A, b, x, 1);
    EXPECT_TRUE(v.finite);
    EXPECT_LE(v.eta_inf, 1e-15);
    EXPECT_LE(v.omega, 1e-15);
    EXPECT_LE(v.relative_residual, 1e-15);
    EXPECT_TRUE(smf::bench::is_qualified_validation(v));
    EXPECT_EQ(smf::bench::qualify_outcome(smf::bench::Outcome::Success, v),
              smf::bench::Outcome::Success);
}

TEST(BenchmarkValidation, CorruptedSolutionFails) {
    smf::CscLower A = make_test_spd_3x3();
    std::vector<double> x = {1.0, 2.05, 3.0}; // perturbed by 5e-2
    std::vector<double> b = {2.0, 4.0, 10.0};

    smf::bench::Validation v = smf::bench::validate(A, b, x, 1);
    EXPECT_TRUE(v.finite);
    EXPECT_GT(v.eta_inf, 1e-3);
    EXPECT_GT(v.omega, 1e-3);
    EXPECT_FALSE(smf::bench::is_qualified_validation(v));
}

TEST(BenchmarkValidation, NanOrInfSolutionFails) {
    smf::CscLower A = make_test_spd_3x3();
    std::vector<double> b = {2.0, 4.0, 10.0};

    std::vector<double> x_nan = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    smf::bench::Validation v_nan = smf::bench::validate(A, b, x_nan, 1);
    EXPECT_FALSE(v_nan.finite);
    EXPECT_FALSE(smf::bench::is_qualified_validation(v_nan));
    EXPECT_EQ(smf::bench::qualify_outcome(smf::bench::Outcome::Success, v_nan),
              smf::bench::Outcome::Inaccurate);

    std::vector<double> x_inf = {1.0, 2.0, std::numeric_limits<double>::infinity()};
    smf::bench::Validation v_inf = smf::bench::validate(A, b, x_inf, 1);
    EXPECT_FALSE(v_inf.finite);
    EXPECT_FALSE(smf::bench::is_qualified_validation(v_inf));
    EXPECT_EQ(smf::bench::qualify_outcome(smf::bench::Outcome::Success, v_inf),
              smf::bench::Outcome::Inaccurate);
}

TEST(BenchmarkValidation, ZeroRhsDenominatorConvention) {
    smf::CscLower A = make_test_spd_3x3();
    std::vector<double> b_zero = {0.0, 0.0, 0.0};
    std::vector<double> x_zero = {0.0, 0.0, 0.0};

    // Zero denominator with zero residual gives 0.0 and passes.
    smf::bench::Validation v_exact = smf::bench::validate(A, b_zero, x_zero, 1);
    EXPECT_TRUE(v_exact.finite);
    EXPECT_DOUBLE_EQ(v_exact.eta_inf, 0.0);
    EXPECT_DOUBLE_EQ(v_exact.omega, 0.0);
    EXPECT_TRUE(smf::bench::is_qualified_validation(v_exact));

    // Zero row in A with nonzero b_i and x = 0 -> zero componentwise denominator?
    // Wait: if A = 0 and x = 0 and b = [0, 0, 0], residual is 0 -> 0.
    // What if x != 0 on b = 0 for nonsingular A? Then Ax != 0, so residual != 0 and denominator > 0:
    std::vector<double> x_nonzero = {1e-3, 0.0, 0.0};
    smf::bench::Validation v_bad = smf::bench::validate(A, b_zero, x_nonzero, 1);
    EXPECT_GT(v_bad.eta_inf, 0.1);
    EXPECT_GT(v_bad.omega, 0.1);
    EXPECT_FALSE(smf::bench::is_qualified_validation(v_bad));

    // Also test a component i where (|A||x| + |b|)_i == 0 while |b - Ax|_i == 0 vs > 0:
    // Row 2 decoupled: A(2,2) = 4, b = [2, 0, 0], x = [0.5, 0, 0] on diag(4, 4, 4)
    smf::CscLower D;
    D.n = 2;
    D.col_ptr = {0, 1, 2};
    D.row_idx = {0, 1};
    D.values = {4.0, 4.0};
    std::vector<double> b_dec = {4.0, 0.0};
    std::vector<double> x_dec = {1.0, 0.0}; // component 1 has (|D||x| + |b|)_1 == 0 and r_1 == 0
    smf::bench::Validation v_dec = smf::bench::validate(D, b_dec, x_dec, 1);
    EXPECT_TRUE(v_dec.finite);
    EXPECT_DOUBLE_EQ(v_dec.omega, 0.0);
    EXPECT_TRUE(smf::bench::is_qualified_validation(v_dec));
}

TEST(BenchmarkValidation, BadSecondRhsFails) {
    smf::CscLower A = make_test_spd_3x3();
    // 2 RHS columns (column-major, length 3 * 2 = 6):
    // RHS 0: x0 = [1, 2, 3], b0 = [2, 4, 10] (exact)
    // RHS 1: x1 = [1, 1, 1] (wrong for b1 = [2, 4, 10])
    std::vector<double> b = {2.0, 4.0, 10.0, 2.0, 4.0, 10.0};
    std::vector<double> x = {1.0, 2.0, 3.0, 1.0, 1.0, 1.0};

    smf::bench::Validation v = smf::bench::validate(A, b, x, 2);
    EXPECT_TRUE(v.finite);
    EXPECT_GT(v.eta_inf, 1e-2);
    EXPECT_GT(v.omega, 1e-2);
    EXPECT_FALSE(smf::bench::is_qualified_validation(v));
    EXPECT_EQ(smf::bench::qualify_outcome(smf::bench::Outcome::Success, v),
              smf::bench::Outcome::Inaccurate);
}

TEST(BenchmarkValidation, SuccessWithExcessiveBackwardErrorBecomesInaccurate) {
    smf::CscLower A = make_test_spd_3x3();
    std::vector<double> b = {2.0, 4.0, 10.0};
    // Perturb x enough to exceed eta_inf <= 1e-10 or omega <= 1e-8
    std::vector<double> x = {1.0 + 1e-6, 2.0, 3.0};

    smf::bench::Validation v = smf::bench::validate(A, b, x, 1);
    EXPECT_TRUE(v.finite);
    EXPECT_GT(v.omega, smf::bench::kMaxQualifiedOmega);
    EXPECT_EQ(smf::bench::qualify_outcome(smf::bench::Outcome::Success, v),
              smf::bench::Outcome::Inaccurate);
}
