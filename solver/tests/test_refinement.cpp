#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/refinement.hpp"
#include "smf/scaling.hpp"
#include "smf/scaling_matching.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace smf;

namespace {

// Badly scaled 4x4 indefinite KKT-like matrix with scales spanning 1e-6 to 1e6:
CscLower make_badly_scaled_indef(double scale_factor = 1e6) {
    CscLower A;
    A.n = 4;
    A.col_ptr = {0, 3, 6, 8, 9};
    A.row_idx = {
        0, 1, 2,
        1, 2, 3,
        2, 3,
        3
    };
    const double s = scale_factor;
    const double inv_s = 1.0 / scale_factor;
    // Let D_scale = diag(s, inv_s, s, inv_s), and base matrix B = [[4, 1, 2, 0], [1, -3, 0.5, 1], [2, 0.5, 0, 2], [0, 1, 2, -1]]
    // A_ij = d_i * B_ij * d_j
    const double d[4] = {s, inv_s, s, inv_s};
    A.values = {
        d[0] * 4.0 * d[0], d[1] * 1.0 * d[0], d[2] * 2.0 * d[0],
        d[1] * (-3.0) * d[1], d[2] * 0.5 * d[1], d[3] * 1.0 * d[1],
        d[2] * 0.0 * d[2], d[3] * 2.0 * d[2],
        d[3] * (-1.0) * d[3]
    };
    return A;
}

} // namespace

TEST(Refinement, OriginalSystemAfterScaling) {
    CscLower A = make_badly_scaled_indef(1e7);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.scaling = ScalingMethod::Equilibration;
    ctrl.refinement_max_iters = 5;

    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);
    ASSERT_EQ(fk.scale.size(), 4u);
    for (double s : fk.scale) {
        EXPECT_GT(s, 0.0);
        EXPECT_TRUE(std::isfinite(s));
    }

    // Manufactured solution x_true = [1e-7, 1e7, -2e-7, 3e7]
    std::vector<double> x_true = {1e-7, 1e7, -2e-7, 3e7};
    std::vector<double> b(4, 0.0);
    for (int j = 0; j < 4; ++j) {
        for (Int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
            int i = A.row_idx[p];
            double v = A.values[p];
            b[i] += v * x_true[j];
            if (i != j) b[j] += v * x_true[i];
        }
    }

    // Perturb initial x slightly so refinement must run at least 1 iteration and converge
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 4, 1, sws), 0);
    std::vector<double> x_perturbed = x;
    x_perturbed[0] *= (1.0 + 1e-6);
    x_perturbed[1] *= (1.0 - 1e-6);

    RefinementResult rr = refine(A, fk, ctrl, b, x_perturbed, 1, sws);
    EXPECT_EQ(rr.status, ErrorCode::Success);
    EXPECT_GE(rr.iterations, 1);
    EXPECT_LE(rr.eta_inf, 1e-10);
    EXPECT_LE(rr.omega, 1e-8);

    auto val = smf::bench::validate(A, b, x_perturbed, 1);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val));
}

TEST(Refinement, StagnationIsNotSuccess) {
    CscLower A = make_badly_scaled_indef(1.0);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.refinement_max_iters = 0; // 0 iterations allowed on a corrupted x -> must fail qualification!

    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);

    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> x_corrupted = {100.0, -50.0, 25.0, 10.0};
    SolveWorkspace sws;

    RefinementResult rr = refine(A, fk, ctrl, b, x_corrupted, 1, sws);
    EXPECT_NE(rr.status, ErrorCode::Success);
    EXPECT_GT(rr.eta_inf, 1e-10);

    // Also test stagnation when FactorKeep is deliberately corrupted so solve corrections do not improve x:
    FactorKeep fk_bad = fk;
    for (auto &de : fk_bad.diag_entries) {
        de.d00 = 1e20;
        de.d10 = 0.0;
        de.d11 = 1e20;
    }
    Control ctrl_iters = ctrl;
    ctrl_iters.refinement_max_iters = 5;
    RefinementResult rr_stag = refine(A, fk_bad, ctrl_iters, b, x_corrupted, 1, sws);
    EXPECT_NE(rr_stag.status, ErrorCode::Success);
}

TEST(Refinement, EveryRhs) {
    CscLower A = make_badly_scaled_indef(1e5);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.scaling = ScalingMethod::Equilibration;
    ctrl.refinement_max_iters = 5;

    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);

    // 3 RHS columns; corrupt the 3rd RHS column before calling refine()
    std::vector<double> b = {
        1.0, 2.0, 3.0, 4.0,
        -1.0, 0.5, 2.5, -3.0,
        4.0, -2.0, 1.0, 0.5
    };
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 4, 3, sws), 0);
    // Corrupt 3rd RHS column (offset 8..11)
    x[8] *= 1.01;
    x[9] *= 0.99;

    RefinementResult rr = refine(A, fk, ctrl, b, x, 3, sws);
    EXPECT_EQ(rr.status, ErrorCode::Success);
    EXPECT_GE(rr.iterations, 1);
    auto val = smf::bench::validate(A, b, x, 3);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val));
}

TEST(Refinement, RepeatedValuesSubstantialScalingChange) {
    CscLower base = make_badly_scaled_indef(1.0);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.scaling = ScalingMethod::Equilibration;

    Solver solver;
    Info info;
    auto ak = solver.analyse(base, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;

    for (double scale : {1.0, 1e3, 1e6, 1e9, 1e-4, 1e8}) {
        CscLower updated = make_badly_scaled_indef(scale);
        Info info_f;
        ASSERT_EQ(solver.factor(*ak, updated, ctrl, info_f, fk, nws), FactorStatus::Success);
        std::vector<double> b = {1.0, -1.0, 2.0, -2.0};
        std::vector<double> x = b;
        Info info_s;
        ASSERT_EQ(solver.solve(fk, ctrl, info_s, x.data(), 4, 1, sws), 0);
        RefinementResult rr = refine(updated, fk, ctrl, b, x, 1, sws);
        EXPECT_EQ(rr.status, ErrorCode::Success) << "scale=" << scale;
        auto val = smf::bench::validate(updated, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "scale=" << scale << " eta_inf=" << val.eta_inf << " omega=" << val.omega;
    }
}

TEST(Matching, SymmetryAndInertiaPreserved) {
    CscLower A = make_badly_scaled_indef(1e6);

    MatchingResult mr = compute_symmetric_matching(A);
    ASSERT_EQ(mr.status, ErrorCode::Success);
    ASSERT_EQ(mr.matched_rank, 4);
    ASSERT_EQ(mr.scale.size(), 4u);
    for (double s : mr.scale) {
        EXPECT_GT(s, 0.0);
        EXPECT_TRUE(std::isfinite(s));
    }

    // Solve with ScalingMethod::Matching and verify inertia matches unscaled B
    Control ctrl_unscaled;
    ctrl_unscaled.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl_unscaled.scaling = ScalingMethod::None;
    CscLower B = make_badly_scaled_indef(1.0);
    Solver solver;
    Info info_b;
    auto ak_b = solver.analyse(B, ctrl_unscaled, info_b);
    FactorKeep fk_b;
    ASSERT_EQ(solver.factor(*ak_b, ctrl_unscaled, info_b, fk_b), FactorStatus::Success);

    Control ctrl_match;
    ctrl_match.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl_match.scaling = ScalingMethod::Matching;
    Info info_m;
    auto ak_m = solver.analyse(A, ctrl_match, info_m);
    ASSERT_NE(ak_m.get(), nullptr);
    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk_m;
    ASSERT_EQ(solver.factor(*ak_m, A, ctrl_match, info_m, fk_m, nws), FactorStatus::Success);
    EXPECT_EQ(info_m.num_positive, info_b.num_positive);
    EXPECT_EQ(info_m.num_negative, info_b.num_negative);
    EXPECT_EQ(info_m.num_zero, info_b.num_zero);

    std::vector<double> b = {1.0, -2.0, 3.0, -4.0};
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk_m, ctrl_match, info_m, x.data(), 4, 1, sws), 0);
    RefinementResult rr = refine(A, fk_m, ctrl_match, b, x, 1, sws);
    EXPECT_EQ(rr.status, ErrorCode::Success);
    auto val = smf::bench::validate(A, b, x, 1);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val));
}

TEST(Matching, StructurallySingular) {
    // 3x3 matrix where row/col 1 has no non-zero entries
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 1, 1, 2};
    A.row_idx = {0, 2};
    A.values = {4.0, 5.0};

    MatchingResult mr = compute_symmetric_matching(A);
    EXPECT_EQ(mr.status, ErrorCode::SingularMatrix);
    EXPECT_LT(mr.matched_rank, 3);

    std::vector<double> scale;
    EXPECT_EQ(compute_matching_scale(A, scale, 0), ErrorCode::SingularMatrix);
}

TEST(Matching, ScaledKktSequenceWithDelayedPivots) {
    // Indefinite KKT sequence where leaf node 0 has 0 diagonal (must delay to parent)
    // and scaling varies by 1e7 across primal/dual blocks:
    CscLower A;
    A.n = 4;
    A.col_ptr = {0, 2, 4, 6, 7};
    A.row_idx = {
        0, 3,    // col 0: A(0,0)=0, A(3,0)=1e7
        1, 2,    // col 1: A(1,1)=0, A(2,1)=1e-7
        2, 3,    // col 2: A(2,2)=1e-14, A(3,2)=1.0
        3        // col 3: A(3,3)=-1e14
    };
    A.values = {
        0.0, 1e7,
        0.0, 1e-7,
        1e-14, 1.0,
        -1e14
    };

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.nemin = 1;
    ctrl.scaling = ScalingMethod::Matching;
    ctrl.pivot_u = 0.01;

    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;

    for (int step = 0; step < 5; ++step) {
        CscLower As = A;
        const double mult = 1.0 + 0.5 * step;
        As.values[1] = 1e7 * mult;
        As.values[3] = 1e-7 / mult;
        As.values[6] = -1e14 * mult;

        ASSERT_EQ(solver.factor(*ak, As, ctrl, info, fk, nws), FactorStatus::Success);
        EXPECT_GT(info.delayed_pivots, 0);
        EXPECT_EQ(info.num_zero, 0);

        std::vector<double> b = {1e7, 1e-7, -2e-7, 3e7};
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 4, 1, sws), 0);
        RefinementResult rr = refine(As, fk, ctrl, b, x, 1, sws);
        EXPECT_EQ(rr.status, ErrorCode::Success);
        auto val = smf::bench::validate(As, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "step=" << step << " eta_inf=" << val.eta_inf << " omega=" << val.omega;
    }
}
