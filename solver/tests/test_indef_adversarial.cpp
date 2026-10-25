#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/dense_indef.hpp"
#include "smf/factor_indef.hpp"
#include "smf/info.hpp"
#include "smf/pivoting.hpp"
#include "smf/solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace smf;

namespace {

CscLower make_exchange_2x2(double scale = 1.0) {
    // A = scale * [[0, 1], [1, 0]]
    // Lower CSC with explicit zero diagonal entries so supernode amalgamation forms a single 2x2 front
    CscLower A;
    A.n = 2;
    A.col_ptr = {0, 2, 3};
    A.row_idx = {0, 1, 1};
    A.values = {0.0, 1.0 * scale, 0.0};
    return A;
}

CscLower make_spec_3x3_secondary_zero_diag() {
    // Spec reproducer #2:
    // A = [[0.005, 1, 0],
    //      [1,     0, 2],
    //      [0,     2, 1]]
    // Store all lower-triangle entries so supernode amalgamation creates one full 3x3 front.
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 3, 5, 6};
    A.row_idx = {0, 1, 2, 1, 2, 2};
    A.values = {0.005, 1.0, 0.0, 0.0, 2.0, 1.0};
    return A;
}

CscLower make_generic_front_exchange_20x20() {
    // Dense 20x20 lower triangle (exercises p > 16 generic solve kernel)
    // First 2x2 block is [[0, 1], [1, 0]], remaining 18x18 block is diagonal dominant SPD.
    const int n = 20;
    CscLower A;
    A.n = n;
    A.col_ptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            A.row_idx.push_back(i);
            double v = 0.0;
            if (j == 0 && i == 0) v = 0.0;
            else if (j == 0 && i == 1) v = 1.0;
            else if (j == 1 && i == 1) v = 0.0;
            else if (i == j) v = 10.0 + static_cast<double>(i);
            else if (i == j + 1 && j >= 2) v = -0.5;
            else v = 1e-12; // structural entry to force a single 20x20 supernode front
            A.values.push_back(v);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

} // namespace

TEST(IndefAdversarial, TwoByTwoExchange) {
    CscLower A = make_exchange_2x2();
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.nemin = 8;
    ctrl.pivot_u = 0.01;

    Info info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_EQ(ak->supernodes.size(), 1u) << "Expected single full 2x2 front";

    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
    EXPECT_EQ(info.num_positive, 1);
    EXPECT_EQ(info.num_negative, 1);
    EXPECT_EQ(info.num_zero, 0);

    // Single RHS: b = [1, 2] -> x = [2, 1]
    std::vector<double> b = {1.0, 2.0};
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 2, 1), 0);
    EXPECT_NEAR(x[0], 2.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0, 1e-12);

    auto val1 = smf::bench::validate(A, b, x, 1);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val1));

    // Multiple RHS (nrhs = 3):
    // col 0: [1, 2] -> [2, 1]
    // col 1: [-3, 5] -> [5, -3]
    // col 2: [0.5, -1.5] -> [-1.5, 0.5]
    std::vector<double> b_multi = {1.0, 2.0, -3.0, 5.0, 0.5, -1.5};
    std::vector<double> x_multi = b_multi;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x_multi.data(), 2, 3), 0);
    EXPECT_NEAR(x_multi[0], 2.0, 1e-12);
    EXPECT_NEAR(x_multi[1], 1.0, 1e-12);
    EXPECT_NEAR(x_multi[2], 5.0, 1e-12);
    EXPECT_NEAR(x_multi[3], -3.0, 1e-12);
    EXPECT_NEAR(x_multi[4], -1.5, 1e-12);
    EXPECT_NEAR(x_multi[5], 0.5, 1e-12);

    auto val_multi = smf::bench::validate(A, b_multi, x_multi, 3);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val_multi));

    // Generic kernel (p = 20 > 16) with a 2x2 exchange block:
    CscLower A20 = make_generic_front_exchange_20x20();
    Info info20;
    auto ak20 = solver.analyse(A20, ctrl, info20);
    ASSERT_NE(ak20.get(), nullptr);
    ASSERT_EQ(ak20->supernodes.size(), 1u);
    FactorKeep fk20;
    ASSERT_EQ(solver.factor(*ak20, ctrl, info20, fk20), FactorStatus::Success);
    std::vector<double> b20(20, 1.0);
    b20[0] = 1.0;
    b20[1] = 2.0;
    std::vector<double> x20 = b20;
    ASSERT_EQ(solver.solve(fk20, ctrl, info20, x20.data(), 20, 1), 0);
    auto val20 = smf::bench::validate(A20, b20, x20, 1);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val20))
        << "eta_inf=" << val20.eta_inf << " omega=" << val20.omega;
}

TEST(IndefAdversarial, SecondaryPivotZeroDiagonal) {
    // Direct front kernel test on A = [[0.005, 1, 0], [1, 0, 2], [0, 2, 1]], u = 0.01
    {
        AlignedArena arena(4096);
        std::vector<Int> rows = {0, 1, 2};
        FrontalMatrix F(3, 3, rows, rows, arena);
        F.zero();
        F.at(0, 0) = 0.005;
        F.at(1, 0) = 1.0;
        F.at(2, 0) = 0.0;
        F.at(1, 1) = 0.0;
        F.at(2, 1) = 2.0;
        F.at(2, 2) = 1.0;

        PivotResult pr = choose_pivot(F, 0, 3, 0.01, 1e-20);
        // Row 1 has a ZERO diagonal (F(1,1) == 0). It must NEVER be selected as a 1x1 pivot!
        EXPECT_FALSE(pr.decision == PivotDecision::Accept1x1 && pr.col0 == 1)
            << "Secondary test falsely selected zero diagonal in row 1 as 1x1 pivot";
        EXPECT_EQ(pr.decision, PivotDecision::Accept2x2);
    }

    // Dense helper test on the same 3x3 matrix
    {
        double A_dense[9] = {
            0.005, 1.0, 0.0,
            0.0,   0.0, 2.0,
            0.0,   0.0, 1.0
        };
        std::vector<Pivot> pivots;
        InertiaCounts inertia{};
        FactorStatus st = dense_ldlt_indef(A_dense, 3, 3, 0.01, 1e-20, pivots, inertia);
        EXPECT_EQ(st, FactorStatus::Success);
        for (int i = 0; i < 9; ++i) {
            EXPECT_TRUE(std::isfinite(A_dense[i])) << "Non-finite entry in dense factor at " << i;
        }
        for (const auto& piv : pivots) {
            if (piv.type == PivotType::OneByOne) {
                EXPECT_GT(std::abs(piv.d00), 1e-12) << "Accepted zero 1x1 divisor in dense_ldlt_indef";
            }
        }
        // Determinant is -1.02 < 0, trace = 1.005 > 0 -> 2 positive, 1 negative
        EXPECT_EQ(inertia.positive, 2);
        EXPECT_EQ(inertia.negative, 1);
        EXPECT_EQ(inertia.zero, 0);
    }

    // Public Solver API test on the same 3x3 matrix
    {
        CscLower A = make_spec_3x3_secondary_zero_diag();
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.nemin = 8;
        ctrl.pivot_u = 0.01;

        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);

        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
        for (double v : fk.factor_values) {
            EXPECT_TRUE(std::isfinite(v));
        }
        for (const auto& de : fk.diag_entries) {
            if (de.tag == 1) {
                EXPECT_GT(std::abs(de.d00), 1e-12);
            }
        }
        EXPECT_EQ(info.num_positive, 2);
        EXPECT_EQ(info.num_negative, 1);
        EXPECT_EQ(info.num_zero, 0);

        std::vector<double> b = {1.0, -2.0, 3.0};
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 3, 1), 0);
        auto val = smf::bench::validate(A, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "eta_inf=" << val.eta_inf << " omega=" << val.omega;
    }

    // Non-adjacent numeric swap case:
    // 3x3 front where col 0 has tiny diagonal (0), zero at (1,0), large at (2,0) = 2.0,
    // and row 2 has a dominant diagonal A(2,2) = 5.0 so secondary test swaps 0 <-> 2 (1x1)
    // OR 2x2 pivot swaps 1 <-> 2! Test both!
    {
        // Case A: Non-adjacent 1x1 swap (0 <-> 2): A(0,0)=0, A(1,0)=0, A(2,0)=1, A(1,1)=3, A(2,1)=0.5, A(2,2)=10
        CscLower A_swap1;
        A_swap1.n = 3;
        A_swap1.col_ptr = {0, 3, 5, 6};
        A_swap1.row_idx = {0, 1, 2, 1, 2, 2};
        A_swap1.values = {0.0, 0.0, 1.0, 3.0, 0.5, 10.0};

        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.pivot_u = 0.1;
        Info info;
        Solver solver;
        auto ak = solver.analyse(A_swap1, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
        std::vector<double> b = {1.0, 2.0, 3.0};
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 3, 1), 0);
        auto val = smf::bench::validate(A_swap1, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "Non-adjacent 1x1 swap failed: eta_inf=" << val.eta_inf << " omega=" << val.omega;

        // Case B: Non-adjacent 2x2 swap (1 <-> 2 for 2x2 pivot (0, 2)):
        // A(0,0)=0, A(1,0)=0, A(2,0)=2.0, A(1,1)=4.0, A(2,1)=0.5, A(2,2)=0.0
        CscLower A_swap2;
        A_swap2.n = 3;
        A_swap2.col_ptr = {0, 3, 5, 6};
        A_swap2.row_idx = {0, 1, 2, 1, 2, 2};
        A_swap2.values = {0.0, 0.0, 2.0, 4.0, 0.5, 0.0};

        auto ak2 = solver.analyse(A_swap2, ctrl, info);
        ASSERT_NE(ak2.get(), nullptr);
        FactorKeep fk2;
        ASSERT_EQ(solver.factor(*ak2, ctrl, info, fk2), FactorStatus::Success);
        std::vector<double> x2 = b;
        ASSERT_EQ(solver.solve(fk2, ctrl, info, x2.data(), 3, 1), 0);
        auto val2 = smf::bench::validate(A_swap2, b, x2, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val2))
            << "Non-adjacent 2x2 swap failed: eta_inf=" << val2.eta_inf << " omega=" << val2.omega;
    }
}

TEST(IndefAdversarial, ExtremeScaling1eMinus100And1ePlus100) {
    for (double scale : {1e-100, 1e100}) {
        CscLower A = make_exchange_2x2(scale);
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.pivot_u = 0.01;
        ctrl.small_pivot = 1e-20;

        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success)
            << "Failed for scale=" << scale;
        EXPECT_EQ(info.num_positive, 1) << "scale=" << scale;
        EXPECT_EQ(info.num_negative, 1) << "scale=" << scale;
        EXPECT_EQ(info.num_zero, 0) << "scale=" << scale;
        EXPECT_EQ(info.numerical_rank, 2) << "scale=" << scale;

        std::vector<double> b = {1.0 * scale, 2.0 * scale};
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 2, 1), 0);
        EXPECT_NEAR(x[0], 2.0, 1e-12) << "scale=" << scale;
        EXPECT_NEAR(x[1], 1.0, 1e-12) << "scale=" << scale;
        auto val = smf::bench::validate(A, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val)) << "scale=" << scale;
    }
}

TEST(IndefAdversarial, ErrorPropagationAndLifetimeSafety) {
    CscLower A_spd;
    A_spd.n = 2;
    A_spd.col_ptr = {0, 2, 3};
    A_spd.row_idx = {0, 1, 1};
    A_spd.values = {4.0, 1.0, 3.0};

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    Info info;
    Solver solver;

    // 1. Returned factor from factor_solve or factor used after AnalysisKeep is destroyed
    std::unique_ptr<FactorKeep> fk_surviving;
    {
        auto ak = solver.analyse(A_spd, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        fk_surviving = std::make_unique<FactorKeep>();
        ASSERT_EQ(solver.factor(*ak, ctrl, info, *fk_surviving), FactorStatus::Success);
        fk_surviving->analysis = nullptr; // simulate destroyed analysis handle
    }
    std::vector<double> b = {5.0, 4.0};
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(*fk_surviving, ctrl, info, x.data(), 2, 1), 0);
    EXPECT_NEAR(x[0], 1.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0, 1e-12);

    // Also test factor returned from factor_solve used for subsequent solve
    {
        std::vector<double> b_fs = {5.0, 4.0};
        auto fk_from_fs = solver.factor_solve(A_spd, ctrl, info, b_fs.data(), 2, 1);
        ASSERT_NE(fk_from_fs.get(), nullptr);
        std::vector<double> b_reuse = {9.0, 5.0};
        ASSERT_EQ(solver.solve(*fk_from_fs, ctrl, info, b_reuse.data(), 2, 1), 0);
        EXPECT_NEAR(b_reuse[0], 2.0, 1e-12);
        EXPECT_NEAR(b_reuse[1], 1.0, 1e-12);
    }

    // 2. Invalid dimension / nrhs / null buffer / unknown SolveJob must fail and set info.status
    {
        Info bad_info;
        EXPECT_NE(solver.solve(*fk_surviving, ctrl, bad_info, x.data(), 999, 1), 0);
        EXPECT_EQ(bad_info.status, ErrorCode::IllegalValue);
    }
    {
        Info bad_info;
        EXPECT_NE(solver.solve(*fk_surviving, ctrl, bad_info, x.data(), 2, 0), 0);
        EXPECT_EQ(bad_info.status, ErrorCode::IllegalValue);
    }
    {
        Info bad_info;
        EXPECT_NE(solver.solve(*fk_surviving, ctrl, bad_info, nullptr, 2, 1), 0);
        EXPECT_EQ(bad_info.status, ErrorCode::IllegalValue);
    }
    {
        Info bad_info;
        EXPECT_NE(solver.solve(*fk_surviving, ctrl, bad_info, x.data(), 2, 1, static_cast<SolveJob>(99)), 0);
        EXPECT_EQ(bad_info.status, ErrorCode::IllegalValue);
    }

    // 3. Singular matrix without continue_on_singular: factor_solve returns nullptr and solve returns nonzero
    {
        CscLower A_sing;
        A_sing.n = 2;
        A_sing.col_ptr = {0, 1, 1};
        A_sing.row_idx = {0};
        A_sing.values = {2.0};

        Control sing_ctrl = ctrl;
        sing_ctrl.continue_on_singular = false;
        Info sing_info;
        std::vector<double> b_sing = {1.0, 1.0};
        auto fk_sing_fs = solver.factor_solve(A_sing, sing_ctrl, sing_info, b_sing.data(), 2, 1);
        EXPECT_EQ(fk_sing_fs.get(), nullptr);
        EXPECT_NE(sing_info.status, ErrorCode::Success);

        auto ak_sing = solver.analyse(A_sing, sing_ctrl, sing_info);
        ASSERT_NE(ak_sing.get(), nullptr);
        FactorKeep fk_sing;
        EXPECT_EQ(solver.factor(*ak_sing, sing_ctrl, sing_info, fk_sing), FactorStatus::Singular);
        EXPECT_NE(solver.solve(fk_sing, sing_ctrl, sing_info, b_sing.data(), 2, 1), 0);
        EXPECT_EQ(sing_info.status, ErrorCode::SingularMatrix);
    }
}
