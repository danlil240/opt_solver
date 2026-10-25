#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/numeric_front.hpp"
#include "smf/solver.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace smf;

TEST(DelayedPivots, CrossFrontNonsingular) {
    // 1. Single cross-front delay:
    // 3x3 matrix with nemin=1 and OrderingMethod detail:
    // Node 0 has A(0,0) = 0.0, connects to node 2 with A(2,0) = 1.0.
    // Node 1 has A(1,1) = 4.0, connects to node 2 with A(2,1) = 1.0.
    // Node 2 has A(2,2) = 2.0.
    // Supernode 0 has 1 eligible column {0} with zero diagonal and extension row {2}.
    // Column 0 MUST be delayed from supernode 0 into its parent (supernode 2),
    // where it meets A(2,0) = 1.0 and is factored nonsingularly!
    {
        CscLower A;
        A.n = 3;
        A.col_ptr = {0, 2, 4, 5};
        A.row_idx = {0, 2, 1, 2, 2};
        A.values = {0.0, 1.0, 4.0, 1.0, 2.0};

        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.nemin = 1;
        ctrl.pivot_u = 0.01;

        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        ASSERT_GE(ak->supernodes.size(), 2u);

        FactorKeep fk;
        FactorStatus st = solver.factor(*ak, ctrl, info, fk);
        ASSERT_EQ(st, FactorStatus::Success);
        EXPECT_GT(info.delayed_pivots, 0);
        EXPECT_EQ(info.num_zero, 0);
        EXPECT_EQ(info.numerical_rank, 3);
        // Eigenvalues of [[0,0,1],[0,4,1],[1,1,2]]: det = -4 < 0, trace = 6 > 0 -> 2 pos, 1 neg
        EXPECT_EQ(info.num_positive, 2);
        EXPECT_EQ(info.num_negative, 1);

        std::vector<double> b = {3.0, 5.0, 4.0};
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 3, 1), 0);
        auto val = smf::bench::validate(A, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "eta_inf=" << val.eta_inf << " omega=" << val.omega;
    }

    // 2. Multiple children delaying into the same parent:
    // Both leaf 0 (A(0,0)=0, A(3,0)=2) and leaf 1 (A(1,1)=0, A(2,1)=3) delay into ancestors!
    {
        CscLower A;
        A.n = 4;
        A.col_ptr = {0, 2, 4, 6, 7};
        A.row_idx = {
            0, 3,    // col 0: A(0,0)=0, A(3,0)=2
            1, 2,    // col 1: A(1,1)=0, A(2,1)=3
            2, 3,    // col 2: A(2,2)=1, A(3,2)=0.5
            3        // col 3: A(3,3)=-1
        };
        A.values = {
            0.0, 2.0,
            0.0, 3.0,
            1.0, 0.5,
            -1.0
        };

        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.nemin = 1;
        ctrl.pivot_u = 0.01;

        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);

        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
        EXPECT_GE(info.delayed_pivots, 1);
        EXPECT_EQ(info.num_zero, 0);
        EXPECT_EQ(info.numerical_rank, 4);
        EXPECT_EQ(info.num_positive, 2);
        EXPECT_EQ(info.num_negative, 2);

        std::vector<double> b = {2.0, -3.0, 1.0, 4.0};
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 4, 1), 0);
        auto val = smf::bench::validate(A, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val));
    }

    // 3. Multi-level delay: leaf 0 has A(0,0)=0, A(1,0)=0 (structural), A(2,0)=2.0;
    // middle node 1 has A(1,1)=5.0, A(2,1)=1.0; root 2 has A(2,2)=0.0.
    // Column 0 is delayed from node 0 -> node 1 (where A(1,0)=0 so it STILL cannot pivot!) -> root 2 (where it pivots with 2)!
    {
        CscLower A;
        A.n = 3;
        A.col_ptr = {0, 3, 5, 6};
        A.row_idx = {0, 1, 2, 1, 2, 2};
        A.values = {0.0, 0.0, 2.0, 5.0, 1.0, 0.0};

        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.nemin = 1;
        ctrl.pivot_u = 0.05;

        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);

        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
        EXPECT_EQ(info.num_zero, 0);
        EXPECT_EQ(info.numerical_rank, 3);
        EXPECT_EQ(info.num_positive, 2);
        EXPECT_EQ(info.num_negative, 1);

        std::vector<double> b = {4.0, 6.0, -2.0};
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 3, 1), 0);
        auto val = smf::bench::validate(A, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val));
    }

    // 4. Genuinely singular root after delay:
    // Column 0 has A(0,0)=0, A(2,0)=0 (stored structurally), so it delays to root 2 and is still zero at root!
    {
        CscLower A;
        A.n = 3;
        A.col_ptr = {0, 2, 4, 5};
        A.row_idx = {0, 2, 1, 2, 2};
        A.values = {0.0, 0.0, 4.0, 1.0, 3.0};

        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.nemin = 1;
        ctrl.pivot_u = 0.01;

        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);

        FactorKeep fk;
        EXPECT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Singular);
        EXPECT_GE(info.num_zero, 1);
        EXPECT_EQ(info.numerical_rank, 2);
    }
}

TEST(DelayedPivots, ExtensionRowGrowth) {
    // Direct NumericFront test:
    // 1 eligible column (row 0) with tiny diagonal F(0,0) = 1e-6,
    // and 1 extension row (row 1) with large coupling F(1,0) = 10.0.
    // With pivot_u = 0.01, |F(0,0)| = 1e-6 < u * |F(1,0)| = 0.1,
    // so column 0 MUST fail the full-column stability check and be delayed!
    {
        NumericFront front({0, 1}, 1);
        front.at(0, 0) = 1e-6;
        front.at(1, 0) = 10.0;
        front.at(1, 1) = 0.0;

        Control ctrl;
        ctrl.pivot_u = 0.01;
        ctrl.small_pivot = 1e-20;

        FrontResult res = factor_front_reference(front, ctrl, ctrl.small_pivot, false);
        EXPECT_EQ(res.status, ErrorCode::Success);
        EXPECT_EQ(res.num_accepted, 0);
        EXPECT_EQ(res.delayed_count, 1);
        ASSERT_EQ(res.delayed_col_ids.size(), 1u);
        EXPECT_EQ(res.delayed_col_ids[0], 0);
    }

    // End-to-end Solver test where an ill-conditioned local diagonal (1e-8) with large
    // extension coupling (10.0) must be delayed to its parent rather than causing 1e8 element growth!
    {
        CscLower A;
        A.n = 3;
        A.col_ptr = {0, 2, 4, 5};
        A.row_idx = {0, 2, 1, 2, 2};
        A.values = {1e-8, 10.0, 5.0, 1.0, 2.0};

        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.nemin = 1;
        ctrl.pivot_u = 0.01;

        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);

        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
        EXPECT_GT(info.delayed_pivots, 0);
        EXPECT_EQ(info.num_zero, 0);

        std::vector<double> b = {10.0, 6.0, 13.0};
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 3, 1), 0);
        auto val = smf::bench::validate(A, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "eta_inf=" << val.eta_inf << " omega=" << val.omega;
    }
}

TEST(DelayedPivots, ParallelThreadsConsistency) {
    CscLower A;
    A.n = 4;
    A.col_ptr = {0, 2, 4, 6, 7};
    A.row_idx = {0, 3, 1, 2, 2, 3, 3};
    A.values = {0.0, 2.0, 0.0, 3.0, 1.0, 0.5, -1.0};

    std::vector<double> b = {2.0, -3.0, 1.0, 4.0};
    std::vector<double> x_ref;

    for (int threads : {1, 2, 4}) {
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.nemin = 1;
        ctrl.pivot_u = 0.01;
        ctrl.num_threads = threads;
        ctrl.deterministic = true;

        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success)
            << "threads=" << threads;
        EXPECT_EQ(info.num_positive, 2);
        EXPECT_EQ(info.num_negative, 2);
        EXPECT_EQ(info.num_zero, 0);

        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 4, 1), 0);
        auto val = smf::bench::validate(A, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val));
        if (threads == 1) {
            x_ref = x;
        } else {
            for (std::size_t i = 0; i < x.size(); ++i) {
                EXPECT_DOUBLE_EQ(x[i], x_ref[i]) << "Thread mismatch at i=" << i;
            }
        }
    }
}
