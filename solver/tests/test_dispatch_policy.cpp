#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/dispatch_policy.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace smf;

TEST(DispatchPolicy, StructuralSelectionAndBandedSpdPreserved) {
    Solver solver;
    Info info;

    // 1. Tiny dense matrix (4x4 full lower triangle) -> TinyDense
    {
        CscLower A;
        A.n = 4;
        A.col_ptr = {0, 4, 7, 9, 10};
        A.row_idx = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
        A.values = {10.0, 1.0, 0.5, 0.2, 10.0, 1.0, 0.5, 10.0, 1.0, 10.0};
        Control ctrl;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        DispatchDecision d = select_native_path(*ak, ctrl, 1, 1);
        EXPECT_EQ(d.path, NativePath::TinyDense);
        EXPECT_FALSE(d.reason.empty());
    }

    // 2. Narrow-band SPD (N=120, tridiagonal, AutoParallel) -> BandedSPD
    {
        CscLower A;
        A.n = 120;
        A.col_ptr.resize(121, 0);
        for (int j = 0; j < 120; ++j) {
            A.row_idx.push_back(j);
            A.values.push_back(4.0);
            if (j + 1 < 120) {
                A.row_idx.push_back(j + 1);
                A.values.push_back(-1.0);
            }
            A.col_ptr[static_cast<std::size_t>(j + 1)] =
                static_cast<Int>(A.row_idx.size());
        }
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
        ctrl.ordering = OrderingMethod::AutoParallel;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        DispatchDecision d = select_native_path(*ak, ctrl, 1, 10);
        EXPECT_EQ(d.path, NativePath::BandedSPD);
    }

    // 3. Repeated KKT candidate (N=20, full diagonal, expected_refactors=10) -> CertifiedQuasiDefinite
    {
        const int n = 20;
        CscLower A;
        A.n = n;
        A.col_ptr.resize(n + 1, 0);
        for (int j = 0; j < n; ++j) {
            A.row_idx.push_back(j);
            A.values.push_back((j < 12) ? 10.0 : -5.0);
            if (j + 1 < n) {
                A.row_idx.push_back(j + 1);
                A.values.push_back(0.5);
            }
            A.col_ptr[static_cast<std::size_t>(j + 1)] =
                static_cast<Int>(A.row_idx.size());
        }
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.expected_refactors = 10;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        DispatchDecision d = select_native_path(*ak, ctrl, 1, ctrl.expected_refactors);
        EXPECT_EQ(d.path, NativePath::CertifiedQuasiDefinite);
        EXPECT_TRUE(certify_quasi_definite(A));

        // Corrupt a diagonal to 0.0 (non-quasi-definite!) -> certify_quasi_definite must return false
        // and Solver::factor must fall back cleanly to GeneralIndefinite and still solve accurately!
        CscLower A_non_qd = A;
        A_non_qd.values[0] = 0.0;
        EXPECT_FALSE(certify_quasi_definite(A_non_qd));

        NumericWorkspace nws;
        SolveWorkspace sws;
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, A_non_qd, ctrl, info, fk, nws), FactorStatus::Success);
        std::vector<double> b(n, 1.0);
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), n, 1, sws), 0);
        auto val = smf::bench::validate(A_non_qd, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val));
    }
}
