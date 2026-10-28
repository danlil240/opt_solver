#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/factor_indef.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace smf;

namespace {

void check_factor_identity(const CscLower& A, const FactorKeep& fk) {
    const int n = static_cast<int>(A.n);
    ASSERT_EQ(static_cast<int>(fk.perm.size()), n);
    ASSERT_EQ(static_cast<int>(fk.iperm.size()), n);

    // Check permutation bijection and inverse consistency
    std::vector<int> sorted_perm = fk.perm;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(sorted_perm[static_cast<std::size_t>(i)], i);
        const int old_idx = fk.perm[static_cast<std::size_t>(i)];
        ASSERT_GE(old_idx, 0);
        ASSERT_LT(old_idx, n);
        EXPECT_EQ(fk.iperm[static_cast<std::size_t>(old_idx)], i);
    }

    // Build full dense A (n x n)
    std::vector<double> A_dense(static_cast<std::size_t>(n * n), 0.0);
    for (int j = 0; j < n; ++j) {
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
            const double v = A.values[static_cast<std::size_t>(p)];
            A_dense[static_cast<std::size_t>(j * n + i)] += v;
            if (i != j) {
                A_dense[static_cast<std::size_t>(i * n + j)] += v;
            }
        }
    }

    // Reconstruct P^T A P using perm[new] = old
    std::vector<double> PAP(static_cast<std::size_t>(n * n), 0.0);
    double norm_A_fro_sq = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int old_i = fk.perm[static_cast<std::size_t>(i)];
            const int old_j = fk.perm[static_cast<std::size_t>(j)];
            const double val = A_dense[static_cast<std::size_t>(old_j * n + old_i)];
            PAP[static_cast<std::size_t>(j * n + i)] = val;
            norm_A_fro_sq += val * val;
        }
    }
    const double norm_A_fro = std::max(1.0, std::sqrt(norm_A_fro_sq));

    // Reconstruct unit lower-triangular L from solve_steps and factor_values
    // (strictly below the diagonal; the diagonal slots hold D when inline_d,
    // checked against diag_entries below).
    std::vector<double> L(static_cast<std::size_t>(n * n), 0.0);
    std::vector<double> D_inline(static_cast<std::size_t>(n * n), 0.0);
    for (int i = 0; i < n; ++i) {
        L[static_cast<std::size_t>(i * n + i)] = 1.0;
    }
    for (const auto& step : fk.solve_steps) {
        const Int* rows = fk.solve_row_indices.data() + step.row_offset;
        const double* fdata = fk.factor_values.data() + step.factor_offset;
        const int p = step.p;
        const int f = step.f;
        for (int c = 0; c < p; ++c) {
            const int gcol = static_cast<int>(rows[c]);
            ASSERT_GE(gcol, 0);
            ASSERT_LT(gcol, n);
            if (fk.inline_d) {
                D_inline[static_cast<std::size_t>(gcol * n + gcol)] = fdata[static_cast<std::size_t>(c * f + c)];
                if (c + 1 < p && fdata[static_cast<std::size_t>((c + 1) * f + c)] != 0.0) {
                    const int g1 = static_cast<int>(rows[c + 1]);
                    const double d10 = fdata[static_cast<std::size_t>((c + 1) * f + c)];
                    D_inline[static_cast<std::size_t>(gcol * n + g1)] = d10;
                    D_inline[static_cast<std::size_t>(g1 * n + gcol)] = d10;
                }
            }
            for (int r = c + 1; r < f; ++r) {
                const int grow = static_cast<int>(rows[r]);
                ASSERT_GE(grow, 0);
                ASSERT_LT(grow, n);
                L[static_cast<std::size_t>(gcol * n + grow)] = fdata[static_cast<std::size_t>(c * f + r)];
            }
        }
    }

    // Reconstruct block-diagonal D exclusively from fk.diag_entries
    std::vector<double> D(static_cast<std::size_t>(n * n), 0.0);
    ASSERT_FALSE(fk.diag_entries.empty());
    for (const auto& de : fk.diag_entries) {
        if (de.tag == 1) {
            D[static_cast<std::size_t>(de.row0 * n + de.row0)] = de.d00;
            EXPECT_DOUBLE_EQ(L[static_cast<std::size_t>(de.row0 * n + de.row0)], 1.0);
        } else if (de.tag == 2) {
            D[static_cast<std::size_t>(de.row0 * n + de.row0)] = de.d00;
            D[static_cast<std::size_t>(de.row0 * n + de.row1)] = de.d10;
            D[static_cast<std::size_t>(de.row1 * n + de.row0)] = de.d10;
            D[static_cast<std::size_t>(de.row1 * n + de.row1)] = de.d11;
            // Internal L block of every 2x2 pivot must be identity!
            EXPECT_DOUBLE_EQ(L[static_cast<std::size_t>(de.row0 * n + de.row0)], 1.0);
            EXPECT_DOUBLE_EQ(L[static_cast<std::size_t>(de.row0 * n + de.row1)], 0.0);
            EXPECT_DOUBLE_EQ(L[static_cast<std::size_t>(de.row1 * n + de.row0)], 0.0);
            EXPECT_DOUBLE_EQ(L[static_cast<std::size_t>(de.row1 * n + de.row1)], 1.0);
        }
    }

    if (fk.inline_d) {
        for (int idx = 0; idx < n * n; ++idx)
            EXPECT_EQ(D_inline[static_cast<std::size_t>(idx)], D[static_cast<std::size_t>(idx)]) << idx;
    }

    // Compute L * D * L^T
    std::vector<double> LD(static_cast<std::size_t>(n * n), 0.0);
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
            const double d_kj = D[static_cast<std::size_t>(j * n + k)];
            if (d_kj == 0.0) continue;
            for (int i = 0; i < n; ++i) {
                LD[static_cast<std::size_t>(j * n + i)] += L[static_cast<std::size_t>(k * n + i)] * d_kj;
            }
        }
    }

    std::vector<double> LDLT(static_cast<std::size_t>(n * n), 0.0);
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
            const double l_jk = L[static_cast<std::size_t>(k * n + j)];
            if (l_jk == 0.0) continue;
            for (int i = 0; i < n; ++i) {
                LDLT[static_cast<std::size_t>(j * n + i)] += LD[static_cast<std::size_t>(k * n + i)] * l_jk;
            }
        }
    }

    double err_fro_sq = 0.0;
    for (int idx = 0; idx < n * n; ++idx) {
        const double diff = PAP[static_cast<std::size_t>(idx)] - LDLT[static_cast<std::size_t>(idx)];
        err_fro_sq += diff * diff;
    }
    const double rel_err = std::sqrt(err_fro_sq) / norm_A_fro;
    EXPECT_LE(rel_err, 1e-12) << "Normalized reconstruction error ||P^T A P - L D L^T||_F / ||A||_F = " << rel_err;
}

} // namespace

TEST(FactorIdentity, NumericPermutationAndBlockD) {
    Solver solver;
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.nemin = 8;
    ctrl.pivot_u = 0.1;

    // Case 1: 3x3 with 2x2 pivot at (0, 1) and 1x1 at 2
    {
        CscLower A;
        A.n = 3;
        A.col_ptr = {0, 3, 5, 6};
        A.row_idx = {0, 1, 2, 1, 2, 2};
        A.values = {0.0, 2.0, 1.0, 0.0, -1.0, 4.0};

        Info info;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
        check_factor_identity(A, fk);
    }

    // Case 2: 4x4 front where col 0 accepts a 1x1 pivot first, and then at k=1
    // a numerical swap occurs between col 1 and col 3 (either 1x1 swap or 2x2 swap)!
    // This verifies that earlier accepted L columns (col 0) have their rows permuted
    // consistently with the numeric swap at k=1!
    {
        CscLower A;
        A.n = 4;
        A.col_ptr = {0, 4, 7, 9, 10};
        A.row_idx = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
        // col 0: A(0,0)=10 (accepted 1x1), A(1,0)=1, A(2,0)=2, A(3,0)=3
        // After k=0 Schur update:
        //   F(1,1) = 0.1 - 0.1 = 0.0
        //   F(2,1) = 0.2 - 0.2 = 0.0
        //   F(3,1) = 5.3 - 0.3 = 5.0  (large off-diagonal at r=3!)
        //   F(2,2) = 4.4 - 0.4 = 4.0
        //   F(3,2) = 0.6 - 0.6 = 0.0
        //   F(3,3) = 0.9 - 0.9 = 0.0
        // At k=1: F(1,1)=0, argmax off-diagonal is r=3 (F(3,1)=5.0), F(3,3)=0 -> 2x2 pivot (1, 3) swaps 2 <-> 3!
        A.values = {
            10.0, 1.0, 2.0, 3.0,
            0.1,  0.2, 5.3,
            4.4,  0.6,
            0.9
        };

        Info info;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
        check_factor_identity(A, fk);
    }

    // Case 3: Multi-supernode tree where a parent front performs a numeric swap
    // on rows that appeared as extension rows of a child front!
    {
        // 4x4 matrix with nemin=1 so supernodes are separate, then parent amalgamated or 2-node tree:
        // Node 0 connects to 2 and 3; nodes 1, 2, 3 form a dense 3x3 block where 2 and 3 swap.
        CscLower A;
        A.n = 4;
        A.col_ptr = {0, 3, 6, 8, 9};
        A.row_idx = {
            0, 2, 3,    // col 0
            1, 2, 3,    // col 1
            2, 3,       // col 2
            3           // col 3
        };
        A.values = {
            8.0, 1.0, -2.0,
            6.0, 0.5, 1.5,
            0.0, 3.0,
            0.0
        };

        Control tree_ctrl = ctrl;
        tree_ctrl.nemin = 1;
        Info info;
        auto ak = solver.analyse(A, tree_ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, tree_ctrl, info, fk), FactorStatus::Success);
        check_factor_identity(A, fk);
    }
}
