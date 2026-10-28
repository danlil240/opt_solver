#include "../benchmarks/common/validation.hpp"
#include "smf/blocked_indef.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/numeric_front.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace smf;

namespace {

// Reconstruct P^T F P and L D L^T + [0, 0; 0, Contrib] on a factored NumericFront
void verify_front_factor_identity(const NumericFront &orig_front,
                                  const FrontResult &res,
                                  double tol = 1e-11) {
    const int f = orig_front.num_rows();
    const int p = res.num_accepted;
    const int q = f - p;
    ASSERT_EQ(res.front_size, f);
    ASSERT_EQ(static_cast<int>(res.l_row_ids.size()), f);
    ASSERT_EQ(static_cast<int>(res.accepted_row_ids.size()), p);

    // Map row_id -> original index in orig_front
    auto find_orig = [&](Int r_id) -> int {
        for (int i = 0; i < f; ++i) {
            if (orig_front.row_ids()[static_cast<std::size_t>(i)] == r_id) return i;
        }
        return -1;
    };

    // Build full symmetric original matrix in res.l_row_ids order
    std::vector<double> PFP(static_cast<std::size_t>(f * f), 0.0);
    double norm_fro_sq = 0.0;
    for (int j = 0; j < f; ++j) {
        const int oj = find_orig(res.l_row_ids[static_cast<std::size_t>(j)]);
        ASSERT_GE(oj, 0);
        for (int i = 0; i < f; ++i) {
            const int oi = find_orig(res.l_row_ids[static_cast<std::size_t>(i)]);
            ASSERT_GE(oi, 0);
            const double v = orig_front.at(std::max(oi, oj), std::min(oi, oj));
            PFP[static_cast<std::size_t>(j * f + i)] = v;
            norm_fro_sq += v * v;
        }
    }
    const double norm_fro = std::max(1.0, std::sqrt(norm_fro_sq));

    // Build f x p L and p x p D
    std::vector<double> D(static_cast<std::size_t>(p * p), 0.0);
    int col_pos = 0;
    for (const auto &db : res.d_blocks) {
        if (db.tag == 1) {
            D[static_cast<std::size_t>(col_pos * p + col_pos)] = db.d00;
            ++col_pos;
        } else if (db.tag == 2) {
            D[static_cast<std::size_t>(col_pos * p + col_pos)] = db.d00;
            D[static_cast<std::size_t>(col_pos * p + (col_pos + 1))] = db.d10;
            D[static_cast<std::size_t>((col_pos + 1) * p + col_pos)] = db.d10;
            D[static_cast<std::size_t>((col_pos + 1) * p + (col_pos + 1))] = db.d11;
            col_pos += 2;
        }
    }
    ASSERT_EQ(col_pos, p);

    // Reconstruct L * D * L^T + [0, 0; 0, Contrib]
    std::vector<double> recon(static_cast<std::size_t>(f * f), 0.0);
    for (int j = 0; j < f; ++j) {
        for (int i = 0; i < f; ++i) {
            double sum = 0.0;
            for (int a = 0; a < p; ++a) {
                const double l_ia = res.l_panel[static_cast<std::size_t>(a * f + i)];
                if (l_ia == 0.0) continue;
                for (int b_idx = 0; b_idx < p; ++b_idx) {
                    const double d_ab = D[static_cast<std::size_t>(b_idx * p + a)];
                    if (d_ab == 0.0) continue;
                    const double l_jb = res.l_panel[static_cast<std::size_t>(b_idx * f + j)];
                    sum += l_ia * d_ab * l_jb;
                }
            }
            recon[static_cast<std::size_t>(j * f + i)] = sum;
        }
    }
    if (q > 0 && !res.contrib.empty()) {
        for (int cj = 0; cj < q; ++cj) {
            for (int ci = cj; ci < q; ++ci) {
                const double cv = res.contrib[static_cast<std::size_t>(cj * q + ci)];
                recon[static_cast<std::size_t>((p + cj) * f + (p + ci))] += cv;
                if (ci != cj) {
                    recon[static_cast<std::size_t>((p + ci) * f + (p + cj))] += cv;
                }
            }
        }
    }

    double diff_sq = 0.0;
    for (int idx = 0; idx < f * f; ++idx) {
        const double d = PFP[static_cast<std::size_t>(idx)] - recon[static_cast<std::size_t>(idx)];
        diff_sq += d * d;
    }
    const double rel_err = std::sqrt(diff_sq) / norm_fro;
    EXPECT_LE(rel_err, tol) << "Front reconstruction error = " << rel_err;
}

} // namespace

TEST(BlockedIndef, PanelBoundary2x2AndSwapsAndDelays) {
    // 1. Panel-boundary 2x2 pivot:
    // Front of size f=10, num_eligible=8, panel_size=3 so col 2 is at panel boundary (k = 2 = panel_size - 1)
    // and col 2 has zero diagonal coupling strongly to col 3 (or col 5 across the panel boundary)!
    {
        std::vector<Int> rows(10);
        std::iota(rows.begin(), rows.end(), 0);
        NumericFront f_ref(rows, 8);
        for (int j = 0; j < 10; ++j) {
            for (int i = j; i < 10; ++i) {
                if (i == j) f_ref.at(i, j) = (i == 2 || i == 5) ? 0.0 : (6.0 + 0.5 * i);
                else f_ref.at(i, j) = 0.1 / static_cast<double>(i - j + 1);
            }
        }
        // Strong 2x2 coupling between col 2 (panel boundary for nb=3) and col 5 (in next panel!)
        f_ref.at(5, 2) = 4.0;
        // Also put a zero diagonal at col 6 with strong extension coupling to row 9 so col 6 is delayed!
        f_ref.at(6, 6) = 0.0;
        for (int i = 0; i < 8; ++i) {
            if (i != 6) f_ref.at(std::max(i, 6), std::min(i, 6)) = 0.0;
        }
        f_ref.at(9, 6) = 8.0;

        NumericFront orig_copy = f_ref;
        NumericFront f_blk = f_ref;

        Control ctrl;
        ctrl.pivot_u = 0.01;
        ctrl.small_pivot = 1e-20;
        ctrl.indef_panel_size = 3;

        NumericWorkspace ws;
        FrontResult r_ref = factor_front_reference(f_ref, ctrl, ctrl.small_pivot, false);
        FrontResult r_blk = factor_front_blocked(f_blk, ctrl, ws, ctrl.small_pivot, false);

        ASSERT_EQ(r_ref.status, ErrorCode::Success);
        ASSERT_EQ(r_blk.status, ErrorCode::Success);
        EXPECT_EQ(r_blk.inertia.positive, r_ref.inertia.positive);
        EXPECT_EQ(r_blk.inertia.negative, r_ref.inertia.negative);
        EXPECT_EQ(r_blk.inertia.zero, r_ref.inertia.zero);
        EXPECT_EQ(r_blk.delayed_count, r_ref.delayed_count);
        EXPECT_EQ(r_blk.num_accepted, r_ref.num_accepted);

        verify_front_factor_identity(orig_copy, r_ref);
        verify_front_factor_identity(orig_copy, r_blk);
    }

    // 2. Extreme scaling (1e-100 and 1e100) with blocked front factorization
    for (double scale : {1e-100, 1e100}) {
        std::vector<Int> rows = {0, 1, 2, 3, 4, 5};
        NumericFront f_blk(rows, 4);
        f_blk.at(0, 0) = 0.0;
        f_blk.at(1, 0) = 2.0 * scale;
        f_blk.at(1, 1) = 0.0;
        f_blk.at(2, 2) = 5.0 * scale;
        f_blk.at(3, 2) = 1.0 * scale;
        f_blk.at(3, 3) = -4.0 * scale;
        f_blk.at(4, 0) = 0.5 * scale;
        f_blk.at(5, 2) = -0.5 * scale;

        Control ctrl;
        ctrl.pivot_u = 0.01;
        ctrl.indef_panel_size = 2;
        NumericWorkspace ws;

        FrontResult r_blk = factor_front_blocked(f_blk, ctrl, ws, -1.0, false);
        ASSERT_EQ(r_blk.status, ErrorCode::Success);
        EXPECT_EQ(r_blk.num_accepted, 4);
        EXPECT_EQ(r_blk.inertia.positive, 2);
        EXPECT_EQ(r_blk.inertia.negative, 2);
        EXPECT_EQ(r_blk.inertia.zero, 0);
    }
}

TEST(BlockedIndef, PanelSizeSweepAndSolverEndToEnd) {
    // Build a 40x40 dense indefinite front with mixed 1x1 and 2x2 pivots and sweep panel sizes 16, 32, 64, 128
    const int n = 40;
    CscLower A;
    A.n = n;
    A.col_ptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            A.row_idx.push_back(i);
            double v = 0.0;
            if (i == j) {
                if (i % 7 == 0) v = 0.0; // force 2x2 pivots at multiples of 7
                else if (i % 3 == 0) v = -12.0 - 0.2 * i;
                else v = 15.0 + 0.3 * i;
            } else if (i == j + 1 && (j % 7 == 0)) {
                v = 6.0;
            } else {
                v = 0.2 / static_cast<double>(i - j + 1);
            }
            A.values.push_back(v);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }

    std::vector<double> b(n, 1.0);
    for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = (i % 2 == 0) ? 1.0 : -2.0;

    Solver solver;
    Control ref_ctrl;
    ref_ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ref_ctrl.indef_use_reference_kernel = true;
    Info ref_info;
    auto ak = solver.analyse(A, ref_ctrl, ref_info);
    ASSERT_NE(ak.get(), nullptr);

    NumericWorkspace nws_ref;
    FactorKeep fk_ref;
    ASSERT_EQ(solver.factor(*ak, A, ref_ctrl, ref_info, fk_ref, nws_ref), FactorStatus::Success);

    for (int panel_size : {16, 32, 64, 128}) {
        Control blk_ctrl;
        blk_ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        blk_ctrl.indef_panel_size = panel_size;
        blk_ctrl.indef_use_reference_kernel = false;

        Info blk_info;
        NumericWorkspace nws;
        SolveWorkspace sws;
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, A, blk_ctrl, blk_info, fk, nws), FactorStatus::Success)
            << "panel_size=" << panel_size;
        EXPECT_EQ(blk_info.num_positive, ref_info.num_positive) << "panel_size=" << panel_size;
        EXPECT_EQ(blk_info.num_negative, ref_info.num_negative) << "panel_size=" << panel_size;
        EXPECT_EQ(blk_info.num_zero, ref_info.num_zero) << "panel_size=" << panel_size;

        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, blk_ctrl, blk_info, x.data(), n, 1, sws), 0);
        auto val = smf::bench::validate(A, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "panel_size=" << panel_size << " eta_inf=" << val.eta_inf << " omega=" << val.omega;
    }
}

// Tiny threshold (IPOPT's u = 1e-8): both front kernels must take the stable
// 2x2 on (0, 1) instead of a 1x1 on the near-null partner diagonal 2e-6,
// which passes 1e-8 * 100 but would put L = 5e7 into the factor.
TEST(BlockedIndef, TinyUPrefers2x2OverNearNullPartner) {
    std::vector<Int> rows = {0, 1, 2};
    NumericFront base(rows, 3);
    base.at(0, 0) = 0.0;
    base.at(1, 0) = 1.0;
    base.at(1, 1) = 2e-6;
    base.at(2, 1) = 100.0;
    base.at(2, 2) = 1.0;

    Control ctrl;
    ctrl.pivot_u = 1e-8;
    ctrl.small_pivot = 1e-20;
    ctrl.indef_panel_size = 2;

    NumericFront f_ref = base;
    NumericFront f_blk = base;
    NumericWorkspace ws;
    const FrontResult r_ref = factor_front_reference(f_ref, ctrl, ctrl.small_pivot, false);
    const FrontResult r_blk = factor_front_blocked(f_blk, ctrl, ws, ctrl.small_pivot, false);
    for (const FrontResult *r : {&r_ref, &r_blk}) {
        ASSERT_EQ(r->status, ErrorCode::Success);
        ASSERT_GE(r->pivot_tags.size(), 2u);
        EXPECT_EQ(r->pivot_tags[0], 2);
        EXPECT_EQ(r->pivot_tags[1], -1);
        EXPECT_EQ(r->inertia.positive, 2);
        EXPECT_EQ(r->inertia.negative, 1);
        EXPECT_EQ(r->inertia.zero, 0);
    }
}
