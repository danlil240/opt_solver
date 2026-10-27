#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/numeric_front.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <span>
#include <utility>
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

namespace {

// KKT [H B^T; B 0] (H: 5-point Laplacian of a g x g grid with diagonal
// hdiag; m zero-diagonal constraint rows, row c coupling variables
// 3c, 3c+1, 3c+2 mod g^2). AMD eliminates the degree-3 constraint rows
// early, where they cannot pivot, so its factorization delays them.
CscLower delayed_kkt(int g, int m, double hdiag)
{
    const int nx = g * g, n = nx + m;
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n));
    for (int y = 0; y < g; ++y)
        for (int x = 0; x < g; ++x) {
            const int j = y * g + x;
            auto &c = cols[static_cast<std::size_t>(j)];
            c.push_back({j, hdiag});
            if (x + 1 < g) c.push_back({j + 1, -1.0});
            if (y + 1 < g) c.push_back({j + g, -1.0});
        }
    const double coef[3] = {1.0, -0.5, 0.25};
    for (int k = 0; k < m; ++k) {
        const int row = nx + k;
        for (int t = 0; t < 3; ++t)
            cols[static_cast<std::size_t>((3 * k + t) % nx)].push_back({row, coef[t]});
        cols[static_cast<std::size_t>(row)].push_back({row, 0.0});
    }
    CscLower A;
    A.n = n;
    A.col_ptr.assign(1, 0);
    for (auto &c : cols) {
        std::sort(c.begin(), c.end());
        for (auto [i, v] : c) {
            A.row_idx.push_back(i);
            A.values.push_back(v);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

} // namespace

TEST(DelayedPivots, RealizedDelayTriggerReanalysesOnce) {
    const int g = 30, m = 300;
    const CscLower A = delayed_kkt(g, m, 4.0);
    const int n = A.n;

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.nemin = 1;  // no amalgamation: a constraint row keeps its own front

    // Precondition: the AMD factorization delays more than 5% of n.
    {
        Solver s0;
        Info i0;
        Control c0 = ctrl;
        c0.ordering = OrderingMethod::AMD;
        auto a0 = s0.analyse(A, c0, i0);
        ASSERT_NE(a0.get(), nullptr);
        FactorKeep f0;
        ASSERT_EQ(s0.factor(*a0, c0, i0, f0), FactorStatus::Success);
        std::printf("[RealizedDelay] AMD factor: %d delays, n = %d\n", i0.delayed_pivots, n);
        ASSERT_GT(i0.delayed_pivots, n / 20);
    }

    ctrl.matching_trigger = MatchingTrigger::RealizedDelays;
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    EXPECT_EQ(info.ordering_used, static_cast<int>(OrderingMethod::AMD));

    // First factor: delays > 5% of n, so it re-analyses with the matching
    // ordering and factors again.
    NumericWorkspace ws;
    FactorKeep fk;
    Info i1;
    ASSERT_EQ(solver.factor(*ak, std::span<const double>(A.values), ctrl, i1, fk, ws),
              FactorStatus::Success);
    std::printf("[RealizedDelay] after re-analysis: %d delays\n", i1.delayed_pivots);
    EXPECT_EQ(i1.matching_reanalyses, 1);
    EXPECT_EQ(i1.ordering_used, static_cast<int>(OrderingMethod::Matching));
    EXPECT_LE(i1.delayed_pivots, n / 20);
    EXPECT_EQ(i1.num_positive, g * g);
    EXPECT_EQ(i1.num_negative, m);
    {
        std::vector<double> b(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = 1.0 + 0.01 * i;
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, i1, x.data(), n, 1), 0);
        auto val = smf::bench::validate(A, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "eta_inf=" << val.eta_inf << " omega=" << val.omega;
    }

    // Second factor, new values (same pattern): keeps the matching analysis,
    // no further re-analysis.
    const CscLower A2 = delayed_kkt(g, m, 5.0);
    Info i2;
    FactorKeep fk2;
    ASSERT_EQ(solver.factor(*ak, A2, ctrl, i2, fk2, ws), FactorStatus::Success);
    EXPECT_EQ(i2.matching_reanalyses, 0);
    EXPECT_EQ(i2.ordering_used, static_cast<int>(OrderingMethod::Matching));
    EXPECT_LE(i2.delayed_pivots, n / 20);
    {
        std::vector<double> b(static_cast<std::size_t>(n), 1.0);
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk2, ctrl, i2, x.data(), n, 1), 0);
        auto val = smf::bench::validate(A2, b, x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "eta_inf=" << val.eta_inf << " omega=" << val.omega;
    }

    // Predicted (the default) never re-analyses inside factor().
    Control pc;
    pc.matrix_type = MatrixType::RealSymmetricIndefinite;
    Solver ps;
    Info pi;
    auto pak = ps.analyse(A, pc, pi);
    ASSERT_NE(pak.get(), nullptr);
    FactorKeep pfk;
    ASSERT_EQ(ps.factor(*pak, pc, pi, pfk), FactorStatus::Success);
    EXPECT_EQ(pi.matching_reanalyses, 0);
}

namespace {

// Two children A = [0, E) and B = [E, 2E) under a root C = [2E, 2E + R),
// in identity order: A11 and B11 structurally dense, every A / B column
// coupled to every C row. With `nasty`, four columns of A are numerically
// zero inside A11 (diagonal included):
//  - a = 5 and b = 200 (different 128-wide APTP tiles) see only A(b, a) = 1:
//    each fails its own tile, so APTP fails both; the TPP fallback takes
//    them as the 2x2 [[0, 1], [1, 0]].
//  - z = 7 and z2 = 250 see only C rows (A(2E, z) = A(2E + 1, z2) = 1):
//    no eligible pivot exists for them in A, so they fail APTP and TPP both
//    and are delayed to C, where they pivot against those rows.
// Without `nasty` every diagonal is large: nothing fails or delays.
CscLower aptp_fallback_matrix(bool nasty) {
    const Int E = 300, R = 20, n = 2 * E + R;
    const Int a = 5, b = 200, z = 7, z2 = 250;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    const auto special = [&](Int j) { return nasty && (j == a || j == b || j == z || j == z2); };
    CscLower A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (Int j = 0; j < n; ++j) {
        const bool in_c = j >= 2 * E;
        const Int blk_end = in_c ? n : (j < E ? E : 2 * E);
        for (Int i = j; i < blk_end; ++i) {
            double v;
            if (i == j) v = (j % 2 == 0 ? 1.0 : -1.0) * (10.0 + uni(rng));
            else v = 0.05 * uni(rng);
            if (special(i) || special(j)) v = 0.0;
            if (nasty && j == a && i == b) v = 1.0;
            A.row_idx.push_back(i);
            A.values.push_back(v);
        }
        if (!in_c) {
            for (Int i = 2 * E; i < n; ++i) {
                double v = special(j) ? 0.0 : 0.01 * uni(rng);
                if (nasty && j == z && i == 2 * E) v = 1.0;
                if (nasty && j == z2 && i == 2 * E + 1) v = 1.0;
                A.row_idx.push_back(i);
                A.values.push_back(v);
            }
        }
        A.col_ptr[static_cast<std::size_t>(j) + 1] = static_cast<Int>(A.row_idx.size());
    }
    return A;
}

Control aptp_fallback_control(std::vector<Int> &perm, Int n) {
    perm.resize(static_cast<std::size_t>(n));
    std::iota(perm.begin(), perm.end(), Int{0});
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.ordering = OrderingMethod::UserSupplied;
    ctrl.user_perm = perm.data();
    ctrl.scaling = ScalingMethod::None;
    ctrl.front_kernel = FrontKernel::Aptp;
    return ctrl;
}

} // namespace

// Columns that fail APTP but pass the TPP fallback, and columns that fail
// both and are delayed to the parent.
TEST(DelayedPivots, AptpFallbackThenDelay) {
    const CscLower A = aptp_fallback_matrix(true);
    std::vector<Int> perm;
    const Control ctrl = aptp_fallback_control(perm, A.n);
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_GE(ak->supernodes.size(), 3u);
    FactorKeep fk;
    NumericWorkspace ws;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, ws), FactorStatus::Success);
    EXPECT_GE(info.aptp_failed_columns, 4);
    EXPECT_EQ(info.delayed_pivots, 2);
    EXPECT_EQ(info.num_zero, 0);
    std::vector<double> b(static_cast<std::size_t>(A.n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::cos(0.1 * static_cast<double>(i));
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), static_cast<int>(A.n), 1), 0);
    const auto val = smf::bench::validate(A, b, x, 1);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val))
        << "eta_inf=" << val.eta_inf << " omega=" << val.omega;

    // The TPP engine agrees on inertia (and delays the same two columns).
    Control tc = ctrl;
    tc.front_kernel = FrontKernel::Tpp;
    Info ti;
    FactorKeep tfk;
    NumericWorkspace tws;
    ASSERT_EQ(solver.factor(*ak, A, tc, ti, tfk, tws), FactorStatus::Success);
    EXPECT_EQ(ti.aptp_failed_columns, 0);
    EXPECT_EQ(ti.delayed_pivots, 2);
    EXPECT_EQ(ti.num_positive, info.num_positive);
    EXPECT_EQ(ti.num_negative, info.num_negative);

    // Auto is Tpp by default (aptp_min_elig = 0); with a cutoff the 300-column
    // front of A (and of B) takes APTP.
    Control ac = ctrl;
    ac.front_kernel = FrontKernel::Auto;
    Info ai;
    ASSERT_EQ(solver.factor(*ak, A, ac, ai, tfk, tws), FactorStatus::Success);
    EXPECT_EQ(ai.aptp_failed_columns, 0);
    ac.aptp_min_elig = 128;
    ASSERT_EQ(solver.factor(*ak, A, ac, ai, tfk, tws), FactorStatus::Success);
    EXPECT_EQ(ai.aptp_failed_columns, info.aptp_failed_columns);
    EXPECT_EQ(ai.delayed_pivots, 2);
    ac.aptp_min_elig = 301;
    ASSERT_EQ(solver.factor(*ak, A, ac, ai, tfk, tws), FactorStatus::Success);
    EXPECT_EQ(ai.aptp_failed_columns, 0);
}

// A refactorization with benign values starts from clean counters: no
// APTP failures and no delays left over from the previous factor().
TEST(DelayedPivots, AptpRefactorStartsClean) {
    const CscLower A = aptp_fallback_matrix(true);
    const CscLower A2 = aptp_fallback_matrix(false);
    std::vector<Int> perm;
    const Control ctrl = aptp_fallback_control(perm, A.n);
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    NumericWorkspace ws;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, ws), FactorStatus::Success);
    ASSERT_GT(info.aptp_failed_columns, 0);
    ASSERT_GT(info.delayed_pivots, 0);
    ASSERT_EQ(solver.factor(*ak, A2, ctrl, info, fk, ws), FactorStatus::Success);
    EXPECT_EQ(info.aptp_failed_columns, 0);
    EXPECT_EQ(info.delayed_pivots, 0);
    std::vector<double> b(static_cast<std::size_t>(A2.n), 1.0);
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), static_cast<int>(A2.n), 1), 0);
    const auto val = smf::bench::validate(A2, b, x, 1);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val))
        << "eta_inf=" << val.eta_inf << " omega=" << val.omega;
}
