#include "../benchmarks/common/replay_refine.hpp"
#include "../benchmarks/common/validation.hpp"
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/refinement.hpp"
#include "smf/scaling.hpp"
#include "smf/scaling_matching.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
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
    // The lean engine keeps D inline in the factor too (FactorKeep::inline_d).
    for (double &v : fk_bad.factor_values) v *= 1e20;
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

TEST(Matching, ExplicitZerosKeepLaterEntries) {
    // [[0 1 0] [1 0 1] [0 1 1]] with explicit zeros at (0,0) and (2,0), as
    // IPOPT passes them. Dropping column 0's zeros used to leave stale slots
    // that made (2,1) and (2,2) look like duplicates, so the nonsingular
    // matrix was reported structurally singular (robot_arm KKT: 40k columns).
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 3, 4, 5};
    A.row_idx = {0, 1, 2, 2, 2};
    A.values = {0.0, 1.0, 0.0, 1.0, 1.0};

    MatchingResult mr = compute_symmetric_matching(A);
    ASSERT_EQ(mr.status, ErrorCode::Success);
    EXPECT_EQ(mr.matched_rank, 3);
    EXPECT_EQ(mr.col_to_row[0], 1);
    EXPECT_EQ(mr.col_to_row[1], 0);
    EXPECT_EQ(mr.col_to_row[2], 2);
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

namespace {

// Random symmetric matrix in lower CSC form: each lower entry is present with
// probability `density`, magnitude log-uniform in [1e-4, 1e4], random sign.
CscLower make_random_sym(int n, double density, bool with_diag, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    CscLower A;
    A.n = n;
    A.col_ptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            if ((i == j && with_diag) || (i != j && unif(rng) < density)) {
                const double mag = std::pow(10.0, -4.0 + 8.0 * unif(rng));
                A.row_idx.push_back(i);
                A.values.push_back(unif(rng) < 0.5 ? -mag : mag);
            }
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

// KKT-structured matrix [H J^T; J 0]: tridiagonal H on nv variables, nc
// constraint rows with `per_row` entries each, absent (2,2) block.
CscLower make_random_kkt(int nv, int nc, int per_row, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::uniform_int_distribution<int> pick(0, nv - 1);
    const auto value = [&] {
        const double mag = std::pow(10.0, -4.0 + 8.0 * unif(rng));
        return unif(rng) < 0.5 ? -mag : mag;
    };
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(nv + nc));
    for (int j = 0; j < nv; ++j) {
        cols[static_cast<std::size_t>(j)].push_back({j, value()});
        if (j + 1 < nv) cols[static_cast<std::size_t>(j)].push_back({j + 1, value()});
    }
    for (int r = 0; r < nc; ++r) {
        for (int k = 0; k < per_row; ++k) {
            cols[static_cast<std::size_t>(pick(rng))].push_back({nv + r, value()});
        }
    }
    CscLower A;
    A.n = nv + nc;
    A.col_ptr.push_back(0);
    for (auto &c : cols) {
        std::sort(c.begin(), c.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        for (const auto &[i, v] : c) {
            A.row_idx.push_back(i);
            A.values.push_back(v);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

// Dense full symmetric |A| (row-major), duplicates summed.
std::vector<double> dense_abs(const CscLower &A) {
    const int n = static_cast<int>(A.n);
    std::vector<double> d(static_cast<std::size_t>(n) * n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const int i = A.row_idx[static_cast<std::size_t>(p)];
            d[static_cast<std::size_t>(i) * n + j] += A.values[static_cast<std::size_t>(p)];
            if (i != j) d[static_cast<std::size_t>(j) * n + i] += A.values[static_cast<std::size_t>(p)];
        }
    }
    for (double &x : d) x = std::abs(x);
    return d;
}

} // namespace

TEST(Matching, OptimalAgainstBruteForce) {
    int compared = 0;
    for (unsigned seed = 1; seed <= 60; ++seed) {
        const int n = 3 + static_cast<int>(seed % 5);  // 3..7
        CscLower A = make_random_sym(n, 0.45, seed % 3 == 0, seed);
        const std::vector<double> d = dense_abs(A);
        std::vector<double> cmax(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                cmax[static_cast<std::size_t>(j)] =
                    std::max(cmax[static_cast<std::size_t>(j)], d[static_cast<std::size_t>(i) * n + j]);
        const auto cost = [&](int i, int j) {
            return std::log(cmax[static_cast<std::size_t>(j)]) - std::log(d[static_cast<std::size_t>(i) * n + j]);
        };

        // Brute force over perm[j] = row matched to column j.
        std::vector<int> perm(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), 0);
        const double inf = std::numeric_limits<double>::infinity();
        double best = inf;
        do {
            double c = 0.0;
            for (int j = 0; j < n && c < inf; ++j) {
                const int i = perm[static_cast<std::size_t>(j)];
                c = d[static_cast<std::size_t>(i) * n + j] > 0.0 ? c + cost(i, j) : inf;
            }
            best = std::min(best, c);
        } while (std::next_permutation(perm.begin(), perm.end()));

        MatchingResult mr = compute_symmetric_matching(A);
        if (best == inf) {
            EXPECT_EQ(mr.status, ErrorCode::SingularMatrix) << "seed=" << seed;
            continue;
        }
        ASSERT_EQ(mr.status, ErrorCode::Success) << "seed=" << seed;
        ASSERT_EQ(mr.matched_rank, n);
        double got = 0.0;
        for (int j = 0; j < n; ++j) {
            const int i = mr.col_to_row[static_cast<std::size_t>(j)];
            ASSERT_GE(i, 0);
            ASSERT_EQ(mr.row_to_col[static_cast<std::size_t>(i)], j);
            ASSERT_GT(d[static_cast<std::size_t>(i) * n + j], 0.0);
            got += cost(i, j);
        }
        EXPECT_NEAR(got, best, 1e-9 * (1.0 + std::abs(best))) << "seed=" << seed << " n=" << n;
        ++compared;
    }
    EXPECT_GT(compared, 30);
}

TEST(Matching, DenseRowsStayOptimal) {
    // KKT with a few dense constraint rows (the boyd1 shape): the matching
    // sets dense rows aside first. Its cost must equal a dense O(n^3)
    // Hungarian reference on the same costs.
    for (unsigned seed = 1; seed <= 3; ++seed) {
        // ~194 entries per constraint row > 10 sqrt(210) = 145: dense.
        const int nv = 200, nc = 10, n = nv + nc;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        const auto value = [&] { return std::pow(10.0, -3.0 + 9.0 * unif(rng)) * (unif(rng) < 0.5 ? -1 : 1); };
        CscLower A;
        A.n = n;
        A.col_ptr.push_back(0);
        for (int j = 0; j < n; ++j) {
            if (j < nv) {
                A.row_idx.push_back(j);
                A.values.push_back(value());
                if (j + 1 < nv) {
                    A.row_idx.push_back(j + 1);
                    A.values.push_back(value());
                }
                for (int r = nv; r < n; ++r) {
                    if (unif(rng) < 0.97) {
                        A.row_idx.push_back(r);
                        A.values.push_back(value());
                    }
                }
            }
            A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
        }

        const std::vector<double> d = dense_abs(A);
        std::vector<double> cmax(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                cmax[static_cast<std::size_t>(j)] =
                    std::max(cmax[static_cast<std::size_t>(j)], d[static_cast<std::size_t>(i) * n + j]);
        const double forbid = 1e9;
        const auto cost = [&](int i, int j) {
            const double a = d[static_cast<std::size_t>(i) * n + j];
            return a > 0.0 ? std::log(cmax[static_cast<std::size_t>(j)]) - std::log(a) : forbid;
        };

        // Hungarian (rows 1..n assigned to columns 1..n), O(n^3).
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<double> hu(static_cast<std::size_t>(n) + 1, 0.0), hv(static_cast<std::size_t>(n) + 1, 0.0);
        std::vector<int> p(static_cast<std::size_t>(n) + 1, 0), way(static_cast<std::size_t>(n) + 1, 0);
        for (int i = 1; i <= n; ++i) {
            p[0] = i;
            int j0 = 0;
            std::vector<double> minv(static_cast<std::size_t>(n) + 1, inf);
            std::vector<char> used(static_cast<std::size_t>(n) + 1, 0);
            do {
                used[static_cast<std::size_t>(j0)] = 1;
                const int i0 = p[static_cast<std::size_t>(j0)];
                double delta = inf;
                int j1 = 0;
                for (int j = 1; j <= n; ++j) {
                    if (used[static_cast<std::size_t>(j)]) continue;
                    const double cur = cost(i0 - 1, j - 1) - hu[static_cast<std::size_t>(i0)] - hv[static_cast<std::size_t>(j)];
                    if (cur < minv[static_cast<std::size_t>(j)]) {
                        minv[static_cast<std::size_t>(j)] = cur;
                        way[static_cast<std::size_t>(j)] = j0;
                    }
                    if (minv[static_cast<std::size_t>(j)] < delta) {
                        delta = minv[static_cast<std::size_t>(j)];
                        j1 = j;
                    }
                }
                for (int j = 0; j <= n; ++j) {
                    if (used[static_cast<std::size_t>(j)]) {
                        hu[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta;
                        hv[static_cast<std::size_t>(j)] -= delta;
                    } else {
                        minv[static_cast<std::size_t>(j)] -= delta;
                    }
                }
                j0 = j1;
            } while (p[static_cast<std::size_t>(j0)] != 0);
            do {
                const int j1 = way[static_cast<std::size_t>(j0)];
                p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
                j0 = j1;
            } while (j0 != 0);
        }
        double best = 0.0;
        for (int j = 1; j <= n; ++j) best += cost(p[static_cast<std::size_t>(j)] - 1, j - 1);
        ASSERT_LT(best, forbid) << "seed=" << seed;

        MatchingResult mr = compute_symmetric_matching(A);
        ASSERT_EQ(mr.status, ErrorCode::Success) << "seed=" << seed;
        ASSERT_EQ(mr.matched_rank, n);
        double got = 0.0;
        for (int j = 0; j < n; ++j) got += cost(mr.col_to_row[static_cast<std::size_t>(j)], j);
        EXPECT_NEAR(got, best, 1e-8 * (1.0 + std::abs(best))) << "seed=" << seed;
    }
}

TEST(Matching, KktScalingBoundsEntries) {
    // Dual feasibility of the matching makes every |s_i a_ij s_j| <= 1.
    for (unsigned seed = 1; seed <= 4; ++seed) {
        CscLower A = make_random_kkt(1500, 600, 3, seed);
        MatchingResult mr = compute_symmetric_matching(A);
        ASSERT_EQ(mr.status, ErrorCode::Success) << "seed=" << seed;
        ASSERT_EQ(mr.matched_rank, A.n);
        double max_scaled = 0.0;
        for (int j = 0; j < A.n; ++j) {
            for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
                const int i = A.row_idx[static_cast<std::size_t>(p)];
                max_scaled = std::max(max_scaled, std::abs(mr.scale[static_cast<std::size_t>(i)] *
                                                           A.values[static_cast<std::size_t>(p)] *
                                                           mr.scale[static_cast<std::size_t>(j)]));
            }
        }
        EXPECT_LE(max_scaled, 1.0 + 1e-9) << "seed=" << seed;
        EXPECT_GT(max_scaled, 0.5) << "seed=" << seed;
    }
}

TEST(Matching, AutoPolicyScalesSaddlePointsOnly) {
    // Default Control (auto ordering, ScalingMethod::Auto): a KKT matrix
    // (zero diagonals) gets the analysis' matching scaling in its factor when
    // the matching ordering is adopted, and no scaling when AMD is kept (the
    // scaling without its 2x2-pair ordering degrades accuracy); a matrix with
    // a full nonzero diagonal gets no scaling.
    const auto solve_accurately = [](const CscLower &A, const Control &ctrl, FactorKeep &fk,
                                     std::unique_ptr<AnalysisKeep> &ak) {
        Solver solver;
        Info info;
        ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace nws;
        SolveWorkspace sws;
        ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);
        std::vector<double> b(static_cast<std::size_t>(A.n));
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.37 * static_cast<double>(i)) + 1.5;
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), static_cast<int>(A.n), 1, sws), 0);
        refine(A, fk, ctrl, b, x, 1, sws);
        EXPECT_TRUE(smf::bench::is_qualified_validation(smf::bench::validate(A, b, x, 1)));
    };

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    // The auto policy starts at n = 10000.
    {
        // This KKT's matching ordering costs more than the auto policy
        // allows, so AMD is kept and the matching scaling is not applied.
        CscLower kkt = make_random_kkt(8000, 3000, 3, 7);
        FactorKeep fk;
        std::unique_ptr<AnalysisKeep> ak;
        solve_accurately(kkt, ctrl, fk, ak);
        EXPECT_TRUE(ak->matching_scale.empty());
        EXPECT_TRUE(fk.scale.empty());
    }
    {
        // Indefinite tridiagonal with a dominant diagonal of alternating sign.
        CscLower full_diag;
        full_diag.n = 12000;
        full_diag.col_ptr.push_back(0);
        for (Int j = 0; j < full_diag.n; ++j) {
            full_diag.row_idx.push_back(j);
            full_diag.values.push_back((j % 2 == 0) ? 1e5 : -1e5);
            if (j + 1 < full_diag.n) {
                full_diag.row_idx.push_back(j + 1);
                full_diag.values.push_back(1e4 * std::sin(0.1 * j));
            }
            full_diag.col_ptr.push_back(static_cast<Int>(full_diag.row_idx.size()));
        }
        FactorKeep fk;
        std::unique_ptr<AnalysisKeep> ak;
        solve_accurately(full_diag, ctrl, fk, ak);
        EXPECT_TRUE(ak->matching_scale.empty());
        EXPECT_TRUE(fk.scale.empty());
    }
    {
        // Explicit matching ordering.
        Control mc = ctrl;
        mc.ordering = OrderingMethod::Matching;
        CscLower kkt = make_random_kkt(8000, 3000, 3, 8);
        Solver solver;
        Info info;
        auto ak = solver.analyse(kkt, mc, info);
        ASSERT_NE(ak.get(), nullptr);
        EXPECT_EQ(info.ordering_used, static_cast<int>(OrderingMethod::Matching));
        FactorKeep fk;
        std::unique_ptr<AnalysisKeep> ak2;
        solve_accurately(kkt, mc, fk, ak2);
        // With its ordering, the matching's scaling is kept and applied.
        ASSERT_FALSE(ak2->matching_scale.empty());
        EXPECT_EQ(fk.scale, ak2->matching_scale);
    }
}

TEST(StaticPivoting, NeverDelaysAndRefinesToAccuracy) {
    // Zero-diagonal KKT blocks force delays under threshold pivoting. With
    // static pivoting on the matching ordering (the combination that solves
    // GHS_indef's cvxqp3 / ncvxqp3) the fronts accept every column,
    // perturbing the ones with no acceptable pivot, and refinement recovers
    // full accuracy.
    CscLower A = make_random_kkt(1500, 600, 3, 5);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.scaling = ScalingMethod::Matching;
    Solver solver;
    NumericWorkspace nws;
    SolveWorkspace sws;

    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk_delay;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk_delay, nws), FactorStatus::Success);
    ASSERT_GT(info.delayed_pivots, 0) << "matrix must exercise delays";
    EXPECT_EQ(info.static_pivots, 0);

    ctrl.ordering = OrderingMethod::Matching;
    ctrl.static_pivot_tol = 1e-10;
    auto akm = solver.analyse(A, ctrl, info);
    ASSERT_NE(akm.get(), nullptr);
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*akm, A, ctrl, info, fk, nws), FactorStatus::Success);
    EXPECT_EQ(info.delayed_pivots, 0);
    EXPECT_EQ(info.num_zero, 0);

    std::vector<double> b(static_cast<std::size_t>(A.n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::cos(0.21 * static_cast<double>(i));
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), static_cast<int>(A.n), 1, sws), 0);
    const RefinementResult rr = refine(A, fk, ctrl, b, x, 1, sws);
    EXPECT_EQ(rr.status, ErrorCode::Success) << "static_pivots=" << info.static_pivots
                                             << " eta=" << rr.eta_inf << " omega=" << rr.omega;
    EXPECT_TRUE(smf::bench::is_qualified_validation(smf::bench::validate(A, b, x, 1)));
}

TEST(StaticPivoting, ForcedPivotsAreCountedAndRefined) {
    // A 2x2 block [[0, 1], [1, 0]] behind tiny couplings: in the leaf front
    // the zero-diagonal column has no partner, so threshold pivoting delays
    // it; static pivoting must accept it as a perturbed 1x1 instead.
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 2, 4, 5};
    A.row_idx = {0, 2, 1, 2, 2};
    A.values = {0.0, 1.0, 4.0, 1.0, 0.0};
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.scaling = ScalingMethod::None;
    ctrl.nemin = 1;
    ctrl.static_pivot_tol = 1e-8;
    Solver solver;
    NumericWorkspace nws;
    SolveWorkspace sws;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);
    EXPECT_EQ(info.delayed_pivots, 0);
    EXPECT_GT(info.static_pivots, 0);

    // x = [1, 2, 3]: b = A x.
    std::vector<double> b = {3.0, 8.0 + 3.0, 1.0 + 2.0};
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 3, 1, sws), 0);
    refine(A, fk, ctrl, b, x, 1, sws);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
    EXPECT_NEAR(x[2], 3.0, 1e-10);
}

TEST(Scaling, UserSuppliedScaleIsApplied) {
    CscLower A = make_badly_scaled_indef(1e6);
    std::vector<double> s;
    ASSERT_EQ(compute_matching_scale(A, s, 0), ErrorCode::Success);

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.scaling = ScalingMethod::UserSupplied;
    ctrl.user_scale = s.data();
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);
    EXPECT_EQ(fk.scale, s);

    std::vector<double> b = {1.0, -2.0, 3.0, -4.0};
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), 4, 1, sws), 0);
    RefinementResult rr = refine(A, fk, ctrl, b, x, 1, sws);
    EXPECT_EQ(rr.status, ErrorCode::Success);
    EXPECT_TRUE(smf::bench::is_qualified_validation(smf::bench::validate(A, b, x, 1)));

    // A missing or non-positive user scale is rejected.
    ctrl.user_scale = nullptr;
    FactorKeep fk_bad;
    EXPECT_NE(solver.factor(*ak, A, ctrl, info, fk_bad, nws), FactorStatus::Success);
    std::vector<double> bad = s;
    bad[2] = 0.0;
    ctrl.user_scale = bad.data();
    EXPECT_NE(solver.factor(*ak, A, ctrl, info, fk_bad, nws), FactorStatus::Success);
}

// acopf_grid_160 (recorded IPOPT KKT sequence, max|A| ~ 3e11 on the last
// step): the replay harness's refinement stopped on a normwise criterion
// (eta <= 1e-14) that is met with a huge margin by ||A|| alone, so it never
// corrected a componentwise backward error of ~1e-6 in rows whose own entries
// are small. One more refinement sweep brings omega to ~1e-15.
TEST(Refinement, Acopf160) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" /
                         "fixtures" / "ipopt_kkt" / "acopf_grid_160";
    const fs::path mtx = dir / "acopf_grid_160.mtx";
    const fs::path vals = dir / "acopf_grid_160.mtx.vals";
    std::error_code ec;
    if (!fs::exists(mtx, ec) || !fs::exists(vals, ec)) {
        GTEST_SKIP() << "fixture not present: " << mtx.string();
    }
    auto mm = read_matrix_market(mtx.string());
    ASSERT_TRUE(mm.ok) << mm.error_message;
    CscLower A = std::move(mm.matrix);

    // Last recorded factorization's values.
    const std::size_t step_bytes = A.values.size() * sizeof(double);
    std::ifstream in(vals, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(in.good());
    const std::size_t nsteps = static_cast<std::size_t>(in.tellg()) / step_bytes;
    ASSERT_GE(nsteps, 1u);
    std::vector<double> last(A.values.size());
    in.seekg(static_cast<std::streamoff>((nsteps - 1) * step_bytes));
    in.read(reinterpret_cast<char *>(last.data()), static_cast<std::streamsize>(step_bytes));
    ASSERT_TRUE(in.good());

    // IPOPT track: u = 1e-8, single thread, auto ordering/scaling.
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.num_threads = 1;
    ctrl.pivot_u = 1e-8;
    Info info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    NumericWorkspace nws;
    SolveWorkspace sws;
    ASSERT_EQ(solver.factor(*ak, last, ctrl, info, fk, nws), FactorStatus::Success);

    const int n = static_cast<int>(A.n);
    const std::vector<double> b(static_cast<std::size_t>(n), 1.0);
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), n, 1, sws), 0);

    // The harness's refinement: up to 3 correction solves.
    bench::refine_solution(
        [&](std::vector<double> &r) {
            return solver.solve(fk, ctrl, info, r.data(), n, 1, sws) == 0;
        },
        A, last, b, x, 3);

    CscLower A_last = A;
    A_last.values = last;
    const auto v = bench::validate(A_last, b, x, 1);
    EXPECT_TRUE(bench::is_qualified_validation(v))
        << "eta_inf=" << v.eta_inf << " omega=" << v.omega;
}

// ---------------------------------------------------------------------------
// FGMRES escalation (RefineMethod::Auto)
// ---------------------------------------------------------------------------

namespace {

// Saddle point [[K, G^T], [G, 0]] with a pressure null space: velocities live
// on the edges of a w x w pressure grid, G is the edge-node incidence
// (gradient) so every column of G sums to zero and [0; 1] spans the null
// space. K = 4 I with -0.5 couplings between consecutive edges (SPD).
CscLower make_singular_stokes(int w) {
    const int np = w * w;
    std::vector<std::pair<int, int>> edges;  // (tail, head) pressure nodes
    for (int r = 0; r < w; ++r) {
        for (int c = 0; c < w; ++c) {
            const int p = r * w + c;
            if (c + 1 < w) edges.push_back({p, p + 1});
            if (r + 1 < w) edges.push_back({p, p + w});
        }
    }
    const int ne = static_cast<int>(edges.size());
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(ne + np));
    for (int e = 0; e < ne; ++e) {
        auto &col = cols[static_cast<std::size_t>(e)];
        col.push_back({e, 4.0});
        if (e + 1 < ne) col.push_back({e + 1, -0.5});
        col.push_back({ne + edges[static_cast<std::size_t>(e)].first, -1.0});
        col.push_back({ne + edges[static_cast<std::size_t>(e)].second, 1.0});
    }
    CscLower A;
    A.n = ne + np;
    A.col_ptr.push_back(0);
    for (auto &c : cols) {
        std::sort(c.begin(), c.end());
        for (const auto &[i, v] : c) {
            A.row_idx.push_back(i);
            A.values.push_back(v);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

// Well-conditioned SPD: 2D 5-point stencil on a g x g grid, diagonal 8.
CscLower make_spd_laplacian(int g) {
    CscLower A;
    A.n = g * g;
    A.col_ptr.push_back(0);
    for (int j = 0; j < g * g; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(8.0);
        if ((j % g) + 1 < g) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(-1.0);
        }
        if (j + g < g * g) {
            A.row_idx.push_back(j + g);
            A.values.push_back(-1.0);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

// Classical refinement steps on ClassicalPathUnchanged's system, recorded
// with the refinement as it was before FGMRES existed.
constexpr int kClassicalStepsBeforeFgmres = 1;

// Nonconvex KKT [H J^T; J 0]: H is a tridiagonal (4, -1) block on nv
// variables plus one 3-variable nonconvex block [[0, 0, 1], [0, 4, 1],
// [1, 1, -3000]] (which sets max|A| = 3000); J has nc rows with two
// entries -1 / +1 on distinct variables.
CscLower make_ncvx_kkt(int nv, int nc, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, nv - 1);
    const int nb = 3;
    const int n = nb + nv + nc;
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n));
    cols[0] = {{0, 0.0}, {2, 1.0}};
    cols[1] = {{1, 4.0}, {2, 1.0}};
    cols[2] = {{2, -3000.0}};
    for (int j = 0; j < nv; ++j) {
        cols[static_cast<std::size_t>(nb + j)].push_back({nb + j, 4.0});
        if (j + 1 < nv) cols[static_cast<std::size_t>(nb + j)].push_back({nb + j + 1, -1.0});
    }
    for (int r = 0; r < nc; ++r) {
        const int j0 = pick(rng);
        const int j1 = (j0 + 1 + pick(rng) % (nv - 1)) % nv;
        cols[static_cast<std::size_t>(nb + j0)].push_back({nb + nv + r, -1.0});
        cols[static_cast<std::size_t>(nb + j1)].push_back({nb + nv + r, 1.0});
    }
    CscLower A;
    A.n = n;
    A.col_ptr.push_back(0);
    for (auto &col : cols) {
        std::sort(col.begin(), col.end());
        for (const auto &[i, v] : col) {
            A.row_idx.push_back(i);
            A.values.push_back(v);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

} // namespace

TEST(Refinement, FgmresRecoversStaticPivots) {
    // Static pivoting with a deliberately large tolerance and no scaling: the
    // zero pivots (every constraint row, the nonconvex block's free variable)
    // are raised to 1e-6 max|A| = 3e-3 against O(1) constraint entries, too
    // far for classical refinement to recover.
    CscLower A = make_ncvx_kkt(400, 150, 3);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.scaling = ScalingMethod::None;
    ctrl.nemin = 1;
    ctrl.static_pivot_tol = 1e-6;
    EXPECT_EQ(ctrl.refine_method, RefineMethod::Auto);
    EXPECT_EQ(ctrl.refine_max_steps, 10);
    EXPECT_EQ(ctrl.fgmres_restart, 20);
    Solver solver;
    NumericWorkspace nws;
    SolveWorkspace sws;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);
    ASSERT_GT(info.static_pivots, 0);
    EXPECT_EQ(fk.static_pivots, info.static_pivots);

    const int n = static_cast<int>(A.n);
    std::vector<double> b(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::cos(0.21 * static_cast<double>(i));
    std::vector<double> x0 = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x0.data(), n, 1, sws), 0);

    Control classical = ctrl;
    classical.refine_method = RefineMethod::Classical;
    std::vector<double> xc = x0;
    const RefinementResult rc = refine(A, fk, classical, b, xc, 1, sws);
    ASSERT_NE(rc.status, ErrorCode::Success)
        << "classical refinement must stagnate here: eta=" << rc.eta_inf << " omega=" << rc.omega;
    EXPECT_EQ(rc.method_used, RefineMethod::Classical);

    std::vector<double> x = x0;
    const RefinementResult ra = refine(A, fk, ctrl, b, x, 1, sws);
    EXPECT_EQ(ra.status, ErrorCode::Success)
        << "iters=" << ra.iterations << " eta=" << ra.eta_inf << " omega=" << ra.omega;
    EXPECT_EQ(ra.method_used, RefineMethod::Fgmres);
    EXPECT_LE(ra.iterations, ctrl.refine_max_steps);
    EXPECT_LE(ra.eta_inf, 1e-10);
    EXPECT_LE(ra.omega, 1e-8);
    EXPECT_TRUE(smf::bench::is_qualified_validation(smf::bench::validate(A, b, x, 1)));

    // Fgmres from the start reaches the gate as well.
    Control fg = ctrl;
    fg.refine_method = RefineMethod::Fgmres;
    std::vector<double> xf = x0;
    const RefinementResult rf = refine(A, fk, fg, b, xf, 1, sws);
    EXPECT_EQ(rf.status, ErrorCode::Success) << "eta=" << rf.eta_inf << " omega=" << rf.omega;
    EXPECT_EQ(rf.method_used, RefineMethod::Fgmres);
    EXPECT_LE(rf.iterations, fg.refine_max_steps);
}

namespace {

// Factor a singular saddle point with static pivoting (so the factor exists)
// and refine an inconsistent right-hand side: no x meets the gate.
void expect_singular_terminates(const CscLower &A, const std::vector<double> &b) {
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.static_pivot_tol = 1e-8;
    Solver solver;
    NumericWorkspace nws;
    SolveWorkspace sws;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);
    const int n = static_cast<int>(A.n);
    std::vector<double> x0 = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x0.data(), n, 1, sws), 0);
    for (RefineMethod m : {RefineMethod::Auto, RefineMethod::Fgmres, RefineMethod::Classical}) {
        Control c = ctrl;
        c.refine_method = m;
        std::vector<double> x = x0;
        const RefinementResult rr = refine(A, fk, c, b, x, 1, sws);
        EXPECT_NE(rr.status, ErrorCode::Success) << "method " << static_cast<int>(m);
        EXPECT_LE(rr.iterations, c.refine_max_steps) << "method " << static_cast<int>(m);
        if (m == RefineMethod::Fgmres) EXPECT_EQ(rr.method_used, RefineMethod::Fgmres);
        if (m == RefineMethod::Classical) EXPECT_EQ(rr.method_used, RefineMethod::Classical);
        bool finite = true;
        for (double v : x) finite = finite && std::isfinite(v);
        EXPECT_TRUE(finite) << "method " << static_cast<int>(m);
    }
}

} // namespace

TEST(Refinement, SingularDoesNotLoop) {
    // Generated pressure-null-space Stokes; b has a component along [0; 1].
    CscLower A = make_singular_stokes(20);
    std::vector<double> b(static_cast<std::size_t>(A.n), 1.0);
    expect_singular_terminates(A, b);
}

TEST(Refinement, SingularDoesNotLoopTuronM) {
    namespace fs = std::filesystem;
    const fs::path mtx = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" / "fixtures" /
                         "large" / "turon_m" / "turon_m.mtx";
    std::error_code ec;
    if (!fs::exists(mtx, ec)) GTEST_SKIP() << "fixture not present: " << mtx.string();
    auto mm = read_matrix_market(mtx.string());
    ASSERT_TRUE(mm.ok) << mm.error_message;
    std::vector<double> b(static_cast<std::size_t>(mm.matrix.n), 1.0);
    expect_singular_terminates(mm.matrix, b);
}

TEST(Refinement, ClassicalPathUnchanged) {
    // Well-conditioned SPD: Auto never escalates and takes exactly the steps
    // classical refinement took before FGMRES existed.
    CscLower A = make_spd_laplacian(30);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    Solver solver;
    NumericWorkspace nws;
    SolveWorkspace sws;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);
    const int n = static_cast<int>(A.n);
    std::vector<double> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = std::sin(0.37 * i) + 0.5;
    std::vector<double> x0 = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x0.data(), n, 1, sws), 0);
    // Perturb so that refinement has to take steps.
    for (int i = 0; i < n; ++i) x0[static_cast<std::size_t>(i)] *= 1.0 + 1e-6 * std::cos(1.3 * i);

    Control classical = ctrl;
    classical.refine_method = RefineMethod::Classical;
    std::vector<double> xc = x0;
    const RefinementResult rc = refine(A, fk, classical, b, xc, 1, sws);
    std::vector<double> xa = x0;
    const RefinementResult ra = refine(A, fk, ctrl, b, xa, 1, sws);

    EXPECT_EQ(rc.status, ErrorCode::Success);
    EXPECT_EQ(ra.status, ErrorCode::Success);
    EXPECT_EQ(rc.method_used, RefineMethod::Classical);
    EXPECT_EQ(ra.method_used, RefineMethod::Classical);
    EXPECT_EQ(rc.iterations, kClassicalStepsBeforeFgmres);
    EXPECT_EQ(ra.iterations, rc.iterations);
    EXPECT_EQ(xa, xc);  // bit-identical iterates
}
