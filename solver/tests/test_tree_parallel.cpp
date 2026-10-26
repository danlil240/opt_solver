// Tree-parallel factorization (Control::num_threads > 1): independent
// subtrees on worker threads, the top of the tree after. Results must match
// the serial factorization: same inertia and delays (each front is factored
// by the same code on the same data), solutions to rounding.

#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace smf;

namespace {

// 5-point Laplacian on an m x m grid (SPD), lower CSC.
CscLower laplacian_2d(int m) {
    CscLower A;
    A.n = m * m;
    A.col_ptr.push_back(0);
    for (int j = 0; j < A.n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(4.0);
        if ((j % m) + 1 < m) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(-1.0);
        }
        if (j + m < A.n) {
            A.row_idx.push_back(j + m);
            A.values.push_back(-1.0);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

// Saddle point [L B^T; B 0]: L a 2D Laplacian on m x m, B picks sums of
// neighbouring pairs (one constraint per `stride` unknowns).
CscLower grid_kkt(int m, int stride) {
    const int nv = m * m;
    const int nc = nv / stride;
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(nv + nc));
    const CscLower L = laplacian_2d(m);
    for (int j = 0; j < nv; ++j)
        for (Int p = L.col_ptr[static_cast<std::size_t>(j)]; p < L.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            cols[static_cast<std::size_t>(j)].push_back({L.row_idx[static_cast<std::size_t>(p)], L.values[static_cast<std::size_t>(p)]});
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> unif(0.5, 2.0);
    for (int r = 0; r < nc; ++r) {
        const int v = r * stride;
        cols[static_cast<std::size_t>(v)].push_back({nv + r, unif(rng)});
        if (v + 1 < nv) cols[static_cast<std::size_t>(v + 1)].push_back({nv + r, -unif(rng)});
    }
    CscLower A;
    A.n = nv + nc;
    A.col_ptr.push_back(0);
    for (auto &c : cols) {
        std::sort(c.begin(), c.end());
        for (const auto &[i, x] : c) {
            A.row_idx.push_back(i);
            A.values.push_back(x);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

struct Run {
    Info info;
    std::vector<double> x;
};

Run factor_and_solve(const CscLower &A, Control ctrl, int threads, int refactors = 1) {
    ctrl.num_threads = threads;
    Solver solver;
    Run run;
    auto ak = solver.analyse(A, ctrl, run.info);
    EXPECT_NE(ak.get(), nullptr);
    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;
    for (int r = 0; r < refactors; ++r)
        EXPECT_EQ(solver.factor(*ak, A, ctrl, run.info, fk, nws), FactorStatus::Success);
    run.x.assign(static_cast<std::size_t>(A.n), 0.0);
    for (std::size_t i = 0; i < run.x.size(); ++i) run.x[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
    const std::vector<double> b = run.x;
    EXPECT_EQ(solver.solve(fk, ctrl, run.info, run.x.data(), static_cast<int>(A.n), 1, sws), 0);
    const auto v = smf::bench::validate(A, b, run.x, 1);
    EXPECT_LT(v.eta_inf, 1e-12);
    return run;
}

double max_rel_diff(const std::vector<double> &a, const std::vector<double> &b) {
    double d = 0.0, s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::abs(a[i] - b[i]));
        s = std::max(s, std::abs(a[i]));
    }
    return d / s;
}

} // namespace

TEST(TreeParallel, SpdMatchesSerial) {
    const CscLower A = laplacian_2d(120);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    const Run serial = factor_and_solve(A, ctrl, 1);
    const Run par = factor_and_solve(A, ctrl, 4, /*refactors=*/2);
    EXPECT_EQ(serial.info.parallel_subtrees, 0);
    ASSERT_GE(par.info.parallel_subtrees, 2) << "parallel path not taken";
    EXPECT_LT(max_rel_diff(serial.x, par.x), 1e-12);
    EXPECT_EQ(par.info.actual_factor_entries, serial.info.actual_factor_entries);
}

TEST(TreeParallel, IndefiniteMatchesSerial) {
    const CscLower A = grid_kkt(110, 7);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.scaling = ScalingMethod::None;
    const Run serial = factor_and_solve(A, ctrl, 1);
    ASSERT_GT(serial.info.num_negative, 0);
    const Run par = factor_and_solve(A, ctrl, 4, /*refactors=*/2);
    ASSERT_GE(par.info.parallel_subtrees, 2) << "parallel path not taken";
    EXPECT_EQ(par.info.num_positive, serial.info.num_positive);
    EXPECT_EQ(par.info.num_negative, serial.info.num_negative);
    EXPECT_EQ(par.info.num_zero, serial.info.num_zero);
    EXPECT_EQ(par.info.delayed_pivots, serial.info.delayed_pivots);
    EXPECT_LT(max_rel_diff(serial.x, par.x), 1e-10);
}

namespace {

// Factor A on 4 threads (tree-parallel), then solve the same factor with
// 1 and with 4 solve threads: 3 right-hand sides, the full solve and the
// split Forward + DiagBack jobs, on one reused workspace (pool kept, then
// re-created at a different size).
void check_parallel_solve(const CscLower &A, Control ctrl, bool expect_delays = false) {
    ctrl.num_threads = 4;
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace nws;
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);
    if (expect_delays) ASSERT_GT(info.delayed_pivots, 0);
    ASSERT_GE(fk.solve_subtrees.size(), 2u) << "no parallel solve layout";
    ASSERT_FALSE(fk.solve_top_rows.empty());

    const int n = static_cast<int>(A.n), nrhs = 3;
    std::vector<double> b(static_cast<std::size_t>(n) * nrhs);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.017 * static_cast<double>(i)) + 0.3;

    Control serial = ctrl;
    serial.num_threads = 1;
    SolveWorkspace sws1;
    std::vector<double> x1 = b;
    ASSERT_EQ(solver.solve(fk, serial, info, x1.data(), n, nrhs, sws1), 0);

    Control par = ctrl;
    par.solve_parallel_min_entries = 0;
    SolveWorkspace sws;
    for (const int threads : {4, 4, 3}) {
        par.num_threads = threads;
        std::vector<double> x = b;
        ASSERT_EQ(solver.solve(fk, par, info, x.data(), n, nrhs, sws), 0);
        EXPECT_LT(max_rel_diff(x1, x), 1e-12) << threads << " threads";

        std::vector<double> y = b;
        ASSERT_EQ(solver.solve(fk, par, info, y.data(), n, nrhs, sws, SolveJob::Forward), 0);
        ASSERT_EQ(solver.solve(fk, par, info, y.data(), n, nrhs, sws, SolveJob::DiagBack), 0);
        EXPECT_LT(max_rel_diff(x1, y), 1e-12) << threads << " threads, split jobs";
    }
    for (int r = 0; r < nrhs; ++r) {
        const std::vector<double> br(b.begin() + r * n, b.begin() + (r + 1) * n);
        const std::vector<double> xr(x1.begin() + r * n, x1.begin() + (r + 1) * n);
        EXPECT_LT(smf::bench::validate(A, br, xr, 1).eta_inf, 1e-12);
    }
}

} // namespace

TEST(TreeParallel, ParallelSolveSpd) {
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    check_parallel_solve(laplacian_2d(120), ctrl);
}

TEST(TreeParallel, ParallelSolveIndefiniteWithDelays) {
    // Unscaled saddle point with a strict threshold: pivots are delayed,
    // some out of the subtrees into the top part.
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.scaling = ScalingMethod::None;
    ctrl.pivot_u = 0.5;
    check_parallel_solve(grid_kkt(110, 7), ctrl, /*expect_delays=*/true);
}
