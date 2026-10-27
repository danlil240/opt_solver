// Tree-parallel factorization (Control::num_threads > 1): independent
// subtrees on worker threads, the top of the tree after. Results must match
// the serial factorization: same inertia and delays (each front is factored
// by the same code on the same data), solutions to rounding.

#include "../benchmarks/common/validation.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/determinism.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
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
    if (deterministic_mode(ctrl)) {
        // Deterministic mode lays every front out in postorder and solves
        // serially: no parallel solve layout. The checks below still hold.
        EXPECT_TRUE(fk.solve_subtrees.empty()) << "deterministic mode: parallel solve layout";
    } else {
        ASSERT_GE(fk.solve_subtrees.size(), 2u) << "no parallel solve layout";
        ASSERT_FALSE(fk.solve_top_rows.empty());
    }

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

namespace {

// 3D Helmholtz on an m^3 grid: 7-point Laplacian minus a shift that puts
// the operator well inside its spectrum (indefinite), lower CSC.
CscLower helmholtz_3d(int m, double shift) {
    const int n = m * m * m;
    CscLower A;
    A.n = n;
    A.col_ptr.push_back(0);
    for (int z = 0; z < m; ++z)
        for (int y = 0; y < m; ++y)
            for (int x = 0; x < m; ++x) {
                const int j = (z * m + y) * m + x;
                A.row_idx.push_back(j);
                A.values.push_back(6.0 - shift);
                if (x + 1 < m) {
                    A.row_idx.push_back(j + 1);
                    A.values.push_back(-1.0);
                }
                if (y + 1 < m) {
                    A.row_idx.push_back(j + m);
                    A.values.push_back(-1.0);
                }
                if (z + 1 < m) {
                    A.row_idx.push_back(j + m * m);
                    A.values.push_back(-1.0);
                }
                A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
            }
    return A;
}

// helm3d01 if the fixture is present, otherwise a generated 3D Helmholtz
// 30^3; `name` says which.
CscLower helmholtz_case(std::string &name) {
    namespace fs = std::filesystem;
    const fs::path mtx = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" /
                         "fixtures" / "large" / "helm3d01" / "helm3d01.mtx";
    std::error_code ec;
    if (fs::exists(mtx, ec)) {
        auto mm = read_matrix_market(mtx.string());
        if (mm.ok) {
            name = "helm3d01";
            return std::move(mm.matrix);
        }
    }
    name = "helmholtz_3d_30";
    return helmholtz_3d(30, 0.9);
}

struct RefinedRun {
    Info info;
    double eta = 1.0;
};

// Factor and solve with a few steps of classical refinement (residual in
// double), then the normwise backward error.
RefinedRun factor_refined(const CscLower &A, Control ctrl, int threads) {
    ctrl.num_threads = threads;
    Solver solver;
    RefinedRun run;
    auto ak = solver.analyse(A, ctrl, run.info);
    EXPECT_NE(ak.get(), nullptr);
    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;
    EXPECT_EQ(solver.factor(*ak, A, ctrl, run.info, fk, nws), FactorStatus::Success);
    const int n = static_cast<int>(A.n);
    std::vector<double> b(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
    std::vector<double> x = b, r(b.size());
    EXPECT_EQ(solver.solve(fk, ctrl, run.info, x.data(), n, 1, sws), 0);
    for (int it = 0; it < 3; ++it) {
        r = b;
        for (Int j = 0; j < A.n; ++j)
            for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
                const std::size_t i = static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)]);
                const double v = A.values[static_cast<std::size_t>(p)];
                r[i] -= v * x[static_cast<std::size_t>(j)];
                if (i != static_cast<std::size_t>(j)) r[static_cast<std::size_t>(j)] -= v * x[i];
            }
        EXPECT_EQ(solver.solve(fk, ctrl, run.info, r.data(), n, 1, sws), 0);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += r[i];
    }
    run.eta = smf::bench::validate(A, b, x, 1).eta_inf;
    return run;
}

} // namespace

// Fronts of >= 512 rows at the top of the tree are factored as 2D tile
// tasks (Control::top_tiles) with 8 threads: same inertia as the serial
// factorization and the same backward error class.
TEST(TreeParallel, TileTasksMatchSerial) {
    std::string name;
    const CscLower A = helmholtz_case(name);
    SCOPED_TRACE(name);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    const RefinedRun serial = factor_refined(A, ctrl, 1);
    ASSERT_GT(serial.info.num_negative, 0) << "not indefinite";
    // Deterministic mode (Control::deterministic or an SMF_DETERMINISTIC
    // build) tiles the large plain fronts at every thread count, 1 included.
    if (deterministic_mode(ctrl))
        EXPECT_GT(serial.info.top_tile_fronts, 0) << "deterministic mode: no tile front at 1 thread";
    else
        EXPECT_EQ(serial.info.top_tile_fronts, 0);
    const RefinedRun par = factor_refined(A, ctrl, 8);
    ASSERT_GE(par.info.parallel_subtrees, 2) << "parallel path not taken";
    EXPECT_GT(par.info.top_tile_fronts, 0) << "no front factored as tile tasks";
    EXPECT_EQ(par.info.num_positive, serial.info.num_positive);
    EXPECT_EQ(par.info.num_negative, serial.info.num_negative);
    EXPECT_EQ(par.info.num_zero, serial.info.num_zero);
    EXPECT_LE(serial.eta, 1e-14);
    EXPECT_LE(par.eta, 1e-14);

    // top_tiles off: today's threaded-BLAS top phase, same inertia.
    ctrl.top_tiles = false;
    const RefinedRun off = factor_refined(A, ctrl, 8);
    EXPECT_EQ(off.info.top_tile_fronts, 0);
    EXPECT_EQ(off.info.num_negative, serial.info.num_negative);
}

// R42: tile tasks (APTP structure) lack TPP's weak-1x1 rule, so below
// pivot u = 1e-6 (kWeakPivotRatio; IPOPT's u = 1e-8) no front is tiled at
// any thread count, deterministic mode included: those fronts keep the TPP
// kernel. At u = 1e-6 tiles are still on.
TEST(TreeParallel, NoTilesBelowWeakPivotU) {
    std::string name;
    const CscLower A = helmholtz_case(name);
    SCOPED_TRACE(name);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.pivot_u = 1e-8;
    const RefinedRun par = factor_refined(A, ctrl, 4);
    ASSERT_GE(par.info.parallel_subtrees, 2) << "parallel path not taken";
    EXPECT_EQ(par.info.top_tile_fronts, 0) << "u = 1e-8, 4 threads: a front was tiled";
    EXPECT_LE(par.eta, 1e-12);
    const RefinedRun serial = factor_refined(A, ctrl, 1);
    EXPECT_EQ(serial.info.top_tile_fronts, 0);
    EXPECT_EQ(par.info.num_negative, serial.info.num_negative);
    EXPECT_EQ(par.info.num_positive, serial.info.num_positive);

    ctrl.deterministic = true;
    for (const int threads : {1, 4}) {
        const RefinedRun det = factor_refined(A, ctrl, threads);
        EXPECT_EQ(det.info.top_tile_fronts, 0) << "deterministic, " << threads << " threads";
        EXPECT_EQ(det.info.num_negative, serial.info.num_negative) << threads << " threads";
    }

    ctrl.deterministic = false;
    ctrl.pivot_u = 1e-6;  // the gate is inclusive
    const RefinedRun on = factor_refined(A, ctrl, 4);
    EXPECT_GT(on.info.top_tile_fronts, 0) << "u = 1e-6, 4 threads: tiles off";
}

// The BLAS threads in use (BlasThreadGuard accounting) never exceed
// num_threads: a tile front's owner drops to serial BLAS and each helper
// counts one thread within the front's share.
TEST(TreeParallel, TileTasksNoOversubscription) {
    std::string name;
    const CscLower A = helmholtz_case(name);
    SCOPED_TRACE(name);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    for (const int threads : {8, 3}) {
        ctrl.num_threads = threads;
        Solver solver;
        Info info;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace nws;
        FactorKeep fk;
        ASSERT_EQ(blas_threads_active(), 0);
        blas_threads_reset_peak();
        ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);
        const int peak = blas_threads_peak();
        EXPECT_EQ(blas_threads_active(), 0);
        EXPECT_GT(info.top_tile_fronts, 0) << threads << " threads";
        EXPECT_GE(peak, 2) << threads << " threads";
        EXPECT_LE(peak, threads) << threads << " threads, peak " << peak;
    }
}
