// Tree-parallel factorization (Control::num_threads > 1): independent
// subtrees on worker threads, the top of the tree after. Results must match
// the serial factorization: same inertia and delays (each front is factored
// by the same code on the same data), solutions to rounding.

#include "../benchmarks/common/validation.hpp"
#include "kkt_test_matrices.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/determinism.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/solve_pool.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <thread>
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
    ctrl.auto_threads = false;  // small test trees
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
    ctrl.auto_threads = false;  // small test trees
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
        EXPECT_GE(fk.solve_threads, 2) << "parallel solve not chosen";
    }

    const int n = static_cast<int>(A.n), nrhs = 3;
    std::vector<double> b(static_cast<std::size_t>(n) * nrhs);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.017 * static_cast<double>(i)) + 0.3;

    Control serial = ctrl;
    serial.num_threads = 1;
    serial.auto_threads = false;  // small test trees
    SolveWorkspace sws1;
    std::vector<double> x1 = b;
    ASSERT_EQ(solver.solve(fk, serial, info, x1.data(), n, nrhs, sws1), 0);

    Control par = ctrl;
    par.solve_parallel_min_entries = 0;
    SolveWorkspace sws;
    for (const int threads : {4, 4, 3}) {
        par.num_threads = threads;
        par.auto_threads = false;  // small test trees
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
// 30^3; `name` says which. The tile tests set Control::top_tile_min_rows =
// kTileTestMinRows: the generated grid's largest fronts (~1700 rows) are
// below the default threshold (2048 outside deterministic mode).
constexpr int kTileTestMinRows = 512;

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
    ctrl.auto_threads = false;  // small test trees
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

// Fronts of >= 512 rows (top_tile_min_rows here) at the top of the tree
// are factored as 2D tile tasks (Control::top_tiles) with 8 threads: same inertia as the serial
// factorization and the same backward error class.
TEST(TreeParallel, TileTasksMatchSerial) {
    std::string name;
    const CscLower A = helmholtz_case(name);
    SCOPED_TRACE(name);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.top_tile_min_rows = kTileTestMinRows;
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
    ctrl.top_tile_min_rows = kTileTestMinRows;
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
    ctrl.top_tile_min_rows = kTileTestMinRows;
    for (const int threads : {8, 3}) {
        ctrl.num_threads = threads;
        ctrl.auto_threads = false;  // small test trees
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

namespace {

// Banded SPD matrix of order n and half-bandwidth bw, lower CSC.
CscLower banded_spd(int n, int bw) {
    CscLower A;
    A.n = n;
    A.col_ptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(2.0 * bw + 1.0);
        for (int i = j + 1; i <= std::min(n - 1, j + bw); ++i) {
            A.row_idx.push_back(i);
            A.values.push_back(-1.0);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

} // namespace

// Control::auto_threads: a chain-like tree (a banded matrix) repays no
// thread, so it factors on one at num_threads = 16; with auto_threads off
// the same tree is partitioned for all 16. (Bandwidth 30, not 10: at 10 the
// tree's front work, 6.5e6, is below the 2e7 partition floor.)
TEST(TreeParallel, AutoThreadsPicksSerialForChain) {
    const CscLower A = banded_spd(20000, 30);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.num_threads = 16;
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    double work = 0.0;
    for (std::size_t s = 0; s < ak->supernodes.size(); ++s) {
        const double f = static_cast<double>(ak->front_size(static_cast<Int>(s)));
        work += static_cast<double>(ak->supernodes[s].width()) * f * f;
    }
    ASSERT_GT(work, 2e7) << "premise: heavy enough to partition";

    for (const bool auto_threads : {true, false}) {
        Control c = ctrl;
        c.auto_threads = auto_threads;
        NumericWorkspace nws;
        FactorKeep fk;
        Info fi;
        ASSERT_EQ(solver.factor(*ak, A, c, fi, fk, nws), FactorStatus::Success);
        if (auto_threads) {
            EXPECT_EQ(fi.threads_used, 1);
            EXPECT_EQ(fi.parallel_subtrees, 0);
        } else {
            EXPECT_GE(fi.parallel_subtrees, 2);
        }
    }
}

// Control::auto_threads: the plan is cached in the NumericWorkspace per
// analysis, num_threads and cost values; changing any of them re-plans
// (threads_used follows), and a refactorization with the same ones reuses it.
TEST(TreeParallel, ThreadPlanCacheFollowsItsInputs) {
    const CscLower grid = laplacian_2d(200);
    const CscLower band = banded_spd(20000, 30);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.num_threads = 16;
    Solver solver;
    Info info;
    auto ak_grid = solver.analyse(grid, ctrl, info);
    ASSERT_NE(ak_grid.get(), nullptr);
    Control band_ctrl = ctrl;
    band_ctrl.ordering = OrderingMethod::AMD;
    auto ak_band = solver.analyse(band, band_ctrl, info);
    ASSERT_NE(ak_band.get(), nullptr);

    NumericWorkspace nws;  // shared by every factorization below
    FactorKeep fk;
    const auto threads_used = [&](const AnalysisKeep &ak, const CscLower &A, const Control &c) {
        Info fi;
        EXPECT_EQ(solver.factor(ak, A, c, fi, fk, nws), FactorStatus::Success);
        return fi.threads_used;
    };
    const int planned = threads_used(*ak_grid, grid, ctrl);
    ASSERT_GT(planned, 2) << "premise: the grid's tree repays more than 2 threads";
    EXPECT_EQ(threads_used(*ak_grid, grid, ctrl), planned);  // cached

    Control dear = ctrl;  // threads cost more than the whole factorization
    dear.thread_cost_per_thread = 1e30;
    EXPECT_EQ(threads_used(*ak_grid, grid, dear), 1);
    Control dear_top = ctrl;
    dear_top.thread_cost_per_top_front = 1e30;
    EXPECT_EQ(threads_used(*ak_grid, grid, dear_top), 1);
    EXPECT_EQ(threads_used(*ak_grid, grid, ctrl), planned);

    Control two = ctrl;
    two.num_threads = 2;
    EXPECT_EQ(threads_used(*ak_grid, grid, two), 2);
    EXPECT_EQ(threads_used(*ak_grid, grid, ctrl), planned);

    // Another analysis on the same workspace: its own plan (the chain: 1).
    EXPECT_EQ(threads_used(*ak_band, band, band_ctrl), 1);
    EXPECT_EQ(threads_used(*ak_grid, grid, ctrl), planned);
}

// Control::auto_threads covers the solve. Off, FactorKeep::solve_threads is
// the factorization's thread count whatever the solve costs: exactly 4 here,
// since auto off partitions the 200 x 200 grid for all 4 threads and it has
// far more than 4 subtrees (threads = min(4, subtrees)). On, a solve cost of
// 1e15 per thread keeps the solve serial on the same parallel layout.
TEST(TreeParallel, AutoThreadsOffFixesSolveThreads) {
    const CscLower A = laplacian_2d(200);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::METIS;
    ctrl.num_threads = 4;
    ctrl.solve_cost_per_thread = 1e15;
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    for (const bool auto_threads : {false, true}) {
        Control c = ctrl;
        c.auto_threads = auto_threads;
        NumericWorkspace nws;
        FactorKeep fk;
        Info fi;
        ASSERT_EQ(solver.factor(*ak, A, c, fi, fk, nws), FactorStatus::Success);
        ASSERT_GE(fi.threads_used, 2) << "premise: tree-parallel, auto " << auto_threads;
        ASSERT_GE(fk.solve_subtrees.size(), 2u) << "premise: parallel solve layout";
        if (auto_threads) {
            EXPECT_EQ(fk.solve_threads, 1);
        } else {
            EXPECT_EQ(fi.threads_used, 4);
            EXPECT_EQ(fk.solve_threads, 4);
        }
    }
}

// FactorKeep::solve_threads (the solve's own cost model) and the call's
// num_threads both cap the parallel solve: it runs on at most the smaller.
TEST(TreeParallel, SolveThreadsCappedByCall) {
    const CscLower A = laplacian_2d(200);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::METIS;
    ctrl.num_threads = 8;
    ctrl.auto_threads = false;  // small test trees
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace nws;
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);
    ASSERT_GE(fk.solve_subtrees.size(), 2u) << "premise: parallel solve layout";
    EXPECT_GE(fk.solve_threads, 2);

    std::vector<double> b(static_cast<std::size_t>(A.n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
    const auto solve_on = [&](int threads, SolveWorkspace &sws) {
        Control c = ctrl;
        c.num_threads = threads;
        std::vector<double> x = b;
        EXPECT_EQ(solver.solve(fk, c, info, x.data(), static_cast<int>(A.n), 1, sws), 0);
        EXPECT_LT(smf::bench::validate(A, b, x, 1).eta_inf, 1e-12) << threads << " threads";
    };
    SolveWorkspace sws;
    solve_on(2, sws);
    ASSERT_NE(sws.pool.get(), nullptr);
    EXPECT_LE(sws.pool->size(), 2);

    // A factor whose solve is serial: no solve pool at any num_threads.
    fk.solve_threads = 1;
    SolveWorkspace serial_ws;
    solve_on(8, serial_ws);
    EXPECT_EQ(serial_ws.pool.get(), nullptr);
}

TEST(TreeParallel, PhaseTimesReported) {
    const CscLower A = laplacian_2d(120);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    const Run serial = factor_and_solve(A, ctrl, 1);
    const Run par = factor_and_solve(A, ctrl, 4);
    EXPECT_EQ(serial.info.threads_used, 1);
    EXPECT_EQ(serial.info.subtree_seconds, 0.0);
    EXPECT_EQ(serial.info.top_seconds, 0.0);
    ASSERT_GE(par.info.parallel_subtrees, 2);
    EXPECT_GE(par.info.threads_used, 2);
    EXPECT_GT(par.info.subtree_seconds, 0.0);
    EXPECT_GT(par.info.top_seconds, 0.0);
    EXPECT_GE(par.info.gather_seconds, 0.0);
    EXPECT_LE(par.info.subtree_seconds + par.info.top_seconds + par.info.gather_seconds,
              par.info.factor_seconds);
}

// The factorization's worker threads live in the workspace's pool: repeated
// factorizations start no new threads.
TEST(TreeParallel, PoolPersistsAcrossFactorizations) {
    const CscLower A = laplacian_2d(120);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.num_threads = 4;
    ctrl.auto_threads = false;  // small test trees
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    FactorKeep fk;
    const long long t0 = SolvePool::threads_started();
    for (int r = 0; r < 3; ++r)
        ASSERT_EQ(s.factor(*ak, A, ctrl, info, fk, ws), FactorStatus::Success);
    ASSERT_GE(info.parallel_subtrees, 2);
    EXPECT_EQ(SolvePool::threads_started() - t0, 3);  // one pool of 4: three workers, once
}

// Idle workers count no BLAS threads.
TEST(TreeParallel, IdlePoolHoldsNoBlasThreads) {
    const CscLower A = laplacian_2d(120);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.num_threads = 4;
    ctrl.auto_threads = false;  // small test trees
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    FactorKeep fk;
    const int before = blas_threads_active();
    ASSERT_EQ(s.factor(*ak, A, ctrl, info, fk, ws), FactorStatus::Success);
    ASSERT_GE(info.parallel_subtrees, 2);
    ASSERT_NE(ws.pool.get(), nullptr);
    EXPECT_EQ(blas_threads_active(), before);
    // Still so after the workers have gone through their spin and sleep.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(blas_threads_active(), before);
}

// The pool is recreated when the thread count changes.
TEST(TreeParallel, PoolFollowsThreadCount) {
    const CscLower A = laplacian_2d(120);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.auto_threads = false;  // small test trees
    Solver s;
    Info info;
    auto ak = s.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace ws;
    SolveWorkspace sws;
    FactorKeep fk;
    long long started = 0;  // by the factorizations only: the solve has its own pool
    for (const int threads : {4, 8, 4}) {
        ctrl.num_threads = threads;
        const long long t0 = SolvePool::threads_started();
        ASSERT_EQ(s.factor(*ak, A, ctrl, info, fk, ws), FactorStatus::Success);
        started += SolvePool::threads_started() - t0;
        ASSERT_EQ(info.threads_used, threads);
        std::vector<double> x(static_cast<std::size_t>(A.n));
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
        const std::vector<double> b = x;
        ASSERT_EQ(s.solve(fk, ctrl, info, x.data(), static_cast<int>(A.n), 1, sws), 0);
        EXPECT_LT(smf::bench::validate(A, b, x, 1).eta_inf, 1e-12) << threads << " threads";
    }
    EXPECT_EQ(started, 3 + 7 + 3);
}

// Two workspaces factoring at once, each with its own pool, do not interfere.
TEST(TreeParallel, TwoWorkspacesConcurrently) {
    const CscLower A = laplacian_2d(120);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.num_threads = 4;
    ctrl.auto_threads = false;  // small test trees
    std::atomic<int> bad{0};
    const auto body = [&] {
        Solver s;
        Info info;
        auto ak = s.analyse(A, ctrl, info);
        if (!ak) {
            ++bad;
            return;
        }
        NumericWorkspace ws;
        SolveWorkspace sws;
        FactorKeep fk;
        for (int r = 0; r < 5; ++r) {
            if (s.factor(*ak, A, ctrl, info, fk, ws) != FactorStatus::Success) {
                ++bad;
                return;
            }
            std::vector<double> x(static_cast<std::size_t>(A.n));
            for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
            const std::vector<double> b = x;
            if (s.solve(fk, ctrl, info, x.data(), static_cast<int>(A.n), 1, sws) != 0 ||
                !(smf::bench::validate(A, b, x, 1).eta_inf < 1e-12))
                ++bad;
        }
    };
    std::thread t1(body), t2(body);
    t1.join();
    t2.join();
    EXPECT_EQ(bad.load(), 0);
}

// Control::thread_plan_front_costs prices the thread plan in time, but the
// delay budget's regions (which decide where fronts static-pivot) keep the
// flops-only view and the 2e7 floor: the factorization is the same either way.
TEST(TreeParallel, DelayBudgetRegionsUnchangedByFrontCosts) {
    const CscLower A = cascade_kkt(40, 600, 4.1, 3.0);
    Control base;
    base.delay_budget = 20;
    base.num_threads = 4;
    base.auto_threads = false;  // same thread count both ways: only the regions could differ
    Solver solver;
    std::vector<std::vector<double>> xs;
    std::vector<Info> infos;
    for (const bool costs : {false, true}) {
        Control ctrl = base;
        ctrl.thread_plan_front_costs = costs;
        ctrl.thread_cost_per_front = 1e5;
        ctrl.thread_cost_per_front_entry = 10.0;
        Info info;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace nws;
        SolveWorkspace sws;
        FactorKeep fk;
        const FactorStatus st = solver.factor(*ak, A, ctrl, info, fk, nws);
        ASSERT_TRUE(st == FactorStatus::Success || st == FactorStatus::Singular);
        std::vector<double> x(static_cast<std::size_t>(A.n));
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
        EXPECT_EQ(solver.solve(fk, ctrl, info, x.data(), static_cast<int>(A.n), 1, sws), 0);
        xs.push_back(x);
        infos.push_back(info);
    }
    EXPECT_EQ(infos[0].delayed_pivots, infos[1].delayed_pivots);
    EXPECT_EQ(infos[0].static_pivots, infos[1].static_pivots);
    EXPECT_EQ(infos[0].num_negative, infos[1].num_negative);
    EXPECT_EQ(infos[0].num_positive, infos[1].num_positive);
    EXPECT_EQ(xs[0], xs[1]);
}

namespace {
// A planned (auto_threads) factorization sequence on one workspace with front
// costs on: V0 delays little, V1 cascades (cascade_kkt's spread) on the same
// pattern, so V1's realized fronts are much larger than the analysis says.
struct ReplanFixture {
    CscLower V0 = cascade_kkt(90, 300, 2.5, 0.0);
    CscLower V1 = cascade_kkt(90, 300, 2.5, 1.5);
    Control ctrl;
    Solver solver;
    std::unique_ptr<AnalysisKeep> ak;
    NumericWorkspace nws;
    FactorKeep fk;
    ReplanFixture() {
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.scaling = ScalingMethod::None;
        ctrl.scale_on_delay_ratio = -1.0;
        ctrl.num_threads = 8;
        ctrl.auto_threads = true;
        ctrl.thread_plan_front_costs = true;
        ctrl.thread_cost_per_front = 1e5;
        ctrl.thread_cost_per_front_entry = 10.0;
        Info info;
        ak = solver.analyse(V1, ctrl, info);
    }
    Info factor(const CscLower &A, Control c) {
        Info info;
        solver.factor(*ak, A, c, info, fk, nws);
        return info;
    }
    Info factor(const CscLower &A) { return factor(A, ctrl); }
};
}  // namespace

TEST(TreeParallel, ReplansWhenDelaysGrowFronts) {
    ReplanFixture fx;
    ASSERT_NE(fx.ak.get(), nullptr);
    ASSERT_EQ(fx.V0.values.size(), fx.V1.values.size());
    const Info a = fx.factor(fx.V0);
    EXPECT_EQ(a.plan_from_realized, 0);
    const Info b = fx.factor(fx.V1);  // planned before V1's sizes were known
    EXPECT_EQ(b.plan_from_realized, 0);
    ASSERT_GT(b.delayed_pivots, 10 * a.delayed_pivots + 100) << "premise: V1 cascades, V0 does not";
    EXPECT_EQ(fx.factor(fx.V1).plan_from_realized, 1);
    EXPECT_EQ(fx.factor(fx.V1).plan_from_realized, 0);  // realized total now within 10%
}

TEST(TreeParallel, PlanCacheSurvivesAbort) {
    ReplanFixture fx;
    ASSERT_NE(fx.ak.get(), nullptr);
    const Info first = fx.factor(fx.V0);
    Control limited = fx.ctrl;
    limited.max_negative_pivots = 1;
    const Info aborted = fx.factor(fx.V1, limited);
    ASSERT_EQ(aborted.factor_status, FactorStatus::InertiaExceeded);
    const Info again = fx.factor(fx.V0);
    EXPECT_EQ(again.plan_from_realized, 0);
    EXPECT_EQ(again.threads_used, first.threads_used);
}
