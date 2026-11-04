// test_parallel_determinism.cpp — M6.G1 acceptance tests
//
// Decision Log (BLAS non-determinism):
//   D-DET-001 (superseded by Task 14): bitwise identity across thread counts
//   was judged impossible and the multi-thread test used a residual
//   tolerance. Deterministic mode (determinism.hpp) now gives it: one BLAS
//   thread per call, tiled fronts whatever the thread count, a thread-count
//   independent ordering, a postorder factor layout and a serial postorder
//   solve, MKL CNR. BitwiseAcrossThreadCounts checks it bit for bit.

#include "smf/blas_thread_guard.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/determinism.hpp"
#include "smf/ipm_session.hpp"
#include "smf/ipopt_plugin_control.hpp"
#include "smf/matrix_market.hpp"
#include "smf/multifrontal_serial.hpp"
#include "smf/solve_sparse_fwd.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <gtest/gtest.h>
#include <string>
#include <span>
#include <vector>

using namespace smf;

// ---------------------------------------------------------------------------
// Matrix builders
// ---------------------------------------------------------------------------

/// 1-D Poisson FD matrix of size n (SPD tridiagonal: diag=2, off=-1).
static CscLower make_poisson_spd(int n) {
    CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        int cnt = 1; // diagonal
        if (j + 1 < n) ++cnt; // sub-diagonal
        A.col_ptr[static_cast<std::size_t>(j) + 1] = cnt;
    }
    for (int j = 0; j < n; ++j)
        A.col_ptr[static_cast<std::size_t>(j) + 1] +=
            A.col_ptr[static_cast<std::size_t>(j)];
    int nnz = A.col_ptr[static_cast<std::size_t>(n)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));
    for (int j = 0; j < n; ++j) {
        int pos = A.col_ptr[static_cast<std::size_t>(j)];
        A.row_idx[static_cast<std::size_t>(pos)] = j;
        A.values[static_cast<std::size_t>(pos)] = 2.0;
        ++pos;
        if (j + 1 < n) {
            A.row_idx[static_cast<std::size_t>(pos)] = j + 1;
            A.values[static_cast<std::size_t>(pos)] = -1.0;
        }
    }
    return A;
}

/// Simple indefinite KKT-like matrix: block [[H, A^T],[A, 0]] stored lower.
/// Use a 2x2 saddle point: H = diag(1,2,...,p), A = ones(q,p).
/// Build as lower-CSC of the full (p+q)x(p+q) symmetric matrix.
static CscLower make_kkt(int p, int q) {
    // Full n = p + q
    // H block: lower triangle columns 0..p-1
    // A^T block: rows p..p+q-1, cols 0..p-1
    // Zero block: cols p..p+q-1 (no entries in lower triangle)
    // We add a small diagonal perturbation to the zero block for stability.
    int n = p + q;
    // Count nnz per column
    std::vector<int> cnt(static_cast<std::size_t>(n), 0);
    for (int j = 0; j < p; ++j) {
        cnt[static_cast<std::size_t>(j)] = 1 + q; // diagonal + A entries
    }
    for (int j = p; j < n; ++j) {
        cnt[static_cast<std::size_t>(j)] = 1; // small negative diagonal
    }
    CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j)
        A.col_ptr[static_cast<std::size_t>(j) + 1] =
            A.col_ptr[static_cast<std::size_t>(j)] + cnt[static_cast<std::size_t>(j)];
    int nnz = A.col_ptr[static_cast<std::size_t>(n)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));
    for (int j = 0; j < p; ++j) {
        int pos = A.col_ptr[static_cast<std::size_t>(j)];
        // diagonal H[j,j]
        A.row_idx[static_cast<std::size_t>(pos)] = j;
        A.values[static_cast<std::size_t>(pos)] = static_cast<double>(j + 1);
        ++pos;
        // A block entries: rows p..p+q-1 all = 1.0
        for (int i = p; i < n; ++i, ++pos) {
            A.row_idx[static_cast<std::size_t>(pos)] = i;
            A.values[static_cast<std::size_t>(pos)] = 1.0;
        }
    }
    // Zero block diagonal: small negative diagonal for indefiniteness
    for (int j = p; j < n; ++j) {
        int pos = A.col_ptr[static_cast<std::size_t>(j)];
        A.row_idx[static_cast<std::size_t>(pos)] = j;
        A.values[static_cast<std::size_t>(pos)] = -1e-4; // small negative
    }
    return A;
}

// ---------------------------------------------------------------------------
// Residual
// ---------------------------------------------------------------------------
static double compute_residual(const CscLower& A, const std::vector<double>& b,
                                const std::vector<double>& x) {
    int n = A.n;
    std::vector<double> r(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        for (int k = A.col_ptr[static_cast<std::size_t>(j)];
             k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            int i = A.row_idx[static_cast<std::size_t>(k)];
            double v = A.values[static_cast<std::size_t>(k)];
            r[static_cast<std::size_t>(i)] += v * x[static_cast<std::size_t>(j)];
            if (i != j)
                r[static_cast<std::size_t>(j)] += v * x[static_cast<std::size_t>(i)];
        }
    }
    double nr = 0.0, nb = 0.0;
    for (int i = 0; i < n; ++i) {
        nr = std::max(nr, std::abs(r[static_cast<std::size_t>(i)] -
                                   b[static_cast<std::size_t>(i)]));
        nb = std::max(nb, std::abs(b[static_cast<std::size_t>(i)]));
    }
    return nr / std::max(1.0, nb);
}

// ---------------------------------------------------------------------------
// Test 1: Serial runs produce bitwise identical solutions
// ---------------------------------------------------------------------------
TEST(ParallelDeterminism, SerialReproducible) {
    const int n = 30;
    CscLower A = make_poisson_spd(n);

    std::vector<double> b_ref(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        b_ref[static_cast<std::size_t>(i)] = 1.0 + 0.1 * (i % 7);

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.num_threads = 1;
    ctrl.auto_threads = false;  // small test trees

    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    // Run 5 times and collect solutions
    std::vector<std::vector<double>> solutions;
    for (int run = 0; run < 5; ++run) {
        std::vector<double> rhs = b_ref;
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
        ASSERT_EQ(solver.solve(fk, ctrl, info, rhs.data(), n), 0);
        solutions.push_back(rhs);
    }

    // All solutions must be bitwise identical to run 0
    for (int run = 1; run < 5; ++run) {
        for (int i = 0; i < n; ++i) {
            uint64_t bits0, bitsr;
            std::memcpy(&bits0, &solutions[0][static_cast<std::size_t>(i)], 8);
            std::memcpy(&bitsr, &solutions[static_cast<std::size_t>(run)]
                                           [static_cast<std::size_t>(i)], 8);
            EXPECT_EQ(bits0, bitsr)
                << "Run " << run << " differs at index " << i;
        }
    }
    double res = compute_residual(A, b_ref, solutions[0]);
    EXPECT_LT(res, 1e-10) << "Residual: " << res;
    std::printf("[SerialReproducible] residual=%.2e  (5 runs bitwise identical)\n", res);
}

// ---------------------------------------------------------------------------
// Test 2: deterministic=true matches deterministic=false residual quality
// ---------------------------------------------------------------------------
TEST(ParallelDeterminism, DeterministicMode) {
    const int n = 30;
    CscLower A = make_poisson_spd(n);

    std::vector<double> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        b[static_cast<std::size_t>(i)] = static_cast<double>(i + 1);
    const std::vector<double> b_orig = b;

    Solver solver;

    // deterministic=true
    {
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
        ctrl.num_threads = 1;
        ctrl.auto_threads = false;  // small test trees
        ctrl.deterministic = true;
        Info info;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
        std::vector<double> rhs = b_orig;
        ASSERT_EQ(solver.solve(fk, ctrl, info, rhs.data(), n), 0);
        double res = compute_residual(A, b_orig, rhs);
        EXPECT_LT(res, 1e-10) << "deterministic=true residual: " << res;
        std::printf("[DeterministicMode] deterministic=true  residual=%.2e\n", res);
    }

    // deterministic=false
    {
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
        ctrl.num_threads = 1;
        ctrl.auto_threads = false;  // small test trees
        ctrl.deterministic = false;
        Info info;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
        std::vector<double> rhs = b_orig;
        ASSERT_EQ(solver.solve(fk, ctrl, info, rhs.data(), n), 0);
        double res = compute_residual(A, b_orig, rhs);
        EXPECT_LT(res, 1e-10) << "deterministic=false residual: " << res;
        std::printf("[DeterministicMode] deterministic=false residual=%.2e\n", res);
    }
}

// ---------------------------------------------------------------------------
// Test 3: Multi-thread correctness (residual test; bitwise not guaranteed)
//   Decision D-DET-001: BLAS may reorder FMAs per thread count → not bitwise
//   identical across thread counts. Residual tolerance used instead.
// ---------------------------------------------------------------------------
TEST(ParallelDeterminism, MultiThread_Residual) {
    const int n = 100;
    CscLower A = make_poisson_spd(n);

    std::vector<double> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        b[static_cast<std::size_t>(i)] = 1.0 + 0.01 * i;
    const std::vector<double> b_orig = b;

    Solver solver;
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.num_threads = 4;
    ctrl.auto_threads = false;  // small test trees
    ctrl.deterministic = true;

    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);

    std::vector<double> rhs = b_orig;
    ASSERT_EQ(solver.solve(fk, ctrl, info, rhs.data(), n), 0);

    double res = compute_residual(A, b_orig, rhs);
    EXPECT_LT(res, 1e-10) << "4-thread deterministic residual: " << res;
    std::printf("[MultiThread_Residual] 4T deterministic residual=%.2e\n", res);
    // Note: bitwise identity across thread counts not asserted — see D-DET-001
}

// ---------------------------------------------------------------------------
// Test 4: Unit test for sort_children_deterministic
// ---------------------------------------------------------------------------
TEST(ParallelDeterminism, SortChildrenDeterministic) {
    std::vector<int> children = {7, 2, 5, 1, 9, 3};
    sort_children_deterministic(children);
    EXPECT_EQ(children, (std::vector<int>{1, 2, 3, 5, 7, 9}));

    std::vector<int> empty;
    sort_children_deterministic(empty);
    EXPECT_TRUE(empty.empty());

    std::vector<int> single = {42};
    sort_children_deterministic(single);
    EXPECT_EQ(single[0], 42);

    // Already sorted
    std::vector<int> sorted = {0, 1, 2, 3};
    sort_children_deterministic(sorted);
    EXPECT_EQ(sorted, (std::vector<int>{0, 1, 2, 3}));
}

// ---------------------------------------------------------------------------
// Test 5: deterministic_reduce correctness
// ---------------------------------------------------------------------------
TEST(ParallelDeterminism, DeterministicReduceCorrect) {
    std::vector<std::vector<double>> contribs = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {0.1, 0.2, 0.3}
    };
    std::vector<double> acc;
    deterministic_reduce(acc, contribs, 3);
    ASSERT_EQ(acc.size(), 3u);
    EXPECT_DOUBLE_EQ(acc[0], 5.1);
    EXPECT_DOUBLE_EQ(acc[1], 7.2);
    EXPECT_DOUBLE_EQ(acc[2], 9.3);
}

// ---------------------------------------------------------------------------
// Test 6: deterministic mode (Control::deterministic / SMF_DETERMINISTIC):
// bitwise-identical solutions, D entries and inertia across runs and across
// 1/2/4/8/16 threads (analyse, factor and solve each run).
// ---------------------------------------------------------------------------
namespace {

namespace fs = std::filesystem;

fs::path fixture_root() {
    return fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" / "fixtures";
}

// 7-point Laplacian on an m^3 grid minus `shift` on the diagonal, lower CSC
// (shift 0: SPD Poisson; shift 0.9: indefinite Helmholtz).
CscLower grid_3d(int m, double shift) {
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
                if (x + 1 < m) { A.row_idx.push_back(j + 1); A.values.push_back(-1.0); }
                if (y + 1 < m) { A.row_idx.push_back(j + m); A.values.push_back(-1.0); }
                if (z + 1 < m) { A.row_idx.push_back(j + m * m); A.values.push_back(-1.0); }
                A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
            }
    return A;
}

struct DetCase {
    std::string name;
    CscLower A;
    MatrixType type;
    bool expect_parallel;  // 8 threads take the tree-parallel path
    bool expect_tiles;     // 8 threads factor top fronts as tile tasks
    OrderingMethod ordering = OrderingMethod::AutoSerial;
};

struct DetRun {
    std::vector<double> x;
    std::vector<double> xm;  // nrhs = 5 blocked solve (Task 16)
    std::vector<double> d;  // D entries (indefinite), elimination order
    int pos = 0, neg = 0, zero = 0;
    int subtrees = 0, tiles = 0;
    long long arena_blocks = 0;  // subtree roots' blocks put in the root arenas
    int ordering_used = 0;
    bool banded = false;  // the banded SPD fast path factored it
};

// caller_blas > 0: the BLAS threads of the caller of the nrhs > 1 solve
// (default: `threads`). blocktime_ms: Control::mkl_blocktime_ms.
DetRun det_run(const DetCase &c, int threads, int caller_blas = 0, int blocktime_ms = -1) {
    Control ctrl;
    ctrl.matrix_type = c.type;
    ctrl.num_threads = threads;
    ctrl.auto_threads = false;  // small test trees
    ctrl.deterministic = true;
    ctrl.mkl_blocktime_ms = blocktime_ms;
    ctrl.ordering = c.ordering;
    // Cached L11^-1 (built at factor time) is part of what must reproduce.
    ctrl.cache_inverse_diag = true;
    // AutoParallel tries METIS only when AMD's flops times the refactors
    // are large enough per row.
    if (c.ordering == OrderingMethod::AutoParallel) ctrl.expected_refactors = 100;
    Solver solver;
    Info info;
    DetRun r;
    auto ak = solver.analyse(c.A, ctrl, info);
    EXPECT_NE(ak.get(), nullptr);
    if (!ak) return r;
    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;
    const long long arena0 = root_arena_blocks_taken();
    EXPECT_EQ(solver.factor(*ak, c.A, ctrl, info, fk, nws), FactorStatus::Success);
    r.arena_blocks = root_arena_blocks_taken() - arena0;
    const int n = static_cast<int>(c.A.n);
    r.x.resize(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < r.x.size(); ++i)
        r.x[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
    EXPECT_EQ(solver.solve(fk, ctrl, info, r.x.data(), n, 1, sws), 0);
    // nrhs > 1: the serial blocked solve (GEMM / TRSM per front). The caller's
    // BLAS runs `threads` threads, as a threaded application's would.
    {
        constexpr int kNrhs = 5;
        r.xm.resize(static_cast<std::size_t>(n) * kNrhs);
        for (std::size_t i = 0; i < r.xm.size(); ++i)
            r.xm[i] = std::cos(0.007 * static_cast<double>(i)) + 0.25;
        BlasThreadGuard blas(caller_blas > 0 ? caller_blas : threads);
        EXPECT_EQ(solver.solve(fk, ctrl, info, std::span<double>(r.xm), kNrhs, n, sws), 0);
    }
    for (const DiagSolveEntry &e : fk.diag_entries) {
        r.d.push_back(e.d00);
        r.d.push_back(e.d10);
        r.d.push_back(e.d11);
    }
    r.pos = info.num_positive;
    r.neg = info.num_negative;
    r.zero = info.num_zero;
    r.subtrees = info.parallel_subtrees;
    r.tiles = info.top_tile_fronts;
    r.ordering_used = info.ordering_used;
    r.banded = fk.use_banded_spd;
    return r;
}

// Entries whose bit patterns differ, and the largest difference among them.
std::pair<std::size_t, double> bit_diff(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size()) return {std::max(a.size(), b.size()), INFINITY};
    std::size_t cnt = 0;
    double mx = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) {
            ++cnt;
            mx = std::max(mx, std::abs(a[i] - b[i]));
        }
    return {cnt, mx};
}

} // namespace

TEST(ParallelDeterminism, BitwiseAcrossThreadCounts) {
    std::vector<DetCase> cases;
    {
        const fs::path mtx = fixture_root() / "optim20" / "kkt_pde_opt.mtx";
        auto mm = read_matrix_market(mtx.string());
        ASSERT_TRUE(mm.ok) << mtx.string() << ": " << mm.error_message;
        cases.push_back({"kkt_pde_opt", std::move(mm.matrix), MatrixType::RealSymmetricIndefinite,
                         false, false});
    }
    cases.push_back({"poisson_3d_26", grid_3d(26, 0.0), MatrixType::RealSymmetricPositiveDefinite,
                     true, false});
    // AutoParallel: its METIS candidate (kept on a 3D grid) is serial METIS
    // at 1 thread and parallel ND above, except in deterministic mode.
    cases.push_back({"poisson_3d_26/AutoParallel", grid_3d(26, 0.0),
                     MatrixType::RealSymmetricPositiveDefinite, true, false,
                     OrderingMethod::AutoParallel});
    {
        // The Task 13 Helmholtz: helm3d01 when the fixture is present.
        const fs::path mtx = fixture_root() / "large" / "helm3d01" / "helm3d01.mtx";
        std::error_code ec;
        auto mm = fs::exists(mtx, ec) ? read_matrix_market(mtx.string()) : MatrixMarketResult{};
        if (mm.ok)
            cases.push_back({"helm3d01", std::move(mm.matrix), MatrixType::RealSymmetricIndefinite,
                             true, true});
        else
            cases.push_back({"helmholtz_3d_30", grid_3d(30, 0.9),
                             MatrixType::RealSymmetricIndefinite, true, true});
    }

    constexpr int kRuns = 10;
    for (const DetCase &c : cases) {
        SCOPED_TRACE(c.name);
        const DetRun ref = det_run(c, 1);
        ASSERT_FALSE(ref.x.empty());
        if (c.ordering == OrderingMethod::AutoParallel)
            EXPECT_EQ(ref.ordering_used, static_cast<int>(OrderingMethod::METIS))
                << "METIS candidate not kept: the ordering path is not exercised";
        int mismatches = 0;
        for (const int threads : {1, 2, 4, 8, 16}) {
            for (int run = 0; run < kRuns; ++run) {
                const DetRun r = det_run(c, threads);
                if (threads == 8 && run == 0) {
                    if (c.expect_parallel) {
                        EXPECT_GE(r.subtrees, 2) << "8 threads: parallel path not taken";
                        // The root-arena path (Task 6b3) is part of what
                        // must reproduce bitwise.
                        EXPECT_GT(r.arena_blocks, 0) << "8 threads: no root block in the arena";
                    }
                    if (c.expect_tiles)
                        EXPECT_GT(r.tiles, 0) << "8 threads: no tile-task front";
                }
                EXPECT_EQ(r.ordering_used, ref.ordering_used) << threads << " threads, run " << run;
                EXPECT_EQ(r.pos, ref.pos) << threads << " threads, run " << run;
                EXPECT_EQ(r.neg, ref.neg) << threads << " threads, run " << run;
                EXPECT_EQ(r.zero, ref.zero) << threads << " threads, run " << run;
                const auto [dx, mx] = bit_diff(r.x, ref.x);
                const auto [dd, md] = bit_diff(r.d, ref.d);
                const auto [dm, mm] = bit_diff(r.xm, ref.xm);
                if (dm != 0)
                    ADD_FAILURE() << c.name << ": " << threads << " threads, run " << run << ": "
                                  << dm << " of " << r.xm.size()
                                  << " nrhs = 5 solution entries differ (max " << mm << ")";
                if (dx != 0 || dd != 0) {
                    ++mismatches;
                    ADD_FAILURE() << c.name << ": " << threads << " threads, run " << run << ": "
                                  << dx << " of " << r.x.size() << " solution entries differ (max "
                                  << mx << "), " << dd << " of " << r.d.size()
                                  << " D values differ (max " << md << ")";
                }
            }
        }
        std::printf("[BitwiseAcrossThreadCounts] %s n=%d: %d of %d runs differ from 1-thread run 0\n",
                    c.name.c_str(), static_cast<int>(c.A.n), mismatches, 5 * kRuns);
    }
#if defined(SMF_USE_MKL) && SMF_USE_MKL
    // MKL's conditional numerical reproducibility took effect.
    EXPECT_EQ(blas_reproducibility_status(), 0)
        << "mkl_cbwr_set returned " << blas_reproducibility_status();
    std::printf("[BitwiseAcrossThreadCounts] MKL CNR branch %d (10 = AVX2, 2 = AUTO)\n",
                blas_reproducibility_branch());
#endif
}

// A pool thread that wakes late (by a scheduler quantum on a loaded
// machine): the subtree task dealt to it (the heaviest tasks go one per
// thread, final review I-2) is taken over by a thread that ran out of
// tasks, so the factorization does not wait for it, and the result is the
// 1-thread one bit for bit whichever thread factored the task.
TEST(ParallelDeterminism, LateThreadsDealtTaskIsTakenOver) {
    struct LateOff {
        ~LateOff() { set_late_phase1_thread_for_test(-1); }
    } late_off;
    std::vector<DetCase> cases;
    cases.push_back({"poisson_3d_26", grid_3d(26, 0.0), MatrixType::RealSymmetricPositiveDefinite,
                     true, false});
    cases.push_back({"helmholtz_3d_30", grid_3d(30, 0.9), MatrixType::RealSymmetricIndefinite,
                     true, true});
    for (const DetCase &c : cases) {
        SCOPED_TRACE(c.name);
        set_late_phase1_thread_for_test(-1);
        const DetRun ref = det_run(c, 1);
        ASSERT_FALSE(ref.x.empty());
        // No thread held back: how often a dealt task changes thread
        // (informational; the deal is meant to hold).
        const long long idle0 = dealt_tasks_taken_over();
        constexpr int kIdleRuns = 10;
        for (int run = 0; run < kIdleRuns; ++run) (void)det_run(c, 8);
        std::printf("[LateThreadsDealtTaskIsTakenOver] %s: %lld dealt tasks taken over in %d "
                    "8-thread factorizations, no thread held back\n",
                    c.name.c_str(), dealt_tasks_taken_over() - idle0, kIdleRuns);
        for (const int late : {1, 7}) {
            set_late_phase1_thread_for_test(late);
            const long long over0 = dealt_tasks_taken_over();
            const auto t0 = std::chrono::steady_clock::now();
            const DetRun r = det_run(c, 8);
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            set_late_phase1_thread_for_test(-1);
            ASSERT_GE(r.subtrees, 8) << "premise: a task dealt to every thread";
            // The late thread waits until its task is taken (10 s at most).
            EXPECT_GE(dealt_tasks_taken_over() - over0, 1) << "late thread " << late;
            EXPECT_LT(secs, 5.0) << "late thread " << late << ": its task waited for it";
            EXPECT_EQ(r.pos, ref.pos);
            EXPECT_EQ(r.neg, ref.neg);
            EXPECT_EQ(r.zero, ref.zero);
            EXPECT_EQ(bit_diff(r.x, ref.x).first, 0u) << "late thread " << late;
            EXPECT_EQ(bit_diff(r.d, ref.d).first, 0u) << "late thread " << late;
            EXPECT_EQ(bit_diff(r.xm, ref.xm).first, 0u) << "late thread " << late;
        }
    }
}

// A caller whose BLAS runs several threads (a threaded-MKL build's default:
// MKL's global count, every core) gets, in deterministic mode, the bits of
// a caller on serial BLAS, at num_threads = 1 too (bug: the 1-thread
// factorization ran its fronts on the caller's MKL threads, so every
// multi-thread result differed from it), and the caller's BLAS setting is
// left as it was. Only a BLAS that honours a thread count can tell: in a
// sequential-MKL build the guard below changes nothing and the test checks
// only reproducibility (SMF_EXPECT_THREADED_BLAS: the build links threaded
// MKL, so the test must exercise it).
TEST(ParallelDeterminism, CallerBlasThreadsDoNotChangeBits) {
    constexpr int kCallerBlas = 4;
    bool threaded = false;
    {
        BlasThreadGuard g(kCallerBlas);
        threaded = blas_max_threads() == kCallerBlas;
    }
#if defined(SMF_EXPECT_THREADED_BLAS)
    ASSERT_TRUE(threaded) << "threaded-MKL build, but the BLAS ignores a thread count";
#endif
    std::printf("[CallerBlasThreadsDoNotChangeBits] caller BLAS threads %s\n",
                threaded ? "honoured (exercised)" : "ignored (sequential BLAS: not exercised)");

    // Banded SPD (half-bandwidth 64, so DPBTRF / DPBTRS take their blocked
    // BLAS-3 path): the banded fast path of AutoParallel.
    CscLower band;
    {
        const int n = 20000;
        constexpr int kd = 64;
        band.n = n;
        band.col_ptr.push_back(0);
        for (int j = 0; j < n; ++j) {
            band.row_idx.push_back(j);
            band.values.push_back(2.0 * kd + 2.0);
            for (int k = 1; k <= kd && j + k < n; ++k) {
                band.row_idx.push_back(j + k);
                band.values.push_back(-1.0);
            }
            band.col_ptr.push_back(static_cast<Int>(band.row_idx.size()));
        }
    }
    std::vector<DetCase> cases;
    {
        const fs::path mtx = fixture_root() / "optim20" / "kkt_pde_opt.mtx";
        auto mm = read_matrix_market(mtx.string());
        ASSERT_TRUE(mm.ok) << mtx.string() << ": " << mm.error_message;
        cases.push_back({"kkt_pde_opt", std::move(mm.matrix), MatrixType::RealSymmetricIndefinite,
                         false, false});
    }
    cases.push_back({"poisson_3d_26", grid_3d(26, 0.0), MatrixType::RealSymmetricPositiveDefinite,
                     true, false});
    cases.push_back({"helmholtz_3d_30", grid_3d(30, 0.9), MatrixType::RealSymmetricIndefinite,
                     true, true});
    cases.push_back({"band_spd/AutoParallel", std::move(band),
                     MatrixType::RealSymmetricPositiveDefinite, false, false,
                     OrderingMethod::AutoParallel});
    for (const DetCase &c : cases) {
        SCOPED_TRACE(c.name);
        DetRun ref;
        {
            BlasThreadGuard serial(1);
            ref = det_run(c, 1);
        }
        ASSERT_FALSE(ref.x.empty());
        if (c.ordering == OrderingMethod::AutoParallel)
            ASSERT_TRUE(ref.banded) << "the banded path is not exercised";
        for (const int threads : {1, 8}) {
            DetRun r;
            {
                BlasThreadGuard caller(kCallerBlas);
                r = det_run(c, threads, kCallerBlas);
                EXPECT_EQ(blas_max_threads(), threaded ? kCallerBlas : 1)
                    << "the caller's BLAS thread count was not restored";
            }
            EXPECT_EQ(r.banded, ref.banded) << threads << " threads";
            EXPECT_EQ(r.pos, ref.pos) << threads << " threads";
            EXPECT_EQ(r.neg, ref.neg) << threads << " threads";
            EXPECT_EQ(r.zero, ref.zero) << threads << " threads";
            EXPECT_EQ(bit_diff(r.x, ref.x).first, 0u) << threads << " threads";
            EXPECT_EQ(bit_diff(r.d, ref.d).first, 0u) << threads << " threads";
            EXPECT_EQ(bit_diff(r.xm, ref.xm).first, 0u) << threads << " threads";
        }
        std::printf("[CallerBlasThreadsDoNotChangeBits] %s n=%d%s\n", c.name.c_str(),
                    static_cast<int>(c.A.n), ref.banded ? " (banded path)" : "");
    }
#if defined(SMF_USE_MKL) && SMF_USE_MKL
    EXPECT_EQ(blas_reproducibility_status(), 0)
        << "mkl_cbwr_set returned " << blas_reproducibility_status();
#endif
}

// ---------------------------------------------------------------------------
// num_threads = 1 outside deterministic mode is one BLAS thread too (bug:
// with threaded MKL the 1-thread factor and solve ran their fronts on the
// caller's MKL thread count, by default every core). Every public entry runs
// here under a caller whose BLAS runs several threads: Solver::factor (the
// banded SPD path included), refactorization with new values, the solve at
// nrhs = 1 and nrhs > 1, IpmSession (the plugin's path) and
// solve_sparse_forward. Its bits must be a serial-BLAS caller's, and the
// caller's setting must come back. num_threads <= 0 keeps the caller's BLAS
// (no guard): reported, as evidence the threaded BLAS changes bits at all.
// ---------------------------------------------------------------------------
namespace {

struct OneThreadRun {
    std::vector<double> l;    // factor values: factorization, then refactorization
    std::vector<double> x;    // solutions: nrhs 1 and 5, after each factorization
    std::vector<double> y;    // solve_sparse_forward
    std::vector<double> ipm;  // IpmSession: factor values and solutions
    int neg = 0;
    bool banded = false;
    bool ipm_ok = false;
};

OneThreadRun one_thread_run(const DetCase &c, int num_threads) {
    Control ctrl;
    ctrl.matrix_type = c.type;
    ctrl.num_threads = num_threads;
    ctrl.ordering = c.ordering;
    ctrl.cache_inverse_diag = true;
    if (c.ordering == OrderingMethod::AutoParallel) ctrl.expected_refactors = 100;
    OneThreadRun r;
    Solver solver;
    Info info;
    auto ak = solver.analyse(c.A, ctrl, info);
    EXPECT_NE(ak.get(), nullptr);
    if (!ak) return r;
    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;
    const int n = static_cast<int>(c.A.n);
    std::vector<double> v = c.A.values;
    for (int it = 0; it < 2; ++it) {
        if (it == 0) {
            EXPECT_EQ(solver.factor(*ak, c.A, ctrl, info, fk, nws), FactorStatus::Success);
        } else {
            // Refactorization: new values in the analysed matrix's entry order.
            for (std::size_t p = 0; p < v.size(); ++p)
                v[p] *= 1.0 + 1e-3 * std::sin(0.37 * static_cast<double>(p));
            EXPECT_EQ(solver.factor(*ak, std::span<const double>(v), ctrl, info, fk, nws),
                      FactorStatus::Success);
        }
        r.l.insert(r.l.end(), fk.factor_values.begin(), fk.factor_values.end());
        r.l.insert(r.l.end(), fk.band_factor.begin(), fk.band_factor.end());
        r.neg += info.num_negative;
        r.banded = fk.use_banded_spd;
        constexpr int kNrhs = 5;
        std::vector<double> b(static_cast<std::size_t>(n) * (1 + kNrhs));
        for (std::size_t i = 0; i < b.size(); ++i)
            b[i] = std::sin(0.013 * static_cast<double>(i) + it) + 0.5;
        EXPECT_EQ(solver.solve(fk, ctrl, info, b.data(), n, 1, sws), 0);
        EXPECT_EQ(solver.solve(fk, ctrl, info, std::span<double>(b.data() + n, b.size() - n),
                               kNrhs, n, sws),
                  0);
        r.x.insert(r.x.end(), b.begin(), b.end());
    }
    if (!fk.use_banded_spd) {
        std::vector<double> b(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; i += 97) b[static_cast<std::size_t>(i)] = 1.0 + 0.01 * i;
        std::vector<int> reach;
        solve_sparse_forward(*ak, fk, b, reach, r.y);
    }
    // IpmSession, as the plugin drives it (indefinite only).
    if (c.type == MatrixType::RealSymmetricIndefinite) {
        std::optional<Control> pc =
            ipopt_plugin_control(Ma97Options{}, -1, std::max(1, num_threads));
        EXPECT_TRUE(pc.has_value());
        if (!pc) return r;
        pc->num_threads = num_threads;
        IpmSession s;
        EXPECT_EQ(s.analyse(c.A, {}, *pc, ipopt_plugin_ipm_control()), ErrorCode::Success);
        r.ipm_ok = s.factor(c.A.values) == IpmFactorStatus::Ok;
        EXPECT_TRUE(r.ipm_ok);
        if (!r.ipm_ok) return r;
        const FactorKeep &ipk = s.factor_keep();
        r.ipm.assign(ipk.factor_values.begin(), ipk.factor_values.end());
        std::vector<double> b(static_cast<std::size_t>(n) * 4);
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::cos(0.011 * static_cast<double>(i));
        EXPECT_EQ(s.solve(b.data(), n, 1), 0);
        EXPECT_EQ(s.solve(b.data() + n, n, 3), 0);
        r.ipm.insert(r.ipm.end(), b.begin(), b.end());
    }
    return r;
}

std::vector<DetCase> one_thread_cases() {
    std::vector<DetCase> cases;
    {
        const fs::path mtx = fixture_root() / "optim20" / "kkt_pde_opt.mtx";
        auto mm = read_matrix_market(mtx.string());
        EXPECT_TRUE(mm.ok) << mtx.string() << ": " << mm.error_message;
        if (mm.ok)
            cases.push_back({"kkt_pde_opt", std::move(mm.matrix),
                             MatrixType::RealSymmetricIndefinite, false, false});
    }
    cases.push_back({"poisson_3d_26", grid_3d(26, 0.0), MatrixType::RealSymmetricPositiveDefinite,
                     false, false});
    cases.push_back({"helmholtz_3d_30", grid_3d(30, 0.9), MatrixType::RealSymmetricIndefinite,
                     false, false});
    // Banded SPD (half-bandwidth 64: DPBTRF / DPBTRS take their blocked
    // BLAS-3 path), AutoParallel's banded fast path.
    CscLower band;
    const int n = 20000;
    constexpr int kd = 64;
    band.n = n;
    band.col_ptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        band.row_idx.push_back(j);
        band.values.push_back(2.0 * kd + 2.0);
        for (int k = 1; k <= kd && j + k < n; ++k) {
            band.row_idx.push_back(j + k);
            band.values.push_back(-1.0);
        }
        band.col_ptr.push_back(static_cast<Int>(band.row_idx.size()));
    }
    cases.push_back({"band_spd/AutoParallel", std::move(band),
                     MatrixType::RealSymmetricPositiveDefinite, false, false,
                     OrderingMethod::AutoParallel});
    return cases;
}

bool caller_blas_honoured(int threads) {
    BlasThreadGuard g(threads);
    return blas_max_threads() == threads;
}

} // namespace

TEST(ParallelDeterminism, OneThreadIsOneBlasThread) {
    constexpr int kCallerBlas = 4;
    const bool threaded = caller_blas_honoured(kCallerBlas);
#if defined(SMF_EXPECT_THREADED_BLAS)
    ASSERT_TRUE(threaded) << "threaded-MKL build, but the BLAS ignores a thread count";
#endif
    std::printf("[OneThreadIsOneBlasThread] caller BLAS threads %s\n",
                threaded ? "honoured (exercised)" : "ignored (sequential BLAS: not exercised)");
    for (const DetCase &c : one_thread_cases()) {
        SCOPED_TRACE(c.name);
        OneThreadRun ref;
        {
            BlasThreadGuard serial(1);
            ref = one_thread_run(c, 1);
        }
        ASSERT_FALSE(ref.x.empty());
        if (c.ordering == OrderingMethod::AutoParallel)
            ASSERT_TRUE(ref.banded) << "the banded path is not exercised";
        OneThreadRun r, inherit;
        {
            BlasThreadGuard caller(kCallerBlas);
            r = one_thread_run(c, 1);
            EXPECT_EQ(blas_max_threads(), threaded ? kCallerBlas : 1)
                << "the caller's BLAS thread count was not restored";
            inherit = one_thread_run(c, 0);  // no guard: the caller's threads
            EXPECT_EQ(blas_max_threads(), threaded ? kCallerBlas : 1);
        }
        EXPECT_EQ(r.banded, ref.banded);
        EXPECT_EQ(r.neg, ref.neg);
        const auto [dl, ml] = bit_diff(r.l, ref.l);
        const auto [dx, mx] = bit_diff(r.x, ref.x);
        EXPECT_EQ(dl, 0u) << dl << " of " << ref.l.size() << " factor entries differ (max " << ml
                          << ")";
        EXPECT_EQ(dx, 0u) << dx << " of " << ref.x.size() << " solution entries differ (max "
                          << mx << ")";
        EXPECT_EQ(bit_diff(r.y, ref.y).first, 0u) << "solve_sparse_forward";
        EXPECT_EQ(r.ipm_ok, ref.ipm_ok);
        EXPECT_EQ(bit_diff(r.ipm, ref.ipm).first, 0u) << "IpmSession";
        const std::size_t inherit_diff = bit_diff(inherit.l, ref.l).first +
                                         bit_diff(inherit.x, ref.x).first +
                                         bit_diff(inherit.ipm, ref.ipm).first;
        std::printf("[OneThreadIsOneBlasThread] %s n=%d%s: num_threads = 0 on the caller's %d "
                    "BLAS threads: %zu entries differ from serial BLAS\n",
                    c.name.c_str(), static_cast<int>(c.A.n), ref.banded ? " (banded path)" : "",
                    threaded ? kCallerBlas : 1, inherit_diff);
    }
}

// Threaded MKL's DTRTRI (m >= 512) dropped the calling thread's local MKL
// thread setting, so a 1-thread call went on with every core after it
// (build_inverse_diag), and num_threads = 0 changed the caller's setting.
// smf_dtrtri_lower keeps it.
TEST(ParallelDeterminism, TrtriKeepsLocalBlasSetting) {
    const bool threaded = caller_blas_honoured(4);
#if defined(SMF_EXPECT_THREADED_BLAS)
    ASSERT_TRUE(threaded) << "threaded-MKL build, but the BLAS ignores a thread count";
#endif
    for (const int local : {1, 4}) {
        BlasThreadGuard g(local);
        for (const int n : {64, 512, 1024}) {
            std::vector<double> A(static_cast<std::size_t>(n) * n, 0.0);
            for (int j = 0; j < n; ++j)
                for (int i = j; i < n; ++i)
                    A[static_cast<std::size_t>(j) * n + i] = i == j ? 2.0 : 0.01 / (1 + i - j);
            EXPECT_EQ(smf_dtrtri_lower(A.data(), n, n, false), 0);
            EXPECT_EQ(blas_max_threads(), threaded ? local : 1) << "n = " << n;
        }
    }
}

// num_threads = N: the call sets the caller's BLAS to N for its duration and
// gives the caller its own setting back, from above and from below N.
TEST(ParallelDeterminism, ExplicitThreadsRestoreCallerBlas) {
    const bool threaded = caller_blas_honoured(4);
#if defined(SMF_EXPECT_THREADED_BLAS)
    ASSERT_TRUE(threaded) << "threaded-MKL build, but the BLAS ignores a thread count";
#endif
    for (const DetCase &c : one_thread_cases()) {
        SCOPED_TRACE(c.name);
        for (const int caller : {1, 2, 8}) {
            BlasThreadGuard g(caller);
            const OneThreadRun r = one_thread_run(c, 4);
            EXPECT_FALSE(r.x.empty());
            EXPECT_EQ(blas_max_threads(), threaded ? caller : 1)
                << "caller on " << caller << " BLAS threads, num_threads = 4";
        }
    }
    // The rule itself: N inside the call, deterministic mode 1, <= 0 none.
    Control ctrl;
    EXPECT_EQ(call_blas_threads(ctrl), 1);
    ctrl.num_threads = 3;
    EXPECT_EQ(call_blas_threads(ctrl), deterministic_mode(ctrl) ? 1 : 3);
    {
        BlasThreadGuard g(2);
        {
            const CallBlasGuard call(ctrl);
            EXPECT_EQ(blas_max_threads(), !threaded ? 1 : deterministic_mode(ctrl) ? 1 : 3);
        }
        EXPECT_EQ(blas_max_threads(), threaded ? 2 : 1);
        ctrl.num_threads = 0;
        EXPECT_EQ(call_blas_threads(ctrl), deterministic_mode(ctrl) ? 1 : 0);
        ctrl.num_threads = -1;
        EXPECT_EQ(call_blas_threads(ctrl), deterministic_mode(ctrl) ? 1 : 0);
        {
            const CallBlasGuard call(ctrl);
            EXPECT_EQ(blas_max_threads(), !threaded ? 1 : deterministic_mode(ctrl) ? 1 : 2);
        }
    }
    ctrl.deterministic = true;
    ctrl.num_threads = 8;
    EXPECT_EQ(call_blas_threads(ctrl), 1);
}

// ---------------------------------------------------------------------------
// The IPOPT plugin's Control (ipopt_plugin_control(), which the plugin now
// gives the OpenMP thread count) through the plugin's IpmSession, with
// Control::deterministic on: factors, D, inertia and solutions bit for bit
// identical at 1 and 4 threads. The session runs as in the plugin: deferred
// analysis (pattern only), the plugin's IpmControl, a BlasThreadGuard of
// num_threads around each call, and a refactorization with new values.
// Not covered: deterministic mode orders with parallel ND at every thread
// count, so the non-deterministic switch from serial METIS (1 thread) to
// parallel ND (more) under AutoParallel is outside this test.
// ---------------------------------------------------------------------------
namespace {

struct PluginDetRun {
    std::vector<double> l;  // factor_values of both factorizations
    std::vector<double> d;  // D entries
    std::vector<double> x;  // nrhs = 1 and nrhs = 3 solutions
    int neg = 0, zero = 0;
    int threads_used = 0, subtrees = 0;
};

PluginDetRun plugin_det_run(const CscLower &A, int threads, bool auto_threads) {
    PluginDetRun r;
    const std::optional<Control> base = ipopt_plugin_control(Ma97Options{}, -1, threads);
    EXPECT_TRUE(base.has_value());
    if (!base) return r;
    Control ctrl = *base;
    EXPECT_EQ(ctrl.num_threads, threads);
    ctrl.deterministic = true;
    ctrl.auto_threads = auto_threads;
    IpmSession s;
    {
        const BlasThreadGuard blas(threads);
        EXPECT_EQ(s.analyse(A, {}, ctrl, ipopt_plugin_ipm_control()), ErrorCode::Success);
    }
    const int n = static_cast<int>(A.n);
    std::vector<double> v = A.values;
    for (int it = 0; it < 2; ++it) {
        if (it == 1)
            for (std::size_t p = 0; p < v.size(); ++p)
                v[p] *= 1.0 + 1e-3 * std::sin(0.37 * static_cast<double>(p));
        const BlasThreadGuard blas(threads);
        s.set_control(ctrl);
        EXPECT_EQ(s.factor(v), IpmFactorStatus::Ok) << threads << " threads, factorization " << it;
        const FactorKeep &fk = s.factor_keep();
        // L supernode by supernode: the tree-parallel engine stores the
        // subtrees' blocks per pass (factor_col_ptr[s] is where supernode s
        // starts), so the flat layout depends on the partition, the blocks
        // must not. A block is its front's rows x pivot columns (f x p,
        // column-major), taken from the front's solve step (the realized
        // front, delayed rows included, which the symbolic size lacks);
        // a supernode whose pivots were all delayed has none.
        {
            std::map<LongInt, const SolveStep *> step_at;
            for (const SolveStep &st : fk.solve_steps)
                if (st.p > 0) step_at[st.factor_offset] = &st;
            const std::size_t ns = fk.factor_col_ptr.empty() ? 0 : fk.factor_col_ptr.size() - 1;
            LongInt total = 0;
            for (std::size_t sn = 0; sn < ns; ++sn) {
                const std::size_t p = sn < fk.pivot_types.size() ? fk.pivot_types[sn].size() : 0;
                LongInt len = 0;
                if (p > 0) {
                    const auto found = step_at.find(fk.factor_col_ptr[sn]);
                    if (found == step_at.end()) {
                        ADD_FAILURE() << "supernode " << sn << ": no solve step at its factor offset";
                        continue;
                    }
                    EXPECT_EQ(static_cast<std::size_t>(found->second->p), p) << "supernode " << sn;
                    len = static_cast<LongInt>(found->second->f) * found->second->p;
                }
                const LongInt b = fk.factor_col_ptr[sn];
                r.l.push_back(static_cast<double>(len));
                r.l.insert(r.l.end(), fk.factor_values.begin() + b, fk.factor_values.begin() + b + len);
                total += len;
            }
            EXPECT_EQ(total, static_cast<LongInt>(fk.factor_values.size())) << "blocks do not tile L";
        }
        for (const DiagSolveEntry &e : fk.diag_entries) {
            r.d.push_back(e.d00);
            r.d.push_back(e.d10);
            r.d.push_back(e.d11);
        }
        r.neg += s.info().num_negative;
        r.zero += s.info().num_zero;
        r.threads_used = std::max(r.threads_used, s.info().threads_used);
        r.subtrees = std::max(r.subtrees, s.info().parallel_subtrees);
        std::vector<double> b(static_cast<std::size_t>(n) * 4);
        for (std::size_t i = 0; i < b.size(); ++i)
            b[i] = std::sin(0.013 * static_cast<double>(i) + it) + 0.5;
        EXPECT_EQ(s.solve(b.data(), n, 1), 0);
        EXPECT_EQ(s.solve(b.data() + n, n, 3), 0);
        r.x.insert(r.x.end(), b.begin(), b.end());
    }
    return r;
}

} // namespace

TEST(ParallelDeterminism, IpoptPluginControlBitwiseAcrossThreads) {
    struct Case {
        std::string name;
        CscLower A;
        bool expect_parallel;  // 4 threads, auto_threads off: subtree-parallel engine
    };
    std::vector<Case> cases;
    {
        const fs::path mtx = fixture_root() / "optim20" / "kkt_pde_opt.mtx";
        auto mm = read_matrix_market(mtx.string());
        ASSERT_TRUE(mm.ok) << mtx.string() << ": " << mm.error_message;
        cases.push_back({"kkt_pde_opt", std::move(mm.matrix), false});
    }
    cases.push_back({"helmholtz_3d_30", grid_3d(30, 0.9), true});
    for (const Case &c : cases) {
        for (const bool auto_threads : {true, false}) {
            SCOPED_TRACE(c.name + (auto_threads ? " auto_threads" : " fixed threads"));
            const PluginDetRun ref = plugin_det_run(c.A, 1, auto_threads);
            ASSERT_FALSE(ref.x.empty());
            const PluginDetRun r = plugin_det_run(c.A, 4, auto_threads);
            if (!auto_threads && c.expect_parallel) {
                EXPECT_GE(r.subtrees, 2) << "4 threads: subtree-parallel engine not taken";
                EXPECT_GT(r.threads_used, 1);
            }
            EXPECT_EQ(r.neg, ref.neg);
            EXPECT_EQ(r.zero, ref.zero);
            const auto [dl, ml] = bit_diff(r.l, ref.l);
            const auto [dd, md] = bit_diff(r.d, ref.d);
            const auto [dx, mx] = bit_diff(r.x, ref.x);
            EXPECT_EQ(dl, 0u) << dl << " of " << ref.l.size() << " factor entries differ (max " << ml << ")";
            EXPECT_EQ(dd, 0u) << dd << " of " << ref.d.size() << " D entries differ (max " << md << ")";
            EXPECT_EQ(dx, 0u) << dx << " of " << ref.x.size() << " solution entries differ (max " << mx << ")";
            std::printf("[IpoptPluginControlBitwiseAcrossThreads] %s %s: n=%d, 4 threads used %d "
                        "(subtrees %d), factor %zu entries\n",
                        c.name.c_str(), auto_threads ? "auto" : "fixed", static_cast<int>(c.A.n),
                        r.threads_used, r.subtrees, ref.l.size());
        }
    }
}

// Control::mkl_blocktime_ms never changes a result. With the knob off, on
// (0 and 1 ms) at 1 and 4 threads, the factor, the D values and both solves
// are bit for bit the same, and (with a fake Intel runtime installed, so it
// runs on every build) the pool workers that ran top fronts set the
// blocktime on their own threads and put it back.
namespace {
thread_local int tl_bt = 7;
std::mutex g_bt_mutex;
std::set<std::thread::id> g_bt_setters;       // threads that set the knob's value
int g_bt_knob_sets = 0, g_bt_restores = 0;
void bt_set(int ms) {
    std::lock_guard<std::mutex> lk(g_bt_mutex);
    tl_bt = ms;
    if (ms == 1) { ++g_bt_knob_sets; g_bt_setters.insert(std::this_thread::get_id()); }
    else if (ms == 7) ++g_bt_restores;
}
int bt_get() { return tl_bt; }
} // namespace

TEST(ParallelDeterminism, MklBlocktimeKnobBitwise) {
    std::vector<DetCase> cases;
    cases.push_back({"poisson_3d_26", grid_3d(26, 0.0), MatrixType::RealSymmetricPositiveDefinite,
                     true, false});
    cases.push_back({"helmholtz_3d_30", grid_3d(30, 0.9), MatrixType::RealSymmetricIndefinite,
                     true, true});
    for (const DetCase &c : cases) {
        SCOPED_TRACE(c.name);
        for (const int threads : {1, 4}) {
            const DetRun ref = det_run(c, threads, 0, -1);
            ASSERT_FALSE(ref.x.empty());
            for (const int bt : {0, 1}) {
                const DetRun r = det_run(c, threads, 0, bt);
                EXPECT_EQ(bit_diff(r.x, ref.x).first, 0u) << threads << " threads, blocktime " << bt;
                EXPECT_EQ(bit_diff(r.xm, ref.xm).first, 0u) << threads << " threads, blocktime " << bt;
                EXPECT_EQ(bit_diff(r.d, ref.d).first, 0u) << threads << " threads, blocktime " << bt;
                EXPECT_EQ(r.pos, ref.pos);
                EXPECT_EQ(r.neg, ref.neg);
                EXPECT_EQ(r.zero, ref.zero);
            }
        }
    }

    // The knob reaches the pool workers, and is undone.
    struct HookReset {
        ~HookReset() { mkl_blocktime_reset_hooks_for_test(); }
    } reset;
    tl_bt = 7;
    g_bt_setters.clear();
    g_bt_knob_sets = g_bt_restores = 0;
    mkl_blocktime_set_hooks_for_test(bt_set, bt_get);
    (void)det_run(cases[0], 4, 0, 1);
    EXPECT_GT(g_bt_knob_sets, 0) << "no front applied the blocktime";
    EXPECT_EQ(g_bt_knob_sets, g_bt_restores) << "a blocktime was not restored";
    EXPECT_EQ(tl_bt, 7) << "the caller's blocktime was not restored";
    std::printf("[MklBlocktimeKnobBitwise] %d applications on %zu thread(s), all restored\n",
                g_bt_knob_sets, g_bt_setters.size());
    EXPECT_GE(g_bt_setters.size(), 1u);
}
