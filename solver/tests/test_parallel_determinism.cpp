// test_parallel_determinism.cpp — M6.G1 acceptance tests
//
// Decision Log (BLAS non-determinism):
//   D-DET-001 (superseded by Task 14): bitwise identity across thread counts
//   was judged impossible and the multi-thread test used a residual
//   tolerance. Deterministic mode (determinism.hpp) now gives it: one BLAS
//   thread per call, tiled fronts whatever the thread count, a thread-count
//   independent ordering, a postorder factor layout and a serial postorder
//   solve, MKL CNR. BitwiseAcrossThreadCounts checks it bit for bit.

#include "smf/determinism.hpp"
#include "smf/matrix_market.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
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
// 1/2/8/16 threads (analyse, factor and solve each run).
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
    std::vector<double> d;  // D entries (indefinite), elimination order
    int pos = 0, neg = 0, zero = 0;
    int subtrees = 0, tiles = 0;
    int ordering_used = 0;
};

DetRun det_run(const DetCase &c, int threads) {
    Control ctrl;
    ctrl.matrix_type = c.type;
    ctrl.num_threads = threads;
    ctrl.deterministic = true;
    ctrl.ordering = c.ordering;
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
    EXPECT_EQ(solver.factor(*ak, c.A, ctrl, info, fk, nws), FactorStatus::Success);
    const int n = static_cast<int>(c.A.n);
    r.x.resize(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < r.x.size(); ++i)
        r.x[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
    EXPECT_EQ(solver.solve(fk, ctrl, info, r.x.data(), n, 1, sws), 0);
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
        for (const int threads : {1, 2, 8, 16}) {
            for (int run = 0; run < kRuns; ++run) {
                const DetRun r = det_run(c, threads);
                if (threads == 8 && run == 0) {
                    if (c.expect_parallel)
                        EXPECT_GE(r.subtrees, 2) << "8 threads: parallel path not taken";
                    if (c.expect_tiles)
                        EXPECT_GT(r.tiles, 0) << "8 threads: no tile-task front";
                }
                EXPECT_EQ(r.ordering_used, ref.ordering_used) << threads << " threads, run " << run;
                EXPECT_EQ(r.pos, ref.pos) << threads << " threads, run " << run;
                EXPECT_EQ(r.neg, ref.neg) << threads << " threads, run " << run;
                EXPECT_EQ(r.zero, ref.zero) << threads << " threads, run " << run;
                const auto [dx, mx] = bit_diff(r.x, ref.x);
                const auto [dd, md] = bit_diff(r.d, ref.d);
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
                    c.name.c_str(), static_cast<int>(c.A.n), mismatches, 4 * kRuns);
    }
#if defined(SMF_USE_MKL) && SMF_USE_MKL
    // MKL's conditional numerical reproducibility took effect.
    EXPECT_EQ(blas_reproducibility_status(), 0)
        << "mkl_cbwr_set returned " << blas_reproducibility_status();
    std::printf("[BitwiseAcrossThreadCounts] MKL CNR branch %d (10 = AVX2, 2 = AUTO)\n",
                blas_reproducibility_branch());
#endif
}
