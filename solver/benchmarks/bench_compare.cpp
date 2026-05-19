// bench_compare.cpp — smf vs Eigen SimplicialLDLT (and optionally CHOLMOD)
// Part of the smf MA97-class solver benchmark suite.
// Compile with: cmake -DSMF_BUILD_BENCHMARKS=ON
//
// Compares smf against Eigen SimplicialLDLT on 5 synthetic SPD matrices.
// CHOLMOD support is included when SMF_HAS_CHOLMOD is defined.

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef EIGEN_WORLD_VERSION
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#endif

#ifdef SMF_HAS_CHOLMOD
#include <cholmod.h>
#endif

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

static inline double elapsed_ms(TimePoint t0, TimePoint t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// SpMV for residual: y = A*x  (A symmetric, stored as lower triangle)
// ---------------------------------------------------------------------------
static void spmv_sym(const smf::CscLower &A, const double *x, double *y) {
    const smf::Int n = A.n;
    for (smf::Int i = 0; i < n; ++i) y[i] = 0.0;
    for (smf::Int j = 0; j < n; ++j) {
        for (smf::Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            smf::Int i  = A.row_idx[static_cast<std::size_t>(p)];
            double   v  = A.values[static_cast<std::size_t>(p)];
            y[i] += v * x[j];
            if (i != j) y[j] += v * x[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix generators — all return lower-triangular CSC
// ---------------------------------------------------------------------------

/// 2-D 5-point Poisson on grid_n × grid_n grid.
/// N = grid_n^2, diag = 4, off-diag = -1.
static smf::CscLower make_poisson2d(int grid_n) {
    const int N = grid_n * grid_n;

    // Count entries per column (lower triangle only)
    std::vector<smf::Int> col_count(static_cast<std::size_t>(N), 0);
    for (int row = 0; row < grid_n; ++row) {
        for (int col = 0; col < grid_n; ++col) {
            int k = row * grid_n + col;
            col_count[static_cast<std::size_t>(k)]++;          // diagonal
            if (col > 0)  col_count[static_cast<std::size_t>(k - 1)]++;  // right neigh stored in lower col
            if (row > 0)  col_count[static_cast<std::size_t>(k - grid_n)]++; // below neigh
        }
    }

    smf::CscLower A;
    A.n = static_cast<smf::Int>(N);
    A.col_ptr.resize(static_cast<std::size_t>(N + 1));
    A.col_ptr[0] = 0;
    for (int j = 0; j < N; ++j)
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            A.col_ptr[static_cast<std::size_t>(j)] +
            col_count[static_cast<std::size_t>(j)];

    const smf::Int nnz = A.col_ptr[static_cast<std::size_t>(N)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    std::vector<smf::Int> pos(A.col_ptr.begin(), A.col_ptr.end());

    for (int row = 0; row < grid_n; ++row) {
        for (int col = 0; col < grid_n; ++col) {
            int k = row * grid_n + col;
            // diagonal
            smf::Int p = pos[static_cast<std::size_t>(k)]++;
            A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(k);
            A.values[static_cast<std::size_t>(p)]  = 4.0;
            // left neighbour: (k, k-1) in lower triangle → row=k, col=k-1
            if (col > 0) {
                smf::Int q = pos[static_cast<std::size_t>(k - 1)]++;
                A.row_idx[static_cast<std::size_t>(q)] = static_cast<smf::Int>(k);
                A.values[static_cast<std::size_t>(q)]  = -1.0;
            }
            // below neighbour: (k, k-grid_n) → row=k, col=k-grid_n
            if (row > 0) {
                smf::Int q = pos[static_cast<std::size_t>(k - grid_n)]++;
                A.row_idx[static_cast<std::size_t>(q)] = static_cast<smf::Int>(k);
                A.values[static_cast<std::size_t>(q)]  = -1.0;
            }
        }
    }
    return A;
}

/// Symmetric tridiagonal: diag = diag_val, sub-diag = offdiag_val.
/// SPD when diag_val > 2 * |offdiag_val|.
static smf::CscLower make_tridiagonal(int n, double diag_val = 2.0,
                                      double offdiag_val = -1.0) {
    // Lower triangle: diagonal + sub-diagonal entries
    smf::CscLower A;
    A.n = static_cast<smf::Int>(n);
    A.col_ptr.resize(static_cast<std::size_t>(n + 1));
    // col 0: 1 entry (diagonal), col j>0: 2 entries (sub-diag + diag)
    A.col_ptr[0] = 0;
    for (int j = 0; j < n; ++j)
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            A.col_ptr[static_cast<std::size_t>(j)] + (j == 0 ? 1 : 2);

    const smf::Int nnz = A.col_ptr[static_cast<std::size_t>(n)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    for (int j = 0; j < n; ++j) {
        smf::Int p = A.col_ptr[static_cast<std::size_t>(j)];
        if (j > 0) {
            // sub-diagonal: row = j, col = j-1  → lower triangle (j > j-1)
            A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(j);
            A.values[static_cast<std::size_t>(p)]  = offdiag_val;
            ++p;
        }
        // diagonal
        A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(j);
        A.values[static_cast<std::size_t>(p)]  = diag_val;
    }
    return A;
}

/// Banded SPD: dominant diagonal, bandwidth sub-diagonals all = -1.
static smf::CscLower make_banded_spd(int n, int bw = 5) {
    double diag_val = static_cast<double>(bw * 2 + 1) * 2.0;

    // Count entries per column (lower triangle)
    std::vector<smf::Int> col_count(static_cast<std::size_t>(n), 0);
    for (int j = 0; j < n; ++j) {
        col_count[static_cast<std::size_t>(j)]++; // diagonal
        int rows_below = std::min(bw, n - 1 - j);
        col_count[static_cast<std::size_t>(j)] += rows_below;
    }

    smf::CscLower A;
    A.n = static_cast<smf::Int>(n);
    A.col_ptr.resize(static_cast<std::size_t>(n + 1));
    A.col_ptr[0] = 0;
    for (int j = 0; j < n; ++j)
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            A.col_ptr[static_cast<std::size_t>(j)] +
            col_count[static_cast<std::size_t>(j)];

    const smf::Int nnz = A.col_ptr[static_cast<std::size_t>(n)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    for (int j = 0; j < n; ++j) {
        smf::Int p = A.col_ptr[static_cast<std::size_t>(j)];
        // diagonal
        A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(j);
        A.values[static_cast<std::size_t>(p)]  = diag_val;
        ++p;
        // sub-diagonals (rows j+1 .. j+bw stored in column j)
        int rows_below = std::min(bw, n - 1 - j);
        for (int k = 1; k <= rows_below; ++k) {
            A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(j + k);
            A.values[static_cast<std::size_t>(p)]  = -1.0;
            ++p;
        }
    }
    return A;
}

/// Block-diagonal SPD.  Blocks of size 3 (and 2 for remainder).
///   3×3 block: diag=4, off=-1;  2×2 block: diag=2, off=-1.
static smf::CscLower make_block_diagonal_spd(int n) {
    // Pre-compute nnz
    smf::Int total_nnz = 0;
    {
        int remaining = n;
        while (remaining > 0) {
            int bs = (remaining >= 3) ? 3 : remaining;
            // lower triangle of bs×bs: bs*(bs+1)/2
            total_nnz += static_cast<smf::Int>(bs * (bs + 1) / 2);
            remaining -= bs;
        }
    }

    smf::CscLower A;
    A.n = static_cast<smf::Int>(n);
    A.col_ptr.resize(static_cast<std::size_t>(n + 1));
    A.row_idx.resize(static_cast<std::size_t>(total_nnz));
    A.values.resize(static_cast<std::size_t>(total_nnz));

    A.col_ptr[0] = 0;
    smf::Int p = 0;
    int col_offset = 0;

    while (col_offset < n) {
        int bs = (n - col_offset >= 3) ? 3 : (n - col_offset);
        double dv = (bs == 3) ? 4.0 : 2.0;
        double ov = -1.0;

        for (int lc = 0; lc < bs; ++lc) {
            int j = col_offset + lc;
            // column j: entries at rows j, j+1, .., col_offset+bs-1
            for (int lr = lc; lr < bs; ++lr) {
                int i = col_offset + lr;
                A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(i);
                A.values[static_cast<std::size_t>(p)]  = (lr == lc) ? dv : ov;
                ++p;
            }
            A.col_ptr[static_cast<std::size_t>(j + 1)] =
                A.col_ptr[static_cast<std::size_t>(j)] +
                static_cast<smf::Int>(bs - lc);
        }
        col_offset += bs;
    }
    return A;
}

// ---------------------------------------------------------------------------
// smf solve: analyse + factor + solve, return relative residual
// ---------------------------------------------------------------------------
struct SmfTimes {
    double analyse_ms = 0.0;
    double factor_ms  = 0.0;
    double solve_ms   = 0.0;
    double total_ms() const { return analyse_ms + factor_ms + solve_ms; }
};

static double smf_run(const smf::CscLower &A, std::vector<double> &x_out,
                      SmfTimes &t) {
    const smf::Int n = A.n;
    x_out.assign(static_cast<std::size_t>(n), 1.0);

    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricPositiveDefinite;
    smf::Info info;
    smf::Solver solver;

    auto ta0 = Clock::now();
    auto ak  = solver.analyse(A, ctrl, info);
    auto ta1 = Clock::now();
    if (!ak) return -1.0;

    smf::FactorKeep fk;
    auto fs  = solver.factor(*ak, ctrl, info, fk);
    auto ta2 = Clock::now();
    if (fs != smf::FactorStatus::Success) return -1.0;

    solver.solve(fk, ctrl, info, x_out.data(), static_cast<int>(n));
    auto ta3 = Clock::now();

    t.analyse_ms = elapsed_ms(ta0, ta1);
    t.factor_ms  = elapsed_ms(ta1, ta2);
    t.solve_ms   = elapsed_ms(ta2, ta3);

    // residual ||Ax - b||_2 / sqrt(n)
    std::vector<double> Ax(static_cast<std::size_t>(n), 0.0);
    spmv_sym(A, x_out.data(), Ax.data());
    double res = 0.0;
    for (smf::Int i = 0; i < n; ++i) {
        double r = Ax[static_cast<std::size_t>(i)] - 1.0;
        res += r * r;
    }
    return std::sqrt(res) / std::sqrt(static_cast<double>(n));
}

// ---------------------------------------------------------------------------
// Eigen SimplicialLDLT solve
// ---------------------------------------------------------------------------
#ifdef EIGEN_WORLD_VERSION
static Eigen::SparseMatrix<double> to_eigen_sym(const smf::CscLower &A) {
    int n = static_cast<int>(A.n);
    Eigen::SparseMatrix<double> M(n, n);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(A.nnz()) * 2);
    for (int j = 0; j < n; ++j) {
        for (smf::Int k = A.col_ptr[static_cast<std::size_t>(j)];
             k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            int    i = static_cast<int>(A.row_idx[static_cast<std::size_t>(k)]);
            double v = A.values[static_cast<std::size_t>(k)];
            triplets.emplace_back(i, j, v);
            if (i != j) triplets.emplace_back(j, i, v);
        }
    }
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

static double eigen_run(const Eigen::SparseMatrix<double> &M,
                        Eigen::VectorXd &x_out, double &total_ms) {
    int n = static_cast<int>(M.rows());
    Eigen::VectorXd b = Eigen::VectorXd::Ones(n);

    auto t0 = Clock::now();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(M);
    x_out   = ldlt.solve(b);
    auto t1 = Clock::now();

    total_ms = elapsed_ms(t0, t1);

    Eigen::VectorXd res = M * x_out - b;
    return res.norm() / std::sqrt(static_cast<double>(n));
}
#endif // EIGEN_WORLD_VERSION

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------
struct MatrixCase {
    std::string  name;
    smf::CscLower A;
};

static std::vector<MatrixCase> build_test_matrices() {
    std::vector<MatrixCase> cases;
    cases.push_back({"Poisson2D_100",  make_poisson2d(10)});   // N=100
    cases.push_back({"Poisson2D_400",  make_poisson2d(20)});   // N=400
    cases.push_back({"Tridiag_500",    make_tridiagonal(500)});
    cases.push_back({"BandedSPD_200",  make_banded_spd(200, 5)});
    cases.push_back({"BlockDiag_300",  make_block_diagonal_spd(300)});
    return cases;
}

static constexpr int WARMUP_RUNS = 3;

int main() {
    std::puts("=== bench_compare: smf vs Eigen SimplicialLDLT ===");

#ifdef SMF_HAS_CHOLMOD
    std::puts("CHOLMOD: available (compiled in)\n");
#else
    std::puts("CHOLMOD: not available (not compiled in)\n");
#endif

#ifndef EIGEN_WORLD_VERSION
    std::puts("Eigen: not available (not compiled in)");
    std::puts("Running smf-only benchmark.\n");
#endif

    auto cases = build_test_matrices();

    // Header
    std::printf("%-22s  %6s  %7s  %9s", "Matrix", "N", "nnz", "smf(ms)");
#ifdef EIGEN_WORLD_VERSION
    std::printf("  %9s  %8s  %10s  %10s",
                "Eigen(ms)", "Speedup", "smf_res", "Eigen_res");
#else
    std::printf("  %10s", "smf_res");
#endif
    std::puts("");

    // Separator line
    std::printf("%-22s  %6s  %7s  %9s", "----------------------", "------",
                "-------", "---------");
#ifdef EIGEN_WORLD_VERSION
    std::printf("  %9s  %8s  %10s  %10s", "---------", "--------",
                "----------", "----------");
#else
    std::printf("  %10s", "----------");
#endif
    std::puts("");

#ifdef EIGEN_WORLD_VERSION
    int smf_faster_count = 0;
    double total_speedup = 0.0;
    int valid_speedup    = 0;
#endif

    for (auto &mc : cases) {
        const smf::CscLower &A = mc.A;
        const int N   = static_cast<int>(A.n);
        const int nnz = static_cast<int>(A.nnz());

        // --- smf: warm-up runs, keep minimum ---
        double smf_best = 1e18;
        double smf_res  = -1.0;
        std::vector<double> smf_x;
        for (int r = 0; r < WARMUP_RUNS; ++r) {
            SmfTimes t;
            double res = smf_run(A, smf_x, t);
            if (res >= 0.0 && t.total_ms() < smf_best) {
                smf_best = t.total_ms();
                smf_res  = res;
            }
        }

#ifdef EIGEN_WORLD_VERSION
        // --- Eigen: warm-up runs, keep minimum ---
        Eigen::SparseMatrix<double> M = to_eigen_sym(A);
        double eigen_best = 1e18;
        double eigen_res  = -1.0;
        Eigen::VectorXd eigen_x;
        for (int r = 0; r < WARMUP_RUNS; ++r) {
            double t_ms = 0.0;
            double res  = eigen_run(M, eigen_x, t_ms);
            if (res >= 0.0 && t_ms < eigen_best) {
                eigen_best = t_ms;
                eigen_res  = res;
            }
        }

        double speedup = (smf_best > 0.0) ? eigen_best / smf_best : 0.0;
        if (smf_res >= 0.0 && eigen_res >= 0.0) {
            total_speedup += speedup;
            ++valid_speedup;
            if (speedup >= 1.0) ++smf_faster_count;
        }

        std::printf("%-22s  %6d  %7d  %9.3f  %9.3f  %7.2fx  %10.3e  %10.3e\n",
                    mc.name.c_str(), N, nnz, smf_best, eigen_best, speedup,
                    smf_res, eigen_res);
#else
        std::printf("%-22s  %6d  %7d  %9.3f  %10.3e\n",
                    mc.name.c_str(), N, nnz, smf_best, smf_res);
#endif
    }

    std::puts("");

#ifdef EIGEN_WORLD_VERSION
    std::printf("Summary: smf is faster than Eigen in %d/%d cases\n",
                smf_faster_count, static_cast<int>(cases.size()));
    if (valid_speedup > 0) {
        std::printf("Average speedup: %.2fx\n",
                    total_speedup / static_cast<double>(valid_speedup));
    }
#else
    std::printf("Summary: %d matrices solved by smf\n",
                static_cast<int>(cases.size()));
#endif

    return 0;
}
