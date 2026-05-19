// bench_poisson.cpp — 2D Poisson (5-point stencil) benchmark for smf
// Part of the smf MA97-class solver benchmark suite.
// Compile with: cmake -DSMF_BUILD_BENCHMARKS=ON
//
// Usage: bench_poisson [grid_size=10]
//   Generates an (n*n) x (n*n) SPD matrix from the 5-point stencil on an n×n grid.

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// --------------------------------------------------------------------------
// Utility: peak memory from /proc/self/status (Linux only)
// --------------------------------------------------------------------------
static long read_vm_peak_kb() {
    std::ifstream ifs("/proc/self/status");
    if (!ifs.is_open()) return -1L;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("VmPeak:", 0) == 0) {
            // Format: "VmPeak:   XXXX kB"
            const char *p = line.c_str() + 7; // skip "VmPeak:"
            while (*p == ' ' || *p == '\t') ++p;
            return std::atol(p);
        }
    }
    return -1L;
}

// --------------------------------------------------------------------------
// Utility: SpMV for residual — lower-triangular CSC, full-matrix multiply
//   y = A * x  (A is symmetric, stored as lower triangle)
// --------------------------------------------------------------------------
static void spmv_sym_lower(const smf::CscLower &A, const double *x, double *y) {
    const smf::Int n = A.n;
    for (smf::Int i = 0; i < n; ++i) y[i] = 0.0;
    for (smf::Int j = 0; j < n; ++j) {
        for (smf::Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            smf::Int i = A.row_idx[static_cast<std::size_t>(p)];
            double  v = A.values[static_cast<std::size_t>(p)];
            y[i] += v * x[j]; // lower: (i,j) with i>=j → contributes to row i
            if (i != j) {
                y[j] += v * x[i]; // symmetric upper part
            }
        }
    }
}

// --------------------------------------------------------------------------
// Build lower-CSC for 2D 5-point Poisson on n×n grid.
// Node (i,j) → index i*n+j.  Diagonal = 4, off-diagonal connections = -1.
// Lower triangle only: we store (row, col) with row >= col.
// Connections in lower triangle from node k=(i,j):
//   - left:  node k-1  = (i, j-1) if j > 0   → col = k-1 < k ✓
//   - below: node k-n  = (i-1, j) if i > 0   → col = k-n < k ✓
// --------------------------------------------------------------------------
static smf::CscLower build_poisson_2d(int n) {
    const int N = n * n;
    // Count nnz per column (lower triangle)
    std::vector<smf::Int> col_count(static_cast<std::size_t>(N), 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int k = i * n + j;
            ++col_count[static_cast<std::size_t>(k)]; // diagonal
            if (j > 0) ++col_count[static_cast<std::size_t>(k)];   // left neighbour stored in col k (row=k, col=k-1 → but lower means row>=col, so this is stored in col k-1, row k)
            if (i > 0) ++col_count[static_cast<std::size_t>(k)];   // below neighbour
        }
    }
    // Actually re-count properly.
    // For each node k, lower-triangle entries stored in column k:
    //   (k, k) = 4   (diagonal)
    //   (k+1, k) = -1  if k+1 is right neighbour (j < n-1)
    //   (k+n, k) = -1  if k+n is upper neighbour (i < n-1)
    // Wait — let me think again.
    // The 5-point stencil: node k=(i,j) connects to:
    //   right: (i, j+1) = k+1    if j < n-1
    //   left:  (i, j-1) = k-1    if j > 0
    //   up:    (i+1, j) = k+n    if i < n-1
    //   down:  (i-1, j) = k-n    if i > 0
    // For lower-CSC: entry (row, col) with row >= col.
    // Column k contains: diagonal (k,k) plus entries (row, k) with row > k.
    // row > k means (row > k):
    //   right neighbour k+1: row=k+1 > k ✓  (but only if they're in same row: j<n-1)
    //   up neighbour k+n:    row=k+n > k ✓  (always k+n > k)
    // So column k has:
    //   1 diagonal entry
    //   1 entry if j < n-1 (right neighbour, row=k+1)
    //   1 entry if i < n-1 (up neighbour,   row=k+n)

    std::fill(col_count.begin(), col_count.end(), 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int k = i * n + j;
            col_count[static_cast<std::size_t>(k)] = 1; // diagonal
            if (j < n - 1) ++col_count[static_cast<std::size_t>(k)]; // right
            if (i < n - 1) ++col_count[static_cast<std::size_t>(k)]; // up
        }
    }

    smf::CscLower A;
    A.n = static_cast<smf::Int>(N);
    A.col_ptr.resize(static_cast<std::size_t>(N) + 1);
    A.col_ptr[0] = 0;
    for (int k = 0; k < N; ++k) {
        A.col_ptr[static_cast<std::size_t>(k) + 1] =
            A.col_ptr[static_cast<std::size_t>(k)] + col_count[static_cast<std::size_t>(k)];
    }
    smf::Int nnz = A.col_ptr[static_cast<std::size_t>(N)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    // Fill entries (column-by-column, sorted by row within column)
    std::vector<smf::Int> pos(col_count.begin(), col_count.end());
    // Use col_ptr as base; fill from position col_ptr[k]
    // We iterate column k and add in sorted row order: diagonal first, then right, then up.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int k = i * n + j;
            smf::Int base = A.col_ptr[static_cast<std::size_t>(k)];
            smf::Int off  = 0;
            // diagonal (row == col == k)
            A.row_idx[static_cast<std::size_t>(base + off)] = static_cast<smf::Int>(k);
            A.values[static_cast<std::size_t>(base + off)]  = 4.0;
            ++off;
            // right neighbour: row = k+1 (only if j < n-1, so k+1 is still in same physical row boundary — actually k+1 is always > k)
            if (j < n - 1) {
                A.row_idx[static_cast<std::size_t>(base + off)] = static_cast<smf::Int>(k + 1);
                A.values[static_cast<std::size_t>(base + off)]  = -1.0;
                ++off;
            }
            // up neighbour: row = k+n
            if (i < n - 1) {
                A.row_idx[static_cast<std::size_t>(base + off)] = static_cast<smf::Int>(k + n);
                A.values[static_cast<std::size_t>(base + off)]  = -1.0;
                ++off;
            }
            (void)pos; // unused after refactor
        }
    }
    return A;
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    int n = 10;
    if (argc >= 2) {
        n = std::atoi(argv[1]);
        if (n < 2) { std::fprintf(stderr, "grid_size must be >= 2\n"); return 1; }
    }

    std::printf("=== bench_poisson ===\n");

    // Build matrix
    smf::CscLower A = build_poisson_2d(n);
    const int N = n * n;
    std::printf("Grid: %dx%d, N=%d, nnz=%d (lower triangular)\n",
                n, n, N, static_cast<int>(A.nnz()));

    // RHS: b[k] = 1.0
    std::vector<double> b(static_cast<std::size_t>(N), 1.0);
    std::vector<double> b_orig(b); // keep for residual computation

    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricPositiveDefinite;
    smf::Info info;

    smf::Solver solver;

    // Analyse
    auto t0 = std::chrono::high_resolution_clock::now();
    auto ak = solver.analyse(A, ctrl, info);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!ak) {
        std::fprintf(stderr, "bench_poisson: analyse() failed\n");
        return 1;
    }
    double t_analyse = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Factor
    smf::FactorKeep fkeep;
    auto t2 = std::chrono::high_resolution_clock::now();
    smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fkeep);
    auto t3 = std::chrono::high_resolution_clock::now();
    if (fs != smf::FactorStatus::Success) {
        std::fprintf(stderr, "bench_poisson: factor() failed with status %d\n",
                     static_cast<int>(fs));
        return 1;
    }
    double t_factor = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Solve
    auto t4 = std::chrono::high_resolution_clock::now();
    int rc = solver.solve(fkeep, ctrl, info, b.data(), N, 1);
    auto t5 = std::chrono::high_resolution_clock::now();
    if (rc != 0) {
        std::fprintf(stderr, "bench_poisson: solve() returned %d\n", rc);
        return 1;
    }
    double t_solve = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // Residual: ||A*x - b_orig|| / ||b_orig||
    std::vector<double> ax(static_cast<std::size_t>(N));
    spmv_sym_lower(A, b.data(), ax.data());
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        double r = ax[static_cast<std::size_t>(i)] - b_orig[static_cast<std::size_t>(i)];
        num += r * r;
        den += b_orig[static_cast<std::size_t>(i)] * b_orig[static_cast<std::size_t>(i)];
    }
    double residual = std::sqrt(num) / std::max(1.0, std::sqrt(den));

    // Output
    std::printf("Analyse : %.3f ms\n", t_analyse);
    std::printf("Factor  : %.3f ms\n", t_factor);
    std::printf("Solve   : %.3f ms\n", t_solve);
    std::printf("Residual: %.3e\n", residual);
    std::printf("Inertia : pos=%d neg=%d zero=%d\n",
                info.num_positive, info.num_negative, info.num_zero);

    long vmpeakKb = read_vm_peak_kb();
    if (vmpeakKb >= 0) {
        std::printf("Peak memory: %ld kB\n", vmpeakKb);
    } else {
        std::printf("Peak memory: N/A\n");
    }

    return 0;
}
