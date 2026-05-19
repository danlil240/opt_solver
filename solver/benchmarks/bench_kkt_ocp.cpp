// bench_kkt_ocp.cpp — KKT saddle-point matrix benchmark for smf
// Part of the smf MA97-class solver benchmark suite.
//
// Generates a synthetic KKT / saddle-point matrix:
//   [ H   A^T ]   where H = diag(1..nz), A is banded (nc x nz)
//   [ A    0  ]
// with nz=10 (primal vars), nc=5 (constraints), total N=15.
// Matrix type: RealSymmetricIndefinite.

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
            const char *p = line.c_str() + 7;
            while (*p == ' ' || *p == '\t') ++p;
            return std::atol(p);
        }
    }
    return -1L;
}

// --------------------------------------------------------------------------
// Utility: SpMV for residual — symmetric lower-triangular CSC
// --------------------------------------------------------------------------
static void spmv_sym_lower(const smf::CscLower &A, const double *x, double *y) {
    const smf::Int n = A.n;
    for (smf::Int i = 0; i < n; ++i) y[i] = 0.0;
    for (smf::Int j = 0; j < n; ++j) {
        for (smf::Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            smf::Int i = A.row_idx[static_cast<std::size_t>(p)];
            double   v = A.values[static_cast<std::size_t>(p)];
            y[i] += v * x[j];
            if (i != j) y[j] += v * x[i];
        }
    }
}

// --------------------------------------------------------------------------
// Build lower-CSC of the KKT matrix.
//
//   Full N×N matrix (N = nz+nc) stored in lower triangle:
//
//     Col j in [0, nz):
//       (j, j)   = H[j]  (diagonal of H)
//       (nz+i, j) for i in [0, nc) where A[i,j] != 0
//         A[i,j] = 1.0 if j==i or j==i+1 (banded, indices clamped to [0,nz))
//
//     Col j in [nz, nz+nc):   (all-zero Schur block — no entries stored)
//       No explicit diagonal entry (zero block): nothing in lower triangle
//       except we do need the diagonal for the sparse solver to handle it.
//       Actually we store an explicit 0 on each diagonal of the (0) block
//       because some solvers require diagonal entries. But smf handles
//       missing diagonals gracefully (matrix_missing_diag counter).
//       We omit them here and let smf report the structural rank.
// --------------------------------------------------------------------------
static smf::CscLower build_kkt(int nz, int nc) {
    // N = nz + nc
    const int N = nz + nc;

    // Determine entries per column
    // A is nc x nz:  A[i,j] = 1.0 if j==i or j==i+1, for i in [0,nc)
    // In the full KKT lower triangle, column j (j < nz) contains:
    //   diagonal (j,j)
    //   for each i s.t. A[i,j]!=0: entry (nz+i, j)

    std::vector<std::vector<std::pair<smf::Int, double>>> cols(
        static_cast<std::size_t>(N));

    // H diagonal
    for (int j = 0; j < nz; ++j) {
        cols[static_cast<std::size_t>(j)].emplace_back(
            static_cast<smf::Int>(j), static_cast<double>(j + 1));
    }

    // A block: A[i,j] entries become (nz+i, j) in KKT lower triangle
    for (int i = 0; i < nc; ++i) {
        // j == i
        if (i < nz) {
            cols[static_cast<std::size_t>(i)].emplace_back(
                static_cast<smf::Int>(nz + i), 1.0);
        }
        // j == i+1
        if (i + 1 < nz) {
            cols[static_cast<std::size_t>(i + 1)].emplace_back(
                static_cast<smf::Int>(nz + i), 1.0);
        }
    }

    // Schur block regularization: add small negative diagonal entries to make
    // the bottom-right block -eps*I (regularized saddle-point, invertible).
    // Without this the solver detects structurally missing diagonals and the
    // numerical rank drops.
    constexpr double schur_reg = -1.0e-6;
    for (int i = 0; i < nc; ++i) {
        cols[static_cast<std::size_t>(nz + i)].emplace_back(
            static_cast<smf::Int>(nz + i), schur_reg);
    }

    // Build col_ptr
    smf::CscLower A;
    A.n = static_cast<smf::Int>(N);
    A.col_ptr.resize(static_cast<std::size_t>(N) + 1);
    A.col_ptr[0] = 0;
    for (int j = 0; j < N; ++j) {
        // sort by row within column
        auto &c = cols[static_cast<std::size_t>(j)];
        std::sort(c.begin(), c.end(),
                  [](const std::pair<smf::Int,double> &a,
                     const std::pair<smf::Int,double> &b) {
                      return a.first < b.first;
                  });
        A.col_ptr[static_cast<std::size_t>(j) + 1] =
            A.col_ptr[static_cast<std::size_t>(j)] +
            static_cast<smf::Int>(c.size());
    }
    smf::Int nnz = A.col_ptr[static_cast<std::size_t>(N)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    smf::Int pos = 0;
    for (int j = 0; j < N; ++j) {
        for (auto &e : cols[static_cast<std::size_t>(j)]) {
            A.row_idx[static_cast<std::size_t>(pos)] = e.first;
            A.values[static_cast<std::size_t>(pos)]  = e.second;
            ++pos;
        }
    }
    return A;
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
int main(int /*argc*/, char * /*argv*/[]) {
    constexpr int nz = 10;
    constexpr int nc = 5;
    const int N = nz + nc;

    std::printf("=== bench_kkt_ocp ===\n");

    smf::CscLower A = build_kkt(nz, nc);
    std::printf("KKT: nz=%d nc=%d N=%d nnz=%d (lower triangular)\n",
                nz, nc, N, static_cast<int>(A.nnz()));

    // RHS b = [1,...,1]
    std::vector<double> b(static_cast<std::size_t>(N), 1.0);
    std::vector<double> b_orig(b);

    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
    smf::Info info;
    smf::Solver solver;

    // Analyse
    auto t0 = std::chrono::high_resolution_clock::now();
    auto ak = solver.analyse(A, ctrl, info);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!ak) {
        std::fprintf(stderr, "bench_kkt_ocp: analyse() failed\n");
        return 1;
    }
    double t_analyse = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Factor
    smf::FactorKeep fkeep;
    auto t2 = std::chrono::high_resolution_clock::now();
    smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fkeep);
    auto t3 = std::chrono::high_resolution_clock::now();
    if (fs != smf::FactorStatus::Success) {
        std::fprintf(stderr, "bench_kkt_ocp: factor() returned status %d\n",
                     static_cast<int>(fs));
        return 1;
    }
    double t_factor = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Solve
    auto t4 = std::chrono::high_resolution_clock::now();
    int rc = solver.solve(fkeep, ctrl, info, b.data(), N, 1);
    auto t5 = std::chrono::high_resolution_clock::now();
    if (rc != 0) {
        std::fprintf(stderr, "bench_kkt_ocp: solve() returned %d\n", rc);
        return 1;
    }
    double t_solve = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // Residual
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
