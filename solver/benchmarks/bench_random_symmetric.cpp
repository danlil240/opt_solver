// bench_random_symmetric.cpp — Random symmetric indefinite matrix benchmark
// Part of the smf MA97-class solver benchmark suite.
//
// Usage: bench_random_symmetric [N=100]
//
// Builds a block-diagonal symmetric indefinite matrix of size N:
//   Blocks of [[2,-1],[-1,2]] (PD 2×2) and [[-1]] (ND 1×1) interleaved.
// Uses MatrixType::RealSymmetricIndefinite.

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

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
// Build a block-diagonal indefinite matrix.
// Pattern (repeating until N is filled):
//   block_2x2 = [[2,-1],[-1,2]]  (pos def)
//   block_1x1 = [[-1]]           (neg def)
// This gives a clearly indefinite matrix.
// --------------------------------------------------------------------------
static smf::CscLower build_block_indef(int N) {
    // Enumerate entries column by column.
    std::vector<std::pair<smf::Int, double>> entries; // (row, val) per col
    std::vector<smf::Int> col_start(static_cast<std::size_t>(N) + 1, 0);

    // We'll build COO first, then convert to CSC.
    struct Entry { smf::Int row, col; double val; };
    std::vector<Entry> coo;

    int k = 0; // current row/col index
    while (k < N) {
        if (k + 1 < N) {
            // 2×2 block starting at k
            // (k,k)=2, (k+1,k)=-1, (k+1,k+1)=2
            coo.push_back({static_cast<smf::Int>(k),   static_cast<smf::Int>(k),   2.0});
            coo.push_back({static_cast<smf::Int>(k+1), static_cast<smf::Int>(k),  -1.0});
            coo.push_back({static_cast<smf::Int>(k+1), static_cast<smf::Int>(k+1), 2.0});
            k += 2;
        } else {
            // 1×1 block (last if N is odd)
            coo.push_back({static_cast<smf::Int>(k), static_cast<smf::Int>(k), -1.0});
            k += 1;
        }
        // After each PD block, add a ND 1×1 if room
        if (k < N) {
            coo.push_back({static_cast<smf::Int>(k), static_cast<smf::Int>(k), -1.0});
            k += 1;
        }
    }

    // Convert COO to lower-CSC
    // Count entries per column
    std::vector<smf::Int> cnt(static_cast<std::size_t>(N), 0);
    for (auto &e : coo) {
        // Only lower triangle (row >= col): already satisfied by construction
        ++cnt[static_cast<std::size_t>(e.col)];
    }

    smf::CscLower A;
    A.n = static_cast<smf::Int>(N);
    A.col_ptr.resize(static_cast<std::size_t>(N) + 1);
    A.col_ptr[0] = 0;
    for (int j = 0; j < N; ++j) {
        A.col_ptr[static_cast<std::size_t>(j) + 1] =
            A.col_ptr[static_cast<std::size_t>(j)] + cnt[static_cast<std::size_t>(j)];
    }
    smf::Int nnz = A.col_ptr[static_cast<std::size_t>(N)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    std::fill(cnt.begin(), cnt.end(), 0);
    for (auto &e : coo) {
        smf::Int base = A.col_ptr[static_cast<std::size_t>(e.col)];
        smf::Int off  = cnt[static_cast<std::size_t>(e.col)];
        A.row_idx[static_cast<std::size_t>(base + off)] = e.row;
        A.values[static_cast<std::size_t>(base + off)]  = e.val;
        ++cnt[static_cast<std::size_t>(e.col)];
    }

    // Sort rows within each column (should already be sorted by construction)
    for (int j = 0; j < N; ++j) {
        smf::Int beg = A.col_ptr[static_cast<std::size_t>(j)];
        smf::Int end = A.col_ptr[static_cast<std::size_t>(j) + 1];
        // Insertion-sort the small column (at most 2 entries)
        for (smf::Int p = beg + 1; p < end; ++p) {
            smf::Int ridx = A.row_idx[static_cast<std::size_t>(p)];
            double   rval = A.values[static_cast<std::size_t>(p)];
            smf::Int q = p - 1;
            while (q >= beg && A.row_idx[static_cast<std::size_t>(q)] > ridx) {
                A.row_idx[static_cast<std::size_t>(q + 1)] = A.row_idx[static_cast<std::size_t>(q)];
                A.values[static_cast<std::size_t>(q + 1)]  = A.values[static_cast<std::size_t>(q)];
                --q;
            }
            A.row_idx[static_cast<std::size_t>(q + 1)] = ridx;
            A.values[static_cast<std::size_t>(q + 1)]  = rval;
        }
    }

    return A;
}

// --------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    int N = 100;
    if (argc >= 2) {
        N = std::atoi(argv[1]);
        if (N < 2) { std::fprintf(stderr, "N must be >= 2\n"); return 1; }
    }

    std::printf("=== bench_random_symmetric ===\n");

    smf::CscLower A = build_block_indef(N);
    std::printf("Symmetric indefinite block-diagonal: N=%d nnz=%d (lower triangular)\n",
                N, static_cast<int>(A.nnz()));

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
        std::fprintf(stderr, "bench_random_symmetric: analyse() failed\n");
        return 1;
    }
    double t_analyse = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Factor
    smf::FactorKeep fkeep;
    auto t2 = std::chrono::high_resolution_clock::now();
    smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fkeep);
    auto t3 = std::chrono::high_resolution_clock::now();
    if (fs != smf::FactorStatus::Success) {
        std::fprintf(stderr, "bench_random_symmetric: factor() returned status %d\n",
                     static_cast<int>(fs));
        return 1;
    }
    double t_factor = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Solve
    auto t4 = std::chrono::high_resolution_clock::now();
    int rc = solver.solve(fkeep, ctrl, info, b.data(), N, 1);
    auto t5 = std::chrono::high_resolution_clock::now();
    if (rc != 0) {
        std::fprintf(stderr, "bench_random_symmetric: solve() returned %d\n", rc);
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
