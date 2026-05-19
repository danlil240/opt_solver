// bench_suite_sparse_matrix_market.cpp — Matrix Market file benchmark for smf
// Part of the smf MA97-class solver benchmark suite.
//
// Usage: bench_suite_sparse_matrix_market <matrix.mtx> [nrhs=1]
//
// Reads a Matrix Market (.mtx) file (coordinate format, real, symmetric),
// builds a lower-CSC matrix, and runs smf analyse/factor/solve.
// Gracefully reports missing/unreadable files.

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
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
// Matrix Market reader.
// Supports: %%MatrixMarket matrix coordinate real symmetric/general
// Returns false on parse error; sets rows, cols, and fills COO arrays.
// --------------------------------------------------------------------------
struct CooEntry { smf::Int row, col; double val; };

static bool read_matrix_market(const char *path,
                                int &rows, int &cols,
                                std::vector<CooEntry> &coo,
                                bool &is_symmetric)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::fprintf(stderr, "Cannot open file: %s\n", path);
        return false;
    }

    is_symmetric = false;
    bool header_parsed = false;
    bool size_parsed   = false;
    int  nnz_file = 0;

    std::string line;
    while (std::getline(ifs, line)) {
        // Strip trailing CR
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (!header_parsed) {
            // First non-empty line must be %%MatrixMarket header
            if (line.empty()) continue;
            // Convert to lowercase for comparison
            std::string lower = line;
            for (char &c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (lower.rfind("%%matrixmarket", 0) != 0) {
                std::fprintf(stderr, "Not a Matrix Market file: missing %%MatrixMarket header\n");
                return false;
            }
            if (lower.find("coordinate") == std::string::npos) {
                std::fprintf(stderr, "Only coordinate format supported\n");
                return false;
            }
            if (lower.find("symmetric") != std::string::npos) {
                is_symmetric = true;
            }
            header_parsed = true;
            continue;
        }

        // Skip comment lines
        if (!line.empty() && line[0] == '%') continue;
        if (line.empty()) continue;

        if (!size_parsed) {
            // Read: rows cols nnz
            std::istringstream ss(line);
            int nrows = 0, ncols = 0, nz = 0;
            if (!(ss >> nrows >> ncols >> nz)) {
                std::fprintf(stderr, "Failed to parse size line: %s\n", line.c_str());
                return false;
            }
            rows     = nrows;
            cols     = ncols;
            nnz_file = nz;
            coo.reserve(static_cast<std::size_t>(nz));
            size_parsed = true;
            continue;
        }

        // Data lines: row col value (1-based)
        std::istringstream ss(line);
        int r = 0, c = 0;
        double v = 0.0;
        if (!(ss >> r >> c >> v)) continue; // skip malformed lines

        // Convert to 0-based
        r -= 1; c -= 1;

        if (r < 0 || c < 0 || r >= rows || c >= cols) continue; // out of range

        if (is_symmetric) {
            // Store only lower triangle (row >= col)
            if (r >= c) {
                coo.push_back({static_cast<smf::Int>(r),
                               static_cast<smf::Int>(c), v});
            } else {
                // Swap so we store lower triangle
                coo.push_back({static_cast<smf::Int>(c),
                               static_cast<smf::Int>(r), v});
            }
        } else {
            // General: use lower triangle entries only (row >= col)
            if (r >= c) {
                coo.push_back({static_cast<smf::Int>(r),
                               static_cast<smf::Int>(c), v});
            }
            // Upper triangle entries (r < c) are ignored for symmetric treatment.
        }
    }

    if (!size_parsed) {
        std::fprintf(stderr, "No size line found in Matrix Market file\n");
        return false;
    }
    if (rows != cols) {
        std::fprintf(stderr, "Matrix is not square (%d x %d) — not supported\n",
                     rows, cols);
        return false;
    }
    (void)nnz_file; // we trust the actual data count
    return true;
}

// --------------------------------------------------------------------------
// COO → lower-CSC (sort by col then row, merge duplicates by summation)
// --------------------------------------------------------------------------
static smf::CscLower coo_to_lower_csc(int N, std::vector<CooEntry> &coo) {
    // Sort by (col, row)
    std::sort(coo.begin(), coo.end(), [](const CooEntry &a, const CooEntry &b) {
        if (a.col != b.col) return a.col < b.col;
        return a.row < b.row;
    });

    smf::CscLower A;
    A.n = static_cast<smf::Int>(N);
    A.col_ptr.resize(static_cast<std::size_t>(N) + 1, 0);

    // Count deduplicated entries per column
    // Merge duplicate (row,col) pairs
    std::vector<CooEntry> merged;
    merged.reserve(coo.size());
    for (auto &e : coo) {
        if (!merged.empty() && merged.back().row == e.row && merged.back().col == e.col) {
            merged.back().val += e.val;
        } else {
            merged.push_back(e);
        }
    }

    for (auto &e : merged) {
        A.col_ptr[static_cast<std::size_t>(e.col) + 1]++;
    }
    for (int j = 0; j < N; ++j) {
        A.col_ptr[static_cast<std::size_t>(j) + 1] +=
            A.col_ptr[static_cast<std::size_t>(j)];
    }

    smf::Int nnz = A.col_ptr[static_cast<std::size_t>(N)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    std::vector<smf::Int> pos(static_cast<std::size_t>(N), 0);
    for (auto &e : merged) {
        smf::Int base = A.col_ptr[static_cast<std::size_t>(e.col)];
        smf::Int off  = pos[static_cast<std::size_t>(e.col)];
        A.row_idx[static_cast<std::size_t>(base + off)] = e.row;
        A.values[static_cast<std::size_t>(base + off)]  = e.val;
        ++pos[static_cast<std::size_t>(e.col)];
    }
    return A;
}

// --------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    std::printf("=== bench_suite_sparse_matrix_market ===\n");

    if (argc < 2) {
        std::printf("bench_suite_sparse_matrix_market: no matrix file provided (pass path as argv[1])\n");
        std::printf("Usage: bench_suite_sparse_matrix_market <path/to/matrix.mtx>\n");
        std::printf("Skipping benchmark.\n");
        return 0;
    }

    const char *path = argv[1];
    int nrhs = 1;
    if (argc >= 3) nrhs = std::atoi(argv[2]);
    if (nrhs < 1) nrhs = 1;

    // Check file exists
    {
        std::ifstream test(path);
        if (!test.is_open()) {
            std::printf("bench_suite_sparse_matrix_market: cannot open file '%s'\n", path);
            std::printf("Usage: bench_suite_sparse_matrix_market <path/to/matrix.mtx>\n");
            std::printf("Skipping benchmark.\n");
            return 0;
        }
    }

    int rows = 0, cols = 0;
    bool is_symmetric = false;
    std::vector<CooEntry> coo;

    if (!read_matrix_market(path, rows, cols, coo, is_symmetric)) {
        std::printf("bench_suite_sparse_matrix_market: failed to parse '%s'\n", path);
        std::printf("Skipping benchmark.\n");
        return 0;
    }

    const int N = rows;
    smf::CscLower A = coo_to_lower_csc(N, coo);

    std::printf("File  : %s\n", path);
    std::printf("Matrix: N=%d nnz=%d symmetric=%s\n",
                N, static_cast<int>(A.nnz()), is_symmetric ? "yes" : "no (using lower only)");
    std::printf("nrhs  : %d\n", nrhs);

    // RHS
    std::vector<double> b(static_cast<std::size_t>(N * nrhs), 1.0);
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
        std::fprintf(stderr, "bench_suite_sparse_matrix_market: analyse() failed\n");
        return 1;
    }
    double t_analyse = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Factor
    smf::FactorKeep fkeep;
    auto t2 = std::chrono::high_resolution_clock::now();
    smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fkeep);
    auto t3 = std::chrono::high_resolution_clock::now();
    if (fs != smf::FactorStatus::Success) {
        std::fprintf(stderr, "bench_suite_sparse_matrix_market: factor() returned status %d\n",
                     static_cast<int>(fs));
        return 1;
    }
    double t_factor = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Solve
    auto t4 = std::chrono::high_resolution_clock::now();
    int rc = solver.solve(fkeep, ctrl, info, b.data(), N, nrhs);
    auto t5 = std::chrono::high_resolution_clock::now();
    if (rc != 0) {
        std::fprintf(stderr, "bench_suite_sparse_matrix_market: solve() returned %d\n", rc);
        return 1;
    }
    double t_solve = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // Residual (first RHS only)
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
