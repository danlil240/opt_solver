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

#ifdef SMF_HAS_EIGEN
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#endif

#ifdef SMF_HAS_CHOLMOD
#include <cholmod.h>
#endif

#ifdef SMF_HAS_MUMPS
#include <dmumps_c.h>
#ifndef USE_COMM_WORLD
#define USE_COMM_WORLD (-987654)
#endif
#endif

#ifdef SMF_HAS_MA27
// MA27 double-precision Fortran interface (from CoinHSL / libcoinhsl)
extern "C"
{
    void ma27id_(int* icntl, double* cntl);
    void ma27ad_(int* n,
                 int* nz,
                 int* irn,
                 int* icn,
                 int* iw,
                 int* liw,
                 int* ikeep,
                 int* iw1,
                 int* nsteps,
                 int* iflag,
                 int* icntl,
                 double* cntl,
                 int* info,
                 double* ops);
    void ma27bd_(int* n,
                 int* nz,
                 int* irn,
                 int* icn,
                 double* a,
                 int* la,
                 int* iw,
                 int* liw,
                 int* ikeep,
                 int* nsteps,
                 int* maxfrt,
                 int* iw1,
                 int* icntl,
                 double* cntl,
                 int* info);
    void ma27cd_(int* n,
                 double* a,
                 int* la,
                 int* iw,
                 int* liw,
                 double* w,
                 int* maxfrt,
                 double* rhs,
                 int* iw1,
                 int* nsteps,
                 int* icntl,
                 int* info);
}
#endif

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

static inline double elapsed_ms(TimePoint t0, TimePoint t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// SpMV for residual: y = A*x  (A symmetric, stored as lower triangle)
// ---------------------------------------------------------------------------
static void spmv_sym(const smf::CscLower &A, const double* x, double* y)
{
    const smf::Int n = A.n;
    for (smf::Int i = 0; i < n; ++i)
        y[i] = 0.0;
    for (smf::Int j = 0; j < n; ++j)
    {
        for (smf::Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
        {
            smf::Int i = A.row_idx[static_cast<std::size_t>(p)];
            double v = A.values[static_cast<std::size_t>(p)];
            y[i] += v * x[j];
            if (i != j)
                y[j] += v * x[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix generators — all return lower-triangular CSC
// ---------------------------------------------------------------------------

/// 2-D 5-point Poisson on grid_n × grid_n grid.
/// N = grid_n^2, diag = 4, off-diag = -1.
static smf::CscLower make_poisson2d(int grid_n)
{
    const int N = grid_n * grid_n;

    // Count entries per column (lower triangle only)
    std::vector<smf::Int> col_count(static_cast<std::size_t>(N), 0);
    for (int row = 0; row < grid_n; ++row)
    {
        for (int col = 0; col < grid_n; ++col)
        {
            int k = row * grid_n + col;
            col_count[static_cast<std::size_t>(k)]++; // diagonal
            if (col > 0)
                col_count[static_cast<std::size_t>(k - 1)]++; // right neigh stored in lower col
            if (row > 0)
                col_count[static_cast<std::size_t>(k - grid_n)]++; // below neigh
        }
    }

    smf::CscLower A;
    A.n = static_cast<smf::Int>(N);
    A.col_ptr.resize(static_cast<std::size_t>(N + 1));
    A.col_ptr[0] = 0;
    for (int j = 0; j < N; ++j)
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            A.col_ptr[static_cast<std::size_t>(j)] + col_count[static_cast<std::size_t>(j)];

    const smf::Int nnz = A.col_ptr[static_cast<std::size_t>(N)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    std::vector<smf::Int> pos(A.col_ptr.begin(), A.col_ptr.end());

    for (int row = 0; row < grid_n; ++row)
    {
        for (int col = 0; col < grid_n; ++col)
        {
            int k = row * grid_n + col;
            // diagonal
            smf::Int p = pos[static_cast<std::size_t>(k)]++;
            A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(k);
            A.values[static_cast<std::size_t>(p)] = 4.0;
            // left neighbour: (k, k-1) in lower triangle → row=k, col=k-1
            if (col > 0)
            {
                smf::Int q = pos[static_cast<std::size_t>(k - 1)]++;
                A.row_idx[static_cast<std::size_t>(q)] = static_cast<smf::Int>(k);
                A.values[static_cast<std::size_t>(q)] = -1.0;
            }
            // below neighbour: (k, k-grid_n) → row=k, col=k-grid_n
            if (row > 0)
            {
                smf::Int q = pos[static_cast<std::size_t>(k - grid_n)]++;
                A.row_idx[static_cast<std::size_t>(q)] = static_cast<smf::Int>(k);
                A.values[static_cast<std::size_t>(q)] = -1.0;
            }
        }
    }
    return A;
}

/// Symmetric tridiagonal: diag = diag_val, sub-diag = offdiag_val.
/// SPD when diag_val > 2 * |offdiag_val|.
static smf::CscLower make_tridiagonal(int n, double diag_val = 2.0, double offdiag_val = -1.0)
{
    // Lower triangle: col j has diagonal (row j) + sub-diagonal below (row j+1),
    // except the last column which has only the diagonal.
    smf::CscLower A;
    A.n = static_cast<smf::Int>(n);
    A.col_ptr.resize(static_cast<std::size_t>(n + 1));
    A.col_ptr[0] = 0;
    for (int j = 0; j < n; ++j)
        A.col_ptr[static_cast<std::size_t>(j + 1)] = A.col_ptr[static_cast<std::size_t>(j)] + (j < n - 1 ? 2 : 1);

    const smf::Int nnz = A.col_ptr[static_cast<std::size_t>(n)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    for (int j = 0; j < n; ++j)
    {
        smf::Int p = A.col_ptr[static_cast<std::size_t>(j)];
        // diagonal
        A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(j);
        A.values[static_cast<std::size_t>(p)] = diag_val;
        ++p;
        // sub-diagonal: entry (j+1, j) in lower triangle
        if (j < n - 1)
        {
            A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(j + 1);
            A.values[static_cast<std::size_t>(p)] = offdiag_val;
        }
    }
    return A;
}

/// Banded SPD: dominant diagonal, bandwidth sub-diagonals all = -1.
static smf::CscLower make_banded_spd(int n, int bw = 5)
{
    double diag_val = static_cast<double>(bw * 2 + 1) * 2.0;

    // Count entries per column (lower triangle)
    std::vector<smf::Int> col_count(static_cast<std::size_t>(n), 0);
    for (int j = 0; j < n; ++j)
    {
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
            A.col_ptr[static_cast<std::size_t>(j)] + col_count[static_cast<std::size_t>(j)];

    const smf::Int nnz = A.col_ptr[static_cast<std::size_t>(n)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    for (int j = 0; j < n; ++j)
    {
        smf::Int p = A.col_ptr[static_cast<std::size_t>(j)];
        // diagonal
        A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(j);
        A.values[static_cast<std::size_t>(p)] = diag_val;
        ++p;
        // sub-diagonals (rows j+1 .. j+bw stored in column j)
        int rows_below = std::min(bw, n - 1 - j);
        for (int k = 1; k <= rows_below; ++k)
        {
            A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(j + k);
            A.values[static_cast<std::size_t>(p)] = -1.0;
            ++p;
        }
    }
    return A;
}

/// Block-diagonal SPD.  Blocks of size 3 (and 2 for remainder).
///   3×3 block: diag=4, off=-1;  2×2 block: diag=2, off=-1.
static smf::CscLower make_block_diagonal_spd(int n)
{
    // Pre-compute nnz
    smf::Int total_nnz = 0;
    {
        int remaining = n;
        while (remaining > 0)
        {
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

    while (col_offset < n)
    {
        int bs = (n - col_offset >= 3) ? 3 : (n - col_offset);
        double dv = (bs == 3) ? 4.0 : 2.0;
        double ov = -1.0;

        for (int lc = 0; lc < bs; ++lc)
        {
            int j = col_offset + lc;
            // column j: entries at rows j, j+1, .., col_offset+bs-1
            for (int lr = lc; lr < bs; ++lr)
            {
                int i = col_offset + lr;
                A.row_idx[static_cast<std::size_t>(p)] = static_cast<smf::Int>(i);
                A.values[static_cast<std::size_t>(p)] = (lr == lc) ? dv : ov;
                ++p;
            }
            A.col_ptr[static_cast<std::size_t>(j + 1)] =
                A.col_ptr[static_cast<std::size_t>(j)] + static_cast<smf::Int>(bs - lc);
        }
        col_offset += bs;
    }
    return A;
}

// ---------------------------------------------------------------------------
// smf solve: analyse + factor + solve, return relative residual
// ---------------------------------------------------------------------------
struct SmfTimes
{
    double analyse_ms = 0.0;
    double factor_ms = 0.0;
    double solve_ms = 0.0;
    double total_ms() const { return analyse_ms + factor_ms + solve_ms; }
};

static double smf_run(const smf::CscLower &A, std::vector<double> &x_out, SmfTimes &t)
{
    const smf::Int n = A.n;
    x_out.assign(static_cast<std::size_t>(n), 1.0);

    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricPositiveDefinite;
    smf::Info info;
    smf::Solver solver;

    auto ta0 = Clock::now();
    auto ak = solver.analyse(A, ctrl, info);
    auto ta1 = Clock::now();
    if (!ak)
        return -1.0;

    smf::FactorKeep fk;
    auto fs = solver.factor(*ak, ctrl, info, fk);
    auto ta2 = Clock::now();
    if (fs != smf::FactorStatus::Success)
        return -1.0;

    solver.solve(fk, ctrl, info, x_out.data(), static_cast<int>(n));
    auto ta3 = Clock::now();

    t.analyse_ms = elapsed_ms(ta0, ta1);
    t.factor_ms = elapsed_ms(ta1, ta2);
    t.solve_ms = elapsed_ms(ta2, ta3);

    // residual ||Ax - b||_2 / sqrt(n)
    std::vector<double> Ax(static_cast<std::size_t>(n), 0.0);
    spmv_sym(A, x_out.data(), Ax.data());
    double res = 0.0;
    for (smf::Int i = 0; i < n; ++i)
    {
        double r = Ax[static_cast<std::size_t>(i)] - 1.0;
        res += r * r;
    }
    return std::sqrt(res) / std::sqrt(static_cast<double>(n));
}

// ---------------------------------------------------------------------------
// Timing helper for repeated-factorisation benchmarks
// ---------------------------------------------------------------------------
struct RepeatedTimes
{
    double analyse_ms    = 0.0; ///< single analyse call
    double avg_factor_ms = 0.0; ///< average over reps factor calls
    double avg_solve_ms  = 0.0; ///< average over reps solve calls
    double amortized_ms(int reps) const
    {
        return analyse_ms / static_cast<double>(reps) + avg_factor_ms + avg_solve_ms;
    }
};

/// smf repeated factorisation: analyse once, then factor+solve reps times.
static bool smf_repeated_run(const smf::CscLower &A, int reps, RepeatedTimes &t)
{
    const smf::Int n = A.n;
    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricPositiveDefinite;
    smf::Info info;
    smf::Solver solver;

    auto ta0 = Clock::now();
    auto ak = solver.analyse(A, ctrl, info);
    auto ta1 = Clock::now();
    if (!ak)
        return false;
    t.analyse_ms = elapsed_ms(ta0, ta1);

    double total_factor = 0.0, total_solve = 0.0;
    for (int i = 0; i < reps; ++i)
    {
        std::vector<double> x(static_cast<std::size_t>(n), 1.0);
        smf::FactorKeep fk;
        auto tf0 = Clock::now();
        auto fs = solver.factor(*ak, ctrl, info, fk);
        auto tf1 = Clock::now();
        if (fs != smf::FactorStatus::Success)
            return false;
        solver.solve(fk, ctrl, info, x.data(), static_cast<int>(n));
        auto tf2 = Clock::now();
        total_factor += elapsed_ms(tf0, tf1);
        total_solve  += elapsed_ms(tf1, tf2);
    }
    t.avg_factor_ms = total_factor / static_cast<double>(reps);
    t.avg_solve_ms  = total_solve  / static_cast<double>(reps);
    return true;
}

// ---------------------------------------------------------------------------
// Eigen SimplicialLDLT solve
// ---------------------------------------------------------------------------
#ifdef SMF_HAS_EIGEN
static Eigen::SparseMatrix<double> to_eigen_sym(const smf::CscLower &A)
{
    int n = static_cast<int>(A.n);
    Eigen::SparseMatrix<double> M(n, n);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(A.nnz()) * 2);
    for (int j = 0; j < n; ++j)
    {
        for (smf::Int k = A.col_ptr[static_cast<std::size_t>(j)]; k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k)
        {
            int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(k)]);
            double v = A.values[static_cast<std::size_t>(k)];
            triplets.emplace_back(i, j, v);
            if (i != j)
                triplets.emplace_back(j, i, v);
        }
    }
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

static double eigen_run(const Eigen::SparseMatrix<double> &M, Eigen::VectorXd &x_out, double &total_ms)
{
    int n = static_cast<int>(M.rows());
    Eigen::VectorXd b = Eigen::VectorXd::Ones(n);

    auto t0 = Clock::now();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(M);
    x_out = ldlt.solve(b);
    auto t1 = Clock::now();

    total_ms = elapsed_ms(t0, t1);

    Eigen::VectorXd res = M * x_out - b;
    return res.norm() / std::sqrt(static_cast<double>(n));
}

/// Eigen repeated factorisation: analyzePattern once, then factorize+solve reps times.
static bool eigen_repeated_run(const Eigen::SparseMatrix<double> &M, int reps, RepeatedTimes &t)
{
    int n = static_cast<int>(M.rows());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    Eigen::VectorXd b = Eigen::VectorXd::Ones(n);

    auto ta0 = Clock::now();
    ldlt.analyzePattern(M);
    auto ta1 = Clock::now();
    t.analyse_ms = elapsed_ms(ta0, ta1);

    double total_factor = 0.0, total_solve = 0.0;
    for (int i = 0; i < reps; ++i)
    {
        auto tf0 = Clock::now();
        ldlt.factorize(M);
        auto tf1 = Clock::now();
        Eigen::VectorXd x = ldlt.solve(b);
        auto tf2 = Clock::now();
        total_factor += elapsed_ms(tf0, tf1);
        total_solve  += elapsed_ms(tf1, tf2);
    }
    t.avg_factor_ms = total_factor / static_cast<double>(reps);
    t.avg_solve_ms  = total_solve  / static_cast<double>(reps);
    return (ldlt.info() == Eigen::Success);
}
#endif // SMF_HAS_EIGEN

// ---------------------------------------------------------------------------
// CHOLMOD SPD solve: analyse + factorize + solve, return relative residual
// ---------------------------------------------------------------------------
#ifdef SMF_HAS_CHOLMOD
struct CholmodTimes
{
    double analyse_ms = 0.0;
    double factor_ms = 0.0;
    double solve_ms = 0.0;
    double total_ms() const { return analyse_ms + factor_ms + solve_ms; }
};

static double cholmod_run(const smf::CscLower &A, std::vector<double> &x_out, CholmodTimes &t)
{
    const int n = static_cast<int>(A.n);
    const int nnz = static_cast<int>(A.nnz());
    x_out.assign(static_cast<std::size_t>(n), 1.0);

    cholmod_common c;
    cholmod_start(&c);
    c.print = 0; // suppress diagnostic output

    // Build lower-triangular CSC in CHOLMOD format (stype=-1 = lower stored)
    cholmod_sparse* Ac =
        cholmod_allocate_sparse(static_cast<std::size_t>(n), static_cast<std::size_t>(n), static_cast<std::size_t>(nnz),
                                1,  // sorted
                                1,  // packed
                                -1, // stype=-1: symmetric, lower triangle stored
                                CHOLMOD_REAL, &c);

    int* Ap = static_cast<int*>(Ac->p);
    int* Ai = static_cast<int*>(Ac->i);
    double* Ax = static_cast<double*>(Ac->x);
    for (int j = 0; j <= n; ++j)
        Ap[j] = static_cast<int>(A.col_ptr[static_cast<std::size_t>(j)]);
    for (int k = 0; k < nnz; ++k)
    {
        Ai[k] = static_cast<int>(A.row_idx[static_cast<std::size_t>(k)]);
        Ax[k] = A.values[static_cast<std::size_t>(k)];
    }

    cholmod_dense* b_ch = cholmod_ones(static_cast<std::size_t>(n), 1, CHOLMOD_REAL, &c);

    auto ta = Clock::now();
    cholmod_factor* L = cholmod_analyze(Ac, &c);
    auto tb = Clock::now();
    cholmod_factorize(Ac, L, &c);
    auto tc = Clock::now();
    cholmod_dense* x_ch = cholmod_solve(CHOLMOD_A, L, b_ch, &c);
    auto td = Clock::now();

    t.analyse_ms = elapsed_ms(ta, tb);
    t.factor_ms = elapsed_ms(tb, tc);
    t.solve_ms = elapsed_ms(tc, td);

    double res = -1.0;
    if (c.status == CHOLMOD_OK && x_ch)
    {
        double* xp = static_cast<double*>(x_ch->x);
        for (int i = 0; i < n; ++i)
            x_out[static_cast<std::size_t>(i)] = xp[i];
        std::vector<double> Ax_vec(static_cast<std::size_t>(n), 0.0);
        spmv_sym(A, x_out.data(), Ax_vec.data());
        double rr = 0.0;
        for (int i = 0; i < n; ++i)
        {
            double r = Ax_vec[static_cast<std::size_t>(i)] - 1.0;
            rr += r * r;
        }
        res = std::sqrt(rr) / std::sqrt(static_cast<double>(n));
    }

    if (x_ch)
        cholmod_free_dense(&x_ch, &c);
    cholmod_free_dense(&b_ch, &c);
    if (L)
        cholmod_free_factor(&L, &c);
    cholmod_free_sparse(&Ac, &c);
    cholmod_finish(&c);
    return res;
}

/// CHOLMOD repeated factorisation: analyze once, then factorize+solve reps times.
static bool cholmod_repeated_run(const smf::CscLower &A, int reps, RepeatedTimes &t)
{
    const int n   = static_cast<int>(A.n);
    const int nnz = static_cast<int>(A.nnz());

    cholmod_common c;
    cholmod_start(&c);
    c.print = 0;

    cholmod_sparse *Ac =
        cholmod_allocate_sparse(static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                                static_cast<std::size_t>(nnz),
                                1, 1, -1, CHOLMOD_REAL, &c);
    {
        int    *Ap = static_cast<int    *>(Ac->p);
        int    *Ai = static_cast<int    *>(Ac->i);
        double *Ax = static_cast<double *>(Ac->x);
        for (int j = 0; j <= n; ++j)
            Ap[j] = static_cast<int>(A.col_ptr[static_cast<std::size_t>(j)]);
        for (int k = 0; k < nnz; ++k)
        {
            Ai[k] = static_cast<int>(A.row_idx[static_cast<std::size_t>(k)]);
            Ax[k] = A.values[static_cast<std::size_t>(k)];
        }
    }

    auto ta = Clock::now();
    cholmod_factor *L = cholmod_analyze(Ac, &c);
    auto tb = Clock::now();
    if (!L || c.status != CHOLMOD_OK)
    {
        cholmod_free_sparse(&Ac, &c);
        cholmod_finish(&c);
        return false;
    }
    t.analyse_ms = elapsed_ms(ta, tb);

    double total_factor = 0.0, total_solve = 0.0;
    for (int i = 0; i < reps; ++i)
    {
        auto tf0 = Clock::now();
        cholmod_factorize(Ac, L, &c);
        auto tf1 = Clock::now();
        cholmod_dense *b_ch = cholmod_ones(static_cast<std::size_t>(n), 1, CHOLMOD_REAL, &c);
        cholmod_dense *x_ch = cholmod_solve(CHOLMOD_A, L, b_ch, &c);
        auto tf2 = Clock::now();
        total_factor += elapsed_ms(tf0, tf1);
        total_solve  += elapsed_ms(tf1, tf2);
        if (x_ch) cholmod_free_dense(&x_ch, &c);
        cholmod_free_dense(&b_ch, &c);
    }
    t.avg_factor_ms = total_factor / static_cast<double>(reps);
    t.avg_solve_ms  = total_solve  / static_cast<double>(reps);

    cholmod_free_factor(&L, &c);
    cholmod_free_sparse(&Ac, &c);
    cholmod_finish(&c);
    return (c.status == CHOLMOD_OK || reps > 0);
}
#endif // SMF_HAS_CHOLMOD

// ---------------------------------------------------------------------------
// MA27 SPD solve (double precision, via CoinHSL): return relative residual
// ---------------------------------------------------------------------------
#ifdef SMF_HAS_MA27
struct Ma27Times
{
    double analyse_ms = 0.0;
    double factor_ms = 0.0;
    double solve_ms = 0.0;
    double total_ms() const { return analyse_ms + factor_ms + solve_ms; }
};

static double ma27_run(const smf::CscLower &A, std::vector<double> &x_out, Ma27Times &t)
{
    int n = static_cast<int>(A.n);
    int nnz = static_cast<int>(A.nnz());
    x_out.assign(static_cast<std::size_t>(n), 1.0);

    // Build coordinate format (1-indexed, lower triangle) for MA27
    std::vector<int> irn(static_cast<std::size_t>(nnz));
    std::vector<int> icn(static_cast<std::size_t>(nnz));
    std::vector<double> a_in(static_cast<std::size_t>(nnz));
    {
        int p = 0;
        for (int j = 0; j < n; ++j)
        {
            for (smf::Int k = A.col_ptr[static_cast<std::size_t>(j)]; k < A.col_ptr[static_cast<std::size_t>(j) + 1];
                 ++k, ++p)
            {
                irn[static_cast<std::size_t>(p)] = static_cast<int>(A.row_idx[static_cast<std::size_t>(k)]) + 1;
                icn[static_cast<std::size_t>(p)] = j + 1;
                a_in[static_cast<std::size_t>(p)] = A.values[static_cast<std::size_t>(k)];
            }
        }
    }

    // Initialise MA27 controls (suppress all output)
    int icntl[30];
    double cntl[5];
    ma27id_(icntl, cntl);
    icntl[0] = 0;  // error output unit (0 = suppress)
    icntl[1] = 0;  // warning output unit (0 = suppress)
    cntl[0] = 0.0; // threshold pivoting = 0 → pure Cholesky for SPD matrices

    // --- Phase 1: symbolic analysis ---
    int liw1 = 2 * nnz + 3 * n + 1;
    std::vector<int> iw(static_cast<std::size_t>(liw1));
    std::vector<int> ikeep(static_cast<std::size_t>(3 * n));
    std::vector<int> iw1_buf(static_cast<std::size_t>(2 * n));
    int nsteps = 0, iflag = 0;
    int info[20] = {};
    double ops = 0.0;

    auto ta = Clock::now();
    ma27ad_(&n, &nnz, irn.data(), icn.data(), iw.data(), &liw1, ikeep.data(), iw1_buf.data(), &nsteps, &iflag, icntl,
            cntl, info, &ops);
    auto tb = Clock::now();
    if (info[0] != 0)
        return -1.0;

    // --- Phase 2: numeric factorisation ---
    // info[3] = Fortran INFO(4) = minimum LA; info[4] = minimum LIW
    // Use generous factor to absorb any pivoting fill-in beyond the symbolic estimate.
    int la = std::max(4 * nnz, static_cast<int>(static_cast<double>(info[3]) * 3.0)) + 2 * n + 200;
    int liw2 = std::max(4 * nnz, static_cast<int>(static_cast<double>(info[4]) * 3.0)) + 2 * n + 200;
    if (liw2 > liw1)
        iw.resize(static_cast<std::size_t>(liw2));

    std::vector<double> a_fac(static_cast<std::size_t>(la));
    for (int k = 0; k < nnz; ++k)
        a_fac[static_cast<std::size_t>(k)] = a_in[static_cast<std::size_t>(k)];

    int maxfrt = 0;
    // Note: CoinHSL's MA27BD uses IW1 as a work stack up to n+nsteps entries;
    // allocate generously to avoid overflow (standard docs say nsteps, but
    // this build needs more).
    std::vector<int> iw1_bd(static_cast<std::size_t>(2 * n + nsteps + 100));

    ma27bd_(&n, &nnz, irn.data(), icn.data(), a_fac.data(), &la, iw.data(), &liw2, ikeep.data(), &nsteps, &maxfrt,
            iw1_bd.data(), icntl, cntl, info);
    auto tc = Clock::now();
    if (info[0] != 0)
        return -1.0;

    // --- Phase 3: solve ---
    std::vector<double> w(static_cast<std::size_t>(maxfrt > 0 ? maxfrt : 1));
    std::vector<double> rhs(x_out.begin(), x_out.end()); // ones
    std::vector<int> iw1_cd(static_cast<std::size_t>(2 * n + nsteps + 100));

    ma27cd_(&n, a_fac.data(), &la, iw.data(), &liw2, w.data(), &maxfrt, rhs.data(), iw1_cd.data(), &nsteps, icntl,
            info);
    auto td = Clock::now();
    if (info[0] != 0)
        return -1.0;

    x_out.assign(rhs.begin(), rhs.end());

    t.analyse_ms = elapsed_ms(ta, tb);
    t.factor_ms  = elapsed_ms(tb, tc);
    t.solve_ms   = elapsed_ms(tc, td);

    std::vector<double> Ax_vec(static_cast<std::size_t>(n), 0.0);
    spmv_sym(A, x_out.data(), Ax_vec.data());
    double res = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double r = Ax_vec[static_cast<std::size_t>(i)] - 1.0;
        res += r * r;
    }
    return std::sqrt(res) / std::sqrt(static_cast<double>(n));
}

/// MA27 repeated factorisation: ma27ad_ once, then (ma27bd_+ma27cd_) reps times.
/// a[] is restored from a_in before each ma27bd_ call (MA27 overwrites a[] in place).
static bool ma27_repeated_run(const smf::CscLower &A, int reps, RepeatedTimes &t)
{
    int n   = static_cast<int>(A.n);
    int nnz = static_cast<int>(A.nnz());

    // Build 1-indexed COO
    std::vector<int>    irn(static_cast<std::size_t>(nnz));
    std::vector<int>    icn(static_cast<std::size_t>(nnz));
    std::vector<double> a_in(static_cast<std::size_t>(nnz));
    {
        int p = 0;
        for (int j = 0; j < n; ++j)
            for (smf::Int k = A.col_ptr[static_cast<std::size_t>(j)];
                 k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k, ++p)
            {
                irn[static_cast<std::size_t>(p)] =
                    static_cast<int>(A.row_idx[static_cast<std::size_t>(k)]) + 1;
                icn[static_cast<std::size_t>(p)] = j + 1;
                a_in[static_cast<std::size_t>(p)] = A.values[static_cast<std::size_t>(k)];
            }
    }

    int icntl[30]; double cntl[5];
    ma27id_(icntl, cntl);
    icntl[0] = 0; icntl[1] = 0; cntl[0] = 0.0;

    // Phase 1: symbolic analysis (once)
    int liw1 = 2 * nnz + 3 * n + 1;
    std::vector<int> iw(static_cast<std::size_t>(liw1));
    std::vector<int> ikeep(static_cast<std::size_t>(3 * n));
    std::vector<int> iw1_buf(static_cast<std::size_t>(2 * n));
    int nsteps = 0, iflag = 0, info[20] = {};
    double ops = 0.0;

    auto ta = Clock::now();
    ma27ad_(&n, &nnz, irn.data(), icn.data(), iw.data(), &liw1, ikeep.data(),
            iw1_buf.data(), &nsteps, &iflag, icntl, cntl, info, &ops);
    auto tb = Clock::now();
    if (info[0] != 0) return false;
    t.analyse_ms = elapsed_ms(ta, tb);

    int la   = std::max(4 * nnz, static_cast<int>(static_cast<double>(info[3]) * 3.0)) + 2 * n + 200;
    int liw2 = std::max(4 * nnz, static_cast<int>(static_cast<double>(info[4]) * 3.0)) + 2 * n + 200;
    if (liw2 > liw1) iw.resize(static_cast<std::size_t>(liw2));

    std::vector<double> a_fac(static_cast<std::size_t>(la));
    std::vector<int>    iw1_bd(static_cast<std::size_t>(2 * n + nsteps + 100));
    std::vector<int>    iw1_cd(static_cast<std::size_t>(2 * n + nsteps + 100));

    double total_factor = 0.0, total_solve = 0.0;
    for (int rep = 0; rep < reps; ++rep)
    {
        // Restore original values (MA27 overwrites a_fac during factorisation)
        for (int k = 0; k < nnz; ++k)
            a_fac[static_cast<std::size_t>(k)] = a_in[static_cast<std::size_t>(k)];

        int maxfrt = 0;
        auto tf0 = Clock::now();
        ma27bd_(&n, &nnz, irn.data(), icn.data(), a_fac.data(), &la, iw.data(), &liw2,
                ikeep.data(), &nsteps, &maxfrt, iw1_bd.data(), icntl, cntl, info);
        auto tf1 = Clock::now();
        if (info[0] != 0) return false;

        std::vector<double> w(static_cast<std::size_t>(maxfrt > 0 ? maxfrt : 1));
        std::vector<double> rhs(static_cast<std::size_t>(n), 1.0);
        ma27cd_(&n, a_fac.data(), &la, iw.data(), &liw2, w.data(), &maxfrt,
                rhs.data(), iw1_cd.data(), &nsteps, icntl, info);
        auto tf2 = Clock::now();
        if (info[0] != 0) return false;

        total_factor += elapsed_ms(tf0, tf1);
        total_solve  += elapsed_ms(tf1, tf2);
    }
    t.avg_factor_ms = total_factor / static_cast<double>(reps);
    t.avg_solve_ms  = total_solve  / static_cast<double>(reps);
    return true;
}
#endif // SMF_HAS_MA27

// ---------------------------------------------------------------------------
// MUMPS SPD solve (sequential, double precision): return relative residual
// ---------------------------------------------------------------------------
#ifdef SMF_HAS_MUMPS
struct MumpsTimes
{
    double analyse_ms = 0.0;
    double factor_ms = 0.0;
    double solve_ms = 0.0;
    double total_ms() const { return analyse_ms + factor_ms + solve_ms; }
};

static double mumps_run(const smf::CscLower &A, std::vector<double> &x_out, MumpsTimes &t)
{
    int n = static_cast<int>(A.n);
    int nnz = static_cast<int>(A.nnz());
    x_out.assign(static_cast<std::size_t>(n), 1.0);

    // Build 1-indexed COO (lower triangle) for MUMPS assembled format
    std::vector<int> irn(static_cast<std::size_t>(nnz));
    std::vector<int> jcn(static_cast<std::size_t>(nnz));
    std::vector<double> a_in(static_cast<std::size_t>(nnz));
    {
        int p = 0;
        for (int j = 0; j < n; ++j)
            for (smf::Int k = A.col_ptr[static_cast<std::size_t>(j)]; k < A.col_ptr[static_cast<std::size_t>(j) + 1];
                 ++k, ++p)
            {
                irn[static_cast<std::size_t>(p)] = static_cast<int>(A.row_idx[static_cast<std::size_t>(k)]) + 1;
                jcn[static_cast<std::size_t>(p)] = j + 1;
                a_in[static_cast<std::size_t>(p)] = A.values[static_cast<std::size_t>(k)];
            }
    }

    std::vector<double> rhs(static_cast<std::size_t>(n), 1.0); // b = ones

    DMUMPS_STRUC_C id;
    std::memset(&id, 0, sizeof(id));
    id.sym = 1; // SPD
    id.par = 1; // host is a worker (required for sequential)
    id.comm_fortran = USE_COMM_WORLD;

    // Initialize
    id.job = -1;
    dmumps_c(&id);

    // Suppress all MUMPS output
    id.icntl[0] = -1; // ICNTL(1): errors
    id.icntl[1] = -1; // ICNTL(2): warnings / diagnostics
    id.icntl[2] = -1; // ICNTL(3): global statistics
    id.icntl[3] = 0;  // ICNTL(4): printing level (0 = none)

    id.n = n;
    id.nz = nnz;
    id.irn = irn.data();
    id.jcn = jcn.data();
    id.a = a_in.data();
    id.nrhs = 1;
    id.lrhs = n;
    id.rhs = rhs.data();

    // Phase 1: analysis
    auto ta = Clock::now();
    id.job = 1;
    dmumps_c(&id);
    auto tb = Clock::now();
    if (id.info[0] < 0)
    {
        id.job = -2;
        dmumps_c(&id);
        return -1.0;
    }

    // Phase 2: factorization
    id.job = 2;
    dmumps_c(&id);
    auto tc = Clock::now();
    if (id.info[0] < 0)
    {
        id.job = -2;
        dmumps_c(&id);
        return -1.0;
    }

    // Phase 3: solve (solution written into rhs)
    id.job = 3;
    dmumps_c(&id);
    auto td = Clock::now();
    int info_solve = id.info[0]; // save before finalize may overwrite

    // Finalize
    id.job = -2;
    dmumps_c(&id);

    if (info_solve < 0)
        return -1.0;

    t.analyse_ms = elapsed_ms(ta, tb);
    t.factor_ms = elapsed_ms(tb, tc);
    t.solve_ms = elapsed_ms(tc, td);

    x_out.assign(rhs.begin(), rhs.end());

    std::vector<double> Ax_vec(static_cast<std::size_t>(n), 0.0);
    spmv_sym(A, x_out.data(), Ax_vec.data());
    double res = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double r = Ax_vec[static_cast<std::size_t>(i)] - 1.0;
        res += r * r;
    }
    return std::sqrt(res) / std::sqrt(static_cast<double>(n));
}
#endif // SMF_HAS_MUMPS

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------
struct MatrixCase
{
    std::string name;
    smf::CscLower A;
};

static std::vector<MatrixCase> build_test_matrices()
{
    std::vector<MatrixCase> cases;
    cases.push_back({"Poisson2D_100", make_poisson2d(10)}); // N=100
    cases.push_back({"Poisson2D_400", make_poisson2d(20)}); // N=400
    cases.push_back({"Tridiag_500", make_tridiagonal(500)});
    cases.push_back({"BandedSPD_200", make_banded_spd(200, 5)});
    cases.push_back({"BlockDiag_300", make_block_diagonal_spd(300)});
    // Larger matrices – exercise BLAS-3 fronts and amortised-analyse advantage
    cases.push_back({"Poisson2D_1000", make_poisson2d(31)});    // N=961,  nnz≈3844
    cases.push_back({"Poisson2D_4000", make_poisson2d(63)});    // N=3969, nnz≈15876
    cases.push_back({"Tridiag_5000",   make_tridiagonal(5000)});
    cases.push_back({"BandedSPD_2000", make_banded_spd(2000, 10)});
    return cases;
}

/// Warmup/timing runs: fewer for large matrices to keep benchmark fast.
static int warmup_runs_for(int N) { return (N > 500) ? 2 : 3; }

int main()
{
    std::puts("=== bench_compare: smf vs CHOLMOD vs MA27 vs MUMPS ===");

#ifdef SMF_HAS_CHOLMOD
    std::puts("CHOLMOD : available");
#else
    std::puts("CHOLMOD : not compiled in");
#endif
#ifdef SMF_HAS_MA27
    std::puts("MA27    : available (CoinHSL)");
#else
    std::puts("MA27    : not compiled in");
#endif
#ifdef SMF_HAS_MUMPS
    std::puts("MUMPS   : available (sequential)");
#else
    std::puts("MUMPS   : not compiled in");
#endif
#ifdef SMF_HAS_EIGEN
    std::puts("Eigen   : available");
#else
    std::puts("Eigen   : not compiled in");
#endif
    std::puts("");

    auto cases = build_test_matrices();

    // Per-case smf phase breakdown (filled inside main loop)
    struct BreakdownRow
    {
        std::string name;
        int         N   = 0;
        SmfTimes    times;
    };
    std::vector<BreakdownRow> breakdown_rows;
    breakdown_rows.reserve(cases.size());

    // --- Print header ---
    std::printf("%-22s  %6s  %7s  %9s", "Matrix", "N", "nnz", "smf(ms)");
#ifdef SMF_HAS_CHOLMOD
    std::printf("  %9s", "chol(ms)");
#endif
#ifdef SMF_HAS_MA27
    std::printf("  %9s", "ma27(ms)");
#endif
#ifdef SMF_HAS_MUMPS
    std::printf("  %9s", "mps(ms)");
#endif
#ifdef SMF_HAS_EIGEN
    std::printf("  %9s", "eig(ms)");
#endif
    std::printf("  %10s", "smf_res");
#ifdef SMF_HAS_CHOLMOD
    std::printf("  %10s", "chol_res");
#endif
#ifdef SMF_HAS_MA27
    std::printf("  %10s", "ma27_res");
#endif
#ifdef SMF_HAS_MUMPS
    std::printf("  %10s", "mps_res");
#endif
#ifdef SMF_HAS_EIGEN
    std::printf("  %10s", "eig_res");
#endif
    std::puts("");

    // --- Separator ---
    std::printf("%-22s  %6s  %7s  %9s", "----------------------", "------", "-------", "---------");
#ifdef SMF_HAS_CHOLMOD
    std::printf("  %9s", "---------");
#endif
#ifdef SMF_HAS_MA27
    std::printf("  %9s", "---------");
#endif
#ifdef SMF_HAS_MUMPS
    std::printf("  %9s", "---------");
#endif
#ifdef SMF_HAS_EIGEN
    std::printf("  %9s", "---------");
#endif
    std::printf("  %10s", "----------");
#ifdef SMF_HAS_CHOLMOD
    std::printf("  %10s", "----------");
#endif
#ifdef SMF_HAS_MA27
    std::printf("  %10s", "----------");
#endif
#ifdef SMF_HAS_MUMPS
    std::printf("  %10s", "----------");
#endif
#ifdef SMF_HAS_EIGEN
    std::printf("  %10s", "----------");
#endif
    std::puts("");

    // --- Speedup accumulators ---
#ifdef SMF_HAS_CHOLMOD
    int chol_faster = 0;
    double chol_spd_sum = 0.0;
    int chol_valid = 0;
#endif
#ifdef SMF_HAS_MA27
    int ma27_faster = 0;
    double ma27_spd_sum = 0.0;
    int ma27_valid = 0;
#endif
#ifdef SMF_HAS_MUMPS
    int mps_faster = 0;
    double mps_spd_sum = 0.0;
    int mps_valid = 0;
#endif
#ifdef SMF_HAS_EIGEN
    int eig_faster = 0;
    double eig_spd_sum = 0.0;
    int eig_valid = 0;
#endif

    for (auto &mc : cases)
    {
        const smf::CscLower &A = mc.A;
        const int N = static_cast<int>(A.n);
        const int nnz = static_cast<int>(A.nnz());

        // smf
        double smf_best = 1e18, smf_res = -1.0;
        SmfTimes smf_best_times;
        std::vector<double> smf_x;
        for (int r = 0; r < warmup_runs_for(N); ++r)
        {
            SmfTimes t;
            double res = smf_run(A, smf_x, t);
            if (res >= 0.0 && t.total_ms() < smf_best)
            {
                smf_best       = t.total_ms();
                smf_res        = res;
                smf_best_times = t;
            }
        }
        breakdown_rows.push_back({mc.name, N, smf_best_times});

#ifdef SMF_HAS_CHOLMOD
        double chol_best = 1e18, chol_res = -1.0;
        std::vector<double> chol_x;
        for (int r = 0; r < warmup_runs_for(N); ++r)
        {
            CholmodTimes t;
            double res = cholmod_run(A, chol_x, t);
            if (res >= 0.0 && t.total_ms() < chol_best)
            {
                chol_best = t.total_ms();
                chol_res = res;
            }
        }
        if (smf_res >= 0.0 && chol_res >= 0.0)
        {
            double sp = (smf_best > 0.0) ? chol_best / smf_best : 0.0;
            chol_spd_sum += sp;
            ++chol_valid;
            if (sp >= 1.0)
                ++chol_faster;
        }
#endif

#ifdef SMF_HAS_MA27
        double ma27_best = 1e18, ma27_res = -1.0;
        std::vector<double> ma27_x;
        for (int r = 0; r < warmup_runs_for(N); ++r)
        {
            Ma27Times t;
            double res = ma27_run(A, ma27_x, t);
            if (res >= 0.0 && t.total_ms() < ma27_best)
            {
                ma27_best = t.total_ms();
                ma27_res = res;
            }
        }
        if (smf_res >= 0.0 && ma27_res >= 0.0)
        {
            double sp = (smf_best > 0.0) ? ma27_best / smf_best : 0.0;
            ma27_spd_sum += sp;
            ++ma27_valid;
            if (sp >= 1.0)
                ++ma27_faster;
        }
#endif

#ifdef SMF_HAS_MUMPS
        double mps_best = 1e18, mps_res = -1.0;
        std::vector<double> mps_x;
        for (int r = 0; r < warmup_runs_for(N); ++r)
        {
            MumpsTimes t;
            double res = mumps_run(A, mps_x, t);
            if (res >= 0.0 && t.total_ms() < mps_best)
            {
                mps_best = t.total_ms();
                mps_res = res;
            }
        }
        if (smf_res >= 0.0 && mps_res >= 0.0)
        {
            double sp = (smf_best > 0.0) ? mps_best / smf_best : 0.0;
            mps_spd_sum += sp;
            ++mps_valid;
            if (sp >= 1.0)
                ++mps_faster;
        }
#endif

#ifdef SMF_HAS_EIGEN
        Eigen::SparseMatrix<double> M = to_eigen_sym(A);
        double eig_best = 1e18, eig_res = -1.0;
        Eigen::VectorXd eig_x;
        for (int r = 0; r < warmup_runs_for(N); ++r)
        {
            double t_ms = 0.0;
            double res = eigen_run(M, eig_x, t_ms);
            if (res >= 0.0 && t_ms < eig_best)
            {
                eig_best = t_ms;
                eig_res = res;
            }
        }
        if (smf_res >= 0.0 && eig_res >= 0.0)
        {
            double sp = (smf_best > 0.0) ? eig_best / smf_best : 0.0;
            eig_spd_sum += sp;
            ++eig_valid;
            if (sp >= 1.0)
                ++eig_faster;
        }
#endif

        // Row output
        std::printf("%-22s  %6d  %7d  %9.3f", mc.name.c_str(), N, nnz, smf_best);
#ifdef SMF_HAS_CHOLMOD
        if (chol_res >= 0.0)
            std::printf("  %9.3f", chol_best);
        else
            std::printf("  %9s", "FAILED");
#endif
#ifdef SMF_HAS_MA27
        if (ma27_res >= 0.0)
            std::printf("  %9.3f", ma27_best);
        else
            std::printf("  %9s", "FAILED");
#endif
#ifdef SMF_HAS_MUMPS
        if (mps_res >= 0.0)
            std::printf("  %9.3f", mps_best);
        else
            std::printf("  %9s", "FAILED");
#endif
#ifdef SMF_HAS_EIGEN
        if (eig_res >= 0.0)
            std::printf("  %9.3f", eig_best);
        else
            std::printf("  %9s", "FAILED");
#endif
        std::printf("  %10.3e", smf_res >= 0.0 ? smf_res : -1.0);
#ifdef SMF_HAS_CHOLMOD
        if (chol_res >= 0.0)
            std::printf("  %10.3e", chol_res);
        else
            std::printf("  %10s", "FAILED");
#endif
#ifdef SMF_HAS_MA27
        if (ma27_res >= 0.0)
            std::printf("  %10.3e", ma27_res);
        else
            std::printf("  %10s", "FAILED");
#endif
#ifdef SMF_HAS_MUMPS
        if (mps_res >= 0.0)
            std::printf("  %10.3e", mps_res);
        else
            std::printf("  %10s", "FAILED");
#endif
#ifdef SMF_HAS_EIGEN
        if (eig_res >= 0.0)
            std::printf("  %10.3e", eig_res);
        else
            std::printf("  %10s", "FAILED");
#endif
        std::puts("");
    }

    // --- Summary ---
    std::puts("");
    std::printf("Speedup = competitor_time / smf_time  (>1 means smf is faster)\n");
#ifdef SMF_HAS_CHOLMOD
    std::printf("  smf vs CHOLMOD : smf faster in %d/%d cases, avg speedup %.2fx\n", chol_faster, chol_valid,
                chol_valid > 0 ? chol_spd_sum / static_cast<double>(chol_valid) : 0.0);
#endif
#ifdef SMF_HAS_MA27
    std::printf("  smf vs MA27    : smf faster in %d/%d cases, avg speedup %.2fx\n", ma27_faster, ma27_valid,
                ma27_valid > 0 ? ma27_spd_sum / static_cast<double>(ma27_valid) : 0.0);
#endif
#ifdef SMF_HAS_MUMPS
    std::printf("  smf vs MUMPS   : smf faster in %d/%d cases, avg speedup %.2fx\n", mps_faster, mps_valid,
                mps_valid > 0 ? mps_spd_sum / static_cast<double>(mps_valid) : 0.0);
#endif
#ifdef SMF_HAS_EIGEN
    std::printf("  smf vs Eigen   : smf faster in %d/%d cases, avg speedup %.2fx\n", eig_faster, eig_valid,
                eig_valid > 0 ? eig_spd_sum / static_cast<double>(eig_valid) : 0.0);
#endif

    // =========================================================================
    // smf Phase Breakdown
    // =========================================================================
    std::puts("\n=== smf Phase Breakdown ===");
    std::printf("%-22s  %6s  %12s  %11s  %10s\n",
                "Matrix", "N", "analyse(ms)", "factor(ms)", "solve(ms)");
    std::printf("%-22s  %6s  %12s  %11s  %10s\n",
                "----------------------", "------", "------------", "-----------", "----------");
    for (const auto &br : breakdown_rows)
    {
        std::printf("%-22s  %6d  %12.3f  %11.3f  %10.3f\n",
                    br.name.c_str(), br.N,
                    br.times.analyse_ms, br.times.factor_ms, br.times.solve_ms);
    }

    // =========================================================================
    // Repeated Factorisation (analyse×1 + factor+solve×10)
    // =========================================================================
    static constexpr int REPEAT_REPS = 10;
    std::puts("\n=== Repeated Factorization (analyse x1 + factor+solve x10) ===");
    std::printf("Note: amortized = analyse/10 + avg_factor + avg_solve\n\n");

    // Header
    std::printf("%-22s  %6s  %7s  %10s",
                "Matrix", "N", "nnz", "smf_amort");
#ifdef SMF_HAS_CHOLMOD
    std::printf("  %10s", "chol_amort");
#endif
#ifdef SMF_HAS_MA27
    std::printf("  %10s", "ma27_amort");
#endif
#ifdef SMF_HAS_EIGEN
    std::printf("  %10s", "eig_amort");
#endif
    std::printf("  %9s  %9s  %9s\n", "smf_a_ms", "smf_f_ms", "smf_s_ms");

    // Separator
    std::printf("%-22s  %6s  %7s  %10s",
                "----------------------", "------", "-------", "----------");
#ifdef SMF_HAS_CHOLMOD
    std::printf("  %10s", "----------");
#endif
#ifdef SMF_HAS_MA27
    std::printf("  %10s", "----------");
#endif
#ifdef SMF_HAS_EIGEN
    std::printf("  %10s", "----------");
#endif
    std::printf("  %9s  %9s  %9s\n", "---------", "---------", "---------");

    for (const auto &mc : cases)
    {
        const smf::CscLower &A  = mc.A;
        const int            N  = static_cast<int>(A.n);
        const int            nz = static_cast<int>(A.nnz());

        RepeatedTimes smf_rt;
        bool smf_ok = smf_repeated_run(A, REPEAT_REPS, smf_rt);

#ifdef SMF_HAS_CHOLMOD
        RepeatedTimes chol_rt;
        bool chol_ok = cholmod_repeated_run(A, REPEAT_REPS, chol_rt);
#endif
#ifdef SMF_HAS_MA27
        RepeatedTimes ma27_rt;
        bool ma27_ok = ma27_repeated_run(A, REPEAT_REPS, ma27_rt);
#endif
#ifdef SMF_HAS_EIGEN
        Eigen::SparseMatrix<double> M_rep = to_eigen_sym(A);
        RepeatedTimes eig_rt;
        bool eig_ok = eigen_repeated_run(M_rep, REPEAT_REPS, eig_rt);
#endif

        // smf amortized
        std::printf("%-22s  %6d  %7d", mc.name.c_str(), N, nz);
        if (smf_ok)
            std::printf("  %10.3f", smf_rt.amortized_ms(REPEAT_REPS));
        else
            std::printf("  %10s", "FAILED");

#ifdef SMF_HAS_CHOLMOD
        if (chol_ok)
            std::printf("  %10.3f", chol_rt.amortized_ms(REPEAT_REPS));
        else
            std::printf("  %10s", "FAILED");
#endif
#ifdef SMF_HAS_MA27
        if (ma27_ok)
            std::printf("  %10.3f", ma27_rt.amortized_ms(REPEAT_REPS));
        else
            std::printf("  %10s", "FAILED");
#endif
#ifdef SMF_HAS_EIGEN
        if (eig_ok)
            std::printf("  %10.3f", eig_rt.amortized_ms(REPEAT_REPS));
        else
            std::printf("  %10s", "FAILED");
#endif
        // smf split columns
        if (smf_ok)
            std::printf("  %9.3f  %9.3f  %9.3f",
                        smf_rt.analyse_ms, smf_rt.avg_factor_ms, smf_rt.avg_solve_ms);
        else
            std::printf("  %9s  %9s  %9s", "n/a", "n/a", "n/a");
        std::puts("");
    }

    return 0;
}
