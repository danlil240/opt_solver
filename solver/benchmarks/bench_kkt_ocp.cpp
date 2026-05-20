// bench_kkt_ocp.cpp — KKT saddle-point matrix benchmark for smf
// Part of the smf MA97-class solver benchmark suite.
//
// Section 1 (original):
//   Small KKT / saddle-point matrix:
//     [ H   A^T ]   where H = diag(1..nz), A is banded (nc x nz)
//     [ A    0  ]
//   with nz=10 (primal vars), nc=5 (constraints), total N=15.
//
// Section 2 (trajectory-optimization KKT):
//   Block-tridiagonal Lagrangian matrix for a linear-quadratic OCP:
//     Variables: x_0, u_0, x_1, u_1, ..., u_{N-1}, x_N  (interleaved layout)
//     Size:      nx*(N+1) + nu*N  (with N=50, nx=6, nu=3 → 456)
//     Structure: block-tridiagonal SPD Hessian of the Lagrangian
//   Times analyse + 10 factor+solve iterations and reports per-operation cost.

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
#include <string>
#include <vector>

// --------------------------------------------------------------------------
// Utility: peak memory from /proc/self/status (Linux only)
// --------------------------------------------------------------------------
static long read_vm_peak_kb()
{
    std::ifstream ifs("/proc/self/status");
    if (!ifs.is_open())
        return -1L;
    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.rfind("VmPeak:", 0) == 0)
        {
            const char* p = line.c_str() + 7;
            while (*p == ' ' || *p == '\t')
                ++p;
            return std::atol(p);
        }
    }
    return -1L;
}

// --------------------------------------------------------------------------
// Utility: SpMV for residual — symmetric lower-triangular CSC
// --------------------------------------------------------------------------
static void spmv_sym_lower(const smf::CscLower &A, const double* x, double* y)
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

// --------------------------------------------------------------------------
// Build lower-CSC of the small KKT matrix.
//   Full N×N matrix (N = nz+nc) stored in lower triangle:
//     Col j in [0, nz): H diagonal + A coupling rows
//     Col j in [nz, nz+nc): Schur block regularization -eps*I
// --------------------------------------------------------------------------
static smf::CscLower build_kkt(int nz, int nc)
{
    const int N = nz + nc;

    std::vector<std::vector<std::pair<smf::Int, double>>> cols(static_cast<std::size_t>(N));

    // H diagonal
    for (int j = 0; j < nz; ++j)
    {
        cols[static_cast<std::size_t>(j)].emplace_back(static_cast<smf::Int>(j), static_cast<double>(j + 1));
    }

    // A block: A[i,j] entries become (nz+i, j) in KKT lower triangle
    for (int i = 0; i < nc; ++i)
    {
        if (i < nz)
        {
            cols[static_cast<std::size_t>(i)].emplace_back(static_cast<smf::Int>(nz + i), 1.0);
        }
        if (i + 1 < nz)
        {
            cols[static_cast<std::size_t>(i + 1)].emplace_back(static_cast<smf::Int>(nz + i), 1.0);
        }
    }

    // Schur block regularization: small negative diagonal entries
    constexpr double schur_reg = -1.0e-6;
    for (int i = 0; i < nc; ++i)
    {
        cols[static_cast<std::size_t>(nz + i)].emplace_back(static_cast<smf::Int>(nz + i), schur_reg);
    }

    smf::CscLower A;
    A.n = static_cast<smf::Int>(N);
    A.col_ptr.resize(static_cast<std::size_t>(N) + 1);
    A.col_ptr[0] = 0;
    for (int j = 0; j < N; ++j)
    {
        auto &c = cols[static_cast<std::size_t>(j)];
        std::sort(c.begin(), c.end(), [](const std::pair<smf::Int, double> &a, const std::pair<smf::Int, double> &b)
                  { return a.first < b.first; });
        A.col_ptr[static_cast<std::size_t>(j) + 1] =
            A.col_ptr[static_cast<std::size_t>(j)] + static_cast<smf::Int>(c.size());
    }
    smf::Int nnz = A.col_ptr[static_cast<std::size_t>(N)];
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));

    smf::Int pos = 0;
    for (int j = 0; j < N; ++j)
    {
        for (auto &e : cols[static_cast<std::size_t>(j)])
        {
            A.row_idx[static_cast<std::size_t>(pos)] = e.first;
            A.values[static_cast<std::size_t>(pos)] = e.second;
            ++pos;
        }
    }
    return A;
}

// --------------------------------------------------------------------------
// Build trajectory-optimization KKT Hessian (block-tridiagonal SPD).
//
// Problem: discrete-time LQR with N steps, nx states, nu inputs.
//
// Variable ordering (INTERLEAVED, adjacent layout to avoid reordering bugs):
//   z = [ x_0, u_0, x_1, u_1, ..., u_{N-1}, x_N ]
//   Size: n = nx*(N+1) + nu*N
//
// Hessian of the Lagrangian (positive definite for unconstrained LQR):
//
//   Diagonal blocks:
//     For x_k (k=0..N-1): Q_k = q_scale * I_nx  (state cost)
//     For u_k (k=0..N-1): R_k = r_scale * I_nu  (control cost)
//     For x_N:            Q_N = qN_scale * I_nx  (terminal state cost)
//
//   Off-diagonal coupling (linearized dynamics x_{k+1} = A*x_k + B*u_k):
//     Between x_k  and u_k  : coupling block S_{xu} = s_xu * I (min(nx,nu))
//     Between u_k  and x_{k+1}: coupling block S_{ux1} = s_ux * I (min(nu,nx))
//
//   The Hessian is made diagonally dominant to ensure positive definiteness.
//
// Only the lower triangle of the full N×N matrix is stored (CscLower).
// --------------------------------------------------------------------------
static smf::CscLower build_traj_opt_hessian(int N_steps, int nx, int nu)
{
    const int nz = nx * (N_steps + 1) + nu * N_steps; // total variables
    const double q = 1.0;                             // state cost diagonal
    const double r = 10.0;                            // control cost diagonal (larger → more stable)
    const double qN = 5.0;                            // terminal state cost
    const double s = 0.1;                             // coupling magnitude (small → diag dominant)

    // Build COO lower-triangle entries
    std::vector<std::pair<smf::Int, smf::Int>> rc_pairs;
    std::vector<double> vals;

    // Reserve conservative upper bound: nz diagonal + coupling
    rc_pairs.reserve(static_cast<std::size_t>(3 * nz));
    vals.reserve(static_cast<std::size_t>(3 * nz));

    // Column base offsets
    // For step k (k = 0..N_steps-1):
    //   x_k starts at col_x[k] = k*(nx+nu)
    //   u_k starts at col_u[k] = k*(nx+nu) + nx
    // x_N starts at col_x[N_steps] = N_steps*(nx+nu)

    auto col_x = [&](int k) { return k * (nx + nu); };
    auto col_u = [&](int k) { return k * (nx + nu) + nx; };
    // x_N
    int col_xN = N_steps * (nx + nu);

    auto add_entry = [&](int row, int col, double val)
    {
        // Only lower triangle (row >= col)
        if (row >= col)
        {
            rc_pairs.push_back({static_cast<smf::Int>(row), static_cast<smf::Int>(col)});
            vals.push_back(val);
        }
    };

    for (int k = 0; k < N_steps; ++k)
    {
        int xk_base = col_x(k);
        int uk_base = col_u(k);

        // Q_k: diagonal block for x_k
        for (int i = 0; i < nx; ++i)
        {
            add_entry(xk_base + i, xk_base + i, q);
        }

        // R_k: diagonal block for u_k
        for (int i = 0; i < nu; ++i)
        {
            add_entry(uk_base + i, uk_base + i, r);
        }

        // S_{xu}: coupling between x_k and u_k (lower triangle: u_k > x_k)
        int lim_xu = (nx < nu) ? nx : nu;
        for (int i = 0; i < lim_xu; ++i)
        {
            // u_k[i] row, x_k[i] col  →  row = uk_base+i, col = xk_base+i
            add_entry(uk_base + i, xk_base + i, s);
        }

        // S_{ux1}: coupling between u_k and x_{k+1}
        int xk1_base = col_x(k + 1);
        int lim_ux1 = (nu < nx) ? nu : nx;
        for (int i = 0; i < lim_ux1; ++i)
        {
            // x_{k+1}[i] row, u_k[i] col  →  row = xk1_base+i, col = uk_base+i
            add_entry(xk1_base + i, uk_base + i, s);
        }
    }

    // Q_N: diagonal block for terminal x_N
    for (int i = 0; i < nx; ++i)
    {
        add_entry(col_xN + i, col_xN + i, qN);
    }

    // Sort COO by (col, row) and build CSC
    // Build index array for sorting
    std::vector<int> order(rc_pairs.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b)
              {
                  if (rc_pairs[a].second != rc_pairs[b].second)
                      return rc_pairs[a].second < rc_pairs[b].second;
                  return rc_pairs[a].first < rc_pairs[b].first;
              });

    smf::CscLower A;
    A.n = static_cast<smf::Int>(nz);
    A.col_ptr.resize(static_cast<std::size_t>(nz) + 1, 0);

    for (int idx : order)
    {
        int c = static_cast<int>(rc_pairs[idx].second);
        A.col_ptr[static_cast<std::size_t>(c) + 1]++;
    }
    for (int j = 0; j < nz; ++j)
    {
        A.col_ptr[static_cast<std::size_t>(j) + 1] += A.col_ptr[static_cast<std::size_t>(j)];
    }

    smf::Int nnz_val = A.col_ptr[static_cast<std::size_t>(nz)];
    A.row_idx.resize(static_cast<std::size_t>(nnz_val));
    A.values.resize(static_cast<std::size_t>(nnz_val));

    std::vector<smf::Int> pos(static_cast<std::size_t>(nz), 0);
    for (int idx : order)
    {
        int c = static_cast<int>(rc_pairs[idx].second);
        int r = static_cast<int>(rc_pairs[idx].first);
        smf::Int base = A.col_ptr[static_cast<std::size_t>(c)];
        smf::Int off = pos[static_cast<std::size_t>(c)];
        A.row_idx[static_cast<std::size_t>(base + off)] = static_cast<smf::Int>(r);
        A.values[static_cast<std::size_t>(base + off)] = vals[static_cast<std::size_t>(idx)];
        ++pos[static_cast<std::size_t>(c)];
    }
    return A;
}

// --------------------------------------------------------------------------
// Run traj-opt benchmark: analyse once, factor+solve N_iter times
// --------------------------------------------------------------------------
static int bench_traj_opt(int N_steps, int nx, int nu, int N_iter)
{
    const int n = nx * (N_steps + 1) + nu * N_steps;

    std::printf("\n=== Trajectory-Optimization KKT Hessian ===\n");
    std::printf("N_steps=%d  nx=%d  nu=%d  matrix_size=%d  nnz=?  iterations=%d\n", N_steps, nx, nu, n, N_iter);

    smf::CscLower A = build_traj_opt_hessian(N_steps, nx, nu);
    std::printf("Matrix: n=%d  nnz=%d (lower triangular)\n", n, static_cast<int>(A.nnz()));

    std::vector<double> b(static_cast<std::size_t>(n), 1.0);
    std::vector<double> b_orig(b);
    std::vector<double> b_work(static_cast<std::size_t>(n));

    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricPositiveDefinite;
    smf::Info info;
    smf::Solver solver;

    // Analyse (once)
    auto t_a0 = std::chrono::high_resolution_clock::now();
    auto ak = solver.analyse(A, ctrl, info);
    auto t_a1 = std::chrono::high_resolution_clock::now();
    if (!ak)
    {
        std::fprintf(stderr, "bench_traj_opt: analyse() failed\n");
        return 1;
    }
    double t_analyse = std::chrono::duration<double, std::milli>(t_a1 - t_a0).count();

    // Factor + solve repeated N_iter times
    double t_factor_total = 0.0;
    double t_solve_total = 0.0;
    double residual_last = 0.0;
    smf::FactorStatus fs_last = smf::FactorStatus::Success;

    for (int iter = 0; iter < N_iter; ++iter)
    {
        smf::FactorKeep fkeep;
        auto t_f0 = std::chrono::high_resolution_clock::now();
        smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fkeep);
        auto t_f1 = std::chrono::high_resolution_clock::now();
        fs_last = fs;
        if (fs != smf::FactorStatus::Success)
        {
            std::fprintf(stderr, "bench_traj_opt: factor() iter=%d status=%d\n", iter, static_cast<int>(fs));
            return 1;
        }
        t_factor_total += std::chrono::duration<double, std::milli>(t_f1 - t_f0).count();

        // Restore RHS
        b_work = b_orig;

        auto t_s0 = std::chrono::high_resolution_clock::now();
        int rc = solver.solve(fkeep, ctrl, info, b_work.data(), n, 1);
        auto t_s1 = std::chrono::high_resolution_clock::now();
        if (rc != 0)
        {
            std::fprintf(stderr, "bench_traj_opt: solve() iter=%d rc=%d\n", iter, rc);
            return 1;
        }
        t_solve_total += std::chrono::duration<double, std::milli>(t_s1 - t_s0).count();

        // Residual (last iteration only)
        if (iter == N_iter - 1)
        {
            std::vector<double> ax(static_cast<std::size_t>(n));
            spmv_sym_lower(A, b_work.data(), ax.data());
            double num = 0.0, den = 0.0;
            for (int i = 0; i < n; ++i)
            {
                double r = ax[static_cast<std::size_t>(i)] - b_orig[static_cast<std::size_t>(i)];
                num += r * r;
                den += b_orig[static_cast<std::size_t>(i)] * b_orig[static_cast<std::size_t>(i)];
            }
            residual_last = std::sqrt(num) / std::max(1.0, std::sqrt(den));
        }
    }

    double t_factor_avg = t_factor_total / static_cast<double>(N_iter);
    double t_solve_avg = t_solve_total / static_cast<double>(N_iter);

    std::printf("Analyse      : %.3f ms\n", t_analyse);
    std::printf("Factor (avg) : %.3f ms  (total %.3f ms over %d iters)\n", t_factor_avg, t_factor_total, N_iter);
    std::printf("Solve (avg)  : %.3f ms  (total %.3f ms over %d iters)\n", t_solve_avg, t_solve_total, N_iter);
    std::printf("Residual     : %.3e\n", residual_last);
    std::printf("Inertia      : pos=%d neg=%d zero=%d\n", info.num_positive, info.num_negative, info.num_zero);
    (void)fs_last;
    return 0;
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
int main(int /*argc*/, char* /*argv*/[])
{
    // -----------------------------------------------------------------------
    // Section 1: small KKT saddle-point benchmark (original)
    // -----------------------------------------------------------------------
    constexpr int nz = 10;
    constexpr int nc = 5;
    const int N = nz + nc;

    std::printf("=== bench_kkt_ocp ===\n");
    std::printf("\n--- Section 1: Small KKT saddle-point (nz=%d, nc=%d, N=%d) ---\n", nz, nc, N);

    smf::CscLower A = build_kkt(nz, nc);
    std::printf("KKT: nz=%d nc=%d N=%d nnz=%d (lower triangular)\n", nz, nc, N, static_cast<int>(A.nnz()));

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
    if (!ak)
    {
        std::fprintf(stderr, "bench_kkt_ocp: analyse() failed\n");
        return 1;
    }
    double t_analyse = std::chrono::duration<double, std::milli>(t1 - t0).count();

    smf::FactorKeep fkeep;
    auto t2 = std::chrono::high_resolution_clock::now();
    smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fkeep);
    auto t3 = std::chrono::high_resolution_clock::now();
    if (fs != smf::FactorStatus::Success)
    {
        std::fprintf(stderr, "bench_kkt_ocp: factor() returned status %d\n", static_cast<int>(fs));
        return 1;
    }
    double t_factor = std::chrono::duration<double, std::milli>(t3 - t2).count();

    auto t4 = std::chrono::high_resolution_clock::now();
    int rc = solver.solve(fkeep, ctrl, info, b.data(), N, 1);
    auto t5 = std::chrono::high_resolution_clock::now();
    if (rc != 0)
    {
        std::fprintf(stderr, "bench_kkt_ocp: solve() returned %d\n", rc);
        return 1;
    }
    double t_solve = std::chrono::duration<double, std::milli>(t5 - t4).count();

    std::vector<double> ax(static_cast<std::size_t>(N));
    spmv_sym_lower(A, b.data(), ax.data());
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double r = ax[static_cast<std::size_t>(i)] - b_orig[static_cast<std::size_t>(i)];
        num += r * r;
        den += b_orig[static_cast<std::size_t>(i)] * b_orig[static_cast<std::size_t>(i)];
    }
    double residual = std::sqrt(num) / std::max(1.0, std::sqrt(den));

    std::printf("Analyse : %.3f ms\n", t_analyse);
    std::printf("Factor  : %.3f ms\n", t_factor);
    std::printf("Solve   : %.3f ms\n", t_solve);
    std::printf("Residual: %.3e\n", residual);
    std::printf("Inertia : pos=%d neg=%d zero=%d\n", info.num_positive, info.num_negative, info.num_zero);

    long vmpeakKb = read_vm_peak_kb();
    if (vmpeakKb >= 0)
        std::printf("Peak memory: %ld kB\n", vmpeakKb);
    else
        std::printf("Peak memory: N/A\n");

    // -----------------------------------------------------------------------
    // Section 2: Trajectory-optimization KKT Hessian benchmark
    // -----------------------------------------------------------------------
    constexpr int traj_N = 50;     // timesteps
    constexpr int traj_nx = 6;     // state dimension
    constexpr int traj_nu = 3;     // control dimension
    constexpr int traj_iters = 10; // factor+solve iterations

    int ret = bench_traj_opt(traj_N, traj_nx, traj_nu, traj_iters);

    vmpeakKb = read_vm_peak_kb();
    if (vmpeakKb >= 0)
        std::printf("Peak memory: %ld kB\n", vmpeakKb);
    else
        std::printf("Peak memory: N/A\n");

    return ret;
}
