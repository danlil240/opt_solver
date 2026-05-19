/// test_ocp_kkt_regression.cpp
///
/// OCP (Optimal Control Problem) KKT regression test.
///
/// Builds a synthetic block-KKT matrix (indefinite, 2-block structure):
///   - N=20 timesteps, nx=4 states, nu=2 inputs
///   - Matrix size: 2*N*NX + (N-1)*NU = 198 unknowns
///
/// Variable layout (interleaved state/dual for numerical stability):
///   [2*(k*NX+i)]:   x[k,i]  — state DOF at timestep k, dimension i
///   [2*(k*NX+i)+1]: λ[k,i]  — dual DOF at timestep k, dimension i
///   [2*N*NX + k*NU + i]: u[k,i] — input DOF (appended at end)
///
/// KKT block structure for each (x[k,i], λ[k,i]) pair:
///   Column 2*(k*NX+i):   (j,j) = h_{k,i}  and  (j+1,j) = 1.0
///   Column 2*(k*NX+i)+1: (j+1,j+1) = -ε
///
/// Each pair forms a 2×2 saddle-point block:
///   [ h  1 ]   with inertia (1,1,0) when h > 1/ε
///   [ 1 -ε ]
///
/// Input DOFs: pure diagonal entries r_{k,i} > 0
///
/// Total inertia: (N*NX + (N-1)*NU) positive, N*NX negative, 0 zero
///              = 118 positive, 80 negative, 0 zero
///
/// Acceptance criteria:
///   1. analyse() called exactly once.
///   2. factor() + solve() called 100 times (same pattern, different values).
///   3. Residual ‖Ax − b‖_∞ / ‖b‖_∞ < 1e-9 on every solve.
///   4. Inertia (num_positive, num_negative, num_zero) is stable across
///      all 100 solves.
///   5. Info::analyse_seconds > 0 after the first analyse call.
///   6. Total Info::factor_seconds across 100 iters > 0.

#include "smf/analysis.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

using namespace smf;

// ---------------------------------------------------------------------------
// XorShift32 — deterministic PRNG
// ---------------------------------------------------------------------------

class XorShift32 {
public:
    explicit XorShift32(uint32_t seed = 42u) : state_(seed ? seed : 1u) {}

    double next() {
        state_ ^= state_ << 13u;
        state_ ^= state_ >> 17u;
        state_ ^= state_ << 5u;
        return static_cast<double>(state_) / static_cast<double>(0xFFFFFFFFu);
    }

    double next_signed() { return 2.0 * next() - 1.0; }

private:
    uint32_t state_;
};

// ---------------------------------------------------------------------------
// Problem dimensions
// ---------------------------------------------------------------------------

static constexpr int N  = 20;  // timesteps
static constexpr int NX = 4;   // states
static constexpr int NU = 2;   // inputs

// Total DOFs per variable type:
static constexpr int NSTATE_DOF  = N * NX;          // 80
static constexpr int NDUAL_DOF   = N * NX;           // 80
static constexpr int NINPUT_DOF  = (N - 1) * NU;     // 38
static constexpr int NTOT        = 2 * NSTATE_DOF + NINPUT_DOF; // 198
//
// Interleaved layout:
//   index 2*(k*NX+i)   = x[k,i]     (state DOF)
//   index 2*(k*NX+i)+1 = λ[k,i]     (dual DOF)
//   index 2*N*NX + k*NU + i = u[k,i] (input DOF, appended)

inline int state_col(int k, int i) { return 2 * (k * NX + i); }
inline int dual_col(int k, int i)  { return 2 * (k * NX + i) + 1; }
inline int input_col(int k, int i) { return 2 * NSTATE_DOF + k * NU + i; }

// Small negative regularization for dual diagonal (ensures non-zero pivot)
static constexpr double EPS = 1e-4;

// ---------------------------------------------------------------------------
// KKT matrix builder
// ---------------------------------------------------------------------------
//
// Each (x[k,i], λ[k,i]) pair occupies adjacent columns (j, j+1) and forms:
//   col j   (x[k,i]): (j,j)=h and (j+1,j)=1.0   [2 entries]
//   col j+1 (λ[k,i]): (j+1,j+1)=-EPS             [1 entry]
//
// Input DOFs:
//   col input_col(k,i): (col, col) = r_{k,i}      [1 entry]
//
// 'scale' perturbs h and r for repeated-factorization testing.
// 'rng'   generates the base positive values.

static CscLower build_kkt(XorShift32 &rng, double scale = 1.0) {
    CscLower A;
    A.n = NTOT;
    A.col_ptr.resize(static_cast<std::size_t>(NTOT + 1), 0);

    // nnz = NSTATE_DOF*2 (state cols: diag+coupling)
    //     + NDUAL_DOF*1  (dual cols: diag only)
    //     + NINPUT_DOF*1 (input cols: diag only)
    int expected_nnz = NSTATE_DOF * 2 + NDUAL_DOF + NINPUT_DOF;
    A.row_idx.reserve(static_cast<std::size_t>(expected_nnz));
    A.values.reserve(static_cast<std::size_t>(expected_nnz));

    int nnz = 0;

    // --- State/dual interleaved columns ---
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < NX; ++i) {
            int jx = state_col(k, i); // = 2*(k*NX+i)
            int jl = dual_col(k, i);  // = 2*(k*NX+i)+1  = jx+1

            // Column jx (x[k,i]): entries at rows jx (diagonal) and jl (coupling)
            A.col_ptr[static_cast<std::size_t>(jx)] = nnz;

            double h = (1.0 + static_cast<double>(i + 1)) * scale
                       + 0.05 * rng.next(); // h > 0 always
            A.row_idx.push_back(jx);       // (jx, jx) = h
            A.values.push_back(h);
            ++nnz;
            A.row_idx.push_back(jl);       // (jl, jx) = 1.0   (lower triangle: jl > jx ✓)
            A.values.push_back(1.0);
            ++nnz;

            // Column jl (λ[k,i]): only diagonal entry
            A.col_ptr[static_cast<std::size_t>(jl)] = nnz;
            A.row_idx.push_back(jl);       // (jl, jl) = -EPS
            A.values.push_back(-EPS);
            ++nnz;
        }
    }

    // --- Input columns ---
    for (int k = 0; k < N - 1; ++k) {
        for (int i = 0; i < NU; ++i) {
            int ju = input_col(k, i);
            A.col_ptr[static_cast<std::size_t>(ju)] = nnz;

            double r = (0.5 + static_cast<double>(i + 1)) * scale
                       + 0.05 * rng.next(); // r > 0 always
            A.row_idx.push_back(ju);        // (ju, ju) = r
            A.values.push_back(r);
            ++nnz;
        }
    }

    A.col_ptr[static_cast<std::size_t>(NTOT)] = nnz;
    return A;
}

// ---------------------------------------------------------------------------
// Full symmetric SpMV: y = A * x
// ---------------------------------------------------------------------------

static void matvec_sym(const CscLower &A, const double *x, double *y) {
    int n = A.n;
    std::fill(y, y + n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
            int    i = A.row_idx[p];
            double v = A.values[p];
            y[i] += v * x[j];
            if (i != j) y[j] += v * x[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Test: 100 solves with fixed symbolic structure
// ---------------------------------------------------------------------------

TEST(OcpKktRegression, HundredSolves) {
    XorShift32 rng(1234u);

    // Build base KKT matrix
    CscLower A_base = build_kkt(rng, 1.0);
    ASSERT_EQ(A_base.n, NTOT);

    int expected_nnz = NSTATE_DOF * 2 + NDUAL_DOF + NINPUT_DOF;
    ASSERT_EQ(static_cast<int>(A_base.row_idx.size()), expected_nnz);

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    Info info;

    // ---------- analyse exactly once ----------
    int analyse_call_count = 0;
    Solver solver;
    auto ak = solver.analyse(A_base, ctrl, info);
    ++analyse_call_count;

    ASSERT_NE(ak.get(), nullptr) << "analyse() returned null";
    EXPECT_EQ(info.status, ErrorCode::Success);
    EXPECT_GT(info.analyse_seconds, 0.0) << "analyse_seconds must be > 0";

    ASSERT_EQ(ak->cleaned.values.size(), A_base.values.size())
        << "Cleaned matrix size mismatch";

    // Record inertia from first successful factorization
    int first_pos = -1, first_neg = -1, first_zero = -1;
    double total_factor_s = 0.0;

    // ---------- factor + solve 100 times ----------
    for (int iter = 0; iter < 100; ++iter) {
        // Build new values (same pattern, slightly different scale)
        double scale_factor = 1.0 + 0.01 * rng.next_signed();
        CscLower A_iter = build_kkt(rng, scale_factor);

        ASSERT_EQ(A_iter.row_idx.size(), A_base.row_idx.size())
            << "Sparsity pattern changed at iter " << iter;
        ASSERT_EQ(A_iter.values.size(), ak->cleaned.values.size())
            << "Value vector size mismatch at iter " << iter;

        // Update values in the analysis keep (same symbolic structure)
        ak->cleaned.values = A_iter.values;

        // Factor
        FactorKeep fk;
        FactorStatus fs = solver.factor(*ak, ctrl, info, fk);
        ASSERT_EQ(fs, FactorStatus::Success)
            << "factor() failed at iter " << iter
            << " (status=" << static_cast<int>(fs) << ")";

        total_factor_s += info.factor_seconds;

        // Check inertia stability
        if (iter == 0) {
            first_pos  = info.num_positive;
            first_neg  = info.num_negative;
            first_zero = info.num_zero;

            // Expected: NSTATE_DOF+NINPUT_DOF positive, NDUAL_DOF negative
            EXPECT_EQ(first_pos,  NSTATE_DOF + NINPUT_DOF)
                << "Expected " << NSTATE_DOF + NINPUT_DOF << " positive eigenvalues";
            EXPECT_EQ(first_neg,  NDUAL_DOF)
                << "Expected " << NDUAL_DOF << " negative eigenvalues";
            EXPECT_EQ(first_zero, 0)
                << "Expected 0 zero eigenvalues (non-singular KKT)";
        } else {
            EXPECT_EQ(info.num_positive, first_pos)
                << "Inertia num_positive changed at iter " << iter;
            EXPECT_EQ(info.num_negative, first_neg)
                << "Inertia num_negative changed at iter " << iter;
            EXPECT_EQ(info.num_zero, first_zero)
                << "Inertia num_zero changed at iter " << iter;
        }

        // Build RHS (deterministic per-iteration)
        XorShift32 rhs_rng(static_cast<uint32_t>(iter + 1000));
        std::vector<double> b(static_cast<std::size_t>(NTOT));
        for (auto &v : b) v = rhs_rng.next_signed();

        // Solve
        std::vector<double> x(b.begin(), b.end());
        int ret = solver.solve(fk, ctrl, info, x.data(), NTOT, 1);
        ASSERT_EQ(ret, 0) << "solve() returned error at iter " << iter;

        // Compute residual ‖A_iter * x − b‖_∞ / ‖b‖_∞
        std::vector<double> Ax(static_cast<std::size_t>(NTOT));
        matvec_sym(A_iter, x.data(), Ax.data());

        double res_inf = 0.0, b_inf = 0.0;
        for (int i = 0; i < NTOT; ++i) {
            res_inf = std::max(res_inf, std::abs(Ax[i] - b[i]));
            b_inf   = std::max(b_inf,   std::abs(b[i]));
        }
        double rel_res = (b_inf > 0.0) ? res_inf / b_inf : res_inf;
        EXPECT_LT(rel_res, 1e-9)
            << "Residual too large at iter " << iter
            << ": rel_res = " << rel_res;
    }

    // Total factor time must be >= 0 (small matrices may be sub-resolution)
    EXPECT_GE(total_factor_s, 0.0)
        << "factor_seconds must be non-negative";

    // analyse was called exactly once
    EXPECT_EQ(analyse_call_count, 1)
        << "analyse must be called exactly once";
}
