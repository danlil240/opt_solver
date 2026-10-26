#pragma once

// Iterative refinement around a factorization (single right-hand side), the
// way IPOPT wraps every linear solve in the replay harness. Header-only and
// solver-agnostic: `solve_correction(r)` overwrites the residual r with the
// solution of A d = r from whatever factorization the caller holds, and
// returns false on failure.

#include "smf/csc_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace smf::bench {

// y = A x for a symmetric matrix stored as lower CSC (values from `vals`).
inline void sym_matvec(const CscLower &A, const std::vector<double> &vals,
                       const double *x, double *y) {
    std::fill(y, y + A.n, 0.0);
    for (Int j = 0; j < A.n; ++j) {
        for (Int k = A.col_ptr[static_cast<std::size_t>(j)];
             k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            const Int i = A.row_idx[static_cast<std::size_t>(k)];
            const double a = vals[static_cast<std::size_t>(k)];
            y[i] += a * x[j];
            if (i != j) y[j] += a * x[i];
        }
    }
}

// Refine x against A x = b: correct with A d = b - A x until both backward
// errors are converged or stop improving, at most `max_steps` correction
// solves.
//   normwise      eta   = ||r||inf / (||A||inf ||x||inf + ||b||inf)  <= kRefineEtaDone
//   componentwise omega = max_i |r_i| / ((|A||x|)_i + |b_i|)         <= kRefineOmegaDone
// Both must hold: eta alone is blind on KKT systems with huge barrier terms
// (||A|| ~ 1e11 makes eta ~ 1e-18 while rows with small entries are left with
// omega ~ 1e-6, which the harness's componentwise gate rejects).
inline constexpr double kRefineEtaDone = 1e-14;
inline constexpr double kRefineOmegaDone = 1e-10;

template <class SolveCorrection>
void refine_solution(SolveCorrection &&solve_correction, const CscLower &A,
                     const std::vector<double> &vals, const std::vector<double> &b,
                     std::vector<double> &x, int max_steps) {
    const std::size_t n = static_cast<std::size_t>(A.n);
    std::vector<double> ax(n), aax(n), r(n), rowsum(n, 0.0);
    for (Int j = 0; j < A.n; ++j) {
        for (Int k = A.col_ptr[static_cast<std::size_t>(j)];
             k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            const Int i = A.row_idx[static_cast<std::size_t>(k)];
            const double a = std::abs(vals[static_cast<std::size_t>(k)]);
            rowsum[static_cast<std::size_t>(i)] += a;
            if (i != j) rowsum[static_cast<std::size_t>(j)] += a;
        }
    }
    const double anorm = *std::max_element(rowsum.begin(), rowsum.end());
    double bnorm = 0.0;
    for (double v : b) bnorm = std::max(bnorm, std::abs(v));
    // merit <= 1 means both backward errors have converged.
    double prev = std::numeric_limits<double>::infinity();
    for (int it = 0; it <= max_steps; ++it) {
        std::fill(aax.begin(), aax.end(), 0.0);
        for (Int j = 0; j < A.n; ++j) {
            const double xj = x[static_cast<std::size_t>(j)];
            double ajx = 0.0;
            for (Int k = A.col_ptr[static_cast<std::size_t>(j)];
                 k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
                const std::size_t i = static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(k)]);
                const double a = std::abs(vals[static_cast<std::size_t>(k)]);
                aax[i] += a * std::abs(xj);
                if (i != static_cast<std::size_t>(j)) ajx += a * std::abs(x[i]);
            }
            aax[static_cast<std::size_t>(j)] += ajx;
        }
        sym_matvec(A, vals, x.data(), ax.data());
        double rnorm = 0.0, xnorm = 0.0, omega = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = b[i] - ax[i];
            rnorm = std::max(rnorm, std::abs(r[i]));
            xnorm = std::max(xnorm, std::abs(x[i]));
            const double den = aax[i] + std::abs(b[i]);
            if (den > 0.0) omega = std::max(omega, std::abs(r[i]) / den);
            else if (r[i] != 0.0) omega = std::numeric_limits<double>::infinity();
        }
        const double eta = rnorm / (anorm * xnorm + bnorm);
        const double merit = std::max(eta / kRefineEtaDone, omega / kRefineOmegaDone);
        if (it == max_steps || merit <= 1.0 || merit >= 0.5 * prev) break;
        prev = merit;
        if (!solve_correction(r)) break;
        for (std::size_t i = 0; i < n; ++i) x[i] += r[i];
    }
}

} // namespace smf::bench
