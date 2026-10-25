#pragma once

#include "smf/dense_indef.hpp"
#include "smf/types.hpp"

#include <cstdint>

namespace smf {

struct LdltKernelParams {
    double u = 0.01;            ///< threshold pivoting parameter
    double eff_small = 1e-20;   ///< absolute 1x1 / 2x2-scale pivot threshold
    double small_pivot = 1e-20; ///< scaled 2x2 determinant threshold
    int nb = 32;                ///< panel width for the blocked (BLAS-3) path
    int blocked_min_front = 96; ///< fronts smaller than this use the unblocked path
    bool posdef = false;        ///< no pivoting: accept a(k,k) > 0 in order, stop otherwise
    double static_pivot = 0.0;  ///< > 0: never stop; force a 1x1 with |d| >= static_pivot
};

struct LdltKernelResult {
    int num_accepted = 0;
    int static_pivots = 0;      ///< forced pivots whose |d| was raised
    InertiaCounts inertia{};
    double flops = 0.0;
};

/// In-place LDL^T of the leading `num_elig` (fully summed) columns of a
/// symmetric front with threshold 1x1 / 2x2 pivoting.
///
/// F is f x f, column-major with leading dimension f; only the lower triangle
/// is read or written. `rows` (length f) is permuted in sync with every
/// symmetric swap. Pivot acceptance follows the classic MA27-style tests also
/// used by factor_front_reference / factor_front_blocked.
///
/// On return, with p = num_accepted:
///  - columns [0, p) hold L strictly below the pivot blocks (unscaled D on
///    the diagonal, and the 2x2 off-diagonal at (c+1, c) for 2x2 pivots);
///  - tags[c] is 1 (1x1), 2 (first of 2x2) or -1 (second of 2x2) for c < p;
///  - the lower triangle of F[p:f, p:f] is the Schur complement, with the
///    unaccepted eligible columns in positions [p, num_elig).
///
/// With prm.posdef the columns are taken in order without any search and the
/// factorization stops at the first non-positive (or non-finite) pivot.
///
/// `work` must hold at least f * (prm.nb + 2) doubles.
LdltKernelResult ldlt_factor_front(double *F, int f, int num_elig, Int *rows,
                                   const LdltKernelParams &prm, int8_t *tags,
                                   double *work);

/// Number of doubles ldlt_factor_front needs in `work` for a front of size f.
inline std::size_t ldlt_work_size(int f, const LdltKernelParams &prm) {
    return static_cast<std::size_t>(f) * static_cast<std::size_t>(prm.nb + 2);
}

} // namespace smf
