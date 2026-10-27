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
    bool static_pivots_as_zero = false; ///< a forced pivot whose |d| was raised counts in inertia.zero
    /// Hybrid accounting: a forced pivot raised from an exact zero (original
    /// |d| < eff_small or not finite) counts in inertia.zero; one raised from
    /// a nonzero |d| keeps its sign. Set on fallback fronts.
    bool static_zero_stays_zero = false;
    /// > 0 (with static_pivot > 0): no pivot search. Column k is the pivot;
    /// a 1x1 with finite |d| >= min_abs_pivot is accepted as it is, any other
    /// goes to the static path, which raises it to static_pivot.
    double min_abs_pivot = 0.0;
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
/// `first` > 0 continues a partial factorization: columns [0, first) are
/// already eliminated (L, D and tags in the layout above) and F[first:f,
/// first:f] holds their Schur complement, as ldlt_aptp_front leaves it. The
/// search starts at column `first`; a symmetric swap also permutes the rows
/// of the earlier L columns. num_accepted is then the end of the eliminated
/// columns (first plus this call's pivots), while inertia, static_pivots and
/// flops count this call's pivots only.
///
/// `work` must hold at least f * (prm.nb + 2) doubles.
LdltKernelResult ldlt_factor_front(double *F, int f, int num_elig, Int *rows,
                                   const LdltKernelParams &prm, int8_t *tags,
                                   double *work, int first = 0);

/// Largest front the small-front kernels (ldlt_factor_small_front) take.
inline constexpr int kSmallFrontMax = 32;

/// Leading dimension of the small-front layout for a front of size
/// 1 <= f <= kSmallFrontMax: 4, 8, 16 or 32.
inline constexpr int small_front_ld(int f) noexcept {
    return f <= 4 ? 4 : (f <= 8 ? 8 : (f <= 16 ? 16 : kSmallFrontMax));
}

/// ldlt_factor_front (first = 0) on a front of size f <= kSmallFrontMax
/// stored in the small-front layout: column-major with leading dimension
/// ld = small_front_ld(f), F 32-byte aligned, and rows [f, fe) of every
/// column zero, fe = f rounded up to a multiple of 4. The update kernels are
/// AVX2 microkernels over whole aligned 4-row blocks; they also write the
/// rows of each column above its diagonal down to the 4-row boundary (the
/// unused upper triangle; keep it finite, zero is best) and keep the padding
/// rows zero. Pivot sequence, tags, rows, inertia, static pivots and flops
/// are those of ldlt_factor_front, and the factor and Schur complement the
/// same values (the same operations per entry, in the same order).
LdltKernelResult ldlt_factor_small_front(double *F, int ld, int f, int num_elig, Int *rows,
                                         const LdltKernelParams &prm, int8_t *tags);

/// Number of doubles ldlt_factor_front needs in `work` for a front of size f.
inline std::size_t ldlt_work_size(int f, const LdltKernelParams &prm) {
    return static_cast<std::size_t>(f) * static_cast<std::size_t>(prm.nb + 2);
}

} // namespace smf
