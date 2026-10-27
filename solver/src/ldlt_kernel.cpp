#include "smf/ldlt_kernel.hpp"
#include "smf/dense_kernel.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace smf {

namespace {

// std::max({a, b, c}) without the initializer_list (MSVC lowers that to a
// vectorized max_element call): same result, NaN handling included.
inline double max3(double a, double b, double c) noexcept {
    double m = a;
    if (m < b) m = b;
    if (m < c) m = c;
    return m;
}

struct PivotChoice {
    int kind = 0; // 0 = none, 1 = 1x1 at a, 2 = 2x2 at (a, b)
    int a = -1;
    int b = -1;
};

// A 1x1 pivot that passes the threshold test with |d| below this fraction
// of its column's largest off-diagonal entry is "weak": it puts entries
// above 1 / kWeakPivotRatio into L. Only IPOPT's tiny u (1e-8) admits such
// pivots; for u >= kWeakPivotRatio the rule below never fires.
constexpr double kWeakPivotRatio = 1e-6;

// Element (i, j) of the symmetric front from its lower-triangle storage.
inline double sym_at(const double *F, std::size_t ld, int i, int j) noexcept {
    return (i >= j) ? F[static_cast<std::size_t>(j) * ld + i]
                    : F[static_cast<std::size_t>(i) * ld + j];
}

// Threshold test of the 2x2 pivot (cand, r): a scaled determinant above
// small_pivot and the growth bound u * max(|D^-1| * off-diagonal maxima) <=
// |det|. m2 is the largest eligible entry of column cand other than r, off1
// the largest entry of column r other than cand, and rows from ext0 on are
// the non-eligible rows of column cand.
//
// On success, *growth (if given) receives the pivot's growth bound
// max(bound0, bound1) / |det|: the largest entry it can put into L relative
// to the off-diagonal entries, comparable to a 1x1's max|a(i, c)| / |a_cc|.
bool two_by_two_ok(const double *F, std::size_t ld, int f, int cand, int r, const double *col,
                   const double *col_r, double m2, double off1, int ext0,
                   const LdltKernelParams &prm, double *growth = nullptr) noexcept {
    const double d00 = col[cand];
    const double d10 = sym_at(F, ld, r, cand);
    const double d11 = col_r[r];
    const double s2 = max3(std::abs(d00), std::abs(d10), std::abs(d11));
    if (!(s2 >= prm.eff_small) || !is_finite_fast(s2)) return false;
    const double a = d00 / s2;
    const double b = d10 / s2;
    const double c = d11 / s2;
    const double det_s = a * c - b * b;
    if (!(std::abs(det_s) > prm.small_pivot)) return false;

    double off0 = m2;
    for (int i = ext0; i < f; ++i) off0 = std::max(off0, std::abs(col[i]));
    double g = 0.0;
    if (off0 > 0.0 || off1 > 0.0) {
        const double bound0 = (std::abs(c) * off0 + std::abs(b) * off1) / s2;
        const double bound1 = (std::abs(b) * off0 + std::abs(a) * off1) / s2;
        if (prm.u * std::max(bound0, bound1) > std::abs(det_s)) return false;
        g = std::max(bound0, bound1) / std::abs(det_s);
    }
    if (growth != nullptr) *growth = g;
    return true;
}

// Threshold pivot selection over candidates [k, limit). Partners for the
// secondary 1x1 / 2x2 tests must also lie in [k, limit) so that every column
// touched by the chosen pivot is up to date.
PivotChoice select_pivot(const double *F, std::size_t ld, int f, int k, int num_elig,
                         int limit, const LdltKernelParams &prm) noexcept {
    const std::size_t sf = ld;
    if (prm.posdef) {
        const double d = F[static_cast<std::size_t>(k) * sf + k];
        if (d > 0.0 && d <= std::numeric_limits<double>::max()) return {1, k, -1};
        return {};
    }
    if (prm.min_abs_pivot > 0.0) {
        // In order, no search: column k or nothing (then the static path).
        const double d = F[static_cast<std::size_t>(k) * sf + k];
        if (is_finite_fast(d) && std::abs(d) >= prm.min_abs_pivot) return {1, k, -1};
        return {};
    }
    for (int cand = k; cand < limit; ++cand) {
        const double *col = F + static_cast<std::size_t>(cand) * sf;
        const double a_cand = std::abs(col[cand]);

        // One pass over the eligible rows of column cand: largest entry (the
        // 2x2 partner) and second largest (its off-diagonal bound).
        double m1 = 0.0;
        double m2 = 0.0;
        int r_best = -1;
        for (int i = k; i < cand; ++i) {
            const double v = std::abs(F[static_cast<std::size_t>(i) * sf + cand]);
            if (v > m1) {
                m2 = m1;
                m1 = v;
                r_best = i;
            } else if (v > m2) {
                m2 = v;
            }
        }
        for (int i = cand + 1; i < num_elig; ++i) {
            const double v = std::abs(col[i]);
            if (v > m1) {
                m2 = m1;
                m1 = v;
                r_best = i;
            } else if (v > m2) {
                m2 = v;
            }
        }
        const int ext0 = std::max(num_elig, cand + 1);

        // 1x1 on cand: a_cand >= u * max|a(i, cand)|. The remaining rows are
        // only scanned when the diagonal can pass, and the scan stops at the
        // first entry that makes the test fail. A weak 1x1 (below) is kept
        // in hand while the 2x2 with the partner is tried.
        bool weak_1x1 = false;
        double weak_growth = 0.0;
        if (a_cand >= prm.eff_small && !(prm.u * m1 > a_cand)) {
            bool ok = true;
            double colmax = m1;
            for (int i = ext0; i < f; ++i) {
                const double v = std::abs(col[i]);
                if (prm.u * v > a_cand) {
                    ok = false;
                    break;
                }
                colmax = std::max(colmax, v);
            }
            if (ok) {
                if (!(a_cand < kWeakPivotRatio * colmax)) return {1, cand, -1};
                weak_1x1 = true;
                weak_growth = colmax / a_cand;
            }
        }

        if (r_best < k || r_best >= limit || m1 <= 0.0) {
            if (weak_1x1) return {1, cand, -1};
            continue;
        }

        // One pass over column r_best: max excluding r_best (1x1 test on the
        // partner) and max excluding cand as well (2x2 bound).
        const double *col_r = F + static_cast<std::size_t>(r_best) * sf;
        double om2 = 0.0;
        double off1 = 0.0;
        for (int i = k; i < r_best; ++i) {
            const double v = std::abs(F[static_cast<std::size_t>(i) * sf + r_best]);
            om2 = std::max(om2, v);
            if (i != cand) off1 = std::max(off1, v);
        }
        for (int i = r_best + 1; i < f; ++i) {
            const double v = std::abs(col_r[i]);
            om2 = std::max(om2, v);
            if (i != cand) off1 = std::max(off1, v);
        }
        // The 2x2 (cand, r_best) is tried before a 1x1 on the partner, as in
        // MA27: at a tiny u (IPOPT's 1e-8) a near-null partner diagonal passes
        // the 1x1 test and puts entries up to 1/u into L, where the 2x2 would
        // have been stable.
        // A weak 1x1 on cand gives way to the 2x2 when that pivot's growth
        // bound is smaller (case500_goc at u = 1e-8: a 1x1 with |d| ~ 5e-8
        // of its column left a row with omega ~ 1 that 3 refinement steps
        // could not repair; MA27 pivots that column in a 2x2).
        double growth2 = 0.0;
        if (k + 1 < limit &&
            two_by_two_ok(F, ld, f, cand, r_best, col, col_r, m2, off1, ext0, prm, &growth2) &&
            (!weak_1x1 || growth2 < weak_growth))
            return {2, cand, r_best};
        if (weak_1x1) return {1, cand, -1};

        const double a_r = std::abs(col_r[r_best]);
        if (a_r >= prm.eff_small && (om2 == 0.0 || a_r >= prm.u * om2)) {
            return {1, r_best, -1};
        }
    }
    return {};
}

// Static pivoting, when no threshold pivot exists among [k, num_elig) (all
// columns current): take the column whose diagonal is largest relative to
// its column as a 1x1 pivot, so the growth in L stays as small as it can,
// and lift |d| to prm.static_pivot (keeping its sign) if it is smaller.
// With prm.min_abs_pivot > 0 (in-order mode) it takes column k as it is.
// `raised`: 0 = taken as it is, 1 = |d| raised from a nonzero value, 2 =
// raised from an exact zero (original |d| below prm.eff_small, or not
// finite: what inertia_1x1 would have counted as zero).
PivotChoice force_static_pivot(double *F, std::size_t ld, int f, int k, int num_elig,
                               const LdltKernelParams &prm,
                               LdltKernelResult &res, int &raised) noexcept {
    const std::size_t sf = ld;
    int best = k;
    double best_ratio = -1.0;
    // In-order mode (prm.min_abs_pivot): column k itself, no search.
    const int search_end = prm.min_abs_pivot > 0.0 ? k : num_elig;
    for (int cand = k; cand < search_end; ++cand) {
        const double *col = F + static_cast<std::size_t>(cand) * sf;
        double colmax = 0.0;
        for (int i = k; i < cand; ++i)
            colmax = std::max(colmax, std::abs(F[static_cast<std::size_t>(i) * sf + cand]));
        for (int i = cand + 1; i < f; ++i) colmax = std::max(colmax, std::abs(col[i]));
        const double a = std::abs(col[cand]);
        const double ratio = (colmax > 0.0) ? a / colmax : a;
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best = cand;
        }
    }
    double &d = F[static_cast<std::size_t>(best) * sf + best];
    raised = 0;
    if (!(std::abs(d) >= prm.static_pivot)) {
        raised = (is_finite_fast(d) && std::abs(d) >= prm.eff_small) ? 1 : 2;
        d = (d < 0.0) ? -prm.static_pivot : prm.static_pivot;
        ++res.static_pivots;
    }
    return {1, best, -1};
}

// Whether a forced pivot (force_static_pivot's `raised`) counts as a zero
// eigenvalue: every raised one with prm.static_pivots_as_zero, else only one
// raised from an exact zero with prm.static_zero_stays_zero.
inline bool static_counts_as_zero(int raised, const LdltKernelParams &prm) noexcept {
    if (raised == 0) return false;
    return prm.static_pivots_as_zero || (raised == 2 && prm.static_zero_stays_zero);
}

// Symmetric swap of positions p < q in the lower-stored front. Columns
// [0, p) (already-eliminated L) swap rows p and q too. W rows follow when
// w_cols > 0.
void sym_swap(double *F, std::size_t ld, int f, int p, int q, Int *rows, double *W,
              int w_cols) noexcept {
    if (p == q) return;
    if (p > q) std::swap(p, q);
    const std::size_t sf = ld;
    const std::size_t wf = static_cast<std::size_t>(f);
    std::swap(F[p * sf + p], F[q * sf + q]);
    for (int m = 0; m < p; ++m) std::swap(F[m * sf + p], F[m * sf + q]);
    for (int m = p + 1; m < q; ++m) std::swap(F[p * sf + m], F[m * sf + q]);
    for (int m = q + 1; m < f; ++m) std::swap(F[p * sf + m], F[q * sf + m]);
    std::swap(rows[p], rows[q]);
    for (int c = 0; c < w_cols; ++c) std::swap(W[c * wf + p], W[c * wf + q]);
}

// Eliminate a 1x1 pivot at k. Columns (k, ucol_end) receive the rank-1
// update immediately; when W is given, the unscaled column is saved at W
// column w_col for a deferred BLAS-3 update of the remaining columns.
void eliminate_1x1(double *F, int f, int k, int ucol_end, double *W, int w_col) noexcept {
    const std::size_t sf = static_cast<std::size_t>(f);
    double *ck = F + k * sf;
    const double inv_d = 1.0 / ck[k];
    if (W != nullptr) {
        double *w = W + static_cast<std::size_t>(w_col) * sf;
        for (int i = k + 1; i < f; ++i) w[i] = ck[i];
    }
    for (int j = k + 1; j < ucol_end; ++j) {
        const double lj = ck[j] * inv_d;
        if (lj == 0.0) continue;
        double *cj = F + j * sf;
        for (int i = j; i < f; ++i) cj[i] -= ck[i] * lj;
    }
    for (int i = k + 1; i < f; ++i) ck[i] *= inv_d;
}

// Inverse [i00 i10; i10 i11] of the 2x2 pivot [d00 d10; d10 d11], scaled by
// its largest entry for stability.
inline void inverse_2x2(double d00, double d10, double d11, double &i00, double &i10,
                        double &i11) noexcept {
    const double s = max3(std::abs(d00), std::abs(d10), std::abs(d11));
    const double a = d00 / s;
    const double b = d10 / s;
    const double c = d11 / s;
    const double inv = 1.0 / ((a * c - b * b) * s);
    i00 = c * inv;
    i10 = -b * inv;
    i11 = a * inv;
}

// Eliminate a 2x2 pivot at (k, k+1); same deferred-update contract as 1x1.
void eliminate_2x2(double *F, int f, int k, int ucol_end, double *W, int w_col) noexcept {
    const std::size_t sf = static_cast<std::size_t>(f);
    double *c0 = F + k * sf;
    double *c1 = F + (k + 1) * sf;
    double i00, i10, i11;
    inverse_2x2(c0[k], c0[k + 1], c1[k + 1], i00, i10, i11);
    if (W != nullptr) {
        double *w0 = W + static_cast<std::size_t>(w_col) * sf;
        double *w1 = w0 + sf;
        for (int i = k + 2; i < f; ++i) {
            w0[i] = c0[i];
            w1[i] = c1[i];
        }
    }
    for (int j = k + 2; j < ucol_end; ++j) {
        const double a0 = c0[j];
        const double a1 = c1[j];
        const double l0 = i00 * a0 + i10 * a1;
        const double l1 = i10 * a0 + i11 * a1;
        if (l0 == 0.0 && l1 == 0.0) continue;
        double *cj = F + j * sf;
        for (int i = j; i < f; ++i) cj[i] -= c0[i] * l0 + c1[i] * l1;
    }
    for (int i = k + 2; i < f; ++i) {
        const double a0 = c0[i];
        const double a1 = c1[i];
        c0[i] = i00 * a0 + i10 * a1;
        c1[i] = i10 * a0 + i11 * a1;
    }
}

// ---------------------------------------------------------------------------
// Small-front microkernels (ldlt_factor_small_front): the front lives in a
// 32-byte aligned buffer with a fixed leading dimension LD in {4, 8, 16, 32},
// its rows padded with zeros to fe = f rounded up to 4. Every column update
// then runs over whole aligned 4-row blocks, from the block holding its
// diagonal to fe: the rows above the diagonal it also writes are the unused
// upper triangle, the padding rows stay zero (0 - 0 * l) and are never read
// as front entries. Each entry gets exactly the operations of
// eliminate_1x1 / eliminate_2x2 in the same order (multiply, then add or
// subtract; no FMA), so the factor is the generic kernel's, bit for bit.

#if defined(__AVX2__)
// c[i] *= s for rows [r0, fe): scalar up to the 4-row boundary, so that the
// rows above r0 (the pivot block) are left alone.
inline void small_scale(double *c, int r0, int fe, double s) noexcept {
    int i = r0;
    for (; (i & 3) != 0 && i < fe; ++i) c[i] *= s;
    const __m256d vs = _mm256_set1_pd(s);
    for (; i < fe; i += 4) _mm256_store_pd(c + i, _mm256_mul_pd(_mm256_load_pd(c + i), vs));
}
#endif

template <int LD>
void small_eliminate_1x1(double *F, int f, int k) noexcept {
    double *ck = F + k * LD;
    const double inv_d = 1.0 / ck[k];
#if defined(__AVX2__)
    const int fe = (f + 3) & ~3;
    for (int j = k + 1; j < f; ++j) {
        const double lj = ck[j] * inv_d;
        if (lj == 0.0) continue;
        double *cj = F + j * LD;
        const __m256d vl = _mm256_set1_pd(lj);
        for (int i = j & ~3; i < fe; i += 4) {
            const __m256d prod = _mm256_mul_pd(_mm256_load_pd(ck + i), vl);
            _mm256_store_pd(cj + i, _mm256_sub_pd(_mm256_load_pd(cj + i), prod));
        }
    }
    small_scale(ck, k + 1, fe, inv_d);
#else
    for (int j = k + 1; j < f; ++j) {
        const double lj = ck[j] * inv_d;
        if (lj == 0.0) continue;
        double *cj = F + j * LD;
        for (int i = j; i < f; ++i) cj[i] -= ck[i] * lj;
    }
    for (int i = k + 1; i < f; ++i) ck[i] *= inv_d;
#endif
}

template <int LD>
void small_eliminate_2x2(double *F, int f, int k) noexcept {
    double *c0 = F + k * LD;
    double *c1 = c0 + LD;
    double i00, i10, i11;
    inverse_2x2(c0[k], c0[k + 1], c1[k + 1], i00, i10, i11);
#if defined(__AVX2__)
    const int fe = (f + 3) & ~3;
    for (int j = k + 2; j < f; ++j) {
        const double a0 = c0[j];
        const double a1 = c1[j];
        const double l0 = i00 * a0 + i10 * a1;
        const double l1 = i10 * a0 + i11 * a1;
        if (l0 == 0.0 && l1 == 0.0) continue;
        double *cj = F + j * LD;
        const __m256d v0 = _mm256_set1_pd(l0);
        const __m256d v1 = _mm256_set1_pd(l1);
        for (int i = j & ~3; i < fe; i += 4) {
            const __m256d t = _mm256_add_pd(_mm256_mul_pd(_mm256_load_pd(c0 + i), v0),
                                            _mm256_mul_pd(_mm256_load_pd(c1 + i), v1));
            _mm256_store_pd(cj + i, _mm256_sub_pd(_mm256_load_pd(cj + i), t));
        }
    }
    int i = k + 2;
    for (; (i & 3) != 0 && i < fe; ++i) {
        const double a0 = c0[i];
        const double a1 = c1[i];
        c0[i] = i00 * a0 + i10 * a1;
        c1[i] = i10 * a0 + i11 * a1;
    }
    const __m256d w00 = _mm256_set1_pd(i00);
    const __m256d w10 = _mm256_set1_pd(i10);
    const __m256d w11 = _mm256_set1_pd(i11);
    for (; i < fe; i += 4) {
        const __m256d a0 = _mm256_load_pd(c0 + i);
        const __m256d a1 = _mm256_load_pd(c1 + i);
        _mm256_store_pd(c0 + i, _mm256_add_pd(_mm256_mul_pd(w00, a0), _mm256_mul_pd(w10, a1)));
        _mm256_store_pd(c1 + i, _mm256_add_pd(_mm256_mul_pd(w10, a0), _mm256_mul_pd(w11, a1)));
    }
#else
    for (int j = k + 2; j < f; ++j) {
        const double a0 = c0[j];
        const double a1 = c1[j];
        const double l0 = i00 * a0 + i10 * a1;
        const double l1 = i10 * a0 + i11 * a1;
        if (l0 == 0.0 && l1 == 0.0) continue;
        double *cj = F + j * LD;
        for (int i = j; i < f; ++i) cj[i] -= c0[i] * l0 + c1[i] * l1;
    }
    for (int i = k + 2; i < f; ++i) {
        const double a0 = c0[i];
        const double a1 = c1[i];
        c0[i] = i00 * a0 + i10 * a1;
        c1[i] = i10 * a0 + i11 * a1;
    }
#endif
}

inline void inertia_1x1(double d, double small, InertiaCounts &out) noexcept {
    if (!is_finite_fast(d) || std::abs(d) < small) ++out.zero;
    else if (d > 0.0) ++out.positive;
    else ++out.negative;
}

// Accept the chosen pivot at position k: swap it into place, record its
// tag / inertia, and eliminate it. `as_zero` (a 1x1 static pivot, see
// static_counts_as_zero) counts it as a zero eigenvalue.
// LD > 0: the small-front layout (leading dimension LD, microkernels).
template <int LD = 0>
void apply_pivot(const PivotChoice &pc, double *F, int f, int &k, Int *rows,
                 int8_t *tags, double *W, int w_cols, int ucol_end,
                 const LdltKernelParams &prm, LdltKernelResult &res,
                 bool as_zero = false) noexcept {
    const std::size_t sf = LD > 0 ? static_cast<std::size_t>(LD) : static_cast<std::size_t>(f);
    if (pc.kind == 1) {
        sym_swap(F, sf, f, k, pc.a, rows, W, w_cols);
        if (as_zero) ++res.inertia.zero;
        else inertia_1x1(F[k * sf + k], prm.eff_small, res.inertia);
        tags[k] = 1;
        if constexpr (LD > 0) small_eliminate_1x1<LD>(F, f, k);
        else eliminate_1x1(F, f, k, ucol_end, W, w_cols);
        const double rem = static_cast<double>(f - (k + 1));
        res.flops += rem + rem * (rem + 1.0);
        ++k;
        return;
    }
    int partner = pc.b;
    if (pc.a != k) {
        sym_swap(F, sf, f, k, pc.a, rows, W, w_cols);
        if (partner == k) partner = pc.a;
    }
    if (partner != k + 1) sym_swap(F, sf, f, k + 1, partner, rows, W, w_cols);
    add_inertia_2x2(F[k * sf + k], F[k * sf + k + 1], F[(k + 1) * sf + k + 1],
                    prm.small_pivot, res.inertia);
    tags[k] = 2;
    tags[k + 1] = -1;
    if constexpr (LD > 0) small_eliminate_2x2<LD>(F, f, k);
    else eliminate_2x2(F, f, k, ucol_end, W, w_cols);
    const double rem = static_cast<double>(f - (k + 2));
    res.flops += 6.0 * rem + 2.0 * rem * (rem + 1.0);
    k += 2;
}

// Panel update of the eligible columns only: F[c0:f, c0:e] -= L[c0:f, p0:p1]
// * W[c0:e, 0:(p1-p0)]^T (lower part), as a lower GEMM on the [c0, e)
// square and a GEMM for the rows below it. Pivot selection reads only these
// columns; the trailing block [e, f) gets one rank-p update at the end
// (schur_update) instead of streaming through memory once per panel.
void deferred_update(double *F, int f, int p0, int p1, int c0, int e, const double *W) {
    const int jb = p1 - p0;
    const int m = e - c0;
    if (jb <= 0 || m <= 0) return;
    const std::size_t sf = static_cast<std::size_t>(f);
    smf_dgemmt_lower_nt_sub(F + c0 * sf + c0, m, jb, F + p0 * sf + c0, f, W + c0, f, f);
    const int r = f - e;
    if (r > 0)
        smf_dgemm_nt(F + c0 * sf + e, r, m, jb, F + p0 * sf + e, f, W + c0, f, f, -1.0, 1.0);
}

// Schur complement of the rows below the eligible block, once and at rank
// p - p0: F[e:f, e:f] -= L[e:f, p0:p] D L[e:f, p0:p]^T (lower). L D is formed
// in the front's unused strict upper triangle, F[p0:p, e:f], as its transpose.
void schur_update(double *F, int f, int p0, int p, int e, const int8_t *tags) {
    const int q = f - e;
    if (p <= p0 || q <= 0) return;
    const std::size_t sf = static_cast<std::size_t>(f);
    for (int c = p0; c < p; ++c) {
        const double *Lc = F + static_cast<std::size_t>(c) * sf;
        if (tags[c] == 2) {
            const double *Lc1 = Lc + sf;
            const double d00 = Lc[c], d10 = Lc[c + 1], d11 = Lc1[c + 1];
            for (int i = e; i < f; ++i) {
                double *row = F + static_cast<std::size_t>(i) * sf;
                row[c] = Lc[i] * d00 + Lc1[i] * d10;
                row[c + 1] = Lc[i] * d10 + Lc1[i] * d11;
            }
            ++c;
        } else {
            const double d = Lc[c];
            for (int i = e; i < f; ++i) F[static_cast<std::size_t>(i) * sf + c] = Lc[i] * d;
        }
    }
    smf_dgemmt_lower_nn_sub(F + e * sf + e, q, p - p0, F + p0 * sf + e, f, F + e * sf + p0, f, f);
}

// ldlt_factor_front's unblocked loop on the small-front layout.
template <int LD>
LdltKernelResult small_front_factor(double *F, int f, int num_elig, Int *rows,
                                    const LdltKernelParams &prm, int8_t *tags) noexcept {
    LdltKernelResult res;
    int k = 0;
    while (k < num_elig) {
        PivotChoice pc = select_pivot(F, LD, f, k, num_elig, num_elig, prm);
        bool as_zero = false;
        if (pc.kind == 0) {
            if (!(prm.static_pivot > 0.0)) break;
            int raised = 0;
            pc = force_static_pivot(F, LD, f, k, num_elig, prm, res, raised);
            as_zero = static_counts_as_zero(raised, prm);
        }
        apply_pivot<LD>(pc, F, f, k, rows, tags, nullptr, 0, f, prm, res, as_zero);
    }
    res.num_accepted = k;
    return res;
}

} // namespace

LdltKernelResult ldlt_factor_small_front(double *F, int ld, int f, int num_elig, Int *rows,
                                         const LdltKernelParams &prm, int8_t *tags) {
    assert(f >= 1 && f <= kSmallFrontMax && ld == small_front_ld(f));
    assert(reinterpret_cast<std::uintptr_t>(F) % 32 == 0);
    switch (ld) {
    case 4: return small_front_factor<4>(F, f, num_elig, rows, prm, tags);
    case 8: return small_front_factor<8>(F, f, num_elig, rows, prm, tags);
    case 16: return small_front_factor<16>(F, f, num_elig, rows, prm, tags);
    default: return small_front_factor<kSmallFrontMax>(F, f, num_elig, rows, prm, tags);
    }
}

LdltKernelResult ldlt_factor_front(double *F, int f, int num_elig, Int *rows,
                                   const LdltKernelParams &prm, int8_t *tags,
                                   double *work, int first) {
    LdltKernelResult res;
    int k = first;

    if (f - first < prm.blocked_min_front || prm.nb <= 1) {
        // Unblocked right-looking: every pivot updates the full trailing
        // triangle. No BLAS calls; fronts this small live in L1/L2.
        while (k < num_elig) {
            PivotChoice pc = select_pivot(F, static_cast<std::size_t>(f), f, k, num_elig, num_elig, prm);
            bool as_zero = false;
            if (pc.kind == 0) {
                if (!(prm.static_pivot > 0.0)) break;
                int raised = 0;
                pc = force_static_pivot(F, static_cast<std::size_t>(f), f, k, num_elig, prm, res, raised);
                as_zero = static_counts_as_zero(raised, prm);
            }
            apply_pivot(pc, F, f, k, rows, tags, nullptr, 0, f, prm, res, as_zero);
        }
        res.num_accepted = k;
        return res;
    }

    // Blocked: pivots within an nb-wide panel update only the panel columns
    // and are saved (unscaled) in W; the trailing triangle is then updated
    // with one lower-only GEMM per panel.
    double *W = work;
    while (k < num_elig) {
        const int ps = k;
        const int pl = std::min(num_elig, ps + prm.nb);
        bool stalled = false;
        while (k < pl) {
            const PivotChoice pc = select_pivot(F, static_cast<std::size_t>(f), f, k, num_elig, pl, prm);
            if (pc.kind == 0) {
                stalled = true;
                break;
            }
            apply_pivot(pc, F, f, k, rows, tags, W, k - ps, pl, prm, res);
        }
        deferred_update(F, f, ps, k, pl, num_elig, W);

        if (stalled && k < num_elig) {
            // The whole trailing matrix is current now: search every
            // remaining eligible column (handles cross-panel 2x2 pivots).
            const int ss = k;
            PivotChoice pc = select_pivot(F, static_cast<std::size_t>(f), f, k, num_elig, num_elig, prm);
            bool as_zero = false;
            if (pc.kind == 0) {
                if (!(prm.static_pivot > 0.0)) break;
                int raised = 0;
                pc = force_static_pivot(F, static_cast<std::size_t>(f), f, k, num_elig, prm, res, raised);
                as_zero = static_counts_as_zero(raised, prm);
            }
            apply_pivot(pc, F, f, k, rows, tags, W, 0, ss, prm, res, as_zero);
            deferred_update(F, f, ss, k, k, num_elig, W);
        }
    }
    schur_update(F, f, first, k, num_elig, tags);
    res.num_accepted = k;
    return res;
}

} // namespace smf
