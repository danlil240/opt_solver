#include "smf/ldlt_kernel.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/dense_kernel.hpp"

#include "parallel/tile_tasks.hpp"
#include "phase_trace.hpp"

#include <chrono>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace smf {

namespace {

// Partition columns [0, m) whose column j has height M - j (M >= m > 0) into
// up to max_strips equal-area column strips.
inline int equal_area_strips(int m, int M, int max_strips, int bounds[65]) noexcept {
    const int nstrips = std::min({max_strips, 64, std::max(1, m / 8)});
    bounds[0] = 0;
    if (nstrips <= 1) {
        bounds[1] = m;
        return 1;
    }
    const double Md = static_cast<double>(M) + 0.5;
    const double total = static_cast<double>(m) * static_cast<double>(M) -
                         0.5 * static_cast<double>(m) * static_cast<double>(m - 1);
    const double Md2 = Md * Md;
    for (int b = 1; b < nstrips; ++b) {
        const double target = total * static_cast<double>(b) / static_cast<double>(nstrips);
        const double disc = std::max(0.0, Md2 - 2.0 * target);
        int c = static_cast<int>(Md - std::sqrt(disc) + 0.5);
        c = std::clamp(c, bounds[b - 1] + 1, m - (nstrips - b));
        bounds[b] = c;
    }
    bounds[nstrips] = m;
    return nstrips;
}

#if defined(_OPENMP)
// Runs strips [0, nstrips) on the current OpenMP team; call it inside the
// parallel region. The team can be smaller than the num_threads(nstrips)
// request (a nested region with nesting off, OMP_DYNAMIC, a thread limit), so
// each thread takes every team-size-th strip: none is skipped.
template <class Fn> inline void run_team_strips(int nstrips, const Fn &fn) {
    const int nth = omp_get_num_threads();
    for (int s = omp_get_thread_num(); s < nstrips; s += nth) fn(s);
}
#endif

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

// Weak 1x1 pivots: see kWeakPivotRatio (ldlt_kernel.hpp).

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
        if (growth != nullptr) g = std::max(bound0, bound1) / std::abs(det_s);
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
    // Fast path for cand = k: contiguous column scan via branchless AVX2 max.
    // Whenever |a(k,k)| satisfies both the threshold test and the non-weak 1x1
    // guard over the full column [k+1, f), select_pivot is guaranteed to pick
    // {1, k, -1} without needing m1/m2/r_best tracking.
    if (k < limit) {
        const double *col0 = F + static_cast<std::size_t>(k) * sf;
        const double a0 = std::abs(col0[k]);
        if (a0 >= prm.eff_small) {
            double colmax = 0.0;
            int i = k + 1;
#if defined(__AVX2__)
            if (f - i >= 8) {
                const __m256d sign_mask = _mm256_set1_pd(-0.0);
                __m256d vmax0 = _mm256_setzero_pd();
                __m256d vmax1 = _mm256_setzero_pd();
                for (; i + 8 <= f; i += 8) {
                    const __m256d v0 = _mm256_andnot_pd(sign_mask, _mm256_loadu_pd(col0 + i));
                    const __m256d v1 = _mm256_andnot_pd(sign_mask, _mm256_loadu_pd(col0 + i + 4));
                    vmax0 = _mm256_max_pd(v0, vmax0);
                    vmax1 = _mm256_max_pd(v1, vmax1);
                }
                vmax0 = _mm256_max_pd(vmax1, vmax0);
                const __m128d lo = _mm256_castpd256_pd128(vmax0);
                const __m128d hi = _mm256_extractf128_pd(vmax0, 1);
                const __m128d m2 = _mm_max_pd(hi, lo);
                const __m128d m1 = _mm_max_sd(_mm_unpackhi_pd(m2, m2), m2);
                colmax = _mm_cvtsd_f64(m1);
            }
#endif
            for (; i < f; ++i) {
                const double v = std::abs(col0[i]);
                if (v > colmax) colmax = v;
            }
            if (!(prm.u * colmax > a0) && !(a0 < kWeakPivotRatio * colmax)) {
                return {1, k, -1};
            }
        }
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
    double best_colmax = 0.0;
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
            best_colmax = colmax;
        }
    }
    double &d = F[static_cast<std::size_t>(best) * sf + best];
    raised = 0;
    if (!(std::abs(d) >= prm.static_pivot)) {
        raised = (is_finite_fast(d) && std::abs(d) >= prm.eff_small) ? 1 : 2;
        d = (d < 0.0) ? -prm.static_pivot : prm.static_pivot;
        ++res.static_pivots;
    }
    // The growth this pivot lets into L (threshold pivoting bounds it by
    // 1 / u): its column's largest entry over the |d| it is taken with.
    if (best_colmax > 0.0 && std::abs(d) > 0.0)
        res.max_growth = std::max(res.max_growth, best_colmax / std::abs(d));
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
        if (k + 1 < f) {
            std::memcpy(w + (k + 1), ck + (k + 1),
                        static_cast<std::size_t>(f - (k + 1)) * sizeof(double));
        }
    }
    int j = k + 1;
    while (j < ucol_end) {
        int jc[4];
        double jl[4];
        int nc = 0;
        while (j < ucol_end && nc < 4) {
            const double lj = ck[j] * inv_d;
            if (lj != 0.0) {
                jc[nc] = j;
                jl[nc] = lj;
                ++nc;
            }
            ++j;
        }
        if (nc == 0) break;
        if (nc == 4) {
            const int j0 = jc[0], j1 = jc[1], j2 = jc[2], j3 = jc[3];
            double *c0 = F + static_cast<std::size_t>(j0) * sf;
            double *c1 = F + static_cast<std::size_t>(j1) * sf;
            double *c2 = F + static_cast<std::size_t>(j2) * sf;
            double *c3 = F + static_cast<std::size_t>(j3) * sf;
            const double l0 = jl[0], l1 = jl[1], l2 = jl[2], l3 = jl[3];
            for (int i = j0; i < j3; ++i) c0[i] -= ck[i] * l0;
            for (int i = j1; i < j3; ++i) c1[i] -= ck[i] * l1;
            for (int i = j2; i < j3; ++i) c2[i] -= ck[i] * l2;
            int i = j3;
#if defined(__AVX2__)
            const __m256d vl0 = _mm256_set1_pd(l0);
            const __m256d vl1 = _mm256_set1_pd(l1);
            const __m256d vl2 = _mm256_set1_pd(l2);
            const __m256d vl3 = _mm256_set1_pd(l3);
            for (; i + 4 <= f; i += 4) {
                const __m256d vk = _mm256_loadu_pd(ck + i);
                _mm256_storeu_pd(c0 + i, _mm256_sub_pd(_mm256_loadu_pd(c0 + i), _mm256_mul_pd(vk, vl0)));
                _mm256_storeu_pd(c1 + i, _mm256_sub_pd(_mm256_loadu_pd(c1 + i), _mm256_mul_pd(vk, vl1)));
                _mm256_storeu_pd(c2 + i, _mm256_sub_pd(_mm256_loadu_pd(c2 + i), _mm256_mul_pd(vk, vl2)));
                _mm256_storeu_pd(c3 + i, _mm256_sub_pd(_mm256_loadu_pd(c3 + i), _mm256_mul_pd(vk, vl3)));
            }
#endif
            for (; i < f; ++i) {
                const double vk = ck[i];
                c0[i] -= vk * l0;
                c1[i] -= vk * l1;
                c2[i] -= vk * l2;
                c3[i] -= vk * l3;
            }
        } else {
            for (int t = 0; t < nc; ++t) {
                const int jt = jc[t];
                const double lt = jl[t];
                double *ct = F + static_cast<std::size_t>(jt) * sf;
                int i = jt;
#if defined(__AVX2__)
                const __m256d vlt = _mm256_set1_pd(lt);
                for (; i + 4 <= f; i += 4) {
                    const __m256d vk = _mm256_loadu_pd(ck + i);
                    _mm256_storeu_pd(ct + i, _mm256_sub_pd(_mm256_loadu_pd(ct + i), _mm256_mul_pd(vk, vlt)));
                }
#endif
                for (; i < f; ++i) ct[i] -= ck[i] * lt;
            }
        }
    }
    int i = k + 1;
#if defined(__AVX2__)
    const __m256d vs = _mm256_set1_pd(inv_d);
    for (; i + 4 <= f; i += 4) {
        _mm256_storeu_pd(ck + i, _mm256_mul_pd(_mm256_loadu_pd(ck + i), vs));
    }
#endif
    for (; i < f; ++i) ck[i] *= inv_d;
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
        if (k + 2 < f) {
            const std::size_t nb_bytes = static_cast<std::size_t>(f - (k + 2)) * sizeof(double);
            std::memcpy(w0 + (k + 2), c0 + (k + 2), nb_bytes);
            std::memcpy(w1 + (k + 2), c1 + (k + 2), nb_bytes);
        }
    }
    int j = k + 2;
    while (j < ucol_end) {
        int jc[2];
        double jl0[2], jl1[2];
        int nc = 0;
        while (j < ucol_end && nc < 2) {
            const double a0 = c0[j];
            const double a1 = c1[j];
            const double l0 = i00 * a0 + i10 * a1;
            const double l1 = i10 * a0 + i11 * a1;
            if (l0 != 0.0 || l1 != 0.0) {
                jc[nc] = j;
                jl0[nc] = l0;
                jl1[nc] = l1;
                ++nc;
            }
            ++j;
        }
        if (nc == 0) break;
        if (nc == 2) {
            const int j0 = jc[0], j1 = jc[1];
            double *d0 = F + static_cast<std::size_t>(j0) * sf;
            double *d1 = F + static_cast<std::size_t>(j1) * sf;
            const double l00 = jl0[0], l01 = jl1[0];
            const double l10 = jl0[1], l11 = jl1[1];
            for (int i = j0; i < j1; ++i) d0[i] -= c0[i] * l00 + c1[i] * l01;
            int i = j1;
#if defined(__AVX2__)
            const __m256d v00 = _mm256_set1_pd(l00);
            const __m256d v01 = _mm256_set1_pd(l01);
            const __m256d v10 = _mm256_set1_pd(l10);
            const __m256d v11 = _mm256_set1_pd(l11);
            for (; i + 4 <= f; i += 4) {
                const __m256d vc0 = _mm256_loadu_pd(c0 + i);
                const __m256d vc1 = _mm256_loadu_pd(c1 + i);
                const __m256d t0 = _mm256_add_pd(_mm256_mul_pd(vc0, v00), _mm256_mul_pd(vc1, v01));
                const __m256d t1 = _mm256_add_pd(_mm256_mul_pd(vc0, v10), _mm256_mul_pd(vc1, v11));
                _mm256_storeu_pd(d0 + i, _mm256_sub_pd(_mm256_loadu_pd(d0 + i), t0));
                _mm256_storeu_pd(d1 + i, _mm256_sub_pd(_mm256_loadu_pd(d1 + i), t1));
            }
#endif
            for (; i < f; ++i) {
                const double vc0 = c0[i], vc1 = c1[i];
                d0[i] -= vc0 * l00 + vc1 * l01;
                d1[i] -= vc0 * l10 + vc1 * l11;
            }
        } else {
            const int j0 = jc[0];
            double *d0 = F + static_cast<std::size_t>(j0) * sf;
            const double l0 = jl0[0], l1 = jl1[0];
            int i = j0;
#if defined(__AVX2__)
            const __m256d v0 = _mm256_set1_pd(l0);
            const __m256d v1 = _mm256_set1_pd(l1);
            for (; i + 4 <= f; i += 4) {
                const __m256d t = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(c0 + i), v0),
                                                _mm256_mul_pd(_mm256_loadu_pd(c1 + i), v1));
                _mm256_storeu_pd(d0 + i, _mm256_sub_pd(_mm256_loadu_pd(d0 + i), t));
            }
#endif
            for (; i < f; ++i) d0[i] -= c0[i] * l0 + c1[i] * l1;
        }
    }
    int i = k + 2;
#if defined(__AVX2__)
    const __m256d w00 = _mm256_set1_pd(i00);
    const __m256d w10 = _mm256_set1_pd(i10);
    const __m256d w11 = _mm256_set1_pd(i11);
    for (; i + 4 <= f; i += 4) {
        const __m256d a0 = _mm256_loadu_pd(c0 + i);
        const __m256d a1 = _mm256_loadu_pd(c1 + i);
        _mm256_storeu_pd(c0 + i, _mm256_add_pd(_mm256_mul_pd(w00, a0), _mm256_mul_pd(w10, a1)));
        _mm256_storeu_pd(c1 + i, _mm256_add_pd(_mm256_mul_pd(w10, a0), _mm256_mul_pd(w11, a1)));
    }
#endif
    for (; i < f; ++i) {
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
    {
        const int nt = blas_current_share();
        const int M = f - c0;
        if (nt >= 2 && m >= 24 && static_cast<long long>(M) * m * jb >= 65536LL) {
            int bounds[65];
            const int nstrips = equal_area_strips(m, M, nt, bounds);
            if (nstrips >= 2) {
                const auto run_strip = [&](int s) {
                    const int j0 = bounds[s];
                    const int j1 = bounds[s + 1];
                    const int w = j1 - j0;
                    const int col = c0 + j0;
                    const int H = f - col;
                    double *C = F + static_cast<std::size_t>(col) * sf + col;
                    const double *A = F + static_cast<std::size_t>(p0) * sf + col;
                    const double *B = W + col;
                    if (2 * w < H) {
                        smf_dgemm_nt_seq(C, H, w, jb, A, f, B, f, f, -1.0, 1.0);
                    } else {
                        smf_dgemmt_lower_nt_sub_seq(C, w, jb, A, f, B, f, f);
                        if (H > w) {
                            smf_dgemm_nt_seq(C + w, H - w, w, jb, A + w, f, B, f, f, -1.0, 1.0);
                        }
                    }
                };
#if !(defined(SMF_USE_MKL) && (SMF_USE_MKL))
                if (TileHelpers *h = current_top_helpers()) {
                    BlasSerialPinScope pin_blas;
                    parallel_for_helpers(nstrips, nstrips, h, run_strip);
                    return;
                }
#endif
#if defined(_OPENMP)
                BlasSerialPinScope pin_blas;
#pragma omp parallel num_threads(nstrips)
                {
                    const int prev_blas = blas_local_set_seq();
                    run_team_strips(nstrips, run_strip);
                    blas_local_restore(prev_blas);
                }
                return;
#endif
            }
        }
    }
    smf_dgemmt_lower_nt_sub(F + c0 * sf + c0, m, jb, F + p0 * sf + c0, f, W + c0, f, f);
    const int r = f - e;
    if (r > 0)
        smf_dgemm_nt(F + c0 * sf + e, r, m, jb, F + p0 * sf + e, f, W + c0, f, f, -1.0, 1.0);
}

// Fill one contiguous column of LD (length q) for pivot column c in [c0, c1).
inline void fill_ld_column(const double *F, std::size_t sf, int e, int q,
                           const int8_t *tags, int c, double *dst) noexcept {
    const double *Lc = F + static_cast<std::size_t>(c) * sf;
    const int8_t tag = tags[c];
    if (tag == 1) {
        const double d = Lc[c];
        const double *src = Lc + e;
        int i = 0;
#if defined(__AVX2__)
        const __m256d vd = _mm256_set1_pd(d);
        for (; i + 4 <= q; i += 4) {
            _mm256_storeu_pd(dst + i, _mm256_mul_pd(_mm256_loadu_pd(src + i), vd));
        }
#endif
        for (; i < q; ++i) dst[i] = src[i] * d;
    } else if (tag == 2) {
        const double *Lc1 = Lc + sf;
        const double d00 = Lc[c];
        const double d10 = Lc[c + 1];
        const double *s0 = Lc + e;
        const double *s1 = Lc1 + e;
        int i = 0;
#if defined(__AVX2__)
        const __m256d v0 = _mm256_set1_pd(d00);
        const __m256d v1 = _mm256_set1_pd(d10);
        for (; i + 4 <= q; i += 4) {
            const __m256d t = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(s0 + i), v0),
                                            _mm256_mul_pd(_mm256_loadu_pd(s1 + i), v1));
            _mm256_storeu_pd(dst + i, t);
        }
#endif
        for (; i < q; ++i) dst[i] = s0[i] * d00 + s1[i] * d10;
    } else if (tag == -1) {
        const double *Lc0 = Lc - sf;
        const double d10 = Lc0[c];
        const double d11 = Lc[c];
        const double *s0 = Lc0 + e;
        const double *s1 = Lc + e;
        int i = 0;
#if defined(__AVX2__)
        const __m256d v0 = _mm256_set1_pd(d10);
        const __m256d v1 = _mm256_set1_pd(d11);
        for (; i + 4 <= q; i += 4) {
            const __m256d t = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(s0 + i), v0),
                                            _mm256_mul_pd(_mm256_loadu_pd(s1 + i), v1));
            _mm256_storeu_pd(dst + i, t);
        }
#endif
        for (; i < q; ++i) dst[i] = s0[i] * d10 + s1[i] * d11;
    } else {
        std::fill_n(dst, q, 0.0);
    }
}

// Schur complement of the rows below the eligible block, once and at rank
// p - p0: F[e:f, e:f] -= L[e:f, p0:p] D L[e:f, p0:p]^T (lower).
void schur_update(double *F, int f, int p0, int p, int e, const int8_t *tags,
                  double *buf, std::size_t buf_cap) {
    const int q = f - e;
    if (p <= p0 || q <= 0) return;
    const std::size_t sf = static_cast<std::size_t>(f);
    if (buf == nullptr || buf_cap < static_cast<std::size_t>(q) * 2u) {
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
        return;
    }

    const std::size_t sq = static_cast<std::size_t>(q);
    const int max_cols = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(p - p0), buf_cap / sq));
    int c0 = p0;
    while (c0 < p) {
        int c1 = std::min(p, c0 + max_cols);
        if (c1 < p && tags[c1 - 1] == 2) {
            if (c1 - 1 > c0) --c1;
            else ++c1;
        }
        const int kc = c1 - c0;
        {
            const int nt = blas_current_share();
            if (nt >= 2 && q >= 64 && static_cast<long long>(q) * q * kc >= 262144LL) {
                int bounds[65];
                const int nstrips = equal_area_strips(q, q, nt, bounds);
                if (nstrips >= 2) {
                    const auto run_strip = [&](int s) {
                        const int j0 = bounds[s];
                        const int j1 = bounds[s + 1];
                        const int w = j1 - j0;
                        const int H = q - j0;
                        double *C = F + static_cast<std::size_t>(e + j0) * sf + (e + j0);
                        const double *A = F + static_cast<std::size_t>(c0) * sf + (e + j0);
                        const double *B = buf + j0;
                        if (2 * w < H) {
                            smf_dgemm_nt_seq(C, H, w, kc, A, f, B, q, f, -1.0, 1.0);
                        } else {
                            smf_dgemmt_lower_nt_sub_seq(C, w, kc, A, f, B, q, f);
                            if (H > w) {
                                smf_dgemm_nt_seq(C + w, H - w, w, kc, A + w, f, B, q, f, -1.0, 1.0);
                            }
                        }
                    };
#if !(defined(SMF_USE_MKL) && (SMF_USE_MKL))
                    if (TileHelpers *h = current_top_helpers()) {
                        BlasSerialPinScope pin_blas;
                        if (static_cast<long long>(kc) * q >= 16384LL && kc >= 8) {
                            const int nfill = std::min(nstrips, kc);
                            parallel_for_helpers(nfill, nfill, h, [&](int s) {
                                const int t0 = (kc * s) / nfill;
                                const int t1 = (kc * (s + 1)) / nfill;
                                for (int t = t0; t < t1; ++t) {
                                    fill_ld_column(F, sf, e, q, tags, c0 + t,
                                                   buf + static_cast<std::size_t>(t) * sq);
                                }
                            });
                        } else {
                            for (int t = 0; t < kc; ++t) {
                                fill_ld_column(F, sf, e, q, tags, c0 + t,
                                               buf + static_cast<std::size_t>(t) * sq);
                            }
                        }
                        parallel_for_helpers(nstrips, nstrips, h, run_strip);
                        c0 = c1;
                        continue;
                    }
#endif
#if defined(_OPENMP)
                    BlasSerialPinScope pin_blas;
#pragma omp parallel num_threads(nstrips)
                    {
                        const int prev_blas = blas_local_set_seq();
#pragma omp for schedule(static)
                        for (int t = 0; t < kc; ++t) {
                            fill_ld_column(F, sf, e, q, tags, c0 + t,
                                           buf + static_cast<std::size_t>(t) * sq);
                        }
                        run_team_strips(nstrips, run_strip);
                        blas_local_restore(prev_blas);
                    }
                    c0 = c1;
                    continue;
#endif
                }
            }
        }
        for (int t = 0; t < kc; ++t) {
            fill_ld_column(F, sf, e, q, tags, c0 + t,
                           buf + static_cast<std::size_t>(t) * sq);
        }
        smf_dgemmt_lower_nt_sub(F + static_cast<std::size_t>(e) * sf + e, q, kc,
                                F + static_cast<std::size_t>(c0) * sf + e, f, buf, q, f);
        c0 = c1;
    }
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
                                   double *work, int first,
                                   std::size_t work_cap) {
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
    // SMF_BENCH_PHASE_TRACE=2 on a top front: time since the last lap goes to
    // a kernel sum (top_panel / top_defer / top_schur).
    const bool lap_on = phase_in_top_front();
    auto lap_t = lap_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto lap = [&](PhaseSum slot) {
        if (!lap_on) return;
        const auto t = std::chrono::steady_clock::now();
        phase_sum_add(slot, std::chrono::duration<double>(t - lap_t).count());
        lap_t = t;
    };
    // Plain threshold pivoting first passes over the columns once, without
    // searching outside the panel. The children's delayed columns come
    // first in the front and mostly fail again; searching every remaining
    // column for each pivot (the loop below) costs a scan of each failed
    // column per pivot and a rank-1 update of the whole eligible block per
    // pivot (ncvxqp9 at u = 0.01: half its factor time). A panel that stalls
    // (every eligible column current then) instead
    // - pulls in partners: each failed column whose largest entry lies
    //   outside the panel gets that column next to it, and the next panel
    //   starts on these pairs (the 2x2 the search would have found, or the
    //   partner's 1x1). Without them a far partner pivots alone first and
    //   the failed column later on what is left (cops_polygon_200 at
    //   u = 0.01: backward error 1e-14 -> 1.4e-8, ncvxqp7 at 1e-8: 2e-16 ->
    //   6e-11). Again only once a pivot was taken since the last pairs;
    // - else moves its failed columns behind the untried ones, to
    //   [live, num_elig), and the next panel starts on fresh columns.
    // The loop below then tries the moved columns again, all current.
    // In-order, static-pivot and SPD fronts keep their order (they pivot or
    // stop at the first failure).
    if (!prm.posdef && prm.static_pivot == 0.0 && prm.min_abs_pivot == 0.0) {
        const std::size_t sf = static_cast<std::size_t>(f);
        int live = num_elig;
        int k_paired = -1;
        while (k < live) {
            const int ps = k;
            const int pl = std::min(live, ps + prm.nb);
            bool stalled = false;
            while (k < pl) {
                const PivotChoice pc = select_pivot(F, sf, f, k, num_elig, pl, prm);
                if (pc.kind == 0) {
                    stalled = true;
                    break;
                }
                apply_pivot(pc, F, f, k, rows, tags, W, k - ps, pl, prm, res);
            }
            lap(PhaseSum::TopPanel);
            deferred_update(F, f, ps, k, pl, num_elig, W);
            lap(PhaseSum::TopDefer);
            if (stalled && k != k_paired) {
                // order: (failed column, partner) row ids, distinct partners.
                Int order[128];
                int np = 0;
                const int max_pairs = std::min(prm.nb / 2, 64);
                for (int pos = k; pos < pl && np / 2 < max_pairs; ++pos) {
                    const double *col = F + static_cast<std::size_t>(pos) * sf;
                    double in_max = 0.0;
                    for (int i = k; i < pos; ++i) in_max = std::max(in_max, std::abs(F[static_cast<std::size_t>(i) * sf + pos]));
                    for (int i = pos + 1; i < pl; ++i) in_max = std::max(in_max, std::abs(col[i]));
                    double out_max = 0.0;
                    int r = -1;
                    for (int i = pl; i < num_elig; ++i) {
                        const double v = std::abs(col[i]);
                        if (v > out_max) {
                            out_max = v;
                            r = i;
                        }
                    }
                    // The search tried the partners inside the panel.
                    if (r < 0 || !(out_max > in_max)) continue;
                    bool taken = false;
                    for (int t = 1; t < np; t += 2) taken = taken || order[t] == rows[r];
                    if (taken) continue;
                    order[np++] = rows[pos];
                    order[np++] = rows[r];
                }
                if (np > 0) {
                    // Each swap fixes position k + t; the column it displaces
                    // moves further back (row ids are distinct).
                    for (int t = 0; t < np; ++t) {
                        int q = k + t;
                        while (rows[q] != order[t]) ++q;
                        sym_swap(F, sf, f, k + t, q, rows, nullptr, 0);
                    }
                    live = std::max(live, k + np);
                    k_paired = k;
                    continue;
                }
            }
            if (stalled) {
                // The m failed columns [k, pl) trade places with the last
                // mv untried ones (all of [pl, live) when fewer remain).
                const int m = pl - k;
                const int mv = std::min(m, live - pl);
                for (int t = 0; t < mv; ++t)
                    sym_swap(F, sf, f, k + t, live - mv + t, rows, nullptr, 0);
                live -= m;
            }
            lap(PhaseSum::TopPanel);
        }
    }
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
        lap(PhaseSum::TopPanel);
        deferred_update(F, f, ps, k, pl, num_elig, W);
        lap(PhaseSum::TopDefer);

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
            lap(PhaseSum::TopPanel);
            deferred_update(F, f, ss, k, k, num_elig, W);
            lap(PhaseSum::TopDefer);
        }
    }
    lap(PhaseSum::TopPanel);
    const std::size_t eff_cap = work_cap > 0 ? work_cap : ldlt_work_size(f, prm);
    schur_update(F, f, first, k, num_elig, tags, work, eff_cap);
    lap(PhaseSum::TopSchur);
    res.num_accepted = k;
    return res;
}

#if !(defined(SMF_USE_MKL) && (SMF_USE_MKL))
namespace {
void posdef_dtrsm_omp(double *B, int n_rows, int k, const double *L, int ldl, int ldb, int nt) {
    if (n_rows <= 0 || k <= 0) return;
    if (nt >= 2 && n_rows >= 64 && static_cast<long long>(n_rows) * k * k >= 65536LL) {
        const int nstrips = std::min({nt, 64, std::max(1, n_rows / 16)});
        if (nstrips >= 2) {
            const auto run_strip = [&](int s) {
                const int r0 = (n_rows * s) / nstrips;
                const int r1 = (n_rows * (s + 1)) / nstrips;
                if (r1 > r0) {
                    smf_dtrsm_right_lower_transpose_seq(B + r0, r1 - r0, k, L, ldl, ldb);
                }
            };
            if (TileHelpers *h = current_top_helpers()) {
                BlasSerialPinScope pin_blas;
                parallel_for_helpers(nstrips, nstrips, h, run_strip);
                return;
            }
#if defined(_OPENMP)
            BlasSerialPinScope pin_blas;
#pragma omp parallel num_threads(nstrips)
            {
                run_team_strips(nstrips, run_strip);
            }
            return;
#endif
        }
    }
    smf_dtrsm_right_lower_transpose(B, n_rows, k, L, ldl, ldb);
}

void posdef_dsyrk_omp(double *C, int m, const double *A, int k, int lda, int ldc, int nt) {
    if (m <= 0 || k <= 0) return;
    if (nt >= 2 && m >= 64 && static_cast<long long>(m) * m * k >= 262144LL) {
        int bounds[65];
        const int nstrips = equal_area_strips(m, m, nt, bounds);
        if (nstrips >= 2) {
            const std::size_t sldc = static_cast<std::size_t>(ldc);
            const auto run_strip = [&](int s) {
                const int j0 = bounds[s];
                const int j1 = bounds[s + 1];
                const int w = j1 - j0;
                const int H = m - j0;
                double *Cj = C + static_cast<std::size_t>(j0) * sldc + j0;
                const double *Aj = A + j0;
                smf_dsyrk_lower_seq(Cj, w, Aj, k, lda, ldc);
                if (H > w) {
                    smf_dgemm_nt_seq(Cj + w, H - w, w, k, Aj + w, lda, Aj, lda, ldc, -1.0, 1.0);
                }
            };
            if (TileHelpers *h = current_top_helpers()) {
                BlasSerialPinScope pin_blas;
                parallel_for_helpers(nstrips, nstrips, h, run_strip);
                return;
            }
#if defined(_OPENMP)
            BlasSerialPinScope pin_blas;
#pragma omp parallel num_threads(nstrips)
            {
                run_team_strips(nstrips, run_strip);
            }
            return;
#endif
        }
    }
    smf_dsyrk_lower(C, m, A, k, lda, ldc);
}

int posdef_dpotrf_omp(double *A, int n, int lda, int nt) {
    if (nt < 2 || n < 256) return smf_dpotrf_lower(A, n, lda);
    constexpr int kNb = 128;
    const std::size_t slda = static_cast<std::size_t>(lda);
    BlasSerialPinScope pin_blas;
    for (int j = 0; j < n; j += kNb) {
        const int jb = std::min(kNb, n - j);
        double *Ajj = A + static_cast<std::size_t>(j) * slda + j;
        const int info = smf_dpotrf_lower(Ajj, jb, lda);
        if (info != 0) return info > 0 ? j + info : info;
        const int rem = n - (j + jb);
        if (rem > 0) {
            double *A21 = Ajj + jb;
            double *A22 = A + static_cast<std::size_t>(j + jb) * slda + (j + jb);
            posdef_dtrsm_omp(A21, rem, jb, Ajj, lda, lda, nt);
            posdef_dsyrk_omp(A22, rem, A21, jb, lda, lda, nt);
        }
    }
    return 0;
}
} // namespace
#endif

LdltKernelResult posdef_factor_front_lapack(double *F, int f, int num_elig, int8_t *tags) {
    LdltKernelResult res;
    const int p = num_elig;
    if (p <= 0) return res;
    const std::size_t sf = static_cast<std::size_t>(f);
#if !(defined(SMF_USE_MKL) && (SMF_USE_MKL))
    const int nt = blas_current_share();
    const int info = posdef_dpotrf_omp(F, p, f, nt);
#else
    // dpotrf stops at the first leading minor whose pivot is not > 0 (or is
    // NaN); info = that column + 1. The columns before it are factored.
    const int info = smf_dpotrf_lower(F, p, f);
#endif
    int ok = info == 0 ? p : (info > 0 ? info - 1 : 0);
    // ldlt_factor_front also rejects an infinite pivot, which dpotrf takes
    // (sqrt(inf) = inf); the pivots before it are finite and positive.
    for (int c = 0; c < ok; ++c) {
        if (!is_finite_fast(F[static_cast<std::size_t>(c) * sf + c])) {
            ok = c;
            break;
        }
    }
    for (int c = 0; c < ok; ++c) {
        tags[c] = 1;
        const double rem = static_cast<double>(f - (c + 1));
        res.flops += rem + rem * (rem + 1.0);
    }
    res.inertia.positive = ok;
    res.num_accepted = ok;
    if (ok < p) return res;
    const int q = f - p;
    if (q > 0) {
        // L21 = A21 L11^-T, then S = A22 - L21 L21^T (lower).
#if !(defined(SMF_USE_MKL) && (SMF_USE_MKL))
        const int nt2 = blas_current_share();
        posdef_dtrsm_omp(F + p, q, p, F, f, f, nt2);
        posdef_dsyrk_omp(F + static_cast<std::size_t>(p) * sf + p, q, F + p, p, f, f, nt2);
#else
        smf_dtrsm_right_lower_transpose(F + p, q, p, F, f, f);
        smf_dsyrk_lower(F + static_cast<std::size_t>(p) * sf + p, q, F + p, p, f, f);
#endif
    }
    return res;
}

} // namespace smf
