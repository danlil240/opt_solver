#include "smf/ldlt_kernel.hpp"
#include "smf/dense_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace smf {

namespace {

struct PivotChoice {
    int kind = 0; // 0 = none, 1 = 1x1 at a, 2 = 2x2 at (a, b)
    int a = -1;
    int b = -1;
};

// Element (i, j) of the symmetric front from its lower-triangle storage.
inline double sym_at(const double *F, int f, int i, int j) noexcept {
    return (i >= j) ? F[static_cast<std::size_t>(j) * f + i]
                    : F[static_cast<std::size_t>(i) * f + j];
}

// Threshold pivot selection over candidates [k, limit). Partners for the
// secondary 1x1 / 2x2 tests must also lie in [k, limit) so that every column
// touched by the chosen pivot is up to date.
PivotChoice select_pivot(const double *F, int f, int k, int num_elig, int limit,
                         const LdltKernelParams &prm) noexcept {
    const std::size_t sf = static_cast<std::size_t>(f);
    if (prm.posdef) {
        const double d = F[static_cast<std::size_t>(k) * sf + k];
        if (d > 0.0 && d <= std::numeric_limits<double>::max()) return {1, k, -1};
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
        // first entry that makes the test fail.
        if (a_cand >= prm.eff_small && !(prm.u * m1 > a_cand)) {
            bool ok = true;
            for (int i = ext0; i < f; ++i) {
                if (prm.u * std::abs(col[i]) > a_cand) {
                    ok = false;
                    break;
                }
            }
            if (ok) return {1, cand, -1};
        }

        if (r_best < k || r_best >= limit || m1 <= 0.0) continue;

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
        const double a_r = std::abs(col_r[r_best]);
        if (a_r >= prm.eff_small && (om2 == 0.0 || a_r >= prm.u * om2)) {
            return {1, r_best, -1};
        }

        if (k + 1 >= limit) continue;

        const double d00 = col[cand];
        const double d10 = sym_at(F, f, r_best, cand);
        const double d11 = col_r[r_best];
        const double s2 = std::max({std::abs(d00), std::abs(d10), std::abs(d11)});
        if (!(s2 >= prm.eff_small) || !is_finite_fast(s2)) continue;
        const double a = d00 / s2;
        const double b = d10 / s2;
        const double c = d11 / s2;
        const double det_s = a * c - b * b;
        if (!(std::abs(det_s) > prm.small_pivot)) continue;

        double off0 = m2;
        for (int i = ext0; i < f; ++i) off0 = std::max(off0, std::abs(col[i]));
        if (off0 > 0.0 || off1 > 0.0) {
            const double bound0 = (std::abs(c) * off0 + std::abs(b) * off1) / s2;
            const double bound1 = (std::abs(b) * off0 + std::abs(a) * off1) / s2;
            if (prm.u * std::max(bound0, bound1) > std::abs(det_s)) continue;
        }
        return {2, cand, r_best};
    }
    return {};
}

// Static pivoting, when no threshold pivot exists among [k, num_elig) (all
// columns current): take the column whose diagonal is largest relative to
// its column as a 1x1 pivot, so the growth in L stays as small as it can,
// and lift |d| to prm.static_pivot (keeping its sign) if it is smaller.
PivotChoice force_static_pivot(double *F, int f, int k, int num_elig,
                               const LdltKernelParams &prm,
                               LdltKernelResult &res) noexcept {
    const std::size_t sf = static_cast<std::size_t>(f);
    int best = k;
    double best_ratio = -1.0;
    for (int cand = k; cand < num_elig; ++cand) {
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
    if (!(std::abs(d) >= prm.static_pivot)) {
        d = (d < 0.0) ? -prm.static_pivot : prm.static_pivot;
        ++res.static_pivots;
    }
    return {1, best, -1};
}

// Symmetric swap of positions p < q in the lower-stored front. Columns
// [0, p) (already-eliminated L) swap rows p and q too. W rows follow when
// w_cols > 0.
void sym_swap(double *F, int f, int p, int q, Int *rows, double *W,
              int w_cols) noexcept {
    if (p == q) return;
    if (p > q) std::swap(p, q);
    const std::size_t sf = static_cast<std::size_t>(f);
    std::swap(F[p * sf + p], F[q * sf + q]);
    for (int m = 0; m < p; ++m) std::swap(F[m * sf + p], F[m * sf + q]);
    for (int m = p + 1; m < q; ++m) std::swap(F[p * sf + m], F[m * sf + q]);
    for (int m = q + 1; m < f; ++m) std::swap(F[p * sf + m], F[q * sf + m]);
    std::swap(rows[p], rows[q]);
    for (int c = 0; c < w_cols; ++c) std::swap(W[c * sf + p], W[c * sf + q]);
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

// Eliminate a 2x2 pivot at (k, k+1); same deferred-update contract as 1x1.
void eliminate_2x2(double *F, int f, int k, int ucol_end, double *W, int w_col) noexcept {
    const std::size_t sf = static_cast<std::size_t>(f);
    double *c0 = F + k * sf;
    double *c1 = F + (k + 1) * sf;
    const double d00 = c0[k];
    const double d10 = c0[k + 1];
    const double d11 = c1[k + 1];
    const double s = std::max({std::abs(d00), std::abs(d10), std::abs(d11)});
    const double a = d00 / s;
    const double b = d10 / s;
    const double c = d11 / s;
    const double inv = 1.0 / ((a * c - b * b) * s);
    const double i00 = c * inv;
    const double i10 = -b * inv;
    const double i11 = a * inv;
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

inline void inertia_1x1(double d, double small, InertiaCounts &out) noexcept {
    if (!is_finite_fast(d) || std::abs(d) < small) ++out.zero;
    else if (d > 0.0) ++out.positive;
    else ++out.negative;
}

// Accept the chosen pivot at position k: swap it into place, record its
// tag / inertia, and eliminate it.
void apply_pivot(const PivotChoice &pc, double *F, int f, int &k, Int *rows,
                 int8_t *tags, double *W, int w_cols, int ucol_end,
                 const LdltKernelParams &prm, LdltKernelResult &res) noexcept {
    const std::size_t sf = static_cast<std::size_t>(f);
    if (pc.kind == 1) {
        sym_swap(F, f, k, pc.a, rows, W, w_cols);
        inertia_1x1(F[k * sf + k], prm.eff_small, res.inertia);
        tags[k] = 1;
        eliminate_1x1(F, f, k, ucol_end, W, w_cols);
        const double rem = static_cast<double>(f - (k + 1));
        res.flops += rem + rem * (rem + 1.0);
        ++k;
        return;
    }
    int partner = pc.b;
    if (pc.a != k) {
        sym_swap(F, f, k, pc.a, rows, W, w_cols);
        if (partner == k) partner = pc.a;
    }
    if (partner != k + 1) sym_swap(F, f, k + 1, partner, rows, W, w_cols);
    add_inertia_2x2(F[k * sf + k], F[k * sf + k + 1], F[(k + 1) * sf + k + 1],
                    prm.small_pivot, res.inertia);
    tags[k] = 2;
    tags[k + 1] = -1;
    eliminate_2x2(F, f, k, ucol_end, W, w_cols);
    const double rem = static_cast<double>(f - (k + 2));
    res.flops += 6.0 * rem + 2.0 * rem * (rem + 1.0);
    k += 2;
}

// F[c0:f, c0:f] -= L[c0:f, p0:p1] * W[c0:f, 0:(p1-p0)]^T (lower triangle).
void deferred_update(double *F, int f, int p0, int p1, int c0, const double *W) {
    const int jb = p1 - p0;
    const int m = f - c0;
    if (jb <= 0 || m <= 0) return;
    const std::size_t sf = static_cast<std::size_t>(f);
    smf_dgemmt_lower_nt_sub(F + c0 * sf + c0, m, jb, F + p0 * sf + c0, f, W + c0, f, f);
}

} // namespace

LdltKernelResult ldlt_factor_front(double *F, int f, int num_elig, Int *rows,
                                   const LdltKernelParams &prm, int8_t *tags,
                                   double *work) {
    LdltKernelResult res;
    int k = 0;

    if (f < prm.blocked_min_front || prm.nb <= 1) {
        // Unblocked right-looking: every pivot updates the full trailing
        // triangle. No BLAS calls; fronts this small live in L1/L2.
        while (k < num_elig) {
            PivotChoice pc = select_pivot(F, f, k, num_elig, num_elig, prm);
            if (pc.kind == 0) {
                if (!(prm.static_pivot > 0.0)) break;
                pc = force_static_pivot(F, f, k, num_elig, prm, res);
            }
            apply_pivot(pc, F, f, k, rows, tags, nullptr, 0, f, prm, res);
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
            const PivotChoice pc = select_pivot(F, f, k, num_elig, pl, prm);
            if (pc.kind == 0) {
                stalled = true;
                break;
            }
            apply_pivot(pc, F, f, k, rows, tags, W, k - ps, pl, prm, res);
        }
        deferred_update(F, f, ps, k, pl, W);

        if (stalled && k < num_elig) {
            // The whole trailing matrix is current now: search every
            // remaining eligible column (handles cross-panel 2x2 pivots).
            const int ss = k;
            PivotChoice pc = select_pivot(F, f, k, num_elig, num_elig, prm);
            if (pc.kind == 0) {
                if (!(prm.static_pivot > 0.0)) break;
                pc = force_static_pivot(F, f, k, num_elig, prm, res);
            }
            apply_pivot(pc, F, f, k, rows, tags, W, 0, ss, prm, res);
            deferred_update(F, f, ss, k, k, W);
        }
    }
    res.num_accepted = k;
    return res;
}

} // namespace smf
