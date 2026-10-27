// APTP (a posteriori threshold pivoting) dense front kernel.
//
// Structure (Duff, Hogg & Lopez, RAL-TR-2018-012, Alg. 1-2), serial:
//
//   sweep(view, block bs):  for each block column [c0, c0 + w) of the view
//     1. back up the block column (diagonal tile and the rows below it);
//     2. factor the diagonal tile speculatively (child, see below), which
//        permutes the tile so its ne_t eliminated pivots come first;
//     3. apply the tile permutation to the rest of the view;
//     4. ApplyN: L = A P (L_t D_t)^{-T} for the rows below the tile (TRSM in
//        place); ApplyT: the same for the rows of previously failed columns,
//        which are stored transposed in those columns (gathered, out of
//        place). nelim = first pivot with an |l| > 1/u; a 2x2 never splits;
//     5. restore columns [nelim, w) from the backup (permuted) - fail in
//        place, no data movement between tiles;
//     6. update the uneliminated part of the view with the nelim pivots only:
//        the eligible columns right of the pivots (UpdateNN), the previously
//        failed columns (UpdateNT / UpdateTN).
//   At the end the failed columns are moved behind the eliminated ones.
//
//   Children: the outer sweep (bs = nb) factors each tile with an inner
//   sweep (bs = ib) whose diagonal blocks use complete pivoting, followed by
//   complete pivoting over the columns that failed inside the tile (the tile
//   is in cache, so this costs little and keeps saddle-point tiles whole).
//
// Storage: the view is a symmetric matrix held in its lower triangle with
// leading dimension ld. Entry (q, p) between a failed column q and a later
// pivot p (q < p) holds L(q, p) at column q, row p, i.e. transposed. The
// final symmetric permutation that moves failed columns to the back turns it
// into column p', row q' as the ldlt_factor_front layout requires.

#include "smf/ldlt_aptp.hpp"
#include "smf/dense_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace smf {

namespace {

struct Ctx {
    double inv_u = 100.0;
    double small = 1e-20;
    int nb = 128; // effective outer block
    int ib = 32;  // effective inner block (<= nb)
    double *bk[2] = {nullptr, nullptr}; // block-column backups per level
    double *lt = nullptr;   // unit lower copy of the tile's L (nb x nb)
    double *ga = nullptr;   // gathered rows (nb x nb)
    double *gb = nullptr;   // gathered L D rows (nb x nb)
    double *gc = nullptr;   // product scratch (nb x nb)
    double *dinv = nullptr; // 2 * nb: diagonal / off-diagonal of D^{-1}
    double *colbuf = nullptr; // f
    int *top_lp = nullptr;  // num_elig
    int *q[2] = {nullptr, nullptr}; // failed-column lists per level
    int *lpc[2] = {nullptr, nullptr}; // child permutations per level
    int *cp_lp = nullptr;   // nb
    int *itmp = nullptr;    // nb
};

inline std::size_t idx(int j, int ld) noexcept {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

inline bool l_ok(double l, double inv_u) noexcept { return std::abs(l) <= inv_u; }

// Symmetric transposition of positions p and q in the lower-stored m x m view
// (columns before p: rows p and q; between: (m, p) <-> (q, m); after q: rows).
void view_swap(double *A, int ld, int m, int p, int q, int8_t *tags, Int *rows,
               int *lp) noexcept {
    if (p == q) return;
    if (p > q) std::swap(p, q);
    std::swap(A[idx(p, ld) + p], A[idx(q, ld) + q]);
    for (int c = 0; c < p; ++c) std::swap(A[idx(c, ld) + p], A[idx(c, ld) + q]);
    for (int c = p + 1; c < q; ++c) std::swap(A[idx(p, ld) + c], A[idx(c, ld) + q]);
    double *cp = A + idx(p, ld);
    double *cq = A + idx(q, ld);
    for (int r = q + 1; r < m; ++r) std::swap(cp[r], cq[r]);
    std::swap(tags[p], tags[q]);
    std::swap(rows[p], rows[q]);
    std::swap(lp[p], lp[q]);
}

// Stable inverse of the 2x2 pivot [d00 d10; d10 d11] (as ldlt_factor_front).
inline void inv_2x2(double d00, double d10, double d11, double &i00, double &i10,
                    double &i11) noexcept {
    const double s = std::max({std::abs(d00), std::abs(d10), std::abs(d11)});
    const double a = d00 / s;
    const double b = d10 / s;
    const double c = d11 / s;
    const double inv = 1.0 / ((a * c - b * b) * s);
    i00 = c * inv;
    i10 = -b * inv;
    i11 = a * inv;
}

// ---------------------------------------------------------------------------
// Complete pivoting (Alg. 2) on an n x n view: the largest remaining entry is
// a 1x1 if diagonal; else the 2x2 it spans if |det| >= |a_mt|^2 / 2; else a
// 1x1 on the larger of its two diagonals. Stops when the remaining block is
// below `small` or a pivot would put an |l| > 1/u into the view.
// ---------------------------------------------------------------------------
int cp_factor(double *A, int ld, int n, int8_t *tags, Int *rows, int *lp,
              const Ctx &cx) noexcept {
    for (int t = 0; t < n; ++t) {
        lp[t] = t;
        tags[t] = 0;
    }
    int k = 0;
    while (k < n) {
        double amax = -1.0;
        int bi = -1;
        int bj = -1;
        for (int j = k; j < n; ++j) {
            const double *col = A + idx(j, ld);
            for (int i = j; i < n; ++i) {
                const double v = std::abs(col[i]);
                if (v > amax) {
                    amax = v;
                    bi = i;
                    bj = j;
                }
            }
        }
        if (!(amax >= cx.small) || !is_finite_fast(amax)) break;

        int kind = 1;
        int a = bj;
        int b = -1;
        if (bi != bj) {
            const double amm = A[idx(bj, ld) + bj];
            const double att = A[idx(bi, ld) + bi];
            const double amt = A[idx(bj, ld) + bi];
            const double delta = amm * att - amt * amt;
            if (std::abs(delta) >= 0.5 * amt * amt) {
                kind = 2;
                b = bi; // a = bj < b
            } else {
                a = (std::abs(att) > std::abs(amm)) ? bi : bj;
            }
        }

        // A posteriori bound inside the view (complete pivoting gives |l| <= 4
        // in exact arithmetic; checked so u > 0.25 and rounding are safe).
        auto sym_at = [&](int i, int j) {
            return (i >= j) ? A[idx(j, ld) + i] : A[idx(i, ld) + j];
        };
        bool ok = true;
        if (kind == 1) {
            const double d = A[idx(a, ld) + a];
            const double lim = cx.inv_u * std::abs(d);
            for (int r = k; r < n && ok; ++r)
                if (r != a && !(std::abs(sym_at(r, a)) <= lim)) ok = false;
        } else {
            double i00, i10, i11;
            inv_2x2(A[idx(a, ld) + a], A[idx(a, ld) + b], A[idx(b, ld) + b], i00, i10, i11);
            for (int r = k; r < n && ok; ++r) {
                if (r == a || r == b) continue;
                const double x0 = sym_at(r, a);
                const double x1 = sym_at(r, b);
                if (!l_ok(i00 * x0 + i10 * x1, cx.inv_u) || !l_ok(i10 * x0 + i11 * x1, cx.inv_u))
                    ok = false;
            }
        }
        if (!ok) break;

        if (kind == 1) {
            view_swap(A, ld, n, k, a, tags, rows, lp);
            double *ck = A + idx(k, ld);
            const double inv_d = 1.0 / ck[k];
            for (int j = k + 1; j < n; ++j) {
                const double lj = ck[j] * inv_d;
                if (lj == 0.0) continue;
                double *cj = A + idx(j, ld);
                for (int i = j; i < n; ++i) cj[i] -= ck[i] * lj;
            }
            for (int i = k + 1; i < n; ++i) ck[i] *= inv_d;
            tags[k] = 1;
            ++k;
        } else {
            int partner = b;
            if (a != k) {
                view_swap(A, ld, n, k, a, tags, rows, lp);
                if (partner == k) partner = a;
            }
            if (partner != k + 1) view_swap(A, ld, n, k + 1, partner, tags, rows, lp);
            double *c0 = A + idx(k, ld);
            double *c1 = A + idx(k + 1, ld);
            double i00, i10, i11;
            inv_2x2(c0[k], c0[k + 1], c1[k + 1], i00, i10, i11);
            for (int j = k + 2; j < n; ++j) {
                const double a0 = c0[j];
                const double a1 = c1[j];
                const double l0 = i00 * a0 + i10 * a1;
                const double l1 = i10 * a0 + i11 * a1;
                if (l0 == 0.0 && l1 == 0.0) continue;
                double *cj = A + idx(j, ld);
                for (int i = j; i < n; ++i) cj[i] -= c0[i] * l0 + c1[i] * l1;
            }
            for (int i = k + 2; i < n; ++i) {
                const double a0 = c0[i];
                const double a1 = c1[i];
                c0[i] = i00 * a0 + i10 * a1;
                c1[i] = i10 * a0 + i11 * a1;
            }
            tags[k] = 2;
            tags[k + 1] = -1;
            k += 2;
        }
    }
    return k;
}

// Row permutation of rows [r0, r0 + w) in columns [0, ncols): new row r0 + t
// takes old row r0 + perm[t].
void permute_rows(double *A, int ld, int ncols, int r0, int w, const int *perm,
                  double *buf) noexcept {
    for (int c = 0; c < ncols; ++c) {
        double *col = A + idx(c, ld) + r0;
        for (int t = 0; t < w; ++t) buf[t] = col[perm[t]];
        std::memcpy(col, buf, sizeof(double) * static_cast<std::size_t>(w));
    }
}

bool is_identity(const int *perm, int w) noexcept {
    for (int t = 0; t < w; ++t)
        if (perm[t] != t) return false;
    return true;
}

// L = X D^{-1} in place on the first k columns of X (rows x k, ld ldx), pivot
// by pivot, stopping at the first pivot with an |l| > 1/u. Returns the start
// of that pivot (k if none failed).
int scale_and_check(double *X, int nrows, int ldx, int k, const int8_t *tags,
                    const double *dinv, double inv_u) noexcept {
    for (int t = 0; t < k; ++t) {
        double *x0 = X + idx(t, ldx);
        if (tags[t] == 2) {
            double *x1 = x0 + ldx;
            const double i00 = dinv[2 * t];
            const double i10 = dinv[2 * t + 1];
            const double i11 = dinv[2 * (t + 1)];
            bool ok = true;
            for (int r = 0; r < nrows; ++r) {
                const double a0 = x0[r];
                const double a1 = x1[r];
                const double l0 = i00 * a0 + i10 * a1;
                const double l1 = i10 * a0 + i11 * a1;
                x0[r] = l0;
                x1[r] = l1;
                ok = ok && l_ok(l0, inv_u) && l_ok(l1, inv_u);
            }
            if (!ok) return t;
            ++t;
        } else {
            const double id = dinv[2 * t];
            bool ok = true;
            for (int r = 0; r < nrows; ++r) {
                const double l = x0[r] * id;
                x0[r] = l;
                ok = ok && l_ok(l, inv_u);
            }
            if (!ok) return t;
        }
    }
    return k;
}

// One row of L D for the pivots [0, k): out[t] = sum_s L(row, s) D(s, t), with
// L(row, s) = lrow[s * stride] and D read from the tile (diagonal at
// tile[t * ld + t], 2x2 off-diagonal at tile[t * ld + t + 1]).
inline void row_times_d(const double *lrow, std::size_t stride, int k,
                        const double *tile, int ld, const int8_t *tags,
                        double *out, std::size_t out_stride) noexcept {
    for (int t = 0; t < k; ++t) {
        const double d00 = tile[idx(t, ld) + t];
        if (tags[t] == 2) {
            const double d10 = tile[idx(t, ld) + t + 1];
            const double d11 = tile[idx(t + 1, ld) + t + 1];
            const double l0 = lrow[t * stride];
            const double l1 = lrow[(t + 1) * stride];
            out[t * out_stride] = l0 * d00 + l1 * d10;
            out[(t + 1) * out_stride] = l0 * d10 + l1 * d11;
            ++t;
        } else {
            out[t * out_stride] = lrow[t * stride] * d00;
        }
    }
}

// X := X L^{-T} for the k x k unit lower L (ld ldl), X rows x k (ld ldx):
// column blocks of 32 with a small TRSM each and a GEMM for the columns to
// their right (faster than one wide right-side TRSM in MKL, single thread).
void trsm_rlt(double *X, int nrows, int k, const double *L, int ldl, int ldx) {
    constexpr int kb = 32;
    if (nrows <= 0 || k <= 0) return;
    for (int j0 = 0; j0 < k; j0 += kb) {
        const int jb = std::min(kb, k - j0);
        smf_dtrsm_right_lower_transpose(X + idx(j0, ldx), nrows, jb, L + idx(j0, ldl) + j0, ldl,
                                        ldx);
        const int rest = k - j0 - jb;
        if (rest > 0)
            smf_dgemm_nt(X + idx(j0 + jb, ldx), nrows, rest, jb, X + idx(j0, ldx), ldx,
                         L + idx(j0, ldl) + j0 + jb, ldl, ldx, -1.0, 1.0);
    }
}

int factor_tile(double *A, int ld, int w, int8_t *tags, Int *rows, int *lp, Ctx &cx);

// ---------------------------------------------------------------------------
// APTP sweep over the leading `ne` columns of the m x m view (A, ld). level 0
// is the front (block nb, child factor_tile), level 1 a tile (block ib, child
// cp_factor). Columns [ne, m) are not pivot candidates; with m > ne (front
// level only) their trailing triangle gets one deferred update at the end.
// On return the eliminated pivots occupy [0, nelim) and lp[t] is the view
// position that moved to t.
// ---------------------------------------------------------------------------
int sweep(double *A, int ld, int m, int ne, int level, int8_t *tags, Int *rows,
          int *lp, Ctx &cx) {
    const int bs = (level == 0) ? cx.nb : cx.ib;
    for (int t = 0; t < ne; ++t) {
        lp[t] = t;
        tags[t] = 0;
    }
    int *Q = cx.q[level];
    int nq = 0;
    int num_elim = 0;
    double *bk = cx.bk[level];
    int *lpc = cx.lpc[level];

    for (int c0 = 0; c0 < ne; c0 += bs) {
        const int w = std::min(bs, ne - c0);
        const int ldb = m - c0;
        const int rb = m - c0 - w; // rows below the tile

        // 1. Backup: column c0 + t, rows [c0 + t, m).
        for (int t = 0; t < w; ++t)
            std::memcpy(bk + idx(t, ldb) + t, A + idx(c0 + t, ld) + c0 + t,
                        sizeof(double) * static_cast<std::size_t>(m - c0 - t));

        // 2. Factor the diagonal tile.
        double *T = A + idx(c0, ld) + c0;
        const int ne_t = (level == 0) ? factor_tile(T, ld, w, tags + c0, rows + c0, lpc, cx)
                                      : cp_factor(T, ld, w, tags + c0, rows + c0, lpc, cx);

        // 3. Apply the tile permutation outside the tile.
        if (!is_identity(lpc, w)) {
            permute_rows(A, ld, c0, c0, w, lpc, cx.colbuf);
            for (int t = 0; t < w; ++t)
                std::memcpy(A + idx(c0 + t, ld) + c0 + w, bk + idx(lpc[t], ldb) + w,
                            sizeof(double) * static_cast<std::size_t>(rb));
            for (int t = 0; t < w; ++t) cx.itmp[t] = lp[c0 + lpc[t]];
            std::memcpy(lp + c0, cx.itmp, sizeof(int) * static_cast<std::size_t>(w));
        }

        int nelim = ne_t;
        if (ne_t > 0) {
            // 4. L_t (unit lower, 2x2 off-diagonals zeroed) and D^{-1}.
            double *lt = cx.lt;
            for (int j = 0; j < ne_t; ++j) {
                double *lj = lt + idx(j, bs);
                lj[j] = 1.0;
                const double *tj = T + idx(j, ld);
                for (int i = j + 1; i < ne_t; ++i) lj[i] = tj[i];
            }
            for (int j = 0; j < ne_t; ++j) {
                const double *tj = T + idx(j, ld);
                if (tags[c0 + j] == 2) {
                    lt[idx(j, bs) + j + 1] = 0.0;
                    double i00, i10, i11;
                    inv_2x2(tj[j], tj[j + 1], T[idx(j + 1, ld) + j + 1], i00, i10, i11);
                    cx.dinv[2 * j] = i00;
                    cx.dinv[2 * j + 1] = i10;
                    cx.dinv[2 * (j + 1)] = i11;
                    cx.dinv[2 * (j + 1) + 1] = 0.0;
                    ++j;
                } else {
                    cx.dinv[2 * j] = 1.0 / tj[j];
                    cx.dinv[2 * j + 1] = 0.0;
                }
            }

            // ApplyN: rows below the tile, in place.
            if (rb > 0) {
                double *X = A + idx(c0, ld) + c0 + w;
                trsm_rlt(X, rb, ne_t, lt, bs, ld);
                nelim = scale_and_check(X, rb, ld, ne_t, tags + c0, cx.dinv, cx.inv_u);
            }

            // ApplyT: rows of earlier failed columns (stored transposed in
            // those columns), out of place so nothing needs restoring.
            for (int i0 = 0; i0 < nq && nelim > 0; i0 += bs) {
                const int nc = std::min(bs, nq - i0);
                for (int t = 0; t < nelim; ++t)
                    for (int ii = 0; ii < nc; ++ii)
                        cx.ga[idx(t, bs) + ii] = A[idx(Q[i0 + ii], ld) + c0 + t];
                trsm_rlt(cx.ga, nc, nelim, lt, bs, bs);
                nelim = scale_and_check(cx.ga, nc, bs, nelim, tags + c0, cx.dinv, cx.inv_u);
            }
            // Adjust: never keep only the first column of a 2x2.
            if (nelim > 0 && tags[c0 + nelim - 1] == 2) --nelim;
        }

        // 5. Restore the failed columns [nelim, w) from the backup.
        if (nelim < w) {
            auto bks = [&](int a, int b) {
                return (a >= b) ? bk[idx(b, ldb) + a] : bk[idx(a, ldb) + b];
            };
            for (int s2 = nelim; s2 < w; ++s2) {
                double *col = T + idx(s2, ld);
                for (int s1 = s2; s1 < w; ++s1) col[s1] = bks(lpc[s1], lpc[s2]);
                std::memcpy(col + w, bk + idx(lpc[s2], ldb) + w,
                            sizeof(double) * static_cast<std::size_t>(rb));
                tags[c0 + s2] = 0;
            }
        }

        if (nelim > 0) {
            // ApplyT, final: L rows of the earlier failed columns for the
            // accepted pivots, written back in place.
            for (int i0 = 0; i0 < nq; i0 += bs) {
                const int nc = std::min(bs, nq - i0);
                if (nq > bs) {
                    for (int t = 0; t < nelim; ++t)
                        for (int ii = 0; ii < nc; ++ii)
                            cx.ga[idx(t, bs) + ii] = A[idx(Q[i0 + ii], ld) + c0 + t];
                    trsm_rlt(cx.ga, nc, nelim, cx.lt, bs, bs);
                    scale_and_check(cx.ga, nc, bs, nelim, tags + c0, cx.dinv, cx.inv_u);
                } // else: ga still holds the single chunk from the check pass
                for (int t = 0; t < nelim; ++t)
                    for (int ii = 0; ii < nc; ++ii)
                        A[idx(Q[i0 + ii], ld) + c0 + t] = cx.ga[idx(t, bs) + ii];
            }

            // 6a. UpdateNN: eligible columns [e0, ne), rows to m. L D for
            // those columns goes, transposed, into the unused strict upper
            // triangle rows [c0, c0 + nelim) of the same columns.
            const int e0 = c0 + nelim;
            const int msz = ne - e0;
            if (msz > 0) {
                for (int col = e0; col < ne; ++col)
                    row_times_d(A + idx(c0, ld) + col, static_cast<std::size_t>(ld), nelim, T,
                                ld, tags + c0, A + idx(col, ld) + c0, 1);
                const double *Wt = A + idx(e0, ld) + c0;
                smf_dgemmt_lower_nn_sub(A + idx(e0, ld) + e0, msz, nelim, A + idx(c0, ld) + e0,
                                        ld, Wt, ld, ld);
                if (m > ne)
                    smf_dgemm(A + idx(e0, ld) + ne, m - ne, msz, nelim, A + idx(c0, ld) + ne, ld,
                              Wt, ld, ld, -1.0, 1.0);
            }

            // 6b. UpdateNT / UpdateTN: earlier failed columns q, rows [e0, m)
            // and rows of other failed columns q' >= q.
            const int mr = m - e0;
            for (int j0 = 0; j0 < nq; j0 += bs) {
                const int nj = std::min(bs, nq - j0);
                for (int jj = 0; jj < nj; ++jj)
                    row_times_d(A + idx(Q[j0 + jj], ld) + c0, 1, nelim, T, ld, tags + c0,
                                cx.gb + jj, static_cast<std::size_t>(bs));
                if (mr > 0) {
                    double *Tm = bk; // backup no longer needed
                    smf_dgemm_nt(Tm, mr, nj, nelim, A + idx(c0, ld) + e0, ld, cx.gb, bs, mr, 1.0,
                                 0.0);
                    for (int jj = 0; jj < nj; ++jj) {
                        double *dst = A + idx(Q[j0 + jj], ld) + e0;
                        const double *src = Tm + idx(jj, mr);
                        for (int r = 0; r < mr; ++r) dst[r] -= src[r];
                    }
                }
                for (int i0 = j0; i0 < nq; i0 += bs) {
                    const int ni = std::min(bs, nq - i0);
                    for (int t = 0; t < nelim; ++t)
                        for (int ii = 0; ii < ni; ++ii)
                            cx.ga[idx(t, bs) + ii] = A[idx(Q[i0 + ii], ld) + c0 + t];
                    smf_dgemm_nt(cx.gc, ni, nj, nelim, cx.ga, bs, cx.gb, bs, bs, 1.0, 0.0);
                    for (int jj = 0; jj < nj; ++jj) {
                        const int qj = Q[j0 + jj];
                        double *colq = A + idx(qj, ld);
                        for (int ii = 0; ii < ni; ++ii) {
                            const int qi = Q[i0 + ii];
                            if (qi >= qj) colq[qi] -= cx.gc[idx(jj, bs) + ii];
                        }
                    }
                }
            }
        }

        num_elim += nelim;
        for (int s = nelim; s < w; ++s) Q[nq++] = c0 + s;
    }

    // Move the failed columns behind the eliminated ones (eliminated order
    // kept, so 2x2 pairs stay adjacent).
    if (num_elim < ne) {
        int dst = 0;
        for (int src = 0; src < ne; ++src) {
            if (tags[src] == 0) continue;
            if (src != dst) view_swap(A, ld, m, dst, src, tags, rows, lp);
            ++dst;
        }
    }

    // Deferred update of the non-eligible trailing triangle, at rank num_elim:
    // F[ne:m, ne:m] -= L D L^T, with (L D)^T formed in the strict upper
    // triangle F[0:p, ne:m].
    const int qn = m - ne;
    if (qn > 0 && num_elim > 0) {
        for (int c = 0; c < num_elim; ++c) {
            const double *Lc = A + idx(c, ld);
            if (tags[c] == 2) {
                const double *Lc1 = Lc + ld;
                const double d00 = Lc[c], d10 = Lc[c + 1], d11 = Lc1[c + 1];
                for (int i = ne; i < m; ++i) {
                    double *row = A + idx(i, ld);
                    row[c] = Lc[i] * d00 + Lc1[i] * d10;
                    row[c + 1] = Lc[i] * d10 + Lc1[i] * d11;
                }
                ++c;
            } else {
                const double d = Lc[c];
                for (int i = ne; i < m; ++i) A[idx(i, ld) + c] = Lc[i] * d;
            }
        }
        smf_dgemmt_lower_nn_sub(A + idx(ne, ld) + ne, qn, num_elim, A + ne, ld, A + idx(ne, ld), ld,
                                ld);
    }
    return num_elim;
}

// Diagonal tile of the outer sweep: inner APTP with block ib (complete
// pivoting on its diagonal blocks), then complete pivoting on the columns
// that failed inside the tile, which are current and at the back.
int factor_tile(double *A, int ld, int w, int8_t *tags, Int *rows, int *lp, Ctx &cx) {
    if (w <= cx.ib) return cp_factor(A, ld, w, tags, rows, lp, cx);
    int ne1 = sweep(A, ld, w, w, 1, tags, rows, lp, cx);
    if (ne1 < w) {
        const int n2 = w - ne1;
        int *lp2 = cx.cp_lp;
        const int ne2 = cp_factor(A + idx(ne1, ld) + ne1, ld, n2, tags + ne1, rows + ne1, lp2, cx);
        if (!is_identity(lp2, n2)) {
            permute_rows(A, ld, ne1, ne1, n2, lp2, cx.colbuf);
            for (int t = 0; t < n2; ++t) cx.itmp[t] = lp[ne1 + lp2[t]];
            std::memcpy(lp + ne1, cx.itmp, sizeof(int) * static_cast<std::size_t>(n2));
        }
        ne1 += ne2;
    }
    return ne1;
}

struct Sizes {
    int nb;
    int ib;
};

Sizes effective_sizes(int num_elig, const AptpParams &prm) noexcept {
    const int ne = std::max(1, num_elig);
    const int nb = std::min(std::max(1, prm.nb), ne);
    const int ib = std::min(std::max(1, prm.ib), nb);
    return {nb, ib};
}

} // namespace

std::size_t aptp_work_size(int f, int num_elig, const AptpParams &prm) {
    const Sizes s = effective_sizes(num_elig, prm);
    const std::size_t F = static_cast<std::size_t>(std::max(f, 1));
    const std::size_t nb = static_cast<std::size_t>(s.nb);
    const std::size_t ib = static_cast<std::size_t>(s.ib);
    const std::size_t ne = static_cast<std::size_t>(std::max(num_elig, 1));
    const std::size_t doubles = F * nb + nb * ib + 4 * nb * nb + 2 * nb + F;
    const std::size_t ints = 2 * ne + 4 * nb + ib;
    return doubles + (ints * sizeof(int) + sizeof(double) - 1) / sizeof(double) + 1;
}

AptpResult ldlt_aptp_front(double *F, int f, int num_elig, Int *rows, const AptpParams &prm,
                           int8_t *tags, double *work) {
    AptpResult res;
    if (f <= 0 || num_elig <= 0) {
        res.num_failed = std::max(0, num_elig);
        return res;
    }
    const Sizes s = effective_sizes(num_elig, prm);
    Ctx cx;
    cx.inv_u = 1.0 / prm.u;
    cx.small = prm.small;
    cx.nb = s.nb;
    cx.ib = s.ib;
    const std::size_t nb = static_cast<std::size_t>(s.nb);
    const std::size_t ib = static_cast<std::size_t>(s.ib);
    double *p = work;
    cx.bk[0] = p;
    p += static_cast<std::size_t>(f) * nb;
    cx.bk[1] = p;
    p += nb * ib;
    cx.lt = p;
    p += nb * nb;
    cx.ga = p;
    p += nb * nb;
    cx.gb = p;
    p += nb * nb;
    cx.gc = p;
    p += nb * nb;
    cx.dinv = p;
    p += 2 * nb;
    cx.colbuf = p;
    p += static_cast<std::size_t>(f);
    int *ip = reinterpret_cast<int *>(p);
    cx.top_lp = ip;
    ip += num_elig;
    cx.q[0] = ip;
    ip += num_elig;
    cx.lpc[0] = ip;
    ip += nb;
    cx.q[1] = ip;
    ip += nb;
    cx.lpc[1] = ip;
    ip += ib;
    cx.cp_lp = ip;
    ip += nb;
    cx.itmp = ip;

    const int pe = sweep(F, f, f, num_elig, 0, tags, rows, cx.top_lp, cx);
    res.num_eliminated = pe;
    res.num_failed = num_elig - pe;

    const std::size_t sf = static_cast<std::size_t>(f);
    for (int k = 0; k < pe; ++k) {
        if (tags[k] == 2) {
            add_inertia_2x2(F[k * sf + k], F[k * sf + k + 1], F[(k + 1) * sf + k + 1], prm.small,
                            res.inertia);
            const double rem = static_cast<double>(f - (k + 2));
            res.flops += 6.0 * rem + 2.0 * rem * (rem + 1.0);
            ++k;
        } else {
            add_inertia_1x1(F[k * sf + k], prm.small, res.inertia);
            const double rem = static_cast<double>(f - (k + 1));
            res.flops += rem + rem * (rem + 1.0);
        }
    }
    return res;
}

} // namespace smf
