#include "smf/dense_indef.hpp"
#include "smf/pivoting.hpp"
#include <algorithm>
#include <cmath>

namespace smf {

// ---------------------------------------------------------------------------
// dense_cholesky_lower
// Standard left-looking Cholesky; col-major storage: (i,j) -> A[i + j*lda].
// ---------------------------------------------------------------------------
FactorStatus dense_cholesky_lower(double* A, int n, int lda) noexcept
{
    for (int j = 0; j < n; ++j) {
        double s = A[j + j * lda];
        for (int col = 0; col < j; ++col)
            s -= A[j + col * lda] * A[j + col * lda];

        if (s <= 0.0 || !std::isfinite(s))
            return FactorStatus::NotPositiveDefinite;

        A[j + j * lda] = std::sqrt(s);
        const double inv_diag = 1.0 / A[j + j * lda];

        for (int i = j + 1; i < n; ++i) {
            double val = A[i + j * lda];
            for (int col = 0; col < j; ++col)
                val -= A[i + col * lda] * A[j + col * lda];
            A[i + j * lda] = val * inv_diag;
        }
    }
    return FactorStatus::Success;
}

// ---------------------------------------------------------------------------
// add_inertia_1x1
// ---------------------------------------------------------------------------
void add_inertia_1x1(double d, double small, InertiaCounts& out) noexcept
{
    if (!std::isfinite(d) || std::abs(d) < small)
        ++out.zero;
    else if (d > 0.0)
        ++out.positive;
    else
        ++out.negative;
}

// ---------------------------------------------------------------------------
// add_inertia_2x2
// Scale-normalized eigenvalue sign classification for [[d00,d10],[d10,d11]].
// ---------------------------------------------------------------------------
void add_inertia_2x2(double d00, double d10, double d11, double small,
                     InertiaCounts& out) noexcept
{
    if (d10 == 0.0) {
        add_inertia_1x1(d00, small, out);
        add_inertia_1x1(d11, small, out);
        return;
    }

    double s = 0.0;
    const double det_s = scaled_det_2x2(d00, d10, d11, s);
    if (s == 0.0 || !std::isfinite(det_s)) {
        out.zero += 2;
        return;
    }

    const double tr_s = (d00 / s) + (d11 / s);

    if (det_s < -small) {
        ++out.positive;
        ++out.negative;
    } else if (std::abs(det_s) <= small) {
        ++out.zero;
        if (tr_s > small)       ++out.positive;
        else if (tr_s < -small) ++out.negative;
        else                    ++out.zero;
    } else {
        if      (tr_s > 0.0) out.positive += 2;
        else if (tr_s < 0.0) out.negative += 2;
        else                 out.zero     += 2;
    }
}

// ---------------------------------------------------------------------------
// dense_ldlt_indef
// Threshold pivoting with scaled 2×2 arithmetic.
// col-major storage: (i,j) -> A[i + j*lda], lower triangle only.
// ---------------------------------------------------------------------------
FactorStatus dense_ldlt_indef(double* A, int n, int lda, double u, double small,
                               std::vector<Pivot>& pivots,
                               InertiaCounts& inertia) noexcept
{
    bool singular = false;

    double max_abs = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            const double v = std::abs(A[i + j * lda]);
            if (v > max_abs) max_abs = v;
        }
    }
    const double eff_small =
        (max_abs > 0.0 && max_abs < 1.0) ? (small * max_abs) : small;

    // Symmetric swap of rows/cols p and q across the full lower-triangular matrix
    // (including already-computed L columns in [0, p)). Requires 0 <= p < q < n.
    auto sym_swap = [&](int p, int q) {
        if (p == q) return;
        if (p > q) std::swap(p, q);
        std::swap(A[p + p * lda], A[q + q * lda]);
        for (int m = 0; m < p; ++m)
            std::swap(A[p + m * lda], A[q + m * lda]);
        for (int m = p + 1; m < q; ++m)
            std::swap(A[m + p * lda], A[q + m * lda]);
        for (int m = q + 1; m < n; ++m)
            std::swap(A[m + p * lda], A[m + q * lda]);
    };

    int k = 0;
    while (k < n) {
        int    r      = k + 1;
        double omega1 = 0.0;
        if (k + 1 < n) {
            omega1 = std::abs(A[(k + 1) + k * lda]);
            r      = k + 1;
            for (int i = k + 2; i < n; ++i) {
                const double v = std::abs(A[i + k * lda]);
                if (v > omega1) { omega1 = v; r = i; }
            }
        }

        const double akk = std::abs(A[k + k * lda]);

        if (omega1 == 0.0 || (akk >= eff_small && akk >= u * omega1)) {
            const double d = A[k + k * lda];
            add_inertia_1x1(d, eff_small, inertia);
            pivots.push_back(Pivot{PivotType::OneByOne, k, -1, d, 0.0, 0.0});

            if (std::abs(d) >= eff_small) {
                for (int i = k + 1; i < n; ++i)
                    A[i + k * lda] /= d;
                for (int j = k + 1; j < n; ++j) {
                    const double w_jk = d * A[j + k * lda];
                    for (int i = j; i < n; ++i)
                        A[i + j * lda] -= A[i + k * lda] * w_jk;
                }
            } else {
                singular = true;
            }
            ++k;

        } else {
            double omega2 = 0.0;
            for (int i = k; i < n; ++i) {
                if (i == r) continue;
                const int lo = (i < r) ? i : r;
                const int hi = (i < r) ? r : i;
                const double v = std::abs(A[hi + lo * lda]);
                if (v > omega2) omega2 = v;
            }

            const double arr = std::abs(A[r + r * lda]);
            if (arr >= eff_small && arr >= u * omega2) {
                sym_swap(k, r);
                const double d = A[k + k * lda];
                add_inertia_1x1(d, eff_small, inertia);
                pivots.push_back(Pivot{PivotType::OneByOne, r, -1, d, 0.0, 0.0});

                if (std::abs(d) >= eff_small) {
                    for (int i = k + 1; i < n; ++i)
                        A[i + k * lda] /= d;
                    for (int j = k + 1; j < n; ++j) {
                        const double w_jk = d * A[j + k * lda];
                        for (int i = j; i < n; ++i)
                            A[i + j * lda] -= A[i + k * lda] * w_jk;
                    }
                } else {
                    singular = true;
                }
                ++k;

            } else {
                if (r != k + 1)
                    sym_swap(k + 1, r);

                const double d00 = A[ k      +  k      * lda];
                const double d10 = A[(k + 1) +  k      * lda];
                const double d11 = A[(k + 1) + (k + 1) * lda];

                add_inertia_2x2(d00, d10, d11, small, inertia);
                pivots.push_back(
                    Pivot{PivotType::TwoByTwo, k, r, d00, d10, d11});

                double s2 = 0.0;
                const double det_s = scaled_det_2x2(d00, d10, d11, s2);
                if (s2 >= eff_small && std::abs(det_s) > small) {
                    for (int i = k + 2; i < n; ++i) {
                        const double ai0 = A[i +  k      * lda];
                        const double ai1 = A[i + (k + 1) * lda];
                        double l0 = 0.0;
                        double l1 = 0.0;
                        solve_2x2_scaled(d00, d10, d11, ai0, ai1, l0, l1);
                        A[i +  k      * lda] = l0;
                        A[i + (k + 1) * lda] = l1;
                    }

                    for (int j = k + 2; j < n; ++j) {
                        const double lj0 = A[j +  k      * lda];
                        const double lj1 = A[j + (k + 1) * lda];
                        const double wj0 = d00 * lj0 + d10 * lj1;
                        const double wj1 = d10 * lj0 + d11 * lj1;
                        for (int i = j; i < n; ++i) {
                            A[i + j * lda] -=
                                  A[i +  k      * lda] * wj0
                                + A[i + (k + 1) * lda] * wj1;
                        }
                    }
                    A[(k + 1) + k * lda] = 0.0;
                } else {
                    singular = true;
                }
                k += 2;
            }
        }
    }

    return singular ? FactorStatus::Singular : FactorStatus::Success;
}

} // namespace smf
