#include "smf/dense_indef.hpp"
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

        if (s <= 0.0)
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
    if (std::abs(d) < small)
        ++out.zero;
    else if (d > 0.0)
        ++out.positive;
    else
        ++out.negative;
}

// ---------------------------------------------------------------------------
// add_inertia_2x2
// Signs inferred from trace / determinant of [[d00,d10],[d10,d11]].
// ---------------------------------------------------------------------------
void add_inertia_2x2(double d00, double d10, double d11, double small,
                     InertiaCounts& out) noexcept
{
    const double det  = d00 * d11 - d10 * d10;
    const double trc  = d00 + d11;
    const double norm = std::abs(d00) + std::abs(d11) + std::abs(d10);

    if (det < -small * norm) {
        // One positive, one negative eigenvalue.
        ++out.positive;
        ++out.negative;
    } else if (std::abs(det) <= small * norm) {
        // One zero eigenvalue; the other has the sign of the trace.
        ++out.zero;
        if (trc >= 0.0) ++out.positive;
        else            ++out.negative;
    } else {
        // det > 0: both eigenvalues have the same sign (= sign of trace).
        if      (trc > 0.0) out.positive += 2;
        else if (trc < 0.0) out.negative += 2;
        else                out.zero     += 2;
    }
}

// ---------------------------------------------------------------------------
// dense_ldlt_indef
// Bounded Bunch-Kaufman threshold pivoting, alpha = (1+sqrt(17))/8 ~ 0.6404.
// col-major storage: (i,j) -> A[i + j*lda], lower triangle only.
// ---------------------------------------------------------------------------
FactorStatus dense_ldlt_indef(double* A, int n, int lda, double u, double small,
                               std::vector<Pivot>& pivots,
                               InertiaCounts& inertia) noexcept
{
    bool singular = false;

    // Symmetric swap of rows/cols p and q within the lower-triangular active
    // submatrix.  cur is the current factorization position (column index at
    // which the active block begins).  Requires cur <= p < q < n.
    auto sym_swap = [&](int cur, int p, int q) {
        std::swap(A[p + p * lda], A[q + q * lda]);
        for (int m = cur; m < p; ++m)
            std::swap(A[p + m * lda], A[q + m * lda]);
        for (int m = p + 1; m < q; ++m)
            std::swap(A[m + p * lda], A[q + m * lda]);
        for (int m = q + 1; m < n; ++m)
            std::swap(A[m + p * lda], A[m + q * lda]);
    };

    int k = 0;
    while (k < n) {

        // ----------------------------------------------------------------
        // Step 1: find r = argmax_{i in [k+1,n)} |A[i + k*lda]|; omega1.
        // ----------------------------------------------------------------
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

        // ----------------------------------------------------------------
        // Step 2: 1×1 pivot at k if diagonal dominates.
        // ----------------------------------------------------------------
        if (omega1 == 0.0 || akk >= u * omega1) {
            const double d = A[k + k * lda];
            add_inertia_1x1(d, small, inertia);
            pivots.push_back(Pivot{PivotType::OneByOne, k, -1, d, 0.0, 0.0});

            if (std::abs(d) >= small) {
                for (int i = k + 1; i < n; ++i)
                    A[i + k * lda] /= d;
                for (int j = k + 1; j < n; ++j)
                    for (int i = j; i < n; ++i)
                        A[i + j * lda] -= A[i + k * lda] * d * A[j + k * lda];
            } else {
                singular = true;
            }
            ++k;

        } else {
            // ----------------------------------------------------------------
            // Step 3: compute omega2 = max_{i in [k,n), i != r}
            //         |A[max(i,r) + min(i,r)*lda]|.
            // ----------------------------------------------------------------
            double omega2 = 0.0;
            for (int i = k; i < n; ++i) {
                if (i == r) continue;
                const int lo = (i < r) ? i : r;
                const int hi = (i < r) ? r : i;
                const double v = std::abs(A[hi + lo * lda]);
                if (v > omega2) omega2 = v;
            }

            if (akk * omega2 >= u * omega1 * omega1) {
                // swap(k, r) then 1×1 at k
                sym_swap(k, k, r);
                const double d = A[k + k * lda];
                add_inertia_1x1(d, small, inertia);
                pivots.push_back(Pivot{PivotType::OneByOne, k, -1, d, 0.0, 0.0});

                if (std::abs(d) >= small) {
                    for (int i = k + 1; i < n; ++i)
                        A[i + k * lda] /= d;
                    for (int j = k + 1; j < n; ++j)
                        for (int i = j; i < n; ++i)
                            A[i + j * lda] -= A[i + k * lda] * d * A[j + k * lda];
                } else {
                    singular = true;
                }
                ++k;

            } else {
                // 2×2 pivot at (k, k+1); swap r into position k+1 if needed.
                if (r != k + 1)
                    sym_swap(k, k + 1, r);

                const double d00 = A[ k        +  k      * lda];
                const double d10 = A[(k + 1)   +  k      * lda];
                const double d11 = A[(k + 1)   + (k + 1) * lda];

                add_inertia_2x2(d00, d10, d11, small, inertia);
                pivots.push_back(
                    Pivot{PivotType::TwoByTwo, k, k + 1, d00, d10, d11});

                const double det = d00 * d11 - d10 * d10;
                if (std::abs(det) > small) {
                    const double inv_det = 1.0 / det;

                    // Compute L columns in-place (store into A[i+k*lda] and
                    // A[i+(k+1)*lda] for i in [k+2, n)).
                    for (int i = k + 2; i < n; ++i) {
                        const double ai0 = A[i +  k      * lda];
                        const double ai1 = A[i + (k + 1) * lda];
                        A[i +  k      * lda] = ( d11 * ai0 - d10 * ai1) * inv_det;
                        A[i + (k + 1) * lda] = (-d10 * ai0 + d00 * ai1) * inv_det;
                    }

                    // Schur complement update using the computed L columns.
                    for (int j = k + 2; j < n; ++j) {
                        for (int i = j; i < n; ++i) {
                            A[i + j * lda] -=
                                  A[i +  k      * lda] * d00 * A[j +  k      * lda]
                                + A[i +  k      * lda] * d10 * A[j + (k + 1) * lda]
                                + A[i + (k + 1) * lda] * d10 * A[j +  k      * lda]
                                + A[i + (k + 1) * lda] * d11 * A[j + (k + 1) * lda];
                        }
                    }
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
