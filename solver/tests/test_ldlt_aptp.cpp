/// @file test_ldlt_aptp.cpp
/// APTP (a posteriori threshold pivoting) dense front kernel, ldlt_aptp_front.
///
///   - LdltAptp.MatchesTppOnRandomIndefinite : 200 random dense indefinite
///     fronts; reconstruction of P A P^T from L, D and the Schur complement,
///     the |l| <= 1/u bound, and the inertia against LAPACK dsyev.
///   - LdltAptp.AllColumnsFailOnZeroBlock    : zero eligible block, nonzero
///     off-diagonals: nothing is eliminated and F keeps its content.
///   - LdltAptp.SaddlePointUsesTwoByTwo      : [[0, B^T], [B, 0]] is fully
///     eliminated with 2x2 pivots only, inertia (50, 50, 0).
///   - LdltAptp.SchurMatchesReference        : the Schur complement equals
///     ldlt_factor_front's whenever both eliminate the same pivot set.
///   - LdltAptp.FailedColumnsKeepLayout      : KKT-like fronts with small
///     nb / ib, so failures and fail-in-place updates happen across tiles;
///     the layout contract still holds for the failed columns.

#include "smf/ldlt_aptp.hpp"
#include "smf/ldlt_kernel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(SMF_USE_MKL) && (SMF_USE_MKL)
#include <mkl.h>
#elif !(defined(SMF_BLAS_REFERENCE) && (SMF_BLAS_REFERENCE))
extern "C" void dsyev_(const char *jobz, const char *uplo, const int *n, double *a,
                       const int *lda, double *w, double *work, const int *lwork,
                       int *info);
#endif

using namespace smf;

namespace {

template <typename... Ts> std::string trace_msg(const Ts &...xs) {
    std::ostringstream os;
    (os << ... << xs);
    return os.str();
}

using Mat = std::vector<double>; // column-major f x f, lower triangle used

double &at(Mat &A, int f, int i, int j) { return A[static_cast<std::size_t>(j) * f + i]; }
double at(const Mat &A, int f, int i, int j) { return A[static_cast<std::size_t>(j) * f + i]; }
double sym(const Mat &A, int f, int i, int j) { return i >= j ? at(A, f, i, j) : at(A, f, j, i); }

Mat random_symmetric(int f, std::mt19937 &rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    Mat A(static_cast<std::size_t>(f) * f, 0.0);
    for (int j = 0; j < f; ++j)
        for (int i = j; i < f; ++i) at(A, f, i, j) = U(rng);
    return A;
}

/// Eigenvalue sign count of the leading n x n block of the lower-stored A.
InertiaCounts eigen_inertia(const Mat &A, int f, int n) {
    std::vector<double> a(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(j) * n + i] = sym(A, f, i, j);
    std::vector<double> w(n);
#if defined(SMF_USE_MKL) && (SMF_USE_MKL)
    const int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'L', n, a.data(), n, w.data());
#elif !(defined(SMF_BLAS_REFERENCE) && (SMF_BLAS_REFERENCE))
    int info = 0;
    int lwork = -1;
    double q = 0.0;
    dsyev_("N", "L", &n, a.data(), &n, w.data(), &q, &lwork, &info);
    lwork = static_cast<int>(q);
    std::vector<double> wk(static_cast<std::size_t>(std::max(1, lwork)));
    dsyev_("N", "L", &n, a.data(), &n, w.data(), wk.data(), &lwork, &info);
#else
    // Reference build: cyclic Jacobi.
    int info = 0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = j + 1; i < n; ++i) off += a[j * n + i] * a[j * n + i];
        if (off < 1e-30) break;
        for (int p = 0; p < n; ++p)
            for (int q2 = p + 1; q2 < n; ++q2) {
                const double apq = a[q2 * n + p];
                if (apq == 0.0) continue;
                const double th = (a[q2 * n + q2] - a[p * n + p]) / (2.0 * apq);
                const double t = (th >= 0 ? 1.0 : -1.0) / (std::abs(th) + std::sqrt(th * th + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = a[p * n + k], akq = a[q2 * n + k];
                    a[p * n + k] = c * akp - s * akq;
                    a[q2 * n + k] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = a[k * n + p], aqk = a[k * n + q2];
                    a[k * n + p] = c * apk - s * aqk;
                    a[k * n + q2] = s * apk + c * aqk;
                }
            }
    }
    for (int i = 0; i < n; ++i) w[i] = a[i * n + i];
#endif
    EXPECT_EQ(info, 0);
    const double scale = std::max(1.0, std::abs(w.empty() ? 0.0 : w.back()));
    InertiaCounts c;
    for (double v : w) {
        if (std::abs(v) <= 1e-12 * scale) ++c.zero;
        else if (v > 0.0) ++c.positive;
        else ++c.negative;
    }
    return c;
}

struct Run {
    Mat F;
    std::vector<Int> rows;
    std::vector<int8_t> tags;
    AptpResult res{};
};

Run run_aptp(const Mat &A, int f, int num_elig, const AptpParams &prm) {
    Run r;
    r.F = A;
    r.rows.resize(f);
    std::iota(r.rows.begin(), r.rows.end(), 0);
    r.tags.assign(static_cast<std::size_t>(std::max(1, f)), 0);
    std::vector<double> work(aptp_work_size(f, num_elig, prm) + 1);
    r.res = ldlt_aptp_front(r.F.data(), f, num_elig, r.rows.data(), prm, r.tags.data(), work.data());
    return r;
}

/// Max |l| over the L part of columns [0, p) (2x2 off-diagonal D excluded).
double max_abs_l(const Mat &F, int f, int p, const std::vector<int8_t> &tags) {
    double m = 0.0;
    for (int c = 0; c < p; ++c) {
        const int first = (tags[c] == 2) ? c + 2 : c + 1;
        for (int i = first; i < f; ++i) m = std::max(m, std::abs(at(F, f, i, c)));
    }
    return m;
}

/// ||P A P^T - (L D L^T + [0 0; 0 S])||_F / ||A||_F over the lower triangle,
/// where p = num_eliminated and S is the trailing lower triangle of F.
double reconstruction_error(const Mat &A, const Run &r, int f) {
    const int p = r.res.num_eliminated;
    const Mat &F = r.F;
    // LD = L * D (f x p), L unit lower in columns [0, p).
    std::vector<double> L(static_cast<std::size_t>(f) * std::max(p, 1), 0.0);
    std::vector<double> LD(static_cast<std::size_t>(f) * std::max(p, 1), 0.0);
    for (int c = 0; c < p; ++c) {
        L[static_cast<std::size_t>(c) * f + c] = 1.0;
        const int first = (r.tags[c] == 2) ? c + 2 : c + 1;
        for (int i = first; i < f; ++i) L[static_cast<std::size_t>(c) * f + i] = at(F, f, i, c);
        if (r.tags[c] == -1) L[static_cast<std::size_t>(c) * f + c - 1] = 0.0; // never set; keep clean
    }
    for (int c = 0; c < p; ++c) {
        if (r.tags[c] == 2) {
            const double d00 = at(F, f, c, c), d10 = at(F, f, c + 1, c), d11 = at(F, f, c + 1, c + 1);
            for (int i = 0; i < f; ++i) {
                const double l0 = L[static_cast<std::size_t>(c) * f + i];
                const double l1 = L[static_cast<std::size_t>(c + 1) * f + i];
                LD[static_cast<std::size_t>(c) * f + i] = l0 * d00 + l1 * d10;
                LD[static_cast<std::size_t>(c + 1) * f + i] = l0 * d10 + l1 * d11;
            }
            ++c;
        } else {
            const double d = at(F, f, c, c);
            for (int i = 0; i < f; ++i)
                LD[static_cast<std::size_t>(c) * f + i] = L[static_cast<std::size_t>(c) * f + i] * d;
        }
    }
    double num = 0.0, den = 0.0;
    std::vector<double> col(f);
    for (int j = 0; j < f; ++j) {
        std::fill(col.begin(), col.end(), 0.0);
        const int kmax = std::min(p, j + 2); // LD(j, j+1) != 0 for a 2x2 at j
        for (int k = 0; k < kmax; ++k) {
            const double ljk = LD[static_cast<std::size_t>(k) * f + j];
            if (ljk == 0.0) continue;
            const double *lk = &L[static_cast<std::size_t>(k) * f];
            for (int i = j; i < f; ++i) col[i] += lk[i] * ljk;
        }
        for (int i = j; i < f; ++i) {
            double rec = col[i];
            if (j >= p) rec += at(F, f, i, j);
            const double a = sym(A, f, r.rows[i], r.rows[j]);
            num += (a - rec) * (a - rec);
            den += a * a;
        }
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

void expect_valid_tags(const Run &r) {
    const int p = r.res.num_eliminated;
    for (int c = 0; c < p; ++c) {
        if (r.tags[c] == 2) {
            ASSERT_LT(c + 1, p) << "2x2 split across nelim";
            ASSERT_EQ(r.tags[c + 1], -1);
            ++c;
        } else {
            ASSERT_EQ(r.tags[c], 1) << "column " << c;
        }
    }
}

void expect_row_permutation(const Run &r, int f, int num_elig) {
    std::vector<Int> s(r.rows.begin(), r.rows.end());
    std::sort(s.begin(), s.end());
    for (int i = 0; i < f; ++i) ASSERT_EQ(s[i], i);
    for (int i = num_elig; i < f; ++i) ASSERT_EQ(r.rows[i], i) << "non-eligible row moved";
}

InertiaCounts inertia_from_d(const Run &r, int f) {
    InertiaCounts c;
    for (int k = 0; k < r.res.num_eliminated; ++k) {
        if (r.tags[k] == 2) {
            const double d00 = at(r.F, f, k, k), d10 = at(r.F, f, k + 1, k), d11 = at(r.F, f, k + 1, k + 1);
            const double det = d00 * d11 - d10 * d10;
            if (det < 0) { ++c.positive; ++c.negative; }
            else if (d00 + d11 > 0) c.positive += 2;
            else c.negative += 2;
            ++k;
        } else {
            (at(r.F, f, k, k) > 0 ? c.positive : c.negative)++;
        }
    }
    return c;
}

} // namespace

TEST(LdltAptp, MatchesTppOnRandomIndefinite) {
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> Df(8, 600);
    const AptpParams prm;
    int complete = 0;
    int multi_tile = 0;
    for (int t = 0; t < 200; ++t) {
        const int f = Df(rng);
        const int num_elig = std::uniform_int_distribution<int>(1, f)(rng);
        const Mat A = random_symmetric(f, rng);
        const Run r = run_aptp(A, f, num_elig, prm);
        SCOPED_TRACE(trace_msg("trial ", t, " f=", f, " elig=", num_elig));

        ASSERT_GE(r.res.num_eliminated, 0);
        ASSERT_EQ(r.res.num_eliminated + r.res.num_failed, num_elig);
        expect_valid_tags(r);
        expect_row_permutation(r, f, num_elig);
        EXPECT_LE(reconstruction_error(A, r, f), 1e-13);
        EXPECT_LE(max_abs_l(r.F, f, r.res.num_eliminated, r.tags), 1.0 / prm.u);
        const InertiaCounts dI = inertia_from_d(r, f);
        EXPECT_EQ(r.res.inertia.positive, dI.positive);
        EXPECT_EQ(r.res.inertia.negative, dI.negative);
        EXPECT_GT(r.res.flops, 0.0);

        if (r.res.num_failed == 0) {
            // All eligible columns eliminated: by Sylvester's law the pivots'
            // inertia is that of the leading num_elig x num_elig block of A.
            ++complete;
            const InertiaCounts ev = eigen_inertia(A, f, num_elig);
            EXPECT_EQ(r.res.inertia.positive, ev.positive);
            EXPECT_EQ(r.res.inertia.negative, ev.negative);
            EXPECT_EQ(r.res.inertia.zero, ev.zero);
        }
        if (num_elig > prm.nb) ++multi_tile;
    }
    EXPECT_GE(complete, 150) << "too few fully eliminated fronts for the inertia check";
    EXPECT_GE(multi_tile, 50);
}

TEST(LdltAptp, AllColumnsFailOnZeroBlock) {
    std::mt19937 rng(5);
    for (const auto &[f, num_elig] : {std::pair{60, 20}, std::pair{400, 300}}) {
        Mat A = random_symmetric(f, rng);
        for (int j = 0; j < num_elig; ++j)
            for (int i = j; i < num_elig; ++i) at(A, f, i, j) = 0.0;
        const AptpParams prm;
        const Run r = run_aptp(A, f, num_elig, prm);
        EXPECT_EQ(r.res.num_eliminated, 0);
        EXPECT_EQ(r.res.num_failed, num_elig);
        EXPECT_EQ(r.res.inertia.positive + r.res.inertia.negative + r.res.inertia.zero, 0);
        expect_row_permutation(r, f, num_elig);
        for (int j = 0; j < f; ++j)
            for (int i = j; i < f; ++i)
                ASSERT_EQ(at(r.F, f, i, j), sym(A, f, r.rows[i], r.rows[j]))
                    << "f=" << f << " (" << i << "," << j << ")";
    }
}

TEST(LdltAptp, SaddlePointUsesTwoByTwo) {
    const int nb = 50;
    const int f = 2 * nb;
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(-0.5, 0.5);
    Mat A(static_cast<std::size_t>(f) * f, 0.0);
    // B well conditioned: 4 I + small random perturbation.
    for (int i = 0; i < nb; ++i)
        for (int j = 0; j < nb; ++j) at(A, f, nb + i, j) = (i == j ? 4.0 : 0.0) + U(rng) / nb;
    const AptpParams prm;
    const Run r = run_aptp(A, f, f, prm);
    ASSERT_EQ(r.res.num_eliminated, f);
    EXPECT_EQ(r.res.num_failed, 0);
    for (int c = 0; c < f; c += 2) {
        EXPECT_EQ(r.tags[c], 2) << c;
        EXPECT_EQ(r.tags[c + 1], -1) << c;
    }
    EXPECT_EQ(r.res.inertia.positive, nb);
    EXPECT_EQ(r.res.inertia.negative, nb);
    EXPECT_EQ(r.res.inertia.zero, 0);
    EXPECT_LE(reconstruction_error(A, r, f), 1e-13);
    EXPECT_LE(max_abs_l(r.F, f, f, r.tags), 1.0 / prm.u);
}

TEST(LdltAptp, SchurMatchesReference) {
    std::mt19937 rng(23);
    const AptpParams prm;
    int compared = 0;
    for (int t = 0; t < 60; ++t) {
        const int f = std::uniform_int_distribution<int>(20, 500)(rng);
        const int num_elig = std::uniform_int_distribution<int>(1, f - 1)(rng);
        // A forward-error check needs a well-conditioned eliminated block: the
        // Schur complement's sensitivity is cond(A11) times the backward
        // error, and on plain uniform fronts cond(A11) reaches 1e2-1e3 (both
        // kernels then reconstruct P A P^T to ~1e-14 but their Schur
        // complements differ by up to ~5e-12). Random signs on a dominant
        // diagonal keep it indefinite; every 5th diagonal is zero with a
        // strong neighbour entry so that 2x2 pivots occur too.
        Mat A = random_symmetric(f, rng);
        const double big = 1.5 * std::sqrt(static_cast<double>(f));
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        for (int i = 0; i < f; ++i) {
            const double sgn = (U01(rng) < 0.5) ? -1.0 : 1.0;
            at(A, f, i, i) = (i % 5 == 0) ? 0.0 : sgn * big * (1.0 + 0.5 * U01(rng));
            if (i % 5 == 0 && i + 1 < f) at(A, f, i + 1, i) = big;
        }
        SCOPED_TRACE(trace_msg("trial ", t, " f=", f, " elig=", num_elig));

        const Run r = run_aptp(A, f, num_elig, prm);

        Mat Fr = A;
        std::vector<Int> rows_r(f);
        std::iota(rows_r.begin(), rows_r.end(), 0);
        std::vector<int8_t> tags_r(f, 0);
        LdltKernelParams kp;
        kp.u = prm.u;
        std::vector<double> wk(ldlt_work_size(f, kp));
        const LdltKernelResult rr =
            ldlt_factor_front(Fr.data(), f, num_elig, rows_r.data(), kp, tags_r.data(), wk.data());

        const int p = r.res.num_eliminated;
        if (rr.num_accepted != p) continue;
        std::set<Int> s1(r.rows.begin(), r.rows.begin() + p);
        std::set<Int> s2(rows_r.begin(), rows_r.begin() + p);
        if (s1 != s2) continue;
        ++compared;

        // Position of each original row in each result (rows >= num_elig fixed).
        std::vector<int> pos_r(f);
        for (int i = 0; i < f; ++i) pos_r[rows_r[i]] = i;
        double num = 0.0, den = 0.0;
        for (int j = p; j < f; ++j)
            for (int i = j; i < f; ++i) {
                const int ri = pos_r[r.rows[i]], rj = pos_r[r.rows[j]];
                const double sref = sym(Fr, f, ri, rj);
                const double d = at(r.F, f, i, j) - sref;
                num += d * d;
                den += sref * sref;
            }
        EXPECT_LE(std::sqrt(num / std::max(den, 1e-300)), 1e-12);
        EXPECT_EQ(r.res.inertia.positive, rr.inertia.positive);
        EXPECT_EQ(r.res.inertia.negative, rr.inertia.negative);
    }
    EXPECT_GE(compared, 40);
}

TEST(LdltAptp, FailedColumnsKeepLayout) {
    // KKT-like fronts [[H, J^T], [J, 0]] in a random symmetric order, with a
    // weak H so that pivots fail the a posteriori test. Small nb / ib force
    // many tiles and inner blocks.
    std::mt19937 rng(41);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    int with_failures = 0;
    for (int t = 0; t < 40; ++t) {
        const int f = std::uniform_int_distribution<int>(30, 260)(rng);
        const int num_elig = std::uniform_int_distribution<int>(f / 2, f)(rng);
        const int np = (2 * f) / 3;
        Mat K(static_cast<std::size_t>(f) * f, 0.0);
        for (int j = 0; j < np; ++j)
            for (int i = j; i < np; ++i) at(K, f, i, j) = (i == j) ? 1e-3 * U(rng) : 1e-3 * U(rng);
        for (int j = 0; j < np; ++j)
            for (int i = np; i < f; ++i)
                if (std::uniform_int_distribution<int>(0, 3)(rng) == 0) at(K, f, i, j) = U(rng);
        std::vector<int> perm(f);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        Mat A(static_cast<std::size_t>(f) * f, 0.0);
        for (int j = 0; j < f; ++j)
            for (int i = j; i < f; ++i) at(A, f, i, j) = sym(K, f, perm[i], perm[j]);

        AptpParams prm;
        prm.nb = (t % 2 == 0) ? 16 : 24;
        prm.ib = (t % 3 == 0) ? 4 : 8;
        prm.u = (t % 4 == 0) ? 0.1 : 0.01;
        const Run r = run_aptp(A, f, num_elig, prm);
        SCOPED_TRACE(trace_msg("trial ", t, " f=", f, " elig=", num_elig));
        ASSERT_EQ(r.res.num_eliminated + r.res.num_failed, num_elig);
        if (r.res.num_failed > 0) ++with_failures;
        expect_valid_tags(r);
        expect_row_permutation(r, f, num_elig);
        EXPECT_LE(reconstruction_error(A, r, f), 1e-12);
        EXPECT_LE(max_abs_l(r.F, f, r.res.num_eliminated, r.tags), 1.0 / prm.u);
        const InertiaCounts dI = inertia_from_d(r, f);
        EXPECT_EQ(r.res.inertia.positive, dI.positive);
        EXPECT_EQ(r.res.inertia.negative, dI.negative);
    }
    EXPECT_GE(with_failures, 5);
}
