#include "smf/dense_indef.hpp"
#include "smf/ldlt_kernel.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace smf;

TEST(LDLT, SPD2x2) {
    // A = [[4,2],[2,3]], SPD, expect 1x1 pivots, inertia (2,0,0)
    double A[4]={4,2, 2,3}; // col-major
    std::vector<Pivot> piv; InertiaCounts in;
    auto s=dense_ldlt_indef(A,2,2,0.01,1e-20,piv,in);
    EXPECT_EQ(s,FactorStatus::Success);
    EXPECT_EQ(in.positive,2); EXPECT_EQ(in.negative,0); EXPECT_EQ(in.zero,0);
}
TEST(LDLT, Indefinite3x3) {
    // A = [[2,1,0],[1,0,1],[0,1,2]] col-major lower: A[0]=2,A[1]=1,A[2]=0,A[3]=0,A[4]=0,A[5]=1,A[6]=0,A[7]=0,A[8]=2
    // lower col-major: col0=[2,1,0], col1=[_,0,1], col2=[_,_,2]
    double A[9]={2,1,0, 0,0,1, 0,0,2};
    std::vector<Pivot> piv; InertiaCounts in;
    dense_ldlt_indef(A,3,3,0.01,1e-20,piv,in);
    // Should produce pivots without crashing; total inertia sums to 3
    EXPECT_EQ(in.positive+in.negative+in.zero, 3);
    // Matrix det = 2*(0*2-1)-1*(1*2-0) = -2-2 = -4 < 0, so 1 neg 2 pos or vice versa
    // Actual: eigenvalues of [[2,1,0],[1,0,1],[0,1,2]]: sum=4>0, so at least 1 pos
    EXPECT_GE(in.positive,1);
}
TEST(LDLT, NearSingular) {
    double A[4]={1e-25,0,0,1.0};
    std::vector<Pivot> piv; InertiaCounts in;
    // Should not crash; singular status expected since |A[0,0]| < small=1e-20
    EXPECT_NO_THROW(dense_ldlt_indef(A,2,2,0.1,1e-20,piv,in));
}
TEST(LDLT, AllNegative2x2) {
    double A[4]={-4,-2, -2,-3};
    std::vector<Pivot> piv; InertiaCounts in;
    dense_ldlt_indef(A,2,2,0.01,1e-20,piv,in);
    EXPECT_EQ(in.negative,2); EXPECT_EQ(in.positive,0);
}
// Random indefinite front of size f, lower triangle, column-major (ld f).
// kind 0: dense; 1: sparse with zero diagonals (2x2 pivots, delays);
// 2: permuted KKT [[H, J^T], [J, 0]]; 3: entries over 12 orders of magnitude;
// 4: diagonally dominant SPD.
static std::vector<double> small_random_front(int f, int kind, std::mt19937 &rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::uniform_real_distribution<double> P(0.0, 1.0);
    std::vector<double> A(static_cast<std::size_t>(f) * f, 0.0);
    auto at = [&](int i, int j) -> double & { return A[static_cast<std::size_t>(j) * f + i]; };
    if (kind == 2) {
        const int np = std::max(1, (2 * f) / 3);
        std::vector<int> perm(f);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int j = 0; j < f; ++j)
            for (int i = j; i < f; ++i) {
                const int a = std::max(perm[i], perm[j]), b = std::min(perm[i], perm[j]);
                double v = 0.0;
                if (a < np) v = (a == b) ? 1e-3 + P(rng) : (P(rng) < 0.3 ? 0.1 * U(rng) : 0.0);
                else if (b < np) v = P(rng) < 0.5 ? U(rng) : 0.0;
                at(i, j) = v;
            }
        return A;
    }
    for (int j = 0; j < f; ++j)
        for (int i = j; i < f; ++i) {
            double v = U(rng);
            if (kind == 1) {
                if (i == j ? P(rng) < 0.4 : P(rng) < 0.6) v = 0.0;
            } else if (kind == 3) {
                v *= std::pow(10.0, 12.0 * P(rng) - 6.0);
            } else if (kind == 4) {
                v = (i == j) ? 2.0 * f + P(rng) : U(rng);
            }
            at(i, j) = v;
        }
    return A;
}

TEST(LdltDense, SmallFrontKernelsMatchGeneric) {
    std::mt19937 rng(20261027);
    constexpr int kFronts = 1000;
    int fronts_2x2 = 0, fronts_static = 0, fronts_delay = 0;
    long long entries = 0, not_bitwise = 0;
    alignas(32) double buf[kSmallFrontMax * kSmallFrontMax];
    for (int t = 0; t < kFronts; ++t) {
        const int f = 1 + static_cast<int>(rng() % kSmallFrontMax);
        const int num_elig = 1 + static_cast<int>(rng() % f);
        const int kind = t % 5;
        const std::vector<double> A = small_random_front(f, kind, rng);

        LdltKernelParams prm;
        const double us[] = {0.01, 1e-8, 0.1, 0.5};
        prm.u = us[(t / 5) % 4];
        if (kind == 4) {
            prm.posdef = (t % 2) == 0;
        } else if (t % 7 == 3) {
            prm.static_pivot = 1e-6;
            prm.static_zero_stays_zero = (t % 3) == 0;
            if (t % 11 == 5) prm.min_abs_pivot = prm.static_pivot;
        }
        SCOPED_TRACE("front " + std::to_string(t) + " f=" + std::to_string(f) +
                     " elig=" + std::to_string(num_elig) + " kind=" + std::to_string(kind));

        // Generic unblocked kernel, in place with leading dimension f.
        std::vector<double> G = A;
        std::vector<Int> grows(f);
        std::iota(grows.begin(), grows.end(), Int{0});
        std::vector<int8_t> gtags(f, 0);
        std::vector<double> work(ldlt_work_size(f, prm));
        const LdltKernelResult gr =
            ldlt_factor_front(G.data(), f, num_elig, grows.data(), prm, gtags.data(), work.data());

        // Small-front kernel on the padded layout; the upper triangle holds
        // junk it must not read.
        const int ld = small_front_ld(f);
        const int fe = (f + 3) & ~3;
        std::fill(std::begin(buf), std::end(buf), 0.0);
        for (int j = 0; j < f; ++j) {
            for (int i = 0; i < j; ++i) buf[j * ld + i] = 3.25 + i - j;
            for (int i = j; i < f; ++i) buf[j * ld + i] = A[static_cast<std::size_t>(j) * f + i];
            for (int i = f; i < fe; ++i) buf[j * ld + i] = 0.0;
        }
        std::vector<Int> srows(f);
        std::iota(srows.begin(), srows.end(), Int{0});
        std::vector<int8_t> stags(f, 0);
        const LdltKernelResult sr =
            ldlt_factor_small_front(buf, ld, f, num_elig, srows.data(), prm, stags.data());

        ASSERT_EQ(sr.num_accepted, gr.num_accepted);
        ASSERT_EQ(sr.static_pivots, gr.static_pivots);
        ASSERT_EQ(sr.inertia.positive, gr.inertia.positive);
        ASSERT_EQ(sr.inertia.negative, gr.inertia.negative);
        ASSERT_EQ(sr.inertia.zero, gr.inertia.zero);
        ASSERT_EQ(sr.flops, gr.flops);
        const int p = gr.num_accepted;
        for (int c = 0; c < p; ++c) ASSERT_EQ(stags[c], gtags[c]) << "tag " << c;
        for (int i = 0; i < f; ++i) ASSERT_EQ(srows[i], grows[i]) << "row " << i;
        if (std::any_of(gtags.begin(), gtags.begin() + p, [](int8_t g) { return g == 2; })) ++fronts_2x2;
        if (gr.static_pivots > 0) ++fronts_static;
        if (p < num_elig) ++fronts_delay;

        // L and D (columns [0, p)) and the Schur complement: lower triangle.
        for (int j = 0; j < f; ++j)
            for (int i = j; i < f; ++i) {
                const double g = G[static_cast<std::size_t>(j) * f + i];
                const double s = buf[j * ld + i];
                ++entries;
                if (std::memcmp(&g, &s, sizeof(double)) != 0) ++not_bitwise;
                if (std::isfinite(g)) {
                    ASSERT_LE(std::abs(s - g), 1e-14 * std::max(1.0, std::abs(g)))
                        << "entry (" << i << ", " << j << ")";
                } else {
                    ASSERT_EQ(std::isnan(s), std::isnan(g));
                }
            }
        // Padding rows stay zero.
        for (int j = 0; j < f; ++j)
            for (int i = f; i < fe; ++i) ASSERT_EQ(buf[j * ld + i], 0.0);
    }
    std::printf("[ small fronts ] %d fronts: %d with 2x2 pivots, %d static, %d delaying; "
                "%lld of %lld entries not bitwise equal\n",
                kFronts, fronts_2x2, fronts_static, fronts_delay, not_bitwise, entries);
    // Coverage of the pivot paths the kernels share with the generic one.
    EXPECT_GT(fronts_2x2, 50);
    EXPECT_GT(fronts_static, 10);
    EXPECT_GT(fronts_delay, 50);
}

// Factors the 4x4 front A (column-major, lower triangle) with both the
// generic unblocked kernel and the small-front kernel, all columns eligible,
// checks that they agree, and returns the pivot tags.
static std::vector<int8_t> weak_front_tags(const std::vector<double> &A, double u) {
    constexpr int f = 4;
    LdltKernelParams prm;
    prm.u = u;
    std::vector<double> G = A;
    std::vector<Int> grows(f);
    std::iota(grows.begin(), grows.end(), Int{0});
    std::vector<int8_t> gtags(f, 0);
    std::vector<double> work(ldlt_work_size(f, prm));
    const LdltKernelResult gr = ldlt_factor_front(G.data(), f, f, grows.data(), prm, gtags.data(), work.data());
    EXPECT_EQ(gr.num_accepted, f);

    alignas(32) double buf[kSmallFrontMax * kSmallFrontMax] = {};
    const int ld = small_front_ld(f);
    for (int j = 0; j < f; ++j)
        for (int i = j; i < f; ++i) buf[j * ld + i] = A[static_cast<std::size_t>(j) * f + i];
    std::vector<Int> srows(f);
    std::iota(srows.begin(), srows.end(), Int{0});
    std::vector<int8_t> stags(f, 0);
    const LdltKernelResult sr = ldlt_factor_small_front(buf, ld, f, f, srows.data(), prm, stags.data());
    EXPECT_EQ(sr.num_accepted, gr.num_accepted);
    EXPECT_EQ(sr.inertia.negative, gr.inertia.negative);
    EXPECT_EQ(stags, gtags) << "small-front and generic kernels disagree";
    EXPECT_EQ(srows, grows);
    return gtags;
}

// A front whose column 0 has a tiny diagonal d00 against a unit coupling to
// row 1 (its 2x2 partner), small entries elsewhere, and a(2, 1) = a21 in the
// partner's column.
static std::vector<double> weak_front(double d00, double a21) {
    constexpr int f = 4;
    std::vector<double> A(f * f, 0.0);
    const auto at = [&](int i, int j) -> double & { return A[static_cast<std::size_t>(j) * f + i]; };
    at(0, 0) = d00;
    at(1, 0) = 1.0;
    at(2, 0) = 0.1;
    at(3, 0) = 0.05;
    at(1, 1) = 1e-3;
    at(2, 1) = a21;
    at(3, 1) = 0.1;
    at(2, 2) = 4.0;
    at(3, 2) = 0.3;
    at(3, 3) = -5.0;
    return A;
}

// select_pivot's weak 1x1 rule: at IPOPT's u = 1e-8, a 1x1 that passes the
// threshold test with |d| below 1e-6 of its column maximum gives way to the
// 2x2 with its partner when the 2x2's growth bound is smaller.
TEST(LdltDense, WeakOneByOneGivesWayToStableTwoByTwo) {
    // (a) |a00| = 5e-8 of its column (1x1 growth 2e7) and a well-conditioned
    // 2x2 (0, 1) with growth bound 0.2: the 2x2 is taken.
    {
        SCOPED_TRACE("(a) weak 1x1, stable 2x2");
        const std::vector<int8_t> t = weak_front_tags(weak_front(5e-8, 0.2), 1e-8);
        EXPECT_EQ(t[0], 2);
        EXPECT_EQ(t[1], -1);
    }
    // (b) The same weak 1x1, but a(2, 1) = 3e7 gives the 2x2 a growth bound
    // of 3e7 (still within 1/u), larger than the 1x1's 2e7: the 1x1 stays.
    {
        SCOPED_TRACE("(b) weak 1x1, 2x2 with larger growth");
        const std::vector<int8_t> t = weak_front_tags(weak_front(5e-8, 3e7), 1e-8);
        EXPECT_EQ(t[0], 1);
    }
    // (c) u = 0.01 (and u = 1e-8 with a 1x1 that is not weak, ratio 0.05):
    // the rule never fires and the 1x1 on column 0 is taken as before.
    for (double u : {0.01, 1e-8}) {
        SCOPED_TRACE("(c) not weak, u = " + std::to_string(u));
        const std::vector<int8_t> t = weak_front_tags(weak_front(0.05, 0.2), u);
        EXPECT_EQ(t[0], 1);
    }
    // At u = 0.01 the weak diagonal of (a) fails the 1x1 test outright: the
    // 2x2, as before the rule.
    {
        SCOPED_TRACE("(c) u = 0.01, failing 1x1");
        const std::vector<int8_t> t = weak_front_tags(weak_front(5e-8, 0.2), 0.01);
        EXPECT_EQ(t[0], 2);
        EXPECT_EQ(t[1], -1);
    }
}

// The blocked kernel's first pass on a KKT-like front whose leading eligible
// columns have no pivot of their own, like the delayed columns a front
// receives: m constraint columns c (zero diagonal) coupled only to variable
// columns at least kFar past them (outside their panel), then nv variable
// columns (an SPD block), then q non-eligible rows. A panel over constraint
// columns stalls and pulls each one's largest partner next to it, and the
// next panel takes those 2x2 pivots. Every eligible column is accepted, the
// inertia is the block's, and the Schur complement of the rows below matches
// a dense reference.
TEST(LdltDense, StalledPanelsPullInPartners) {
    constexpr int m = 40, nv = 120, q = 40, e = m + nv, f = e + q, kFar = 100;
    std::mt19937 rng(20261102);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> A(static_cast<std::size_t>(f) * f, 0.0);
    const auto at = [&](int i, int j) -> double & { return A[static_cast<std::size_t>(j) * f + i]; };
    for (int c = 0; c < m; ++c)
        for (int t = 0; t < 3; ++t) at(kFar + (c * 3 + t) % (e - kFar) , c) += 1.0 + 0.5 * U(rng);
    for (int j = m; j < e; ++j) {
        at(j, j) = 4.0 + U(rng);
        for (int i = j + 1; i < f; ++i)
            if (rng() % 8 == 0) at(i, j) = 0.3 * U(rng);
    }
    for (int j = e; j < f; ++j)
        for (int i = j; i < f; ++i) at(i, j) = (i == j ? 2.0 : 0.1 * U(rng));

    LdltKernelParams prm;
    prm.u = 0.01;
    ASSERT_GE(f, prm.blocked_min_front);
    std::vector<double> F = A;
    std::vector<Int> rows(f);
    std::iota(rows.begin(), rows.end(), Int{0});
    std::vector<int8_t> tags(f, 0);
    std::vector<double> work(ldlt_work_size(f, prm));
    const LdltKernelResult r = ldlt_factor_front(F.data(), f, e, rows.data(), prm, tags.data(), work.data());
    EXPECT_EQ(r.num_accepted, e);
    EXPECT_EQ(r.inertia.positive, nv);
    EXPECT_EQ(r.inertia.negative, m);
    EXPECT_EQ(r.inertia.zero, 0);
    // The first pivot pairs constraint column 0 with a variable past kFar.
    EXPECT_EQ(tags[0], 2);
    EXPECT_EQ(rows[0], 0);
    EXPECT_GE(rows[1], kFar);
    // Non-eligible rows keep their positions.
    for (int i = e; i < f; ++i) EXPECT_EQ(rows[i], i);

    // Reference Schur complement S = A22 - A21 A11^-1 A12 (Gaussian
    // elimination with partial pivoting on the full symmetric A11).
    std::vector<double> M(static_cast<std::size_t>(e) * (e + q), 0.0);  // [A11 | A12], row-major
    const auto sym = [&](int i, int j) { return i >= j ? at(i, j) : at(j, i); };
    for (int i = 0; i < e; ++i)
        for (int j = 0; j < e + q; ++j) M[static_cast<std::size_t>(i) * (e + q) + j] = sym(i, j);
    for (int k = 0; k < e; ++k) {
        int p = k;
        for (int i = k + 1; i < e; ++i)
            if (std::abs(M[static_cast<std::size_t>(i) * (e + q) + k]) > std::abs(M[static_cast<std::size_t>(p) * (e + q) + k])) p = i;
        if (p != k)
            for (int j = 0; j < e + q; ++j)
                std::swap(M[static_cast<std::size_t>(k) * (e + q) + j], M[static_cast<std::size_t>(p) * (e + q) + j]);
        const double d = M[static_cast<std::size_t>(k) * (e + q) + k];
        ASSERT_NE(d, 0.0);
        for (int i = k + 1; i < e; ++i) {
            const double l = M[static_cast<std::size_t>(i) * (e + q) + k] / d;
            if (l == 0.0) continue;
            for (int j = k; j < e + q; ++j)
                M[static_cast<std::size_t>(i) * (e + q) + j] -= l * M[static_cast<std::size_t>(k) * (e + q) + j];
        }
    }
    // X = A11^-1 A12 by back substitution, then S.
    std::vector<double> X(static_cast<std::size_t>(e) * q, 0.0);
    for (int c = 0; c < q; ++c)
        for (int i = e - 1; i >= 0; --i) {
            double s = M[static_cast<std::size_t>(i) * (e + q) + e + c];
            for (int j = i + 1; j < e; ++j) s -= M[static_cast<std::size_t>(i) * (e + q) + j] * X[static_cast<std::size_t>(j) * q + c];
            X[static_cast<std::size_t>(i) * q + c] = s / M[static_cast<std::size_t>(i) * (e + q) + i];
        }
    double worst = 0.0;
    for (int j = e; j < f; ++j)
        for (int i = j; i < f; ++i) {
            double s = at(i, j);
            for (int k = 0; k < e; ++k) s -= sym(i, k) * X[static_cast<std::size_t>(k) * q + (j - e)];
            worst = std::max(worst, std::abs(F[static_cast<std::size_t>(j) * f + i] - s));
        }
    EXPECT_LT(worst, 1e-10);
}

int main(int argc,char**argv){testing::InitGoogleTest(&argc,argv);return RUN_ALL_TESTS();}
