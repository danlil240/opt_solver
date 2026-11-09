// TPP kernel with panel lookahead (solver/src/ldlt_lookahead.hpp, 8T engine
// Part 6): panel k+1 is searched and factored while panel k's update of the
// later columns and its Schur chunk run on other threads. The pivots (rows,
// tags), the accepted count and the inertia are today's kernel's; L and the
// Schur complement differ by rounding only.

#include "ldlt_lookahead.hpp"

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

namespace {

using Mat = std::vector<double>;  // f x f, column-major, lower triangle used

// Random symmetric indefinite: diagonal of random sign and size, off-diagonals
// in [-1, 1] at density `dens`.
Mat random_indefinite(int f, unsigned seed, double dens = 0.3) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  Mat A(static_cast<std::size_t>(f) * f, 0.0);
  for (int j = 0; j < f; ++j) {
    A[static_cast<std::size_t>(j) * f + j] = 4.0 * U(rng);
    for (int i = j + 1; i < f; ++i)
      if (U(rng) * 0.5 + 0.5 < dens) A[static_cast<std::size_t>(j) * f + i] = U(rng);
  }
  return A;
}

// Saddle point: [H B^T; B 0] with H (f - z) x (f - z) SPD-ish and a zero
// z x z block, so the zero block's columns pivot in 2x2s.
Mat saddle(int f, int z, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const int h = f - z;
  Mat A(static_cast<std::size_t>(f) * f, 0.0);
  for (int j = 0; j < h; ++j) {
    A[static_cast<std::size_t>(j) * f + j] = 6.0 + U(rng);
    for (int i = j + 1; i < h; ++i)
      if (rng() % 6 == 0) A[static_cast<std::size_t>(j) * f + i] = 0.5 * U(rng);
    for (int i = h; i < f; ++i)
      if (rng() % 5 == 0) A[static_cast<std::size_t>(j) * f + i] = U(rng);
  }
  // Interleave: move the zero block's rows/columns among the H ones so that
  // panels hold both (a symmetric permutation keeps A symmetric).
  std::vector<int> perm(static_cast<std::size_t>(f));
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  Mat P(A.size(), 0.0);
  const auto sym = [&](int i, int j) {
    return i >= j ? A[static_cast<std::size_t>(j) * f + i] : A[static_cast<std::size_t>(i) * f + j];
  };
  for (int j = 0; j < f; ++j)
    for (int i = j; i < f; ++i) P[static_cast<std::size_t>(j) * f + i] = sym(perm[i], perm[j]);
  return P;
}

// LdltDense.StalledPanelsPullInPartners' pattern, larger: m constraint
// columns at the front whose largest entries lie far behind (> kFar), then
// variables, then q non-eligible rows.
Mat stalled_pattern(int m, int nv, int q, int kFar, int &e) {
  e = m + nv;
  const int f = e + q;
  std::mt19937 rng(20261102);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  Mat A(static_cast<std::size_t>(f) * f, 0.0);
  const auto at = [&](int i, int j) -> double & { return A[static_cast<std::size_t>(j) * f + i]; };
  for (int c = 0; c < m; ++c)
    for (int t = 0; t < 3; ++t) at(kFar + (c * 3 + t) % (e - kFar), c) += 1.0 + 0.5 * U(rng);
  for (int j = m; j < e; ++j) {
    at(j, j) = 4.0 + U(rng);
    for (int i = j + 1; i < f; ++i)
      if (rng() % 8 == 0) at(i, j) = 0.3 * U(rng);
  }
  for (int j = e; j < f; ++j)
    for (int i = j; i < f; ++i) at(i, j) = (i == j ? 2.0 : 0.1 * U(rng));
  return A;
}

struct Out {
  Mat F;
  std::vector<Int> rows;
  std::vector<int8_t> tags;
  LdltKernelResult r;
};

Out run_tpp(const Mat &A, int f, int ne, const LdltKernelParams &prm) {
  Out o{A, std::vector<Int>(static_cast<std::size_t>(f)), std::vector<int8_t>(static_cast<std::size_t>(f), 0), {}};
  std::iota(o.rows.begin(), o.rows.end(), Int{0});
  std::vector<double> work(ldlt_work_size(f, prm));
  o.r = ldlt_factor_front(o.F.data(), f, ne, o.rows.data(), prm, o.tags.data(), work.data());
  return o;
}

Out run_la(const Mat &A, int f, int ne, const LdltKernelParams &prm, int threads) {
  Out o{A, std::vector<Int>(static_cast<std::size_t>(f)), std::vector<int8_t>(static_cast<std::size_t>(f), 0), {}};
  std::iota(o.rows.begin(), o.rows.end(), Int{0});
  std::vector<double> work(ldlt_lookahead_work_size(f, prm));
  o.r = ldlt_factor_front_lookahead(o.F.data(), f, ne, o.rows.data(), prm, o.tags.data(), work.data(), threads,
                                    nullptr);
  return o;
}

// Same pivots, count and inertia exactly; the lower triangle within 1e-9 of
// the reference's largest entry. The Schur update is summed in per-panel
// chunks, so on matrices with heavy cancellation (random indefinite at
// u = 0.01) entries differ by rounding up to ~3e-11 of the largest; a missing
// or doubled update would differ by O(1). Accuracy itself is checked by the
// backward error (EdgeShapes, ne == f).
void expect_same(const Out &ref, const Out &got, int f, const std::string &what) {
  EXPECT_EQ(got.r.num_accepted, ref.r.num_accepted) << what;
  EXPECT_EQ(got.r.inertia.positive, ref.r.inertia.positive) << what;
  EXPECT_EQ(got.r.inertia.negative, ref.r.inertia.negative) << what;
  EXPECT_EQ(got.r.inertia.zero, ref.r.inertia.zero) << what;
  EXPECT_EQ(got.rows, ref.rows) << what;
  EXPECT_EQ(got.tags, ref.tags) << what;
  double mx = 0.0, diff = 0.0;
  for (int j = 0; j < f; ++j)
    for (int i = j; i < f; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * f + i;
      mx = std::max(mx, std::abs(ref.F[k]));
      diff = std::max(diff, std::abs(ref.F[k] - got.F[k]));
    }
  EXPECT_LE(diff, 1e-9 * mx) << what << ": max diff " << diff << " of " << mx;
}

// A full factorization (num_elig == f): max |A(rows, rows) - L D L^T| over
// max |A|, with L unit lower, D's 1x1 / 2x2 blocks from tags.
double backward_error(const Mat &A, const Out &o, int f) {
  const auto a = [&](int i, int j) {
    return i >= j ? A[static_cast<std::size_t>(j) * f + i] : A[static_cast<std::size_t>(i) * f + j];
  };
  const auto F = [&](int i, int j) { return o.F[static_cast<std::size_t>(j) * f + i]; };
  // L (unit lower, below the D blocks) and D, dense.
  Mat L(static_cast<std::size_t>(f) * f, 0.0), D(static_cast<std::size_t>(f) * f, 0.0);
  for (int c = 0; c < f; ++c) {
    L[static_cast<std::size_t>(c) * f + c] = 1.0;
    const bool second = o.tags[static_cast<std::size_t>(c)] == -1;
    const bool first2 = o.tags[static_cast<std::size_t>(c)] == 2;
    D[static_cast<std::size_t>(c) * f + c] = F(c, c);
    if (first2) {
      D[static_cast<std::size_t>(c) * f + c + 1] = F(c + 1, c);
      D[static_cast<std::size_t>(c + 1) * f + c] = F(c + 1, c);
    }
    for (int i = c + (first2 ? 2 : 1); i < f; ++i) L[static_cast<std::size_t>(c) * f + i] = F(i, c);
    (void)second;
  }
  double err = 0.0, amax = 0.0;
  std::vector<double> LD(static_cast<std::size_t>(f) * f, 0.0);
  for (int j = 0; j < f; ++j)
    for (int k = 0; k < f; ++k) {
      const double d = D[static_cast<std::size_t>(j) * f + k];
      if (d == 0.0) continue;
      for (int i = 0; i < f; ++i) LD[static_cast<std::size_t>(j) * f + i] += L[static_cast<std::size_t>(k) * f + i] * d;
    }
  for (int j = 0; j < f; ++j)
    for (int i = j; i < f; ++i) {
      double m = 0.0;
      for (int k = 0; k <= j; ++k) m += LD[static_cast<std::size_t>(k) * f + i] * L[static_cast<std::size_t>(k) * f + j];
      const double ai = a(static_cast<int>(o.rows[static_cast<std::size_t>(i)]), static_cast<int>(o.rows[static_cast<std::size_t>(j)]));
      err = std::max(err, std::abs(m - ai));
      amax = std::max(amax, std::abs(ai));
    }
  return err / amax;
}

LdltKernelParams params(double u) {
  LdltKernelParams p;
  p.u = u;
  p.nb = 32;
  return p;
}

}  // namespace

TEST(LdltLookahead, SamePivotsAsTpp) {
  int e_st = 0;
  const Mat st = stalled_pattern(48, 200, 72, 150, e_st);
  struct Case {
    const char *name;
    Mat A;
    int f, ne;
  } cases[] = {{"random", random_indefinite(400, 7), 400, 300},
               {"saddle", saddle(384, 128, 11), 384, 300},
               {"stalled", st, e_st + 72, e_st}};
  for (const double u : {0.01, 1e-8}) {
    for (const Case &c : cases) {
      const Out ref = run_tpp(c.A, c.f, c.ne, params(u));
      for (const int t : {2, 4, 8}) {
        const Out got = run_la(c.A, c.f, c.ne, params(u), t);
        expect_same(ref, got, c.f, std::string(c.name) + " u " + std::to_string(u) + " t " + std::to_string(t));
      }
    }
  }
}

TEST(LdltLookahead, StallTakesTodaysPath) {
  // Columns [64, 96) have a zero diagonal and tiny couplings except to their
  // partners in [160, 192): panel 2 stalls and pulls the partners in.
  const int f = 256, ne = 192;
  Mat A = random_indefinite(f, 3, 0.05);
  for (int j = 0; j < f; ++j) A[static_cast<std::size_t>(j) * f + j] += (j % 2 ? 6.0 : -6.0);
  for (int c = 64; c < 96; ++c) {
    for (int i = c; i < f; ++i) A[static_cast<std::size_t>(c) * f + i] *= 1e-3;
    for (int j = 0; j < c; ++j) A[static_cast<std::size_t>(j) * f + c] *= 1e-3;
    A[static_cast<std::size_t>(c) * f + c] = 0.0;
    A[static_cast<std::size_t>(c) * f + (96 + c)] = 1.0;  // partner c + 96 in [160, 192)
  }
  const Out ref = run_tpp(A, f, ne, params(0.01));
  bool far_partner = false;
  for (int pos = 64; pos < 96; ++pos) far_partner = far_partner || ref.rows[static_cast<std::size_t>(pos)] >= 160;
  ASSERT_TRUE(far_partner) << "premise: panel 2 pulled in a partner from [160, 192)";
  expect_same(ref, run_la(A, f, ne, params(0.01), 4), f, "stall");
}

TEST(LdltLookahead, EdgeShapes) {
  struct Shape {
    const char *name;
    Mat A;
    int f, ne;
  };
  std::vector<Shape> shapes;
  shapes.push_back({"ne 301", random_indefinite(400, 21), 400, 301});
  shapes.push_back({"ne == f", random_indefinite(200, 22), 200, 200});
  shapes.push_back({"two short panels", random_indefinite(200, 23), 200, 40});
  {
    Mat A = random_indefinite(200, 24);
    for (int j = 0; j < 32; ++j)
      for (int i = j; i < 32; ++i) A[static_cast<std::size_t>(j) * 200 + i] = 0.0;
    shapes.push_back({"first panel stalls", A, 200, 150});
  }
  {
    Mat A = random_indefinite(200, 25);
    for (int j = 0; j < 100; ++j)
      for (int i = j; i < 200; ++i) A[static_cast<std::size_t>(j) * 200 + i] = 0.0;
    shapes.push_back({"zero eligible block", A, 200, 100});
  }
  for (const Shape &s : shapes) {
    const Out ref = run_tpp(s.A, s.f, s.ne, params(0.01));
    expect_same(ref, run_la(s.A, s.f, s.ne, params(0.01), 4), s.f, s.name);
  }
  // ne == f: the lookahead's factorization is as accurate as today's (its
  // differences from today's are rounding of the chunked Schur sums).
  {
    const Out ref = run_tpp(shapes[1].A, 200, 200, params(0.01));
    const Out got = run_la(shapes[1].A, 200, 200, params(0.01), 4);
    const double er = backward_error(shapes[1].A, ref, 200), eg = backward_error(shapes[1].A, got, 200);
    std::printf("[ne == f] backward error: tpp %.3e lookahead %.3e\n", er, eg);
    EXPECT_LE(eg, 4.0 * er + 1e-15);
  }
  // The zero eligible block delays every column.
  EXPECT_EQ(run_la(shapes.back().A, 200, 100, params(0.01), 4).r.num_accepted, 0);
}

TEST(LdltLookahead, RepeatedRunsAgree) {
  // nthreads 2: the update and Schur tasks run on one BLAS thread, so the run
  // is deterministic; any race between panel k+1 and panel k's update would
  // show as a difference.
  const Mat A = saddle(384, 128, 11);
  const Out first = run_la(A, 384, 300, params(1e-8), 2);
  for (int it = 0; it < 20; ++it) {
    const Out again = run_la(A, 384, 300, params(1e-8), 2);
    ASSERT_EQ(std::memcmp(again.F.data(), first.F.data(), first.F.size() * sizeof(double)), 0) << it;
    ASSERT_EQ(again.rows, first.rows) << it;
    ASSERT_EQ(again.tags, first.tags) << it;
  }
}
