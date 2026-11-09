// Column-block assembly of a front (solver/src/front_assembly.hpp, 8T engine
// Part 5): any split of the columns, on any number of threads, gives the
// serial assembly's values bit for bit.

#include "front_assembly.hpp"
#include "parallel/tile_tasks.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace smf;

namespace {

constexpr int kF = 700;

// A synthetic front: original entries in pivot columns [0, 40) (identity
// pos), and three children with increasing maps; the third is strided.
struct Front {
  std::vector<Int> acp, ari, pos;
  std::vector<double> av;
  std::vector<std::vector<Int>> maps;
  std::vector<std::vector<double>> vals;
  std::vector<AssemblyChild> kids;
  AssemblyOriginal orig;
};

Front make_front() {
  Front fr;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  fr.pos.resize(kF);
  for (int i = 0; i < kF; ++i) fr.pos[i] = i;
  fr.acp.push_back(0);
  for (int j = 0; j < 40; ++j) {
    for (int i = j; i < kF; i += 1 + static_cast<int>(rng() % 9)) {
      fr.ari.push_back(i);
      fr.av.push_back(u(rng));
    }
    fr.acp.push_back(static_cast<Int>(fr.ari.size()));
  }
  fr.orig.acp = fr.acp.data();
  fr.orig.ari = fr.ari.data();
  fr.orig.pos = fr.pos.data();
  fr.orig.av = fr.av.data();
  fr.orig.col_start = 0;
  fr.orig.col_end = 40;
  const int qs[3] = {300, 450, 120};
  for (int c = 0; c < 3; ++c) {
    std::vector<Int> all(kF);
    for (int i = 0; i < kF; ++i) all[i] = i;
    std::shuffle(all.begin(), all.end(), rng);
    std::vector<Int> m(all.begin(), all.begin() + qs[c]);
    std::sort(m.begin(), m.end());
    fr.maps.push_back(m);
    const std::size_t q = static_cast<std::size_t>(qs[c]);
    const std::size_t ld = c == 2 ? 128 : 0;
    std::vector<double> v(ld > 0 ? ld * ld : q * (q + 1) / 2);
    for (double &x : v) x = u(rng);
    fr.vals.push_back(std::move(v));
  }
  for (int c = 0; c < 3; ++c) {
    AssemblyChild k;
    k.map = fr.maps[c].data();
    k.values = fr.vals[c].data();
    k.ld = c == 2 ? 128 : 0;
    k.q = static_cast<Int>(fr.maps[c].size());
    fr.kids.push_back(k);
  }
  return fr;
}

// Every entry starts at the same value (the upper triangle is never written).
std::vector<double> blank() { return std::vector<double>(static_cast<std::size_t>(kF) * kF, 7.0); }

// The serial assembly as process_front writes it: zero the lower triangle,
// the original entries, then each child in jj-outer order.
std::vector<double> reference(const Front &fr) {
  std::vector<double> F = blank();
  const std::size_t f = kF;
  for (std::size_t j = 0; j < f; ++j) std::fill_n(F.data() + j * f + j, f - j, 0.0);
  for (Int j = fr.orig.col_start; j < fr.orig.col_end; ++j)
    for (Int k = fr.acp[j]; k < fr.acp[j + 1]; ++k)
      F[static_cast<std::size_t>(fr.pos[j]) * f + fr.pos[fr.ari[k]]] += fr.av[k];
  for (const AssemblyChild &c : fr.kids) {
    const double *cv = c.values;
    for (Int jj = 0; jj < c.q; ++jj) {
      if (c.ld > 0) cv = c.values + static_cast<std::size_t>(jj) * (c.ld + 1);
      double *Fc = F.data() + static_cast<std::size_t>(c.map[jj]) * f;
      for (Int ii = jj; ii < c.q; ++ii) Fc[c.map[ii]] += *cv++;
    }
  }
  return F;
}

bool same(const std::vector<double> &a, const std::vector<double> &b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

}  // namespace

TEST(FrontAssembly, ColumnBlocksEqualOneBlock) {
  const Front fr = make_front();
  std::vector<double> one = blank();
  assemble_front_columns(one.data(), kF, fr.orig, fr.kids, 0, kF);
  std::vector<double> blocks = blank();
  for (const auto [c0, c1] : {std::pair{0, 1}, std::pair{1, 333}, std::pair{333, kF}})
    assemble_front_columns(blocks.data(), kF, fr.orig, fr.kids, c0, c1);
  EXPECT_TRUE(same(one, blocks));
  EXPECT_TRUE(same(one, reference(fr)));
}

TEST(FrontAssembly, ParallelEqualsSerialAtEveryThreadCount) {
  const Front fr = make_front();
  std::vector<double> serial = blank();
  assemble_front_columns(serial.data(), kF, fr.orig, fr.kids, 0, kF);
  ASSERT_TRUE(same(serial, reference(fr)));
  for (const int t : {1, 2, 4, 8}) {
    std::vector<double> par = blank();
    assemble_front_parallel(par.data(), kF, fr.orig, fr.kids, t, nullptr);
    EXPECT_TRUE(same(serial, par)) << t << " threads";
  }
}

namespace {
// Records what assemble_front_parallel lends to the top-part scheduler's
// workers; lends nothing back (the runner then runs every task itself).
class RecordingHelpers final : public smf::TileHelpers {
public:
  void offer(smf::TileRunner &, int slots) override { offered += slots; }
  void withdraw(smf::TileRunner &) override {}
  int offered = 0;
};
}  // namespace

// With OpenMP the blocks run on the calling thread's OpenMP team, the one
// its MKL calls use, never on the scheduler's workers: workers next to
// MKL's spinning team made the front's following BLAS calls slower by
// about what the assembly saved (2026-11-09 acopf profile).
TEST(FrontAssembly, ParallelRunsOnTheCallersOpenMPTeam) {
#ifndef _OPENMP
  GTEST_SKIP() << "built without OpenMP";
#else
  const Front fr = make_front();
  std::vector<double> serial = blank();
  assemble_front_columns(serial.data(), kF, fr.orig, fr.kids, 0, kF);
  RecordingHelpers helpers;
  std::vector<double> par = blank();
  assemble_front_parallel(par.data(), kF, fr.orig, fr.kids, 4, &helpers);
  EXPECT_EQ(helpers.offered, 0);
  EXPECT_TRUE(same(serial, par));
#endif
}

// The contribution block (rows and columns [p, f) of the front, lower part)
// copied out packed, column by column, on any number of threads.
TEST(FrontAssembly, ContributionBlockCopyEqualsSerial) {
  const int f = 640, p = 140, q = f - p;
  std::vector<double> F(static_cast<std::size_t>(f) * f);
  for (std::size_t i = 0; i < F.size(); ++i) F[i] = std::sin(0.37 * static_cast<double>(i));
  const std::size_t packed = static_cast<std::size_t>(q) * (q + 1) / 2;
  std::vector<double> ref(packed);
  double *d = ref.data();
  for (int jj = 0; jj < q; ++jj) {
    const double *src = F.data() + static_cast<std::size_t>(p + jj) * f + (p + jj);
    d = std::copy(src, src + (q - jj), d);
  }
  for (const int t : {1, 2, 4, 8}) {
    std::vector<double> out(packed + 1, -1.0);
    copy_contribution_block(out.data(), F.data() + static_cast<std::size_t>(p) * f + p, f, q, t);
    EXPECT_TRUE(std::memcmp(out.data(), ref.data(), packed * sizeof(double)) == 0) << t << " threads";
    EXPECT_EQ(out[packed], -1.0) << "wrote past the block";
  }
}

// AssemblyChild needs a strictly increasing map; the caller checks each
// child's with child_maps_increasing and keeps today's loops otherwise.
TEST(FrontAssembly, NonMonotoneChildKeepsSerialOrder) {
  const std::vector<Int> up = {1, 2, 5, 9}, down = {1, 3, 2, 9}, flat = {4, 4}, none = {};
  EXPECT_TRUE(child_maps_increasing(up));
  EXPECT_FALSE(child_maps_increasing(down));
  EXPECT_FALSE(child_maps_increasing(flat));
  EXPECT_TRUE(child_maps_increasing(none));
}
