// Column-block assembly of a front (see front_assembly.hpp).

#include "front_assembly.hpp"

#include "parallel/tile_tasks.hpp"

#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace smf {

bool child_maps_increasing(std::span<const Int> map) noexcept {
  for (std::size_t t = 1; t < map.size(); ++t)
    if (!(map[t] > map[t - 1])) return false;
  return true;
}

void assemble_front_columns(double *F, int f, const AssemblyOriginal &orig,
                            std::span<const AssemblyChild> kids, int c0, int c1) {
  const std::size_t sf = static_cast<std::size_t>(f);
  for (int j = c0; j < c1; ++j) {
    const std::size_t uj = static_cast<std::size_t>(j);
    std::fill_n(F + uj * sf + uj, sf - uj, 0.0);
  }
  // Original entries of the pivot columns (lower CSC, so pos[row] >= pos[col]).
  for (Int j = orig.col_start; j < orig.col_end; ++j) {
    const Int pj = orig.pos[j];
    if (pj < c0 || pj >= c1) continue;
    double *Fj = F + static_cast<std::size_t>(pj) * sf;
    for (Int k = orig.acp[j]; k < orig.acp[j + 1]; ++k) Fj[orig.pos[orig.ari[k]]] += orig.av[k];
  }
  // Children in order: a child's column jj goes to front column map[jj] (its
  // rows map below it), so the block's columns are the jj with map[jj] in
  // [c0, c1), contiguous because the map increases.
  for (const AssemblyChild &c : kids) {
    if (c.q <= 0) continue;
    const Int *lo = std::lower_bound(c.map, c.map + c.q, static_cast<Int>(c0));
    const Int *hi = std::lower_bound(lo, c.map + c.q, static_cast<Int>(c1));
    const Int j0 = static_cast<Int>(lo - c.map);
    const Int j1 = static_cast<Int>(hi - c.map);
    const std::size_t q = static_cast<std::size_t>(c.q);
    for (Int jj = j0; jj < j1; ++jj) {
      const std::size_t ujj = static_cast<std::size_t>(jj);
      // Packed lower: column jj starts after columns 0 .. jj-1 (q - t entries each).
      const double *cv = c.ld > 0 ? c.values + ujj * (c.ld + 1) : c.values + ujj * q - ujj * (ujj - 1) / 2;
      double *Fc = F + static_cast<std::size_t>(c.map[jj]) * sf;
      for (Int ii = jj; ii < c.q; ++ii) Fc[c.map[ii]] += *cv++;
    }
  }
}

namespace {
// Independent column blocks, no dependencies.
class BlockDag final : public TileDag {
public:
  BlockDag(double *F, int f, const AssemblyOriginal &orig, std::span<const AssemblyChild> kids,
           const std::vector<int> &bounds)
      : F_(F), f_(f), orig_(orig), kids_(kids), bounds_(bounds) {}
  void execute(int t) override {
    const std::size_t ut = static_cast<std::size_t>(t);
    assemble_front_columns(F_, f_, orig_, kids_, bounds_[ut], bounds_[ut + 1]);
  }
  void release(int, TileRunner &) override {}
  bool urgent(int) const override { return false; }

private:
  double *F_;
  int f_;
  const AssemblyOriginal &orig_;
  std::span<const AssemblyChild> kids_;
  const std::vector<int> &bounds_;
};
} // namespace

void assemble_front_parallel(double *F, int f, const AssemblyOriginal &orig,
                             std::span<const AssemblyChild> kids, int nthreads, TileHelpers *helpers) {
  if (nthreads <= 1 || f <= 1) {
    assemble_front_columns(F, f, orig, kids, 0, f);
    return;
  }
  const int nblocks = std::min(f, 4 * nthreads);
  // Block boundaries at equal shares of the lower triangle's area: columns
  // [0, c) hold c f - c (c - 1) / 2 entries.
  const double ff = static_cast<double>(f);
  const double total = ff * (ff + 1.0) / 2.0;
  std::vector<int> bounds;
  bounds.reserve(static_cast<std::size_t>(nblocks) + 1);
  bounds.push_back(0);
  int c = 0;
  for (int b = 1; b < nblocks; ++b) {
    const double target = total * b / nblocks;
    while (c < f) {
      const double cd = static_cast<double>(c);
      if (cd * ff - cd * (cd - 1.0) / 2.0 >= target) break;
      ++c;
    }
    if (c > bounds.back()) bounds.push_back(c);
  }
  if (bounds.back() < f) bounds.push_back(f);
  const int n = static_cast<int>(bounds.size()) - 1;
#ifdef _OPENMP
  // On the calling thread's OpenMP team, the one its MKL calls run on (smf's
  // OpenMP and MKL's intel threading share one runtime): the team is hot
  // from the front's previous BLAS call and stays so for the next one.
  // Lending the blocks to the scheduler's workers instead put extra threads
  // next to MKL's spinning team, and the front's following BLAS calls lost
  // about what the assembly saved (2026-11-09 acopf profile).
  (void)helpers;
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
  for (int b = 0; b < n; ++b)
    assemble_front_columns(F, f, orig, kids, bounds[static_cast<std::size_t>(b)],
                           bounds[static_cast<std::size_t>(b) + 1]);
#else
  std::vector<int> scratch(3 * static_cast<std::size_t>(n), 0);
  int *pending = scratch.data();               // all zero: no predecessors
  int *roots = scratch.data() + n;
  int *ready = scratch.data() + 2 * n;
  for (int t = 0; t < n; ++t) roots[t] = t;
  std::vector<int> ready_lo(static_cast<std::size_t>(n));
  BlockDag dag(F, f, orig, kids, bounds);
  TileRunner runner;
  runner.run(dag, pending, roots, n, n, ready, ready_lo.data(), nthreads, helpers);
#endif
}

void copy_contribution_block(double *dst, const double *cb, std::size_t lda, int q, int nthreads) {
  // Column jj starts at jj q - jj (jj - 1) / 2 in the packed block.
  const auto column = [&](int jj) {
    const std::size_t u = static_cast<std::size_t>(jj);
    const std::size_t sq = static_cast<std::size_t>(q);
    const double *src = cb + u * lda + u;
    std::copy(src, src + (sq - u), dst + u * sq - u * (u - std::size_t{1}) / 2);
  };
#ifdef _OPENMP
  if (nthreads >= 2) {
    // The calling thread's team, as assemble_front_parallel.
#pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads)
    for (int jj = 0; jj < q; ++jj) column(jj);
    return;
  }
#else
  (void)nthreads;
#endif
  for (int jj = 0; jj < q; ++jj) column(jj);
}

} // namespace smf
