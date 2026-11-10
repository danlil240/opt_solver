#pragma once
// Column-block assembly of a front (8T engine Part 5).
//
// A front's assembly writes each entry of its lower triangle: zero, then the
// front's own original entries, then every child's contribution block in
// child order. Split by destination column, the blocks are independent: each
// entry still receives the same additions in the same order, so assembling
// the columns in any blocks, on any number of threads, gives the serial
// values bit for bit. process_front uses it for large top fronts with a BLAS
// share of 2 or more, lending the share's threads through TileHelpers.

#include "smf/types.hpp"

#include <cstddef>
#include <span>

namespace smf {

class TileHelpers;

// One child's contribution block, as process_front reads it.
struct AssemblyChild {
  const Int *map = nullptr;       // front position of each of the q rows; strictly increasing
  const double *values = nullptr; // packed lower block (ld == 0), or first column of a strided block
  std::size_t ld = 0;             // > 0: strided (small-front buffer), column jj starts at values + jj*(ld+1)
  Int q = 0;
};

// The front's own pivot columns' original entries (lower CSC): column j goes
// to front column pos[j], row ari[k] to pos[ari[k]].
struct AssemblyOriginal {
  const Int *acp = nullptr;
  const Int *ari = nullptr;
  const Int *pos = nullptr;
  const double *av = nullptr;
  Int col_start = 0;
  Int col_end = 0;
};

// True when map is strictly increasing (an AssemblyChild's requirement).
bool child_maps_increasing(std::span<const Int> map) noexcept;

// Columns [c0, c1) of the f x f front F (leading dimension f, lower
// triangle): zero fill, original entries, then each child in order. Each
// entry gets the serial assembly's additions in the same order.
void assemble_front_columns(double *F, int f, const AssemblyOriginal &orig,
                            std::span<const AssemblyChild> kids, int c0, int c1);

// The whole front in 4 * nthreads column blocks of equal lower-triangle area,
// on nthreads threads: the calling thread's OpenMP team when built with
// OpenMP (helpers unused), else a TileRunner (helpers, or the runner's own
// threads when null). nthreads <= 1: one block on the caller.
void assemble_front_parallel(double *F, int f, const AssemblyOriginal &orig,
                             std::span<const AssemblyChild> kids, int nthreads, TileHelpers *helpers);

// The q x q contribution block whose top-left entry is cb (leading dimension
// lda, lower part) into dst, packed column by column (q (q + 1) / 2 values),
// on nthreads threads (the calling thread's OpenMP team; serial without
// OpenMP).
void copy_contribution_block(double *dst, const double *cb, std::size_t lda, int q, int nthreads);

} // namespace smf
