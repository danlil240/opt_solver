#pragma once
// TPP kernel with panel lookahead (8T engine Part 6).
//
// ldlt_factor_front's blocked path, with panel k+1 searched and factored
// while panel k's deferred update of the columns past panel k+1 and its chunk
// of the Schur update run on the other threads. Per panel k three tasks:
//   P_k  search and apply panel k's pivots (BLAS on 1 thread; urgent);
//   N_k  panel k's deferred update of panel k+1's columns (1 thread; urgent);
//   R_k  panel k's deferred update of the columns past panel k+1, then its
//        Schur chunk (BLAS on the other threads);
// with N_k after P_k and R_{k-1}, P_{k+1} after N_k, R_k after P_k and
// R_{k-1}. Panel k+1's pivot search and 2x2 partners stay inside its panel,
// so its symmetric swaps touch only rows of panel k+1 in earlier columns and
// panel k+1's own columns: none of what R_k reads or writes, and no
// interchange has to be recorded. A panel that stalls ends the lookahead: the
// later tasks run as no-ops and ldlt_factor_front's code continues from that
// panel (its deferred update, the partner pairing / moves, the full search).
// Pivots (rows, tags), num_accepted and inertia are ldlt_factor_front's; the
// Schur update is summed in per-panel chunks, so L's rows and the Schur
// complement differ from it by rounding only.

#include "smf/ldlt_kernel.hpp"

#include <cstddef>

namespace smf {

class TileHelpers;

// Doubles of `work` ldlt_factor_front_lookahead needs: two panel buffers,
// each laid out as ldlt_factor_front's work.
inline std::size_t ldlt_lookahead_work_size(int f, const LdltKernelParams &prm) {
  return 2 * ldlt_work_size(f, prm);
}

// ldlt_factor_front (first = 0) with panel lookahead on nthreads threads
// (helpers, or the TileRunner's own threads when null). Only a plain
// threshold front (!posdef, static_pivot == 0, min_abs_pivot == 0, f >=
// blocked_min_front, nb > 1, more than one panel) with nthreads >= 2 takes
// the lookahead; any other calls ldlt_factor_front.
LdltKernelResult ldlt_factor_front_lookahead(double *F, int f, int num_elig, Int *rows,
                                             const LdltKernelParams &prm, int8_t *tags, double *work,
                                             int nthreads, TileHelpers *helpers);

} // namespace smf
