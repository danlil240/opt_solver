#pragma once
// Internal steps of the APTP front kernel (ldlt_aptp.cpp), shared by its
// serial sweep and by the 2D tile tasks of the top of the tree
// (parallel/tile_tasks.cpp). Not part of the public interface.
//
// One block column [c0, c0 + w) of a sweep is
//   block_factor  - back up the column, factor the diagonal tile, apply its
//                   permutation to the rest of the view, form L_t / D^{-1};
//   apply_n       - ApplyN on any range of rows below the tile (rows are
//                   independent: the tile tasks split them into row blocks);
//   block_finish  - ApplyT, restore the failed columns, UpdateNN (all of it,
//                   or only the block column's own failed columns, leaving
//                   the other eligible columns to update_tile), UpdateNT/TN.
// After the last block column: sweep_compact, then the deferred update of
// the non-eligible trailing triangle (trailing_update, or per tile with
// trailing_form_w + trailing_tile).
//
// Scratch use: block_factor writes cx.bk/lt/dinv/lpc (and the inner level's
// scratch); apply_n only reads cx.lt / cx.dinv; block_finish uses cx.bk,
// ga, gb, gc. update_tile / trailing_* use no scratch, so they may run
// concurrently with each other and with the next block column's steps on
// disjoint data.

#include "smf/ldlt_aptp.hpp"

#include <cstddef>
#include <cstdint>

namespace smf::aptp_detail {

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

/// One APTP sweep over the leading `ne` columns of the m x m view (A, ld):
/// level 0 is the front (block nb), level 1 a diagonal tile (block ib).
struct Sweep {
    double *A = nullptr;
    int ld = 0;
    int m = 0;
    int ne = 0;
    int level = 0;
    int bs = 0;
    int8_t *tags = nullptr;
    Int *rows = nullptr;
    int *lp = nullptr; // lp[t]: view position that moved to t
    int *Q = nullptr;  // failed columns so far (positions)
    int nq = 0;
    int num_elim = 0;
};

inline std::size_t idx(int j, int ld) noexcept {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

/// Carves the scratch of aptp_work_size(f, num_elig, prm) out of `work`.
Ctx make_ctx(int f, int num_elig, const AptpParams &prm, double *work);

Sweep sweep_begin(double *A, int ld, int m, int ne, int level, int8_t *tags, Int *rows,
                  int *lp, Ctx &cx);

/// Steps 1-3 of block column [c0, c0 + w) plus L_t and D^{-1} (cx.lt,
/// cx.dinv) when anything was eliminated. Returns ne_t, the tile's pivots.
int block_factor(Sweep &sw, int c0, int w, Ctx &cx);

/// ApplyN on rows [r0, r1), r0 >= c0 + w: L = X L_t^{-T} D^{-1} in place,
/// pivot by pivot until the first with an |l| > 1/u in these rows, whose
/// start it returns (ne_t if none failed).
int apply_n(const Sweep &sw, int c0, int ne_t, int r0, int r1, const Ctx &cx);

enum class UpdateNN {
    Full,     ///< every eligible column right of the pivots (serial sweep)
    OwnBlock, ///< only [c0 + nelim, c0 + w); (L D)^T formed for all of them
};

/// Rest of the block column given `nelim` = min over ApplyN results (ne_t
/// if no rows below): ApplyT, restore, UpdateNN per `mode`, UpdateNT/TN.
/// Records the failed columns in sw.Q and the pivots in sw.num_elim.
/// Returns the block column's accepted pivots.
int block_finish(Sweep &sw, int c0, int w, int ne_t, int nelim, UpdateNN mode, Ctx &cx);

/// UpdateNN of one tile after block_finish(..., OwnBlock):
/// A[r0:r1, j0:j1] -= L[r0:r1, c0:c0+nelim] (L D)^T, j0 >= c0 + w. A
/// diagonal tile (r0 == j0, r1 == j1) is updated in its lower triangle.
void update_tile(const Sweep &sw, int c0, int nelim, int j0, int j1, int r0, int r1);

/// Moves the failed columns behind the eliminated ones.
void sweep_compact(Sweep &sw);

/// Deferred update of the non-eligible trailing triangle F[ne:m, ne:m] at
/// rank num_elim, after sweep_compact: in one call, or per tile with
/// trailing_form_w for columns [j0, j1) and then trailing_tile.
void trailing_update(Sweep &sw);
void trailing_form_w(const Sweep &sw, int j0, int j1);
void trailing_tile(const Sweep &sw, int j0, int j1, int r0, int r1);

/// Inertia and flops of a front whose sweep eliminated `pe` pivots.
AptpResult front_result(const double *F, int f, int num_elig, int pe, const AptpParams &prm,
                        const int8_t *tags);

} // namespace smf::aptp_detail
