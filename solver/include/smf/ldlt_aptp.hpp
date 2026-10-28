#pragma once

#include "smf/dense_indef.hpp"
#include "smf/types.hpp"

#include <cstddef>
#include <cstdint>

namespace smf {

/// Parameters of the APTP (a posteriori threshold pivoting) front kernel.
struct AptpParams {
    double u = 0.01;      ///< a posteriori bound: every accepted |l| <= 1/u
    int nb = 128;         ///< outer block (tile) width
    int ib = 32;          ///< inner block width inside a tile
    double small = 1e-20; ///< entries below this are treated as zero
};

struct AptpResult {
    int num_eliminated = 0;
    int num_failed = 0;   ///< num_elig - num_eliminated
    InertiaCounts inertia{};
    double flops = 0.0;   ///< same accounting as ldlt_factor_front
};

/// In-place LDL^T of the leading `num_elig` (fully summed) columns of a
/// symmetric front with SSIDS-style a posteriori threshold pivoting.
///
/// The fully summed columns are processed in block columns of width nb. Each
/// diagonal tile is factorized speculatively (inner APTP with block ib and
/// complete pivoting on the ib x ib diagonal blocks, then complete pivoting on
/// whatever failed inside the tile); the L tiles below are formed with a TRSM
/// and the block column keeps only its leading pivots whose |l| <= 1/u. The
/// rest are restored from a backup and stay in place, kept up to date, until
/// the end of the sweep. Failed columns are not retried at the front level:
/// they are the caller's fallback.
///
/// Output layout and tags are those of ldlt_factor_front (ldlt_kernel.hpp):
/// F is f x f column-major (ld f), lower triangle; with p = num_eliminated,
///  - columns [0, p) hold L strictly below the pivot blocks, unscaled D on
///    the diagonal and the 2x2 off-diagonal at (c+1, c);
///  - tags[c] is 1 (1x1), 2 (first of 2x2) or -1 (second of 2x2) for c < p,
///    and 0 for c in [p, num_elig);
///  - the lower triangle of F[p:f, p:f] is the Schur complement, with the
///    failed eligible columns in positions [p, num_elig);
///  - `rows` is permuted in sync with every symmetric swap (rows from
///    num_elig on never move).
/// Like ldlt_factor_front, the strict upper triangle is used as scratch.
///
/// `work` must hold at least aptp_work_size(f, num_elig, prm) doubles.
AptpResult ldlt_aptp_front(double *F, int f, int num_elig, Int *rows,
                           const AptpParams &prm, int8_t *tags, double *work);

/// Doubles ldlt_aptp_front needs in `work`: the backup of one outer block
/// column (f x nb) and of one inner block column (nb x ib), four nb x nb
/// scratch tiles, and index arrays.
std::size_t aptp_work_size(int f, int num_elig, const AptpParams &prm);

} // namespace smf
