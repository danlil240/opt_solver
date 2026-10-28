#pragma once
#include "smf/control.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/info.hpp"

namespace smf {

/// Apply D^{-1} to x in-place (indefinite case only).
/// x is in permuted ordering, length n.
/// For each supernode s and each fully-summed column k:
///   1×1: x[front_rows(s)[k]] /= D[k,k]
///   2×2: solve [d00 d10; d10 d11]^{-1} * [x[r0]; x[r1]]
/// For SPD (is_posdef): this is a no-op.
void solve_diag(const FactorKeep &fkeep, double *x, int n);

/// Smallest pivot block (p) whose L11^-1 Control::cache_inverse_diag caches.
inline constexpr int kInvDiagMinPivots = 16;
/// Largest kappa_1(L11) whose inverse is cached (above it the front keeps
/// its triangular solve): 1e5 let robot_arm_8k's one large front through,
/// moving its unrefined solution by 4e-7 in backward-error terms, and kept
/// indef_rand_1500's ill-conditioned blocks (backward error 2e-12 against
/// 3e-14 with TRSV); 1e4 keeps every acopf / obstacle_surface front.
inline constexpr double kInvDiagMaxCond = 1e4;

/// Fill fkeep.inv_diag / inv_diag_offset with L11^-1 of every solve step of
/// at least kInvDiagMinPivots pivots (ctrl.cache_inverse_diag), or clear
/// them (cache off, no solve steps, or a factor the solve refuses). The
/// buffers keep their capacity: a refactor of the same structure allocates
/// nothing. Sets info.inverse_diag_fronts / inverse_diag_seconds and adds
/// the time to info.factor_seconds. Deterministic: one front after another
/// in step order, serial BLAS in deterministic mode.
void build_inverse_diag(FactorKeep &fkeep, const Control &ctrl, Info &info);

/// Apply permutation P: x_out[j] = x_in[perm[j]] (perm[new]=old).
void apply_perm(const FactorKeep &fkeep, const double *x_in, double *x_out,
                int n);

/// Apply inverse permutation: x_out[j] = x_in[iperm[j]].
void apply_inv_perm(const FactorKeep &fkeep, const double *x_in, double *x_out,
                    int n);

} // namespace smf
