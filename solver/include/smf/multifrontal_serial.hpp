#pragma once

#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/dense_indef.hpp"
#include "smf/error.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"

namespace smf {

struct SerialFactorStats {
  InertiaCounts inertia{};
  int delayed = 0;
  int static_pivots = 0;
  int parallel_subtrees = 0;
  double flops = 0.0;
  int max_front = 0;
  // ErrorCode::InertiaExceeded: predicted flops of the fronts not factored.
  double aborted_flops_skipped = 0.0;
};

/// Lean serial multifrontal factorization shared by the SPD and indefinite
/// drivers.
///
/// Fronts are assembled into one reusable dense buffer through a global
/// row->position map, contribution blocks live on a LIFO stack in packed
/// lower-triangular form, and the factor goes straight into the solve layout
/// of `fkeep` (rows labelled by their symbolic index, so the numerical
/// permutation is the identity). Buffers are planned from the symbolic
/// structure and reused across calls: a refactorization allocates nothing
/// unless delayed pivots make a front, the stack or the factor larger than
/// planned, which is what ws.capacity_growths_last_call reports.
///
/// `acp/ari/av` are the permuted lower-CSC matrix. With `posdef`, columns
/// are eliminated in order without pivoting, L is written as the Cholesky
/// factor (fkeep.is_posdef layout) and a non-positive pivot returns
/// ErrorCode::NotPositiveDefinite. Otherwise threshold 1x1/2x2 pivoting with
/// delayed pivots produces unit L plus fkeep.diag_entries.
///
/// Indefinite with ctrl.max_negative_pivots >= 0: as soon as the negative
/// pivots accepted so far exceed it, the remaining fronts are skipped and
/// ErrorCode::InertiaExceeded is returned, `st.inertia` holding the partial
/// counts and `st.aborted_flops_skipped` the skipped fronts' predicted flops.
///
/// Indefinite with `static_pivot` > 0: every front static-pivots (no
/// delays). `fallback_static_pivot` > 0 (Control::fallback_static_pivot_tol,
/// absolute): only a root front, or a front receiving more delayed columns
/// than its supernode width, static-pivots, with this value in place of
/// `static_pivot`. ctrl.static_pivots_as_zero counts raised pivots as zeros.
ErrorCode multifrontal_factor_serial(const AnalysisKeep &keep, const Int *acp,
                                     const Int *ari, const double *av,
                                     const Control &ctrl, double eff_small,
                                     double static_pivot, double fallback_static_pivot,
                                     bool posdef, FactorKeep &fkeep,
                                     NumericWorkspace &ws, SerialFactorStats &st);

} // namespace smf
