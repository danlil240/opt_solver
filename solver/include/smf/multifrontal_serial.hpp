#pragma once

#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/dense_indef.hpp"
#include "smf/error.hpp"
#include "smf/factor_indef.hpp"  // EarlyWorkStop
#include "smf/factor_posdef.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"

namespace smf {

struct SerialFactorStats {
  InertiaCounts inertia{};
  int delayed = 0;
  int static_pivots = 0;
  double max_growth = 0.0;  // largest |l_ij| below a forced (static path) pivot
  int aptp_failed = 0;  // columns the APTP front kernel failed (before the TPP fallback)
  int budget_retries = 0;  // fronts refactored static because they overshot Control::delay_budget
  int parallel_subtrees = 0;
  int top_tile_fronts = 0;  // fronts factored as 2D tile tasks (Control::top_tiles)
  int posdef_lapack_fronts = 0;  // SPD fronts on dpotrf + dtrsm + dsyrk (Control::posdef_front_kernel)
  int top_part_fronts = 0;  // subtree engine: fronts of the top part factored (after the subtrees)
  // Subtree engine: top fronts taken from the scheduler's shared ready list;
  // the others were continued by the thread that factored their last child.
  int top_part_queue_pops = 0;
  int parallel_assembly_fronts = 0;  // Info::parallel_assembly_fronts
  int top_priority_from_realized = 0;  // Info::top_priority_from_realized
  int small_front_handovers = 0;  // contribution blocks handed to the parent in a small-front buffer
  int threads_used = 1;  // threads the factorization ran on (1 when serial)
  // Control::thread_plan_front_costs: the thread plan was rebuilt from the
  // previous factorization's realized front sizes (Info::plan_from_realized).
  int plan_from_realized = 0;
  int factor_store_regathers = 0;  // Info::factor_store_regathers
  int two_by_two_pivots = 0;       // Info::two_by_two_pivots
  int solve_layout_reused = 0;     // Info::solve_layout_reused
  // Subtree engine phases in seconds (0 when serial): 1 the subtrees, 2 the top
  // part (task DAG), 3 the gather of the per-thread factors.
  double subtree_seconds = 0.0;
  double top_seconds = 0.0;
  double gather_seconds = 0.0;
  double flops = 0.0;
  // Sum over fronts of (fully summed columns) x (rows)^2, as realized: the
  // delay-inflated counterpart of Info::predicted_flops (Info::front_work).
  // An integer, so sums are exact in any order.
  long long front_work = 0;
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
/// Indefinite with `work_limit` >= 0: as soon as the front work counted so
/// far (st.front_work, an exact integer sum) exceeds it, the remaining
/// fronts are skipped and ErrorCode::WorkLimitExceeded is returned. Work is only ever added, so
/// this happens exactly when the finished factorization's count would
/// exceed it, and a run within the limit does the work of an unarmed one.
/// `early` (armed only with `work_limit` >= 0): the conditional early stop
/// (EarlyWorkStop), evaluated over the fronts in postorder on every path
/// (the subtree engine publishes each front's work and advances a postorder
/// cursor), so its decision does not depend on the thread count or timing.
///
/// Indefinite with `static_pivot` > 0: every front static-pivots (no
/// delays). `fallback_static_pivot` > 0 (Control::fallback_static_pivot_tol,
/// absolute): only a root front, or a front receiving more delayed columns
/// than its supernode width, static-pivots, with this value in place of
/// `static_pivot`. `budget_static_pivot` > 0 (absolute) arms
/// ctrl.delay_budget when `static_pivot` is 0: fronts that run once the
/// budget is spent static-pivot with it, as `static_pivot` fronts do (see
/// Control::delay_budget for the count on the tree-parallel path).
/// ctrl.static_pivots_as_zero counts raised pivots as zeros;
/// ctrl.static_zero_stays_zero gives `static_pivot` fronts the hybrid
/// accounting of fallback fronts. ctrl.static_pivot_in_order (with
/// `static_pivot` > 0): 1x1 pivots in order, no search.
ErrorCode multifrontal_factor_serial(const AnalysisKeep &keep, const Int *acp,
                                     const Int *ari, const double *av,
                                     const Control &ctrl, double eff_small,
                                     double static_pivot, double fallback_static_pivot,
                                     double budget_static_pivot, bool posdef, FactorKeep &fkeep,
                                     NumericWorkspace &ws, SerialFactorStats &st,
                                     double work_limit = -1.0,
                                     const EarlyWorkStop &early = {});

/// Subtree roots' contribution blocks the tree-parallel factorizations of
/// this process put in their root arenas so far (tests use the difference).
long long root_arena_blocks_taken();

/// Tests: pool thread `tid` (> 0) of every tree-parallel factorization
/// starts its subtree phase only once another thread has taken the task
/// dealt to it (or after 10 s), as a thread that wakes late would; -1 (the
/// default) turns this off.
void set_late_phase1_thread_for_test(int tid);

/// Dealt subtree tasks (the heaviest, one per thread) that a thread other
/// than their own took, by every factorization of this process so far.
long long dealt_tasks_taken_over();

} // namespace smf
