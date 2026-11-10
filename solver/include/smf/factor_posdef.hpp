#pragma once
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#include <utility>
#include <cstdint>
#include <vector>

namespace smf {

struct DiagSolveEntry {
  int8_t tag = 0;
  /// 1 for a column the kernel left unaccepted at a root and the factor turned
  /// into a zero pivot (tag 1, d00 = 0). Lets realized_pivot_order tell it from
  /// a genuine 1x1 pivot without relying on d00 == 0. Fits in the padding.
  int8_t was_zero = 0;
  Int row0 = 0;
  Int row1 = 0;
  double d00 = 0.0;
  double d10 = 0.0;
  double d11 = 0.0;
};

struct SolveStep {
  LongInt row_offset = 0;
  LongInt factor_offset = 0;
  int p = 0;
  int f = 0;
};

struct NumericWorkspace;
struct FactorValues;
struct SolveWorkspace;

/// Result of SPD or indefinite multifrontal factorization.
struct SlackFactor;

struct FactorKeep {
  /// A factorization through the analysis' slack plan (slack_elimination.hpp):
  /// the pairs' inverses; every other member is then the reduced matrix's
  /// factor (its order SlackPlan::nr). Null: the matrix's own factor.
  std::shared_ptr<SlackFactor> slack;
  std::vector<double>
      factor_values; ///< packed L columns (per supernode, col-major)
  std::vector<LongInt> factor_col_ptr; ///< factor_col_ptr[s] = 64-bit start offset of supernode s
  std::vector<int> perm;           ///< total permutation perm[new] = old
  std::vector<int> iperm;          ///< total inverse permutation iperm[old] = new
  std::vector<int> num_perm;       ///< numerical permutation num_perm[new_piv] = sym_idx
  std::vector<int> num_iperm;      ///< inverse numerical permutation num_iperm[sym_idx] = new_piv
  Int n = 0;                       ///< matrix order (through a slack plan: the reduced matrix's)
  FactorStatus status = FactorStatus::Success; ///< completion status of factorization
  // [s][k] = pivot type for fully-summed column k of supernode s:
  //  1 = 1×1 accepted, 2 = first of 2×2, -1 = second of 2×2, 0 = delayed
  std::vector<std::vector<int8_t>> pivot_types; ///< [s][k] = pivot type per supernode column
  std::vector<DiagSolveEntry> diag_entries; ///< flattened D blocks for solve_diag
  std::vector<SolveStep> solve_steps; ///< flattened front traversal for triangular solves
  std::vector<Int> solve_row_indices; ///< row indices referenced by solve_steps
  // Tree-parallel factorizations also leave the solve its parallel layout:
  // solve_steps[0, solve_top_begin) is a sequence of complete, independent
  // subtrees (solve_subtrees, [begin, end) step ranges), the rest the top of
  // the tree. solve_top_rows lists the top part's pivot rows and
  // solve_top_slot maps a row to its index there (-1 for subtree rows).
  // solve_top_parent[i] is the top step (relative to solve_top_begin) of
  // top step i's parent front, -1 at a root.
  std::vector<std::pair<LongInt, LongInt>> solve_subtrees;
  LongInt solve_top_begin = 0;
  std::vector<Int> solve_top_rows;
  std::vector<Int> solve_top_slot;
  std::vector<Int> solve_top_parent;
  /// AnalysisKeep::serial of the analysis whose per-analysis setup (perm,
  /// iperm, pivot_types sized per supernode) this FactorKeep holds from its
  /// last lean indefinite factorization; 0 after any other outcome. A
  /// refactorization of the same analysis skips re-initialising it.
  std::uint64_t setup_analysis = 0;
  // The solve's thread count: with Control::auto_threads, the solve's cost
  // model (src/parallel/thread_plan.hpp, plan_solve_threads, at the
  // factorization's Control::num_threads); off, the factorization's thread
  // count. Either at most solve_subtrees.size(), since a solve runs at most
  // one thread per subtree. The solves use at most this many threads (and
  // at most their num_threads). Every factorization sets it; 1, a serial
  // solve, without a parallel layout.
  int solve_threads = 1;
  // Control::cache_inverse_diag: inv_diag_offset[k] is where solve step k's
  // L11^-1 starts in inv_diag (p x p column-major, zero above the diagonal),
  // -1 for a step without one (p < kInvDiagMinPivots). Empty: no cache. The
  // inverse is of L11 as the triangular solves use it: unit lower for an
  // indefinite factor (D left out), non-unit for SPD.
  std::vector<double> inv_diag;
  std::vector<LongInt> inv_diag_offset;
  // inline_d: D^-1 of every pivot block, built once per factorization
  // (build_inverse_diag) so the backward solve multiplies instead of
  // dividing (a 2x2 block's scaled Cramer solve on every right-hand side).
  // Three entries per pivot, pivots in solve-step order (step k's first at
  // 3 * dinv_offset[k]): a 1x1 pivot [1/d, 0, 0] (0 for a zero or
  // non-finite d); the first pivot of a 2x2 block [e00, e10, e11] of the
  // block's inverse (all zero for a singular block), its second pivot
  // unused. dinv_kind: 1, 2 (first of a 2x2), 0 (second of a 2x2). Empty:
  // apply D from the factor's diagonal slots (apply_inline_d).
  std::vector<double> dinv;
  std::vector<std::int8_t> dinv_kind;
  std::vector<LongInt> dinv_offset;
  // Every solve step has fewer than kInvDiagMinPivots pivots and no cached
  // L11^-1 (build_inverse_diag): the solve's front kernels are all scalar
  // loops, and solve_full_interleaved takes every serial full solve.
  bool solve_small_fronts = false;
  std::vector<double> solve_tmp; ///< legacy scratch (not mutated by const solve)
  std::vector<double> solve_loc; ///< legacy scratch (not mutated by const solve)
  std::vector<double> solve_ext; ///< legacy scratch (not mutated by const solve)
  bool is_posdef = false;          ///< true if factorization used SPD path (D = I)
  /// Pivots perturbed by static pivoting (Info::static_pivots of the
  /// factorization that produced this factor): refine() escalates sooner.
  int static_pivots = 0;
  /// The factor came from the static path (Control::static_fallback_after_
  /// cascade, Pivoting::Static): refine() aims at Control::refine_omega_target
  /// (see refinement_advised()).
  bool static_path = false;
  /// Largest |l_ij| below a pivot the static path forced (one that failed
  /// the threshold test; 0 when none was): growth threshold pivoting would
  /// have bounded by 1/pivot_u.
  double pivot_growth = 0.0;
  /// Control::static_fallback_after_cascade: later factorizations of
  /// delay_analysis into this FactorKeep take the static path (with
  /// delay_scale when it is not empty).
  bool delay_static = false;
  /// The analysis (serial) for which a static-path factor of that fallback
  /// was thrown away (it perturbed pivots): no further static tries for it
  /// into this FactorKeep.
  std::uint64_t static_rejected_analysis = 0;
  /// Indefinite factor whose unit-L fronts keep D in their diagonal slots
  /// (a 2x2 block's off-diagonal at (c, c + 1)): the backward solve applies
  /// D^-1 front by front instead of a separate pass over diag_entries.
  bool inline_d = false;
  bool use_banded_spd = false;     ///< true when SPD factor stored in banded LAPACK form
  int band_n = 0;                  ///< order for banded SPD factor
  int band_kd = 0;                 ///< lower half-bandwidth for banded SPD factor
  std::vector<double> band_factor; ///< banded Cholesky factor (ldab=band_kd+1, col-major)
  std::vector<double> scale;       ///< positive diagonal scaling S applied during factorization
  /// Control::scale_on_delay_ratio: the analysis (AnalysisKeep::serial;
  /// 0 = none) whose factorization into this FactorKeep fired the delay
  /// trigger, and what came of it: the matching scale it factored with
  /// (delay_scale), which later factorizations of that analysis into this
  /// FactorKeep start from; or, delay_scale empty, that the values had no
  /// matching scaling, so later ones with the same zero pattern
  /// (delay_signature) run unarmed.
  std::uint64_t delay_analysis = 0;
  std::vector<double> delay_scale;
  /// The backward error of the accuracy probe (scale_on_delay_probe) when
  /// delay_scale was fresh; a later factor with it must stay close. -1:
  /// none recorded (reuse off, or a factor with zero pivots).
  double delay_omega = -1.0;
  /// delay_scale empty: the zero pattern of the values that had no
  /// matching scaling (another pattern is tried again).
  std::uint64_t delay_signature = 0;
  /// Control::scale_on_delay_next_ratio: the factorization of
  /// delay_analysis was let through but costly, so the next one runs as
  /// Auto resolves it under the tighter limit, or with
  /// Control::scale_on_delay_carry starts from a new matching scaling
  /// without a try (delay_scale is empty).
  bool delay_next = false;
  // keep the analysis tree for the solve phase
  const AnalysisKeep *analysis = nullptr;
};

/// Perform SPD multifrontal factorization.
/// Returns FactorStatus::Success or FactorStatus::NotPositiveDefinite.
FactorStatus factor_posdef(const AnalysisKeep &keep, const Control &ctrl,
                           Info &info, FactorKeep &fkeep);

FactorStatus factor_posdef(const AnalysisKeep &keep, const CscLower &updated,
                           const Control &ctrl, Info &info,
                           FactorKeep &fkeep, NumericWorkspace &ws);

/// Same, with values in the analysed matrix's entry order.
FactorStatus factor_posdef(const AnalysisKeep &keep, const FactorValues &values,
                          const Control &ctrl, Info &info,
                          FactorKeep &fkeep, NumericWorkspace &ws);

} // namespace smf
