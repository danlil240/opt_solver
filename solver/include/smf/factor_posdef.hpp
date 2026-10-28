#pragma once
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#include <utility>
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
struct FactorKeep {
  std::vector<double>
      factor_values; ///< packed L columns (per supernode, col-major)
  std::vector<LongInt> factor_col_ptr; ///< factor_col_ptr[s] = 64-bit start offset of supernode s
  std::vector<int> perm;           ///< total permutation perm[new] = old
  std::vector<int> iperm;          ///< total inverse permutation iperm[old] = new
  std::vector<int> num_perm;       ///< numerical permutation num_perm[new_piv] = sym_idx
  std::vector<int> num_iperm;      ///< inverse numerical permutation num_iperm[sym_idx] = new_piv
  Int n = 0;                       ///< matrix order
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
  // Control::cache_inverse_diag: inv_diag_offset[k] is where solve step k's
  // L11^-1 starts in inv_diag (p x p column-major, zero above the diagonal),
  // -1 for a step without one (p < kInvDiagMinPivots). Empty: no cache. The
  // inverse is of L11 as the triangular solves use it: unit lower for an
  // indefinite factor (D left out), non-unit for SPD.
  std::vector<double> inv_diag;
  std::vector<LongInt> inv_diag_offset;
  std::vector<double> solve_tmp; ///< legacy scratch (not mutated by const solve)
  std::vector<double> solve_loc; ///< legacy scratch (not mutated by const solve)
  std::vector<double> solve_ext; ///< legacy scratch (not mutated by const solve)
  bool is_posdef = false;          ///< true if factorization used SPD path (D = I)
  /// Pivots perturbed by static pivoting (Info::static_pivots of the
  /// factorization that produced this factor): refine() escalates sooner.
  int static_pivots = 0;
  /// Indefinite factor whose unit-L fronts keep D in their diagonal slots
  /// (a 2x2 block's off-diagonal at (c, c + 1)): the backward solve applies
  /// D^-1 front by front instead of a separate pass over diag_entries.
  bool inline_d = false;
  bool use_banded_spd = false;     ///< true when SPD factor stored in banded LAPACK form
  int band_n = 0;                  ///< order for banded SPD factor
  int band_kd = 0;                 ///< lower half-bandwidth for banded SPD factor
  std::vector<double> band_factor; ///< banded Cholesky factor (ldab=band_kd+1, col-major)
  std::vector<double> scale;       ///< positive diagonal scaling S applied during factorization
  /// Control::scale_on_delay_ratio: the analysis whose factorization into
  /// this FactorKeep last fired the delay trigger (only compared, never
  /// dereferenced). Later factorizations of it into this FactorKeep use the
  /// matching scaling of their own values from the start (a new analysis
  /// that happens to get the same address inherits that: harmless, any
  /// matching scaling is valid). Null: none.
  const AnalysisKeep *delay_matching_analysis = nullptr;
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
