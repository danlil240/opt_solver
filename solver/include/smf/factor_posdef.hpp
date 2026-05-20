#pragma once
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#include <vector>

namespace smf {

struct DiagSolveEntry {
  int8_t tag = 0;
  Int row0 = 0;
  Int row1 = 0;
  double d00 = 0.0;
  double d10 = 0.0;
  double d11 = 0.0;
};

struct SolveStep {
  Int row_offset = 0;
  int factor_offset = 0;
  int p = 0;
  int f = 0;
};

/// Result of SPD multifrontal factorization.
struct FactorKeep {
  std::vector<double>
      factor_values; ///< packed L columns (per supernode, col-major)
  std::vector<int> factor_col_ptr; ///< factor_col_ptr[s] = start of supernode s
                                   ///< in factor_values
  std::vector<int> perm;           ///< copy of analysis permutation
  std::vector<int> iperm;          ///< copy of inverse permutation
  // [s][k] = pivot type for fully-summed column k of supernode s:
  //  1 = 1×1 accepted, 2 = first of 2×2, -1 = second of 2×2, 0 = delayed
  std::vector<std::vector<int8_t>> pivot_types; ///< [s][k] = pivot type per supernode column
  std::vector<DiagSolveEntry> diag_entries; ///< flattened D blocks for solve_diag
  std::vector<SolveStep> solve_steps; ///< flattened front traversal for triangular solves
  std::vector<Int> solve_row_indices; ///< row indices referenced by solve_steps
  mutable std::vector<double> solve_tmp; ///< reusable solve permutation scratch
  mutable std::vector<double> solve_loc; ///< reusable local pivot scratch
  mutable std::vector<double> solve_ext; ///< reusable extension scratch
  bool is_posdef = false;          ///< true if factorization used SPD path (D = I)
  // keep the analysis tree for the solve phase
  const AnalysisKeep *analysis = nullptr;
};

/// Perform SPD multifrontal factorization.
/// Returns FactorStatus::Success or FactorStatus::NotPositiveDefinite.
FactorStatus factor_posdef(const AnalysisKeep &keep, const Control &ctrl,
                           Info &info, FactorKeep &fkeep);

} // namespace smf
