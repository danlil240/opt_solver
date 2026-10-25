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
  LongInt row_offset = 0;
  LongInt factor_offset = 0;
  int p = 0;
  int f = 0;
};

struct NumericWorkspace;
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
  std::vector<double> solve_tmp; ///< legacy scratch (not mutated by const solve)
  std::vector<double> solve_loc; ///< legacy scratch (not mutated by const solve)
  std::vector<double> solve_ext; ///< legacy scratch (not mutated by const solve)
  bool is_posdef = false;          ///< true if factorization used SPD path (D = I)
  bool use_banded_spd = false;     ///< true when SPD factor stored in banded LAPACK form
  int band_n = 0;                  ///< order for banded SPD factor
  int band_kd = 0;                 ///< lower half-bandwidth for banded SPD factor
  std::vector<double> band_factor; ///< banded Cholesky factor (ldab=band_kd+1, col-major)
  std::vector<double> scale;       ///< positive diagonal scaling S applied during factorization
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

} // namespace smf
