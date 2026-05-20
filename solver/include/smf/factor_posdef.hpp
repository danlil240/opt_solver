#pragma once
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/factor_stack.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#include "smf/utils/memory.hpp"
#include <vector>

namespace smf {

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
  bool is_posdef = false;          ///< true if factorization used SPD path (D = I)
  bool use_banded_spd = false;     ///< true when SPD factor stored in banded LAPACK form
  int band_n = 0;                  ///< order for banded SPD factor
  int band_kd = 0;                 ///< lower half-bandwidth for banded SPD factor
  std::vector<double> band_factor; ///< banded Cholesky factor (ldab=band_kd+1, col-major)
  // keep the analysis tree for the solve phase
  const AnalysisKeep *analysis = nullptr;
};

/// Perform SPD multifrontal factorization.
/// Returns FactorStatus::Success or FactorStatus::NotPositiveDefinite.
FactorStatus factor_posdef(const AnalysisKeep &keep, const Control &ctrl,
                           Info &info, FactorKeep &fkeep);

} // namespace smf
