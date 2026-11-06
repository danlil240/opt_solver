#pragma once
#include "smf/analysis.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/types.hpp"
#include <vector>

namespace smf {

/// The order in which an indefinite factor actually eliminated its columns,
/// in the original numbering. Delayed columns appear where they were finally
/// eliminated (in an ancestor front), so feeding perm and pairs back as
/// OrderingMethod::UserSupplied makes them start there.
struct RealizedOrder {
  std::vector<Int> perm;  ///< perm[k] = original index eliminated k-th (perm[new] = old)
  std::vector<Int> pairs; ///< flattened original-index pairs (first, second) of 2x2 pivots
  Int zero_pivots = 0;    ///< columns that ended as zero pivots (placed last)
};

/// Read the realized elimination order off fk.diag_entries (any engine: the
/// lean serial and tree-parallel ones, and the legacy Control::tree_parallel
/// one, whose entries are in its own numerical order), mapped to original
/// columns through fk.perm. fk must be a factor of the matrix analysed by ak. perm is
/// empty if fk does not cover all n columns (no factor, or an engine that does
/// not fill diag_entries); an SPD factor returns the analysis order unchanged.
RealizedOrder realized_pivot_order(const AnalysisKeep &ak, const FactorKeep &fk);

} // namespace smf
