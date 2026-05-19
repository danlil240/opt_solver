#pragma once
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/dense_indef.hpp"   // InertiaCounts, Pivot, add_inertia_1x1/2x2
#include "smf/factor_posdef.hpp" // reuse FactorKeep
#include "smf/info.hpp"
#include "smf/types.hpp"

namespace smf {

/// Perform indefinite LDLᵀ multifrontal factorization.
/// Returns FactorStatus (Success, Singular, etc.).
/// Populates info.num_negative, num_positive, num_zero, delayed_pivots.
FactorStatus factor_indef(const AnalysisKeep &keep, const Control &ctrl,
                          Info &info, FactorKeep &fkeep);

} // namespace smf
