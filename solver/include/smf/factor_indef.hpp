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
/// `work_limit` >= 0 (the scale-on-delay trigger, Control::
/// scale_on_delay_ratio; Solver::factor sets it): once the front work
/// (Info::front_work) counted exceeds it the factorization stops, unusable,
/// with FactorStatus::MaxPivotDelays (info.front_work and delayed_pivots the
/// partial counts).
FactorStatus factor_indef(const AnalysisKeep &keep, const Control &ctrl,
                          Info &info, FactorKeep &fkeep, double work_limit = -1.0);

FactorStatus factor_indef(const AnalysisKeep &keep, const CscLower &updated,
                          const Control &ctrl, Info &info,
                          FactorKeep &fkeep, NumericWorkspace &ws, double work_limit = -1.0);

/// Same, with values in the analysed matrix's entry order.
FactorStatus factor_indef(const AnalysisKeep &keep, const FactorValues &values,
                          const Control &ctrl, Info &info,
                          FactorKeep &fkeep, NumericWorkspace &ws, double work_limit = -1.0);

/// Kernel flops of a factorization of `keep` without delays, counted as the
/// kernels count Info::actual_flops: a pivot with r rows below it in its
/// front costs r + r (r + 1) (2x2 pivots about the same per column).
double delay_free_kernel_flops(const AnalysisKeep &keep);

/// Front work (Info::front_work) of a factorization of `keep` without
/// delays: sum over fronts of pivots x rows^2 (Info::predicted_flops).
double delay_free_front_work(const AnalysisKeep &keep);

} // namespace smf
