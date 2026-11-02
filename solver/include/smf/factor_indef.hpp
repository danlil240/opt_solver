#pragma once
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/dense_indef.hpp"   // InertiaCounts, Pivot, add_inertia_1x1/2x2
#include "smf/factor_posdef.hpp" // reuse FactorKeep
#include "smf/info.hpp"
#include "smf/types.hpp"

namespace smf {

/// Conditional early stop of a work-limited factorization (the scale-on-
/// delay rescue's first try, Control::scale_on_delay_early_ratio).
/// `free_work` is its delay-free front work F (delay_free_front_work).
/// Taking the fronts in postorder, at the front whose work w takes the
/// count `done` of the fronts before it past floor(ratio x F) (done <=
/// floor(ratio x F) < done + w), the factorization stops as at the work
/// limit (WorkLimitExceeded, FactorStatus::MaxPivotDelays, w counted) when
/// (done + w) / (delay-free work of the fronts up to that one, included) >
/// proj; otherwise only the work limit applies. Armed only with a work
/// limit >= 0; ratio < 0: off.
struct EarlyWorkStop {
  double free_work = 0.0;
  double ratio = -1.0;
  double proj = 0.0;
  bool on() const { return ratio >= 0.0; }
};

/// Perform indefinite LDLᵀ multifrontal factorization.
/// Returns FactorStatus (Success, Singular, etc.).
/// Populates info.num_negative, num_positive, num_zero, delayed_pivots.
/// `work_limit` >= 0 (the scale-on-delay trigger, Control::
/// scale_on_delay_ratio; Solver::factor sets it): once the front work
/// (Info::front_work) counted exceeds it the factorization stops, unusable,
/// with FactorStatus::MaxPivotDelays (info.front_work and delayed_pivots the
/// partial counts). `early`: the conditional early stop under that limit
/// (EarlyWorkStop).
FactorStatus factor_indef(const AnalysisKeep &keep, const Control &ctrl,
                          Info &info, FactorKeep &fkeep, double work_limit = -1.0,
                          const EarlyWorkStop &early = {});

FactorStatus factor_indef(const AnalysisKeep &keep, const CscLower &updated,
                          const Control &ctrl, Info &info,
                          FactorKeep &fkeep, NumericWorkspace &ws, double work_limit = -1.0,
                          const EarlyWorkStop &early = {});

/// Same, with values in the analysed matrix's entry order.
FactorStatus factor_indef(const AnalysisKeep &keep, const FactorValues &values,
                          const Control &ctrl, Info &info,
                          FactorKeep &fkeep, NumericWorkspace &ws, double work_limit = -1.0,
                          const EarlyWorkStop &early = {});

/// Kernel flops of a factorization of `keep` without delays, counted as the
/// kernels count Info::actual_flops: a pivot with r rows below it in its
/// front costs r + r (r + 1) (2x2 pivots about the same per column).
double delay_free_kernel_flops(const AnalysisKeep &keep);

/// Front work (Info::front_work) of a factorization of `keep` without
/// delays: sum over fronts of pivots x rows^2 (Info::predicted_flops).
double delay_free_front_work(const AnalysisKeep &keep);

} // namespace smf
