#include "smf/ipm_session.hpp"

#include "smf/pivot_order.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>

namespace smf {

namespace {

// Flops of a factorization of `ak` without delays, counted as the kernel
// counts Info::actual_flops: a pivot with r rows below it in its front costs
// r + r (r + 1) (2x2 pivots count about the same per column).
double kernel_flops(const AnalysisKeep &ak) {
  double total = 0.0;
  for (std::size_t s = 0; s < ak.supernodes.size(); ++s) {
    const double f = static_cast<double>(ak.fronts[s].front_size());
    const Int p = ak.supernodes[s].width();
    for (Int k = 0; k < p; ++k) {
      const double r = f - static_cast<double>(k) - 1.0;
      total += r + r * (r + 1.0);
    }
  }
  return total;
}

} // namespace

ErrorCode IpmSession::analyse(const CscLower& pattern, std::span<const double> values,
                              const Control& ctrl, const IpmControl& ipm) {
  // A new pattern starts a new history (the factor before the analyses it
  // points at).
  fk_ = FactorKeep{};
  ak_.reset();
  nws_ = NumericWorkspace{};
  sws_ = SolveWorkspace{};
  info_ = Info{};
  solve_info_ = Info{};
  stats_ = IpmStats{};
  ctrl_ = ctrl;
  ipm_ = ipm;
  factor_ctrl_ = ctrl;
  factor_scale_.clear();
  expected_neg_.reset();
  user_scale_once_ = false;
  solvable_ = false;
  aborted_ = false;
  pending_accept_ = false;
  last_spd_ok_ = false;
  prev_ak_.reset();
  prev_info_ = Info{};
  order_perm_.clear();
  order_pairs_.clear();
  reanalyse_armed_ = false;
  trigger_delays_ = 0;
  trigger_flops_ = 0.0;
  last_attempt_ = -1;
  probation_ = false;
  probation_delays_ = 0;
  revert_pending_ = false;
  reanalyse_stopped_ = false;

  mat_.n = pattern.n;
  mat_.col_ptr = pattern.col_ptr;
  mat_.row_idx = pattern.row_idx;
  const std::size_t nnz = mat_.row_idx.size();
  mat_.values.assign(nnz, 0.0);
  if (mat_.n <= 0 || mat_.validate_shape() != ErrorCode::Success ||
      static_cast<std::size_t>(mat_.nnz()) != nnz || (!values.empty() && values.size() != nnz)) {
    mat_ = CscLower{};
    info_.status = ErrorCode::IllegalValue;
    return info_.status;
  }
  if (values.empty()) return ErrorCode::Success;  // analysis deferred to factor()

  std::copy(values.begin(), values.end(), mat_.values.begin());
  return run_analysis();
}

ErrorCode IpmSession::run_analysis() {
  const auto t0 = std::chrono::steady_clock::now();
  ak_ = solver_.analyse(mat_, ctrl_, info_);
  info_.analyse_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (!ak_ || info_.status != ErrorCode::Success) {
    ak_.reset();
    if (info_.status == ErrorCode::Success) info_.status = ErrorCode::NotInitialised;
    return info_.status;
  }
  return ErrorCode::Success;
}

IpmFactorStatus IpmSession::factor(std::span<const double> values,
                                   std::optional<int> expected_negatives) {
  solvable_ = false;
  aborted_ = false;
  pending_accept_ = false;
  last_spd_ok_ = false;
  if (mat_.n <= 0) {
    info_.status = ErrorCode::NotInitialised;
    return IpmFactorStatus::Error;
  }
  if (values.size() != mat_.values.size()) {
    info_.status = ErrorCode::IllegalValue;
    return IpmFactorStatus::Error;
  }
  std::copy(values.begin(), values.end(), mat_.values.begin());

  // This factor's own Control, kept for completing an abort in solve(): the
  // caller may replace ctrl_ (solve options) and drop its scale buffer.
  factor_ctrl_ = ctrl_;
  if (ctrl_.user_scale != nullptr) {
    factor_scale_.assign(ctrl_.user_scale, ctrl_.user_scale + mat_.n);
    factor_ctrl_.user_scale = factor_scale_.data();
  } else {
    factor_scale_.clear();
  }
  if (user_scale_once_) {
    ctrl_.user_scale = nullptr;
    user_scale_once_ = false;
  }
  // Feature 3: the static mode. Fallback static-pivots only a root front or
  // one over its delay budget. A raised pivot counts by its kept sign, or,
  // with IpmControl::static_as_zero, as a zero eigenvalue (the caller then
  // sees a singular matrix).
  if (ipm_.static_mode == StaticMode::Fallback) {
    factor_ctrl_.fallback_static_pivot_tol = 1e-8;
    factor_ctrl_.static_pivots_as_zero = ipm_.static_as_zero;
  }

  if (!ak_ && run_analysis() != ErrorCode::Success) return IpmFactorStatus::Error;
  prepare_analysis();

  // Early abort: armed with an expectation (explicit, else learned) on an
  // indefinite matrix. Negative pivots only accumulate, so exceeding the
  // expectation proves the inertia wrong.
  int limit = -1;
  const std::optional<int> expected = expected_negatives ? expected_negatives : expected_neg_;
  if (ipm_.abort_on_wrong_inertia && expected && *expected >= 0 &&
      factor_ctrl_.matrix_type != MatrixType::RealSymmetricPositiveDefinite)
    limit = *expected;

  ++stats_.factorizations;
  const IpmFactorStatus result = run_factor(limit);
  if (result == IpmFactorStatus::WrongInertia) {
    ++stats_.aborted;
    stats_.flops_avoided += info_.aborted_flops_skipped;
  }
  return result;
}

IpmFactorStatus IpmSession::run_factor(int max_negative_pivots) {
  factor_ctrl_.max_negative_pivots = max_negative_pivots;
  const FactorStatus fs = solver_.factor(*ak_, mat_, factor_ctrl_, info_, fk_, nws_);
  factor_ctrl_.max_negative_pivots = -1;
  info_.factor_status = fs;

  IpmFactorStatus result = IpmFactorStatus::Error;
  switch (fs) {
  case FactorStatus::Success:
  case FactorStatus::MaxPivotDelays:
    result = IpmFactorStatus::Ok;
    break;
  case FactorStatus::Singular:
    // Input, scaling or allocation failures also come back as Singular.
    if (info_.status == ErrorCode::SingularMatrix) result = IpmFactorStatus::Singular;
    break;
  case FactorStatus::InertiaExceeded:
    result = IpmFactorStatus::WrongInertia;
    break;
  case FactorStatus::NotPositiveDefinite:
    break;
  }
  solvable_ = result == IpmFactorStatus::Ok || result == IpmFactorStatus::Singular;
  aborted_ = result == IpmFactorStatus::WrongInertia;
  pending_accept_ = solvable_;
  last_spd_ok_ = factor_ctrl_.matrix_type == MatrixType::RealSymmetricPositiveDefinite &&
                 fs == FactorStatus::Success;

  // Probation: judged on the adopted analysis' first complete factorization
  // (an aborted one is judged when solve() completes it). Only a factor the
  // engine actually ran counts: a failure before it starts (e.g. the
  // matching scaling finds no matching: Singular with SingularMatrix) leaves
  // fk_ as it was, pointing at the previous analysis, with no factor values.
  const bool ran = fk_.analysis == ak_.get() && !fk_.factor_values.empty();
  if (solvable_ && ran) stats_.static_pivots += info_.static_pivots;
  if (probation_ && solvable_ && ran) {
    probation_ = false;
    if (info_.delayed_pivots > probation_delays_) {
      // Worse than the old analysis. fk_ points at the adopted analysis and
      // may still be solved with, so the switch back waits for factor().
      ++stats_.reanalyses_reverted;
      revert_pending_ = true;
      reanalyse_stopped_ = true;
    } else {
      drop_previous_analysis();
    }
  }
  return result;
}

void IpmSession::drop_previous_analysis() {
  // Never free an analysis the factor still points at.
  assert(fk_.analysis == nullptr || fk_.analysis != prev_ak_.get());
  if (fk_.analysis != nullptr && fk_.analysis == prev_ak_.get()) return;
  prev_ak_.reset();
  prev_info_ = Info{};
}

void IpmSession::consider_reanalysis() {
  if (!ipm_.reanalyse || reanalyse_stopped_ || probation_ || revert_pending_ || !ak_) return;
  if (factor_ctrl_.matrix_type == MatrixType::RealSymmetricPositiveDefinite) return;
  if (!(static_cast<double>(info_.delayed_pivots) > ipm_.reanalyse_delay_frac * static_cast<double>(mat_.n)))
    return;
  // The attempt would run in the next factorization.
  if (last_attempt_ >= 0 && stats_.factorizations + 1 - last_attempt_ < ipm_.reanalyse_min_gap) return;
  // fk_ is overwritten by the next factor(): read its order now.
  RealizedOrder ro = realized_pivot_order(*ak_, fk_);
  order_perm_ = std::move(ro.perm);
  order_pairs_ = std::move(ro.pairs);
  trigger_delays_ = info_.delayed_pivots;
  trigger_flops_ = info_.actual_flops;
  reanalyse_armed_ = true;
}

void IpmSession::prepare_analysis() {
  if (revert_pending_) {
    // fk_ is about to be overwritten; until then it must not point at the
    // analysis being freed.
    revert_pending_ = false;
    fk_.analysis = nullptr;
    ak_ = std::move(prev_ak_);
    info_ = prev_info_;
    prev_info_ = Info{};
    return;
  }
  if (!reanalyse_armed_) return;
  reanalyse_armed_ = false;
  last_attempt_ = stats_.factorizations + 1;
  ++stats_.reanalyses_tried;
  std::vector<Int> perm = std::move(order_perm_);
  std::vector<Int> pairs = std::move(order_pairs_);
  order_perm_.clear();
  order_pairs_.clear();
  if (perm.size() != static_cast<std::size_t>(mat_.n)) return;  // cannot re-analyse: rejected

  Control actrl = factor_ctrl_;
  actrl.ordering = OrderingMethod::UserSupplied;
  actrl.user_perm = perm.data();
  actrl.user_pair_first = pairs.empty() ? nullptr : pairs.data();
  actrl.user_num_pairs = static_cast<Int>(pairs.size() / 2);
  Info ainfo;
  const auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<AnalysisKeep> nak = solver_.analyse(mat_, actrl, ainfo);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  stats_.reanalyse_seconds += secs;
  if (!nak || ainfo.status != ErrorCode::Success) return;
  if (kernel_flops(*nak) > ipm_.reanalyse_flops_ratio * trigger_flops_) return;
  // A user-supplied order computes no matching scaling. Keep the current
  // analysis' (ScalingMethod::Auto applies it; Matching reuses it only for
  // the values it was computed from), so the scaling does not change.
  if (nak->matching_scale.empty() && !ak_->matching_scale.empty() &&
      nak->cleaned.values.size() == ak_->cleaned.values.size()) {
    nak->matching_scale = ak_->matching_scale;
    nak->cleaned.values = ak_->cleaned.values;
  }

  // Adopt. The old analysis stays alive for the probation (and fk_, the
  // accepted factor, still points at it until run_factor replaces it).
  ainfo.analyse_seconds = secs;
  assert(!prev_ak_);  // no probation is open here
  prev_ak_ = std::move(ak_);
  prev_info_ = info_;
  ak_ = std::move(nak);
  info_ = ainfo;
  ++stats_.reanalyses_adopted;
  probation_ = true;
  probation_delays_ = trigger_delays_;
}

int IpmSession::solve(double* b, int n, int nrhs, SolveJob job) {
  // Safety: the caller wants to solve with a factor that stopped early (a
  // wrong expectation). Factor the same values in full first.
  if (aborted_ && ak_) run_factor(-1);
  if (!solvable_) {
    solve_info_.status = ErrorCode::NotInitialised;
    return static_cast<int>(ErrorCode::NotInitialised);
  }
  const int rc = solver_.solve(fk_, ctrl_, solve_info_, b, n, nrhs, sws_, job);
  if (rc == 0 && pending_accept_) {
    ++stats_.accepted;
    pending_accept_ = false;
    if (ipm_.learn_inertia && !last_spd_ok_) expected_neg_ = info_.num_negative;
    consider_reanalysis();
  }
  return rc;
}

InertiaCounts IpmSession::inertia() const {
  if (last_spd_ok_) return InertiaCounts{static_cast<int>(mat_.n), 0, 0};
  return InertiaCounts{info_.num_positive, info_.num_negative, info_.num_zero};
}

void IpmSession::set_control(const Control& ctrl) {
  ctrl_ = ctrl;
  user_scale_once_ = false;
}

void IpmSession::set_user_scale(const double* scale) {
  ctrl_.user_scale = scale;
  user_scale_once_ = true;
}

} // namespace smf
