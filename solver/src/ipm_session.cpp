#include "smf/ipm_session.hpp"

#include <algorithm>
#include <chrono>

namespace smf {

ErrorCode IpmSession::analyse(const CscLower& pattern, std::span<const double> values,
                              const Control& ctrl, const IpmControl& ipm) {
  // A new pattern starts a new history.
  ak_.reset();
  fk_ = FactorKeep{};
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

  if (!ak_ && run_analysis() != ErrorCode::Success) return IpmFactorStatus::Error;

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
  return result;
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
