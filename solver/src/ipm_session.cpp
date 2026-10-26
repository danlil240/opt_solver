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
  user_scale_once_ = false;
  solvable_ = false;
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
                                   std::optional<int> /*expected_negatives*/) {
  solvable_ = false;
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

  const auto clear_user_scale = [this] {
    if (user_scale_once_) {
      ctrl_.user_scale = nullptr;
      user_scale_once_ = false;
    }
  };

  if (!ak_ && run_analysis() != ErrorCode::Success) {
    clear_user_scale();
    return IpmFactorStatus::Error;
  }

  const FactorStatus fs = solver_.factor(*ak_, mat_, ctrl_, info_, fk_, nws_);
  info_.factor_status = fs;
  clear_user_scale();
  ++stats_.factorizations;

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
    ++stats_.aborted;
    result = IpmFactorStatus::WrongInertia;
    break;
  case FactorStatus::NotPositiveDefinite:
    break;
  }
  solvable_ = result == IpmFactorStatus::Ok || result == IpmFactorStatus::Singular;
  pending_accept_ = solvable_;
  last_spd_ok_ = ctrl_.matrix_type == MatrixType::RealSymmetricPositiveDefinite &&
                 fs == FactorStatus::Success;
  return result;
}

int IpmSession::solve(double* b, int n, int nrhs, SolveJob job) {
  if (!solvable_) {
    solve_info_.status = ErrorCode::NotInitialised;
    return static_cast<int>(ErrorCode::NotInitialised);
  }
  const int rc = solver_.solve(fk_, ctrl_, solve_info_, b, n, nrhs, sws_, job);
  if (rc == 0 && pending_accept_) {
    ++stats_.accepted;
    pending_accept_ = false;
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
