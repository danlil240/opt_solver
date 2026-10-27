#include "smf/solve_backward.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/scaling.hpp"
#include "smf/solve_diag.hpp"
#include "smf/solve_forward.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"
#include <chrono>
#include <memory>
#include <vector>

namespace smf {

namespace {

// MatchingTrigger::RealizedDelays fires when a factorization delays more
// than this share of n pivots (IPOPT's MA97 od_hd_reuse threshold).
constexpr double kRealizedDelayShare = 0.05;

// `values` in the cleaned entry order of ak.
std::vector<double> cleaned_values(const AnalysisKeep &ak, const FactorValues &values) {
  if (!values.original_layout || ak.orig_to_clean_idx.empty())
    return std::vector<double>(values.data, values.data + values.size);
  if (values.size != ak.orig_to_clean_idx.size()) return {};
  std::vector<double> cv(ak.cleaned.row_idx.size(), 0.0);
  for (std::size_t k = 0; k < values.size; ++k) {
    const Int c = ak.orig_to_clean_idx[k];
    if (c >= 0) cv[static_cast<std::size_t>(c)] += values.data[k];
  }
  return cv;
}

// ak's pattern re-analysed with the matching ordering (its matching taken
// on `cv`, cleaned order), wired to accept the same value layouts as ak:
// the original input pattern and the cleaned one. Null when the matching
// ordering cannot be used (structurally singular).
std::shared_ptr<AnalysisKeep> matching_reanalysis(const AnalysisKeep &ak, std::vector<double> cv,
                                                  const Control &ctrl) {
  CscLower C;
  C.n = ak.n;
  C.col_ptr = ak.cleaned.col_ptr;
  C.row_idx = ak.cleaned.row_idx;
  C.values = std::move(cv);
  Control c2 = ctrl;
  c2.ordering = OrderingMethod::Matching;
  c2.check_input = false;  // the cleaned pattern is clean
  Solver s;
  Info i2;
  std::unique_ptr<AnalysisKeep> re = s.analyse(C, c2, i2);
  if (!re || i2.status != ErrorCode::Success ||
      i2.ordering_used != static_cast<int>(OrderingMethod::Matching))
    return nullptr;
  // Original-layout values reach the new permuted pattern through ak's
  // cleaning map, then the new clean-to-permuted map.
  const std::vector<Int> &c2p = re->orig_to_perm_idx;  // empty: identity
  std::vector<Int> direct(ak.orig_to_clean_idx.size(), -1);
  for (std::size_t k = 0; k < direct.size(); ++k) {
    const Int c = ak.orig_to_clean_idx[k];
    if (c >= 0) direct[k] = c2p.empty() ? c : c2p[static_cast<std::size_t>(c)];
  }
  re->orig_to_perm_direct = std::move(direct);
  re->orig_col_ptr = ak.orig_col_ptr;
  re->orig_row_idx = ak.orig_row_idx;
  re->orig_to_clean_idx = ak.orig_to_clean_idx;
  re->input_clean = ak.input_clean;
  re->cleaned.values = ak.cleaned.values;  // what factor(ak, ctrl, ...) factors
  return std::shared_ptr<AnalysisKeep>(std::move(re));
}

// Indefinite factor() with the realized-delay matching trigger: `run`
// factors with a given analysis. A keep that already holds a matching
// re-analysis factors with it; otherwise, under RealizedDelays, a factor
// that delays more than kRealizedDelayShare * n pivots re-analyses once with
// the matching ordering and factors again. `values` (called only then)
// gives the factored values for that re-analysis. The normal path adds no
// work.
template <class Values, class Run>
FactorStatus factor_with_matching_trigger(const AnalysisKeep &ak, const Control &ctrl,
                                          Info &info, Values &&values, Run &&run) {
  info.matching_reanalyses = 0;
  if (ak.matching_reanalysis) {
    const FactorStatus st = run(*ak.matching_reanalysis);
    info.ordering_used = static_cast<int>(OrderingMethod::Matching);
    return st;
  }
  // delayed_pivots is reset at the start of a factorization: an input error
  // returns before that (status != Success), so only a factorization that
  // ran to the end (Success, or Singular reported as SingularMatrix; an
  // input or allocation error reports its own ErrorCode) may fire it.
  info.delayed_pivots = 0;
  const FactorStatus st = run(ak);
  const bool ran = st == FactorStatus::Success ||
                   (st == FactorStatus::Singular && info.status == ErrorCode::SingularMatrix);
  if (ctrl.matching_trigger != MatchingTrigger::RealizedDelays || !ak.matching_trigger_eligible ||
      ak.matching_reanalysis_tried || !ran ||
      static_cast<double>(info.delayed_pivots) <= kRealizedDelayShare * static_cast<double>(ak.n))
    return st;
  ak.matching_reanalysis_tried = true;
  const double first_seconds = info.factor_seconds;
  const auto t0 = std::chrono::steady_clock::now();
  std::shared_ptr<AnalysisKeep> re = matching_reanalysis(ak, values(), ctrl);
  const double analyse_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (!re) return st;
  ak.matching_reanalysis = std::move(re);
  const FactorStatus st2 = run(*ak.matching_reanalysis);
  // factor_seconds covers the whole call: both factorizations and the
  // re-analysis between them.
  info.factor_seconds += first_seconds + analyse_seconds;
  info.matching_reanalyses = 1;
  info.ordering_used = static_cast<int>(OrderingMethod::Matching);
  return st2;
}

} // namespace

FactorStatus Solver::factor(const AnalysisKeep &ak, const Control &ctrl,
                            Info &info, FactorKeep &fkeep) const {
  if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
    return factor_posdef(ak, ctrl, info, fkeep);
  }
  return factor_with_matching_trigger(ak, ctrl, info, [&] { return ak.cleaned.values; },
                                      [&](const AnalysisKeep &k) {
    return factor_indef(k, ctrl, info, fkeep);
  });
}

FactorStatus Solver::factor(const AnalysisKeep &ak, const CscLower &updated,
                            const Control &ctrl, Info &info,
                            FactorKeep &fkeep, NumericWorkspace &ws) const {
  if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
    return factor_posdef(ak, updated, ctrl, info, fkeep, ws);
  }
  const auto values = [&] {
    FactorValues fv;
    return resolve_factor_values(ak, updated, fv) ? cleaned_values(ak, fv) : std::vector<double>{};
  };
  return factor_with_matching_trigger(ak, ctrl, info, values, [&](const AnalysisKeep &k) {
    return factor_indef(k, updated, ctrl, info, fkeep, ws);
  });
}

FactorStatus Solver::factor(const AnalysisKeep &ak, std::span<const double> values,
                            const Control &ctrl, Info &info,
                            FactorKeep &fkeep, NumericWorkspace &ws) const {
  const FactorValues fv{values.data(), values.size(), true};
  if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
    return factor_posdef(ak, fv, ctrl, info, fkeep, ws);
  }
  return factor_with_matching_trigger(ak, ctrl, info, [&] { return cleaned_values(ak, fv); },
                                      [&](const AnalysisKeep &k) {
    return factor_indef(k, fv, ctrl, info, fkeep, ws);
  });
}

int Solver::solve(const FactorKeep &fkeep, const Control &ctrl, Info &info,
                  double *b, int n, int nrhs, SolveWorkspace &ws,
                  SolveJob job) const {
  auto t0 = std::chrono::steady_clock::now();

  const int expected_n =
      (fkeep.n > 0) ? static_cast<int>(fkeep.n)
                    : (fkeep.use_banded_spd ? fkeep.band_n
                                            : static_cast<int>(fkeep.perm.size()));

  if (n <= 0 || nrhs <= 0 || b == nullptr ||
      (expected_n > 0 && n != expected_n) ||
      static_cast<int>(fkeep.perm.size()) != n ||
      static_cast<int>(fkeep.iperm.size()) != n) {
    info.status = ErrorCode::IllegalValue;
    return static_cast<int>(ErrorCode::IllegalValue);
  }

  switch (job) {
  case SolveJob::Full:
  case SolveJob::Forward:
  case SolveJob::DiagOnly:
  case SolveJob::Backward:
  case SolveJob::DiagBack:
    break;
  default:
    info.status = ErrorCode::IllegalValue;
    return static_cast<int>(ErrorCode::IllegalValue);
  }

  // An aborted factorization is incomplete: never solve with it.
  if (fkeep.status == FactorStatus::InertiaExceeded) {
    info.status = ErrorCode::InertiaExceeded;
    return static_cast<int>(ErrorCode::InertiaExceeded);
  }
  if (fkeep.status != FactorStatus::Success && !ctrl.continue_on_singular) {
    info.status = (fkeep.status == FactorStatus::NotPositiveDefinite)
                      ? ErrorCode::NotPositiveDefinite
                      : ErrorCode::SingularMatrix;
    return static_cast<int>(info.status);
  }

  ws.ensure_capacity(n, 0, 0, nrhs);

  const bool has_scale =
      (!fkeep.scale.empty() &&
       static_cast<int>(fkeep.scale.size()) == n);

  if (fkeep.use_banded_spd) {
    if (job != SolveJob::Full) {
      info.status = ErrorCode::FeatureNotAvailable;
      return static_cast<int>(ErrorCode::FeatureNotAvailable);
    }
    if (n != fkeep.band_n) {
      info.status = ErrorCode::IllegalValue;
      return static_cast<int>(ErrorCode::IllegalValue);
    }

    if (has_scale) {
      apply_scale_to_rhs(b, n, nrhs, fkeep.scale);
    }

    for (int rhs = 0; rhs < nrhs; ++rhs) {
      const double *src = b + static_cast<std::ptrdiff_t>(rhs) * n;
      double *dst = ws.rhs_perm.data() + static_cast<std::ptrdiff_t>(rhs) * n;
      for (int j = 0; j < n; ++j) {
        dst[static_cast<std::size_t>(
            fkeep.iperm[static_cast<std::size_t>(j)])] = src[j];
      }
    }

    const int rc = smf_dpbtrs_lower(ws.rhs_perm.data(), n, fkeep.band_kd, nrhs,
                                    fkeep.band_factor.data(),
                                    fkeep.band_kd + 1, n);
    if (rc != 0) {
      info.status = ErrorCode::SingularMatrix;
      return rc;
    }

    for (int rhs = 0; rhs < nrhs; ++rhs) {
      const double *src =
          ws.rhs_perm.data() + static_cast<std::ptrdiff_t>(rhs) * n;
      double *dst = b + static_cast<std::ptrdiff_t>(rhs) * n;
      for (int new_i = 0; new_i < n; ++new_i) {
        dst[static_cast<std::size_t>(
            fkeep.perm[static_cast<std::size_t>(new_i)])] =
            src[new_i];
      }
    }

    if (has_scale) {
      apply_scale_to_solution(b, n, nrhs, fkeep.scale);
    }

    auto t1 = std::chrono::steady_clock::now();
    info.solve_seconds += std::chrono::duration<double>(t1 - t0).count();
    info.status = ErrorCode::Success;
    return 0;
  }

  // Subtree-parallel triangular solves once the factor is big enough to
  // repay waking the pool (needs a tree-parallel factorization's layout).
  const int threads =
      (ctrl.num_threads > 1 &&
       static_cast<int64_t>(fkeep.factor_values.size()) >= ctrl.solve_parallel_min_entries)
          ? ctrl.num_threads
          : 1;

  // D then L^T: with D inline the backward pass applies it front by front.
  const auto diag_back = [&](double *y, bool permute) {
    const bool fused = backward_applies_diag(fkeep);
    if (!fused)
      for (int j = 0; j < nrhs; ++j)
        solve_diag(fkeep, y + j * n, n);
    solve_backward(fkeep, y, n, nrhs, ws, threads, fused, permute);
  };

  switch (job) {
  case SolveJob::Full: {
    // One gather (P^T S b) into the work vector and one scatter (S P y) at
    // the end, instead of a scaling pass, a permute-and-copy pass in each
    // triangular solve and a scaling pass again.
    double *w = ws.rhs_perm.data();
    const int *perm = fkeep.perm.data();
    const double *s = has_scale ? fkeep.scale.data() : nullptr;
    for (int r = 0; r < nrhs; ++r) {
      const double *src = b + static_cast<std::ptrdiff_t>(r) * n;
      double *dst = w + static_cast<std::ptrdiff_t>(r) * n;
      if (s)
        for (int i = 0; i < n; ++i)
          dst[i] = src[perm[i]] * s[perm[i]];
      else
        for (int i = 0; i < n; ++i)
          dst[i] = src[perm[i]];
    }
    solve_forward(fkeep, w, n, nrhs, ws, threads, /*permute=*/false);
    diag_back(w, /*permute=*/false);
    for (int r = 0; r < nrhs; ++r) {
      const double *src = w + static_cast<std::ptrdiff_t>(r) * n;
      double *dst = b + static_cast<std::ptrdiff_t>(r) * n;
      if (s)
        for (int i = 0; i < n; ++i)
          dst[perm[i]] = src[i] * s[perm[i]];
      else
        for (int i = 0; i < n; ++i)
          dst[perm[i]] = src[i];
    }
    break;
  }
  case SolveJob::Forward:
    if (has_scale)
      apply_scale_to_rhs(b, n, nrhs, fkeep.scale);
    solve_forward(fkeep, b, n, nrhs, ws, threads);
    break;
  case SolveJob::DiagOnly:
    for (int j = 0; j < nrhs; ++j)
      solve_diag(fkeep, b + j * n, n);
    break;
  case SolveJob::Backward:
    solve_backward(fkeep, b, n, nrhs, ws, threads);
    if (has_scale)
      apply_scale_to_solution(b, n, nrhs, fkeep.scale);
    break;
  case SolveJob::DiagBack:
    diag_back(b, /*permute=*/true);
    if (has_scale)
      apply_scale_to_solution(b, n, nrhs, fkeep.scale);
    break;
  }

  auto t1 = std::chrono::steady_clock::now();
  info.solve_seconds += std::chrono::duration<double>(t1 - t0).count();
  info.status = ErrorCode::Success;
  return 0;
}

int Solver::solve(const FactorKeep &fkeep, const Control &ctrl, Info &info,
                  double *b, int n, int nrhs, SolveJob job) const {
  SolveWorkspace ws;
  return solve(fkeep, ctrl, info, b, n, nrhs, ws, job);
}

} // namespace smf
