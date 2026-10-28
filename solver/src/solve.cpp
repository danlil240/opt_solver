#include "smf/solve_backward.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/determinism.hpp"
#include "smf/factor_indef.hpp"
#include "smf/ldlt_kernel.hpp"
#include "smf/scaling.hpp"
#include "smf/solve_diag.hpp"
#include "smf/solve_forward.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace smf {

namespace {

// MatchingTrigger::RealizedDelays fires when a factorization delays more
// than this share of n pivots (IPOPT's MA97 od_hd_reuse threshold).
constexpr double kRealizedDelayShare = 0.05;

// Control::scale_on_delay_reuse_tol: a factor with a kept scale stands only
// if its accuracy probe's backward error is at most kReuseOmegaFactor times
// the one measured when the scale was fresh, and at most kReuseOmegaMax. On
// the tuning split kept factors that stayed accurate probed at most 5x their
// fresh error, the ones that lost accuracy 39x - 3800x; fresh errors reach
// 1.1e-10 on large PGLib cases (1e-10 outright would reject every reuse
// there), 1e-9 keeps 10x of headroom to the 1e-8 accuracy gate.
constexpr double kReuseOmegaFactor = 10.0;
constexpr double kReuseOmegaMax = 1e-9;

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

// Control::scale_on_delay_ratio applies: ScalingMethod::Auto, plain
// threshold pivoting at u >= kWeakPivotRatio (never IPOPT's 1e-8), no
// static pivoting or delay budget, no max_negative_pivots (its stop would
// race the work limit's on the subtree engine, so the outcome kept in the
// FactorKeep would depend on thread timing), and not the legacy
// tree_parallel engine (it cannot stop early).
bool scale_on_delay_armed(const AnalysisKeep &ak, const Control &ctrl) {
  return ctrl.scale_on_delay_ratio >= 0.0 && ctrl.scaling == ScalingMethod::Auto &&
         ctrl.matrix_type == MatrixType::RealSymmetricIndefinite && ak.n > 0 &&
         ctrl.pivot_u >= kWeakPivotRatio && ctrl.static_pivot_tol == 0.0 &&
         ctrl.fallback_static_pivot_tol == 0.0 && ctrl.delay_budget < 0 &&
         ctrl.max_negative_pivots < 0 &&
         !(ctrl.tree_parallel && ctrl.num_threads > 1 && !deterministic_mode(ctrl));
}

// The matching scaling (ScalingMethod::Matching's) of values `cv` of ak's
// pattern (cleaned order): the analysis' own matching scale, adopted or
// rejected, when they are the analysed values. Empty when there is none
// (e.g. structurally singular).
std::vector<double> matching_scale_of(const AnalysisKeep &ak, std::vector<double> cv,
                                      int print_level) {
  if (cv.size() != ak.cleaned.row_idx.size()) return {};
  const std::vector<double> &analysed =
      !ak.matching_scale.empty() ? ak.matching_scale : ak.rejected_matching_scale;
  if (!analysed.empty() && cv == ak.cleaned.values) return analysed;
  CscLower C;
  C.n = ak.n;
  C.col_ptr = ak.cleaned.col_ptr;
  C.row_idx = ak.cleaned.row_idx;
  C.values = std::move(cv);
  std::vector<double> scale;
  if (compute_scaling_for_method(C, ScalingMethod::Matching, scale, print_level) !=
          ErrorCode::Success ||
      scale.size() != static_cast<std::size_t>(ak.n))
    return {};
  return scale;
}

// How far a kept scale s is from a matching scaling of values cv (cleaned
// order): a matching scaling of cv has every row's largest scaled entry
// m_i = max_j |s_i a_ij s_j| equal to 1 (all entries at most 1, the matched
// one exactly 1). Returns the 90th percentile over rows of |log10 m_i|
// (rows without a nonzero left out); infinity when nothing can be measured.
double scale_staleness(const AnalysisKeep &ak, const std::vector<double> &cv,
                       const std::vector<double> &s) {
  const std::size_t n = static_cast<std::size_t>(ak.n);
  std::vector<double> m(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (Int p = ak.cleaned.col_ptr[j]; p < ak.cleaned.col_ptr[j + 1]; ++p) {
      const std::size_t i = static_cast<std::size_t>(ak.cleaned.row_idx[static_cast<std::size_t>(p)]);
      const double a = std::abs(cv[static_cast<std::size_t>(p)]) * s[i] * s[j];
      m[i] = std::max(m[i], a);
      m[j] = std::max(m[j], a);
    }
  }
  std::size_t k = 0;
  for (double v : m)
    if (v > 0.0) m[k++] = std::abs(std::log10(v));
  if (k == 0) return std::numeric_limits<double>::infinity();
  const auto q = m.begin() + static_cast<std::ptrdiff_t>((k - 1) * 9 / 10);
  std::nth_element(m.begin(), q, m.begin() + static_cast<std::ptrdiff_t>(k));
  return *q;
}

// Accuracy probe of a factor of the values cv (cleaned order) of ak's
// pattern: solve A x = b for a fixed right-hand side with one serial solve
// (the same result at every num_threads in deterministic mode) and return
// the componentwise backward error max_i |b - A x|_i / (|A| |x| + |b|)_i,
// the accuracy gate's omega.
// Infinity when the solve fails or the error is not finite.
double probe_backward_error(const AnalysisKeep &ak, const std::vector<double> &cv,
                            const FactorKeep &fkeep, const Control &ctrl) {
  const std::size_t n = static_cast<std::size_t>(ak.n);
  const auto &cp = ak.cleaned.col_ptr;
  const auto &ri = ak.cleaned.row_idx;
  std::vector<double> b(n, 0.0);
  // A fixed pseudo-random right-hand side (xorshift64, entries of either
  // sign with magnitudes in [0.5, 1.5)): b = A 1 was tried and measured
  // 1e-8 - 1e-6 already with a fresh scale on PGLib (no room to see a stale
  // one), b = 1 is the benchmark's own right-hand side.
  std::uint64_t z = 0x9e3779b97f4a7c15ull;
  for (std::size_t i = 0; i < n; ++i) {
    z ^= z << 13;
    z ^= z >> 7;
    z ^= z << 17;
    const double u = static_cast<double>(z >> 11) * 0x1.0p-53;
    b[i] = (u < 0.5 ? -1.0 : 1.0) * (0.5 + u);
  }
  std::vector<double> x = b;
  Control sc = ctrl;
  sc.num_threads = 1;
  Info si;
  Solver solver;
  if (solver.solve(fkeep, sc, si, x.data(), static_cast<int>(n), 1) != 0)
    return std::numeric_limits<double>::infinity();
  std::vector<double> r = b, d(n, 0.0);
  for (std::size_t j = 0; j < n; ++j)
    for (Int p = cp[j]; p < cp[j + 1]; ++p) {
      const std::size_t i = static_cast<std::size_t>(ri[static_cast<std::size_t>(p)]);
      const double v = cv[static_cast<std::size_t>(p)];
      r[i] -= v * x[j];
      d[i] += std::abs(v * x[j]);
      if (i != j) {
        r[j] -= v * x[i];
        d[j] += std::abs(v * x[i]);
      }
    }
  double omega = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double den = d[i] + std::abs(b[i]);
    const double e = den > 0.0 ? std::abs(r[i]) / den
                               : (r[i] == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    if (!(e <= omega)) omega = e;  // NaN propagates as "not finite"
  }
  return std::isfinite(omega) ? omega : std::numeric_limits<double>::infinity();
}

// Positions (and count) of the exactly zero or non-finite values of cv:
// whether a matching scaling exists depends on nothing else.
std::uint64_t zero_signature(const std::vector<double> &cv) {
  std::uint64_t h = 1469598103934665603ull;  // FNV-1a over the positions
  std::uint64_t count = 0;
  for (std::size_t k = 0; k < cv.size(); ++k)
    if (cv[k] == 0.0 || !std::isfinite(cv[k])) {
      ++count;
      for (int byte = 0; byte < 8; ++byte) {
        h ^= (static_cast<std::uint64_t>(k) >> (8 * byte)) & 0xffu;
        h *= 1099511628211ull;
      }
    }
  return h ^ (count * 0x9e3779b97f4a7c15ull);
}

// Scale on delay (Control::scale_on_delay_ratio): `run(keep, ctrl, limit)`
// factors with a front-work limit. A first factorization runs as Auto
// resolves it (the analysis' matching scale with the matching ordering,
// else none) and stops once its front work exceeds the ratio times the
// delay-free one; it is then redone with the matching scaling of its
// values, which the FactorKeep keeps for this analysis together with the
// backward error of a probe solve. A later factorization of that analysis
// into that FactorKeep starts from the kept scale when it still fits the
// new values (Control::scale_on_delay_reuse_tol), under the same limit; it
// keeps that factor only if its probe's backward error is within
// kReuseOmegaFactor of the kept one and at most kReuseOmegaMax. Otherwise
// (no fit, the limit trips, or the probe fails) it computes a new matching
// scaling. Without a matching scaling (structurally singular values) the
// factorization runs as Auto resolves it, to the end, and later ones run
// unarmed while the values' zero pattern stays the same. Only a
// factorization that ran to the end updates the FactorKeep.
template <class Values, class Run>
FactorStatus factor_scale_on_delay(const AnalysisKeep &ak, const Control &ctrl, Info &info,
                                   FactorKeep &fkeep, Values &&values, Run &&run) {
  const auto ran = [&](FactorStatus st) {
    return st == FactorStatus::Success ||
           (st == FactorStatus::Singular && info.status == ErrorCode::SingularMatrix);
  };
  const auto now = [] { return std::chrono::steady_clock::now(); };
  const auto secs = [](auto t0, auto t1) { return std::chrono::duration<double>(t1 - t0).count(); };
  const bool reuse_on = ctrl.scale_on_delay_reuse_tol >= 0.0;
  const bool ours = fkeep.delay_analysis == ak.serial;
  std::vector<double> kept;
  double kept_omega = std::numeric_limits<double>::infinity();
  std::uint64_t no_matching_signature = 0;
  const bool no_matching = ours && fkeep.delay_scale.empty();
  if (ours) {
    kept = std::move(fkeep.delay_scale);
    kept_omega = fkeep.delay_omega;
    no_matching_signature = fkeep.delay_signature;
  }
  fkeep.delay_analysis = 0;
  fkeep.delay_scale.clear();
  // Work outside factor_indef (value copies, the fit test, probes,
  // matchings, stopped tries): added to factor_seconds on every return.
  double extra_seconds = 0.0;
  const auto finish = [&](FactorStatus st) {
    info.factor_seconds += extra_seconds;
    return st;
  };
  std::vector<double> cv;
  bool have_cv = false;
  const auto get_values = [&] {
    if (!have_cv) {
      const auto t0 = now();
      cv = values();
      have_cv = true;
      extra_seconds += secs(t0, now());
    }
  };
  if (no_matching) {
    // These values' zero pattern had no matching scaling: while it is the
    // same there is still none, so no stopped try.
    get_values();
    if (zero_signature(cv) == no_matching_signature) {
      const FactorStatus st = run(ak, ctrl, -1.0);
      if (ran(st)) {
        fkeep.delay_analysis = ak.serial;
        fkeep.delay_signature = no_matching_signature;
      }
      return finish(st);
    }
  }
  const double limit = std::floor(ctrl.scale_on_delay_ratio * delay_free_front_work(ak));
  if (ours && !no_matching && reuse_on) {
    // Start from the kept scale while it still fits the new values.
    get_values();
    const auto t0 = now();
    const bool fits = cv.size() == ak.cleaned.row_idx.size() &&
                      scale_staleness(ak, cv, kept) <= ctrl.scale_on_delay_reuse_tol;
    extra_seconds += secs(t0, now());
    if (fits) {
      Control c = ctrl;
      c.scaling = ScalingMethod::UserSupplied;
      c.user_scale = kept.data();
      const FactorStatus st = run(ak, c, limit);
      if (st != FactorStatus::MaxPivotDelays && !ran(st)) return finish(st);
      if (st != FactorStatus::MaxPivotDelays) {
        const auto tp = now();
        const double omega = probe_backward_error(ak, cv, fkeep, ctrl);
        extra_seconds += secs(tp, now());
        if (omega <= kReuseOmegaFactor * kept_omega && omega <= kReuseOmegaMax) {
          info.scaling_used = static_cast<int>(ScalingMethod::Matching);
          fkeep.delay_analysis = ak.serial;
          fkeep.delay_scale = std::move(kept);
          fkeep.delay_omega = kept_omega;
          return finish(st);
        }
      }
      // The limit tripped or the probe failed: this try is spent.
      extra_seconds += info.factor_seconds;
    }
  } else if (!ours || no_matching || !reuse_on) {
    if (!ours || no_matching) {
      // First try: as Auto resolves it, under the work limit.
      const FactorStatus st = run(ak, ctrl, limit);
      if (st != FactorStatus::MaxPivotDelays) return finish(st);
      extra_seconds += info.factor_seconds;
    }
  }
  // A new matching scaling of these values.
  get_values();
  const auto t0 = now();
  std::vector<double> scale = matching_scale_of(ak, cv, ctrl.print_level);
  extra_seconds += secs(t0, now());
  FactorStatus st;
  if (scale.empty()) {
    st = run(ak, ctrl, -1.0);
    if (ran(st)) {  // no matching scaling: unarmed while the zero pattern holds
      fkeep.delay_analysis = ak.serial;
      fkeep.delay_signature = zero_signature(cv);
    }
  } else {
    Control c = ctrl;
    c.scaling = ScalingMethod::UserSupplied;
    c.user_scale = scale.data();
    st = run(ak, c, -1.0);
    info.delay_rescales = 1;
    if (ran(st)) {
      info.scaling_used = static_cast<int>(ScalingMethod::Matching);
      double omega = std::numeric_limits<double>::infinity();
      if (reuse_on) {
        const auto tp = now();
        omega = probe_backward_error(ak, cv, fkeep, ctrl);
        extra_seconds += secs(tp, now());
      }
      fkeep.delay_analysis = ak.serial;
      fkeep.delay_scale = std::move(scale);
      fkeep.delay_omega = omega;
    }
  }
  return finish(st);
}

// Indefinite factor() with the realized-delay matching trigger and the
// scale-on-delay trigger: `run(keep, ctrl, work_limit)` factors with a
// given analysis. A keep that already holds a matching re-analysis factors
// with it; otherwise, under RealizedDelays, a factor that delays more than
// kRealizedDelayShare * n pivots re-analyses once with the matching ordering
// and factors again. When the RealizedDelays trigger is not pending,
// Control::scale_on_delay_ratio may apply (factor_scale_on_delay). `values`
// (called only when a trigger fires) gives the factored values in cleaned
// order. The normal path adds no work.
template <class Values, class Run>
FactorStatus factor_with_matching_trigger(const AnalysisKeep &ak, const Control &ctrl,
                                          Info &info, FactorKeep &fkeep, Values &&values,
                                          Run &&run) {
  info.matching_reanalyses = 0;
  if (ak.matching_reanalysis) {
    const FactorStatus st = run(*ak.matching_reanalysis, ctrl, -1.0);
    info.ordering_used = static_cast<int>(OrderingMethod::Matching);
    return st;
  }
  const bool realized_pending = ctrl.matching_trigger == MatchingTrigger::RealizedDelays &&
                                ak.matching_trigger_eligible && !ak.matching_reanalysis_tried;
  if (!realized_pending && scale_on_delay_armed(ak, ctrl))
    return factor_scale_on_delay(ak, ctrl, info, fkeep, values, run);
  // delayed_pivots is reset at the start of a factorization: an input error
  // returns before that (status != Success), so only a factorization that
  // ran to the end (Success, or Singular reported as SingularMatrix; an
  // input or allocation error reports its own ErrorCode) may fire it.
  info.delayed_pivots = 0;
  info.aptp_failed_columns = 0;
  const FactorStatus st = run(ak, ctrl, -1.0);
  const bool ran = st == FactorStatus::Success ||
                   (st == FactorStatus::Singular && info.status == ErrorCode::SingularMatrix);
  if (!realized_pending || !ran ||
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
  const FactorStatus st2 = run(*ak.matching_reanalysis, ctrl, -1.0);
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
  info.scaling_used = 0;
  info.delay_rescales = 0;
  FactorStatus st;
  if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
    st = factor_posdef(ak, ctrl, info, fkeep);
  } else {
    st = factor_with_matching_trigger(ak, ctrl, info, fkeep, [&] { return ak.cleaned.values; },
                                      [&](const AnalysisKeep &k, const Control &c, double limit) {
      return factor_indef(k, c, info, fkeep, limit);
    });
  }
  build_inverse_diag(fkeep, ctrl, info);
  return st;
}

FactorStatus Solver::factor(const AnalysisKeep &ak, const CscLower &updated,
                            const Control &ctrl, Info &info,
                            FactorKeep &fkeep, NumericWorkspace &ws) const {
  info.scaling_used = 0;
  info.delay_rescales = 0;
  FactorStatus st;
  if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
    st = factor_posdef(ak, updated, ctrl, info, fkeep, ws);
  } else {
    const auto values = [&] {
      FactorValues fv;
      return resolve_factor_values(ak, updated, fv) ? cleaned_values(ak, fv)
                                                    : std::vector<double>{};
    };
    st = factor_with_matching_trigger(ak, ctrl, info, fkeep, values,
                                      [&](const AnalysisKeep &k, const Control &c, double limit) {
      return factor_indef(k, updated, c, info, fkeep, ws, limit);
    });
  }
  build_inverse_diag(fkeep, ctrl, info);
  return st;
}

FactorStatus Solver::factor(const AnalysisKeep &ak, std::span<const double> values,
                            const Control &ctrl, Info &info,
                            FactorKeep &fkeep, NumericWorkspace &ws) const {
  const FactorValues fv{values.data(), values.size(), true};
  info.scaling_used = 0;
  info.delay_rescales = 0;
  FactorStatus st;
  if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
    st = factor_posdef(ak, fv, ctrl, info, fkeep, ws);
  } else {
    st = factor_with_matching_trigger(ak, ctrl, info, fkeep,
                                      [&] { return cleaned_values(ak, fv); },
                                      [&](const AnalysisKeep &k, const Control &c, double limit) {
      return factor_indef(k, fv, c, info, fkeep, ws, limit);
    });
  }
  build_inverse_diag(fkeep, ctrl, info);
  return st;
}

namespace {

// The solve with column stride ldb >= n. Only the full non-banded solve
// reads and writes b with stride ldb (its gather and scatter); the other
// paths need ldb == n (Solver::solve packs B for them).
int solve_impl(const FactorKeep &fkeep, const Control &ctrl, Info &info,
               double *b, int n, int nrhs, int ldb, SolveWorkspace &ws,
               SolveJob job) {
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
  // Deterministic mode: one thread, the factor's steps in order (postorder).
  const bool det = deterministic_mode(ctrl);
  if (det) enable_blas_reproducibility();
  // Deterministic mode pins the solve's BLAS to one thread, whatever the
  // caller's BLAS runs: the nrhs > 1 blocked solve calls GEMM / TRSM per
  // front, and their results must not depend on the BLAS thread count (as
  // the factor's kernels run serial BLAS in this mode). Outside it the
  // caller's BLAS threads stay, so a serial-tree solve of many right-hand
  // sides can still use threaded GEMM on its large fronts.
  std::optional<BlasThreadGuard> det_blas;
  if (det) det_blas.emplace(1);
  const int threads =
      (ctrl.num_threads > 1 && !det &&
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
      const double *src = b + static_cast<std::ptrdiff_t>(r) * ldb;
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
      double *dst = b + static_cast<std::ptrdiff_t>(r) * ldb;
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

} // namespace

int Solver::solve(const FactorKeep &fkeep, const Control &ctrl, Info &info,
                  double *b, int n, int nrhs, SolveWorkspace &ws,
                  SolveJob job) const {
  return solve_impl(fkeep, ctrl, info, b, n, nrhs, n, ws, job);
}

int Solver::solve(const FactorKeep &fkeep, const Control &ctrl, Info &info,
                  std::span<double> B, int nrhs, int ldb, SolveWorkspace &ws,
                  SolveJob job) const {
  const int n = fkeep.use_banded_spd ? fkeep.band_n : static_cast<int>(fkeep.perm.size());
  if (n <= 0 || nrhs <= 0 || ldb < n ||
      B.size() < static_cast<std::size_t>(ldb) * static_cast<std::size_t>(nrhs - 1) +
                     static_cast<std::size_t>(n)) {
    info.status = ErrorCode::IllegalValue;
    return static_cast<int>(ErrorCode::IllegalValue);
  }
  if (ldb == n || (job == SolveJob::Full && !fkeep.use_banded_spd))
    return solve_impl(fkeep, ctrl, info, B.data(), n, nrhs, ldb, ws, job);
  // Partial jobs and the banded path take packed columns.
  std::vector<double> packed(static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs));
  for (int r = 0; r < nrhs; ++r)
    std::copy_n(B.data() + static_cast<std::ptrdiff_t>(r) * ldb, n,
                packed.data() + static_cast<std::ptrdiff_t>(r) * n);
  const int rc = solve_impl(fkeep, ctrl, info, packed.data(), n, nrhs, n, ws, job);
  if (rc == 0)
    for (int r = 0; r < nrhs; ++r)
      std::copy_n(packed.data() + static_cast<std::ptrdiff_t>(r) * n, n,
                  B.data() + static_cast<std::ptrdiff_t>(r) * ldb);
  return rc;
}

int Solver::solve(const FactorKeep &fkeep, const Control &ctrl, Info &info,
                  double *b, int n, int nrhs, SolveJob job) const {
  SolveWorkspace ws;
  return solve(fkeep, ctrl, info, b, n, nrhs, ws, job);
}

} // namespace smf
