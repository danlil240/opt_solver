#include "smf/solve_backward.hpp"
#include "smf/solve_interleaved.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/determinism.hpp"
#include "smf/factor_indef.hpp"
#include "smf/ldlt_kernel.hpp"
#include "smf/scaling.hpp"
#include "smf/scaling_matching.hpp"
#include "smf/slack_elimination.hpp"
#include "smf/solve_diag.hpp"
#include "smf/solve_forward.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
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
// The fresh error the relative test compares with is at least this: below
// about 100 eps the probe measures rounding (one reuse of slightly
// different values can move it 10x), not a stale scale.
constexpr double kReuseOmegaFloor = 1e-14;

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
// The conditions of scale_on_delay_armed other than its ratio and pivot
// threshold.
bool scale_on_delay_applies(const AnalysisKeep &ak, const Control &ctrl) {
  return ctrl.scaling == ScalingMethod::Auto &&
         ctrl.matrix_type == MatrixType::RealSymmetricIndefinite && ak.n > 0 &&
         ctrl.static_pivot_tol == 0.0 && ctrl.fallback_static_pivot_tol == 0.0 &&
         ctrl.delay_budget < 0 && ctrl.max_negative_pivots < 0 &&
         !(ctrl.tree_parallel && ctrl.num_threads > 1 && !deterministic_mode(ctrl));
}

bool scale_on_delay_armed(const AnalysisKeep &ak, const Control &ctrl) {
  return ctrl.scale_on_delay_ratio >= 0.0 && ctrl.pivot_u >= kWeakPivotRatio &&
         scale_on_delay_applies(ak, ctrl);
}

// Control::static_fallback_after_cascade keeps a static-path factor only up
// to this pivot growth (FactorKeep::pivot_growth): past it eps * growth is
// above 1e-6, more than refinement reliably recovers. pde_control rec, the
// cascades it is kept on, measured 1.3e6 and 5.6e6.
constexpr double kStaticMaxGrowth = 1e10;

// The static path (Control::static_fallback_after_cascade, Pivoting::Static):
// the delay budget spent from the first front, so every front static-pivots
// (no delays; a column with no threshold pivot is taken as the best 1x1, |d|
// raised to kDelayBudgetStaticTol * max|A| if smaller) on the budget's fixed
// partition. A pivot raised from an exact zero still counts as a zero
// eigenvalue (hybrid accounting, as on fallback fronts). Never on the legacy
// tree_parallel engine, which ignores the delay budget (it would threshold-
// pivot and delay while static_path reports otherwise): the default
// subtree engine at the same num_threads.
Control static_path_control(const Control &ctrl) {
  Control c = ctrl;
  c.delay_budget = 0;
  c.static_zero_stays_zero = true;
  c.tree_parallel = false;
  return c;
}

// The skip-trial gate's safety net (Control::skip_trial_rescue_ratio): an
// analysis whose auto matching trial the gate skipped arms the rescue at
// any pivot threshold (the gate itself only fires below kWeakPivotRatio,
// where scale_on_delay_armed never holds), under the same other conditions.
bool skip_trial_rescue_armed(const AnalysisKeep &ak, const Control &ctrl) {
  return ak.matching_trial_skipped && ctrl.skip_trial_rescue_ratio >= 0.0 &&
         scale_on_delay_applies(ak, ctrl);
}

// The scale-on-delay rescue's matching (matching_scale_of) ends its searches
// once at most this share of the columns is free, and scales by that
// partial matching's duals. The searches that complete a matching are its
// dearest and change the rescued factor little: ncvxqp9 at u = 0.01, 25 ->
// 11.6 ms for the matching, 3588 -> 3989 delays; on PGLib OPF the full
// matchings took 70-370 ms per rescue. The analysis keeps an initial
// matching's scale up to the same share (stokes128, 1% free).
constexpr double kRescueMaxFree = 0.02;

// The matching scaling of values `cv` of ak's pattern (cleaned order) for
// the scale-on-delay rescue: the analysis' own matching scale, adopted or
// rejected, when they are the analysed values, else a matching's that may
// leave kRescueMaxFree of the columns free. Empty when there is none
// (structurally singular: an all-zero column, or more unmatched columns
// than that share). cv is borrowed (moved in and back), not copied.
std::vector<double> matching_scale_of(const AnalysisKeep &ak, std::vector<double> &cv) {
  if (cv.size() != ak.cleaned.row_idx.size()) return {};
  const std::vector<double> &analysed =
      !ak.matching_scale.empty() && !ak.matching_scale_partial ? ak.matching_scale
                                                               : ak.rejected_matching_scale;
  if (!analysed.empty() && cv == ak.cleaned.values) return analysed;
  CscLower C;
  C.n = ak.n;
  C.col_ptr = ak.cleaned.col_ptr;
  C.row_idx = ak.cleaned.row_idx;
  C.values = std::move(cv);
  MatchingResult mr;
  {
    SymmetricMatcher m(C);
    m.set_stop_free(static_cast<int>(kRescueMaxFree * static_cast<double>(ak.n)));
    m.advance();
    mr = m.current();
  }
  cv = std::move(C.values);
  if (mr.status != ErrorCode::Success || mr.scale.size() != static_cast<std::size_t>(ak.n)) return {};
  for (const double s : mr.scale)
    if (!(s > 0.0) || !std::isfinite(s)) return {};
  return std::move(mr.scale);
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

} // namespace

bool scale_on_delay_reuse_passes(double probe_omega, double fresh_omega) {
  const double base = fresh_omega >= 0.0 ? std::max(fresh_omega, kReuseOmegaFloor)
                                         : std::numeric_limits<double>::infinity();
  return probe_omega <= kReuseOmegaFactor * base && probe_omega <= kReuseOmegaMax;
}

double scale_on_delay_probe(const AnalysisKeep &ak, std::span<const double> cv,
                            const FactorKeep &fkeep, const Control &ctrl) {
  const std::size_t n = static_cast<std::size_t>(ak.n);
  if (cv.size() != ak.cleaned.row_idx.size()) return std::numeric_limits<double>::infinity();
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
    // Any non-finite row (an overflowed or NaN solution) fails the probe.
    if (!std::isfinite(e)) return std::numeric_limits<double>::infinity();
    omega = std::max(omega, e);
  }
  return omega;
}

namespace {

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
// unarmed while the values' zero pattern stays the same. A first try let
// through past Control::scale_on_delay_next_ratio leaves the next one to
// run under that ratio as its limit (Control::scale_on_delay_carry: rescaled
// without a try, as after a limit trip). A first try under the full limit may
// stop earlier, at the early ratio's crossing, when its inflation so far
// projects past Control::scale_on_delay_early_proj (EarlyWorkStop); that
// stop takes the limit trip's path. Only a factorization that ran to the
// end updates the FactorKeep.
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
  const bool next = ours && fkeep.delay_next;
  std::vector<double> kept;
  double kept_omega = std::numeric_limits<double>::infinity();
  std::uint64_t no_matching_signature = 0;
  // Control::static_fallback_after_cascade: an earlier factorization of
  // this analysis into this FactorKeep cascaded, so this one takes the
  // static path with no try.
  const bool fallback = ctrl.static_fallback_after_cascade;
  const bool go_static = ours && fallback && fkeep.delay_static;
  const bool no_matching = ours && !next && !go_static && fkeep.delay_scale.empty();
  const bool first_try = !ours || no_matching || next;
  if (ours) {
    kept = std::move(fkeep.delay_scale);
    kept_omega = fkeep.delay_omega;
    no_matching_signature = fkeep.delay_signature;
  }
  fkeep.delay_analysis = 0;
  fkeep.delay_next = false;
  fkeep.delay_static = false;
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
  const double free_work = delay_free_front_work(ak);
  const double limit = std::floor(ctrl.scale_on_delay_ratio * free_work);
  // Control::static_fallback_after_cascade: a static-path factorization of
  // these values, scaled as Auto resolves it (never with a rescue's matching
  // scale: on pde_control_120 rec at 8T the tripped factor's scale on the
  // later values took 627 perturbed pivots and an unrefined omega of 5e-2;
  // unscaled, none and 7e-11). It is kept only when it perturbed no pivot
  // and its growth stays within kStaticMaxGrowth: unscaled static pivoting
  // of a badly scaled KKT perturbs hundreds to tens of thousands of pivots
  // (PGLib OPF, COPS catmix / glider / pinene / rocket: singular or
  // inaccurate), which only the matching scaling avoids. A rejected one is
  // spent work (a stopped try) and the rescue follows; the sequence then
  // goes on from that rescue's kept scale, never static again.
  // Returns true when st is final: the static factor kept, or a hard error
  // (allocation, invalid values) that the rescue would only repeat.
  const auto try_static = [&](FactorStatus &st) {
    st = run(ak, static_path_control(ctrl), -1.0);
    if (!ran(st)) return true;
    if (st == FactorStatus::Success && info.static_pivots == 0 &&
        info.pivot_growth <= kStaticMaxGrowth) {
      fkeep.static_path = true;
      info.static_fallback = 1;
      fkeep.delay_analysis = ak.serial;
      fkeep.delay_static = true;
      return true;
    }
    extra_seconds += info.factor_seconds;
    info.delay_try_seconds += info.factor_seconds;
    info.delay_stops = 1;
    info.static_rejects = 1;
    // Never again for this analysis and FactorKeep, whatever follows (also
    // when the rescue finds no matching scaling and re-arms the tries).
    fkeep.static_rejected_analysis = ak.serial;
    return false;
  };
  if (go_static) {
    FactorStatus st;
    if (try_static(st)) return finish(st);
    goto rescale;
  }
  if (!first_try && reuse_on) {
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
        // A factor with zero pivots (Singular) has no solve to probe: it
        // stands on the fit test and the work limit. Otherwise the probe's
        // error must stay within kReuseOmegaFactor of the fresh one (no
        // fresh one recorded: only the cap applies) and under the cap.
        bool pass = st == FactorStatus::Singular;
        if (!pass) {
          const auto tp = now();
          const double omega = scale_on_delay_probe(ak, cv, fkeep, ctrl);
          extra_seconds += secs(tp, now());
          pass = scale_on_delay_reuse_passes(omega, kept_omega);
        }
        if (pass) {
          info.scaling_used = static_cast<int>(ScalingMethod::Matching);
          fkeep.delay_analysis = ak.serial;
          fkeep.delay_scale = std::move(kept);
          fkeep.delay_omega = kept_omega;
          return finish(st);
        }
      }
      // The limit tripped or the probe failed: this try is spent.
      extra_seconds += info.factor_seconds;
      info.delay_try_seconds += info.factor_seconds;
      info.delay_stops = 1;
    }
  } else if (first_try || !reuse_on) {
    if (first_try) {
      // Control::scale_on_delay_carry: after a costly let-through (and while
      // the next limit is on) no next try; the matching scaling of these
      // values follows directly, as after a limit trip.
      if (next && ctrl.scale_on_delay_carry && ctrl.scale_on_delay_next_ratio >= 0.0)
        goto rescale;
      // First try: as Auto resolves it, under the work limit (the tighter
      // one after a costly factorization, Control::scale_on_delay_next_ratio).
      const double try_limit =
          next && ctrl.scale_on_delay_next_ratio >= 0.0
              ? std::min(limit, std::floor(ctrl.scale_on_delay_next_ratio * free_work))
              : limit;
      // Control::scale_on_delay_early_ratio: a try whose limit is above
      // the early ratio may stop at its crossing (EarlyWorkStop), as a
      // limit trip. A next try (FactorKeep::delay_next) uses
      // Control::scale_on_delay_next_early_ratio instead (off by default).
      EarlyWorkStop early;
      if (!next && ctrl.scale_on_delay_early_ratio >= 0.0 &&
          try_limit > std::floor(ctrl.scale_on_delay_early_ratio * free_work))
        early = {free_work, ctrl.scale_on_delay_early_ratio, ctrl.scale_on_delay_early_proj};
      if (next && ctrl.scale_on_delay_next_early_ratio >= 0.0 &&
          try_limit > std::floor(ctrl.scale_on_delay_next_early_ratio * free_work))
        early = {free_work, ctrl.scale_on_delay_next_early_ratio,
                 ctrl.scale_on_delay_next_early_proj};
      const FactorStatus st = run(ak, ctrl, try_limit, early);
      if (st != FactorStatus::MaxPivotDelays) {
        // Let through, but costly: the next one runs under the tighter limit.
        if (ran(st) && ctrl.scale_on_delay_next_ratio >= 0.0 &&
            info.front_work > ctrl.scale_on_delay_next_ratio * free_work) {
          // A partial matching's scale (AnalysisKeep::matching_scale_partial)
          // is not the values' own matching scaling: the next one stays armed.
          bool own_scale = false;
          if (info.scaling_used == static_cast<int>(ScalingMethod::Matching) &&
              !ak.matching_scale_partial) {
            get_values();
            own_scale = cv == ak.cleaned.values;
          }
          if (!own_scale) {
            fkeep.delay_analysis = ak.serial;
            // Control::static_fallback_after_cascade: the later ones take
            // the static path instead of a next try or a carried rescale.
            const bool to_static = fallback && fkeep.static_rejected_analysis != ak.serial;
            (to_static ? fkeep.delay_static : fkeep.delay_next) = true;
          }
        }
        return finish(st);
      }
      extra_seconds += info.factor_seconds;
      info.delay_try_seconds += info.factor_seconds;
      info.delay_stops = 1;
      // A limit trip counts the front past the limit; within it, the early
      // crossing stopped the try.
      if (early.on() && info.front_work <= try_limit) info.delay_early_stops = 1;
      // Control::static_fallback_after_cascade: the factor that tripped is
      // redone on the static path, and so are the later ones, while it
      // holds (try_static). (Finishing it with the rescue instead cost
      // pde_control_120 rec at 8T 4 ms of matching and a 10 ms scaled
      // refactor against a 7 ms static one.)
      if (fallback && fkeep.static_rejected_analysis != ak.serial) {
        FactorStatus st;
        if (try_static(st)) return finish(st);
      }
    }
  }
rescale:
  // A new matching scaling of these values.
  get_values();
  const auto t0 = now();
  std::vector<double> scale = matching_scale_of(ak, cv);
  info.delay_matching_seconds = secs(t0, now());
  extra_seconds += info.delay_matching_seconds;
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
      // The fresh scale's probe error, the reuse baseline; none (-1) when
      // reuse is off or the factor has zero pivots (nothing to solve with).
      double omega = -1.0;
      if (reuse_on && st == FactorStatus::Success) {
        const auto tp = now();
        omega = scale_on_delay_probe(ak, cv, fkeep, ctrl);
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
  info.skip_trial_rescues = 0;
  // Control::pivoting = Static (the fast mode): every factorization on the
  // static path, no rescue. Unlike the cascade fallback's static path, a
  // pivot raised from an exact zero counts by the sign of its perturbed
  // value (+), as PARDISO counts it: a zero-diagonal KKT (PGLib, COPS,
  // stokes) raises one per zero diagonal block entry it meets first, and
  // static_zero_stays_zero would report every such factor Singular. Only the
  // inertia accounting differs; the factor's values are the same. The
  // caller's own static_zero_stays_zero is kept (false by default; IpmSession's
  // regularized mode sets it, and its accounting stays its own), and so are
  // fallback fronts' hybrid accounting (fallback_static_pivot_tol > 0).
  if (ctrl.pivoting == Pivoting::Static) {
    const AnalysisKeep &k = ak.matching_reanalysis ? *ak.matching_reanalysis : ak;
    Control sc = static_path_control(ctrl);
    sc.static_zero_stays_zero = ctrl.static_zero_stays_zero;
    const FactorStatus st = run(k, sc, -1.0);
    if (ak.matching_reanalysis) info.ordering_used = static_cast<int>(OrderingMethod::Matching);
    // No rescue state survives a static factorization.
    fkeep.delay_analysis = 0;
    fkeep.delay_next = false;
    fkeep.delay_static = false;
    fkeep.delay_scale.clear();
    if (st == FactorStatus::Success || st == FactorStatus::Singular) {
      fkeep.static_path = true;
      info.static_fallback = 1;
    }
    return st;
  }
  if (ak.matching_reanalysis) {
    const FactorStatus st = run(*ak.matching_reanalysis, ctrl, -1.0);
    info.ordering_used = static_cast<int>(OrderingMethod::Matching);
    return st;
  }
  const bool realized_pending = ctrl.matching_trigger == MatchingTrigger::RealizedDelays &&
                                ak.matching_trigger_eligible && !ak.matching_reanalysis_tried;
  if (!realized_pending && scale_on_delay_armed(ak, ctrl))
    return factor_scale_on_delay(ak, ctrl, info, fkeep, values, run);
  // The skip-trial gate's safety net: the rescue under its own work limit,
  // with no early stop and no tightened next limit (only a delay cascade
  // past that limit pays for a matching).
  if (!realized_pending && skip_trial_rescue_armed(ak, ctrl)) {
    Control c = ctrl;
    c.scale_on_delay_ratio = ctrl.skip_trial_rescue_ratio;
    c.scale_on_delay_early_ratio = -1.0;
    c.scale_on_delay_next_ratio = -1.0;
    // Not the recommended track: no static fallback.
    c.static_fallback_after_cascade = false;
    const FactorStatus st = factor_scale_on_delay(ak, c, info, fkeep, values, run);
    info.skip_trial_rescues = info.delay_rescales;
    return st;
  }
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

// Control::slack_elimination: factor through the analysis' slack plan
// (slack_elimination.hpp), the cleaned values `cv`. False when it does not
// run (not armed, the plan given up on, or a pair's pivot failing): the
// caller factors the full matrix as without the plan.
static bool factor_slack(const Solver &solver, const AnalysisKeep &ak, std::vector<double> cv,
                         const Control &ctrl, Info &info, FactorKeep &fkeep, NumericWorkspace &ws,
                         FactorStatus &st) {
  if (!slack_armed(ak, ctrl) || ak.slack_fallbacks_in_row >= kSlackMaxFallbacks) return false;
  const SlackPlan &plan = *ak.slack;
  if (cv.size() != ak.cleaned.row_idx.size()) return false;
  auto sf = std::make_shared<SlackFactor>();
  sf->plan = ak.slack;
  int pos = 0, neg = 0;
  const bool passed = slack_pair_pass(plan, cv.data(), ctrl, ws.slack_values, *sf, pos, neg);
  if (const char *tr = std::getenv("SMF_SLACK_TRACE"); tr != nullptr)
    std::fprintf(stderr, "[slack] pairs %zu fails %d max_growth %.3e max_slack %.3e%s\n", plan.pairs.size(),
                 sf->fails, sf->max_growth, sf->max_slack, passed ? "" : " -> full matrix");
  if (!passed) {
    ++ak.slack_fallbacks_in_row;
    info.slack_fallbacks = 1;
    return false;
  }
  ak.slack_fallbacks_in_row = 0;
  Control rc = ctrl;
  rc.slack_elimination = SlackElimination::Off;
  if (!ws.slack_ws) ws.slack_ws = std::make_shared<NumericWorkspace>();
  st = solver.factor(*plan.reduced, std::span<const double>(ws.slack_values), rc, info, fkeep, *ws.slack_ws);
  // The full matrix, for the solve's accuracy check (Control::slack_omega_target).
  if (ctrl.slack_omega_target > 0.0 && plan.clean_ptr.size() == static_cast<std::size_t>(plan.n_full) + 1) {
    sf->cv = std::move(cv);
    sf->ctrl = ctrl;
    sf->ctrl.slack_elimination = SlackElimination::Off;
  }
  fkeep.slack = std::move(sf);
  info.num_positive += pos;
  info.num_negative += neg;
  info.numerical_rank += pos + neg;
  info.slack_pairs = static_cast<int>(plan.pairs.size());
  return true;
}

FactorStatus Solver::factor(const AnalysisKeep &ak_in, const Control &ctrl,
                            Info &info, FactorKeep &fkeep) const {
  info.scaling_used = 0;
  info.delay_rescales = 0;
  info.skip_trial_rescues = 0;
  info.delay_early_stops = 0;
  info.delay_stops = 0;
  info.delay_try_seconds = 0.0;
  info.delay_matching_seconds = 0.0;
  info.static_fallback = 0;
  info.static_rejects = 0;
  info.pivot_growth = 0.0;
  FactorStatus st;
  fkeep.slack.reset();
  info.slack_pairs = 0;
  info.slack_fallbacks = 0;
  if (ak_in.slack) {
    NumericWorkspace ws;
    if (factor_slack(*this, ak_in, ak_in.cleaned.values, ctrl, info, fkeep, ws, st)) return st;
  }
  // A factorization off the slack plan takes the full matrix's own analysis.
  const AnalysisKeep &ak = slack_full_analysis(ak_in);
  if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
    st = factor_posdef(ak, ctrl, info, fkeep);
  } else {
    st = factor_with_matching_trigger(ak, ctrl, info, fkeep, [&] { return ak.cleaned.values; },
                                      [&](const AnalysisKeep &k, const Control &c, double limit,
                                          const EarlyWorkStop &early = {}) {
      return factor_indef(k, c, info, fkeep, limit, early);
    });
  }
  build_inverse_diag(fkeep, ctrl, info);
  return st;
}

FactorStatus Solver::factor(const AnalysisKeep &ak_in, const CscLower &updated,
                            const Control &ctrl, Info &info,
                            FactorKeep &fkeep, NumericWorkspace &ws) const {
  info.scaling_used = 0;
  info.delay_rescales = 0;
  info.skip_trial_rescues = 0;
  info.delay_early_stops = 0;
  info.delay_stops = 0;
  info.delay_try_seconds = 0.0;
  info.delay_matching_seconds = 0.0;
  info.static_fallback = 0;
  info.static_rejects = 0;
  info.pivot_growth = 0.0;
  FactorStatus st;
  fkeep.slack.reset();
  info.slack_pairs = 0;
  info.slack_fallbacks = 0;
  if (ak_in.slack) {
    FactorValues fv;
    if (factor_slack(*this, ak_in, resolve_factor_values(ak_in, updated, fv) ? cleaned_values(ak_in, fv) : std::vector<double>{},
                     ctrl, info, fkeep, ws, st))
      return st;
  }
  // A factorization off the slack plan takes the full matrix's own analysis.
  const AnalysisKeep &ak = slack_full_analysis(ak_in);
  if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
    st = factor_posdef(ak, updated, ctrl, info, fkeep, ws);
  } else {
    const auto values = [&] {
      FactorValues fv;
      return resolve_factor_values(ak, updated, fv) ? cleaned_values(ak, fv)
                                                    : std::vector<double>{};
    };
    st = factor_with_matching_trigger(ak, ctrl, info, fkeep, values,
                                      [&](const AnalysisKeep &k, const Control &c, double limit,
                                          const EarlyWorkStop &early = {}) {
      return factor_indef(k, updated, c, info, fkeep, ws, limit, early);
    });
  }
  build_inverse_diag(fkeep, ctrl, info);
  return st;
}

FactorStatus Solver::factor(const AnalysisKeep &ak_in, std::span<const double> values,
                            const Control &ctrl, Info &info,
                            FactorKeep &fkeep, NumericWorkspace &ws) const {
  const FactorValues fv{values.data(), values.size(), true};
  info.scaling_used = 0;
  info.delay_rescales = 0;
  info.skip_trial_rescues = 0;
  info.delay_early_stops = 0;
  info.delay_stops = 0;
  info.delay_try_seconds = 0.0;
  info.delay_matching_seconds = 0.0;
  info.static_fallback = 0;
  info.static_rejects = 0;
  info.pivot_growth = 0.0;
  FactorStatus st;
  fkeep.slack.reset();
  info.slack_pairs = 0;
  info.slack_fallbacks = 0;
  if (ak_in.slack && factor_slack(*this, ak_in, cleaned_values(ak_in, fv), ctrl, info, fkeep, ws, st)) return st;
  // A factorization off the slack plan takes the full matrix's own analysis.
  const AnalysisKeep &ak = slack_full_analysis(ak_in);
  if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
    st = factor_posdef(ak, fv, ctrl, info, fkeep, ws);
  } else {
    st = factor_with_matching_trigger(ak, ctrl, info, fkeep,
                                      [&] { return cleaned_values(ak, fv); },
                                      [&](const AnalysisKeep &k, const Control &c, double limit,
                                          const EarlyWorkStop &early = {}) {
      return factor_indef(k, fv, c, info, fkeep, ws, limit, early);
    });
  }
  build_inverse_diag(fkeep, ctrl, info);
  return st;
}

namespace {

// The solve with column stride ldb >= n. Only the full non-banded solve
// reads and writes b with stride ldb (its gather and scatter); the other
// paths need ldb == n (Solver::solve packs B for them).
int solve_core(const FactorKeep &fkeep, const Control &ctrl, Info &info,
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
  // Stopped at a work limit (factor_indef's work_limit): no factor values.
  if (fkeep.status == FactorStatus::MaxPivotDelays) {
    info.status = ErrorCode::WorkLimitExceeded;
    return static_cast<int>(ErrorCode::WorkLimitExceeded);
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

  // The solve's BLAS runs num_threads threads, whatever the caller's BLAS
  // runs (the banded solve's DPBTRS included): num_threads = 1 is one BLAS
  // thread, num_threads = N lets a serial-tree solve of many right-hand
  // sides (GEMM / TRSM per front) use N. Deterministic mode pins it to one,
  // so the nrhs > 1 blocked solve's results do not depend on the BLAS
  // thread count (as the factor's kernels run serial BLAS in this mode).
  // num_threads <= 0: the caller's setting (CallBlasGuard).
  const bool det = deterministic_mode(ctrl);
  if (det) enable_blas_reproducibility();
  const CallBlasGuard call_blas(ctrl);

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
    // Serial solves of up to kInterleavedMaxRhs right-hand sides, and any
    // serial solve of a factor of small fronts only (one right-hand side
    // there runs in place, without gathering each front): interleaved, each
    // factor entry loaded once for all of them.
    if (ctrl.small_solve_kernels && threads == 1 && interleaved_solve_supported(fkeep) &&
        (fkeep.solve_small_fronts ? true : (nrhs > 1 && nrhs <= kInterleavedMaxRhs))) {
      solve_full_interleaved(fkeep, b, n, nrhs, ldb, ws);
      break;
    }
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

// A solve through a slack factor (slack_elimination.hpp): b (n x nrhs,
// column stride ldb, original order) is gathered into the reduced right-hand
// sides with the pairs' forward elimination, solved with the reduced factor
// (fkeep's own arrays), and the pairs substituted back. The factorization's
// first pivots are the pairs: A = L1 diag(P, A_r) L1^T with L1 = [I 0; B P^-1 I],
// so every SolveJob splits into the pairs' part and the reduced job.
int slack_substitute(const FactorKeep &fkeep, const Control &ctrl, Info &info, double *b, int n, int nrhs,
                     int ldb, SolveWorkspace &ws, SolveJob job) {
  const SlackFactor &sf = *fkeep.slack;
  const SlackPlan &plan = *sf.plan;
  const int nr = static_cast<int>(plan.nr);
  if (n != plan.n_full || nrhs <= 0 || b == nullptr || ldb < n) {
    info.status = ErrorCode::IllegalValue;
    return static_cast<int>(ErrorCode::IllegalValue);
  }
  const bool forward = job == SolveJob::Full || job == SolveJob::Forward;
  ws.slack_r.resize(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nrhs));
  const std::size_t np = plan.pairs.size();
  for (int j = 0; j < nrhs; ++j) {
    const double *bj = b + static_cast<std::ptrdiff_t>(j) * ldb;
    double *r = ws.slack_r.data() + static_cast<std::ptrdiff_t>(j) * nr;
    for (int i = 0; i < nr; ++i) r[i] = bj[plan.red_to_full[static_cast<std::size_t>(i)]];
    if (!forward) continue;
    // r_r -= B P^-1 b_p: only the multiplier row of B is nonzero.
    for (std::size_t q = 0; q < np; ++q) {
      const SlackPair &pr = plan.pairs[q];
      const double zy = sf.pinv[3 * q + 1] * bj[pr.s] + sf.pinv[3 * q + 2] * bj[pr.y];
      if (zy == 0.0) continue;
      for (Int t = 0; t < pr.nb; ++t)
        r[plan.b_row[static_cast<std::size_t>(pr.b0 + t)]] -= sf.bval[static_cast<std::size_t>(pr.b0 + t)] * zy;
    }
  }
  const int rc = solve_core(fkeep, ctrl, info, ws.slack_r.data(), nr, nrhs, nr, ws, job);
  if (rc != 0) return rc;
  for (int j = 0; j < nrhs; ++j) {
    double *bj = b + static_cast<std::ptrdiff_t>(j) * ldb;
    const double *r = ws.slack_r.data() + static_cast<std::ptrdiff_t>(j) * nr;
    for (int i = 0; i < nr; ++i) bj[plan.red_to_full[static_cast<std::size_t>(i)]] = r[i];
    if (job == SolveJob::Forward) continue;  // L1^-1 leaves the pairs' rows as they are
    for (std::size_t q = 0; q < np; ++q) {
      const SlackPair &pr = plan.pairs[q];
      const double iss = sf.pinv[3 * q], isy = sf.pinv[3 * q + 1], iyy = sf.pinv[3 * q + 2];
      double bs = bj[pr.s], by = bj[pr.y];
      if (job != SolveJob::Backward) {  // P^-1 on the pairs' rows
        const double ts = iss * bs + isy * by;
        by = isy * bs + iyy * by;
        bs = ts;
      }
      if (job != SolveJob::DiagOnly) {  // L1^-T: x_p -= P^-1 B^T x_r
        double t = 0.0;
        for (Int k = 0; k < pr.nb; ++k)
          t += sf.bval[static_cast<std::size_t>(pr.b0 + k)] * r[plan.b_row[static_cast<std::size_t>(pr.b0 + k)]];
        bs -= isy * t;
        by -= iyy * t;
      }
      bj[pr.s] = bs;
      bj[pr.y] = by;
    }
  }
  return 0;
}

// The componentwise backward error of x for right-hand side b on the slack
// factor's full matrix: max_i |b - A x|_i / max((|A| |x| + |b|)_i, tau_i), with
// tau = n eps max_i (|A| |x| + |b|)_i for rows at roundoff level.
// r becomes b - A x; d is scratch.
double slack_omega(const SlackFactor &sf, const double *b, const double *x, double *r, double *d, int n) {
  const SlackPlan &plan = *sf.plan;
  for (int i = 0; i < n; ++i) {
    r[i] = b[i];
    d[i] = std::abs(b[i]);
  }
  const LongInt *cp = plan.clean_ptr.data();
  const Int *ri_ = plan.clean_row.data();
  const double *cv = sf.cv.data();
  for (Int j = 0; j < plan.n_full; ++j) {
    const double xj = x[j], axj = std::abs(xj);
    double rj = 0.0, dj = 0.0;
    for (LongInt p = cp[j]; p < cp[j + 1]; ++p) {
      const Int i = ri_[p];
      const double v = cv[p], av = std::abs(v);
      r[i] -= v * xj;
      d[i] += av * axj;
      if (i != j) {
        rj -= v * x[i];
        dj += av * std::abs(x[i]);
      }
    }
    r[j] += rj;
    d[j] += dj;
  }
  double dmax = 0.0;
  for (int i = 0; i < n; ++i) dmax = std::max(dmax, d[i]);
  const double tau = static_cast<double>(n) * std::numeric_limits<double>::epsilon() * dmax;
  // The largest |r_i| / den_i, as the ratios themselves (a cross-multiplied
  // comparison overflows or underflows with the right-hand side's scale).
  double w = 0.0;
  for (int i = 0; i < n; ++i) {
    const double den = std::max(d[i], tau), ai = std::abs(r[i]);
    if (!(ai <= std::numeric_limits<double>::max())) return std::numeric_limits<double>::infinity();
    if (den > 0.0) {
      w = std::max(w, ai / den);
    } else if (ai > 0.0) {
      return 1.0;
    }
  }
  return w;
}

// A solve through a slack factor. A full solve is checked on the full matrix
// (Control::slack_omega_target): when a column's componentwise backward error
// is above it, up to kSlackRefineSteps refinement steps with the reduced
// factor follow; if one still is, the full matrix is factored with its own
// analysis (once per factor) and the right-hand sides solved with that, as
// are the factor's later solves.
int slack_solve(const FactorKeep &fkeep, const Control &ctrl, Info &info, double *b, int n, int nrhs,
                int ldb, SolveWorkspace &ws, SolveJob job) {
  const SlackFactor &sf = *fkeep.slack;
  info.slack_refine_steps = 0;
  info.slack_solve_fallbacks = 0;
  if (n != sf.plan->n_full || nrhs <= 0 || b == nullptr || ldb < n) {
    info.status = ErrorCode::IllegalValue;
    return static_cast<int>(ErrorCode::IllegalValue);
  }
  std::shared_ptr<FactorKeep> full;
  {
    std::lock_guard<std::mutex> lock(sf.full_mu);
    full = sf.full;
  }
  if (full) {
    info.slack_solve_fallbacks = 1;
    return solve_core(*full, ctrl, info, b, n, nrhs, ldb, ws, job);
  }
  if (job != SolveJob::Full || !(ctrl.slack_omega_target > 0.0) || sf.cv.empty() || n != sf.plan->n_full)
    return slack_substitute(fkeep, ctrl, info, b, n, nrhs, ldb, ws, job);
  const std::size_t sn = static_cast<std::size_t>(n);
  ws.slack_b.resize(sn * static_cast<std::size_t>(nrhs));
  for (int j = 0; j < nrhs; ++j)
    std::copy_n(b + static_cast<std::ptrdiff_t>(j) * ldb, n, ws.slack_b.data() + static_cast<std::ptrdiff_t>(j) * n);
  int rc = slack_substitute(fkeep, ctrl, info, b, n, nrhs, ldb, ws, job);
  if (rc != 0) return rc;
  ws.slack_res.resize(sn);
  ws.slack_den.resize(sn);
  bool accurate = true;
  double worst = 0.0;
  int steps = 0;
  for (int j = 0; j < nrhs; ++j) {
    double *x = b + static_cast<std::ptrdiff_t>(j) * ldb;
    const double *bj = ws.slack_b.data() + static_cast<std::ptrdiff_t>(j) * n;
    double w = slack_omega(sf, bj, x, ws.slack_res.data(), ws.slack_den.data(), n);
    for (int it = 0; it < kSlackRefineSteps && w > ctrl.slack_omega_target; ++it) {
      rc = slack_substitute(fkeep, ctrl, info, ws.slack_res.data(), n, 1, n, ws, SolveJob::Full);
      if (rc != 0) return rc;
      for (int i = 0; i < n; ++i) x[i] += ws.slack_res[static_cast<std::size_t>(i)];
      ++steps;
      w = slack_omega(sf, bj, x, ws.slack_res.data(), ws.slack_den.data(), n);
    }
    worst = std::max(worst, w);
    if (w > ctrl.slack_omega_target) accurate = false;
  }
  info.slack_refine_steps = steps;
  if (accurate) {
    if (steps > 0 && std::getenv("SMF_SLACK_TRACE") != nullptr)
      std::fprintf(stderr, "[slack] solve refined in %d steps (omega %.3e)\n", steps, worst);
    return 0;
  }
  {
    std::lock_guard<std::mutex> lock(sf.full_mu);
    if (!sf.full) {
      if (const AnalysisKeep *fa = slack_plan_full_analysis(*sf.plan)) {
        auto fk = std::make_shared<FactorKeep>();
        NumericWorkspace nws;
        Info fi;
        // The cleaned values in the cleaned pattern (the full analysis'
        // own, as the plan's: the same input), which factor() takes as is.
        const SlackPlan &plan = *sf.plan;
        CscLower C;
        C.n = plan.n_full;
        C.col_ptr.assign(plan.clean_ptr.begin(), plan.clean_ptr.end());
        C.row_idx = plan.clean_row;
        C.values = sf.cv;
        if (Solver().factor(*fa, C, sf.ctrl, fi, *fk, nws) == FactorStatus::Success)
          sf.full = std::move(fk);
      }
    }
    full = sf.full;
  }
  if (const char *tr = std::getenv("SMF_SLACK_TRACE"); tr != nullptr)
    std::fprintf(stderr, "[slack] solve omega %.3e after %d refinement steps -> %s\n", worst, steps,
                 full ? "full matrix factor" : "kept (full factor failed)");
  if (!full) return 0;
  for (int j = 0; j < nrhs; ++j)
    std::copy_n(ws.slack_b.data() + static_cast<std::ptrdiff_t>(j) * n, n, b + static_cast<std::ptrdiff_t>(j) * ldb);
  info.slack_solve_fallbacks = 1;
  return solve_core(*full, ctrl, info, b, n, nrhs, ldb, ws, job);
}

int solve_impl(const FactorKeep &fkeep, const Control &ctrl, Info &info, double *b, int n, int nrhs,
               int ldb, SolveWorkspace &ws, SolveJob job) {
  if (fkeep.slack) return slack_solve(fkeep, ctrl, info, b, n, nrhs, ldb, ws, job);
  return solve_core(fkeep, ctrl, info, b, n, nrhs, ldb, ws, job);
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
  const int n = fkeep.slack ? static_cast<int>(fkeep.slack->plan->n_full)
                : fkeep.use_banded_spd ? fkeep.band_n : static_cast<int>(fkeep.perm.size());
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
