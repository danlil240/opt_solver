#include "smf/solve_diag.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/determinism.hpp"
#include "smf/pivoting.hpp"

#include <algorithm>
#include <mutex>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace smf {

namespace {

// kappa_1(L11) = ||L11||_1 ||L11^-1||_1 <= kInvDiagMaxCond. Multiplying by a
// computed inverse loses accuracy as kappa grows, where the triangular solve
// stays backward stable: an ill-conditioned block keeps its TRSV. Column
// sums: contiguous, no scratch. A NaN or infinite column sum (a non-finite
// entry in L11 or its inverse) rejects the block: std::max would drop a NaN.
bool inverse_well_conditioned(const double *L, std::ptrdiff_t f, const double *X, int p,
                              bool unit) {
  double nl = 0.0, nx = 0.0;
  for (int j = 0; j < p; ++j) {
    const double *lc = L + j * f;
    const double *xc = X + static_cast<std::ptrdiff_t>(j) * p;
    double sl = unit ? 1.0 : std::abs(lc[j]);
    double sx = std::abs(xc[j]);
    for (int i = j + 1; i < p; ++i) {
      sl += std::abs(lc[i]);
      sx += std::abs(xc[i]);
    }
    if (!std::isfinite(sl) || !std::isfinite(sx))
      return false;
    nl = std::max(nl, sl);
    nx = std::max(nx, sx);
  }
  return nl * nx <= kInvDiagMaxCond;
}

// FactorKeep::dinv from the D blocks in the fronts' diagonal slots, with
// apply_inline_d's semantics: a nonzero (c, c + 1) slot starts a 2x2 block;
// a zero or non-finite 1x1 pivot and a singular 2x2 block give 0. The 2x2
// inverse is solve_2x2_scaled's scaled Cramer rule with the right-hand
// side left out: [c, -b; -b, a] / det_s / s.
void build_inline_dinv(FactorKeep &fkeep, bool usable, const RangeRunner &run = {}) {
  if (!usable || fkeep.is_posdef || !fkeep.inline_d || fkeep.solve_steps.empty() ||
      fkeep.use_banded_spd || fkeep.factor_values.empty()) {
    fkeep.dinv.clear();
    fkeep.dinv_kind.clear();
    fkeep.dinv_offset.clear();
    return;
  }
  const std::size_t ns = fkeep.solve_steps.size();
  const auto nvals = static_cast<LongInt>(fkeep.factor_values.size());
  LongInt total = 0;
  for (const SolveStep &st : fkeep.solve_steps) {
    if (st.p < 0 || st.f < st.p || st.factor_offset < 0 ||
        (st.p > 0 && st.factor_offset + static_cast<LongInt>(st.p) * st.f > nvals)) {
      fkeep.dinv.clear();
      fkeep.dinv_kind.clear();
      fkeep.dinv_offset.clear();
      return;
    }
    total += st.p;
  }
  // resize() keeps the capacity: same structure, no allocation.
  fkeep.dinv.resize(3 * static_cast<std::size_t>(total));
  fkeep.dinv_kind.resize(static_cast<std::size_t>(total));
  fkeep.dinv_offset.resize(ns);
  LongInt off = 0;
  for (std::size_t k = 0; k < ns; ++k) {
    fkeep.dinv_offset[k] = off;
    off += fkeep.solve_steps[k].p;
  }
  const auto fill = [&fkeep](std::size_t k0, std::size_t k1) {
    for (std::size_t k = k0; k < k1; ++k) {
      const SolveStep &st = fkeep.solve_steps[k];
      const LongInt off = fkeep.dinv_offset[k];
      const int p = st.p;
      const std::ptrdiff_t f = st.f;
      const double *L = fkeep.factor_values.data() + st.factor_offset;
      double *d = fkeep.dinv.data() + 3 * off;
      std::int8_t *kind = fkeep.dinv_kind.data() + off;
      for (int c = 0; c < p;) {
        const double d00 = L[c * f + c];
        if (c + 1 < p) {
          const double d10 = L[(c + 1) * f + c];
          if (d10 != 0.0) {
            const double d11 = L[(c + 1) * f + c + 1];
            double sc = 0.0;
            const double det_s = scaled_det_2x2(d00, d10, d11, sc);
            double *e = d + 3 * c;
            if (sc == 0.0 || !std::isfinite(det_s) || det_s == 0.0) {
              e[0] = e[1] = e[2] = 0.0;
            } else {
              e[0] = (d11 / sc) / det_s / sc;
              e[1] = -(d10 / sc) / det_s / sc;
              e[2] = (d00 / sc) / det_s / sc;
            }
            e[3] = e[4] = e[5] = 0.0;
            kind[c] = 2;
            kind[c + 1] = 0;
            c += 2;
            continue;
          }
        }
        double *e = d + 3 * c;
        e[0] = (d00 == 0.0 || !is_finite_fast(d00)) ? 0.0 : 1.0 / d00;
        e[1] = e[2] = 0.0;
        kind[c] = 1;
        ++c;
      }
    }
  };
  if (run) run(ns, fill);
  else fill(0, ns);
}

} // namespace

void finish_solve_layout(FactorKeep &fkeep, bool small_solve_kernels, const RangeRunner &run) {
  build_inline_dinv(fkeep, small_solve_kernels, run);
  // No step reaches kInvDiagMinPivots: none can have a cached L11^-1 either.
  fkeep.solve_small_fronts =
      small_solve_kernels && !fkeep.solve_steps.empty() &&
      std::all_of(fkeep.solve_steps.begin(), fkeep.solve_steps.end(),
                  [](const SolveStep &st) { return st.p < kInvDiagMinPivots; });
}

void build_inverse_diag(FactorKeep &fkeep, const Control &ctrl, Info &info) {
  info.inverse_diag_fronts = 0;
  info.inverse_diag_seconds = 0.0;
  fkeep.inv_diag_offset.clear();
  // Only a factorization that ran to the end: Success, or Singular
  // reported as SingularMatrix under continue_on_singular. A hard error
  // (input, mapping, scaling, allocation) also reports Singular but clears
  // factor_values and may leave stale or partial solve steps behind.
  const bool usable =
      (fkeep.status == FactorStatus::Success && info.status == ErrorCode::Success) ||
      (fkeep.status == FactorStatus::Singular && ctrl.continue_on_singular &&
       info.status == ErrorCode::SingularMatrix);
  // finish_solve_layout built D^-1 with the solve steps; a factor the
  // solve refuses keeps neither.
  if (!usable) {
    build_inline_dinv(fkeep, false);
    fkeep.solve_small_fronts = false;
  }
  if (!ctrl.cache_inverse_diag || fkeep.solve_steps.empty() || fkeep.use_banded_spd ||
      !usable || fkeep.factor_values.empty()) {
    fkeep.inv_diag.clear();
    return;
  }
  // Every step's front must lie inside factor_values.
  const auto nvals = static_cast<LongInt>(fkeep.factor_values.size());
  for (const SolveStep &st : fkeep.solve_steps)
    if (st.p < 0 || st.f < st.p || st.factor_offset < 0 ||
        (st.p > 0 && st.factor_offset + static_cast<LongInt>(st.p) * st.f > nvals)) {
      fkeep.inv_diag.clear();
      return;
    }
  const auto t0 = std::chrono::steady_clock::now();
  const std::size_t ns = fkeep.solve_steps.size();
  LongInt total = 0;
  int fronts = 0;
  for (const SolveStep &st : fkeep.solve_steps)
    if (st.p >= kInvDiagMinPivots) {
      total += static_cast<LongInt>(st.p) * st.p;
      ++fronts;
    }
  if (fronts == 0) {
    fkeep.inv_diag.clear();
    return;
  }
  // resize() keeps the capacity: same structure, no allocation.
  fkeep.inv_diag_offset.resize(ns);
  fkeep.inv_diag.resize(static_cast<std::size_t>(total));
  const bool unit = !fkeep.is_posdef;
  // num_threads BLAS threads (deterministic mode one; <= 0 the caller's).
  const CallBlasGuard blas(ctrl);
  LongInt off = 0;
  for (std::size_t k = 0; k < ns; ++k) {
    const SolveStep &st = fkeep.solve_steps[k];
    const int p = st.p;
    if (p < kInvDiagMinPivots) {
      fkeep.inv_diag_offset[k] = -1;
      continue;
    }
    const std::ptrdiff_t f = st.f;
    const double *L = fkeep.factor_values.data() + st.factor_offset;
    double *X = fkeep.inv_diag.data() + off;
    // X = L11 as the solves read it: the strict lower triangle, the
    // diagonal (ones when unit; D sits there with D inline) and zeros above.
    for (int j = 0; j < p; ++j) {
      double *xc = X + static_cast<std::ptrdiff_t>(j) * p;
      const double *lc = L + j * f;
      for (int i = 0; i < j; ++i) xc[i] = 0.0;
      xc[j] = unit ? 1.0 : lc[j];
      for (int i = j + 1; i < p; ++i) xc[i] = lc[i];
    }
    if (smf_dtrtri_lower(X, p, p, unit) != 0 || !inverse_well_conditioned(L, f, X, p, unit)) {
      fkeep.inv_diag_offset[k] = -1;  // keep the TRSV for this front
      --fronts;
    } else {
      fkeep.inv_diag_offset[k] = off;
    }
    off += static_cast<LongInt>(p) * p;
  }
  info.inverse_diag_fronts = fronts;
  info.inverse_diag_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  info.factor_seconds += info.inverse_diag_seconds;
}

const std::vector<DiagSolveEntry> &FactorKeep::diag() const {
  if (!diag_merge.pending.load(std::memory_order_acquire)) return diag_entries;
  std::lock_guard<std::mutex> lk(*diag_merge.mu);
  if (diag_merge.pending.load(std::memory_order_relaxed)) {
    std::size_t total = 0;
    for (const auto &c : diag_merge.chunks) total += c.e - c.b;
    diag_entries.clear();
    diag_entries.reserve(total);
    for (const auto &c : diag_merge.chunks) {
      const auto &d = diag_merge.parts[static_cast<std::size_t>(c.part)];
      diag_entries.insert(diag_entries.end(), d.begin() + static_cast<std::ptrdiff_t>(c.b),
                          d.begin() + static_cast<std::ptrdiff_t>(c.e));
    }
    diag_merge.pending.store(false, std::memory_order_release);
  }
  return diag_entries;
}

void solve_diag(const FactorKeep &fkeep, double *x, int n) {
  (void)n;
  if (fkeep.is_posdef)
    return;
  if (!fkeep.diag().empty()) {
    for (const DiagSolveEntry &entry : fkeep.diag()) {
      if (entry.tag == 1) {
        if (entry.d00 == 0.0 || !is_finite_fast(entry.d00)) {
          x[entry.row0] = 0.0;
        } else {
          x[entry.row0] /= entry.d00;
        }
      } else if (entry.tag == 2) {
        const double x0 = x[entry.row0];
        const double x1 = x[entry.row1];
        double y0 = 0.0;
        double y1 = 0.0;
        solve_2x2_scaled(entry.d00, entry.d10, entry.d11, x0, x1, y0, y1);
        x[entry.row0] = y0;
        x[entry.row1] = y1;
      }
    }
    return;
  }
  if (!fkeep.analysis)
    return;

  const AnalysisKeep &analysis = *fkeep.analysis;
  const int ns = static_cast<int>(analysis.supernodes.size());

  for (int s = 0; s < ns; ++s) {
    const std::size_t si = static_cast<std::size_t>(s);
    if (si >= fkeep.pivot_types.size())
      continue;
    const auto &ptypes = fkeep.pivot_types[si];
    if (ptypes.empty())
      continue;

    const Supernode &sn = analysis.supernodes[si];
    const std::span<const Int> rows = analysis.front_rows(s);

    const int p = static_cast<int>(sn.width());
    const int f = static_cast<int>(rows.size());
    const LongInt start = fkeep.factor_col_ptr[si];

    for (int k = 0; k < p; ++k) {
      const std::size_t ki = static_cast<std::size_t>(k);
      if (ki >= ptypes.size())
        break;
      const int8_t tag = ptypes[ki];

      if (tag == 1) {
        // 1×1 pivot: divide x[r] by D[k,k]
        const double dkk =
            fkeep.factor_values[static_cast<std::size_t>(start + k * f + k)];
        x[rows[ki]] /= dkk;

      } else if (tag == 2) {
        // 2×2 pivot block at columns k and k+1
        // D stored col-major (lda=f): (row, col) → start + col*f + row
        const double d00 =
            fkeep.factor_values[static_cast<std::size_t>(start + k * f + k)];
        const double d10 = fkeep.factor_values[static_cast<std::size_t>(
            start + k * f + (k + 1))];
        const double d11 = fkeep.factor_values[static_cast<std::size_t>(
            start + (k + 1) * f + (k + 1))];

        const double det = d00 * d11 - d10 * d10;
        const int r0 = static_cast<int>(rows[ki]);
        const int r1 = static_cast<int>(rows[ki + 1]);
        const double x0 = x[r0];
        const double x1 = x[r1];
        // D^{-1} = [[d11, -d10], [-d10, d00]] / det
        x[r0] = (d11 * x0 - d10 * x1) / det;
        x[r1] = (-d10 * x0 + d00 * x1) / det;
      }
      // tag == -1: second half of 2×2 block, already handled above
      // tag ==  0: delayed pivot, skip
    }
  }
}

void apply_perm(const FactorKeep &fkeep, const double *x_in, double *x_out,
                int n) {
  for (int j = 0; j < n; ++j) {
    x_out[j] = x_in[fkeep.perm[static_cast<std::size_t>(j)]];
  }
}

void apply_inv_perm(const FactorKeep &fkeep, const double *x_in, double *x_out,
                    int n) {
  for (int j = 0; j < n; ++j) {
    x_out[j] = x_in[fkeep.iperm[static_cast<std::size_t>(j)]];
  }
}

} // namespace smf
