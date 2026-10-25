#include "smf/pivoting.hpp"
#include <algorithm>
#include <cmath>

namespace smf {

double scaled_det_2x2(double d00, double d10, double d11, double &scale_out) noexcept {
  const double s = std::max({std::abs(d00), std::abs(d10), std::abs(d11)});
  scale_out = s;
  if (!std::isfinite(s) || s == 0.0) {
    return 0.0;
  }
  const double a = d00 / s;
  const double b = d10 / s;
  const double c = d11 / s;
  return a * c - b * b;
}

bool solve_2x2_scaled(double d00, double d10, double d11,
                      double rhs0, double rhs1,
                      double &y0, double &y1) noexcept {
  double s = 0.0;
  const double det_s = scaled_det_2x2(d00, d10, d11, s);
  if (s == 0.0 || !std::isfinite(det_s) || det_s == 0.0) {
    y0 = 0.0;
    y1 = 0.0;
    return false;
  }
  if (!std::isfinite(rhs0) || !std::isfinite(rhs1)) {
    y0 = rhs0;
    y1 = rhs1;
    return false;
  }
  const double rhs_scale = std::max(std::abs(rhs0), std::abs(rhs1));
  if (rhs_scale == 0.0) {
    y0 = 0.0;
    y1 = 0.0;
    return true;
  }
  const double a = d00 / s;
  const double b = d10 / s;
  const double c = d11 / s;
  const double r0 = rhs0 / rhs_scale;
  const double r1 = rhs1 / rhs_scale;
  const double u0 = (c * r0 - b * r1) / det_s;
  const double u1 = (-b * r0 + a * r1) / det_s;
  const double ratio = rhs_scale / s;
  y0 = u0 * ratio;
  y1 = u1 * ratio;
  return std::isfinite(y0) && std::isfinite(y1);
}

// ---------------------------------------------------------------------------
// sym_swap_front
// ---------------------------------------------------------------------------

void sym_swap_front(FrontalMatrix &F, int p, int q, int num_fs) {
  if (p == q)
    return;
  if (p > q)
    std::swap(p, q); // ensure p < q

  // Diagonal entries
  std::swap(F.at(p, p), F.at(q, q));

  // Column-segment [0, p): rows p and q share the same column m
  for (int m = 0; m < p; ++m)
    std::swap(F.at(p, m), F.at(q, m));

  // Cross-band (p, q): column p row m  vs.  column m row q
  for (int m = p + 1; m < q; ++m)
    std::swap(F.at(m, p), F.at(q, m));

  // Column-segment (q, num_fs): same row m, columns p and q
  for (int m = q + 1; m < num_fs; ++m)
    std::swap(F.at(m, p), F.at(m, q));

  // Extension rows: swap column p and column q for rows >= num_fs
  for (int m = num_fs; m < F.front_size(); ++m)
    std::swap(F.at(m, p), F.at(m, q));
}

// ---------------------------------------------------------------------------
// choose_pivot  (Threshold pivoting, read-only on F)
// ---------------------------------------------------------------------------

PivotResult choose_pivot(FrontalMatrix &F, int k, int num_fs, double u,
                         double small) {
  const int f = F.front_size();
  if (k < 0 || k >= num_fs) {
    return {PivotDecision::Reject, -1, -1};
  }

  double front_max_abs = 0.0;
  for (int j = k; j < num_fs; ++j) {
    for (int i = j; i < f; ++i) {
      const double v = std::abs(F.at(i, j));
      if (!std::isfinite(v)) {
        return {PivotDecision::Reject, -1, -1};
      }
      if (v > front_max_abs) {
        front_max_abs = v;
      }
    }
  }
  const double eff_small =
      (front_max_abs > 0.0 && front_max_abs < small) ? (small * front_max_abs) : small;

  // Compute ω1 = max |F.at(i,k)| for i in (k, num_fs); track argmax r
  double omega1 = 0.0;
  int r = k + 1;
  for (int i = k + 1; i < num_fs; ++i) {
    const double v = std::abs(F.at(i, k));
    if (v > omega1) {
      omega1 = v;
      r = i;
    }
  }

  double omega1_ext = 0.0;
  for (int i = num_fs; i < f; ++i) {
    const double v = std::abs(F.at(i, k));
    if (v > omega1_ext) {
      omega1_ext = v;
    }
  }
  const double omega1_full = std::max(omega1, omega1_ext);
  const double akk = std::abs(F.at(k, k));

  // Only 1×1 is possible if last column in block or zero off-diagonal in fs block
  if (omega1 == 0.0 || k + 1 >= num_fs) {
    if (akk < eff_small || (omega1_full > 0.0 && akk < u * omega1_full))
      return {PivotDecision::Reject, -1, -1};
    return {PivotDecision::Accept1x1, k, -1};
  }

  // Primary 1×1 acceptance test at column k
  if (akk >= eff_small && akk >= u * omega1_full)
    return {PivotDecision::Accept1x1, k, -1};

  // Compute ω2 = max |F.at(i,r)| for i in [k, f), i ≠ r
  double omega2 = 0.0;
  for (int i = k; i < num_fs; ++i) {
    if (i == r)
      continue;
    const double v = (i < r) ? std::abs(F.at(r, i)) : std::abs(F.at(i, r));
    if (v > omega2)
      omega2 = v;
  }
  for (int i = num_fs; i < f; ++i) {
    const double v = std::abs(F.at(i, r));
    if (v > omega2)
      omega2 = v;
  }

  // Secondary 1×1 test: check candidate diagonal at r (|F(r,r)| >= u * omega2)
  const double arr = std::abs(F.at(r, r));
  if (arr >= eff_small && arr >= u * omega2)
    return {PivotDecision::Accept1x1, r, -1};

  // 2×2 pivot test at (k, r) using scaled determinant and threshold stability bound
  const double d00 = F.at(k, k);
  const double d10 = F.at(r, k);
  const double d11 = F.at(r, r);
  double s2 = 0.0;
  const double det_s = scaled_det_2x2(d00, d10, d11, s2);
  if (s2 < eff_small || std::abs(det_s) <= small)
    return {PivotDecision::Reject, -1, -1};

  double off_k = 0.0;
  double off_r = 0.0;
  for (int i = k; i < f; ++i) {
    if (i == k || i == r)
      continue;
    const double vk = (i < num_fs)
                          ? ((i < k) ? std::abs(F.at(k, i)) : std::abs(F.at(i, k)))
                          : std::abs(F.at(i, k));
    const double vr = (i < num_fs)
                          ? ((i < r) ? std::abs(F.at(r, i)) : std::abs(F.at(i, r)))
                          : std::abs(F.at(i, r));
    if (vk > off_k) off_k = vk;
    if (vr > off_r) off_r = vr;
  }
  if (off_k > 0.0 || off_r > 0.0) {
    const double a = std::abs(d00) / s2;
    const double b = std::abs(d10) / s2;
    const double c = std::abs(d11) / s2;
    const double bound0 = (c * off_k + b * off_r) / s2;
    const double bound1 = (b * off_k + a * off_r) / s2;
    if (u * std::max(bound0, bound1) > std::abs(det_s)) {
      return {PivotDecision::Reject, -1, -1};
    }
  }

  return {PivotDecision::Accept2x2, k, r};
}

// ---------------------------------------------------------------------------
// apply_pivot_1x1
// ---------------------------------------------------------------------------

void apply_pivot_1x1(FrontalMatrix &F, int k, int num_fs) {
  const double d = F.at(k, k);
  const int f = F.front_size();

  // Form L column k: divide sub-diagonal by pivot (including extension rows)
  for (int i = k + 1; i < f; ++i)
    F.at(i, k) /= d;

  // Schur complement: F(i,j) -= l_ik * d * l_jk  (lower triangle, j > k)
  for (int j = k + 1; j < num_fs; ++j) {
    const double w_jk = d * F.at(j, k);
    for (int i = j; i < f; ++i)
      F.at(i, j) -= F.at(i, k) * w_jk;
  }
}

// ---------------------------------------------------------------------------
// apply_pivot_2x2
// ---------------------------------------------------------------------------

void apply_pivot_2x2(FrontalMatrix &F, int k, int num_fs, double small) {
  const double d00 = F.at(k, k);
  const double d10 = F.at(k + 1, k);
  const double d11 = F.at(k + 1, k + 1);
  double s2 = 0.0;
  const double det_s = scaled_det_2x2(d00, d10, d11, s2);

  // Guard against degenerate 2×2 block
  if (s2 == 0.0 || std::abs(det_s) <= small)
    return;

  const int f = F.front_size();

  // Form L columns k and k+1 for rows i in (k+1, f), including extension rows
  for (int i = k + 2; i < f; ++i) {
    const double a0 = F.at(i, k);
    const double a1 = F.at(i, k + 1);
    double l0 = 0.0;
    double l1 = 0.0;
    solve_2x2_scaled(d00, d10, d11, a0, a1, l0, l1);
    F.at(i, k) = l0;
    F.at(i, k + 1) = l1;
  }

  // Schur complement: F(i,j) -= l_i0 * w_j0 + l_i1 * w_j1
  for (int j = k + 2; j < num_fs; ++j) {
    const double lj0 = F.at(j, k);
    const double lj1 = F.at(j, k + 1);
    const double wj0 = d00 * lj0 + d10 * lj1;
    const double wj1 = d10 * lj0 + d11 * lj1;
    for (int i = j; i < f; ++i) {
      const double li0 = F.at(i, k);
      const double li1 = F.at(i, k + 1);
      F.at(i, j) -= li0 * wj0 + li1 * wj1;
    }
  }
}

} // namespace smf
