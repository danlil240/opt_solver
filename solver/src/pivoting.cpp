#include "smf/pivoting.hpp"
#include <algorithm>
#include <cmath>

namespace smf {

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
}

// ---------------------------------------------------------------------------
// choose_pivot  (Bounded Bunch-Kaufman, read-only on F)
// ---------------------------------------------------------------------------

PivotResult choose_pivot(FrontalMatrix &F, int k, int num_fs, double u,
                         double small) {
  // Compute ω1 = max |F.at(i,k)| for i in (k, num_fs); track argmax r
  double omega1 = 0.0;
  int r = k + 1; // default argmax (only used when omega1 > 0)
  for (int i = k + 1; i < num_fs; ++i) {
    double v = std::abs(F.at(i, k));
    if (v > omega1) {
      omega1 = v;
      r = i;
    }
  }

  // Only 1×1 is possible: last column in block, or column is zero off-diagonal
  if (omega1 == 0.0 || k + 1 >= num_fs) {
    if (std::abs(F.at(k, k)) < small)
      return {PivotDecision::Reject, -1, -1};
    return {PivotDecision::Accept1x1, k, -1};
  }

  // 1×1 acceptance test
  double akk = std::abs(F.at(k, k));
  if (akk >= u * omega1)
    return {PivotDecision::Accept1x1, k, -1};

  // Compute ω2 = max |F.at(i,r)| for i in [k, num_fs), i ≠ r
  // Lower-triangular: element (i,r) → F.at(max(i,r), min(i,r))
  double omega2 = 0.0;
  for (int i = k; i < num_fs; ++i) {
    if (i == r)
      continue;
    double v = (i < r) ? std::abs(F.at(r, i)) : std::abs(F.at(i, r));
    if (v > omega2)
      omega2 = v;
  }

  // BBK secondary test: swap k↔r and Accept1×1 if condition holds
  if (akk * omega2 >= u * omega1 * omega1)
    return {PivotDecision::Accept1x1, r, -1}; // col0=r: caller swaps r→k first

  // 2×2 pivot at (k, r): caller swaps r→k+1 if r ≠ k+1
  return {PivotDecision::Accept2x2, k, r};
}

// ---------------------------------------------------------------------------
// apply_pivot_1x1
// ---------------------------------------------------------------------------

void apply_pivot_1x1(FrontalMatrix &F, int k, int num_fs) {
  double d = F.at(k, k);

  // Form L column k: divide sub-diagonal by pivot
  for (int i = k + 1; i < num_fs; ++i)
    F.at(i, k) /= d;

  // Schur complement: F(i,j) -= l_ik * d * l_jk  (lower triangle, j > k)
  for (int j = k + 1; j < num_fs; ++j) {
    double ljk = F.at(j, k);
    for (int i = j; i < num_fs; ++i)
      F.at(i, j) -= F.at(i, k) * d * ljk;
  }
}

// ---------------------------------------------------------------------------
// apply_pivot_2x2
// ---------------------------------------------------------------------------

void apply_pivot_2x2(FrontalMatrix &F, int k, int num_fs, double small) {
  double d00 = F.at(k, k);
  double d10 = F.at(k + 1, k);
  double d11 = F.at(k + 1, k + 1);
  double det = d00 * d11 - d10 * d10;

  // Guard against degenerate 2×2 block
  if (std::abs(det) < small)
    return;

  // Form L columns k and k+1 for rows i in (k+1, num_fs)
  for (int i = k + 2; i < num_fs; ++i) {
    double a0 = F.at(i, k);
    double a1 = F.at(i, k + 1);
    F.at(i, k) = (d11 * a0 - d10 * a1) / det;
    F.at(i, k + 1) = (-d10 * a0 + d00 * a1) / det;
  }

  // Schur complement: F(i,j) -= l_i0*d00*l_j0 + l_i0*d10*l_j1
  //                            + l_i1*d10*l_j0 + l_i1*d11*l_j1
  for (int j = k + 2; j < num_fs; ++j) {
    double lj0 = F.at(j, k);
    double lj1 = F.at(j, k + 1);
    for (int i = j; i < num_fs; ++i) {
      double li0 = F.at(i, k);
      double li1 = F.at(i, k + 1);
      F.at(i, j) -=
          li0 * d00 * lj0 + li0 * d10 * lj1 + li1 * d10 * lj0 + li1 * d11 * lj1;
    }
  }
}

} // namespace smf
