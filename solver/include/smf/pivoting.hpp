#pragma once
#include "smf/dense_indef.hpp"
#include "smf/frontal_matrix.hpp"
#include "smf/types.hpp"

namespace smf {

/// Result of a pivot selection attempt.
enum class PivotDecision { Accept1x1, Accept2x2, Reject };

struct PivotResult {
  PivotDecision decision;
  int col0; ///< front-local column index of pivot (-1 if Reject)
  int col1; ///< second column for 2×2; -1 for 1×1 or Reject
};

/// Select a pivot at front column k among the fully-summed columns [0, num_fs).
///
/// Bounded Bunch-Kaufman selection within FrontalMatrix F:
///   α = (1 + √17) / 8 ≈ 0.6404
///
///   ω1 = max |F.at(i, k)| for i in (k, num_fs)   (off-diagonal in col k, below
///   diag)
///
///   1×1 acceptance:  |F.at(k,k)| >= u * ω1
///   Otherwise:
///     find r = argmax in col k;  ω2 = max |F.at(i,r)| for i in [k,num_fs), i≠r
///     if |F.at(k,k)| * ω2 >= u * ω1^2  → swap k↔r, Accept1x1 at k
///     else                               → Accept2x2 at (k, r) [swap r→k+1 if
///     r≠k+1]
///
///   If ω1 == 0 (column k is zero off-diagonal): always Accept1x1.
///   If |F.at(k,k)| < small: Reject (treat as zero pivot).
///
/// Returns PivotResult with front-local column indices.
/// col0 != k in the Accept1x1 case means the caller must
/// sym_swap_front(F,k,col0,num_fs) before calling apply_pivot_1x1.
PivotResult choose_pivot(FrontalMatrix &F, int k, int num_fs, double u,
                         double small);

/// Apply a 1×1 pivot at front column k:
///   - Divide F.at(i,k) by d=F.at(k,k) for i in (k, num_fs) — forms L column k.
///   - Schur complement: F.at(i,j) -= F.at(i,k)*d*F.at(j,k) for j in
///   (k,num_fs), i in [j,num_fs).
void apply_pivot_1x1(FrontalMatrix &F, int k, int num_fs);

/// Apply a 2×2 pivot at front columns (k, k+1):
///   d00=F.at(k,k), d10=F.at(k+1,k), d11=F.at(k+1,k+1).
///   For each i in (k+1, num_fs):
///     l_i0 = (d11*F.at(i,k) - d10*F.at(i,k+1)) / det
///     l_i1 = (-d10*F.at(i,k) + d00*F.at(i,k+1)) / det
///     F.at(i,k) = l_i0;  F.at(i,k+1) = l_i1
///   Schur: for j in (k+1,num_fs), i in [j,num_fs):
///     F.at(i,j) -= l_i0*d00*l_j0 + l_i0*d10*l_j1 + l_i1*d10*l_j0 +
///     l_i1*d11*l_j1
void apply_pivot_2x2(FrontalMatrix &F, int k, int num_fs, double small);

/// Symmetric swap of columns (and corresponding rows) p and q within [0,
/// num_fs).
void sym_swap_front(FrontalMatrix &F, int p, int q, int num_fs);

} // namespace smf
