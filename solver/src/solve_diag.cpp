#include "smf/solve_diag.hpp"

#include <cstddef>
#include <cstdint>

namespace smf {

void solve_diag(const FactorKeep &fkeep, double *x, int n) {
  (void)n;
  if (fkeep.is_posdef)
    return;
  if (!fkeep.diag_entries.empty()) {
    for (const DiagSolveEntry &entry : fkeep.diag_entries) {
      if (entry.tag == 1) {
        x[entry.row0] /= entry.d00;
      } else if (entry.tag == 2) {
        const double det = entry.d00 * entry.d11 - entry.d10 * entry.d10;
        const double x0 = x[entry.row0];
        const double x1 = x[entry.row1];
        x[entry.row0] = (entry.d11 * x0 - entry.d10 * x1) / det;
        x[entry.row1] = (-entry.d10 * x0 + entry.d00 * x1) / det;
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
    const FrontalInfo &fi = analysis.fronts[si];

    const int p = static_cast<int>(sn.width());
    const int f = static_cast<int>(fi.front_size());
    const int start = fkeep.factor_col_ptr[si];

    for (int k = 0; k < p; ++k) {
      const std::size_t ki = static_cast<std::size_t>(k);
      if (ki >= ptypes.size())
        break;
      const int8_t tag = ptypes[ki];

      if (tag == 1) {
        // 1×1 pivot: divide x[r] by D[k,k]
        const double dkk =
            fkeep.factor_values[static_cast<std::size_t>(start + k * f + k)];
        x[fi.row_indices[ki]] /= dkk;

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
        const int r0 = static_cast<int>(fi.row_indices[ki]);
        const int r1 = static_cast<int>(fi.row_indices[ki + 1]);
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
