#include "smf/solve_forward.hpp"

#include "smf/analysis.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/supernode.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace smf {

namespace {

} // anonymous namespace

/// Return a postorder reference: precomputed from analysis if available,
/// otherwise compute on demand into `local_buf` (tests / legacy callers).
static const std::vector<Int>&
get_or_compute_postorder(const AnalysisKeep& ak, std::vector<Int>& local_buf)
{
    if (!ak.postorder.empty())
        return ak.postorder;

    // Fallback: compute via iterative DFS (mirrors symbolic_analysis.cpp).
    const Int ns = static_cast<Int>(ak.supernodes.size());
    local_buf.reserve(static_cast<std::size_t>(ns));
    std::vector<std::pair<Int,Int>> stk;
    stk.reserve(static_cast<std::size_t>(ns));
    for (Int i = 0; i < ns; ++i)
        if (ak.supernodes[static_cast<std::size_t>(i)].parent == -1)
            stk.push_back({i, 0});
    while (!stk.empty()) {
        auto& [node, ci] = stk.back();
        const Int nch = static_cast<Int>(ak.supernodes[static_cast<std::size_t>(node)].children.size());
        if (ci < nch) {
            stk.push_back({ak.supernodes[static_cast<std::size_t>(node)].children[static_cast<std::size_t>(ci++)], 0});
        } else {
            local_buf.push_back(node);
            stk.pop_back();
        }
    }
    return local_buf;
}

void solve_forward(const FactorKeep &fkeep, double *x, int n, int nrhs) {
  if (n == 0 || nrhs == 0)
    return;

  const AnalysisKeep &ak = *fkeep.analysis;
  const std::vector<Supernode> &supernodes = ak.supernodes;
  const std::vector<FrontalInfo> &fronts = ak.fronts;
  const std::vector<int> &iperm = fkeep.iperm;

  const Int ns = static_cast<Int>(supernodes.size());
  std::vector<Int> po_buf;
  const std::vector<Int>& postorder = get_or_compute_postorder(ak, po_buf); // precomputed in analysis

  // Scratch buffers (reused across all RHS columns and supernodes)
  std::vector<double> b_loc;
  std::vector<double> b_ext;
  std::vector<double> tmp_perm(static_cast<std::size_t>(n));

  for (int rhs = 0; rhs < nrhs; ++rhs) {
    double *xcol = x + rhs * n;

    // Step 1: Apply P^T permutation: xcol_perm[iperm[j]] = xcol[j]
    {
      for (int j = 0; j < n; ++j)
        tmp_perm[static_cast<std::size_t>(iperm[static_cast<std::size_t>(j)])] =
            xcol[j];
      std::copy(tmp_perm.begin(), tmp_perm.end(), xcol);
    }

    // Step 2: Forward substitution — process supernodes in postorder
    for (Int pos = 0; pos < ns; ++pos) {
      const Int s = postorder[static_cast<std::size_t>(pos)];
      const FrontalInfo &fi = fronts[static_cast<std::size_t>(s)];
      const Supernode &sn = supernodes[static_cast<std::size_t>(s)];

      const int p = static_cast<int>(sn.width());      // pivot columns
      const int f = static_cast<int>(fi.front_size()); // total front rows
      const int q = f - p;                             // extended rows

      if (p <= 0)
        continue;

      // Pointer to L data for supernode s: f×p col-major block, lda=f
      const double *L = &fkeep.factor_values[static_cast<std::size_t>(
          fkeep.factor_col_ptr[static_cast<std::size_t>(s)])];

      // (a) Gather pivot values: b_loc[k] = xcol[ row_indices[k] ]
      b_loc.resize(static_cast<std::size_t>(p));
      for (int k = 0; k < p; ++k)
        b_loc[static_cast<std::size_t>(k)] =
            xcol[fi.row_indices[static_cast<std::size_t>(k)]];

      // (b) Solve L11 * y = b_loc
      // SPD: non-unit lower triangular (dpotrf stores full L including
      // diagonal). Indef: unit lower triangular (diagonal stores D; actual L
      // diagonal = 1).
      if (fkeep.is_posdef)
        smf_dtrsv_lower(b_loc.data(), p, L, f);
      else
        smf_dtrsv_lower_unit(b_loc.data(), p, L, f);

      // (c) Scatter pivot values back
      for (int k = 0; k < p; ++k)
        xcol[fi.row_indices[static_cast<std::size_t>(k)]] =
            b_loc[static_cast<std::size_t>(k)];

      // (d) Update extended rows: b_ext -= L21 * b_loc
      if (q > 0) {
        b_ext.resize(static_cast<std::size_t>(q));

        // Gather extended rows
        for (int k = 0; k < q; ++k)
          b_ext[static_cast<std::size_t>(k)] =
              xcol[fi.row_indices[static_cast<std::size_t>(p + k)]];

        // L21 is rows [p, f) of the f×p block (col-major, lda=f).
        // b_ext = -1 * L21 * b_loc + 1 * b_ext
        smf_dgemv(b_ext.data(), q, p, L + p, f, b_loc.data(), -1.0, 1.0);

        // Scatter extended rows back
        for (int k = 0; k < q; ++k)
          xcol[fi.row_indices[static_cast<std::size_t>(p + k)]] =
              b_ext[static_cast<std::size_t>(k)];
      }
    }
  }
}

} // namespace smf
