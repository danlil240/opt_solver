#include "smf/solve_backward.hpp"
#include "smf/analysis.hpp"
#include "smf/dense_kernel.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace smf {

namespace {
/// Compute postorder traversal matching factor_posdef / solve_forward.
static std::vector<Int>
compute_postorder_bwd(const std::vector<Supernode> &supernodes) {
  const Int ns = static_cast<Int>(supernodes.size());
  std::vector<Int> order;
  order.reserve(static_cast<std::size_t>(ns));
  std::vector<std::pair<Int, Int>> stk;
  stk.reserve(static_cast<std::size_t>(ns));
  for (Int i = 0; i < ns; ++i)
    if (supernodes[static_cast<std::size_t>(i)].parent == -1)
      stk.push_back({i, 0});
  while (!stk.empty()) {
    auto &[node, ci] = stk.back();
    const Int nch = static_cast<Int>(
        supernodes[static_cast<std::size_t>(node)].children.size());
    if (ci < nch) {
      const Int child = supernodes[static_cast<std::size_t>(node)]
                            .children[static_cast<std::size_t>(ci)];
      ++ci;
      stk.push_back({child, 0});
    } else {
      order.push_back(node);
      stk.pop_back();
    }
  }
  return order;
}
} // anonymous namespace

void solve_backward(const FactorKeep &fkeep, double *x, int n, int nrhs) {
  if (n == 0 || nrhs == 0)
    return;

  const AnalysisKeep &ak = *fkeep.analysis;
  const int ns = static_cast<int>(ak.supernodes.size());

  // Compute postorder (children before parents), then reverse for backward
  const std::vector<Int> postorder = compute_postorder_bwd(ak.supernodes);

  std::vector<double> b_loc;
  std::vector<double> b_ext;

  for (int rhs = 0; rhs < nrhs; ++rhs) {
    double *xc = x + static_cast<std::ptrdiff_t>(rhs) * n;

    // Reverse-postorder: roots first, then toward leaves
    for (int pos = ns - 1; pos >= 0; --pos) {
      const Int s = postorder[static_cast<std::size_t>(pos)];
      const Supernode &sn = ak.supernodes[static_cast<std::size_t>(s)];
      const FrontalInfo &fi = ak.fronts[static_cast<std::size_t>(s)];

      const int p = sn.width();
      const int f = fi.front_size();
      const int q = f - p;

      const double *factor_data =
          fkeep.factor_values.data() +
          fkeep.factor_col_ptr[static_cast<std::size_t>(s)];

      // Gather pivot rows
      b_loc.resize(static_cast<std::size_t>(p));
      for (int i = 0; i < p; ++i)
        b_loc[static_cast<std::size_t>(i)] =
            xc[fi.row_indices[static_cast<std::size_t>(i)]];

      // Update b_loc -= L21^T * b_ext
      if (q > 0) {
        b_ext.resize(static_cast<std::size_t>(q));
        for (int i = 0; i < q; ++i)
          b_ext[static_cast<std::size_t>(i)] =
              xc[fi.row_indices[static_cast<std::size_t>(p + i)]];

        for (int j = 0; j < p; ++j) {
          const double *col =
              factor_data + static_cast<std::ptrdiff_t>(j) * f + p;
          double dot = 0.0;
          for (int i = 0; i < q; ++i)
            dot += col[i] * b_ext[static_cast<std::size_t>(i)];
          b_loc[static_cast<std::size_t>(j)] -= dot;
        }
      }

      // Solve L11^T z = b_loc (SPD: non-unit; indef: unit lower diagonal)
      if (fkeep.is_posdef)
        smf_dtrsv_lower_transpose(b_loc.data(), p, factor_data, f);
      else
        smf_dtrsv_lower_transpose_unit(b_loc.data(), p, factor_data, f);

      // Scatter
      for (int i = 0; i < p; ++i)
        xc[fi.row_indices[static_cast<std::size_t>(i)]] =
            b_loc[static_cast<std::size_t>(i)];
    }

    // Apply fill-reducing permutation: x_original[perm[j]] = x_permuted[j]
    std::vector<double> tmp(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j)
      tmp[static_cast<std::size_t>(fkeep.perm[static_cast<std::size_t>(j)])] =
          xc[j];
    for (int j = 0; j < n; ++j)
      xc[j] = tmp[static_cast<std::size_t>(j)];
  }
}

} // namespace smf
