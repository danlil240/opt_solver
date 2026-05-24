#include "smf/solve_backward.hpp"
#include "smf/analysis.hpp"
#include "smf/dense_kernel.hpp"
#include <cstddef>
#include <vector>

namespace smf {

namespace {

static void solve_lower_transpose_nonunit_small(double *b, int p,
                                                const double *L, int lda) {
  for (int col = p - 1; col >= 0; --col) {
    double value = b[col];
    const double *lcol = L + static_cast<std::ptrdiff_t>(col) * lda;
    for (int row = col + 1; row < p; ++row)
      value -= lcol[row] * b[row];
    value /= lcol[col];
    b[col] = value;
  }
}

static void solve_lower_transpose_unit_small(double *b, int p,
                                             const double *L, int lda) {
  for (int col = p - 1; col >= 0; --col) {
    double value = b[col];
    const double *lcol = L + static_cast<std::ptrdiff_t>(col) * lda;
    for (int row = col + 1; row < p; ++row)
      value -= lcol[row] * b[row];
    b[col] = value;
  }
}

} // anonymous namespace

static const std::vector<Int> &
get_or_compute_solve_postorder(const AnalysisKeep &ak,
                               std::vector<Int> &local_buf) {
  if (!ak.solve_postorder.empty())
    return ak.solve_postorder;
  if (!ak.postorder.empty())
    return ak.postorder;

  const Int ns = static_cast<Int>(ak.supernodes.size());
  local_buf.reserve(static_cast<std::size_t>(ns));
  std::vector<std::pair<Int, Int>> stk;
  stk.reserve(static_cast<std::size_t>(ns));
  for (Int i = 0; i < ns; ++i)
    if (ak.supernodes[static_cast<std::size_t>(i)].parent == -1)
      stk.push_back({i, 0});
  while (!stk.empty()) {
    auto &[node, ci] = stk.back();
    const Int nch = static_cast<Int>(
        ak.supernodes[static_cast<std::size_t>(node)].children.size());
    if (ci < nch) {
      stk.push_back(
          {ak.supernodes[static_cast<std::size_t>(node)]
               .children[static_cast<std::size_t>(ci++)],
           0});
    } else {
      local_buf.push_back(node);
      stk.pop_back();
    }
  }
  return local_buf;
}

void solve_backward(const FactorKeep &fkeep, double *x, int n, int nrhs) {
  if (n == 0 || nrhs == 0)
    return;

  const bool use_steps = !fkeep.solve_steps.empty();
  const AnalysisKeep *ak = fkeep.analysis;
  std::vector<Int> local_postorder;
  const std::vector<Int> *solve_postorder = nullptr;
  if (!use_steps) {
    solve_postorder = &get_or_compute_solve_postorder(*ak, local_postorder);
  }
  const int ns = use_steps ? static_cast<int>(fkeep.solve_steps.size())
                           : static_cast<int>(solve_postorder->size());

  std::vector<double> &b_loc = fkeep.solve_loc;
  std::vector<double> &tmp = fkeep.solve_tmp;
  tmp.resize(static_cast<std::size_t>(n));

  for (int rhs = 0; rhs < nrhs; ++rhs) {
    double *xc = x + static_cast<std::ptrdiff_t>(rhs) * n;

    // Reverse-postorder: roots first, then toward leaves
    for (int pos = ns - 1; pos >= 0; --pos) {
      const Int *rows = nullptr;
      const double *factor_data = nullptr;
      int p = 0;
      int f = 0;
      if (use_steps) {
        const SolveStep &step = fkeep.solve_steps[static_cast<std::size_t>(pos)];
        rows = fkeep.solve_row_indices.data() + step.row_offset;
        factor_data = fkeep.factor_values.data() + step.factor_offset;
        p = step.p;
        f = step.f;
      } else {
        const Int s = (*solve_postorder)[static_cast<std::size_t>(pos)];
        const Supernode &sn = ak->supernodes[static_cast<std::size_t>(s)];
        const FrontalInfo &fi = ak->fronts[static_cast<std::size_t>(s)];
        rows = fi.row_indices.data();
        factor_data = fkeep.factor_values.data() +
                      fkeep.factor_col_ptr[static_cast<std::size_t>(s)];
        p = sn.width();
        f = fi.front_size();
      }
      const int q = f - p;

      if (p == 1) {
        double value = xc[rows[0]];
        for (int i = 1; i < f; ++i)
          value -= factor_data[i] * xc[rows[i]];
        if (fkeep.is_posdef)
          value /= factor_data[0];
        xc[rows[0]] = value;
        continue;
      }

      if (p == 2) {
        const double *col0 = factor_data;
        const double *col1 = factor_data + static_cast<std::ptrdiff_t>(f);
        double x1 = xc[rows[1]];
        for (int i = 2; i < f; ++i)
          x1 -= col1[i] * xc[rows[i]];
        if (fkeep.is_posdef)
          x1 /= col1[1];

        double x0 = xc[rows[0]] - col0[1] * x1;
        for (int i = 2; i < f; ++i)
          x0 -= col0[i] * xc[rows[i]];
        if (fkeep.is_posdef)
          x0 /= col0[0];

        xc[rows[1]] = x1;
        xc[rows[0]] = x0;
        continue;
      }

      // Gather pivot rows
      b_loc.resize(static_cast<std::size_t>(p));
      for (int i = 0; i < p; ++i)
        b_loc[static_cast<std::size_t>(i)] =
            xc[rows[i]];

      // Update b_loc -= L21^T * b_ext
      if (q > 0) {
        for (int j = 0; j < p; ++j) {
          const double *col =
              factor_data + static_cast<std::ptrdiff_t>(j) * f + p;
          double dot = 0.0;
          for (int i = 0; i < q; ++i)
            dot += col[i] * xc[rows[p + i]];
          b_loc[static_cast<std::size_t>(j)] -= dot;
        }
      }

      // Solve L11^T z = b_loc (SPD: non-unit; indef: unit lower diagonal)
      if (fkeep.is_posdef) {
        if (p <= 16)
          solve_lower_transpose_nonunit_small(b_loc.data(), p, factor_data, f);
        else
          smf_dtrsv_lower_transpose(b_loc.data(), p, factor_data, f);
      } else {
        if (p <= 16)
          solve_lower_transpose_unit_small(b_loc.data(), p, factor_data, f);
        else
          smf_dtrsv_lower_transpose_unit(b_loc.data(), p, factor_data, f);
      }

      // Scatter
      for (int i = 0; i < p; ++i)
        xc[rows[i]] = b_loc[static_cast<std::size_t>(i)];
    }

    // Apply fill-reducing permutation: x_original[perm[j]] = x_permuted[j]
    for (int j = 0; j < n; ++j)
      tmp[static_cast<std::size_t>(fkeep.perm[static_cast<std::size_t>(j)])] =
          xc[j];
    for (int j = 0; j < n; ++j)
      xc[j] = tmp[static_cast<std::size_t>(j)];
  }
}

} // namespace smf
