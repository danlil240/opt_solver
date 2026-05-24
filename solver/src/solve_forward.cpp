#include "smf/solve_forward.hpp"

#include "smf/analysis.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/supernode.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace smf {

namespace {

static const std::vector<Int> &get_or_compute_postorder(
    const AnalysisKeep &ak, std::vector<Int> &local_buf) {
  if (!ak.postorder.empty())
    return ak.postorder;

  const Int ns = static_cast<Int>(ak.supernodes.size());
  local_buf.reserve(static_cast<std::size_t>(ns));
  std::vector<std::pair<Int, Int>> stack;
  stack.reserve(static_cast<std::size_t>(ns));
  for (Int i = 0; i < ns; ++i) {
    if (ak.supernodes[static_cast<std::size_t>(i)].parent == -1)
      stack.push_back({i, 0});
  }
  while (!stack.empty()) {
    auto &[node, child_pos] = stack.back();
    const auto &children = ak.supernodes[static_cast<std::size_t>(node)].children;
    if (child_pos < static_cast<Int>(children.size())) {
      stack.push_back({children[static_cast<std::size_t>(child_pos++)], 0});
    } else {
      local_buf.push_back(node);
      stack.pop_back();
    }
  }
  return local_buf;
}

static void solve_lower_nonunit_small(double *b, int p, const double *L,
                                      int lda) {
  for (int col = 0; col < p; ++col) {
    b[col] /= L[static_cast<std::ptrdiff_t>(col) * lda + col];
    const double xcol = b[col];
    const double *lcol = L + static_cast<std::ptrdiff_t>(col) * lda;
    for (int row = col + 1; row < p; ++row)
      b[row] -= lcol[row] * xcol;
  }
}

static void solve_lower_unit_small(double *b, int p, const double *L,
                                   int lda) {
  for (int col = 0; col < p; ++col) {
    const double xcol = b[col];
    const double *lcol = L + static_cast<std::ptrdiff_t>(col) * lda;
    for (int row = col + 1; row < p; ++row)
      b[row] -= lcol[row] * xcol;
  }
}

} // anonymous namespace

void solve_forward(const FactorKeep &fkeep, double *x, int n, int nrhs) {
  if (n == 0 || nrhs == 0)
    return;

  const std::vector<int> &iperm = fkeep.iperm;
  const bool use_steps = !fkeep.solve_steps.empty();
  const AnalysisKeep *ak = fkeep.analysis;
  std::vector<Int> po_buf;
  const std::vector<Int> *postorder = nullptr;
  const Int ns = use_steps ? static_cast<Int>(fkeep.solve_steps.size())
                           : static_cast<Int>((!ak->solve_postorder.empty()
                                                   ? ak->solve_postorder
                                                   : get_or_compute_postorder(*ak, po_buf))
                                                  .size());
  if (!use_steps) {
    postorder = &(!ak->solve_postorder.empty()
                      ? ak->solve_postorder
                      : get_or_compute_postorder(*ak, po_buf));
  }

  // Per-call scratch keeps solve_forward reentrant for concurrent callers.
  std::vector<double> tmp;
  std::vector<double> b_loc;
  std::vector<double> b_ext;
  tmp.resize(static_cast<std::size_t>(n));

  for (int rhs = 0; rhs < nrhs; ++rhs) {
    double *xcol = x + rhs * n;

    // Step 1: Apply P^T permutation: xcol_perm[iperm[j]] = xcol[j]
    for (int j = 0; j < n; ++j)
      tmp[static_cast<std::size_t>(iperm[static_cast<std::size_t>(j)])] =
          xcol[j];
    std::copy(tmp.begin(), tmp.end(), xcol);

    // Step 2: Forward substitution — process supernodes in postorder
    for (Int pos = 0; pos < ns; ++pos) {
      const Int *rows = nullptr;
      const double *L = nullptr;
      int p = 0;
      int f = 0;
      if (use_steps) {
        const SolveStep &step = fkeep.solve_steps[static_cast<std::size_t>(pos)];
        rows = fkeep.solve_row_indices.data() + step.row_offset;
        L = fkeep.factor_values.data() + step.factor_offset;
        p = step.p;
        f = step.f;
      } else {
        const Int s = (*postorder)[static_cast<std::size_t>(pos)];
        const FrontalInfo &fi = ak->fronts[static_cast<std::size_t>(s)];
        const Supernode &sn = ak->supernodes[static_cast<std::size_t>(s)];
        rows = fi.row_indices.data();
        L = &fkeep.factor_values[static_cast<std::size_t>(
            fkeep.factor_col_ptr[static_cast<std::size_t>(s)])];
        p = static_cast<int>(sn.width());
        f = static_cast<int>(fi.front_size());
      }
      const int q = f - p;                             // extended rows

      if (p <= 0)
        continue;

      if (p == 1) {
        const int row0 = rows[0];
        double x0 = xcol[row0];
        if (fkeep.is_posdef)
          x0 /= L[0];
        xcol[row0] = x0;
        for (int k = 1; k < f; ++k)
          xcol[rows[k]] -= L[k] * x0;
        continue;
      }

      if (p == 2) {
        const int row0 = rows[0];
        const int row1 = rows[1];
        double x0 = xcol[row0];
        if (fkeep.is_posdef)
          x0 /= L[0];
        double x1 = xcol[row1] - L[1] * x0;
        if (fkeep.is_posdef)
          x1 /= L[static_cast<std::ptrdiff_t>(f) + 1];
        xcol[row0] = x0;
        xcol[row1] = x1;
        const double *col0 = L;
        const double *col1 = L + static_cast<std::ptrdiff_t>(f);
        for (int k = 2; k < f; ++k)
          xcol[rows[k]] -= col0[k] * x0 + col1[k] * x1;
        continue;
      }

      // (a) Gather pivot values: b_loc[k] = xcol[ row_indices[k] ]
      b_loc.resize(static_cast<std::size_t>(p));
      for (int k = 0; k < p; ++k)
        b_loc[static_cast<std::size_t>(k)] =
            xcol[rows[k]];

      // (b) Solve L11 * y = b_loc. Most captured KKT fronts are tiny; direct
      // kernels avoid thousands of CBLAS calls while preserving BLAS for larger
      // dense fronts.
      if (fkeep.is_posdef) {
        if (p <= 16)
          solve_lower_nonunit_small(b_loc.data(), p, L, f);
        else
          smf_dtrsv_lower(b_loc.data(), p, L, f);
      } else {
        if (p <= 16)
          solve_lower_unit_small(b_loc.data(), p, L, f);
        else
          smf_dtrsv_lower_unit(b_loc.data(), p, L, f);
      }

      // (c) Scatter pivot values back
      for (int k = 0; k < p; ++k)
        xcol[rows[k]] = b_loc[static_cast<std::size_t>(k)];

      // (d) Update extended rows: b_ext -= L21 * b_loc
      if (q > 0) {
        if (p * q <= 256) {
          for (int j = 0; j < p; ++j) {
            const double xj = b_loc[static_cast<std::size_t>(j)];
            const double *col = L + static_cast<std::ptrdiff_t>(j) * f + p;
            for (int k = 0; k < q; ++k) {
              const int row = rows[p + k];
              xcol[row] -= col[k] * xj;
            }
          }
        } else {
          b_ext.resize(static_cast<std::size_t>(q));
          for (int k = 0; k < q; ++k)
            b_ext[static_cast<std::size_t>(k)] =
                xcol[rows[p + k]];

          smf_dgemv(b_ext.data(), q, p, L + p, f, b_loc.data(), -1.0, 1.0);

          for (int k = 0; k < q; ++k)
            xcol[rows[p + k]] = b_ext[static_cast<std::size_t>(k)];
        }
      }
    }
  }
}

} // namespace smf
