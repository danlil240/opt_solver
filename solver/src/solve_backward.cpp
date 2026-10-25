#include "smf/solve_backward.hpp"
#include "smf/analysis.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/workspace.hpp"
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

// L^T z = b for small p, row-oriented: once z[row] is final it is
// subtracted from every earlier entry. The updates are independent, unlike
// the dot-product form whose single accumulator serializes every FMA.
static void solve_lower_transpose_nonunit_small(double *b, int p,
                                                 const double *L, int lda) {
  for (int row = p - 1; row >= 0; --row) {
    const double zr = b[row] / L[static_cast<std::ptrdiff_t>(row) * lda + row];
    b[row] = zr;
    for (int col = 0; col < row; ++col)
      b[col] -= L[static_cast<std::ptrdiff_t>(col) * lda + row] * zr;
  }
}

static void solve_lower_transpose_unit_small(double *b, int p,
                                             const double *L, int lda) {
  for (int row = p - 1; row >= 0; --row) {
    const double zr = b[row];
    for (int col = 0; col < row; ++col)
      b[col] -= L[static_cast<std::ptrdiff_t>(col) * lda + row] * zr;
  }
}

// Dot product of two contiguous vectors; four accumulators keep the FMA
// latency off the critical path.
static double dot_contig(const double *a, const double *b, int n) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i)
    s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

} // anonymous namespace

void solve_backward(const FactorKeep &fkeep, double *x, int n, int nrhs,
                    SolveWorkspace &ws) {
  if (n <= 0 || nrhs <= 0 || x == nullptr)
    return;

  const bool use_steps = !fkeep.solve_steps.empty();
  const AnalysisKeep *ak = fkeep.analysis;
  if (!use_steps && ak == nullptr)
    return;
  std::vector<Int> po_buf;
  const std::vector<Int> *postorder = nullptr;
  const int ns = use_steps ? static_cast<int>(fkeep.solve_steps.size())
                           : static_cast<int>((!ak->solve_postorder.empty()
                                                   ? ak->solve_postorder
                                                   : get_or_compute_postorder(*ak, po_buf))
                                                  .size());
  if (!use_steps) {
    postorder = &(!ak->solve_postorder.empty()
                      ? ak->solve_postorder
                      : get_or_compute_postorder(*ak, po_buf));
  }

  std::vector<double> &b_loc = ws.loc;
  std::vector<double> &b_ext = ws.ext;
  std::vector<double> &tmp = ws.tmp;
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
        const Int s = (*postorder)[static_cast<std::size_t>(pos)];
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
      if (b_loc.size() < static_cast<std::size_t>(p))
        b_loc.resize(static_cast<std::size_t>(p));
      for (int i = 0; i < p; ++i)
        b_loc[static_cast<std::size_t>(i)] =
            xc[rows[i]];

      // Update b_loc -= L21^T * b_ext, with b_ext gathered once so the
      // per-column work runs on contiguous data.
      if (q > 0) {
        if (b_ext.size() < static_cast<std::size_t>(q))
          b_ext.resize(static_cast<std::size_t>(q));
        double *be = b_ext.data();
        for (int i = 0; i < q; ++i)
          be[i] = xc[rows[p + i]];
        if (p * q > 4096) {
          smf_dgemv_transpose(b_loc.data(), q, p, factor_data + p, f, be,
                              -1.0, 1.0);
        } else {
          for (int j = 0; j < p; ++j) {
            const double *col =
                factor_data + static_cast<std::ptrdiff_t>(j) * f + p;
            b_loc[static_cast<std::size_t>(j)] -= dot_contig(col, be, q);
          }
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

void solve_backward(const FactorKeep &fkeep, double *x, int n, int nrhs) {
  SolveWorkspace ws;
  solve_backward(fkeep, x, n, nrhs, ws);
}

} // namespace smf
