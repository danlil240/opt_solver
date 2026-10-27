#include "smf/solve_backward.hpp"
#include "smf/analysis.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/pivoting.hpp"
#include "smf/solve_pool.hpp"
#include "smf/workspace.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <span>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

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
    const std::span<const Int> children = ak.children(node);
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

// b[j] -= L(:, j) . be for the p columns of a front's L21 (column stride
// lda, length q). Under /fp:precise the dot products stay scalar, which
// made this loop 38% of the solve on KKT trees of small fronts; with AVX2
// four columns go at once, sharing every load of be.
static void sub_dots(double *b, int p, const double *L, std::ptrdiff_t lda,
                     const double *be, int q) {
  int j = 0;
#if defined(__AVX2__)
  for (; j + 4 <= p; j += 4) {
    const double *c0 = L + j * lda;
    const double *c1 = c0 + lda;
    const double *c2 = c1 + lda;
    const double *c3 = c2 + lda;
    __m256d a0 = _mm256_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    int k = 0;
    for (; k + 4 <= q; k += 4) {
      const __m256d x = _mm256_loadu_pd(be + k);
      a0 = _mm256_fmadd_pd(_mm256_loadu_pd(c0 + k), x, a0);
      a1 = _mm256_fmadd_pd(_mm256_loadu_pd(c1 + k), x, a1);
      a2 = _mm256_fmadd_pd(_mm256_loadu_pd(c2 + k), x, a2);
      a3 = _mm256_fmadd_pd(_mm256_loadu_pd(c3 + k), x, a3);
    }
    // Horizontal sums of the four accumulators in one pass.
    const __m256d h01 = _mm256_hadd_pd(a0, a1);
    const __m256d h23 = _mm256_hadd_pd(a2, a3);
    const __m256d sum = _mm256_add_pd(_mm256_permute2f128_pd(h01, h23, 0x20),
                                      _mm256_permute2f128_pd(h01, h23, 0x31));
    alignas(32) double s[4];
    _mm256_store_pd(s, sum);
    for (; k < q; ++k) {
      s[0] += c0[k] * be[k];
      s[1] += c1[k] * be[k];
      s[2] += c2[k] * be[k];
      s[3] += c3[k] * be[k];
    }
    b[j] -= s[0];
    b[j + 1] -= s[1];
    b[j + 2] -= s[2];
    b[j + 3] -= s[3];
  }
#endif
  for (; j < p; ++j)
    b[j] -= dot_contig(L + j * lda, be, q);
}

// v[0, p) = D^-1 v for a front's pivot block with D inline
// (FactorKeep::inline_d): d on the diagonal, a nonzero (c, c + 1) slot marks
// a 2x2 block (one with a zero off-diagonal is two 1x1 pivots anyway). Zero
// or non-finite 1x1 pivots give 0, as solve_diag does.
void apply_inline_d(double *v, int p, const double *L, std::ptrdiff_t f) {
  for (int c = 0; c < p;) {
    const double d00 = L[c * f + c];
    if (c + 1 < p) {
      const double d10 = L[(c + 1) * f + c];
      if (d10 != 0.0) {
        double y0 = 0.0, y1 = 0.0;
        solve_2x2_scaled(d00, d10, L[(c + 1) * f + c + 1], v[c], v[c + 1], y0, y1);
        v[c] = y0;
        v[c + 1] = y1;
        c += 2;
        continue;
      }
    }
    v[c] = (d00 == 0.0 || !is_finite_fast(d00)) ? 0.0 : v[c] / d00;
    ++c;
  }
}

// One front of L^{-T}: reads the extended rows rows[p, f) (already
// solved: they belong to ancestors) and writes only the pivot rows
// rows[0, p), so fronts of disjoint subtrees can run concurrently.
// With apply_d, the pivot values first take D^-1 (inline D).
void backward_front(const double *factor_data, const Int *rows, int p, int f,
                    bool posdef, bool apply_d, double *xc, std::vector<double> &b_loc,
                    std::vector<double> &b_ext) {
  const int q = f - p;
  if (p <= 0)
    return;

  // p = 1, 2: dot products over gathered entries, split over two
  // accumulators so the FMA chain is not the critical path.
  if (p == 1) {
    double v0 = xc[rows[0]];
    if (apply_d)
      apply_inline_d(&v0, 1, factor_data, f);
    double s0 = 0.0, s1 = 0.0;
    int i = 1;
    for (; i + 2 <= f; i += 2) {
      s0 += factor_data[i] * xc[rows[i]];
      s1 += factor_data[i + 1] * xc[rows[i + 1]];
    }
    if (i < f)
      s0 += factor_data[i] * xc[rows[i]];
    double value = v0 - (s0 + s1);
    if (posdef)
      value /= factor_data[0];
    xc[rows[0]] = value;
    return;
  }

  if (p == 2) {
    const double *col0 = factor_data;
    const double *col1 = factor_data + static_cast<std::ptrdiff_t>(f);
    double v[2] = {xc[rows[0]], xc[rows[1]]};
    if (apply_d)
      apply_inline_d(v, 2, factor_data, f);
    double s0 = 0.0, s1 = 0.0;
    for (int i = 2; i < f; ++i) {
      const double xi = xc[rows[i]];
      s0 += col0[i] * xi;
      s1 += col1[i] * xi;
    }
    double x1 = v[1] - s1;
    if (posdef)
      x1 /= col1[1];

    double x0 = v[0] - col0[1] * x1 - s0;
    if (posdef)
      x0 /= col0[0];

    xc[rows[1]] = x1;
    xc[rows[0]] = x0;
    return;
  }

  // Gather pivot rows
  if (b_loc.size() < static_cast<std::size_t>(p))
    b_loc.resize(static_cast<std::size_t>(p));
  for (int i = 0; i < p; ++i)
    b_loc[static_cast<std::size_t>(i)] = xc[rows[i]];
  if (apply_d)
    apply_inline_d(b_loc.data(), p, factor_data, f);

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
      sub_dots(b_loc.data(), p, factor_data + p, f, be, q);
    }
  }

  // Solve L11^T z = b_loc (SPD: non-unit; indef: unit lower diagonal)
  if (posdef) {
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

} // anonymous namespace

void solve_backward(const FactorKeep &fkeep, double *x, int n, int nrhs,
                    SolveWorkspace &ws, int num_threads, bool apply_diag,
                    bool permute) {
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
  const bool posdef = fkeep.is_posdef;
  const bool apply_d = apply_diag && backward_applies_diag(fkeep);
  const auto step_front = [&](std::size_t k, const double *&L, const Int *&rows,
                              int &p, int &f) {
    const SolveStep &step = fkeep.solve_steps[k];
    rows = fkeep.solve_row_indices.data() + step.row_offset;
    L = fkeep.factor_values.data() + step.factor_offset;
    p = step.p;
    f = step.f;
  };

  // Parallel layout (see solve_forward): the top of the tree first, then
  // the independent subtrees concurrently.
  const std::size_t nsub = fkeep.solve_subtrees.size();
  const int threads = (use_steps && num_threads > 1 && nsub >= 2 &&
                       static_cast<LongInt>(fkeep.solve_top_parent.size()) ==
                           static_cast<LongInt>(fkeep.solve_steps.size()) - fkeep.solve_top_begin)
                          ? std::min<int>(num_threads, static_cast<int>(nsub))
                          : 1;

  std::vector<double> &b_loc = ws.loc;
  std::vector<double> &b_ext = ws.ext;
  std::vector<double> &tmp = ws.tmp;
  tmp.resize(static_cast<std::size_t>(n));

  for (int rhs = 0; rhs < nrhs; ++rhs) {
    double *xc = x + static_cast<std::ptrdiff_t>(rhs) * n;

    if (threads > 1) {
      SolvePool &pool = ws.ensure_pool(threads);
      BlasSerialGuard serial_blas; // the caller is thread 0

      // The top of the tree as a DAG from the roots: a front runs once its
      // parent is done (its extended rows are then final), and releases its
      // children.
      const auto top_begin = static_cast<std::size_t>(fkeep.solve_top_begin);
      const std::size_t nts = fkeep.solve_steps.size() - top_begin;
      std::vector<int> child_ptr(nts + 1, 0), child(nts), ready;
      for (std::size_t i = 0; i < nts; ++i) {
        const Int par = fkeep.solve_top_parent[i];
        if (par >= 0)
          ++child_ptr[static_cast<std::size_t>(par) + 1];
        else
          ready.push_back(static_cast<int>(i));
      }
      for (std::size_t i = 0; i < nts; ++i)
        child_ptr[i + 1] += child_ptr[i];
      {
        std::vector<int> head(child_ptr.begin(), child_ptr.end() - 1);
        for (std::size_t i = 0; i < nts; ++i) {
          const Int par = fkeep.solve_top_parent[i];
          if (par >= 0)
            child[static_cast<std::size_t>(head[static_cast<std::size_t>(par)]++)] = static_cast<int>(i);
        }
      }
      run_dag(pool, static_cast<int>(nts), std::move(ready),
              [&](int i, int tid, const auto &release) {
                auto &sc = ws.threads[static_cast<std::size_t>(tid)];
                const double *L;
                const Int *rows;
                int p, f;
                step_front(top_begin + static_cast<std::size_t>(i), L, rows, p, f);
                backward_front(L, rows, p, f, posdef, apply_d, xc, sc.loc, sc.ext);
                // Run the first child here, queue the others.
                const int c0 = child_ptr[static_cast<std::size_t>(i)];
                const int c1 = child_ptr[static_cast<std::size_t>(i) + 1];
                for (int c = c0 + 1; c < c1; ++c)
                  release(child[static_cast<std::size_t>(c)]);
                return c0 < c1 ? child[static_cast<std::size_t>(c0)] : -1;
              });

      std::atomic<std::size_t> next{0};
      const auto subtrees = [&](int tid) {
        auto &sc = ws.threads[static_cast<std::size_t>(tid)];
        for (;;) {
          const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= nsub)
            break;
          const auto [b, e] = fkeep.solve_subtrees[i];
          for (auto k = static_cast<std::size_t>(e); k-- > static_cast<std::size_t>(b);) {
            const double *L;
            const Int *rows;
            int p, f;
            step_front(k, L, rows, p, f);
            backward_front(L, rows, p, f, posdef, apply_d, xc, sc.loc, sc.ext);
          }
        }
      };
      pool.run(subtrees);
    } else {
      // Reverse-postorder: roots first, then toward leaves
      for (int pos = ns - 1; pos >= 0; --pos) {
        const Int *rows = nullptr;
        const double *factor_data = nullptr;
        int p = 0;
        int f = 0;
        if (use_steps) {
          step_front(static_cast<std::size_t>(pos), factor_data, rows, p, f);
        } else {
          const Int s = (*postorder)[static_cast<std::size_t>(pos)];
          const Supernode &sn = ak->supernodes[static_cast<std::size_t>(s)];
          rows = ak->front_rows(s).data();
          factor_data = fkeep.factor_values.data() +
                        fkeep.factor_col_ptr[static_cast<std::size_t>(s)];
          p = sn.width();
          f = ak->front_size(s);
        }
        backward_front(factor_data, rows, p, f, posdef, apply_d, xc, b_loc, b_ext);
      }
    }

    // Apply fill-reducing permutation: x_original[perm[j]] = x_permuted[j]
    if (permute) {
      for (int j = 0; j < n; ++j)
        tmp[static_cast<std::size_t>(fkeep.perm[static_cast<std::size_t>(j)])] =
            xc[j];
      for (int j = 0; j < n; ++j)
        xc[j] = tmp[static_cast<std::size_t>(j)];
    }
  }
}

void solve_backward(const FactorKeep &fkeep, double *x, int n, int nrhs) {
  SolveWorkspace ws;
  solve_backward(fkeep, x, n, nrhs, ws);
}

} // namespace smf
