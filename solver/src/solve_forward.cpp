#include "smf/solve_forward.hpp"

#include "smf/analysis.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/solve_pool.hpp"
#include "smf/supernode.hpp"
#include "smf/workspace.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <type_traits>
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

// Updates to the extended rows of a front: x[r] -= v. In a parallel
// subtree, rows the top part pivots go to the thread's buffer instead
// (applied before the top is solved), since sibling subtrees share them.
struct DirectUpdate {
  double *x;
  void operator()(Int r, double v) const { x[r] -= v; }
};

struct SplitUpdate {
  double *x;
  const Int *top_slot;
  double *top;
  void operator()(Int r, double v) const {
    const Int s = top_slot[r];
    if (s < 0)
      x[r] -= v;
    else
      top[s] += v;
  }
};

// In the top part's DAG, sibling fronts run concurrently and may share
// ancestor rows: updates are atomic adds (rarely contended).
struct AtomicUpdate {
  double *x;
  void operator()(Int r, double v) const {
    std::atomic_ref<double>(x[r]).fetch_sub(v, std::memory_order_relaxed);
  }
};

// One front of L^{-1}: pivot rows rows[0, p), extended rows rows[p, f),
// L stored column-major with leading dimension f.
template <class Update>
void forward_front(const double *L, const Int *rows, int p, int f, bool posdef,
                   double *xcol, std::vector<double> &b_loc,
                   std::vector<double> &b_ext, Update upd) {
  const int q = f - p; // extended rows
  if (p <= 0)
    return;

  if (p == 1) {
    const Int row0 = rows[0];
    double x0 = xcol[row0];
    if (posdef)
      x0 /= L[0];
    xcol[row0] = x0;
    for (int k = 1; k < f; ++k)
      upd(rows[k], L[k] * x0);
    return;
  }

  if (p == 2) {
    const Int row0 = rows[0];
    const Int row1 = rows[1];
    double x0 = xcol[row0];
    if (posdef)
      x0 /= L[0];
    double x1 = xcol[row1] - L[1] * x0;
    if (posdef)
      x1 /= L[static_cast<std::ptrdiff_t>(f) + 1];
    xcol[row0] = x0;
    xcol[row1] = x1;
    const double *col0 = L;
    const double *col1 = L + static_cast<std::ptrdiff_t>(f);
    for (int k = 2; k < f; ++k)
      upd(rows[k], col0[k] * x0 + col1[k] * x1);
    return;
  }

  // Small fronts, direct updates: gather all f rows once and eliminate
  // column by column over the contiguous vector (pivot and extended rows
  // alike), then scatter once. One loop nest instead of the triangular
  // kernel, the L21 product and the update pass (robot_arm's fronts: f <= 27).
  if constexpr (std::is_same_v<Update, DirectUpdate>) {
    if (p * q <= 4096 && p <= 16) {
      if (b_ext.size() < static_cast<std::size_t>(f))
        b_ext.resize(static_cast<std::size_t>(f));
      double *w = b_ext.data();
      for (int i = 0; i < f; ++i)
        w[i] = xcol[rows[i]];
      for (int j = 0; j < p; ++j) {
        const double *col = L + static_cast<std::ptrdiff_t>(j) * f;
        if (posdef)
          w[j] /= col[j];
        const double xj = w[j];
        for (int i = j + 1; i < f; ++i)
          w[i] -= col[i] * xj;
      }
      for (int i = 0; i < f; ++i)
        xcol[rows[i]] = w[i];
      return;
    }
  }

  // (a) Gather pivot values: b_loc[k] = xcol[ row_indices[k] ]
  if (b_loc.size() < static_cast<std::size_t>(p))
    b_loc.resize(static_cast<std::size_t>(p));
  for (int k = 0; k < p; ++k)
    b_loc[static_cast<std::size_t>(k)] = xcol[rows[k]];

  // (b) Solve L11 * y = b_loc. Most captured KKT fronts are tiny; direct
  // kernels avoid thousands of CBLAS calls while preserving BLAS for larger
  // dense fronts.
  if (posdef) {
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

  // (d) Extended rows: t = L21 * b_loc, accumulated contiguously and
  // applied once per row.
  if (q > 0) {
    if (b_ext.size() < static_cast<std::size_t>(q))
      b_ext.resize(static_cast<std::size_t>(q));
    double *t = b_ext.data();
    if (p * q <= 4096) {
      {
        const double x0 = b_loc[0];
        const double *col = L + p;
        for (int k = 0; k < q; ++k)
          t[k] = col[k] * x0;
      }
      for (int j = 1; j < p; ++j) {
        const double xj = b_loc[static_cast<std::size_t>(j)];
        const double *col = L + static_cast<std::ptrdiff_t>(j) * f + p;
        for (int k = 0; k < q; ++k)
          t[k] += col[k] * xj;
      }
    } else {
      smf_dgemv(t, q, p, L + p, f, b_loc.data(), 1.0, 0.0);
    }
    for (int k = 0; k < q; ++k)
      upd(rows[p + k], t[k]);
  }
}

} // anonymous namespace

void solve_forward(const FactorKeep &fkeep, double *x, int n, int nrhs,
                   SolveWorkspace &ws, int num_threads, bool permute) {
  if (n <= 0 || nrhs <= 0 || x == nullptr)
    return;

  const std::vector<int> &iperm = fkeep.iperm;
  const bool use_steps = !fkeep.solve_steps.empty();
  const AnalysisKeep *ak = fkeep.analysis;
  if (!use_steps && ak == nullptr)
    return;
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
  const bool posdef = fkeep.is_posdef;
  const auto step_front = [&](std::size_t k, const double *&L, const Int *&rows,
                              int &p, int &f) {
    const SolveStep &step = fkeep.solve_steps[k];
    rows = fkeep.solve_row_indices.data() + step.row_offset;
    L = fkeep.factor_values.data() + step.factor_offset;
    p = step.p;
    f = step.f;
  };

  // Parallel layout from a tree-parallel factorization: independent
  // subtrees first, then the top of the tree.
  const std::size_t nsub = fkeep.solve_subtrees.size();
  const int threads =
      (use_steps && num_threads > 1 && nsub >= 2 &&
       fkeep.solve_top_slot.size() == static_cast<std::size_t>(n) &&
       static_cast<LongInt>(fkeep.solve_top_parent.size()) ==
           static_cast<LongInt>(fkeep.solve_steps.size()) - fkeep.solve_top_begin)
          ? std::min<int>(num_threads, static_cast<int>(nsub))
          : 1;

  std::vector<double> &tmp = ws.tmp;
  std::vector<double> &b_loc = ws.loc;
  std::vector<double> &b_ext = ws.ext;
  tmp.resize(static_cast<std::size_t>(n));

  for (int rhs = 0; rhs < nrhs; ++rhs) {
    double *xcol = x + rhs * n;

    // Step 1: Apply P^T permutation: xcol_perm[iperm[j]] = xcol[j]
    if (permute) {
      for (int j = 0; j < n; ++j)
        tmp[static_cast<std::size_t>(iperm[static_cast<std::size_t>(j)])] =
            xcol[j];
      std::copy(tmp.begin(), tmp.end(), xcol);
    }

    // Step 2: Forward substitution — process supernodes in postorder
    if (threads > 1) {
      SolvePool &pool = ws.ensure_pool(threads);
      BlasSerialGuard serial_blas; // the caller is thread 0
      const std::size_t ntop = fkeep.solve_top_rows.size();
      std::atomic<std::size_t> next{0};
      const auto subtrees = [&](int tid) {
        auto &sc = ws.threads[static_cast<std::size_t>(tid)];
        sc.top.assign(ntop, 0.0);
        const SplitUpdate upd{xcol, fkeep.solve_top_slot.data(), sc.top.data()};
        for (;;) {
          const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= nsub)
            break;
          const auto [b, e] = fkeep.solve_subtrees[i];
          for (auto k = static_cast<std::size_t>(b); k < static_cast<std::size_t>(e); ++k) {
            const double *L;
            const Int *rows;
            int p, f;
            step_front(k, L, rows, p, f);
            forward_front(L, rows, p, f, posdef, xcol, sc.loc, sc.ext, upd);
          }
        }
      };
      pool.run(subtrees);

      // Subtree updates to top rows, summed over the threads' buffers in
      // contiguous slices.
      pool.run([&](int tid) {
        const std::size_t lo = ntop * static_cast<std::size_t>(tid) / static_cast<std::size_t>(threads);
        const std::size_t hi = ntop * static_cast<std::size_t>(tid + 1) / static_cast<std::size_t>(threads);
        for (std::size_t s = lo; s < hi; ++s) {
          double v = 0.0;
          for (int t = 0; t < threads; ++t)
            v += ws.threads[static_cast<std::size_t>(t)].top[s];
          xcol[fkeep.solve_top_rows[s]] -= v;
        }
      });

      // The top of the tree as a DAG: a front runs once its children are
      // done (all updates to its pivot rows are then in).
      const auto top_begin = static_cast<std::size_t>(fkeep.solve_top_begin);
      const std::size_t nts = fkeep.solve_steps.size() - top_begin;
      std::vector<std::atomic<int>> pending(nts);
      for (auto &c : pending)
        c.store(0, std::memory_order_relaxed);
      for (std::size_t i = 0; i < nts; ++i)
        if (fkeep.solve_top_parent[i] >= 0)
          pending[static_cast<std::size_t>(fkeep.solve_top_parent[i])].fetch_add(
              1, std::memory_order_relaxed);
      std::vector<int> ready;
      for (std::size_t i = 0; i < nts; ++i)
        if (pending[i].load(std::memory_order_relaxed) == 0)
          ready.push_back(static_cast<int>(i));
      run_dag(pool, static_cast<int>(nts), std::move(ready),
              [&](int i, int tid, const auto & /*release*/) {
                auto &sc = ws.threads[static_cast<std::size_t>(tid)];
                const double *L;
                const Int *rows;
                int p, f;
                step_front(top_begin + static_cast<std::size_t>(i), L, rows, p, f);
                forward_front(L, rows, p, f, posdef, xcol, sc.loc, sc.ext, AtomicUpdate{xcol});
                const Int par = fkeep.solve_top_parent[static_cast<std::size_t>(i)];
                // The last child to finish runs the parent itself.
                if (par >= 0 &&
                    pending[static_cast<std::size_t>(par)].fetch_sub(1, std::memory_order_acq_rel) == 1)
                  return static_cast<int>(par);
                return -1;
              });
      continue;
    }

    for (Int pos = 0; pos < ns; ++pos) {
      const Int *rows = nullptr;
      const double *L = nullptr;
      int p = 0;
      int f = 0;
      if (use_steps) {
        step_front(static_cast<std::size_t>(pos), L, rows, p, f);
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
      forward_front(L, rows, p, f, posdef, xcol, b_loc, b_ext, DirectUpdate{xcol});
    }
  }
}

void solve_forward(const FactorKeep &fkeep, double *x, int n, int nrhs) {
  SolveWorkspace ws;
  solve_forward(fkeep, x, n, nrhs, ws);
}

} // namespace smf
