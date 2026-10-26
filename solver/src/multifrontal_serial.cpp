#include "smf/multifrontal_serial.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/ldlt_kernel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace smf {

// Per-thread output and scratch of the tree-parallel path, kept in the
// NumericWorkspace so refactorizations reuse it.
struct TreeParallelBuffers {
  // Reusable, uninitialised contribution-block buffers for the top of the
  // tree: a fresh zero-filled multi-MB allocation per front cost more than
  // the front itself (page faults), so a parent hands its children's
  // buffers back as soon as it has assembled them.
  struct BlockPool {
    std::mutex mtx;
    std::vector<std::pair<std::size_t, std::unique_ptr<double[]>>> free;
    std::unique_ptr<double[]> acquire(std::size_t need, std::size_t &cap) {
      {
        std::lock_guard<std::mutex> lk(mtx);
        std::size_t best = free.size();
        for (std::size_t i = 0; i < free.size(); ++i)
          if (free[i].first >= need && (best == free.size() || free[i].first < free[best].first))
            best = i;
        if (best < free.size()) {
          cap = free[best].first;
          auto buf = std::move(free[best].second);
          free.erase(free.begin() + static_cast<std::ptrdiff_t>(best));
          return buf;
        }
      }
      cap = need;
      return std::unique_ptr<double[]>(new double[need]);
    }
    void release(std::unique_ptr<double[]> buf, std::size_t cap) {
      if (!buf) return;
      std::lock_guard<std::mutex> lk(mtx);
      free.emplace_back(cap, std::move(buf));
    }
  };
  BlockPool pool;
  struct Thread {
    std::vector<double> fv, cbv, work;
    std::vector<Int> sri, cbr, pos, rows, cmap;
    std::vector<int8_t> tags;
    std::vector<SolveStep> steps;
    std::vector<DiagSolveEntry> diag;
  };
  std::vector<Thread> threads;
};

namespace {

// One pass over a sequence of fronts: where it writes and its scratch. The
// serial path is a single pass bound to the FactorKeep and the workspace;
// the tree-parallel path runs one pass per thread over whole subtrees, then
// the serial pass over the top of the tree.
struct FrontPass {
  std::vector<double> *fv;   // factor values; fronts are assembled in place here
  std::vector<Int> *sri;     // solve row indices
  std::vector<SolveStep> *steps;
  std::vector<DiagSolveEntry> *diag;
  std::vector<double> *cbv;  // contribution stack: packed lower blocks
  std::vector<Int> *cbr;     // contribution stack: row ids
  std::vector<Int> *rows;
  std::vector<Int> *cmap;
  std::vector<int8_t> *tags;
  std::vector<double> *work;
  Int *pos;                  // row -> front position, -1 between fronts
  std::size_t fo = 0, ro = 0, cv_top = 0, cr_top = 0;
  SerialFactorStats st;
  int64_t growths = 0;
};

struct NodeBlock {
  std::unique_ptr<double[]> values;
  std::size_t cap = 0;
  std::vector<Int> rows;
};

struct FactorContext {
  const AnalysisKeep &keep;
  const Int *acp;
  const Int *ari;
  const double *av;
  const LdltKernelParams &prm;
  bool posdef;
  FactorKeep &fkeep;
  NumericWorkspace &ws;
  FrontPass *const *passes; // indexed by ws.cb_owner (>= 0)
  int *front_pass;          // per supernode: the pass that factored it
  // Contribution blocks kept per node (ws.cb_owner == -1): used where
  // stacks are not popped in order, so that no reader's block can move.
  std::vector<NodeBlock> *node_blocks;
  TreeParallelBuffers::BlockPool *pool;
};

// Size-as-high-water-mark growth: only the excess beyond the previous size
// is value-initialised, and capacity grows geometrically.
template <class Vec> void grow(Vec &vec, std::size_t need, int64_t &growths) {
  if (vec.size() >= need) return;
  if (vec.capacity() < need) {
    ++growths;
    vec.reserve(std::max(need, vec.capacity() + vec.capacity() / 2));
  }
  vec.resize(need);
}

// Assemble, factor and store supernode s with pass P (index `me` in
// cx.passes). Children's contribution blocks are read from whichever pass
// produced them; with `pop` (postorder passes, where they are the top of the
// stack) blocks on P's own stack are popped.
ErrorCode process_front(const FactorContext &cx, FrontPass &P, int me, Int s, bool pop) {
  const AnalysisKeep &keep = cx.keep;
  NumericWorkspace &ws = cx.ws;
  FactorKeep &fkeep = cx.fkeep;
  const std::size_t si = static_cast<std::size_t>(s);
  const Supernode &sn = keep.supernodes[si];
  const FrontalInfo &fi = keep.fronts[si];
  const int p_sym = static_cast<int>(sn.width());
  const int f_sym = static_cast<int>(fi.front_size());
  Int *const pos = P.pos;
  cx.front_pass[si] = me;

  const auto child_rows = [&](std::size_t ci) -> const Int * {
    if (ws.cb_owner[ci] < 0) return (*cx.node_blocks)[ci].rows.data();
    return cx.passes[ws.cb_owner[ci]]->cbr->data() + ws.cb_row_off[ci];
  };
  const auto child_values = [&](std::size_t ci) -> const double * {
    if (ws.cb_owner[ci] < 0) return (*cx.node_blocks)[ci].values.get();
    return cx.passes[ws.cb_owner[ci]]->cbv->data() + ws.cb_value_off[ci];
  };

  // Front rows: delayed columns from children, own pivots, then the
  // symbolic extension rows.
  int nd = 0;
  for (const Int c : sn.children) nd += ws.cb_delayed[static_cast<std::size_t>(c)];
  grow(*P.rows, static_cast<std::size_t>(nd + f_sym), P.growths);
  Int *rows = P.rows->data();
  int f = 0;
  if (nd > 0) {
    for (const Int c : sn.children) {
      const std::size_t ci = static_cast<std::size_t>(c);
      const Int *cr = child_rows(ci);
      for (Int t = 0; t < ws.cb_delayed[ci]; ++t) {
        rows[f] = cr[t];
        pos[cr[t]] = f++;
      }
    }
  }
  for (int t = 0; t < f_sym; ++t) {
    const Int r = fi.row_indices[static_cast<std::size_t>(t)];
    rows[f] = r;
    pos[r] = f++;
  }
  const int num_elig = nd + p_sym;
  // Defensive: a child row outside the symbolic structure joins the front.
  for (const Int c : sn.children) {
    const std::size_t ci = static_cast<std::size_t>(c);
    const Int *cr = child_rows(ci);
    for (Int t = ws.cb_delayed[ci]; t < ws.cb_size[ci]; ++t) {
      if (pos[cr[t]] < 0) {
        grow(*P.rows, static_cast<std::size_t>(f) + 1, P.growths);
        rows = P.rows->data();
        rows[f] = cr[t];
        pos[cr[t]] = f++;
      }
    }
  }

  const std::size_t sf = static_cast<std::size_t>(f);
  // The front is assembled in place at its factor offset: after the
  // kernel, columns [0, p) are already L in the solve layout, and the
  // trailing block is dead once pushed to the stack (the next front
  // starts at fo + f * p and reuses it).
  grow(*P.fv, P.fo + sf * sf, P.growths);
  grow(*P.cmap, sf, P.growths);
  grow(*P.tags, sf, P.growths);
  grow(*P.work, ldlt_work_size(f, cx.prm), P.growths);
  double *F = P.fv->data() + P.fo;
  for (std::size_t j = 0; j < sf; ++j) {
    std::fill_n(F + j * sf + j, sf - j, 0.0);
  }

  // Original entries of the pivot columns (lower CSC, so pos[row] >= pos[col]).
  for (Int j = sn.col_start; j < sn.col_end; ++j) {
    double *Fj = F + static_cast<std::size_t>(pos[j]) * sf;
    for (Int k = cx.acp[j]; k < cx.acp[j + 1]; ++k) {
      Fj[pos[cx.ari[k]]] += cx.av[k];
    }
  }

  // Children's contribution blocks (the top of P's stack when P made them).
  std::size_t new_cv_top = P.cv_top;
  std::size_t new_cr_top = P.cr_top;
  Int *cm = P.cmap->data();
  for (const Int c : sn.children) {
    const std::size_t ci = static_cast<std::size_t>(c);
    const Int q = ws.cb_size[ci];
    if (q <= 0) continue;
    const Int *cr = child_rows(ci);
    const double *cv = child_values(ci);
    bool monotone = true;
    Int prev = -1;
    for (Int t = 0; t < q; ++t) {
      const Int l = pos[cr[t]];
      cm[t] = l;
      monotone = monotone && (l > prev);
      prev = l;
    }
    if (monotone) {
      for (Int jj = 0; jj < q; ++jj) {
        double *Fc = F + static_cast<std::size_t>(cm[jj]) * sf;
        for (Int ii = jj; ii < q; ++ii) Fc[cm[ii]] += *cv++;
      }
    } else {
      for (Int jj = 0; jj < q; ++jj) {
        const std::size_t lj = static_cast<std::size_t>(cm[jj]);
        for (Int ii = jj; ii < q; ++ii) {
          const std::size_t li = static_cast<std::size_t>(cm[ii]);
          if (li >= lj) F[lj * sf + li] += *cv++;
          else F[li * sf + lj] += *cv++;
        }
      }
    }
    if (pop && ws.cb_owner[ci] == me) {
      new_cv_top = std::min(new_cv_top, static_cast<std::size_t>(ws.cb_value_off[ci]));
      new_cr_top = std::min(new_cr_top, static_cast<std::size_t>(ws.cb_row_off[ci]));
    }
  }
  P.cv_top = new_cv_top;
  P.cr_top = new_cr_top;
  if (!pop) {
    for (const Int c : sn.children) {
      const std::size_t ci = static_cast<std::size_t>(c);
      if (ws.cb_size[ci] > 0 && ws.cb_owner[ci] < 0) {
        NodeBlock &nb = (*cx.node_blocks)[ci];
        cx.pool->release(std::move(nb.values), nb.cap);
      }
    }
  }

  int8_t *tags = P.tags->data();
  const LdltKernelResult kr =
      ldlt_factor_front(F, f, num_elig, rows, cx.prm, tags, P.work->data());
  const int p = kr.num_accepted;
  SerialFactorStats &st = P.st;
  st.inertia.positive += kr.inertia.positive;
  st.inertia.negative += kr.inertia.negative;
  st.inertia.zero += kr.inertia.zero;
  st.static_pivots += kr.static_pivots;
  st.flops += kr.flops;
  st.max_front = std::max(st.max_front, f);

  if (cx.posdef && p < num_elig) {
    for (int t = 0; t < f; ++t) pos[rows[t]] = -1;
    return ErrorCode::NotPositiveDefinite;
  }

  // A root has no parent to delay to: unaccepted columns are zero pivots.
  const bool is_root = (sn.parent == -1);
  int p_store = p;
  if (is_root && p < num_elig) {
    for (int c = p; c < num_elig; ++c) tags[c] = 0;
    st.inertia.zero += num_elig - p;
    p_store = num_elig;
  }

  // Overflow anywhere in a front propagates into a later pivot of the same
  // or an ancestor front, so checking every D entry catches it without a
  // per-entry test elsewhere. Only the pivot blocks are rewritten: D is
  // read out first, then the strict upper part is zeroed and the unit (or,
  // for SPD, sqrt(d)) diagonal written.
  bool finite_d = true;
  if (cx.posdef) {
    for (int c = 0; c < p_store; ++c) {
      double *Lc = F + static_cast<std::size_t>(c) * sf;
      const double sd = std::sqrt(Lc[c]);
      std::fill_n(Lc, c, 0.0);
      Lc[c] = sd;
      for (int i = c + 1; i < f; ++i) Lc[i] *= sd;
    }
  } else {
    // L is unit lower, so its diagonal slots keep D (FactorKeep::inline_d):
    // d on the diagonal, a 2x2 block's off-diagonal in the strictly upper
    // slot (c, c + 1), which no triangular kernel reads.
    double d10 = 0.0;
    for (int c = 0; c < p_store; ++c) {
      double *Lc = F + static_cast<std::size_t>(c) * sf;
      const int8_t tag = tags[c];
      if (tag == 0) {
        P.diag->push_back(DiagSolveEntry{1, 1, rows[c], 0, 0.0, 0.0, 0.0});
        std::fill_n(Lc + c + 1, f - c - 1, 0.0);
        Lc[c] = 0.0;
      } else if (tag == 2) {
        P.diag->push_back(DiagSolveEntry{
            2, 0, rows[c], rows[c + 1], Lc[c], Lc[c + 1], Lc[sf + c + 1]});
        finite_d = finite_d && is_finite_fast(Lc[c]) && is_finite_fast(Lc[c + 1]) &&
                   is_finite_fast(Lc[sf + c + 1]);
        d10 = Lc[c + 1];
        Lc[c + 1] = 0.0;
      } else if (tag == 1) {
        P.diag->push_back(DiagSolveEntry{1, 0, rows[c], 0, Lc[c], 0.0, 0.0});
        finite_d = finite_d && is_finite_fast(Lc[c]);
      }
      std::fill_n(Lc, c, 0.0);
      if (tag == -1) Lc[c - 1] = d10;
    }
    fkeep.pivot_types[si].assign(tags, tags + p_store);
  }
  fkeep.factor_col_ptr[si] = static_cast<LongInt>(P.fo);
  SolveStep step;
  step.row_offset = static_cast<LongInt>(P.ro);
  step.factor_offset = static_cast<LongInt>(P.fo);
  step.p = p_store;
  step.f = f;
  P.steps->push_back(step);
  grow(*P.sri, P.ro + sf, P.growths);
  std::copy(rows, rows + f, P.sri->begin() + static_cast<std::ptrdiff_t>(P.ro));
  P.fo += sf * static_cast<std::size_t>(p_store);
  P.ro += sf;

  // Push the Schur complement (delayed columns first) for the parent: on
  // P's stack, or into its own node buffer when stacks are not popped.
  const int q = f - p;
  if (!is_root && q > 0) {
    const std::size_t sq = static_cast<std::size_t>(q);
    double *dst;
    Int *dst_rows;
    if (pop) {
      grow(*P.cbv, P.cv_top + sq * (sq + 1) / 2, P.growths);
      grow(*P.cbr, P.cr_top + sq, P.growths);
      dst = P.cbv->data() + P.cv_top;
      dst_rows = P.cbr->data() + P.cr_top;
      ws.cb_owner[si] = me;
      ws.cb_value_off[si] = static_cast<LongInt>(P.cv_top);
      ws.cb_row_off[si] = static_cast<LongInt>(P.cr_top);
      P.cv_top += sq * (sq + 1) / 2;
      P.cr_top += sq;
    } else {
      NodeBlock &nb = (*cx.node_blocks)[si];
      nb.values = cx.pool->acquire(sq * (sq + 1) / 2, nb.cap);
      nb.rows.resize(sq);
      dst = nb.values.get();
      dst_rows = nb.rows.data();
      ws.cb_owner[si] = -1;
      ws.cb_value_off[si] = 0;
      ws.cb_row_off[si] = 0;
    }
    for (int jj = 0; jj < q; ++jj) {
      const double *src = F + static_cast<std::size_t>(p + jj) * sf + (p + jj);
      std::copy(src, src + (q - jj), dst);
      dst += q - jj;
    }
    std::copy(rows + p, rows + f, dst_rows);
    ws.cb_size[si] = q;
    ws.cb_delayed[si] = num_elig - p;
    st.delayed += num_elig - p;
  } else {
    ws.cb_size[si] = 0;
    ws.cb_delayed[si] = 0;
  }

  for (int t = 0; t < f; ++t) pos[rows[t]] = -1;
  return finite_d ? ErrorCode::Success : ErrorCode::IllegalValue;
}

void add_stats(SerialFactorStats &to, const SerialFactorStats &from) {
  to.inertia.positive += from.inertia.positive;
  to.inertia.negative += from.inertia.negative;
  to.inertia.zero += from.inertia.zero;
  to.delayed += from.delayed;
  to.static_pivots += from.static_pivots;
  to.flops += from.flops;
  to.max_front = std::max(to.max_front, from.max_front);
}

// Predicted flops of supernode si's front, the term Info::predicted_flops
// sums (assembly_tree.cpp): p f^2 for the symbolic pivot count and size.
double predicted_front_flops(const AnalysisKeep &keep, std::size_t si) {
  const double f = static_cast<double>(keep.fronts[si].front_size());
  const double p = static_cast<double>(keep.supernodes[si].width());
  return p * f * f;
}

// Below this much front work (sum of f^2 p) thread start-up is not repaid.
constexpr double kTreeParallelMinWork = 2e7;

// Tree parallelism: split the assembly tree into independent subtrees of
// bounded work (largest-first, the heaviest subtree root moves to the top
// part until every remaining subtree is below total / (4 threads)). Returns
// the subtrees as [first, last] postorder index ranges, heaviest first, and
// marks the top part. Leaves too heavy to split stay in the top part, where
// BLAS is threaded. Returns nothing when the tree is too light to share.
std::vector<std::pair<std::size_t, std::size_t>>
partition_subtrees(const AnalysisKeep &keep, int threads, std::vector<std::uint8_t> &top) {
  const std::size_t ns = keep.supernodes.size();
  std::vector<double> work(ns, 0.0);
  std::vector<std::size_t> first(ns), at(ns);
  for (std::size_t k = 0; k < keep.postorder.size(); ++k) {
    const std::size_t si = static_cast<std::size_t>(keep.postorder[k]);
    const Supernode &sn = keep.supernodes[si];
    const double f = static_cast<double>(keep.fronts[si].row_indices.size());
    work[si] = f * f * static_cast<double>(sn.width());
    first[si] = k;
    at[si] = k;
    for (const Int c : sn.children) {
      const std::size_t ci = static_cast<std::size_t>(c);
      work[si] += work[ci];
      first[si] = std::min(first[si], first[ci]);
    }
  }
  double total = 0.0;
  using Item = std::pair<double, Int>;
  std::priority_queue<Item> heap;
  for (std::size_t s = 0; s < ns; ++s) {
    if (keep.supernodes[s].parent < 0) {
      total += work[s];
      heap.emplace(work[s], static_cast<Int>(s));
    }
  }
  top.assign(ns, 0);
  if (total < kTreeParallelMinWork) return {};
  const double target = total / (4.0 * threads);
  while (!heap.empty() && heap.top().first > target) {
    const Int s = heap.top().second;
    heap.pop();
    top[static_cast<std::size_t>(s)] = 1;
    for (const Int c : keep.supernodes[static_cast<std::size_t>(s)].children)
      heap.emplace(work[static_cast<std::size_t>(c)], c);
  }
  std::vector<std::pair<std::size_t, std::size_t>> subtrees;
  while (!heap.empty()) {
    const std::size_t s = static_cast<std::size_t>(heap.top().second);
    heap.pop();
    subtrees.emplace_back(first[s], at[s]);
  }
  return subtrees;
}

} // namespace

ErrorCode multifrontal_factor_serial(const AnalysisKeep &keep, const Int *acp,
                                     const Int *ari, const double *av,
                                     const Control &ctrl, double eff_small,
                                     double static_pivot, bool posdef,
                                     FactorKeep &fkeep,
                                     NumericWorkspace &ws, SerialFactorStats &st) {
  const std::size_t n = static_cast<std::size_t>(keep.n);
  const std::size_t ns = keep.supernodes.size();

  // `plan` sizes from the symbolic structure are not counted as growth.
  auto reserve_plan = [](auto &vec, std::size_t need) {
    if (vec.size() < need) vec.resize(need);
  };

  LdltKernelParams prm;
  prm.u = ctrl.pivot_u;
  prm.eff_small = eff_small;
  prm.small_pivot = ctrl.small_pivot;
  prm.nb = std::max(2, ctrl.indef_panel_size);
  prm.posdef = posdef;
  prm.static_pivot = posdef ? 0.0 : static_pivot;

  // Early abort once the negative pivots exceed the limit (indefinite only).
  // Pivots are only ever added, so a run whose final count is within the
  // limit never trips it and does exactly the work of an unarmed run.
  const int neg_limit = posdef ? -1 : ctrl.max_negative_pivots;
  const bool armed = neg_limit >= 0;

  // Symbolic plan: largest front, factor size, and the peak of the
  // contribution stack when no pivot is delayed.
  std::size_t max_f = 1;
  std::size_t pred_factor = 0;
  std::size_t pred_rows = 0;
  ws.cb_value_off.resize(ns);
  ws.cb_row_off.resize(ns);
  ws.cb_size.resize(ns);
  ws.cb_delayed.resize(ns);
  ws.cb_owner.assign(ns, 0);
  {
    std::size_t vtop = 0, rtop = 0, vpeak = 0, rpeak = 0;
    for (const Int s : keep.postorder) {
      const std::size_t si = static_cast<std::size_t>(s);
      const Supernode &sn = keep.supernodes[si];
      const std::size_t f = keep.fronts[si].row_indices.size();
      const std::size_t p = static_cast<std::size_t>(sn.width());
      max_f = std::max(max_f, f);
      pred_factor += f * p;
      pred_rows += f;
      for (const Int c : sn.children) {
        const std::size_t ci = static_cast<std::size_t>(c);
        if (ws.cb_size[ci] > 0) {
          vtop = std::min(vtop, static_cast<std::size_t>(ws.cb_value_off[ci]));
          rtop = std::min(rtop, static_cast<std::size_t>(ws.cb_row_off[ci]));
        }
      }
      const std::size_t q = f - p;
      if (sn.parent >= 0 && q > 0) {
        ws.cb_value_off[si] = static_cast<LongInt>(vtop);
        ws.cb_row_off[si] = static_cast<LongInt>(rtop);
        ws.cb_size[si] = static_cast<Int>(q);
        vtop += q * (q + 1) / 2;
        rtop += q;
        vpeak = std::max(vpeak, vtop);
        rpeak = std::max(rpeak, rtop);
      } else {
        ws.cb_size[si] = 0;
      }
    }
    reserve_plan(ws.cb_values, vpeak);
    reserve_plan(ws.cb_rows, rpeak);
  }

  if (ws.row_pos.size() != n) ws.row_pos.assign(n, -1);
  reserve_plan(ws.front_rows, max_f);
  reserve_plan(ws.child_map, max_f);
  reserve_plan(ws.pivot_tags, max_f);
  reserve_plan(ws.kernel_work, ldlt_work_size(static_cast<int>(max_f), prm));

  auto &fv = fkeep.factor_values;
  auto &sri = fkeep.solve_row_indices;
  reserve_plan(fv, pred_factor);
  reserve_plan(sri, pred_rows);
  fkeep.factor_col_ptr.resize(ns + 1);
  if (!posdef) fkeep.pivot_types.resize(ns);
  fkeep.solve_steps.clear();
  fkeep.solve_steps.reserve(ns);
  fkeep.solve_subtrees.clear();
  fkeep.inline_d = !posdef;
  fkeep.solve_top_begin = 0;
  fkeep.solve_top_rows.clear();
  fkeep.solve_top_parent.clear();
  fkeep.diag_entries.clear();
  if (!posdef) fkeep.diag_entries.reserve(n);

  FrontPass main;
  main.fv = &fv;
  main.sri = &sri;
  main.steps = &fkeep.solve_steps;
  main.diag = &fkeep.diag_entries;
  main.cbv = &ws.cb_values;
  main.cbr = &ws.cb_rows;
  main.rows = &ws.front_rows;
  main.cmap = &ws.child_map;
  main.tags = &ws.pivot_tags;
  main.work = &ws.kernel_work;
  main.pos = ws.row_pos.data();

  std::vector<FrontPass *> passes{&main};
  std::vector<int> front_pass(ns, 0);
  std::vector<NodeBlock> node_blocks;
  ErrorCode status = ErrorCode::Success;

  // Tree-parallel when there are threads to use and at least two subtrees.
  // Plain std::threads, not OpenMP: MSVC's OpenMP runtime would be loaded
  // next to the ones MKL / CHOLMOD / OpenBLAS bring, which aborts.
  int threads = 1;
  std::vector<std::pair<std::size_t, std::size_t>> subtrees;
  std::vector<std::uint8_t> top;
  if (ctrl.num_threads > 1 && ns >= 2) {
    subtrees = partition_subtrees(keep, ctrl.num_threads, top);
    if (subtrees.size() >= 2)
      threads = std::min<int>(ctrl.num_threads, static_cast<int>(subtrees.size()));
  }

  if (threads <= 1) {
    const FactorContext cx{keep, acp,         ari,         av,          prm,      posdef,
                           fkeep, ws,         passes.data(), front_pass.data(), &node_blocks, nullptr};
    std::size_t k = 0;
    for (; k < keep.postorder.size(); ++k) {
      status = process_front(cx, main, 0, keep.postorder[k], true);
      if (status == ErrorCode::Success && armed && main.st.inertia.negative > neg_limit)
        status = ErrorCode::InertiaExceeded;
      if (status != ErrorCode::Success) break;
    }
    if (status == ErrorCode::InertiaExceeded)
      for (std::size_t r = k + 1; r < keep.postorder.size(); ++r)
        st.aborted_flops_skipped +=
            predicted_front_flops(keep, static_cast<std::size_t>(keep.postorder[r]));
  } else {
    if (!ws.tree_par) ws.tree_par = std::make_shared<TreeParallelBuffers>();
    auto &tb = ws.tree_par->threads;
    if (tb.size() < static_cast<std::size_t>(threads)) tb.resize(static_cast<std::size_t>(threads));
    std::vector<FrontPass> tpass(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) {
      auto &b = tb[static_cast<std::size_t>(t)];
      if (b.pos.size() != n) b.pos.assign(n, -1);
      b.steps.clear();
      b.diag.clear();
      FrontPass &P = tpass[static_cast<std::size_t>(t)];
      P.fv = &b.fv;
      P.sri = &b.sri;
      P.steps = &b.steps;
      P.diag = &b.diag;
      P.cbv = &b.cbv;
      P.cbr = &b.cbr;
      P.rows = &b.rows;
      P.cmap = &b.cmap;
      P.tags = &b.tags;
      P.work = &b.work;
      P.pos = b.pos.data();
      passes.push_back(&P);
    }
    node_blocks.resize(ns);
    const FactorContext cx{keep, acp,         ari,         av,          prm,      posdef,
                           fkeep, ws,         passes.data(), front_pass.data(), &node_blocks, &ws.tree_par->pool};
    // `failed` doubles as the stop flag: every thread polls it before each
    // front. With the limit armed, the threads add each front's negative
    // pivots to one shared count; the add that takes it past the limit
    // raises InertiaExceeded (unless another error came first).
    std::atomic<int> failed{static_cast<int>(ErrorCode::Success)};
    std::atomic<int> negatives{0};
    const auto count_negatives = [&](int added) {
      if (added <= 0 || negatives.fetch_add(added, std::memory_order_relaxed) + added <= neg_limit)
        return;
      int ok = static_cast<int>(ErrorCode::Success);
      failed.compare_exchange_strong(ok, static_cast<int>(ErrorCode::InertiaExceeded),
                                     std::memory_order_relaxed);
    };
    const auto run_pool = [threads](const auto &fn) {
      std::vector<std::thread> pool;
      pool.reserve(static_cast<std::size_t>(threads) - 1);
      for (int t = 1; t < threads; ++t) pool.emplace_back(fn, t);
      fn(0);
      for (auto &th : pool) th.join();
    };

    // Phase 1: subtrees, heaviest first, one thread each on serial BLAS.
    // Each thread's step ranges per subtree become the solve's parallel
    // layout.
    std::atomic<std::size_t> next{0};
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> subtree_steps(
        static_cast<std::size_t>(threads));
    run_pool([&](int tid) {
      BlasSerialGuard serial_blas;
      FrontPass &P = tpass[static_cast<std::size_t>(tid)];
      for (;;) {
        const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= subtrees.size() ||
            failed.load(std::memory_order_relaxed) != static_cast<int>(ErrorCode::Success))
          break;
        const auto [lo, hi] = subtrees[i];
        const std::size_t first_step = P.steps->size();
        for (std::size_t k = lo; k <= hi; ++k) {
          if (failed.load(std::memory_order_relaxed) != static_cast<int>(ErrorCode::Success))
            break;
          const int neg0 = P.st.inertia.negative;
          const ErrorCode e = process_front(cx, P, 1 + tid, keep.postorder[k], true);
          if (e != ErrorCode::Success) {
            failed.store(static_cast<int>(e), std::memory_order_relaxed);
            break;
          }
          if (armed) count_negatives(P.st.inertia.negative - neg0);
        }
        subtree_steps[static_cast<std::size_t>(tid)].emplace_back(first_step, P.steps->size());
      }
    });
    std::vector<std::size_t> phase1_steps(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t)
      phase1_steps[static_cast<std::size_t>(t)] = tpass[static_cast<std::size_t>(t)].steps->size();

    // Phase 2: the top of the tree as a task DAG. A node is ready once its
    // children are factored; ready nodes run concurrently under a budget of
    // `threads` BLAS threads in total (oversubscribing MKL's spinning teams
    // is ruinous): a front takes its share of the free threads among the
    // fronts ready with it, so sibling separators run side by side and the
    // root, alone, gets them all.
    // Stacks are no longer popped (a worker's blocks are consumed out of
    // order); the top part holds only tens of blocks.
    std::vector<Int> pending(ns, 0), ready;
    std::size_t remaining = 0;
    for (std::size_t si = 0; si < ns; ++si) {
      if (!top[si]) continue;
      ++remaining;
      for (const Int c : keep.supernodes[si].children)
        if (top[static_cast<std::size_t>(c)]) ++pending[si];
      if (pending[si] == 0) ready.push_back(static_cast<Int>(si));
    }
    struct Done {
      int pass = 0;
      std::size_t step = 0;
      Int node = -1;
    };
    std::vector<Done> done_order;
    done_order.reserve(remaining);
    std::mutex mtx;
    std::condition_variable cv;
    int free_threads = threads;
    if (failed.load() == static_cast<int>(ErrorCode::Success)) {
      run_pool([&](int tid) {
        FrontPass &P = tpass[static_cast<std::size_t>(tid)];
        for (;;) {
          Int s = -1;
          int blas_threads = 1;
          {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&] {
              return (!ready.empty() && free_threads > 0) || remaining == 0 ||
                     failed.load() != static_cast<int>(ErrorCode::Success);
            });
            if (ready.empty() || failed.load() != static_cast<int>(ErrorCode::Success)) return;
            s = ready.back();
            ready.pop_back();
            blas_threads = std::max(1, free_threads / static_cast<int>(ready.size() + 1));
            free_threads -= blas_threads;
          }
          ErrorCode e;
          const int neg0 = P.st.inertia.negative;
          {
            BlasThreadGuard blas(blas_threads);
            e = process_front(cx, P, 1 + tid, s, false);
          }
          std::lock_guard<std::mutex> lk(mtx);
          free_threads += blas_threads;
          --remaining;
          // Under the mutex, so a waiter cannot miss the stop.
          if (e == ErrorCode::Success && armed) count_negatives(P.st.inertia.negative - neg0);
          if (e != ErrorCode::Success) {
            failed.store(static_cast<int>(e));
          } else {
            done_order.push_back({tid, P.steps->size() - 1, s});
            const Int parent = keep.supernodes[static_cast<std::size_t>(s)].parent;
            if (parent >= 0 && top[static_cast<std::size_t>(parent)] &&
                --pending[static_cast<std::size_t>(parent)] == 0)
              ready.push_back(parent);
          }
          cv.notify_all();
        }
      });
    }
    status = static_cast<ErrorCode>(failed.load());
    st.parallel_subtrees = static_cast<int>(subtrees.size());

    if (status != ErrorCode::Success) {
      // Partial counts (the threads are joined). A front never started
      // still has front_pass 0: the threads' passes are numbered from 1.
      for (int t = 0; t < threads; ++t) add_stats(st, tpass[static_cast<std::size_t>(t)].st);
      if (status == ErrorCode::InertiaExceeded)
        for (std::size_t si = 0; si < ns; ++si)
          if (front_pass[si] == 0) st.aborted_flops_skipped += predicted_front_flops(keep, si);
    } else {
      // Phase 3: concatenate the per-thread factors (copied in parallel) and
      // relocate offsets. Solve order: every subtree (each complete and
      // independent), then the top in completion order, which is
      // topological since a front starts only after its children finish.
      std::vector<std::size_t> fbase(static_cast<std::size_t>(threads) + 1, 0);
      std::vector<std::size_t> rbase(static_cast<std::size_t>(threads) + 1, 0);
      for (int t = 0; t < threads; ++t) {
        const std::size_t ti = static_cast<std::size_t>(t);
        fbase[ti + 1] = fbase[ti] + tpass[ti].fo;
        rbase[ti + 1] = rbase[ti] + tpass[ti].ro;
      }
      main.fo = fbase[static_cast<std::size_t>(threads)];
      main.ro = rbase[static_cast<std::size_t>(threads)];
      grow(fv, main.fo, main.growths);
      grow(sri, main.ro, main.growths);
      run_pool([&](int tid) {
        const std::size_t ti = static_cast<std::size_t>(tid);
        const FrontPass &P = tpass[ti];
        std::copy(P.fv->begin(), P.fv->begin() + static_cast<std::ptrdiff_t>(P.fo),
                  fv.begin() + static_cast<std::ptrdiff_t>(fbase[ti]));
        std::copy(P.sri->begin(), P.sri->begin() + static_cast<std::ptrdiff_t>(P.ro),
                  sri.begin() + static_cast<std::ptrdiff_t>(rbase[ti]));
      });
      const auto relocated = [&](int t, SolveStep step) {
        step.factor_offset += static_cast<LongInt>(fbase[static_cast<std::size_t>(t)]);
        step.row_offset += static_cast<LongInt>(rbase[static_cast<std::size_t>(t)]);
        return step;
      };
      for (int t = 0; t < threads; ++t) {
        const std::size_t ti = static_cast<std::size_t>(t);
        const auto &steps = *tpass[ti].steps;
        const auto base = static_cast<LongInt>(fkeep.solve_steps.size());
        for (const auto &[b, e] : subtree_steps[ti])
          fkeep.solve_subtrees.emplace_back(base + static_cast<LongInt>(b),
                                            base + static_cast<LongInt>(e));
        for (std::size_t k = 0; k < phase1_steps[ti]; ++k)
          fkeep.solve_steps.push_back(relocated(t, steps[k]));
        fkeep.diag_entries.insert(fkeep.diag_entries.end(), tpass[ti].diag->begin(),
                                  tpass[ti].diag->end());
        add_stats(st, tpass[ti].st);
        main.growths += tpass[ti].growths;
      }
      fkeep.solve_top_begin = static_cast<LongInt>(fkeep.solve_steps.size());
      // The solve takes subtrees in this order: heaviest (stored entries)
      // first, so the last ones to start are small.
      {
        std::vector<std::pair<double, std::pair<LongInt, LongInt>>> by_work;
        by_work.reserve(fkeep.solve_subtrees.size());
        for (const auto &r : fkeep.solve_subtrees) {
          double w = 0.0;
          for (LongInt k = r.first; k < r.second; ++k) {
            const SolveStep &step = fkeep.solve_steps[static_cast<std::size_t>(k)];
            w += static_cast<double>(step.p) * step.f;
          }
          by_work.push_back({w, r});
        }
        std::sort(by_work.begin(), by_work.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });
        for (std::size_t i = 0; i < by_work.size(); ++i) fkeep.solve_subtrees[i] = by_work[i].second;
      }
      for (const Done &d : done_order)
        fkeep.solve_steps.push_back(
            relocated(d.pass, (*tpass[static_cast<std::size_t>(d.pass)].steps)[d.step]));
      // Top-part tree over solve steps (parents are top nodes too).
      {
        std::vector<Int> top_pos(ns, -1);
        for (std::size_t i = 0; i < done_order.size(); ++i)
          top_pos[static_cast<std::size_t>(done_order[i].node)] = static_cast<Int>(i);
        fkeep.solve_top_parent.resize(done_order.size());
        for (std::size_t i = 0; i < done_order.size(); ++i) {
          const Int par = keep.supernodes[static_cast<std::size_t>(done_order[i].node)].parent;
          fkeep.solve_top_parent[i] = par >= 0 ? top_pos[static_cast<std::size_t>(par)] : -1;
        }
      }
      for (std::size_t si = 0; si < ns; ++si)
        fkeep.factor_col_ptr[si] +=
            static_cast<LongInt>(fbase[static_cast<std::size_t>(front_pass[si] - 1)]);
    }
  }

  ws.capacity_growths_last_call += main.growths;
  ws.front_heap_allocations_last_call += main.growths;
  add_stats(st, main.st);

  // Leave the stack bookkeeping consistent for the next planning pass.
  if (status != ErrorCode::Success) {
    fkeep.solve_subtrees.clear();
    std::fill(ws.cb_size.begin(), ws.cb_size.end(), Int{0});
    std::fill(ws.cb_delayed.begin(), ws.cb_delayed.end(), Int{0});
    return status;
  }
  fkeep.factor_col_ptr[ns] = static_cast<LongInt>(main.fo);
  fv.resize(main.fo);
  sri.resize(main.ro);
  if (!fkeep.solve_subtrees.empty()) {
    // Pivot rows of the top part (delayed subtree pivots included): the
    // parallel forward solve sends subtree updates to them through
    // per-thread buffers.
    if (fkeep.solve_top_slot.size() != n) fkeep.solve_top_slot.assign(n, -1);
    else std::fill(fkeep.solve_top_slot.begin(), fkeep.solve_top_slot.end(), Int{-1});
    for (std::size_t k = static_cast<std::size_t>(fkeep.solve_top_begin);
         k < fkeep.solve_steps.size(); ++k) {
      const SolveStep &step = fkeep.solve_steps[k];
      const Int *rows = sri.data() + step.row_offset;
      for (int i = 0; i < step.p; ++i) {
        fkeep.solve_top_slot[static_cast<std::size_t>(rows[i])] =
            static_cast<Int>(fkeep.solve_top_rows.size());
        fkeep.solve_top_rows.push_back(rows[i]);
      }
    }
  }
  return ErrorCode::Success;
}

} // namespace smf
