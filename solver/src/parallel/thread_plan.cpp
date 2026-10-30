// Tree partition and thread-count model (see thread_plan.hpp).

#include "thread_plan.hpp"

#include "smf/ldlt_kernel.hpp"

#include <algorithm>
#include <queue>

namespace smf {

bool top_tiles_on(const Control &ctrl, bool posdef, double static_pivot, double min_abs_pivot,
                  bool det) {
  return ctrl.top_tiles && (ctrl.num_threads > 1 || det) && !posdef && static_pivot == 0.0 &&
         min_abs_pivot == 0.0 && ctrl.pivot_u >= kWeakPivotRatio;
}

TreeView tree_view(const AnalysisKeep &keep, bool tiles_on, std::vector<Int> &parent,
                   std::vector<double> &work, std::vector<std::uint8_t> &split) {
  const std::size_t ns = keep.supernodes.size();
  parent.resize(ns);
  work.resize(ns);
  split.resize(ns);
  for (std::size_t s = 0; s < ns; ++s) {
    const Int f = keep.front_size(static_cast<Int>(s));
    const double fd = static_cast<double>(f);
    parent[s] = keep.supernodes[s].parent;
    work[s] = fd * fd * static_cast<double>(keep.supernodes[s].width());
    split[s] = tiles_front(tiles_on, static_cast<int>(f)) ? 1 : 0;
  }
  return TreeView{keep.postorder, parent, work, split};
}

namespace {

// Subtree work of every node, with each subtree's [first, at] postorder
// range. Children are summed in descending postorder (AnalysisKeep::
// children' order), so the sums add up exactly as they always did: the
// delay-budget regions depend on them.
struct SubtreeSums {
  std::vector<std::size_t> cptr;  // children, CSR
  std::vector<Int> cidx;
  std::vector<double> work;
  std::vector<std::size_t> first, at;
  double total = 0.0;  // over the roots, in index order

  std::span<const Int> children(std::size_t s) const {
    return std::span<const Int>(cidx.data() + cptr[s], cptr[s + 1] - cptr[s]);
  }
};

void subtree_sums(const TreeView &t, SubtreeSums &u) {
  const std::size_t ns = t.parent.size();
  u.cptr.assign(ns + 1, 0);
  for (std::size_t s = 0; s < ns; ++s)
    if (t.parent[s] >= 0) ++u.cptr[static_cast<std::size_t>(t.parent[s]) + 1];
  for (std::size_t s = 0; s < ns; ++s) u.cptr[s + 1] += u.cptr[s];
  u.cidx.resize(u.cptr[ns]);
  {
    std::vector<std::size_t> fill(u.cptr.begin(), u.cptr.end() - 1);
    for (std::size_t k = t.postorder.size(); k-- > 0;) {
      const Int s = t.postorder[k];
      const Int p = t.parent[static_cast<std::size_t>(s)];
      if (p >= 0) u.cidx[fill[static_cast<std::size_t>(p)]++] = s;
    }
  }
  u.work.assign(ns, 0.0);
  u.first.resize(ns);
  u.at.resize(ns);
  for (std::size_t k = 0; k < t.postorder.size(); ++k) {
    const std::size_t si = static_cast<std::size_t>(t.postorder[k]);
    u.work[si] = t.work[si];
    u.first[si] = k;
    u.at[si] = k;
    for (const Int c : u.children(si)) {
      const std::size_t ci = static_cast<std::size_t>(c);
      u.work[si] += u.work[ci];
      u.first[si] = std::min(u.first[si], u.first[ci]);
    }
  }
  u.total = 0.0;
  for (std::size_t s = 0; s < ns; ++s)
    if (t.parent[s] < 0) u.total += u.work[s];
}

// The two phases of a candidate thread count, from the partition at this
// target (the one partition_subtrees makes, without its heap: subtree work
// never decreases towards the root, so the top part is exactly the nodes
// heavier than the target, and the subtrees are the other nodes whose
// parent is in the top part, or that are roots).
struct PhaseTerms {
  PartitionCount pc;
  double sub_total = 0.0;    // phase 1: work inside the subtrees
  double max_subtree = 0.0;  //   and its heaviest subtree
  double top_total = 0.0;    // phase 2: work of the top part
  double cp_top = 0.0;       //   and its heaviest path, split fronts / threads
};

void phase_terms(const TreeView &t, const std::vector<double> &sub, double target, int threads,
                 std::vector<double> &below, PhaseTerms &pt) {
  pt = PhaseTerms{};
  below.assign(t.parent.size(), 0.0);  // heaviest top path below each top node
  for (const Int s : t.postorder) {
    const std::size_t si = static_cast<std::size_t>(s);
    const Int p = t.parent[si];
    if (sub[si] > target) {
      ++pt.pc.top;
      pt.top_total += t.work[si];
      const double w = t.split[si] != 0 ? t.work[si] / threads : t.work[si];
      const double path = w + below[si];
      if (p >= 0) {
        double &b = below[static_cast<std::size_t>(p)];
        b = std::max(b, path);
      } else {
        pt.cp_top = std::max(pt.cp_top, path);
      }
    } else if (p < 0 || sub[static_cast<std::size_t>(p)] > target) {
      ++pt.pc.subtrees;
      pt.sub_total += sub[si];
      pt.max_subtree = std::max(pt.max_subtree, sub[si]);
    }
  }
}

// The candidate rule of both models (max_threads >= 2): 1, 2, 4, 8, ...
// and max_threads, cost_of(T) < 0 meaning T is no candidate; in ascending
// thread count, the first within 2% of the least cost. Returns it and its
// cost.
template <class CostOf>
std::pair<int, double> least_cost_threads(int max_threads, double cost1, CostOf cost_of) {
  std::vector<std::pair<int, double>> costs{{1, cost1}};
  const auto add = [&](int T) {
    const double cost = cost_of(T);
    if (cost >= 0.0) costs.emplace_back(T, cost);
  };
  for (int T = 2; T < max_threads; T *= 2) add(T);
  add(max_threads);
  double least = cost1;
  for (const auto &k : costs) least = std::min(least, k.second);
  for (const auto &k : costs)
    if (k.second <= least * 1.02) return k;
  return costs.front();
}

} // namespace

std::vector<std::pair<std::size_t, std::size_t>>
partition_subtrees(const TreeView &t, int threads, double min_work, std::vector<std::uint8_t> &top) {
  const std::size_t ns = t.parent.size();
  SubtreeSums u;
  subtree_sums(t, u);
  using Item = std::pair<double, Int>;
  std::priority_queue<Item> heap;
  for (std::size_t s = 0; s < ns; ++s)
    if (t.parent[s] < 0) heap.emplace(u.work[s], static_cast<Int>(s));
  top.assign(ns, 0);
  if (u.total < min_work) return {};
  const double target = u.total / (4.0 * threads);
  while (!heap.empty() && heap.top().first > target) {
    const Int s = heap.top().second;
    heap.pop();
    top[static_cast<std::size_t>(s)] = 1;
    for (const Int c : u.children(static_cast<std::size_t>(s)))
      heap.emplace(u.work[static_cast<std::size_t>(c)], c);
  }
  std::vector<std::pair<std::size_t, std::size_t>> subtrees;
  while (!heap.empty()) {
    const std::size_t s = static_cast<std::size_t>(heap.top().second);
    heap.pop();
    subtrees.emplace_back(u.first[s], u.at[s]);
  }
  return subtrees;
}

PartitionCount partition_count(const TreeView &t, int threads, double min_work) {
  SubtreeSums u;
  subtree_sums(t, u);
  if (u.total < min_work) return {};
  PhaseTerms pt;
  std::vector<double> below;
  phase_terms(t, u.work, u.total / (4.0 * threads), threads, below, pt);
  return pt.pc;
}

ThreadPlan plan_threads(const TreeView &t, int max_threads, const ThreadCosts &c) {
  ThreadPlan plan;
  if (max_threads < 2) return plan;
  SubtreeSums u;
  subtree_sums(t, u);
  const double total = u.total;
  if (total < kTreeParallelMinWork || !(total > 0.0)) return plan;

  std::vector<double> below;
  const auto [threads, cost] = least_cost_threads(max_threads, total, [&](int T) {
    PhaseTerms pt;
    phase_terms(t, u.work, total / (4.0 * T), T, below, pt);
    if (pt.pc.subtrees < 2) return -1.0;
    return std::max(pt.max_subtree, pt.sub_total / T) + std::max(pt.cp_top, pt.top_total / T) +
           T * c.per_thread + static_cast<double>(pt.pc.top) * c.per_top_front;
  });
  plan.threads = threads;
  plan.predicted_speedup = total / cost;
  return plan;
}

int plan_solve_threads(std::span<const double> subtree_work, std::span<const double> top_work,
                       std::span<const Int> top_parent, std::size_t top_rows, int max_threads,
                       const SolveCosts &c) {
  if (max_threads < 2 || subtree_work.size() < 2) return 1;
  double sub_total = 0.0, max_subtree = 0.0, top_total = 0.0;
  for (const double w : subtree_work) {
    sub_total += w;
    max_subtree = std::max(max_subtree, w);
  }
  for (const double w : top_work) top_total += w;

  // Heaviest top path, children before parents (the order of a step's
  // pending children reaching zero; top_parent need not be ascending).
  const std::size_t nt = top_work.size();
  std::vector<int> pending(nt, 0);
  for (std::size_t i = 0; i < nt; ++i)
    if (top_parent[i] >= 0) ++pending[static_cast<std::size_t>(top_parent[i])];
  std::vector<double> below(nt, 0.0);  // heaviest path below each step
  std::vector<std::size_t> ready;
  for (std::size_t i = 0; i < nt; ++i)
    if (pending[i] == 0) ready.push_back(i);
  double cp_top = 0.0;
  while (!ready.empty()) {
    const std::size_t i = ready.back();
    ready.pop_back();
    const double path = top_work[i] + below[i];
    const Int p = top_parent[i];
    if (p < 0) {
      cp_top = std::max(cp_top, path);
      continue;
    }
    const std::size_t pi = static_cast<std::size_t>(p);
    below[pi] = std::max(below[pi], path);
    if (--pending[pi] == 0) ready.push_back(pi);
  }

  const double n_top = static_cast<double>(nt);
  const double rows = static_cast<double>(top_rows);
  return least_cost_threads(max_threads, sub_total + top_total, [&](int T) {
           return std::max(max_subtree, sub_total / T) + cp_top + T * c.per_thread +
                  n_top * c.per_top_step + T * rows * c.per_top_row;
         })
      .first;
}

} // namespace smf
