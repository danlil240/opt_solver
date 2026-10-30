// Tree partition and thread-count model (see thread_plan.hpp).

#include "thread_plan.hpp"

#include "smf/ldlt_kernel.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
  double sub_total = 0.0;       // phase 1: work inside the subtrees
  double max_subtree = 0.0;     //   and its heaviest subtree
  double top_total = 0.0;       // phase 2: work of the top part
  double cp_top = 0.0;          //   its heaviest path, split fronts / threads
  std::size_t split_top = 0;    //   its split fronts
  double split_top_work = 0.0;  //   and their work
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
      if (t.split[si] != 0) {
        ++pt.split_top;
        pt.split_top_work += t.work[si];
      }
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

// The tie rule of both models over (thread count, cost) pairs in ascending
// thread count, the first T = 1 and a cost < 0 meaning no candidate: the
// first within 2% of the least cost. Returns it and its cost.
std::pair<int, double> least_cost_threads(std::span<const std::pair<int, double>> costs) {
  double least = costs.front().second;
  for (const auto &k : costs)
    if (k.second >= 0.0) least = std::min(least, k.second);
  for (const auto &k : costs)
    if (k.second >= 0.0 && k.second <= least * 1.02) return k;
  return costs.front();
}

void append_line(std::string &out, const char *line, int n) {
  if (n > 0) out.append(line, static_cast<std::size_t>(n));
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

std::vector<int> candidate_threads(int max_threads) {
  std::vector<int> out{1};
  if (max_threads < 2) return out;
  for (int T = 2; T < max_threads; T *= 2) out.push_back(T);
  out.push_back(max_threads);
  return out;
}

std::vector<FactorPlanTerms> factor_plan_terms(const TreeView &t, int max_threads) {
  SubtreeSums u;
  subtree_sums(t, u);
  const double total = u.total;
  std::vector<FactorPlanTerms> out(1);
  out[0].total = total;
  if (max_threads < 2 || total < kTreeParallelMinWork || !(total > 0.0)) return out;
  std::vector<double> below;
  for (const int T : candidate_threads(max_threads)) {
    if (T == 1) continue;
    PhaseTerms pt;
    phase_terms(t, u.work, total / (4.0 * T), T, below, pt);
    FactorPlanTerms k;
    k.threads = T;
    k.ok = pt.pc.subtrees >= 2;
    k.sub_total = pt.sub_total;
    k.max_subtree = pt.max_subtree;
    k.top_total = pt.top_total;
    k.cp_top = pt.cp_top;
    k.top_fronts = pt.pc.top;
    k.split_top_fronts = pt.split_top;
    k.split_top_work = pt.split_top_work;
    k.total = total;
    out.push_back(k);
  }
  return out;
}

double factor_plan_cost(const FactorPlanTerms &k, const ThreadCosts &c) {
  if (k.threads <= 1) return k.total;
  if (!k.ok) return -1.0;
  const int T = k.threads;
  return std::max(k.max_subtree, k.sub_total / T) + std::max(k.cp_top, k.top_total / T) +
         T * c.per_thread + static_cast<double>(k.top_fronts) * c.per_top_front;
}

std::string factor_plan_trace(std::span<const FactorPlanTerms> terms, int chosen) {
  std::string out;
  char line[512];
  for (const auto &k : terms)
    append_line(out, line,
                std::snprintf(line, sizeof line,
                              "plan-factor T=%d ok=%d sub_total=%.6g max_subtree=%.6g "
                              "top_total=%.6g cp_top=%.6g top_fronts=%d split_top_fronts=%d "
                              "split_top_work=%.6g total=%.6g\n",
                              k.threads, k.ok ? 1 : 0, k.sub_total, k.max_subtree, k.top_total,
                              k.cp_top, static_cast<int>(k.top_fronts),
                              static_cast<int>(k.split_top_fronts), k.split_top_work, k.total));
  append_line(out, line, std::snprintf(line, sizeof line, "plan-factor chosen=%d\n", chosen));
  return out;
}

bool plan_trace_enabled() {
  static const bool on = [] {
#if defined(_MSC_VER)
    char *e = nullptr;
    std::size_t len = 0;
    const bool set = _dupenv_s(&e, &len, "SMF_BENCH_PLAN_TRACE") == 0 && e != nullptr &&
                     std::atoi(e) != 0;
    std::free(e);
    return set;
#else
    const char *e = std::getenv("SMF_BENCH_PLAN_TRACE");
    return e != nullptr && std::atoi(e) != 0;
#endif
  }();
  return on;
}

ThreadPlan plan_threads(const TreeView &t, int max_threads, const ThreadCosts &c) {
  const std::vector<FactorPlanTerms> terms = factor_plan_terms(t, max_threads);
  std::vector<std::pair<int, double>> costs;
  costs.reserve(terms.size());
  for (const auto &k : terms) costs.emplace_back(k.threads, factor_plan_cost(k, c));
  const auto [threads, cost] = least_cost_threads(costs);
  ThreadPlan plan;
  plan.threads = threads;
  plan.predicted_speedup = cost > 0.0 ? terms.front().total / cost : 1.0;
  if (plan_trace_enabled()) std::fputs(factor_plan_trace(terms, threads).c_str(), stderr);
  return plan;
}

SolvePlanTerms solve_plan_terms(std::span<const double> subtree_work,
                                std::span<const double> top_work, std::span<const Int> top_parent,
                                std::size_t top_rows) {
  SolvePlanTerms k;
  k.subtrees = subtree_work.size();
  k.n_top = top_work.size();
  k.top_rows = top_rows;
  for (const double w : subtree_work) {
    k.sub_total += w;
    k.max_subtree = std::max(k.max_subtree, w);
  }
  for (const double w : top_work) k.top_total += w;

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
  while (!ready.empty()) {
    const std::size_t i = ready.back();
    ready.pop_back();
    const double path = top_work[i] + below[i];
    const Int p = top_parent[i];
    if (p < 0) {
      k.top_cp = std::max(k.top_cp, path);
      continue;
    }
    const std::size_t pi = static_cast<std::size_t>(p);
    below[pi] = std::max(below[pi], path);
    if (--pending[pi] == 0) ready.push_back(pi);
  }
  return k;
}

double solve_plan_cost(const SolvePlanTerms &k, int threads, const SolveCosts &c) {
  if (threads <= 1) return k.sub_total + k.top_total;
  const int T = threads;
  const double n_top = static_cast<double>(k.n_top);
  const double rows = static_cast<double>(k.top_rows);
  return std::max(k.max_subtree, k.sub_total / T) + k.top_cp + T * c.per_thread +
         n_top * c.per_top_step + T * rows * c.per_top_row;
}

std::string solve_plan_trace(const SolvePlanTerms &k, int max_threads, int chosen) {
  std::string out;
  char line[512];
  for (const int T : candidate_threads(max_threads))
    append_line(out, line,
                std::snprintf(line, sizeof line,
                              "plan-solve T=%d sub_total=%.6g max_subtree=%.6g top_total=%.6g "
                              "top_cp=%.6g n_top=%d top_rows=%zu\n",
                              T, k.sub_total, k.max_subtree, k.top_total, k.top_cp,
                              static_cast<int>(k.n_top), k.top_rows));
  append_line(out, line, std::snprintf(line, sizeof line, "plan-solve chosen=%d\n", chosen));
  return out;
}

int plan_solve_threads(std::span<const double> subtree_work, std::span<const double> top_work,
                       std::span<const Int> top_parent, std::size_t top_rows, int max_threads,
                       const SolveCosts &c) {
  const bool parallel = max_threads >= 2 && subtree_work.size() >= 2;
  const bool trace = plan_trace_enabled();
  if (!parallel && !trace) return 1;
  const SolvePlanTerms k = solve_plan_terms(subtree_work, top_work, top_parent, top_rows);
  int chosen = 1;
  if (parallel) {
    std::vector<std::pair<int, double>> costs;
    for (const int T : candidate_threads(max_threads))
      costs.emplace_back(T, solve_plan_cost(k, T, c));
    chosen = least_cost_threads(costs).first;
  }
  if (trace) std::fputs(solve_plan_trace(k, parallel ? max_threads : 1, chosen).c_str(), stderr);
  return chosen;
}

} // namespace smf
