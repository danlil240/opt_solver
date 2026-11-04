#pragma once
// Tree partition and thread count of the subtree-parallel factorization
// (multifrontal_serial.cpp).
//
// The partition splits the assembly tree into independent subtrees (phase
// 1, one thread each) and a top part (phase 2, TopScheduler). The thread
// count comes from a cost model of those two phases: each takes at least
// its work / T and its longest serial chain (a subtree in phase 1, the
// heaviest top path in phase 2, where fronts run as tile tasks or untiled
// fronts of kTopTileMinFront rows and more count 1/T),
// plus a cost per thread and per top-part front. Work is f^2 p per front,
// the unit of Info::predicted_flops.

#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace smf {

// Below this much front work (sum of f^2 p) a tree is not partitioned
// (one delay-budget region, and no candidate thread count beyond 1).
inline constexpr double kTreeParallelMinWork = 2e7;

// Fronts at least this large are factored as 2D tile tasks at the top of the
// tree (Control::top_tiles).
inline constexpr int kTopTileMinFront = 512;

// Control::top_tiles for a factorization: the plain threshold-pivoting
// fronts (indefinite, no static or in-order pivoting, pivot_u >=
// kWeakPivotRatio) are tiled at more than one thread, and in deterministic
// mode at every thread count. static_pivot / min_abs_pivot: the kernel's.
bool top_tiles_on(const Control &ctrl, bool posdef, double static_pivot, double min_abs_pivot,
                  bool det);

// Per front, with top_tiles_on: a front of front_rows rows is tiled (unless
// it turns into a fallback or delay-budget static front, decided while
// factoring).
inline bool tiles_front(bool tiles_on, int front_rows) {
  return tiles_on && front_rows >= kTopTileMinFront;
}

struct TreeView {                       // one entry per supernode
  std::span<const Int> postorder;
  std::span<const Int> parent;          // -1: root
  std::span<const double> work;         // f^2 p
  std::span<const std::uint8_t> split;  // kSplitTiles / kSplitBlas, else 0
};

// TreeView::split: the front runs as tile tasks at > 1 thread.
inline constexpr std::uint8_t kSplitTiles = 1;
// TreeView::split: a front of at least kTopTileMinFront rows that is not
// tiled (tiles off: weak pivot threshold, static or in-order pivoting,
// positive definite). It runs as one task with its BLAS share of the
// threads (TopScheduler), all of them on the heaviest top path, where such
// fronts follow one another near the root.
inline constexpr std::uint8_t kSplitBlas = 2;

// The view of keep's assembly tree; fills the buffers it points into.
// tiles_on: top_tiles_on(...) at more than one thread.
TreeView tree_view(const AnalysisKeep &keep, bool tiles_on, std::vector<Int> &parent,
                   std::vector<double> &work, std::vector<std::uint8_t> &split);

// Subtree work of every node of a tree view, with each subtree's [first,
// at] postorder range and the children lists: what the partitions and the
// plan walk. Children are summed in descending postorder (AnalysisKeep::
// children' order), so the sums add up exactly as they always did: the
// delay-budget regions depend on them. subtree_sums refills one in place,
// so a caller that keeps it (multifrontal_serial.cpp, per workspace) and
// passes it to the overloads below walks a new analysis without fresh O(ns)
// buffers; the overloads without one build their own.
struct SubtreeSums {
  std::vector<std::size_t> cptr;  // children, CSR
  std::vector<Int> cidx;
  std::vector<double> work;
  std::vector<std::size_t> first, at;
  double total = 0.0;  // over the roots, in index order
  // Scratch: subtree_sums' CSR fill pointers, factor_plan_terms' heaviest
  // top path below each node.
  std::vector<std::size_t> fill;
  std::vector<double> below;

  std::span<const Int> children(std::size_t s) const {
    return std::span<const Int>(cidx.data() + cptr[s], cptr[s + 1] - cptr[s]);
  }
};
void subtree_sums(const TreeView &t, SubtreeSums &u);

// Splits the tree into independent subtrees of bounded work (largest-first:
// the heaviest subtree root moves to the top part until every remaining
// subtree is below total / (4 threads)). Returns the subtrees as [first,
// last] postorder index ranges, heaviest first, and marks the top part.
// Leaves too heavy to split stay in the top part. Returns nothing (and no
// top part) when the tree's work is below min_work.
// The delay budget's regions (multifrontal_serial.cpp) are these subtrees.
std::vector<std::pair<std::size_t, std::size_t>>
partition_subtrees(const TreeView &t, int threads, double min_work, std::vector<std::uint8_t> &top);
// The same with u = subtree_sums(t).
std::vector<std::pair<std::size_t, std::size_t>>
partition_subtrees(const TreeView &t, const SubtreeSums &u, int threads, double min_work,
                   std::vector<std::uint8_t> &top);

// The phase-1 tasks of the tree-parallel factorization without a delay
// budget: partition_subtrees' subtrees, with runs of light ones merged into
// tasks of at most task_grain(sub_total, threads) work, sub_total the
// subtrees' work. Ranges merge in ascending postorder, only when adjacent
// (the first one's last index + 1 is the next one's first) and while the
// merged work stays within the grain; a subtree heavier than the grain
// stays alone. A merged range is a union of whole subtrees, so it is still
// closed under children and runs as one subtree does. Heaviest first (ties:
// the larger node at the range's end first, the heap's order), with
// partition_subtrees' top part. Without the merge, a tree that splits into
// thousands of one-front subtrees under a few top fronts pays a dispatch
// and a solve range per front.
std::vector<std::pair<std::size_t, std::size_t>>
partition_tasks(const TreeView &t, int threads, double min_work, std::vector<std::uint8_t> &top);
// The same with u = subtree_sums(t).
std::vector<std::pair<std::size_t, std::size_t>>
partition_tasks(const TreeView &t, const SubtreeSums &u, int threads, double min_work,
                std::vector<std::uint8_t> &top);

// partition_tasks' merge grain: sub_total / (kTaskGrainsPerThread threads).
// Two adjacent merged tasks weigh more than a grain together, so there are
// at most about 2 kTaskGrainsPerThread merged tasks per thread, and a
// heaviest-first schedule of tasks within a grain ends within a grain of
// sub_total / threads. Swept at 4 to 64 per thread in Task 6a2: within noise.
inline constexpr double kTaskGrainsPerThread = 16.0;
inline double task_grain(double sub_total, int threads) {
  return sub_total / (kTaskGrainsPerThread * static_cast<double>(threads));
}

// The sizes of partition_tasks' result (tasks, top-part nodes) in O(nodes)
// without building it: what plan_threads costs per candidate.
struct PartitionCount {
  std::size_t subtrees = 0;  // tasks
  std::size_t top = 0;
};
PartitionCount partition_count(const TreeView &t, int threads, double min_work);

// Cost model terms, in work units (f^2 p). Control::thread_cost_per_thread,
// thread_cost_per_top_front, thread_cost_per_split_front_thread and
// thread_split_scaling_cap (the fitted defaults and their evidence are
// there).
struct ThreadCosts {
  double per_thread = 7e5;              // a pool thread's wake-up, spin, barrier, dispatch
  double per_top_front = 5e3;           // handing one top-part front between threads
  double per_split_front_thread = 5e5;  // a tile-split front's team set-up, per thread
  int split_cap = 8;                    // threads past which split work stops speeding up (< 1: 1)
};
struct ThreadPlan {
  int threads = 1;
  double predicted_speedup = 1.0;  // total / cost(threads)
};

// The thread count of least cost among 1, 2, 4, 8, ... and max_threads (at
// most max_threads); on a tie within 2%, the smaller. cost(1) = total. For
// T > 1, the engine's two phases on partition_tasks' split at T threads (T
// is no candidate when that gives fewer than 2 tasks): the tasks in
// parallel, a barrier, then the top part:
//   cost(T) = max(max_subtree, sub_total / T) + top(T)
//             + T per_thread + top_fronts(T) per_top_front
//             + split_fronts(T) T per_split_front_thread,
//   top(T) = max(cp_top(T), (top_total - split_work) / T
//                           + split_work / min(T, split_cap)),
// with sub_total / max_subtree the work inside the subtrees / of the
// heaviest task, top_total the top part's work, split_work that of its
// tile-split fronts (which stop speeding up past split_cap threads),
// cp_top(T) the heaviest path through top nodes only (the work of split
// fronts and of kSplitBlas fronts / T), top_fronts(T) the number of top nodes and split_fronts(T) of them
// run as tile tasks.
ThreadPlan plan_threads(const TreeView &t, int max_threads, const ThreadCosts &c);
// The same with u = subtree_sums(t) (its scratch is used).
ThreadPlan plan_threads(const TreeView &t, SubtreeSums &u, int max_threads, const ThreadCosts &c);

// True when plan_threads picks 1 for every tree of this total work (the sum
// of f^2 p over its fronts) and every max_threads, so the plan need not be
// built: below kTreeParallelMinWork (no candidate past 1), or, with no
// negative cost, at most 4 per_thread. Every T > 1 costs at least
// sub_total / T + top_total / T + T per_thread = total / T + T per_thread,
// which reaches total when total <= 4 per_thread (the difference is
// per_thread (T - 2)^2 / T at total = 4 per_thread), so cost(1) = total is
// the least (rounding stays far inside the 2% tie band). Below 2^53 the
// work sums are exact integers in any order, so a total summed in another
// order than plan_threads' meets kTreeParallelMinWork exactly as its does.
bool plan_is_serial(double total, const ThreadCosts &c);

// The candidate thread counts of both models, ascending: 1, 2, 4, 8, ...
// below max_threads, and max_threads (1 alone below 2).
std::vector<int> candidate_threads(int max_threads);

// The terms plan_threads costs, per candidate thread count (independent of
// the cost constants). T = 1 carries the tree's total alone (cost(1) =
// total). For T > 1: ok when partition_tasks gives at least 2 tasks, the
// phase terms of the cost formula above, top_fronts = top(T), and the
// top-part fronts that run as tile tasks (split) with their work.
struct FactorPlanTerms {
  int threads = 1;
  bool ok = true;
  double sub_total = 0.0;
  double max_subtree = 0.0;
  double top_total = 0.0;
  double cp_top = 0.0;
  std::size_t top_fronts = 0;
  std::size_t split_top_fronts = 0;
  double split_top_work = 0.0;
  double total = 0.0;
};

// Candidate 1 alone when max_threads < 2 or the tree's work is below
// kTreeParallelMinWork; else one entry per candidate_threads(max_threads).
std::vector<FactorPlanTerms> factor_plan_terms(const TreeView &t, int max_threads);
// The same with u = subtree_sums(t) (its scratch is used).
std::vector<FactorPlanTerms> factor_plan_terms(const TreeView &t, SubtreeSums &u,
                                               int max_threads);

// cost(T) of plan_threads from one candidate's terms; < 0 when !ok.
double factor_plan_cost(const FactorPlanTerms &k, const ThreadCosts &c);

// The plan trace (SMF_BENCH_PLAN_TRACE=1, printed to stderr when a plan is
// computed): one "plan-factor T=.." line per candidate, then "plan-factor
// chosen=..". tools/bench/fit_thread_costs.py parses it.
std::string factor_plan_trace(std::span<const FactorPlanTerms> terms, int chosen);

// SMF_BENCH_PLAN_TRACE=1 in the environment (read once).
bool plan_trace_enabled();

// Cost model terms of the parallel triangular solve, in step-work units
// (p f per front). Control::solve_cost_per_thread, solve_cost_per_top_step,
// solve_cost_per_top_row.
struct SolveCosts {
  double per_thread = 2e4;    // a pool thread's wake-up, spin, barrier, dispatch
  double per_top_step = 100;  // handing one top step between threads
  double per_top_row = 1.0;   // zeroing and summing one top row's buffer, per thread
};

// The solve's thread count (FactorKeep::solve_threads) for a parallel solve
// layout: subtree_work / top_work the p f work of each subtree / top step,
// top_parent the top steps' tree (FactorKeep::solve_top_parent), top_rows
// the top part's pivot rows. Candidates and tie rule as plan_threads';
// cost(1) = sum(subtree) + sum(top), and for T > 1 the subtrees in parallel,
// then the top part as a DAG, with the per-thread buffers of top rows:
//   cost(T) = max(max_subtree, sum(subtree) / T) + cp_top + T per_thread
//             + n_top per_top_step + T top_rows per_top_row,
// cp_top the heaviest path through top_parent. Fewer than 2 subtrees (the
// solve then runs serially) or max_threads < 2: 1.
int plan_solve_threads(std::span<const double> subtree_work, std::span<const double> top_work,
                       std::span<const Int> top_parent, std::size_t top_rows, int max_threads,
                       const SolveCosts &c);

// The terms plan_solve_threads costs (the same at every thread count).
struct SolvePlanTerms {
  std::size_t subtrees = 0;
  double sub_total = 0.0;
  double max_subtree = 0.0;
  double top_total = 0.0;
  double top_cp = 0.0;  // cp_top above
  std::size_t n_top = 0;
  std::size_t top_rows = 0;
};
SolvePlanTerms solve_plan_terms(std::span<const double> subtree_work,
                                std::span<const double> top_work, std::span<const Int> top_parent,
                                std::size_t top_rows);

// cost(T) of plan_solve_threads (T = 1: sub_total + top_total).
double solve_plan_cost(const SolvePlanTerms &k, int threads, const SolveCosts &c);

// The plan trace: one "plan-solve T=.." line per candidate_threads(
// max_threads), then "plan-solve chosen=..".
std::string solve_plan_trace(const SolvePlanTerms &k, int max_threads, int chosen);

} // namespace smf
