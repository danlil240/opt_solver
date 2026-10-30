#pragma once
// Tree partition and thread count of the subtree-parallel factorization
// (multifrontal_serial.cpp).
//
// The partition splits the assembly tree into independent subtrees (phase
// 1, one thread each) and a top part (phase 2, TopScheduler). The thread
// count comes from a cost model of those two phases: each takes at least
// its work / T and its longest serial chain (a subtree in phase 1, the
// heaviest top path in phase 2, where fronts run as tile tasks count 1/T),
// plus a cost per thread and per top-part front. Work is f^2 p per front,
// the unit of Info::predicted_flops.

#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
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
  std::span<const std::uint8_t> split;  // 1: runs as tile tasks at > 1 thread
};

// The view of keep's assembly tree; fills the buffers it points into.
// tiles_on: top_tiles_on(...) at more than one thread.
TreeView tree_view(const AnalysisKeep &keep, bool tiles_on, std::vector<Int> &parent,
                   std::vector<double> &work, std::vector<std::uint8_t> &split);

// Splits the tree into independent subtrees of bounded work (largest-first:
// the heaviest subtree root moves to the top part until every remaining
// subtree is below total / (4 threads)). Returns the subtrees as [first,
// last] postorder index ranges, heaviest first, and marks the top part.
// Leaves too heavy to split stay in the top part. Returns nothing (and no
// top part) when the tree's work is below min_work.
std::vector<std::pair<std::size_t, std::size_t>>
partition_subtrees(const TreeView &t, int threads, double min_work, std::vector<std::uint8_t> &top);

// The sizes of partition_subtrees' result (subtrees, top-part nodes) in
// O(nodes) without building it: what plan_threads costs per candidate.
struct PartitionCount {
  std::size_t subtrees = 0;
  std::size_t top = 0;
};
PartitionCount partition_count(const TreeView &t, int threads, double min_work);

// Cost model terms, in work units (f^2 p). Control::thread_cost_per_thread
// and thread_cost_per_top_front.
struct ThreadCosts {
  double per_thread = 1e5;     // starting and joining one thread
  double per_top_front = 5e3;  // handing one top-part front between threads
};
struct ThreadPlan {
  int threads = 1;
  double predicted_speedup = 1.0;  // total / cost(threads)
};

// The thread count of least cost among 1, 2, 4, 8, ... and max_threads (at
// most max_threads); on a tie within 2%, the smaller. cost(1) = total. For
// T > 1, the engine's two phases on partition_subtrees' split at T threads
// (T is no candidate when that gives fewer than 2 subtrees): the subtrees in
// parallel, a barrier, then the top part:
//   cost(T) = max(max_subtree, sub_total / T) + max(cp_top(T), top_total / T)
//             + T per_thread + top(T) per_top_front,
// with sub_total / max_subtree the work inside the subtrees / of the
// heaviest one, top_total the top part's work, cp_top(T) the heaviest path
// through top nodes only (split fronts' work / T), and top(T) the number of
// top nodes.
ThreadPlan plan_threads(const TreeView &t, int max_threads, const ThreadCosts &c);

// Cost model terms of the parallel triangular solve, in step-work units
// (p f per front). Control::solve_cost_per_thread, solve_cost_per_top_step,
// solve_cost_per_top_row.
struct SolveCosts {
  double per_thread = 2e4;    // starting and joining one thread
  double per_top_step = 2e3;  // handing one top step between threads
  double per_top_row = 2.0;   // zeroing and summing one top row's buffer, per thread
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

} // namespace smf
