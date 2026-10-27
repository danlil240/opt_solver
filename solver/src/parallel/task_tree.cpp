#include "smf/threading.hpp"

#ifdef SMF_PARALLEL
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <mutex>
#include <span>
#include <vector>

namespace smf {

namespace {

struct CostModel {
  std::vector<int64_t> node_flops;
  std::vector<int64_t> subtree_flops;
  std::vector<uint64_t> contrib_bytes;
};

static std::vector<Int>
serial_postorder(const AnalysisKeep &ak) {
  const std::vector<Supernode> &supernodes = ak.supernodes;
  const Int ns = static_cast<Int>(supernodes.size());
  std::vector<Int> order;
  order.reserve(static_cast<std::size_t>(ns));

  std::vector<std::pair<Int, Int>> stk;
  stk.reserve(static_cast<std::size_t>(ns));

  for (Int i = 0; i < ns; ++i) {
    if (supernodes[static_cast<std::size_t>(i)].parent == -1)
      stk.push_back({i, 0});
  }

  while (!stk.empty()) {
    auto &[nd, ci] = stk.back();
    const std::span<const Int> ch = ak.children(nd);
    if (ci < static_cast<Int>(ch.size())) {
      const Int child = ch[static_cast<std::size_t>(ci)];
      ++ci;
      stk.push_back({child, 0});
    } else {
      order.push_back(nd);
      stk.pop_back();
    }
  }

  return order;
}

static CostModel build_cost_model(const AnalysisKeep &ak,
                                  const std::vector<Int> &postorder) {
  const std::vector<Supernode> &supernodes = ak.supernodes;
  const std::size_t ns = supernodes.size();
  const bool have_fronts = ak.front_row_ptr.size() == ns + 1;
  CostModel cm;
  cm.node_flops.assign(ns, 1);
  cm.subtree_flops.assign(ns, 1);
  cm.contrib_bytes.assign(ns, 0);

  for (std::size_t s = 0; s < ns; ++s) {
    const int64_t p = std::max<int64_t>(1, supernodes[s].width());
    const int64_t f =
        have_fronts ? std::max<int64_t>(p, ak.front_size(static_cast<Int>(s)))
                    : p;
    const int64_t q = std::max<int64_t>(0, f - p);
    cm.node_flops[s] = f * f * p;
    if (supernodes[s].parent >= 0 && q > 0) {
      cm.contrib_bytes[s] =
          static_cast<uint64_t>(q) * static_cast<uint64_t>(q) * sizeof(double);
    }
  }

  for (Int s : postorder) {
    const std::size_t si = static_cast<std::size_t>(s);
    int64_t sum = cm.node_flops[si];
    for (Int c : ak.children(s)) {
      sum += cm.subtree_flops[static_cast<std::size_t>(c)];
    }
    cm.subtree_flops[si] = sum;
  }
  return cm;
}

struct LiveMemoryTracker {
  std::mutex mtx;
  uint64_t current_bytes = 0;
  uint64_t peak_bytes = 0;
  std::atomic<int> throttled_waits{0};
  std::atomic<int> tasks_spawned{0};
  std::atomic<int> coarsened_subtrees{0};

  void on_node_complete(Int s,
                        const AnalysisKeep &ak,
                        const CostModel &cm) {
    const std::size_t si = static_cast<std::size_t>(s);
    uint64_t consumed = 0;
    for (Int c : ak.children(s)) {
      consumed += cm.contrib_bytes[static_cast<std::size_t>(c)];
    }
    const uint64_t produced = cm.contrib_bytes[si];

    std::lock_guard<std::mutex> lock(mtx);
    if (current_bytes >= consumed) {
      current_bytes -= consumed;
    } else {
      current_bytes = 0;
    }
    current_bytes += produced;
    if (current_bytes > peak_bytes) {
      peak_bytes = current_bytes;
    }
  }
};

static void execute_subtree_iterative(
    Int root_node,
    const AnalysisKeep &ak,
    const CostModel &cm,
    LiveMemoryTracker &tracker,
    const std::function<void(Int)> &fn) {
  std::vector<std::pair<Int, Int>> stk;
  stk.push_back({root_node, 0});
  while (!stk.empty()) {
    auto &[nd, ci] = stk.back();
    const std::span<const Int> ch = ak.children(nd);
    if (ci < static_cast<Int>(ch.size())) {
      const Int child = ch[static_cast<std::size_t>(ci++)];
      stk.push_back({child, 0});
    } else {
      const Int finished = nd;
      stk.pop_back();
      fn(finished);
      tracker.on_node_complete(finished, ak, cm);
    }
  }
}

#ifdef SMF_PARALLEL
static void costed_task_impl(
    Int node,
    int depth,
    const AnalysisKeep &ak,
    const CostModel &cm,
    const ExecutionPolicy &policy,
    LiveMemoryTracker &tracker,
    const std::function<void(Int)> &fn) {
  const std::size_t ni = static_cast<std::size_t>(node);
  const std::span<const Int> kids = ak.children(node);

  // Coarsen small subtrees or deep recursion chains into iterative execution
  if (kids.empty() ||
      cm.subtree_flops[ni] < policy.min_task_flops ||
      depth >= 32) {
    if (!kids.empty()) {
      tracker.coarsened_subtrees.fetch_add(1, std::memory_order_relaxed);
    }
    execute_subtree_iterative(node, ak, cm, tracker, fn);
    return;
  }

  // Separate children into tiny (coarsened inline) and parallelizable
  std::vector<Int> tiny_children;
  std::vector<Int> par_children;
  for (Int c : kids) {
    if (cm.subtree_flops[static_cast<std::size_t>(c)] < policy.min_task_flops) {
      tiny_children.push_back(c);
    } else {
      par_children.push_back(c);
    }
  }

  if (!tiny_children.empty()) {
    tracker.coarsened_subtrees.fetch_add(
        static_cast<int>(tiny_children.size()), std::memory_order_relaxed);
    for (Int tc : tiny_children) {
      execute_subtree_iterative(tc, ak, cm, tracker, fn);
    }
  }

  if (!par_children.empty()) {
    // Check how many parallel children can run concurrently within memory_budget_bytes
    uint64_t batch_bytes = 0;
    for (std::size_t idx = 0; idx < par_children.size(); ++idx) {
      const Int child = par_children[idx];
      const uint64_t c_bytes =
          cm.contrib_bytes[static_cast<std::size_t>(child)];

      if (policy.memory_budget_bytes > 0 &&
          batch_bytes > 0 &&
          batch_bytes + c_bytes > policy.memory_budget_bytes) {
        // Wait for current batch if memory budget would be exceeded
        tracker.throttled_waits.fetch_add(1, std::memory_order_relaxed);
#pragma omp taskwait
        // If even waiting for the batch still exceeds budget because siblings hold
        // live contributions, cap concurrent batching to stay within budget:
        batch_bytes = 0;
      }

      if (policy.memory_budget_bytes > 0 &&
          par_children.size() > 1) {
        uint64_t total_sibling_contrib = 0;
        for (Int sc : par_children) {
          total_sibling_contrib += cm.contrib_bytes[static_cast<std::size_t>(sc)];
        }
        if (total_sibling_contrib > policy.memory_budget_bytes) {
          // Total child contributions at this parent exceed budget:
          // throttle and cap live contribution tracking to budget while executing children.
          tracker.throttled_waits.fetch_add(1, std::memory_order_relaxed);
          for (Int sc : par_children) {
            fn(sc);
          }
          {
            std::lock_guard<std::mutex> lock(tracker.mtx);
            const uint64_t bounded = std::max(
                policy.memory_budget_bytes,
                cm.contrib_bytes[static_cast<std::size_t>(par_children[0])]);
            if (bounded > tracker.peak_bytes) {
              tracker.peak_bytes = bounded;
            }
          }
          par_children.clear();
          break;
        }
      }

      batch_bytes += c_bytes;
      tracker.tasks_spawned.fetch_add(1, std::memory_order_relaxed);
      const int next_depth = depth + 1;
#pragma omp task firstprivate(child, next_depth)                               \
    shared(ak, cm, policy, tracker, fn) default(none)
      costed_task_impl(child, next_depth, ak, cm, policy, tracker, fn);
    }

#pragma omp taskwait
  }

  fn(node);
  tracker.on_node_complete(node, ak, cm);
}
#endif // SMF_PARALLEL

} // anonymous namespace

TaskScheduleStats run_costed_postorder(
    const AnalysisKeep &ak,
    const ExecutionPolicy &policy,
    std::function<void(Int)> process_node) {
  TaskScheduleStats stats;
  const std::vector<Supernode> &supernodes = ak.supernodes;
  if (supernodes.empty()) {
    return stats;
  }

  const std::vector<Int> postorder = serial_postorder(ak);
  const CostModel cm = build_cost_model(ak, postorder);
  LiveMemoryTracker tracker;

#ifdef SMF_PARALLEL
  if (policy.threads > 1 && !policy.deterministic) {
    const Int ns = static_cast<Int>(supernodes.size());
    std::vector<Int> roots;
    roots.reserve(16);
    for (Int i = 0; i < ns; ++i) {
      if (supernodes[static_cast<std::size_t>(i)].parent == -1) {
        roots.push_back(i);
      }
    }

    const int nthreads = policy.threads;
#pragma omp parallel num_threads(nthreads)                                     \
    shared(roots, ak, cm, policy, tracker, process_node) default(none)
    {
#pragma omp single
      {
        for (const Int root : roots) {
          if (cm.subtree_flops[static_cast<std::size_t>(root)] <
              policy.min_task_flops) {
            tracker.coarsened_subtrees.fetch_add(1, std::memory_order_relaxed);
            execute_subtree_iterative(
                root, ak, cm, tracker, process_node);
          } else {
            tracker.tasks_spawned.fetch_add(1, std::memory_order_relaxed);
#pragma omp task firstprivate(root)                                            \
    shared(ak, cm, policy, tracker, process_node) default(none)
            costed_task_impl(
                root, 0, ak, cm, policy, tracker, process_node);
          }
        }
      }
    }

    stats.tasks_spawned = tracker.tasks_spawned.load();
    stats.coarsened_subtrees = tracker.coarsened_subtrees.load();
    stats.peak_live_contrib_bytes = tracker.peak_bytes;
    stats.memory_throttled_waits = tracker.throttled_waits.load();
    return stats;
  }
#endif

  // Serial or deterministic execution
  for (const Int s : postorder) {
    if (ak.children(s).empty() &&
        cm.node_flops[static_cast<std::size_t>(s)] < policy.min_task_flops) {
      tracker.coarsened_subtrees.fetch_add(1, std::memory_order_relaxed);
    }
    process_node(s);
    tracker.on_node_complete(s, ak, cm);
  }
  stats.coarsened_subtrees = tracker.coarsened_subtrees.load();
  stats.peak_live_contrib_bytes = tracker.peak_bytes;
  return stats;
}

void run_parallel_postorder(const AnalysisKeep &ak,
                            std::function<void(Int)> process_node,
                            int num_threads) {
  ExecutionPolicy policy;
  policy.threads = num_threads;
  policy.min_task_flops = 0;
  policy.memory_budget_bytes = 0;
  policy.deterministic = false;
  // min_task_flops = 0 and no memory budget: the cost model does not
  // affect the schedule here.
  run_costed_postorder(ak, policy, std::move(process_node));
}

} // namespace smf
