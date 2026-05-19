// task_tree.cpp — OpenMP task-based parallel postorder traversal of the
// SMF supernode assembly tree.
//
// Pattern (§8.2):
//   #pragma omp parallel → #pragma omp single → spawn one task per root
//   Each subtree task: spawn one #pragma omp task per child, then
//   #pragma omp taskwait, then invoke the callback for the current node.
//
// Compiled unconditionally; the OpenMP code path is guarded by
// #ifdef SMF_PARALLEL so the file is always linkable even without -fopenmp.

#include "smf/threading.hpp"

#ifdef SMF_PARALLEL
#include <omp.h>
#endif

#include <vector>

namespace smf {

namespace {

#ifdef SMF_PARALLEL
// Forward declaration so the recursive lambda-free approach can be used.
static void subtree_task_impl(Int node, const std::vector<Supernode> &supernodes,
                              const std::function<void(Int)> &fn);

static void subtree_task_impl(Int node,
                              const std::vector<Supernode> &supernodes,
                              const std::function<void(Int)> &fn) {
  const Supernode &sn = supernodes[static_cast<std::size_t>(node)];

  // Spawn an independent task for each child subtree.
  // "child" is firstprivate (local copy per task) to avoid aliasing.
  // "supernodes" and "fn" are implicitly shared (const refs to outer scope).
  for (const Int child : sn.children) {
#pragma omp task firstprivate(child) shared(supernodes, fn) default(none)
    subtree_task_impl(child, supernodes, fn);
  }

  // Wait for all child tasks to finish before processing this node.
#pragma omp taskwait

  fn(node);
}
#endif // SMF_PARALLEL

/// Serial iterative postorder (children before parents) — fallback path.
static std::vector<Int>
serial_postorder(const std::vector<Supernode> &supernodes) {
  const Int ns = static_cast<Int>(supernodes.size());
  std::vector<Int> order;
  order.reserve(static_cast<std::size_t>(ns));

  // Iterative DFS: stack entries are (node, next_child_to_visit).
  std::vector<std::pair<Int, Int>> stk;
  stk.reserve(static_cast<std::size_t>(ns));

  for (Int i = 0; i < ns; ++i) {
    if (supernodes[static_cast<std::size_t>(i)].parent == -1)
      stk.push_back({i, 0});
  }

  while (!stk.empty()) {
    auto &[nd, ci] = stk.back();
    const Int nch = static_cast<Int>(
        supernodes[static_cast<std::size_t>(nd)].children.size());
    if (ci < nch) {
      const Int child =
          supernodes[static_cast<std::size_t>(nd)]
              .children[static_cast<std::size_t>(ci)];
      ++ci;
      stk.push_back({child, 0});
    } else {
      order.push_back(nd);
      stk.pop_back();
    }
  }

  return order;
}

} // anonymous namespace

void run_parallel_postorder(const std::vector<Supernode> &supernodes,
                            std::function<void(Int)> process_node,
                            int num_threads) {
#ifdef SMF_PARALLEL
  if (num_threads > 1 && !supernodes.empty()) {
    omp_set_num_threads(num_threads);

    // Collect forest roots (supernodes whose parent == -1).
    const Int ns = static_cast<Int>(supernodes.size());
    std::vector<Int> roots;
    roots.reserve(16);
    for (Int i = 0; i < ns; ++i) {
      if (supernodes[static_cast<std::size_t>(i)].parent == -1)
        roots.push_back(i);
    }

    // Launch one #pragma omp parallel region. The #pragma omp single thread
    // seeds a task per root; the OpenMP runtime schedules them across threads.
    // The implicit barrier at the end of the parallel region guarantees that
    // all tasks (including nested ones) have completed before we return.
#pragma omp parallel shared(roots, supernodes, process_node) default(none)
    {
#pragma omp single
      {
        for (const Int root : roots) {
#pragma omp task firstprivate(root) shared(supernodes, process_node)         \
    default(none)
          subtree_task_impl(root, supernodes, process_node);
        }
      } // end omp single — tasks may still be running
    }   // end omp parallel — implicit barrier, all tasks done here

    return;
  }
#else
  (void)num_threads; // suppress unused-parameter warning
#endif

  // Serial fallback: iterate the precomputed postorder list.
  const std::vector<Int> order = serial_postorder(supernodes);
  for (const Int s : order)
    process_node(s);
}

} // namespace smf
