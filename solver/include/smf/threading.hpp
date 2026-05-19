#pragma once
// threading.hpp — OpenMP task-tree helpers for SMF parallel factorization.
//
// §8.2 of the implementation plan: children processed before parent
// (taskwait), independent subtrees processed in parallel (omp task).
//
// All declarations are always visible.  The OpenMP task implementation is
// compiled only when SMF_PARALLEL is defined; otherwise the traversal
// degenerates to a serial postorder loop with zero OpenMP overhead.

#include "smf/supernode.hpp"
#include "smf/types.hpp"

#include <functional>
#include <vector>

namespace smf {

/// Run a postorder traversal of the supernode assembly tree using OpenMP
/// tasks when @p num_threads > 1 and SMF_PARALLEL is defined.
///
/// Guarantees:
///   • process_node(s) is called exactly once per supernode s.
///   • All children of s have fully returned from process_node before
///     process_node(s) is entered (#pragma omp taskwait).
///   • Independent sibling subtrees (different children of the same parent)
///     may execute concurrently — no synchronisation between them.
///
/// Thread-safety contract for the callback:
///   • Two tasks writing to the *same* supernode never run concurrently.
///   • A parent task begins only after all children tasks have returned.
///   • Therefore process_node only needs to protect per-node data (e.g.
///     atomic error flags) from concurrent writes originating in *different
///     subtrees*.
///
/// @param supernodes   Supernode assembly tree (read-only inside the parallel
///                     region).
/// @param process_node Callable invoked for each node in postorder.
/// @param num_threads  Number of OpenMP threads (≥1).  Values ≤1 or an
///                     un-compiled SMF_PARALLEL result in serial execution.
void run_parallel_postorder(const std::vector<Supernode> &supernodes,
                            std::function<void(Int)> process_node,
                            int num_threads = 1);

} // namespace smf
