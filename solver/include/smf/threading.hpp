#pragma once

#include "smf/analysis.hpp"
#include "smf/types.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace smf {

struct ExecutionPolicy {
    int threads = 1;
    uint64_t memory_budget_bytes = 0;
    int64_t min_task_flops = 10000;
    bool deterministic = false;
};

struct TaskScheduleStats {
    int tasks_spawned = 0;
    int coarsened_subtrees = 0;
    uint64_t peak_live_contrib_bytes = 0;
    int memory_throttled_waits = 0;
};

/// Run process_node over the supernode tree of `ak` children-first (a node's
/// children in ak.children() order). Costs use ak.front_size() when
/// ak.front_row_ptr is populated, else the supernode widths.
TaskScheduleStats run_costed_postorder(
    const AnalysisKeep &ak,
    const ExecutionPolicy &policy,
    std::function<void(Int)> process_node);

void run_parallel_postorder(const AnalysisKeep &ak,
                            std::function<void(Int)> process_node,
                            int num_threads = 1);

} // namespace smf
