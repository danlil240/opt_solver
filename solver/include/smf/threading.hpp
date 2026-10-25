#pragma once

#include "smf/assembly_tree.hpp"
#include "smf/supernode.hpp"
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

TaskScheduleStats run_costed_postorder(
    const std::vector<Supernode> &supernodes,
    const std::vector<FrontalInfo> &fronts,
    const ExecutionPolicy &policy,
    std::function<void(Int)> process_node);

void run_parallel_postorder(const std::vector<Supernode> &supernodes,
                            std::function<void(Int)> process_node,
                            int num_threads = 1);

} // namespace smf
