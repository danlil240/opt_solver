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

/// The physical cores this process may run on (SMT siblings counted once),
/// at least 1: cores with at least one logical CPU in the process affinity
/// (Linux: sched_getaffinity, so cpusets of Docker --cpuset-cpus, Slurm,
/// taskset; Windows: the process' groups and, within a single group, its
/// affinity mask, so job objects and `start /affinity`), and never more
/// than the allowed CPUs. Falls back to std::thread::hardware_concurrency()
/// where the topology cannot be read. Computed once per process (later
/// affinity changes are not seen). CPU quotas (docker --cpus) are not
/// affinity and are not taken into account.
int physical_core_count() noexcept;

/// The counting rule of physical_core_count(): of `core_cpus` (one entry per
/// physical core, its logical CPU ids), the cores with at least one id in
/// `allowed_cpus` (empty: every core counts). Exposed for tests.
int count_allowed_cores(const std::vector<std::vector<int>> &core_cpus,
                        const std::vector<int> &allowed_cpus);

/// physical_core_count() restricted to the performance cores of a hybrid
/// CPU: of the physical cores the process may run on, those of the highest
/// performance class among them. Classes: Windows, each core's
/// EfficiencyClass (GetLogicalProcessorInformationEx); Linux, P-core when
/// one of its CPUs is in /sys/devices/cpu_core/cpus (read only when
/// /sys/devices/cpu_atom/cpus exists too, i.e. on hybrid Intel CPUs). Where
/// all allowed cores share one class, or the classes are unknown, it equals
/// physical_core_count(). An i9-12900K (8 P + 8 E-cores) gives 8. The IPOPT
/// plugin's default thread count. Computed once per process.
int performance_core_count() noexcept;

/// The counting rule of performance_core_count(): `core_class` holds one
/// performance class per entry of `core_cpus` (higher: faster). Of the
/// cores with at least one id in `allowed_cpus` (empty: every core), those
/// whose class is the highest among them, so an affinity without the
/// P-cores counts its E-cores. `core_class` empty or of another size than
/// `core_cpus` (unknown): count_allowed_cores(). Exposed for tests.
int count_allowed_performance_cores(const std::vector<std::vector<int>> &core_cpus,
                                    const std::vector<int> &core_class,
                                    const std::vector<int> &allowed_cpus);

} // namespace smf
