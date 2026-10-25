#pragma once

#include "smf/analysis.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/error.hpp"
#include "smf/numeric_front.hpp"
#include "smf/types.hpp"
#include "smf/utils/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace smf {

struct SolveWorkspace {
    std::vector<double> tmp;
    std::vector<double> loc;
    std::vector<double> ext;
    std::vector<double> rhs_perm;
    std::vector<double> refine_r;
    std::vector<double> refine_dx;
    std::vector<double> refine_best_x;

    void ensure_capacity(int n, int max_p, int max_q, int nrhs = 1);
};

struct NumericWorkspace {
    std::vector<double> clean_values;
    std::vector<double> perm_values;
    std::vector<double> front_scratch;
    std::vector<Int> row_map_scratch;
    std::vector<FrontResult> node_results;
    std::vector<std::vector<double>> contrib_pool;
    std::unique_ptr<AlignedArena> serial_arena;

    int warmup_calls = 0;
    int64_t front_heap_allocations_last_call = 0;
    int64_t capacity_growths_last_call = 0;
    std::size_t peak_pool_bytes = 0;

    void prepare_for_analysis(const AnalysisKeep &ak);
    std::vector<double> acquire_contrib(std::size_t elems, bool &allocated_new);
    void release_contrib(std::vector<double> &&buf);
};

ErrorCode map_updated_values_to_permuted(const AnalysisKeep &ak,
                                         const CscLower &updated,
                                         NumericWorkspace &ws);

} // namespace smf
