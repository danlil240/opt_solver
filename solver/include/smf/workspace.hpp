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
    std::size_t contrib_pool_bytes = 0;
    std::unique_ptr<AlignedArena> serial_arena;

    // Lean serial multifrontal engine state (sized on first use, reused by
    // every refactorization of the same analysis).
    std::vector<Int> row_pos;          ///< global row -> front-local index, -1 when unused
    std::vector<Int> front_rows;       ///< row ids of the current front
    std::vector<Int> child_map;        ///< child contribution row -> parent-local index
    std::vector<double> kernel_work;   ///< W panel scratch for the blocked kernel
    std::vector<int8_t> pivot_tags;    ///< per-front pivot tags from the kernel
    std::vector<double> cb_values;     ///< LIFO stack of packed lower contribution blocks
    std::vector<Int> cb_rows;          ///< LIFO stack of contribution row ids
    std::vector<LongInt> cb_value_off; ///< per-supernode offset into cb_values
    std::vector<LongInt> cb_row_off;   ///< per-supernode offset into cb_rows
    std::vector<Int> cb_size;          ///< per-supernode contribution order q
    std::vector<Int> cb_delayed;       ///< per-supernode delayed columns (leading rows of its block)

    int warmup_calls = 0;
    int64_t front_heap_allocations_last_call = 0;
    int64_t capacity_growths_last_call = 0;
    std::size_t peak_pool_bytes = 0;

    void prepare_for_analysis(const AnalysisKeep &ak);
    std::vector<double> acquire_contrib(std::size_t elems, bool &allocated_new);
    void release_contrib(std::vector<double> &&buf);
};

/// Scatter `updated` values (original or cleaned pattern) into
/// ws.perm_values (permuted pattern) and, when `need_clean`, ws.clean_values.
/// Rejects non-finite values; `max_abs` (optional) receives max |value|.
ErrorCode map_updated_values_to_permuted(const AnalysisKeep &ak,
                                         const CscLower &updated,
                                         NumericWorkspace &ws,
                                         bool need_clean = true,
                                         double *max_abs = nullptr);

} // namespace smf
