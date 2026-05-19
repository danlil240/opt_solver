#pragma once
#include "smf/types.hpp"
#include "smf/error.hpp"

namespace smf {

struct Info {
    // Top-level status
    ErrorCode    status              = ErrorCode::Success;
    FactorStatus factor_status       = FactorStatus::Success;

    // Matrix input diagnostics
    int          matrix_duplicates   = 0;
    int          matrix_missing_diag = 0;
    int          matrix_out_of_range = 0;

    // Rank information
    int          structural_rank     = 0;
    int          numerical_rank      = 0;

    // Ordering used (value matches OrderingMethod enum)
    int          ordering_used       = 0;

    // Inertia
    int          num_negative        = 0;   // negative eigenvalues
    int          num_zero            = 0;   // zero eigenvalues
    int          num_positive        = 0;   // positive eigenvalues

    // Factorization counts
    LongInt      actual_factor_entries  = 0;
    LongInt      predicted_factor_entries = 0;
    double       predicted_flops        = 0.0;
    double       actual_flops           = 0.0;
    int          delayed_pivots         = 0;

    // Assembly tree stats
    int          max_front_size         = 0;
    int          max_supernode_size     = 0;
    int          max_tree_depth         = 0;

    // Timings (seconds)
    double       analyse_seconds        = 0.0;
    double       factor_seconds         = 0.0;
    double       solve_seconds          = 0.0;

    // Arena memory diagnostics
    long         arena_peak_bytes       = 0;
    int          arena_growths          = 0;
};

} // namespace smf
