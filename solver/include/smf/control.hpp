#pragma once
#include "smf/types.hpp"
#include <cstdint>

namespace smf {

struct Control {
    // Matrix type
    MatrixType     matrix_type   = MatrixType::RealSymmetricIndefinite;

    // Ordering
    OrderingMethod ordering      = OrderingMethod::AutoSerial;
    int            nemin         = 8;       // min columns for supernode amalgamation

    // Scaling
    ScalingMethod  scaling       = ScalingMethod::None;

    // Pivoting
    double         pivot_u       = 0.01;    // threshold pivoting parameter
    double         small_pivot   = 1e-20;   // zero-pivot threshold

    // Factorization behaviour
    bool           continue_on_singular = false;
    double         factor_memory_multiplier = 1.1;

    // Parallelism
    int            num_threads   = 1;
    bool           deterministic = false;
    int64_t        factor_parallel_min_flops = 10000;
    int64_t        solve_parallel_min_entries = 100000;

    // Solve options
    bool           solve_use_blas3_single_rhs    = false;
    bool           solve_multifrontal_forward     = false;

    // Diagnostics / logging
    int            print_level   = 0;
};

} // namespace smf
