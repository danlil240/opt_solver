#pragma once
#include <cstdint>

namespace smf {

enum class ErrorCode : int32_t {
    Success             =   0,
    NotInitialised      =  -1,
    AllocationError     =  -2,
    IllegalValue        =  -3,
    SingularMatrix      =  -4,
    NotPositiveDefinite =  -5,
    MaxIterations       =  -6,
    OrderingFailed      =  -7,
    FeatureNotAvailable =  -8,
    InertiaExceeded     =  -9,  // see FactorStatus::InertiaExceeded
    // Internal: a factorization stopped at its front-work limit (the
    // scale-on-delay trigger, Control::scale_on_delay_ratio). Solver::factor
    // always factors again after it, so no caller ever sees it.
    WorkLimitExceeded   = -10,
    InternalError       = -99
};

} // namespace smf
