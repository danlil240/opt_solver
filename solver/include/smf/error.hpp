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
    InternalError       = -99
};

} // namespace smf
