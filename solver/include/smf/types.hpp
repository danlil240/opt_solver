#pragma once
#include <cstdint>

namespace smf {

using Int     = int32_t;
using LongInt = int64_t;

enum class MatrixType {
    RealSymmetricPositiveDefinite = 3,
    RealSymmetricIndefinite       = 4
};

enum class OrderingMethod {
    AutoSerial   = 0,
    AutoParallel = 1,
    AMD          = 2,
    METIS        = 3,
    UserSupplied = 4
};

enum class ScalingMethod {
    None         = 0,
    Equilibration = 1,
    Matching     = 2,
    UserSupplied = 3
};

enum class FactorStatus {
    Success             = 0,
    NotPositiveDefinite = 1,
    Singular            = 2,
    MaxPivotDelays      = 3
};

enum class SolveJob {
    Full     = 0,   // AX = B
    Forward  = 1,   // PLX = SB
    DiagOnly = 2,   // DX = B (indef)
    Backward = 3,   // (PL)^T S^{-1} X = B
    DiagBack = 4    // D(PL)^T S^{-1} X = B (indef)
};

} // namespace smf
