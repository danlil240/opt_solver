#pragma once
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace smf {

using Int     = int32_t;
using LongInt = int64_t;

inline constexpr LongInt kMaxFactorStorageEntries = (LongInt{1} << 29); // ~4 GiB of doubles

/// std::isfinite without the out-of-line classification call MSVC emits:
/// finite iff the exponent bits are not all ones.
inline bool is_finite_fast(double v) noexcept {
    const auto bits = std::bit_cast<std::uint64_t>(v);
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

inline bool checked_add_i64(LongInt a, LongInt b, LongInt &out) noexcept {
    if (a < 0 || b < 0) return false;
    if (a > std::numeric_limits<LongInt>::max() - b) return false;
    out = a + b;
    return true;
}

inline bool checked_mul_i64(LongInt a, LongInt b, LongInt &out) noexcept {
    if (a < 0 || b < 0) return false;
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > std::numeric_limits<LongInt>::max() / b) return false;
    out = a * b;
    return true;
}

inline bool checked_storage_elements(LongInt rows, LongInt cols, LongInt &out,
                                     LongInt max_entries = kMaxFactorStorageEntries) noexcept {
    if (!checked_mul_i64(rows, cols, out)) return false;
    return out <= max_entries;
}

enum class MatrixType {
    RealSymmetricPositiveDefinite = 3,
    RealSymmetricIndefinite       = 4
};

enum class OrderingMethod {
    AutoSerial   = 0,
    AutoParallel = 1,
    AMD          = 2,
    METIS        = 3,
    UserSupplied = 4,
    Matching     = 5    // matching-based compressed AMD (indefinite, needs values)
};

// When an auto ordering of an indefinite matrix switches to the matching
// ordering (OrderingMethod::Matching).
//   Predicted      - in analyse(), when the matching ordering's predicted
//                    flops are within a constant factor of AMD's.
//   RealizedDelays - analyse() keeps AMD; a factor() that delays more than
//                    5% of n pivots re-analyses once with the matching
//                    ordering, and later factor() calls with that analysis
//                    use it (IPOPT's MA97 od_hd_reuse rule).
enum class MatchingTrigger {
    Predicted      = 0,
    RealizedDelays = 1
};

enum class ScalingMethod {
    None         = 0,
    Equilibration = 1,
    Matching     = 2,
    UserSupplied = 3,
    Auto         = 4    // matching scaling from the analysis when it computed one (else none)
};

enum class FactorStatus {
    Success             = 0,
    NotPositiveDefinite = 1,
    Singular            = 2,
    MaxPivotDelays      = 3,
    // More negative pivots than Control::max_negative_pivots: the
    // factorization stopped early and the factor is unusable.
    InertiaExceeded     = 4
};

enum class SolveJob {
    Full     = 0,   // AX = B
    Forward  = 1,   // PLX = SB
    DiagOnly = 2,   // DX = B (indef)
    Backward = 3,   // (PL)^T S^{-1} X = B
    DiagBack = 4    // D(PL)^T S^{-1} X = B (indef)
};

} // namespace smf
