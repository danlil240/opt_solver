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
// Relaxed-amalgamation merge rule (analyse).
//   Smf             - nemin plus the bounded-zeros rule (16 free zeros or
//                     <= 30% zeros).
//   SmfPlusCholmod  - the SMF rule, or CHOLMOD's defaults (which ignore
//                     nemin): merged pivots <= 4, no new zeros, <= 16 with
//                     zero share < 0.8, <= 48 with < 0.1, or any size with
//                     < 0.05. Collapses chains of thin top fronts.
enum class Amalgamation {
    Smf            = 0,
    SmfPlusCholmod = 1
};

enum class MatchingTrigger {
    Predicted      = 0,
    RealizedDelays = 1
};

// How analyse() decided the auto matching trial (Info::matching_trial; see
// Control::matching_search_budget).
//   None            - no trial (not an auto ordering of a KKT-like matrix),
//                     or the trial failed: structurally singular matrix, or
//                     the matching ordering could not be built.
//   Exact           - the exact matching decided; its searches stayed
//                     within the budget.
//   RejectedInitial - the searches exceeded the budget, and the matching
//                     ordering of the matching's initial phase was clearly
//                     too dear (past a guard band above the 1.5 flops rule):
//                     AMD/METIS kept, no exact matching.
//   ResumedExact    - the searches exceeded the budget, the initial phase's
//                     matching ordering was not clearly too dear, so the
//                     searches resumed and the exact matching decided.
enum class MatchingTrial {
    None            = 0,
    Exact           = 1,
    RejectedInitial = 2,
    ResumedExact    = 3
};

// Dense kernel for the fully summed columns of an indefinite front.
//   Tpp  - threshold partial pivoting (ldlt_factor_front), blocked BLAS-3,
//          on every front.
//   Aptp - a posteriori threshold pivoting (ldlt_aptp_front, SSIDS-style)
//          on every front, whatever its size (aptp_min_elig is ignored),
//          with ldlt_factor_front as a fallback on the columns APTP fails;
//          what still fails is delayed to the parent as with Tpp.
//   Auto - aptp_min_elig = 0 (the default): never APTP, i.e. Tpp. Serially
//          APTP measured slower on the tuning split and did not reduce
//          delays overall (it is meant to pay off with parallel tiles).
//          aptp_min_elig > 0: Aptp on fronts with at least that many fully
//          summed columns, Tpp below.
// Fronts that static-pivot (static_pivot_tol, a fallback front, the
// in-order quasidefinite mode) or are SPD always use Tpp: APTP has no
// static-pivot mode. Only the lean engine honours front_kernel: the legacy
// task-parallel engine (Control::tree_parallel) always uses its own TPP
// front kernel.
// Iterative refinement method (Control::refine_method, refine()).
//  Classical: x += M^-1 (b - A x) with the factor M, at most
//             Control::refinement_max_iters steps.
//  Fgmres:    right-preconditioned FGMRES(Control::fgmres_restart) with M as
//             the preconditioner, at most Control::refine_max_steps iterations.
//  Auto:      Classical, escalating to Fgmres on stagnation or on a
//             statically pivoted factor that classical refinement has not
//             brought to the gate within 3 steps (see refinement.hpp).
enum class RefineMethod {
    Classical = 0,
    Fgmres    = 1,
    Auto      = 2
};

enum class FrontKernel {
    Tpp  = 0,
    Aptp = 1,
    Auto = 2
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
