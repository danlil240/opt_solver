// ============================================================
// scaling_matching.cpp — Matching-based scaling (stub)
// Part of the smf MA97-class sparse direct solver.
//
// ScalingMethod::Matching is deferred to Phase 9.
// This translation unit provides the stub that wires into the
// solver's scaling dispatch so that passing Matching as the
// chosen method produces a clear, actionable error rather than
// undefined behaviour or a silent no-op.
//
// See docs/scaling_matching_research.md for the planned
// MC64 (Duff & Koster 2001) implementation notes.
// ============================================================

#include "smf/scaling_matching.hpp"

#include <cstdio>   // fprintf, stderr

namespace smf {

ErrorCode compute_matching_scale(const CscLower& /*A*/,
                                 std::vector<double>& /*scale*/,
                                 int print_level)
{
    if (print_level > 0) {
        fprintf(stderr,
            "[smf] ScalingMethod::Matching is not yet implemented "
            "(Phase 9 deferred). Returning FeatureNotAvailable.\n");
    }
    return ErrorCode::FeatureNotAvailable;
}

} // namespace smf
