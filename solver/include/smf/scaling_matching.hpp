#pragma once
// ============================================================
// scaling_matching.hpp — Matching-based scaling interface
// Part of the smf MA97-class sparse direct solver.
//
// STUB: ScalingMethod::Matching is reserved for Phase 9.
//       compute_matching_scale always returns
//       ErrorCode::FeatureNotAvailable.
//
// See docs/scaling_matching_research.md for the planned
// MC64 (Duff & Koster 2001) implementation path.
// ============================================================

#include "types.hpp"
#include "error.hpp"
#include "csc_matrix.hpp"
#include <vector>

namespace smf {

/// Matching-based (MC64-like) diagonal scaling for a symmetric sparse matrix
/// stored in CSC lower-triangular form.
///
/// @param A           Input matrix (lower triangle, CSC storage).
/// @param scale       Output row/column scale vector.  Left unchanged by
///                    this stub — call sites should not rely on its contents
///                    when the function returns FeatureNotAvailable.
/// @param print_level Verbosity.  When > 0 a log message is printed to stderr
///                    explaining that the feature is deferred to Phase 9.
///
/// @return ErrorCode::FeatureNotAvailable  (always, until Phase 9).
ErrorCode compute_matching_scale(const CscLower& A,
                                 std::vector<double>& scale,
                                 int print_level = 0);

} // namespace smf
