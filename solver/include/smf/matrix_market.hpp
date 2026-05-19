// matrix_market.hpp — Matrix Market (.mtx) file reader for smf
//
// Reads %%MatrixMarket matrix coordinate real symmetric/general files.
// Returns a CscLower (for symmetric) or lower-triangle-only CscLower (for
// general matrices treated symmetrically).
//
// Usage:
//   smf::MatrixMarketResult r = smf::read_matrix_market("file.mtx");
//   if (!r.ok) { /* r.error_message has details */ }
//   else { smf::CscLower A = std::move(r.matrix); }
//
// NOTE: No exceptions thrown.  All errors reported via MatrixMarketResult.

#pragma once

#include "smf/csc_matrix.hpp"

#include <string>

namespace smf {

/// Result returned by read_matrix_market().
struct MatrixMarketResult {
    bool       ok            = false;  ///< true iff parse succeeded
    CscLower   matrix;                 ///< loaded matrix (empty if !ok)
    bool       is_symmetric  = false;  ///< true if header says "symmetric"
    bool       is_general    = false;  ///< true if header says "general"
    int        rows          = 0;      ///< declared rows in file header
    int        cols          = 0;      ///< declared cols in file header
    int        nnz_declared  = 0;      ///< nnz declared in size line
    std::string error_message;         ///< non-empty iff !ok
};

/// Read a Matrix Market file from disk.
/// @param path   Filesystem path to the .mtx file.
/// @returns      MatrixMarketResult (never throws).
///
/// Supported formats:
///   %%MatrixMarket matrix coordinate real symmetric
///   %%MatrixMarket matrix coordinate real general
///   %%MatrixMarket matrix coordinate integer symmetric
///   %%MatrixMarket matrix coordinate integer general
///
/// For symmetric matrices every listed entry (r,c,v) with r < c is
/// mirrored: both (r,c) and (c,r) are implied.  Only lower-triangle
/// entries (row >= col) are stored in the returned CscLower.
///
/// For general matrices, upper-triangle entries are silently dropped
/// (only lower-triangle entries, row >= col, are kept).
///
/// Entries are 1-based in the file; this reader converts to 0-based.
/// Duplicate (row,col) pairs are summed.
/// Out-of-range indices are silently discarded.
MatrixMarketResult read_matrix_market(const char *path);

/// Convenience overload accepting std::string.
inline MatrixMarketResult read_matrix_market(const std::string &path) {
    return read_matrix_market(path.c_str());
}

} // namespace smf
