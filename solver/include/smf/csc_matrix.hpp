#pragma once
#include "smf/types.hpp"
#include "smf/error.hpp"
#include <vector>

namespace smf {

struct CscLower {
    Int              n = 0;
    std::vector<Int> col_ptr;   // size n+1; col_ptr[0] == 0
    std::vector<Int> row_idx;   // size nnz, lower-triangle only (row >= col)
    std::vector<double> values; // size nnz

    /// Return the number of stored entries (from col_ptr).
    Int nnz() const noexcept { return col_ptr.empty() ? 0 : col_ptr.back(); }

    /// Cheap structural check (no value walk).
    /// Verifies sizes, col_ptr[0]==0, monotone col_ptr,
    /// row_idx/values size consistency, and row_idx[k] in [col, n).
    /// Does NOT check for duplicate entries or within-column sorting.
    ErrorCode validate_shape() const noexcept;
};

// Forward declarations — implementations are filled in later phases.
struct AnalysisKeep;   // concrete definition in analysis.hpp (Phase 2)
struct FactorKeep;     // concrete definition in factor.hpp (Phase 3)

using AnalysisHandle = AnalysisKeep;
using FactorHandle   = FactorKeep;

} // namespace smf
