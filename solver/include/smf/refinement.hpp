#pragma once

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/error.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/workspace.hpp"

#include <span>
#include <vector>

namespace smf {

struct RefinementResult {
    ErrorCode status = ErrorCode::Success;
    int iterations = 0;
    double eta_inf = 0.0;
    double omega = 0.0;
};

RefinementResult refine(const CscLower &original,
                        const FactorKeep &fkeep,
                        const Control &ctrl,
                        std::span<const double> b,
                        std::span<double> x,
                        int nrhs,
                        SolveWorkspace &ws);

/// One entry (row >= col) of a canonical symmetric matrix: sorted by
/// (col, row), duplicates summed.
struct RefineEntry {
    Int row = 0;
    Int col = 0;
    long double val = 0.0L;
};

/// The matrix refine() checks residuals against, prepared once per pattern
/// so that repeated refinements neither copy nor sort it.
struct RefineMatrix {
    Int n = 0;
    std::vector<RefineEntry> entries;  ///< canonical entries
    std::vector<Int> slot;             ///< input entry k -> its canonical entry
    long double norm_inf = 0.0L;       ///< ||A||_inf of the current values
    bool finite = false;               ///< current values set and all finite
};

/// Build the canonical pattern of A (lower CSC; its values are ignored).
ErrorCode prepare_refine_matrix(const CscLower &A, RefineMatrix &m);
/// Set the values, in A's entry order (duplicates summed in input order),
/// and ||A||_inf. O(nnz), no sorting.
void set_refine_values(std::span<const double> values, RefineMatrix &m);
/// refine() against a prepared matrix: the same iteration as the CscLower
/// overload (which builds its canonical entries on every call).
RefinementResult refine(const RefineMatrix &original,
                        const FactorKeep &fkeep,
                        const Control &ctrl,
                        std::span<const double> b,
                        std::span<double> x,
                        int nrhs,
                        SolveWorkspace &ws);

} // namespace smf
