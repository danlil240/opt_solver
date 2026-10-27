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
    /// Classical unless FGMRES ran (RefineMethod::Fgmres, or Auto escalated).
    RefineMethod method_used = RefineMethod::Classical;
};

/// Refine x (n x nrhs, column-major) towards A x = b, A the original
/// (unscaled) matrix and fkeep its factor (possibly of a perturbed matrix:
/// static pivots, regularization). Success when both backward errors meet
/// the accuracy gate on every column: normwise eta = ||b - A x||inf /
/// (||A||inf ||x||inf + ||b||inf) <= 1e-10 and componentwise omega =
/// max_i |b - A x|_i / (|A||x| + |b|)_i <= 1e-8 (residuals in extended
/// precision). Otherwise MaxIterations (IllegalValue on bad input); x is
/// always left at the best iterate found. Control::refinement_max_iters = 0
/// disables refinement for every method (MaxIterations if x misses the gate).
///
/// Control::refine_method:
///  - Classical: x += M^-1 r, at most refinement_max_iters steps; stops after
///    two steps that contract max(eta / 1e-10, omega / 1e-8) by less than
///    0.85, or one that does not reduce it. Bit-identical to the refinement
///    before FGMRES existed.
///  - Fgmres: right-preconditioned FGMRES(fgmres_restart) in FP64 with the
///    factor as preconditioner (one solve per iteration), modified
///    Gram-Schmidt, restarted from the best iterate with its extended-
///    precision residual; at most refine_max_steps iterations. Every
///    iteration forms the iterate and stops once both measures meet the gate.
///    An iterate more than 1e8 times larger (inf-norm) than max(||x||,
///    ||b|| / ||A||) at the start is rejected (singular, inconsistent systems).
///  - Auto (default): Classical with at most min(refinement_max_iters,
///    refine_max_steps) steps, escalating to FGMRES when a step contracts the
///    measure by less than 0.5 (or does not improve it), or when the factor
///    has static pivots (FactorKeep::static_pivots > 0) and the gate is still
///    not met after min(3, that cap) steps. Until it escalates it takes the
///    Classical steps exactly, so a run whose steps each contract by at
///    least 0.5 (within 3 steps with static pivots) is the Classical run.
/// Budget (Fgmres, Auto): classical steps and FGMRES iterations together never
/// exceed refine_max_steps; FGMRES gets what classical refinement left, and
/// when a full restart cycle does not fit it simply runs up to the remaining
/// budget. With nrhs > 1 each column that misses the gate runs its own FGMRES
/// on that budget; iterations reports the classical steps plus the most
/// FGMRES iterations any column used.
/// RefinementResult::method_used is Fgmres when FGMRES ran, else Classical.
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
