#pragma once
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/factor_indef.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#include <memory>

namespace smf {

struct AnalysisKeep; // forward declaration (full definition via
                     // factor_posdef.hpp)

/// Main solver class.
class Solver {
public:
  Solver() = default;
  ~Solver() = default;

  // Non-copyable, movable
  Solver(const Solver &) = delete;
  Solver &operator=(const Solver &) = delete;
  Solver(Solver &&) = default;
  Solver &operator=(Solver &&) = default;

  /// Run symbolic analysis on the given lower-triangular CSC matrix.
  ///
  /// Ordering policy (documented here — follows MA97 convention):
  ///   AutoSerial   → AMD (always)
  ///   AutoParallel → METIS if n >= 5000 and SMF_HAS_METIS, else AMD
  ///   AMD          → AMD (regardless of n)
  ///   METIS        → METIS (fallback to AMD if SMF_HAS_METIS not compiled in)
  ///   UserSupplied → perm in control.user_perm (NOT IMPLEMENTED — returns
  ///   ErrorCode::FeatureNotAvailable)
  ///
  /// On success: fills info, returns a valid AnalysisHandle.
  /// On failure: sets info.status and returns nullptr.
  ///
  /// @param A        lower-triangular CSC input (need not be clean)
  /// @param control  solver options
  /// @param info     output statistics
  /// @returns        opaque handle owning the AnalysisKeep; nullptr on error
  std::unique_ptr<AnalysisKeep> analyse(const CscLower &A,
                                        const Control &control, Info &info);

  /// Factor the matrix A using the symbolic analysis from a prior analyse()
  /// call.
  /// @param ak     the AnalysisKeep from analyse()
  /// @param ctrl   solver control
  /// @param info   output: updated with factor statistics
  /// @param fkeep  output: factorization result
  /// @returns FactorStatus::Success, NotPositiveDefinite, or Singular
  FactorStatus factor(const AnalysisKeep &ak, const Control &ctrl, Info &info,
                      FactorKeep &fkeep) const;

  /// Solve A x = b for a single or multiple RHS.
  /// @param fkeep  from a prior factor() call
  /// @param ctrl   solver control
  /// @param info   output: updated with solve statistics
  /// @param b      in-place RHS / solution buffer, n × nrhs column-major
  /// @param n      matrix order
  /// @param nrhs   number of right-hand sides
  /// @param job    SolveJob::Full (default) or partial
  /// @returns 0 on success
  int solve(const FactorKeep &fkeep, const Control &ctrl, Info &info, double *b,
            int n, int nrhs = 1, SolveJob job = SolveJob::Full) const;

  /// Combined analyse + factor + solve in a single call.
  /// Solves A x = b in-place; returns the FactorKeep for inspection.
  /// @param A      lower-triangular CSC input
  /// @param ctrl   solver control
  /// @param info   output: updated with all phase statistics
  /// @param b      in-place RHS / solution buffer, n × nrhs column-major
  /// @param n      matrix order
  /// @param nrhs   number of right-hand sides
  /// @returns FactorKeep on success; nullptr if analysis or factorization fails
  std::unique_ptr<FactorKeep> factor_solve(const CscLower &A,
                                           const Control &ctrl, Info &info,
                                           double *b, int n, int nrhs = 1);
};

} // namespace smf
