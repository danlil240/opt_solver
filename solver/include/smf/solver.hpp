#pragma once
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/factor_indef.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"
#include <memory>
#include <span>

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
  /// @returns FactorStatus::Success, NotPositiveDefinite, or Singular;
  ///          InertiaExceeded when ctrl.max_negative_pivots was exceeded
  ///          (the factor is then unusable and solve() refuses it)
  FactorStatus factor(const AnalysisKeep &ak, const Control &ctrl, Info &info,
                      FactorKeep &fkeep) const;

  FactorStatus factor(const AnalysisKeep &ak, const CscLower &updated,
                      const Control &ctrl, Info &info, FactorKeep &fkeep,
                      NumericWorkspace &ws) const;

  /// Refactor with new values in the entry order of the matrix passed to
  /// analyse() (same pattern): nothing to copy into a CscLower, and no
  /// pattern comparison.
  FactorStatus factor(const AnalysisKeep &ak, std::span<const double> values,
                      const Control &ctrl, Info &info, FactorKeep &fkeep,
                      NumericWorkspace &ws) const;

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

  int solve(const FactorKeep &fkeep, const Control &ctrl, Info &info, double *b,
            int n, int nrhs, SolveWorkspace &ws,
            SolveJob job = SolveJob::Full) const;

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

private:
  /// Ordering workspace and analysis temporaries reused by successive
  /// analyse() calls.
  std::vector<Int> analyse_scratch_;
  std::vector<Int> analyse_work_;
};

} // namespace smf
